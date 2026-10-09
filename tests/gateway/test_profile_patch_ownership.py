"""Guard canonical patch seams for the extracted profile domain."""
from __future__ import annotations

import ast
from pathlib import Path
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MOVED_FACADE_EXPORTS = {
    "PROFILE_ROLES",
    "SETUP_ROLE",
    "current_profile_name",
    "drop_profile_role",
    "format_profile_label",
    "get_active_profile",
    "get_active_profile_name",
    "get_profile_dir",
    "list_profile_names",
    "normalize_profile_name",
    "parked_marker_path",
    "profile_exists",
    "profile_is_parked",
    "profile_is_standalone",
    "profile_matches_home",
    "profile_root_for_env_home",
    "profiles_to_serve",
    "read_profile_meta",
    "resolve_profile_env",
    "set_active_profile",
    "validate_alias_name",
    "validate_profile_name",
    "write_profile_meta",
    "_get_default_hermes_home",
    "_get_profiles_root",
}


def _import_bindings(tree: ast.AST) -> dict[str, str]:
    """Map local import bindings to their canonical dotted identities."""
    bindings: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name.split(".", 1)[0]
                canonical = alias.name if alias.asname else local
                bindings[local] = canonical
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                if alias.name == "*":
                    continue
                bindings[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return bindings


def _canonical_name(node: ast.AST, bindings: dict[str, str]) -> str | None:
    """Resolve a Name/Attribute expression through imports without executing it."""
    if isinstance(node, ast.Name):
        return bindings.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        parent = _canonical_name(node.value, bindings)
        return f"{parent}.{node.attr}" if parent else None
    return None


def _profile_facade_aliases(tree: ast.AST) -> set[str]:
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases.update(
                alias.asname or "hermes_cli.profiles"
                for alias in node.names
                if alias.name == "hermes_cli.profiles"
            )
        elif isinstance(node, ast.ImportFrom) and node.module == "hermes_cli":
            aliases.update(
                alias.asname or "profiles"
                for alias in node.names
                if alias.name == "profiles"
            )
    return aliases


def _is_facade_target(node: ast.AST, aliases: set[str]) -> bool:
    if isinstance(node, ast.Name):
        return node.id in aliases
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "profiles"
        and isinstance(node.value, ast.Name)
        and node.value.id == "hermes_cli"
        and "hermes_cli.profiles" in aliases
    )


def _patched_facade_names(path: Path) -> list[tuple[int, str]]:
    source = path.read_text(encoding="utf-8", errors="ignore")
    if "hermes_cli" not in source or "profiles" not in source:
        return []
    tree = ast.parse(source, filename=str(path))
    aliases = _profile_facade_aliases(tree)
    bindings = _import_bindings(tree)
    offenders: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func

        if (
            isinstance(func, ast.Attribute)
            and func.attr == "setattr"
            and len(node.args) >= 2
        ):
            target, name = node.args[:2]
            if (
                _is_facade_target(target, aliases)
                and isinstance(name, ast.Constant)
                and name.value in MOVED_FACADE_EXPORTS
            ):
                offenders.append((node.lineno, str(name.value)))
                continue
            if (
                isinstance(target, ast.Constant)
                and isinstance(target.value, str)
                and target.value.startswith("hermes_cli.profiles.")
            ):
                patched = target.value.rsplit(".", 1)[-1]
                if patched in MOVED_FACADE_EXPORTS:
                    offenders.append((node.lineno, patched))
                    continue

        if (
            _canonical_name(func, bindings) == "unittest.mock.patch.object"
            and len(node.args) >= 2
            and _is_facade_target(node.args[0], aliases)
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in MOVED_FACADE_EXPORTS
        ):
            offenders.append((node.lineno, str(node.args[1].value)))
            continue

        if (
            _canonical_name(func, bindings) == "unittest.mock.patch"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and node.args[0].value.startswith("hermes_cli.profiles.")
        ):
            patched = node.args[0].value.rsplit(".", 1)[-1]
            if patched in MOVED_FACADE_EXPORTS:
                offenders.append((node.lineno, patched))

    return offenders


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "from hermes_cli import profiles\n"
            "monkeypatch.setattr(profiles, 'profile_exists', lambda _: True)\n",
            "profile_exists",
        ),
        (
            "import hermes_cli.profiles\n"
            "monkeypatch.setattr(hermes_cli.profiles, 'get_profile_dir', fake)\n",
            "get_profile_dir",
        ),
        (
            "monkeypatch.setattr('hermes_cli.profiles.get_active_profile_name', fake)\n",
            "get_active_profile_name",
        ),
        (
            "from unittest.mock import patch\n"
            "patch('hermes_cli.profiles.profiles_to_serve', return_value=[])\n",
            "profiles_to_serve",
        ),
    ],
)
def test_patch_collector_detects_facade_behavior_seams(
    tmp_path: Path,
    source: str,
    expected: str,
) -> None:
    probe = tmp_path / "probe.py"
    probe.write_text(source, encoding="utf-8")
    assert _patched_facade_names(probe) == [(2 if "\n" in source.rstrip("\n") else 1, expected)]


@pytest.mark.parametrize(
    "source",
    [
        "from unittest.mock import patch\npatch('target')\n",
        "from unittest.mock import patch as p\np('target')\n",
        "from unittest import mock\nmock.patch('target')\n",
        "import unittest.mock\nunittest.mock.patch('target')\n",
        "import unittest.mock as mock\nmock.patch('target')\n",
        "import unittest as ut\nut.mock.patch('target')\n",
    ],
)
def test_import_resolution_normalizes_patch_calls(source: str) -> None:
    tree = ast.parse(source)
    bindings = _import_bindings(tree)
    call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call))
    assert _canonical_name(call.func, bindings) == "unittest.mock.patch"


@pytest.mark.parametrize(
    "source",
    [
        "from unittest import mock\n"
        "mock.patch('hermes_cli.profiles.profile_exists', return_value=True)\n",
        "import unittest.mock\n"
        "unittest.mock.patch('hermes_cli.profiles.profile_exists', return_value=True)\n",
    ],
)
def test_patch_collector_uses_canonical_patch_identity(tmp_path: Path, source: str) -> None:
    probe = tmp_path / "probe.py"
    probe.write_text(source, encoding="utf-8")
    assert _patched_facade_names(probe) == [(2, "profile_exists")]


@pytest.mark.parametrize(
    "source",
    [
        "from unittest.mock import patch\nfrom hermes_cli import profiles\n"
        "patch.object(profiles, 'profile_exists', fake)\n",
        "from unittest.mock import patch as p\nfrom hermes_cli import profiles as hp\n"
        "p.object(hp, 'profile_exists', fake)\n",
        "from unittest import mock\nfrom hermes_cli import profiles\n"
        "mock.patch.object(profiles, 'profile_exists', fake)\n",
        "import unittest.mock\nfrom hermes_cli import profiles\n"
        "unittest.mock.patch.object(profiles, 'profile_exists', fake)\n",
        "import unittest as ut\nfrom hermes_cli import profiles\n"
        "ut.mock.patch.object(profiles, 'profile_exists', fake)\n",
    ],
)
def test_patch_collector_normalizes_patch_object_forms(tmp_path: Path, source: str) -> None:
    probe = tmp_path / "probe.py"
    probe.write_text(source, encoding="utf-8")
    expected_line = len(source.rstrip("\n").splitlines())
    assert _patched_facade_names(probe) == [(expected_line, "profile_exists")]


@pytest.mark.parametrize(
    "source",
    [
        "from unittest.mock import patch as p\n"
        "p('hermes_cli.profiles.profile_exists', return_value=True)\n",
        "from unittest.mock import patch as replace\n"
        "replace('hermes_cli.profiles.get_profile_dir', fake)\n",
    ],
)
def test_patch_aliases_cannot_bypass_source_prefilter(tmp_path: Path, source: str) -> None:
    probe = tmp_path / "probe.py"
    probe.write_text(source, encoding="utf-8")
    expected_name = "profile_exists" if "profile_exists" in source else "get_profile_dir"
    assert _patched_facade_names(probe) == [(2, expected_name)]


def test_patch_collector_allows_canonical_owner(tmp_path: Path) -> None:
    probe = tmp_path / "probe.py"
    probe.write_text(
        "monkeypatch.setattr('profiles.registry.profile_exists', lambda _: True)\n",
        encoding="utf-8",
    )
    assert _patched_facade_names(probe) == []


def _tracked_profile_test_files() -> list[str]:
    result = subprocess.run(
        ["git", "grep", "-l", "-e", "hermes_cli.profiles",
         "-e", "from hermes_cli import profiles", "--", "tests"],
        cwd=REPO_ROOT, check=False, capture_output=True, text=True,
    )
    if result.returncode not in (0, 1):
        raise AssertionError(
            f"profile patch ownership discovery failed with git grep exit "
            f"{result.returncode}: {result.stderr.strip() or 'no stderr'}"
        )
    return result.stdout.splitlines()


def test_profile_patch_discovery_fails_closed(monkeypatch) -> None:
    def fail(*args, **kwargs):
        return subprocess.CompletedProcess(
            args[0], 128, stdout="", stderr="fatal: not a git repository"
        )

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(AssertionError, match="git grep exit 128"):
        _tracked_profile_test_files()


def test_profile_tests_patch_canonical_owners_not_facade_aliases() -> None:
    offenders: list[tuple[str, int, str]] = []
    tests_root = REPO_ROOT / "tests"
    tracked = _tracked_profile_test_files()
    for relative_text in tracked:
        if not relative_text.endswith(".py"):
            continue
        path = REPO_ROOT / relative_text
        if path == Path(__file__).resolve():
            continue
        for line, name in _patched_facade_names(path):
            offenders.append((str(path.relative_to(tests_root)), line, name))

    assert offenders == []
