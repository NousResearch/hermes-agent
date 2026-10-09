"""Architecture guard for the Phase 1 profile-domain extraction."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"), filename=str(path))
    refs: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            refs.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            refs.add(node.module)
            refs.update(f"{node.module}.{alias.name}" for alias in node.names)
    return refs


def _is_cli_profiles_ref(ref: str) -> bool:
    return ref == "hermes_cli.profiles" or ref.startswith("hermes_cli.profiles.")


MOVED_PROFILE_EXPORTS = frozenset("""
PROFILE_ROLES SETUP_ROLE current_profile_name drop_profile_role format_profile_label
get_active_profile get_active_profile_name get_profile_dir list_profile_names normalize_profile_name
parked_marker_path profile_exists profile_is_parked profile_is_standalone profile_matches_home
profile_root_for_env_home profiles_to_serve read_profile_meta resolve_profile_env set_active_profile
validate_alias_name validate_profile_name write_profile_meta _get_default_hermes_home _get_profiles_root
""".split())

_RUNTIME_ROOTS = ("agent", "gateway", "tui_gateway", "acp_adapter", "cron", "tools", "plugins")


def _import_bindings(tree: ast.AST) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    bindings[alias.asname] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                if alias.name != "*":
                    bindings[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return bindings


def _canonical_expr(node: ast.AST, bindings: dict[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        return bindings.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        parent = _canonical_expr(node.value, bindings)
        return f"{parent}.{node.attr}" if parent else None
    return None


def _moved_profile_facade_refs(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"), filename=str(path))
    bindings = _import_bindings(tree)
    offenders: set[tuple[int, str]] = set()
    prefix = "hermes_cli.profiles."

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "hermes_cli.profiles":
            for alias in node.names:
                if alias.name in MOVED_PROFILE_EXPORTS:
                    offenders.add((node.lineno, alias.name))
            continue

        if isinstance(node, ast.Attribute):
            canonical = _canonical_expr(node, bindings)
            if canonical and canonical.startswith(prefix):
                name = canonical[len(prefix):].split(".", 1)[0]
                if name in MOVED_PROFILE_EXPORTS:
                    offenders.add((node.lineno, name))

        if isinstance(node, ast.Call):
            if (
                len(node.args) >= 2
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "hermes_cli.profiles"
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value in MOVED_PROFILE_EXPORTS
            ):
                offenders.add((node.lineno, str(node.args[1].value)))
            if (
                _canonical_expr(node.func, bindings) == "getattr"
                and len(node.args) >= 2
                and _canonical_expr(node.args[0], bindings) == "hermes_cli.profiles"
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value in MOVED_PROFILE_EXPORTS
            ):
                offenders.add((node.lineno, str(node.args[1].value)))

    return sorted(offenders)


def _runtime_python_files():
    for root_name in _RUNTIME_ROOTS:
        root = REPO_ROOT / root_name
        if root.is_dir():
            yield from root.rglob("*.py")
    yield from REPO_ROOT.glob("*.py")


@pytest.mark.parametrize(
    "source",
    [
        "import hermes_cli.profiles\n",
        "import hermes_cli.profiles as profiles\n",
        "from hermes_cli.profiles import get_profile_dir\n",
        "from hermes_cli.profiles import get_profile_dir as profile_dir\n",
        "from hermes_cli import profiles\n",
        "from hermes_cli import profiles as profile_module\n",
        "from hermes_cli import (profiles,)\n",
        "from hermes_cli import (\n    profiles as profile_module,\n)\n",
    ],
)
def test_import_collector_detects_cli_profiles_spellings(
    tmp_path: Path, source: str
) -> None:
    probe = tmp_path / "probe.py"
    probe.write_text(source, encoding="utf-8")
    assert any(_is_cli_profiles_ref(ref) for ref in _imports(probe))


def test_import_collector_does_not_flag_other_cli_modules(tmp_path: Path) -> None:
    probe = tmp_path / "probe.py"
    probe.write_text("from hermes_cli import config\n", encoding="utf-8")
    assert not any(_is_cli_profiles_ref(ref) for ref in _imports(probe))


def test_gateway_does_not_depend_on_cli_profiles() -> None:
    offenders: list[tuple[str, str]] = []
    gateway_root = REPO_ROOT / "gateway"
    for path in gateway_root.rglob("*.py"):
        for ref in _imports(path):
            if _is_cli_profiles_ref(ref):
                offenders.append((str(path.relative_to(gateway_root)), ref))
    assert offenders == []


def test_profiles_domain_does_not_depend_on_cli() -> None:
    offenders: list[tuple[str, str]] = []
    profiles_root = REPO_ROOT / "profiles"
    for path in profiles_root.rglob("*.py"):
        for ref in _imports(path):
            if ref == "hermes_cli" or ref.startswith("hermes_cli."):
                offenders.append((str(path.relative_to(profiles_root)), ref))
    assert offenders == []


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("from hermes_cli.profiles import read_profile_meta\n", "read_profile_meta"),
        ("from hermes_cli import profiles as hp\nhp.write_profile_meta(home)\n", "write_profile_meta"),
        ("import hermes_cli.profiles\nhermes_cli.profiles.profile_exists('x')\n", "profile_exists"),
        ("_lazy('hermes_cli.profiles', 'get_profile_dir')(name)\n", "get_profile_dir"),
    ],
)
def test_moved_symbol_collector_detects_runtime_facade_use(
    tmp_path: Path, source: str, expected: str
) -> None:
    probe = tmp_path / "probe.py"
    probe.write_text(source, encoding="utf-8")
    assert any(name == expected for _line, name in _moved_profile_facade_refs(probe))


def test_moved_symbol_collector_allows_lifecycle_only_facade_use(tmp_path: Path) -> None:
    probe = tmp_path / "probe.py"
    probe.write_text(
        "from hermes_cli.profiles import list_profiles\n"
        "list_profiles()\n",
        encoding="utf-8",
    )
    assert _moved_profile_facade_refs(probe) == []


def test_runtime_consumers_use_canonical_profile_domain_owners() -> None:
    offenders: list[tuple[str, int, str]] = []
    for path in _runtime_python_files():
        for line, name in _moved_profile_facade_refs(path):
            offenders.append((str(path.relative_to(REPO_ROOT)), line, name))
    assert offenders == []
