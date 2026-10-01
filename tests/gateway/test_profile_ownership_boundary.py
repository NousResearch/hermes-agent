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
