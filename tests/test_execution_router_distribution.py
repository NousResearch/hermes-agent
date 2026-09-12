from __future__ import annotations

import os
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOC_SOURCE = "website/docs/developer-guide/execution-router-plugin.md"
DOC_WHEEL = "share/hermes-agent/docs/execution-router-plugin.md"
PUBLIC_MODULES = {
    "agent/execution_router.py",
    "agent/execution_router_reference.py",
}


def test_execution_router_distribution_declares_only_the_public_doc():
    config = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert config["tool"]["setuptools"]["data-files"] == {
        "share/hermes-agent/docs": [DOC_SOURCE]
    }


def _archive_members(path: Path) -> set[str]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return set(archive.namelist())
    with tarfile.open(path) as archive:
        return {
            name.split("/", 1)[1]
            for name in archive.getnames()
            if "/" in name
        }


def _extract(path: Path, destination: Path) -> Path:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            archive.extractall(destination)
        return destination
    with tarfile.open(path) as archive:
        archive.extractall(destination)
    roots = [item for item in destination.iterdir() if item.is_dir()]
    assert len(roots) == 1
    return roots[0]


def _assert_isolated_import(root: Path) -> None:
    code = """
from agent.execution_router import CONTRACT_VERSION, ExecutionKind, discover_execution_router_capabilities
from hermes_cli.plugins import get_plugin_manager
assert get_plugin_manager().get_execution_router_registration() is None
from agent.execution_router_reference import FirstEligibleExecutionRouterProviderV1
assert get_plugin_manager().get_execution_router_registration() is None
caps = discover_execution_router_capabilities()
assert CONTRACT_VERSION == '1.0'
assert tuple(kind.value for kind in caps.supported_execution_kinds) == ('main_turn', 'native_child', 'kanban_worker')
assert caps.operational_execution_kinds == ()
assert FirstEligibleExecutionRouterProviderV1.__module__ == 'agent.execution_router_reference'
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    result = subprocess.run(
        [sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, {str(root)!r});{code}"],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_built_archives_include_and_import_execution_router_contract_docs_and_fixture(tmp_path):
    wheel_value = os.environ.get("EXECUTION_ROUTER_TEST_WHEEL")
    sdist_value = os.environ.get("EXECUTION_ROUTER_TEST_SDIST")
    wheel_candidates = [Path(wheel_value)] if wheel_value else sorted((PROJECT_ROOT / "dist").glob("hermes_agent-*.whl"))
    sdist_candidates = [Path(sdist_value)] if sdist_value else sorted((PROJECT_ROOT / "dist").glob("hermes_agent-*.tar.gz"))
    if len(wheel_candidates) != 1 or len(sdist_candidates) != 1:
        pytest.skip("archive qualification requires one wheel and one sdist in dist/ or explicit paths")

    wheel = wheel_candidates[0]
    sdist = sdist_candidates[0]
    assert wheel.is_file() and sdist.is_file()

    wheel_members = _archive_members(wheel)
    assert PUBLIC_MODULES <= wheel_members
    assert any(name.endswith(f".data/data/{DOC_WHEEL}") for name in wheel_members)

    sdist_members = _archive_members(sdist)
    assert PUBLIC_MODULES | {DOC_SOURCE} <= sdist_members

    _assert_isolated_import(_extract(wheel, tmp_path / "wheel"))
    _assert_isolated_import(_extract(sdist, tmp_path / "sdist"))
