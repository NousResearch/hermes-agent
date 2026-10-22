"""The optional AG-UI runtime and launcher survive declared package discovery."""

from importlib import import_module
from pathlib import Path

from setuptools.config.pyprojecttoml import read_configuration

from scripts.build.inputs import project_entries


ROOT = Path(__file__).resolve().parents[1]


def test_agui_console_entry_resolves_to_a_callable():
    entry = project_entries(ROOT / "pyproject.toml").get("hermes-agui")
    assert entry is not None, "the installed AG-UI launcher must be declared"
    module, name = entry.split(":")
    assert callable(getattr(import_module(module), name))


def test_declared_package_discovery_includes_the_adapter():
    configuration = read_configuration(str(ROOT / "pyproject.toml"), expand=True)
    packages = configuration["tool"]["setuptools"]["packages"]
    assert "agui_adapter" in packages
    assert "tests.agui_adapter" not in packages


def test_pm_recognizes_the_installed_agui_extra():
    import ag_ui.core
    import pm

    assert callable(ag_ui.core.RunAgentInput)
    assert pm.available("agui")
