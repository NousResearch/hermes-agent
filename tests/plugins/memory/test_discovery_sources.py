"""Discovery parity for out-of-tree memory providers.

Upstream policy closed ``plugins/memory/`` to new providers, so every new
memory backend now lives outside this tree. These tests cover the two sources
that reach it — project-local directories and pip entry points — and the
integration points a directory install gets for free but a pip install
historically did not: the dashboard config panel, the provider's CLI
subcommands, and the ``memory.provider`` dropdown.
"""

from __future__ import annotations

import importlib.metadata
import sys
import textwrap
from pathlib import Path

import pytest

import plugins.memory as memory_plugins

PROVIDER_SOURCE = """\
from agent.memory_provider import MemoryProvider


class Provider(MemoryProvider):
    @property
    def name(self):
        return "{name}"

    def is_available(self):
        return True

    def initialize(self, *a, **kw):
        pass

    def get_tool_schemas(self):
        return []


def register(ctx):
    ctx.register_memory_provider(Provider())
"""


class FakeEntryPoint:
    """Mirrors the importlib.metadata EntryPoint surface discovery uses."""

    group = "hermes_agent.memory_providers"

    def __init__(self, name, value):
        self.name = name
        self.value = value

    def load(self):
        import importlib

        module_name, _, attr = self.value.partition(":")
        module = importlib.import_module(module_name)
        return getattr(module, attr) if attr else module


class FakeEntryPoints(list):
    def select(self, *, group):
        return [ep for ep in self if ep.group == group]


@pytest.fixture
def entry_points(monkeypatch):
    """Install a replaceable entry-point set for the memory group."""
    registry = FakeEntryPoints()
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: registry)
    return registry


def _write_provider_dir(root: Path, name: str) -> Path:
    provider = root / name
    provider.mkdir(parents=True)
    (provider / "__init__.py").write_text(PROVIDER_SOURCE.format(name=name), encoding="utf-8")
    return provider


# ---------------------------------------------------------------------------
# Project-local providers
# ---------------------------------------------------------------------------


def test_project_dir_is_ignored_without_opt_in(tmp_path, monkeypatch):
    """A repo you merely cd into must not be able to offer a memory backend."""
    _write_provider_dir(tmp_path / ".hermes" / "plugins", "projectmem")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HERMES_ENABLE_PROJECT_PLUGINS", raising=False)

    assert "projectmem" not in memory_plugins.list_memory_provider_names()
    assert memory_plugins.find_provider_dir("projectmem") is None


def test_project_dir_is_discovered_when_opted_in(tmp_path, monkeypatch):
    provider = _write_provider_dir(tmp_path / ".hermes" / "plugins", "projectmem")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", "1")

    assert "projectmem" in memory_plugins.list_memory_provider_names()
    assert memory_plugins.find_provider_dir("projectmem") == provider


def test_bundled_still_wins_over_project(tmp_path, monkeypatch):
    """Precedence here is bundled-first, the reverse of the general
    PluginManager's later-wins order. A provider is activated by name, so a
    directory dropped into the working tree must not be able to shadow a
    shipped one and silently redirect the agent's memory."""
    _write_provider_dir(tmp_path / ".hermes" / "plugins", "honcho")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", "1")

    resolved = memory_plugins.find_provider_dir("honcho")
    assert resolved == Path(memory_plugins.__file__).parent / "honcho"


# ---------------------------------------------------------------------------
# Pip entry-point providers
# ---------------------------------------------------------------------------


def test_entry_point_provider_is_listed(entry_points, tmp_path, monkeypatch):
    """list_memory_provider_names() fills the dashboard's memory.provider
    dropdown. Enumerating entry points reads distribution metadata without
    executing any of it, so this stays safe to call at import time."""
    entry_points.append(FakeEntryPoint("pipmem", "pipmem_pkg"))
    assert "pipmem" in memory_plugins.list_memory_provider_names()


def test_find_provider_dir_resolves_a_package_entry_point(entry_points, tmp_path, monkeypatch):
    """Without a directory, a pip-installed provider silently loses its
    dashboard config panel and its `hermes <provider>` subcommands — both are
    read from disk rather than imported."""
    package = tmp_path / "pipmem_pkg"
    package.mkdir()
    (package / "__init__.py").write_text(PROVIDER_SOURCE.format(name="pipmem"), encoding="utf-8")
    (package / "config_schema.py").write_text("CONFIG_SCHEMA = None\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    entry_points.append(FakeEntryPoint("pipmem", "pipmem_pkg:register"))

    assert memory_plugins.find_provider_dir("pipmem") == package


def test_resolving_an_entry_point_does_not_import_it(entry_points, tmp_path, monkeypatch):
    """Discovery runs before the operator has chosen a provider. Importing
    every installed candidate would execute third-party code on the strength of
    a package merely being present."""
    package = tmp_path / "sideeffect_pkg"
    package.mkdir()
    (package / "__init__.py").write_text(
        textwrap.dedent(
            """\
            import pathlib
            pathlib.Path(__file__).with_name("IMPORTED").write_text("x")
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    entry_points.append(FakeEntryPoint("sideeffect", "sideeffect_pkg"))

    assert memory_plugins.find_provider_dir("sideeffect") == package
    assert not (package / "IMPORTED").exists()
    assert "sideeffect_pkg" not in sys.modules


def test_bare_module_entry_point_has_no_directory(entry_points, tmp_path, monkeypatch):
    """A single-file provider has nowhere to put a sibling config_schema.py, so
    it resolves to None rather than handing back the whole site-packages root."""
    (tmp_path / "flatmem.py").write_text(PROVIDER_SOURCE.format(name="flatmem"), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    entry_points.append(FakeEntryPoint("flatmem", "flatmem"))

    assert memory_plugins.find_provider_dir("flatmem") is None


# ---------------------------------------------------------------------------
# Registration surface
# ---------------------------------------------------------------------------


def test_a_secondary_registration_cannot_cost_the_provider(tmp_path, monkeypatch):
    """register_auxiliary_task used to raise AttributeError on the collector —
    which the loader caught, discarded the registered provider, and replaced
    with a bare second instance built by the subclass scan. A silent downgrade
    that looked like success."""
    plugins_root = tmp_path / "plugins"
    provider = _write_provider_dir(plugins_root, "auxmem")
    (provider / "__init__.py").write_text(
        PROVIDER_SOURCE.format(name="auxmem").replace(
            "    ctx.register_memory_provider(Provider())\n",
            "    instance = Provider()\n"
            "    instance.marked = True\n"
            "    ctx.register_memory_provider(instance)\n"
            "    ctx.register_auxiliary_task(\n"
            "        'auxmem_filter', display_name='Aux', description='d'\n"
            "    )\n",
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    loaded = memory_plugins.load_memory_provider("auxmem")
    assert loaded is not None
    assert loaded.name == "auxmem"
    # The instance register() handed over, not a replacement.
    assert getattr(loaded, "marked", False)


def test_activation_is_not_gated_on_plugins_enabled(tmp_path, monkeypatch):
    """Memory providers are activated by naming them in memory.provider. Using
    a real PluginContext for secondary registrations must not start also
    requiring the plugin in plugins.enabled — that would break every existing
    user-installed provider."""
    _write_provider_dir(tmp_path / "plugins", "gatedmem")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    assert memory_plugins.load_memory_provider("gatedmem") is not None
