"""``hermes update`` must re-check the pip dependency of a web provider plugin (#108711).

A web provider plugin (``provides_web_providers``) declares its pip package once. The only path that
installed it was the ``hermes tools`` post_setup hook, which skips an already-importable package — so
after the first install nothing ever checked or moved it, and ``hermes update`` had no step looking at
plugin dependencies at all. These tests pin the refresh contract: a dependency of a provider that is in
use (selected in config, or already installed) is (re)installed with an upgrade; a plugin the user never
selected and never installed is left alone, because installing it would silently change provider
resolution.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import tools.lazy_deps as ld


def _write_web_plugin(
    root: Path, name: str, *, providers: list[str], deps: list[str] | None = None
) -> Path:
    """Write a minimal web-provider plugin manifest under *root* and return its directory."""
    plugin_dir = root / name
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "__init__.py").write_text("", encoding="utf-8")
    lines = ["name: %s" % name, "version: 1.0.0", "kind: backend", "provides_web_providers:"]
    lines += ["  - %s" % p for p in providers]
    if deps is not None:
        lines.append("pip_dependencies:")
        lines += ['  - "%s"' % d for d in deps]
    (plugin_dir / "plugin.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return plugin_dir


@pytest.fixture
def bundled_web_root(tmp_path, monkeypatch):
    """An empty bundled-plugins root, so only manifests the test writes are discovered."""
    root = tmp_path / "bundled-plugins"
    (root / "web").mkdir(parents=True)
    monkeypatch.setattr("hermes_cli.plugins.get_bundled_plugins_dir", lambda: root)
    return root / "web"


@pytest.fixture
def installed_spy(monkeypatch):
    """Spy on the installer; ``presence`` decides whether a declared dep reads as already installed."""
    calls: list[tuple[list[str], dict]] = []

    def _install(specs, **kwargs):
        calls.append((list(specs), kwargs))
        return ld.InstallSpecsResult(ok=True, command="uv pip install " + " ".join(specs))

    monkeypatch.setattr(ld, "install_specs", _install)
    return calls


@pytest.fixture
def presence(monkeypatch):
    """Control the "is this spec already installed?" probe the refresh uses."""
    state = {"installed": False}
    monkeypatch.setattr(ld, "spec_installed", lambda spec: state["installed"])
    return state


def test_selected_web_provider_missing_dependency_is_installed(bundled_web_root, monkeypatch, installed_spy, presence):
    """The reported case: the dep is gone (venv rebuilt / never installed) but the provider is in use."""
    _write_web_plugin(bundled_web_root, "ddgs", providers=["ddgs"], deps=["ddgs>=9,<10"])
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {"web": {"backend": "ddgs"}})
    presence["installed"] = False

    from hermes_cli.update_cmd_plugins import _refresh_active_web_provider_dependencies

    results = _refresh_active_web_provider_dependencies()

    assert [specs for specs, _ in installed_spy] == [["ddgs>=9,<10"]]
    assert installed_spy[0][1]["upgrade"] is True
    assert results == {"web/ddgs": "refreshed"}


def test_installed_but_stale_web_provider_dependency_is_reapplied(bundled_web_root, monkeypatch, installed_spy, presence):
    """The other half: the package is installed but stale, and no config key selects it."""
    _write_web_plugin(bundled_web_root, "ddgs", providers=["ddgs"], deps=["ddgs>=9,<10"])
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {})
    presence["installed"] = True

    from hermes_cli.update_cmd_plugins import _refresh_active_web_provider_dependencies

    _refresh_active_web_provider_dependencies()

    assert [specs for specs, _ in installed_spy] == [["ddgs>=9,<10"]]
    assert installed_spy[0][1]["upgrade"] is True


def test_unused_optional_web_provider_dependency_is_left_alone(
    bundled_web_root, monkeypatch, installed_spy, presence
):
    """Never activate an optional provider: that would change which provider resolves."""
    _write_web_plugin(bundled_web_root, "trafilatura", providers=["trafilatura"], deps=["trafilatura>=2,<3"])
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {"web": {"backend": "firecrawl"}})
    presence["installed"] = False

    from hermes_cli.update_cmd_plugins import _refresh_active_web_provider_dependencies

    assert _refresh_active_web_provider_dependencies() == {}
    assert installed_spy == []


def test_bundled_ddgs_plugin_is_discovered_as_pip_backed():
    """Contract with the shipped manifest: the bundled ddgs provider declares what update re-checks."""
    from hermes_cli.update_cmd_plugins import _web_provider_plugin_dependencies

    found = {plugin.label: plugin for plugin in _web_provider_plugin_dependencies()}
    assert "web/ddgs" in found, sorted(found)
    package_names = {spec.split(">")[0].split("<")[0].split("=")[0].strip() for spec in found["web/ddgs"].dependencies}
    assert "ddgs" in package_names
