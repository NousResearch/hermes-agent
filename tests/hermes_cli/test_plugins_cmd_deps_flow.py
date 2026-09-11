"""hermes plugins install: python-deps consent + try-resolve flow.

Settled 2026-09-02 (.hermes/plans/2026-09-02_164500-plugin-deps-workspace-
union.md): install drops the folder, prompts y/n for declared python
deps, resolves through the pm workspace union, and ENABLES ONLY IF THE
DEPS RESOLVE — a conflicting plugin stays installed-but-disabled with
the resolver's reason.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

import hermes_cli.plugins_cmd as pc


class _Console:
    def __init__(self):
        self.lines: list[str] = []

    def print(self, *args, **kwargs):
        for arg in args:
            self.lines.append(str(arg))


@pytest.fixture
def resolve_env(monkeypatch):
    """Lazy installs on + no pre-existing enabled members, so the resolve
    runs the would-be union with just this plugin."""
    import importlib
    import sys

    if "pm.ensure" not in sys.modules:
        importlib.import_module("pm.ensure")
    ensure_mod = sys.modules["pm.ensure"]
    monkeypatch.setattr(ensure_mod, "lazy_installs_allowed", lambda: True)


def _legacy_plugin(target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    (target / "plugin.yaml").write_text(
        "name: dep-plug\npip_dependencies:\n  - \"somepkg>=1,<2\"\n",
        encoding="utf-8",
    )


def test_no_deps_short_circuits_true(tmp_path):
    plug = tmp_path / "plain"
    plug.mkdir()
    (plug / "plugin.yaml").write_text("name: plain\n", encoding="utf-8")
    ok, reason = pc._install_plugin_python_deps(
        {"name": "plain", "python_dependencies": []}, plug, _Console()
    )
    assert ok is True and reason is None


def test_noninteractive_skips_install(tmp_path, monkeypatch, resolve_env):
    plug = tmp_path / "dep-plug"
    _legacy_plugin(plug)
    monkeypatch.setattr(pc.sys.stdin, "isatty", lambda: False, raising=False)
    called = []
    monkeypatch.setattr(
        "pm.client.sync_venv", lambda *a, **k: called.append(a)
    )
    ok, reason = pc._install_plugin_python_deps(
        {"name": "dep-plug", "python_dependencies": ["somepkg>=1,<2"]},
        plug,
        _Console(),
    )
    assert ok is False
    assert "non-interactive" in reason
    assert not called  # nothing installed, nothing enabled


def test_decline_skips_install(tmp_path, monkeypatch, resolve_env):
    plug = tmp_path / "dep-plug"
    _legacy_plugin(plug)
    monkeypatch.setattr(pc.sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(pc.sys.stdout, "isatty", lambda: True, raising=False)
    monkeypatch.setattr("builtins.input", lambda *a: "n")
    called = []
    monkeypatch.setattr(
        "pm.client.sync_venv", lambda *a, **k: called.append(a)
    )
    ok, reason = pc._install_plugin_python_deps(
        {"name": "dep-plug", "python_dependencies": ["somepkg>=1,<2"]},
        plug,
        _Console(),
    )
    assert ok is False
    assert "declined" in reason
    assert not called


def test_conflict_surfaces_at_admission_not_consent(tmp_path, monkeypatch, resolve_env):
    home = tmp_path / ".hermes"
    (home / "plugins").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    plug = tmp_path / "dep-plug"
    _legacy_plugin(plug)
    monkeypatch.setattr(pc.sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(pc.sys.stdout, "isatty", lambda: True, raising=False)
    monkeypatch.setattr("builtins.input", lambda *a: "y")

    # Consent succeeds — resolution moved OUT of consent (C13):
    ok, reason = pc._install_plugin_python_deps(
        {"name": "dep-plug", "python_dependencies": ["somepkg==9.9.9"]},
        plug,
        _Console(),
    )
    assert ok is True and reason is None

    # The conflict surfaces when the enable COMMITS, with config untouched:
    from pm import client

    def boom(*a, **k):
        raise RuntimeError(
            "uv lock exited 1: Because dep-plug depends on somepkg==9.9.9 "
            "and hermes-agent depends on somepkg==1.0.0, unsatisfiable"
        )

    monkeypatch.setattr(client, "sync_venv", boom)
    from hermes_cli import plugins_admission as adm

    with pytest.raises(adm.AdmissionRefused) as excinfo:
        adm.admit_plugin_set_change(
            {"dep-plug"}, set(), active_plugins_dir=home / "plugins", extra_dirs=[plug]
        )
    assert "unsatisfiable" in str(excinfo.value)
    from hermes_cli.config import load_config

    assert not ((load_config().get("plugins") or {}).get("enabled"))


def test_success_consents_without_plugin_dir_writes(tmp_path, monkeypatch, resolve_env):
    plug = tmp_path / "dep-plug"
    _legacy_plugin(plug)
    monkeypatch.setattr(pc.sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(pc.sys.stdout, "isatty", lambda: True, raising=False)
    monkeypatch.setattr("builtins.input", lambda *a: "y")
    synced = []
    monkeypatch.setattr("pm.client.sync_venv", lambda *a, **k: synced.append(a))
    ok, reason = pc._install_plugin_python_deps(
        {"name": "dep-plug", "python_dependencies": ["somepkg>=1,<2"]},
        plug,
        _Console(),
    )
    assert ok is True and reason is None
    assert not synced  # consent only — the resolve runs in the admission transaction
    # no plugin-dir writes: legacy deps stage as virtual members in pm's workspace
    assert not (plug / "pyproject.toml").exists()
