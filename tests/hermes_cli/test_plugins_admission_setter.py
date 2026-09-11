"""C13 enable admission: every UI path's proposed enabled/disabled sets go
through ONE authority — the candidate union resolves against the active
environment and the config commits inside pm's single locked transaction
(sync's before_publish hook). Refusal (dep conflict OR config-write
failure) publishes nothing: previous config bytes and previous environment
stay exactly in place. Real temp HERMES_HOME — no live user writes."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_cli.plugins_cmd as pc
from hermes_cli import plugins_admission as adm


class _Console:
    def __init__(self):
        self.lines: list[str] = []

    def print(self, *args, **kwargs):
        for arg in args:
            self.lines.append(str(arg))


@pytest.fixture
def plugin_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME: config.yaml, plugins/, facts all under tmp."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "plugins").mkdir()
    return home


def _enabled_set(home) -> set:
    from hermes_cli.config import load_config

    cfg = load_config()
    return set(((cfg.get("plugins") or {}).get("enabled")) or [])


def _write_sets(home, enabled=(), disabled=()):
    pc._write_config_value("plugins", "enabled", sorted(enabled))
    pc._write_config_value("plugins", "disabled", sorted(disabled))


@pytest.fixture
def sync_calls(monkeypatch):
    """Stub the admission caller seam; real engine transactions are tested below."""
    from pm import client
    calls = []

    def _sync(extras=None, *, explicit=False, plugin_dirs=None, before_publish=None):
        members = plugin_dirs() if callable(plugin_dirs) else plugin_dirs
        calls.append(list(members or []))
        if before_publish is not None:
            before_publish()  # same contract: config commits under the sync

    monkeypatch.setattr(client, "sync_venv", _sync)
    return calls


def _dep_plugin(home) -> Path:
    d = home / "plugins" / "dep-plug"
    d.mkdir(exist_ok=True)
    (d / "plugin.yaml").write_text(
        'name: dep-plug\npython_dependencies:\n  - "somepkg>=1"\n', encoding="utf-8"
    )
    return d


def _entry(dir_path: Path):
    return ("dep-plug", "1.0", "test plugin", "user", dir_path, "dep-plug")


def _discovery(monkeypatch, *entries):
    monkeypatch.setattr(pc, "_discover_all_plugins", lambda: list(entries))
    monkeypatch.setattr(pc, "_declared_capabilities_for_key", lambda key: [])


def _refusing_sync(monkeypatch, message="uv lock exited 1: unsatisfiable"):
    from pm import client

    def _boom(*a, **k):
        raise RuntimeError(message)

    monkeypatch.setattr(client, "sync_venv", _boom)


# ── cmd_enable (CLI path) ────────────────────────────────────────────────────


def test_enable_refusal_keeps_config(plugin_home, monkeypatch, sync_calls):
    plug = _dep_plugin(plugin_home)
    _discovery(monkeypatch, _entry(plug))
    _refusing_sync(monkeypatch)
    console = _Console()
    monkeypatch.setattr(pc, "_console", lambda: console)
    with pytest.raises(adm.AdmissionRefused):
        pc.cmd_enable("dep-plug", allow_tool_override=False)
    assert _enabled_set(plugin_home) == set()  # config untouched
    assert any("refused" in line for line in console.lines)


def test_enable_success_commits_candidate_union_once(plugin_home, monkeypatch, sync_calls):
    plug = _dep_plugin(plugin_home)
    _discovery(monkeypatch, _entry(plug))
    pc.cmd_enable("dep-plug", allow_tool_override=False)
    assert sync_calls == [[plug]]  # exactly ONE sync: the candidate union
    assert _enabled_set(plugin_home) == {"dep-plug"}


def test_enable_removal_filters_candidate(plugin_home, monkeypatch, sync_calls):
    other = plugin_home / "plugins" / "other"
    other.mkdir()
    (other / "pyproject.toml").write_text("[project]\nname='o'\n", encoding="utf-8")
    _write_sets(plugin_home, enabled=["other"])
    plug = _dep_plugin(plugin_home)
    _discovery(monkeypatch, _entry(plug))
    pc.cmd_enable("dep-plug", allow_tool_override=False)
    # proposed-sets driven: 'other' stays enabled (it IS in the candidate set)
    assert sync_calls[-1] and set(sync_calls[-1]) == {other, plug}
    assert _enabled_set(plugin_home) == {"other", "dep-plug"}


def test_deps_free_enable_does_not_churn_venv(plugin_home, monkeypatch, sync_calls):
    plain = plugin_home / "plugins" / "plain"
    plain.mkdir()
    (plain / "plugin.yaml").write_text("name: plain\n", encoding="utf-8")
    _discovery(monkeypatch, ("plain", "1.0", "p", "user", plain, "plain"))
    pc.cmd_enable("plain", allow_tool_override=False)
    assert sync_calls == [[]]  # no members: cheap stamp check only


# ── composite UI + dashboard paths ──────────────────────────────────────────


def test_composite_selection_refusal_not_saved(plugin_home, monkeypatch, sync_calls):
    plug = _dep_plugin(plugin_home)
    _discovery(monkeypatch, _entry(plug))
    _refusing_sync(monkeypatch, "conflict")
    with pytest.raises(adm.AdmissionRefused):
        pc._persist_plugin_selection(["dep-plug"], {0}, set())
    assert _enabled_set(plugin_home) == set()


def test_dashboard_enable_refusal_reports_not_ok(plugin_home, monkeypatch, sync_calls):
    plug = _dep_plugin(plugin_home)
    monkeypatch.setattr(pc, "_resolve_plugin_key", lambda name: "dep-plug")
    _refusing_sync(monkeypatch, "conflict")
    result = pc.dashboard_set_agent_plugin_enabled("dep-plug", enabled=True)
    assert result["ok"] is False
    assert _enabled_set(plugin_home) == set()


# ── the before_publish hook: config commits under the lock, undo on facts failure ─


def test_config_commit_undo_restores_previous_bytes(plugin_home):
    _write_sets(plugin_home, enabled=["old"])
    config_path = plugin_home / "config.yaml"
    previous = config_path.read_bytes()

    undo = adm._config_commit({"dep-plug"}, set())
    assert _enabled_set(plugin_home) == {"dep-plug"}  # committed exactly once
    undo()  # facts write failed afterwards → restore
    assert config_path.read_bytes() == previous


def test_facts_failure_triggers_undo_inside_one_transaction(plugin_home, monkeypatch):
    """Real sync_venv: apply stages → before_publish commits config → facts
    write fails → undo restores the previous config bytes atomically; the
    receipt records the failure."""
    import importlib
    import sys

    if "pm.ensure" not in sys.modules:
        importlib.import_module("pm.ensure")
    ensure = sys.modules["pm.ensure"]

    rdir = plugin_home / "receipts"
    monkeypatch.setattr("pm.receipt._receipt_dir", lambda: rdir)
    plug = _dep_plugin(plugin_home)
    _write_sets(plugin_home, enabled=["old"])
    previous = (plugin_home / "config.yaml").read_bytes()

    monkeypatch.setattr(ensure, "_runtime_state_matches", lambda fact, stamp: False)
    monkeypatch.setattr(ensure, "_facts", lambda: {})
    monkeypatch.setattr(
        ensure, "get_package",
        lambda name: SimpleNamespace(
            expected_stamp=lambda enabled, **kw: "stamp1", apply=lambda enabled, **kw: {}
        ),
    )

    order = []

    def before_publish():
        order.append("before_publish")
        return adm._config_commit({"dep-plug"}, set())

    facts_path = plugin_home / "runtime" / "facts.json"
    ensure.Facts(facts_path).record_state("venv", "previous-stamp", [])
    previous_facts = facts_path.read_bytes()
    monkeypatch.setattr(ensure.paths, "runtime_facts_path", lambda: facts_path)
    monkeypatch.setattr(ensure.paths, "repo_root", lambda: plugin_home / "runtime")

    def fail_publication(self, *args, **kwargs):
        order.append("record_state")
        raise OSError("facts disk full")

    monkeypatch.setattr(ensure.Facts, "record_state", fail_publication)
    with pytest.raises(OSError, match="facts disk full"):
        ensure.sync_venv(explicit=True, plugin_dirs=[plug], before_publish=before_publish)
    assert order == ["before_publish", "record_state"]  # config committed under the lock first
    assert (plugin_home / "config.yaml").read_bytes() == previous  # undone atomically
    assert facts_path.read_bytes() == previous_facts
    latest = json.loads((rdir / "latest.json").read_text(encoding="utf-8-sig"))
    assert latest["outcome"] == "failed"


# ── pm receipt invariant: every sync outcome writes a receipt ───────────────


def test_lazy_refusal_writes_failed_receipt(plugin_home, monkeypatch):
    import importlib
    import sys

    if "pm.ensure" not in sys.modules:
        importlib.import_module("pm.ensure")
    ensure = sys.modules["pm.ensure"]
    rdir = plugin_home / "receipts"
    monkeypatch.setattr("pm.receipt._receipt_dir", lambda: rdir)
    monkeypatch.setattr(ensure, "lazy_installs_allowed", lambda: False)
    monkeypatch.setattr(ensure, "_runtime_state_matches", lambda fact, stamp: False)
    with pytest.raises(Exception):
        ensure.sync_venv(extras=["somepkg"])
    latest = json.loads((rdir / "latest.json").read_text(encoding="utf-8-sig"))
    assert latest["kind"] == "sync"
    assert latest["outcome"] == "failed"


def test_noop_sync_writes_ok_receipt_rebuild_false(plugin_home, monkeypatch):
    import importlib
    import sys

    if "pm.ensure" not in sys.modules:
        importlib.import_module("pm.ensure")
    ensure = sys.modules["pm.ensure"]
    rdir = plugin_home / "receipts"
    monkeypatch.setattr("pm.receipt._receipt_dir", lambda: rdir)
    monkeypatch.setattr(ensure, "_runtime_state_matches", lambda fact, stamp: True)
    committed = []
    ensure.sync_venv(extras=[], before_publish=lambda: committed.append(True))
    assert committed == [True], "unchanged dependencies must still commit a plugin selection"
    latest = json.loads((rdir / "latest.json").read_text(encoding="utf-8-sig"))
    assert latest["outcome"] == "ok"
    assert latest["venv_rebuild"]["ok"] is False
    assert latest["venv_rebuild"]["reason"] == "already in sync"
