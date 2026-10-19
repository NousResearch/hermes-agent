"""Contract: deleting an MCP entry that never connected still leaves a log line.

Regression for the qmt/8765 shape — the reconciler's ``removed`` comes from the LIVE server set, so
an entry whose server never came up (dead port, typo, missing token) produced no ``removed`` line
when it was deleted minutes later. The config-key diff is advisory only: it logs, dedupes by key
set, and never changes reconcile behaviour.

The first-change-after-start case is the one that matters: the key baseline is recorded on the
first tick, BEFORE the "signature unchanged" early return, so a change happening right after
startup is visible instead of being swallowed.
"""
from __future__ import annotations

import contextlib
import logging
import os

import pytest

from gateway import run_profile_reconcile as rpr


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """Drive the reconciler's tick with a config file whose signature we control."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("mcp_servers: {}\n")

    servers: dict = {}
    known: set = set()
    state = {"n": 0, "fail": False}

    def _load():
        if state["fail"]:
            raise RuntimeError("config unreadable")
        return dict(servers)

    monkeypatch.setattr("hermes_cli.config.get_config_path", lambda: cfg_path)
    monkeypatch.setattr("tools.mcp_tool_config._load_mcp_config", _load)
    monkeypatch.setattr(
        "tools.mcp_tool_discovery.reconcile_mcp_servers_with_config",
        lambda: {"removed": [], "added": [], "pending": []},
    )
    monkeypatch.setattr("tools.mcp_tool_discovery.get_registered_mcp_server_names", lambda: set(known))
    monkeypatch.setattr("tools.mcp_oauth.suppress_interactive_oauth", contextlib.nullcontext)

    tick_fn = rpr._mcp_config_reconciler(None)

    def tick():
        # Bump the file signature so the reconciler actually runs this tick.
        state["n"] += 1
        cfg_path.write_text("mcp_servers: {}\n" + "#" * state["n"] + "\n")
        stamp = 1_700_000_000 + state["n"] * 10
        os.utime(cfg_path, (stamp, stamp))
        tick_fn()  # one closure for the whole test: its state IS the thing under test

    return {"tick": tick, "servers": servers, "known": known, "state": state}


def _lines(caplog):
    return [rec.getMessage() for rec in caplog.records if rec.name == rpr.__name__]


def test_key_baseline_is_not_logged_as_change(harness, caplog):
    caplog.set_level(logging.INFO, logger=rpr.__name__)
    harness["servers"].update({"alpha": {}, "beta": {}})
    harness["known"].update({"alpha", "beta"})
    harness["tick"]()
    assert not [ln for ln in _lines(caplog) if "config keys changed" in ln], _lines(caplog)


def test_first_change_after_start_is_logged(harness, caplog):
    """The case the early return used to swallow: baseline on tick 1, change on tick 2."""
    caplog.set_level(logging.INFO, logger=rpr.__name__)
    harness["servers"].update({"alpha": {}})
    harness["known"].add("alpha")
    harness["tick"]()  # baseline
    caplog.clear()

    harness["servers"]["gamma"] = {}
    harness["known"].add("gamma")
    harness["tick"]()  # first change after start
    assert [ln for ln in _lines(caplog) if "config keys changed" in ln and "gamma" in ln], _lines(caplog)


def test_deleting_never_connected_entry_right_after_start_is_logged(harness, caplog):
    caplog.set_level(logging.INFO, logger=rpr.__name__)
    # `dead` is configured but never registered (the qmt/8765 shape).
    harness["servers"].update({"alive": {}, "dead": {}})
    harness["known"].add("alive")
    harness["tick"]()  # baseline; surfaces the never-connected entry once
    assert any("configured but not connected" in ln and "dead" in ln for ln in _lines(caplog)), _lines(caplog)
    caplog.clear()

    del harness["servers"]["dead"]
    harness["tick"]()  # first change after start — the line the old code never emitted
    lines = _lines(caplog)
    assert len([ln for ln in lines if "config keys changed" in ln and "'dead'" in ln]) == 1, lines

    caplog.clear()
    harness["tick"]()  # unchanged set never repeats
    assert not [ln for ln in _lines(caplog) if "'dead'" in ln], _lines(caplog)


def test_unchanged_key_set_does_not_relog(harness, caplog):
    caplog.set_level(logging.INFO, logger=rpr.__name__)
    harness["servers"].update({"alpha": {}})
    harness["known"].add("alpha")
    harness["tick"]()
    harness["servers"]["gamma"] = {}
    harness["known"].add("gamma")
    harness["tick"]()
    caplog.clear()

    # Same key set, different file signature — dedupe is by configured keys, not by mtime.
    harness["tick"]()
    harness["tick"]()
    assert not [ln for ln in _lines(caplog) if "config keys changed" in ln], _lines(caplog)


def test_unconnected_but_still_configured_is_reported(harness, caplog):
    caplog.set_level(logging.INFO, logger=rpr.__name__)
    harness["servers"].update({"alpha": {}})
    harness["known"].add("alpha")
    harness["tick"]()
    caplog.clear()

    harness["servers"]["shadow"] = {}  # configured, never registers
    harness["tick"]()
    lines = _lines(caplog)
    assert any("configured but not connected" in ln and "shadow" in ln for ln in lines), lines


def test_unconnected_hint_fails_quiet_when_registry_unavailable(harness, caplog, monkeypatch):
    """A broken registry import must not report every configured server as unconnected."""
    caplog.set_level(logging.INFO, logger=rpr.__name__)
    harness["servers"].update({"alpha": {}, "beta": {}})
    harness["known"].update({"alpha", "beta"})
    harness["tick"]()
    caplog.clear()

    def _boom():
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr("tools.mcp_tool_discovery.get_registered_mcp_server_names", _boom)
    harness["servers"]["shadow"] = {}
    harness["tick"]()
    assert not [ln for ln in _lines(caplog) if "configured but not connected" in ln], _lines(caplog)


def test_disabled_entries_are_not_counted(harness, caplog):
    caplog.set_level(logging.INFO, logger=rpr.__name__)
    harness["servers"].update({"alpha": {}})
    harness["known"].add("alpha")
    harness["tick"]()
    caplog.clear()

    harness["servers"]["off"] = {"enabled": False}
    harness["tick"]()
    assert not [ln for ln in _lines(caplog) if "config keys changed" in ln], _lines(caplog)


def test_unreadable_config_at_baseline_does_not_become_an_empty_snapshot(harness, caplog):
    """A failed read is unknown, not empty — otherwise the next good read looks like 'all added'."""
    caplog.set_level(logging.INFO, logger=rpr.__name__)
    harness["servers"].update({"alpha": {}, "beta": {}, "gamma": {}})
    harness["known"].update({"alpha", "beta", "gamma"})

    harness["state"]["fail"] = True
    harness["tick"]()  # baseline attempt fails
    harness["state"]["fail"] = False
    harness["tick"]()  # baseline recorded now — must NOT report three additions
    lines = _lines(caplog)
    assert not [ln for ln in lines if "config keys changed" in ln], lines

    caplog.clear()
    del harness["servers"]["gamma"]
    harness["tick"]()
    removed = [ln for ln in _lines(caplog) if "config keys changed" in ln]
    assert len(removed) == 1 and "'gamma'" in removed[0], removed


def test_unreadable_config_midrun_keeps_last_good_snapshot(harness, caplog):
    caplog.set_level(logging.INFO, logger=rpr.__name__)
    harness["servers"].update({"alpha": {}, "beta": {}})
    harness["known"].update({"alpha", "beta"})
    harness["tick"]()  # baseline
    caplog.clear()

    harness["state"]["fail"] = True
    harness["tick"]()  # unreadable: log nothing, keep snapshot
    harness["state"]["fail"] = False
    harness["tick"]()  # recovered, same key set: still nothing
    lines = _lines(caplog)
    assert not [ln for ln in lines if "config keys changed" in ln], lines
