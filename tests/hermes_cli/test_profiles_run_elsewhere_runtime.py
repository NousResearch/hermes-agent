"""Host processes that were already running, or that start work without building an agent, do not
run a profile a plugin runs elsewhere.

``AIAgent.__init__`` refuses a marked profile, but four paths never reach it or reach it late:

- the cron ticker (gateway, Desktop, an external provider's built-in fallback) fires ``no_agent``
  script jobs without an agent, and an agent job's occurrence is consumed before the refusal;
- an external provider's inbound fire (``claim_fire``) does the same without a tick;
- an agent built before the profile was marked (a cached gateway or dashboard session agent)
  runs later turns without being constructed again;
- the dashboard's deferred agent build starts the profile's MCP discovery before it builds the
  agent, and ``hermes gateway run`` starts plugin and hook discovery before the gateway refuses.
"""
from __future__ import annotations

import errno
import os
import sys
import threading
import types
from argparse import Namespace
from pathlib import Path

import pytest

from hermes_cli import profiles_run_elsewhere as pre

PLUGIN = "sandbox-plugin"


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A Hermes root (the default profile) with the named profiles ``ops`` and ``dev``; none marked."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    for name in ("ops", "dev"):
        profile = home / "profiles" / name
        profile.mkdir(parents=True)
        (profile / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _deny_marker_folder(monkeypatch):
    """What the backend the plugin runs sees: the marker folder cannot be listed."""
    real_listdir = os.listdir

    def listdir(path):
        if Path(path).name == pre.RUN_ELSEWHERE_DIRNAME:
            raise PermissionError(errno.EACCES, "Access is denied")
        return real_listdir(path)

    monkeypatch.setattr(pre.os, "listdir", listdir)


class _OneCycle(threading.Event):
    """A stop event that ends a ticker after its first cycle (no timing involved)."""

    def wait(self, timeout=None):
        self.set()
        return True


def _record_ticks(monkeypatch) -> list[Path]:
    """Homes whose store a tick was admitted for (the lock, the due scan and the jobs come after)."""
    from hermes_constants import get_hermes_home

    ticked: list[Path] = []
    monkeypatch.setattr("cron.scheduler_tick._tick_admitted",
                        lambda *a, **k: ticked.append(Path(get_hermes_home())) or 0)
    return ticked


# ── cron ───────────────────────────────────────────────────────────────────────────────────────


def test_tick_leaves_a_marked_profile_store_alone(root, monkeypatch):
    from cron.scheduler import tick
    from cron.scheduler_provider import _profile_cron_scope

    ops, dev = root / "profiles" / "ops", root / "profiles" / "dev"
    ticked = _record_ticks(monkeypatch)
    pre.mark_profile_run_elsewhere("ops", PLUGIN, root=root)

    for home in (ops, dev):  # a multi-profile host ticks each store under that profile's scope
        with _profile_cron_scope(home):
            assert tick(verbose=False) == 0
    assert ticked == [dev]

    monkeypatch.setenv("HERMES_HOME", str(ops))  # a process whose own profile is the marked one
    assert tick(verbose=False) == 0
    assert ticked == [dev]

    _deny_marker_folder(monkeypatch)  # the backend the plugin runs ticks its own store
    tick(verbose=False)
    assert ticked == [dev, ops]


def test_gateway_ticker_does_not_tick_its_marked_launch_profile(root, monkeypatch):
    """``hermes -p ops gateway run`` already running when ``ops`` is marked: the ticker adds its
    launch profile back to the served set and its gate only looks for another gateway process."""
    from cron.scheduler_provider import InProcessCronScheduler
    from gateway.run import _cron_profile_gate, _cron_tick_profile_homes

    ops = root / "profiles" / "ops"
    monkeypatch.setenv("HERMES_HOME", str(ops))
    config = types.SimpleNamespace(_runtime_profile_homes=[])
    ticked = _record_ticks(monkeypatch)

    def one_cycle():
        InProcessCronScheduler().start(
            _OneCycle(), profile_homes=lambda: _cron_tick_profile_homes(config),
            profile_gate=_cron_profile_gate, default_profile="ops")

    one_cycle()
    assert ticked == [ops]  # unmarked: the launch profile is ticked

    pre.mark_profile_run_elsewhere("ops", PLUGIN, root=root)
    assert [name for name, _home in _cron_tick_profile_homes(config)] == ["ops"]
    assert _cron_profile_gate("ops", ops) is True
    one_cycle()
    assert ticked == [ops]  # marked: no further tick


def test_external_provider_fallback_ticker_does_not_tick_a_marked_profile(root, monkeypatch):
    """Chronos hands this process's fires to the built-in single-store ticker after NAS rejects its
    identity; that ticker gets no ``profile_homes`` and no ``profile_gate``."""
    from plugins.cron_providers.chronos import ChronosCronScheduler

    ops = root / "profiles" / "ops"
    monkeypatch.setenv("HERMES_HOME", str(ops))
    ticked = _record_ticks(monkeypatch)

    def fall_back():
        provider = ChronosCronScheduler()
        provider._stop_event = _OneCycle()
        provider._ticker_kwargs = {"adapters": None, "loop": None, "interval": 60}
        provider._note_identity_rejected()
        for thread in threading.enumerate():
            if thread.name == "cron-scheduler-chronos-fallback":
                thread.join(timeout=60)
                assert not thread.is_alive()

    fall_back()
    assert ticked == [ops]  # unmarked: the fallback ticks the process's own store

    pre.mark_profile_run_elsewhere("ops", PLUGIN, root=root)
    fall_back()
    assert ticked == [ops]  # marked: no further tick


def test_an_inbound_fire_is_not_claimed_for_a_marked_profile(root, monkeypatch):
    """An external provider's fire webhook (and the misfire sweep) claims through ``claim_fire``."""
    from cron.scheduler_provider import InProcessCronScheduler

    monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "ops"))
    pre.mark_profile_run_elsewhere("ops", PLUGIN, root=root)
    monkeypatch.setattr("cron.executions.create_execution",
                        lambda *a, **k: pytest.fail("a fire was recorded for a profile a plugin runs elsewhere"))
    monkeypatch.setattr("cron.jobs.claim_job_for_fire",
                        lambda *a, **k: pytest.fail("a fire was claimed for a profile a plugin runs elsewhere"))
    provider = InProcessCronScheduler()
    assert provider.claim_fire("job-1") is None
    assert provider.fire_due("job-1") is False


# ── turns of an agent built before the profile was marked ──────────────────────────────────────


class _TurnAdmitted(Exception):
    """The turn got past admission (raised by the first step of ``run_conversation``)."""


def _built_agent(monkeypatch):
    from run_agent import AIAgent

    def admitted(_agent):
        raise _TurnAdmitted

    monkeypatch.setattr("agent.background_review.cancel_background_review_for_live_turn", admitted)
    return object.__new__(AIAgent)  # built earlier: the constructor's refusal is behind it


def test_a_built_agent_runs_no_turn_once_its_profile_is_marked(root, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "ops"))
    agent = _built_agent(monkeypatch)
    with pytest.raises(_TurnAdmitted):
        agent.run_conversation("hi")

    pre.mark_profile_run_elsewhere("ops", PLUGIN, root=root)
    with pytest.raises(pre.ProfileRunsElsewhere) as refused:
        agent.run_conversation("hi")
    assert "'ops'" in str(refused.value) and PLUGIN in str(refused.value)

    _deny_marker_folder(monkeypatch)  # the backend the plugin runs keeps running its turns
    with pytest.raises(_TurnAdmitted):
        agent.run_conversation("hi")


def test_a_multi_profile_host_refuses_only_the_marked_siblings_turns(root, monkeypatch):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    agent = _built_agent(monkeypatch)
    pre.mark_profile_run_elsewhere("ops", PLUGIN, root=root)
    for name, expected in (("ops", pre.ProfileRunsElsewhere), ("dev", _TurnAdmitted)):
        token = set_hermes_home_override(str(root / "profiles" / name))
        try:
            with pytest.raises(expected):
                agent.run_conversation("hi")
        finally:
            reset_hermes_home_override(token)
    with pytest.raises(_TurnAdmitted):
        agent.run_conversation("hi")  # the default (launch) profile is not marked


# ── the dashboard's deferred agent build ───────────────────────────────────────────────────────


def _deferred_build(monkeypatch, *, profile_home=None):
    """Run ``_start_agent_build`` for a registered session; returns (MCP discovery starts, agent
    builds, error events, the session)."""
    from tui_gateway import entry, server

    discovered, built, errors = [], [], []
    monkeypatch.setattr(entry, "ensure_mcp_discovery_started", lambda: discovered.append(True))
    monkeypatch.setattr(server, "_make_agent", lambda *a, **k: built.append(a) or object())
    monkeypatch.setattr(server, "_deferred_build_agent_kwargs", lambda *a, **k: {})
    monkeypatch.setattr(server, "_await_resume_history", lambda *a, **k: True)
    monkeypatch.setattr(server, "_set_session_context", lambda *a, **k: None)
    monkeypatch.setattr(server, "_clear_session_context", lambda *a, **k: None)
    monkeypatch.setattr(server, "_session_cwd", lambda *a, **k: None)
    monkeypatch.setattr(server, "_bind_build_profile_scopes", lambda *a, **k: None)
    monkeypatch.setattr(server, "_open_profile_session_db", lambda *a, **k: None)
    monkeypatch.setattr(server, "_attach_built_agent", lambda *a, **k: False)
    monkeypatch.setattr(server, "_finish_agent_build", lambda *a, **k: None)
    monkeypatch.setattr(server, "_emit", lambda kind, sid, payload: errors.append(payload) if kind == "error" else None)
    session = {"session_key": "key", "agent_ready": threading.Event(),
               "profile_home": str(profile_home) if profile_home else None}
    monkeypatch.setitem(server._sessions, "sid", session)
    server._start_agent_build("sid", session)
    session["_agent_build_thread"].join(timeout=60)
    assert session["agent_ready"].is_set()
    return discovered, built, errors, session


def test_dashboard_build_refuses_a_marked_launch_profile_before_mcp_discovery(root, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "ops"))
    pre.mark_profile_run_elsewhere("ops", PLUGIN, root=root)
    discovered, built, errors, session = _deferred_build(monkeypatch)
    assert discovered == [] and built == []
    assert "'ops'" in session["agent_error"] and PLUGIN in session["agent_error"]
    assert errors and PLUGIN in errors[0]["message"]


def test_dashboard_build_refuses_a_marked_sibling_before_mcp_discovery(root, monkeypatch):
    """A multi-profile backend (launch profile: default) asked for a session in ``ops``."""
    pre.mark_profile_run_elsewhere("ops", PLUGIN, root=root)
    discovered, built, _errors, session = _deferred_build(monkeypatch, profile_home=root / "profiles" / "ops")
    assert discovered == [] and built == []
    assert "'ops'" in session["agent_error"]


def test_dashboard_build_of_an_unmarked_profile_starts_discovery_and_builds(root, monkeypatch):
    pre.mark_profile_run_elsewhere("ops", PLUGIN, root=root)
    discovered, built, errors, _session = _deferred_build(monkeypatch, profile_home=root / "profiles" / "dev")
    assert discovered == [True] and len(built) == 1 and errors == []


# ── hermes gateway run ─────────────────────────────────────────────────────────────────────────


def _gateway_run_args():
    return Namespace(accept_hooks=False, command="gateway", cron_command=None, gateway_command="run",
                     mcp_action=None, tui=False, yolo=False, safe_mode=False, ignore_user_config=False)


def test_cli_gateway_run_exits_78_before_discovery_or_host_gateway_contact(root, monkeypatch, capsys, caplog):
    from gateway.restart import GATEWAY_FATAL_CONFIG_EXIT_CODE
    from hermes_cli import main as main_mod

    monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "ops"))
    pre.mark_profile_run_elsewhere("ops", PLUGIN, root=root)
    monkeypatch.setattr(main_mod, "_guard_noninteractive_user_config", lambda _a: None)
    monkeypatch.setitem(sys.modules, "hermes_cli.plugins", types.SimpleNamespace(
        start_background_plugin_discovery=lambda: pytest.fail("plugin discovery before the refusal")))
    monkeypatch.setattr("agent.shell_hooks.register_from_config",
                        lambda *a, **k: pytest.fail("shell hooks registered before the refusal"))

    with caplog.at_level("ERROR"), pytest.raises(SystemExit) as exited:
        main_mod._prepare_agent_startup(_gateway_run_args())
    assert exited.value.code == GATEWAY_FATAL_CONFIG_EXIT_CODE
    err = capsys.readouterr().err
    assert "'ops'" in err and PLUGIN in err
    # A parked service unit is diagnosed from the profile's logs: the verdict is logged, not only printed.
    assert any("Refusing to start the gateway" in r.getMessage() and PLUGIN in r.getMessage() for r in caplog.records)


def test_cli_gateway_run_of_an_unmarked_profile_still_starts_discovery(root, monkeypatch):
    from hermes_cli import main as main_mod

    pre.mark_profile_run_elsewhere("ops", PLUGIN, root=root)  # HERMES_HOME is the default profile
    started = []
    monkeypatch.setattr(main_mod, "_guard_noninteractive_user_config", lambda _a: None)
    monkeypatch.setitem(sys.modules, "hermes_cli.plugins", types.SimpleNamespace(
        start_background_plugin_discovery=lambda: started.append("plugins")))
    monkeypatch.setattr("agent.shell_hooks.register_from_config", lambda *a, **k: started.append("hooks"))
    monkeypatch.setattr("agent.outbound_webhooks.register_from_config", lambda *a, **k: None)
    main_mod._prepare_agent_startup(_gateway_run_args())
    assert started == ["plugins", "hooks"]
