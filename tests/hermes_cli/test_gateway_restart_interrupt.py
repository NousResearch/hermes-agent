"""A yes to "Interrupt them and restart the gateway?" restarts now instead of waiting for the work.

The live-work prompt (and ``--yes``/``--interrupt``) promised to interrupt the named turns, but every
supervised restart then sent a bare SIGUSR1 and waited ``agent.restart_after_turn_timeout`` (30 min by
default) plus the drain envelope for them to finish. With consent the CLI now asks the gateway it is
about to signal, by PID, to skip the after-turn wait, and shrinks its own wait to match. Without
consent nothing changes: the gateway is never asked and the CLI keeps the full budget.

Every ``launchctl``/``systemctl``/signal seam is replaced by a recorder; nothing here executes one.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hermes_cli import gateway as gw

_PID = 4242
_WORK = {"active_agents": 2, "active_work": [
    {"kind": "chat", "session": "agent:main:telegram:dm:7", "pid": _PID, "elapsed_s": 617.0,
     "current_tool": "terminal"},
    {"kind": "chat", "session": "agent:main:telegram:dm:8", "pid": _PID, "elapsed_s": 540.0},
]}


@pytest.fixture
def gateway(monkeypatch):
    """A running gateway (PID 4242) answering the control socket; records what it is asked."""
    from gateway import control_socket
    state = SimpleNamespace(work=_WORK, accepts=True, asked=[], waited=[])

    def _query(home, verb, *, params=None, timeout=None):
        if verb == "active-work":
            return state.work
        if verb == "restart-interrupt":
            state.asked.append(params)
            return {"interrupting": True, "pid": _PID} if state.accepts else None
        return None

    monkeypatch.setattr(control_socket, "query_gateway_control", _query)
    monkeypatch.setattr("gateway.status.get_running_pid", lambda *a, **k: _PID)
    monkeypatch.setattr(gw, "_graceful_restart_via_sigusr1",
                        lambda pid, budget, **kw: state.waited.append((pid, budget)) or True)
    return state


@pytest.fixture
def launchd(monkeypatch, tmp_path):
    """``hermes gateway restart`` resolving to an installed LaunchAgent whose relaunch succeeds."""
    plist = tmp_path / "ai.hermes.gateway.plist"
    plist.write_text("<plist>whatever</plist>", encoding="utf-8")
    for name, fake in {
        "_host_multiplexer_for_all_verb": lambda: None,
        "_guard_named_profile_under_multiplexer": lambda force=False: None,
        "_dispatch_via_service_manager_if_s6": lambda *a, **k: False,
        "_reap_unsupervised_gateway_orphans": lambda: None,
        "_installed_service_kind_for": lambda probe: "launchd",
        "get_launchd_label": lambda: "ai.hermes.gateway",
        "_launchd_domain": lambda: "gui/501",
        "get_launchd_plist_path": lambda: plist,
        "refresh_launchd_plist_if_needed": lambda: False,
        "_request_gateway_self_restart": lambda pid: False,
        "probe_gateway_loop_liveness": lambda pid, **kw: gw.GATEWAY_LOOP_ALIVE,
        "_wait_for_launchd_service_pid": lambda label, old_pid, timeout=10.0, *, domain: True,
        "_clear_launchd_unsupported_marker": lambda: None,
    }.items():
        monkeypatch.setattr(gw, name, fake)


def _restart(interrupt: bool) -> None:
    gw._cmd_restart(SimpleNamespace(all=False, system=False, force=False, interrupt=interrupt))


def test_consent_restarts_launchd_gateway_without_waiting_for_its_turns(gateway, launchd, capsys):
    full_budget = gw._get_restart_exit_wait_budget()
    assert full_budget > 1800, "the default budget waits the 30 min after-turn cap"

    _restart(interrupt=True)

    assert gateway.asked == [{"pid": _PID}], "the signalled gateway is asked to interrupt, by PID"
    [(pid, budget)] = gateway.waited
    assert pid == _PID and budget == full_budget - 1800
    assert "interrupting live work" in capsys.readouterr().out


def test_without_live_work_the_restart_is_unchanged(gateway, launchd):
    gateway.work = {"active_agents": 0, "active_work": []}

    _restart(interrupt=False)

    assert gateway.asked == []
    assert gateway.waited == [(_PID, gw._get_restart_exit_wait_budget())]


def test_a_gateway_that_cannot_take_the_interrupt_keeps_the_full_wait(gateway, launchd, capsys):
    gateway.accepts = False  # predates the verb, or another PID answered

    _restart(interrupt=True)

    assert gateway.waited == [(_PID, gw._get_restart_exit_wait_budget())]
    assert "did not accept an immediate restart" in capsys.readouterr().out


@pytest.mark.parametrize("path", ["systemd", "external-supervisor"])
def test_every_sigusr1_restart_path_carries_the_consent(gateway, monkeypatch, path):
    from hermes_cli import gateway_supervised_restart as supervised
    full_budget = gw._get_restart_exit_wait_budget()
    if path == "systemd":
        monkeypatch.setattr(gw, "_wait_for_systemd_service_restart", lambda **kw: True)
        gw._systemd_graceful_restart_action(False, _PID, interrupt=True)
    else:
        monkeypatch.setattr(supervised, "_wait_for_supervised_gateway_replacement", lambda old_pid: 4343)
        supervised.restart_externally_supervised_gateway(_PID, interrupt=True)

    assert gateway.asked == [{"pid": _PID}]
    assert gateway.waited == [(_PID, full_budget - 1800)]
