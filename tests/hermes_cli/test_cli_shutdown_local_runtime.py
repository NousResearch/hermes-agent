"""CLI exit stops the managed llama-server the CLI itself started.

Any Hermes process can boot the managed router on demand (a llamacpp endpoint lookup with
``local_runtime.enabled`` and nothing running). The desktop backend stops its router in the
lifespan teardown; the CLI did not. On POSIX the router is a plain child, so it outlived the CLI,
the next Hermes process adopted it as the managed endpoint (recovery accepts a reparented router
whose owner is dead), and the idle sweep that unloads a quiet model had died with the CLI — a
loaded model then stayed resident indefinitely.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch


class _FakeSupervisor:
    def __init__(self, events: list[str]):
        self.events = events

    def stop(self) -> None:
        self.events.append("router stopped")


def _run_cleanup(cli_mod) -> None:
    cli_mod._cleanup_done = False
    try:
        with (
            patch.object(cli_mod, "_cleanup_all_terminals", lambda: None),
            patch.object(cli_mod, "_cleanup_all_browsers", lambda: None),
            patch("tools.mcp_tool_lifecycle.shutdown_mcp_servers", lambda *a, **k: None),
            patch("agent.auxiliary_client.shutdown_cached_clients", lambda *a, **k: None),
            patch("hermes_cli.plugins.invoke_hook", lambda *a, **k: None),
        ):
            cli_mod._run_cleanup()
    finally:
        cli_mod._cleanup_done = False


def test_cli_exit_stops_the_router_it_owns(monkeypatch):
    import cli as cli_mod
    from hermes_cli.local_runtime import bootstrap

    events: list[str] = []
    monkeypatch.setattr(bootstrap, "_SUPERVISOR", _FakeSupervisor(events))

    _run_cleanup(cli_mod)

    assert events == ["router stopped"]
    assert bootstrap.get_supervisor() is None


def test_router_stops_after_memory_shutdown(monkeypatch):
    """Session finalize and memory-provider shutdown can still call the local model, so the
    router goes last."""
    import cli as cli_mod
    from hermes_cli.local_runtime import bootstrap

    events: list[str] = []
    monkeypatch.setattr(bootstrap, "_SUPERVISOR", _FakeSupervisor(events))
    agent = MagicMock()
    agent.session_id = "cli-session-id"
    agent._session_messages = []
    agent.shutdown_memory_provider.side_effect = lambda *a: events.append("memory shutdown")
    monkeypatch.setattr(cli_mod, "_active_agent_ref", agent)

    _run_cleanup(cli_mod)

    assert events == ["memory shutdown", "router stopped"]


def test_router_adopted_from_another_process_is_left_alone(monkeypatch):
    """No in-process supervisor: the router belongs to someone else; exit must not touch it."""
    import cli as cli_mod
    from hermes_cli.local_runtime import bootstrap

    monkeypatch.setattr(bootstrap, "_SUPERVISOR", None)
    monkeypatch.setattr(bootstrap, "_stop_state_server",
                        lambda state: (_ for _ in ()).throw(AssertionError("stopped a foreign router")))

    _run_cleanup(cli_mod)


def test_sessions_that_never_used_the_local_runtime_do_not_import_it(monkeypatch):
    from hermes_cli import cli_shutdown

    monkeypatch.delitem(sys.modules, "hermes_cli.local_runtime.bootstrap", raising=False)

    cli_shutdown._shutdown_local_runtime()

    assert "hermes_cli.local_runtime.bootstrap" not in sys.modules
