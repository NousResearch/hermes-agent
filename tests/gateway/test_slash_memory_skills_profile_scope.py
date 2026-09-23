"""/memory and /skills gateway handlers must run in the routed profile's home scope (#119915).

On a multiplexed gateway the process-level home is the multiplexer's (default)
profile, so a served profile's bot saw "No pending memory writes" and an
approve would write into the default profile's memories/.  The handlers now
run inside ``_profile_runtime_scope`` for the profile owning the source.
"""
import asyncio
import contextlib
import types

import gateway.slash_commands as slash_commands


def _handler_cls():
    for name in dir(slash_commands):
        obj = getattr(slash_commands, name)
        if isinstance(obj, type) and hasattr(obj, "_handle_memory_command"):
            return obj
    raise AssertionError("no slash-command handler class found")


def _runner(multiplex: bool):
    runner = object.__new__(_handler_cls())
    runner.config = types.SimpleNamespace(multiplex_profiles=multiplex)
    runner._resolve_profile_home_for_source = lambda source: "/profiles/project"
    runner._write_approval_setter = lambda section, event: (lambda enabled: None)
    return runner


def _event():
    return types.SimpleNamespace(
        source=types.SimpleNamespace(platform=None, user_id="u", chat_id="c"),
        get_command_args=lambda: "")


def _install_scope_recorder(monkeypatch, state):
    @contextlib.contextmanager
    def fake_scope(home):
        state["scope_homes"].append(home)
        state["in_scope"] = True
        try:
            yield
        finally:
            state["in_scope"] = False

    import gateway.run
    monkeypatch.setattr(gateway.run, "_profile_runtime_scope", fake_scope)


def _install_pending_recorder(monkeypatch, state, reply="ok"):
    import hermes_cli.write_approval_commands as wac

    def fake_handle(*args, **kwargs):
        state["called_in_scope"].append(state["in_scope"])
        return reply

    monkeypatch.setattr(wac, "handle_pending_subcommand", fake_handle)


def test_memory_command_runs_inside_profile_scope_when_multiplexed(monkeypatch):
    state = {"in_scope": False, "scope_homes": [], "called_in_scope": []}
    _install_scope_recorder(monkeypatch, state)
    _install_pending_recorder(monkeypatch, state)
    import tools.memory_tool
    monkeypatch.setattr(tools.memory_tool, "load_on_disk_store", lambda: None)

    out = asyncio.run(_runner(True)._handle_memory_command(_event()))
    assert out == "ok"
    assert state["called_in_scope"] == [True], "handle_pending_subcommand ran outside the profile scope"
    assert state["scope_homes"] == ["/profiles/project"]


def test_memory_command_skips_scope_when_not_multiplexed(monkeypatch):
    state = {"in_scope": False, "scope_homes": [], "called_in_scope": []}
    _install_scope_recorder(monkeypatch, state)
    _install_pending_recorder(monkeypatch, state)
    import tools.memory_tool
    monkeypatch.setattr(tools.memory_tool, "load_on_disk_store", lambda: None)

    asyncio.run(_runner(False)._handle_memory_command(_event()))
    assert state["scope_homes"] == [], "scope must not be entered when multiplex_profiles is off"
    assert state["called_in_scope"] == [False]


def test_skills_command_runs_inside_profile_scope_when_multiplexed(monkeypatch):
    state = {"in_scope": False, "scope_homes": [], "called_in_scope": []}
    _install_scope_recorder(monkeypatch, state)
    _install_pending_recorder(monkeypatch, state)
    import tools.write_approval as wa
    monkeypatch.setattr(wa, "write_approval_enabled", lambda section: True)
    monkeypatch.setattr(wa, "pending_count", lambda section: 0)

    out = asyncio.run(_runner(True)._handle_skills_command(_event()))
    assert out == "ok"
    assert state["called_in_scope"] == [True], "skills review ran outside the profile scope"
    assert state["scope_homes"] == ["/profiles/project"]


def test_profile_home_scope_is_noop_without_multiplex():
    runner = _runner(False)
    with runner._profile_home_scope(_event()):
        pass  # nullcontext: must not raise
