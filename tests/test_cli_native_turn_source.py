from __future__ import annotations

import queue
import types

import pytest

from cli import HermesCLI
from hermes_cli.native_turn_sources import (
    NativeSessionView, NativeTurnAdmission, NativeTurnLease,
    RegisteredNativeTurnSource, register_profile_source,
)


def bare_cli(tmp_path):
    cli = object.__new__(HermesCLI)
    cli._agent_running = False
    cli._should_exit = False
    cli._pending_input = queue.Queue()
    cli._interrupt_queue = queue.Queue()
    cli._active_session_lease = types.SimpleNamespace(lease_id="owner")
    cli.agent = types.SimpleNamespace(session_id="session")
    view = NativeSessionView(
        profile="default",
        profile_home=tmp_path,
        session_id="session",
        surface="cli",
        compression_lineage=("session",),
        session_key="session",
        owner_token="owner",
    )
    cli._native_session_view = lambda: view
    return cli, view


def test_cli_reserves_slot_commits_then_uses_ordinary_turn_entry(monkeypatch, tmp_path):
    cli, view = bare_cli(tmp_path)
    calls = []
    admission = NativeTurnAdmission(
        NativeTurnLease(
            "lease",
            "continue",
            lambda seen: calls.append(("commit", cli._agent_running, seen)) or True,
            lambda outcome, reason: calls.append((outcome, reason)),
        ),
        view,
    )
    monkeypatch.setattr("hermes_cli.native_turn_sources.poll_native_turn", lambda seen: admission)

    def process(text):
        calls.append(("turn", text))
        cli._agent_running = False

    cli._tui_process_one_input = process

    assert cli._maybe_start_native_turn() is True
    assert calls == [("commit", True, view), ("turn", "continue")]


def test_cli_no_source_preserves_old_idle_behavior(monkeypatch, tmp_path):
    cli, _view = bare_cli(tmp_path)
    monkeypatch.setattr("hermes_cli.native_turn_sources.poll_native_turn", lambda _seen: None)
    cli._tui_process_one_input = lambda _text: (_ for _ in ()).throw(AssertionError("no turn"))

    assert cli._maybe_start_native_turn() is False
    assert cli._agent_running is False


def test_cli_never_polls_with_pending_human_input(monkeypatch, tmp_path):
    cli, _view = bare_cli(tmp_path)
    cli._pending_input.put("human")
    monkeypatch.setattr(
        "hermes_cli.native_turn_sources.poll_native_turn",
        lambda _seen: (_ for _ in ()).throw(AssertionError("must not poll")),
    )

    assert cli._maybe_start_native_turn() is False
    assert cli._pending_input.get_nowait() == "human"


def test_cli_race_with_human_input_aborts_without_running(monkeypatch, tmp_path):
    cli, view = bare_cli(tmp_path)
    aborts = []
    admission = NativeTurnAdmission(
        NativeTurnLease(
            "lease",
            "continue",
            lambda _seen: (_ for _ in ()).throw(AssertionError("must not commit")),
            lambda outcome, reason: aborts.append((outcome, reason)),
        ),
        view,
    )

    def poll(_seen):
        cli._pending_input.put("human")
        return admission

    monkeypatch.setattr("hermes_cli.native_turn_sources.poll_native_turn", poll)
    cli._tui_process_one_input = lambda _text: (_ for _ in ()).throw(AssertionError("must not run"))

    assert cli._maybe_start_native_turn() is False
    assert cli._agent_running is False
    assert admission.state == "not_sent"
    assert aborts[0][0] == "not_sent"


@pytest.mark.parametrize("owned", [True, False])
def test_real_cli_fence_notifies_only_the_owned_session(tmp_path, owned):
    cli, view = bare_cli(tmp_path)
    calls = []

    class BoundaryProbe:
        name = "boundary-probe"

        def poll(self, session: NativeSessionView) -> NativeTurnLease | None:
            return None

        def on_user_boundary(self, session: NativeSessionView, reason: str) -> None:
            calls.append((session, reason))

    source = BoundaryProbe()
    unregister = register_profile_source(
        view.profile_home,
        RegisteredNativeTurnSource(source, frozenset({"cli"}), "fixture-plugin"),
    )
    if not owned:
        cli._active_session_lease = None
    try:
        cli._fence_native_turn_sources("new_session")
        assert calls == ([(view, "new_session")] if owned else [])
    finally:
        unregister()