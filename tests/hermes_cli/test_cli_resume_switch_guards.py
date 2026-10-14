"""/resume and /sessions session-switch subscription guards.

Companion to ``tests/hermes_cli/test_cli_resume_command.py``, which already pins the
baseline arms of the subscription fix (both switch spellings with no agent, the refusal
abort, the already-on arm, the unreachable-gateway arm and the transient refusal that
must not block the switch). This file guards the properties those arms leave open:

* a refusal is atomic — the session being left is neither ended nor reloaded, and a live
  agent is not repointed onto a session this viewer cannot write;
* the refusal copy is a real catalog entry, not a raw dotted key;
* the numbered-picker entry point (``_consume_pending_resume_selection`` -> the handler)
  subscribes the session it selects, and its refusal keeps the current session;
* a store without ``attach`` (no gateway facade) still switches;
* the Kanban guard refuses *before* the new attach is attempted;
* the sibling ``/branch`` still subscribes its child, and not the parent.

``_open_switched_session`` (``hermes_cli/cli_session_switch.py``) is the seam: it subscribes
``session_id`` on this viewer's connection and only then may the switch commit.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pytest

from agent.i18n import t
from cli import HermesCLI
from hermes_cli.gateway_client import GatewayClientError, GatewayRPCError


def _make_cli():
    """HermesCLI host double: ``_session_db`` is a MagicMock, so ``attach`` is observable."""
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.session_id = "current_session"
    cli_obj._resumed = False
    cli_obj._pending_title = None
    cli_obj.conversation_history = []
    cli_obj.agent = None
    cli_obj._session_db = MagicMock()
    cli_obj._pending_resume_sessions = None
    # _display_resumed_history reads resume_display; "minimal" short-circuits the recap.
    cli_obj.resume_display = "minimal"
    return cli_obj


def _cli_switching_to(target):
    cli_obj = _make_cli()
    cli_obj._session_db.get_session.return_value = {"id": target, "title": "T"}
    cli_obj._session_db.get_resume_conversations.return_value = ([], [])
    cli_obj._session_db.resolve_resume_session_id.return_value = target
    return cli_obj


def _resume(cli_obj, cmd="/resume target"):
    """Run ``/resume`` with the name resolver pinned; return the ``_cprint`` mock."""
    with (
        patch("hermes_cli.main._resolve_session_by_name_or_id", return_value="target"),
        patch("cli._cprint") as mock_cprint,
    ):
        cli_obj._handle_resume_command(cmd)
    return mock_cprint


def _printed(mock_cprint):
    return " ".join(str(call) for call in mock_cprint.call_args_list)


class TestRefusalIsAtomic:
    """A refused subscription must abort the whole switch, not half of it."""

    def test_a_refused_subscription_does_not_end_or_reload_the_session_being_left(self):
        cli_obj = _cli_switching_to("target")
        cli_obj._session_db.attach.side_effect = GatewayRPCError("permission_denied")

        _resume(cli_obj)

        assert cli_obj.session_id == "current_session"
        # The old row is ended only after the target is subscribed; on a refusal it is untouched,
        # so history is not reloaded onto a session the viewer may not write.
        cli_obj._session_db.end_session.assert_not_called()
        cli_obj._session_db.get_resume_conversations.assert_not_called()
        cli_obj._session_db.reopen_session.assert_not_called()

    def test_a_refused_subscription_leaves_an_existing_agent_on_the_old_session(self):
        cli_obj = _cli_switching_to("target")
        agent = MagicMock()
        agent.session_id = "current_session"
        cli_obj.agent = agent
        cli_obj._session_db.attach.side_effect = GatewayRPCError("permission_denied")

        _resume(cli_obj)

        # _sync_agent_to_session must not fire: repointing the live agent would flush the next
        # turn's messages onto a session the gateway refuses.
        assert agent.session_id == "current_session"
        agent.reset_session_state.assert_not_called()
        agent._flush_messages_to_session_db.assert_not_called()
        cli_obj._session_db.end_session.assert_not_called()


class TestRefusalCopy:
    def test_the_refusal_copy_comes_from_the_catalog_not_a_raw_key(self):
        cli_obj = _cli_switching_to("target")
        cli_obj._session_db.attach.side_effect = GatewayRPCError("permission_denied")

        mock_cprint = _resume(cli_obj)
        printed = _printed(mock_cprint)

        # The handler renders the existing `resume.switch_failed` entry the gateway shares
        # verbatim (already translated in every catalog), and names the session it stayed on.
        rendered = t("gateway.resume.switch_failed")
        assert rendered != "gateway.resume.switch_failed", "the refusal copy is missing from the catalog"
        assert rendered in printed
        assert "target" in printed
        assert "cli.commands." not in printed, "a raw catalog key reached the user"


class TestNumberedPickerEntryPoint:
    """A bare number after a bare /resume re-enters the handler; the select must subscribe too."""

    def _armed_cli(self):
        cli_obj = _make_cli()
        sessions = [{"id": "sess_002", "title": "Coding"}, {"id": "sess_001", "title": "Research"}]
        cli_obj._pending_resume_sessions = sessions
        # The handler re-resolves the index through _list_recent_sessions.
        cli_obj._list_recent_sessions = MagicMock(return_value=sessions)
        cli_obj._session_db.get_session.return_value = {"id": "sess_001", "title": "Research"}
        cli_obj._session_db.get_resume_conversations.return_value = ([], [])
        cli_obj._session_db.resolve_resume_session_id.return_value = "sess_001"
        return cli_obj

    def test_the_numbered_picker_subscribes_the_selected_session_exactly_once(self):
        cli_obj = self._armed_cli()

        with patch("cli._cprint"):
            consumed = cli_obj._consume_pending_resume_selection("2")

        assert consumed is True
        assert cli_obj.session_id == "sess_001"
        # Exactly one attach for the switch's target, and never the session being left.
        cli_obj._session_db.attach.assert_called_once_with("sess_001")

    def test_the_numbered_picker_refusal_keeps_the_current_session(self):
        cli_obj = self._armed_cli()
        cli_obj._session_db.attach.side_effect = GatewayRPCError("permission_denied")

        with patch("cli._cprint"):
            consumed = cli_obj._consume_pending_resume_selection("2")

        assert consumed is True
        assert cli_obj.session_id == "current_session"
        cli_obj._session_db.end_session.assert_not_called()
        # One-shot prompt is still disarmed after the refused selection.
        assert cli_obj._pending_resume_sessions is None


class TestStoreWithoutAttach:
    """Off the gateway facade there is no `attach`: the switch proceeds as it always did."""

    @pytest.mark.parametrize("blank", ["delete", "none"])
    def test_a_store_without_a_callable_attach_still_switches(self, blank):
        cli_obj = _cli_switching_to("target")
        if blank == "delete":
            del cli_obj._session_db.attach
        else:
            cli_obj._session_db.attach = None

        _resume(cli_obj)

        assert cli_obj.session_id == "target"
        # The committed switch still ends the row it left.
        cli_obj._session_db.end_session.assert_called_once_with("current_session", "resumed_other")


class TestKanbanGuardPrecedesTheAttach:
    def test_the_kanban_guard_refuses_before_any_subscription(self, monkeypatch):
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
        cli_obj = _cli_switching_to("target")
        cli_obj._session_db.is_kanban_owned_session.return_value = True

        mock_cprint = _resume(cli_obj)

        assert cli_obj.session_id == "current_session"
        cli_obj._session_db.attach.assert_not_called()
        cli_obj._session_db.end_session.assert_not_called()
        assert t("cli.resume.kanban_owned", session_id="target") in _printed(mock_cprint)


class _Agent:
    """Minimal agent double: no `_memory_manager`, no `_todo_store` (keeps the switch inert)."""

    def __init__(self, sid):
        self.session_id = sid
        self._last_flushed_db_idx = 0

    def _flush_messages_to_session_db(self, history, conversation_history=None):
        pass

    def reset_session_state(self):
        pass


class _ScriptedGateway:
    """The store's connection: answers `session.resume` for the ids it knows and forks on branch."""

    def __init__(self, child="child"):
        self.child = child
        self.final = False  # the store's __bool__ reads connection.final
        self.calls: list[tuple[str, dict]] = []
        self.known = {"parent"}
        self._states: dict[str, object] = {}

    def call(self, method, **params):
        self.calls.append((method, dict(params)))
        if method == "session.resume":
            sid = params["session_id"]
            if sid not in self.known:
                raise GatewayClientError("not_found")
            return {"session_id": sid, "stored_session_id": sid, "messages": [], "revision": 1,
                    "execution_generation": 0, "subscription_id": "sub-" + sid, "info": {}}
        if method == "session.mutate" and params.get("operation") == "branch":
            self.known.add(self.child)
            return {"branched_session_id": self.child, "copied_messages": 1}
        return {"status": "ok"}

    def state(self, sid):
        return self._states.setdefault(sid, types.SimpleNamespace(adopt_snapshot=lambda _snapshot: None))


def test_branch_still_subscribes_its_child_not_the_parent(monkeypatch, tmp_path):
    """/branch shares _end_current_session/_sync_agent_to_session with /resume: the fix must not
    cost the sibling command its child subscription (the child is opened by the store's branch)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_state import SessionDB
    from hermes_cli.gateway_session_store import GatewaySessionStore

    writer = SessionDB(db_path=tmp_path / "state.db")
    writer.create_session("parent", source="local", model="m")
    writer.set_session_title("parent", "Parent")
    writer.append_message("parent", "user", "hi")
    writer.close()

    gateway = _ScriptedGateway()
    store = GatewaySessionStore(gateway)
    attached: list[str] = []
    _real_attach = store.attach

    def _recording_attach(session_id):
        attached.append(session_id)
        return _real_attach(session_id)

    store.attach = _recording_attach

    host = object.__new__(HermesCLI)
    host._session_db = store
    host._gateway_session_open = True
    host.session_id = "parent"
    host._agent_running = False
    host.conversation_history = [{"role": "user", "content": "hi"}]
    host.model = "m"
    host.max_turns = 5
    host.reasoning_config = {}
    host._pending_title = None
    host._resumed = False
    host.agent = _Agent("parent")
    host._new_session_id = lambda now, **kw: HermesCLI._new_session_id(host, now, **kw)

    with patch("cli._cprint"):
        host._handle_branch_command("/branch explore")

    assert host.session_id == "child"
    # The viewer is subscribed to its child, exactly once, and never to the session it left.
    assert attached == ["child"]
    assert set(store.snapshots) == {"child"}
