"""Restart drain-window recovery must be able to dedup an interrupted turn.

The Discord missed-message backfill (``_run_missed_message_backfill``) exists
to recover messages the bot never saw while it was down.  A gateway RESTART
produces a harder case: the message WAS received and a turn WAS started, then
the drain window force-interrupted it.  The transcript is the only durable
record of that, and the transcript row for the user turn is written WITHOUT
the platform-side message id — so nothing downstream can ask "did this
Discord message already reach the transcript?" and the recovery pass has no
authority to dedup against.

``SessionDB`` already carries a ``platform_message_id`` column, a partial
unique index over ``(session_id, platform_message_id)``, and a
``has_platform_message_id`` lookup — the storage and the query exist.  What is
missing is the WRITE on the normal agent-persisted turn path: the id is only
attached on the gateway-side transient-failure fallback
(``_handle_message_with_agent``), never on the path the agent itself flushes.
"""

from __future__ import annotations

import types

import pytest

from hermes_state import SessionDB


def _make_db(tmp_path) -> SessionDB:
    return SessionDB(db_path=tmp_path / "state.db")


class _MinimalAgent:
    """The narrow slice of AIAgent that ``_apply_persist_user_message_override``
    and ``_flush_messages_to_session_db`` read."""

    def __init__(self, db: SessionDB, session_id: str):
        self._session_db = db
        self._session_db_created = True
        self.session_id = session_id
        self._last_flushed_db_idx = 0
        self._flushed_db_message_ids = set()
        self._flushed_db_message_session_id = session_id
        self._persist_user_message_idx = None
        self._persist_user_message_override = None
        self._persist_user_message_timestamp = None
        self._persist_disabled = False

    def _ensure_db_session(self):  # pragma: no cover - already created
        return None


def test_build_turn_context_stamps_the_platform_message_id_on_the_user_turn():
    """The turn prologue must carry the platform id onto the user turn dict.

    This is the row the early crash-resilience persist writes, so it is the
    only place a drain-interrupted turn can pick the id up.
    """
    from agent.turn_context import build_turn_context

    agent = types.SimpleNamespace()
    ctx = _build_turn_context_for_test(
        build_turn_context, agent, persist_user_platform_id="discord-991"
    )

    user_msgs = [m for m in ctx.messages if m.get("role") == "user"]
    assert user_msgs, "no user turn in the built context"
    assert user_msgs[-1].get("platform_message_id") == "discord-991", (
        "the user turn reached persistence without its platform message id — a "
        "drain-interrupted turn is then unrecoverable/undedupable by "
        "has_platform_message_id"
    )


def test_persisted_interrupted_turn_is_findable_by_platform_message_id(tmp_path):
    """E2E: flush a turn the way the agent does, then ask the dedup authority.

    This is the exact question the restart drain-window recovery pass asks
    before re-dispatching a message.  On main the answer is False even though
    the turn IS in the transcript, so recovery would re-run a turn that
    already ran (duplicate work, duplicate spend, duplicate reply).
    """
    from run_agent import AIAgent

    db = _make_db(tmp_path)
    session_id = db.create_session("sess-drain-window", "gateway")

    agent = _MinimalAgent(db, session_id)
    agent._persist_user_message_idx = 0
    agent._persist_user_message_platform_id = "discord-4242"

    messages = [{"role": "user", "content": "please do the thing"}]

    AIAgent._apply_persist_user_message_override(agent, messages)
    AIAgent._flush_messages_to_session_db_unlocked(
        agent, messages, conversation_history=None
    )

    assert db.has_platform_message_id(session_id, "discord-4242"), (
        "the interrupted turn is in the transcript but carries no "
        "platform_message_id, so restart drain-window recovery cannot tell it "
        "already ran and will re-dispatch it"
    )


def test_platform_message_id_survives_a_persist_content_override(tmp_path):
    """The id must not be lost on the override path.

    Group-chat / observed-context turns route through
    ``_persist_user_message_override``; the id has to survive that rewrite or
    the dedup authority is blind for exactly the busy channels that need it.
    """
    from run_agent import AIAgent

    db = _make_db(tmp_path)
    session_id = db.create_session("sess-override", "gateway")

    agent = _MinimalAgent(db, session_id)
    agent._persist_user_message_idx = 0
    agent._persist_user_message_override = "clean transcript text"
    agent._persist_user_message_platform_id = "discord-7777"

    messages = [{"role": "user", "content": "api-facing text with context"}]

    AIAgent._apply_persist_user_message_override(agent, messages)
    AIAgent._flush_messages_to_session_db_unlocked(
        agent, messages, conversation_history=None
    )

    assert db.has_platform_message_id(session_id, "discord-7777")


def _build_turn_context_for_test(build_turn_context, agent, **overrides):
    """Construct a minimal build_turn_context call.

    Mirrors ``tests/agent/test_turn_context.py::_build`` but is kept local so
    this file stays self-contained.
    """
    from tests.agent.test_turn_context import _FakeAgent, _stub_runtime_main

    fake = _FakeAgent()
    kwargs = dict(
        agent=fake,
        user_message="hello",
        system_message=None,
        conversation_history=None,
        task_id=None,
        stream_callback=None,
        persist_user_message=None,
        restore_or_build_system_prompt=lambda *a, **k: None,
        install_safe_stdio=lambda: None,
        sanitize_surrogates=lambda s: s,
        summarize_user_message_for_log=lambda s: s,
        set_session_context=lambda _sid: None,
        set_current_write_origin=lambda _o: None,
        ra=lambda: types.SimpleNamespace(_set_interrupt=lambda *a, **k: None),
    )
    kwargs.update(overrides)
    return build_turn_context(**kwargs)


def test_gateway_run_agent_threads_the_event_message_id_into_the_turn():
    """AST proof that the gateway call site passes the id down.

    The unit tests above prove the persistence layer STORES the id once it is
    given one.  This pins the wiring: without the gateway forwarding
    ``event_message_id`` as ``persist_user_platform_id``, the whole path is
    dead code and every real inbound turn still persists without its id.
    """
    import ast
    import inspect

    import gateway.run as gateway_run

    source = inspect.getsource(gateway_run)
    tree = ast.parse(source)

    forwards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == "persist_user_platform_id"
    ]
    assert forwards, (
        "gateway/run.py never forwards persist_user_platform_id — the inbound "
        "platform message id never reaches the persisted user turn, so a "
        "drain-interrupted turn stays undedupable"
    )


def test_run_conversation_accepts_persist_user_platform_id():
    """The public forwarder must expose the kwarg the gateway passes."""
    import inspect

    from agent.conversation_loop import run_conversation
    from run_agent import AIAgent

    assert (
        "persist_user_platform_id"
        in inspect.signature(run_conversation).parameters
    )
    assert (
        "persist_user_platform_id"
        in inspect.signature(AIAgent.run_conversation).parameters
    )
