"""Affinity contract on Hermes Studio's group-chat bridge shape (#96811).

Studio's group chat is the reproduction reported on #96811, and it reaches
Hermes as a LIBRARY rather than through the gateway: its Python bridge
constructs ``AIAgent(...)`` directly with a fresh physical ``session_id`` for
every reply, and only then persists the session row.

``tests/agent/test_declared_conversation_scope.py`` pins the declaration
contract on synthetic agent doubles, and
``test_prompt_cache_scope.py::TestPerResponseRunNonceIsolation`` pins the
id-level isolation the scope rule may never break.  Neither runs the host's
own construction path, so nothing today would fail if a refactor made that
path stop honouring a declaration.  This suite closes that: it builds a real
``AIAgent`` in the bridge's order and states exactly what the host adoption
buys — and what it must not cross.

The shape reproduced here:

1. ``groupRuntimeSessionId(room, profile, name)`` mints
   ``gc_run_<room>_<profile>_<name>`` truncated to 96 characters plus a fresh
   UUID4 hex — a NEW physical id per reply;
2. the agent is constructed before the row exists (same as Studio's pool);
   the bridge then writes the row, so the resolution reads a landed row;
3. ``AIAgent(...)`` is constructed with ``platform`` / ``session_id`` /
   ``session_db`` and no routing identity of any kind.

``groupBridgeSessionId(room, profile, name, sessionSeed, runtimeConfig)`` is
the value that IS stable across those replies.  It already exists host-side,
already carries the room, profile, agent name and the room-owned
``sessionSeed``, and is already hashed and length-bounded, so passing it as
``gateway_session_key`` is the entire adoption.
"""

from __future__ import annotations

import hashlib
import re
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from agent.prompt_cache_scope import resolve_prompt_cache_scope
from hermes_state import SessionDB
from run_agent import AIAgent

# The bridge stamps its rows with the ``source`` carried by ``bridge.chat``
# while ``AIAgent.platform`` is the bridge's own platform name; the two need
# not match.  Keeping them different here is deliberate — the scope has to
# resolve under the source the row actually landed on.
BRIDGE_ROW_SOURCE = "api_server"
BRIDGE_PLATFORM = "hermes_bridge"

ROOM = "room-7c1f"
PROFILE = "default"


def _runtime_session_id(room: str = ROOM, name: str = "Reviewer") -> str:
    """``groupRuntimeSessionId`` — a new physical id for every reply."""
    prefix = f"gc_run_{room}_{PROFILE}_{name}"[:96]
    return f"{prefix}_{uuid4().hex}"


def _bridge_session_key(
    room: str = ROOM, name: str = "Reviewer", seed: str = "0"
) -> str:
    """``groupBridgeSessionId`` — stable for one conversation in one room."""
    raw = f"gc_{room}_{PROFILE}_{name}_{seed}"
    safe_prefix = re.sub(r"[^a-zA-Z0-9_-]", "_", raw)
    suffix = f"_h_{hashlib.sha256(raw.encode()).hexdigest()[:16]}"
    return f"{safe_prefix[: max(0, 120 - len(suffix))]}{suffix}"


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    try:
        yield session_db
    finally:
        session_db.close()


def _bridge_agent(session_db, session_id: str, *, declared_key: str | None):
    """Construct an agent the way the Studio bridge pool does."""
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            verbose_logging=False,
            skip_context_files=True,
            skip_memory=True,
            session_db=session_db,
            session_id=session_id,
            platform=BRIDGE_PLATFORM,
            gateway_session_key=declared_key,
        )
    agent.client = MagicMock()
    return agent


def _reply_scope(
    session_db,
    *,
    declared_key: str | None,
    room: str = ROOM,
    name: str = "Reviewer",
    row_source: str = BRIDGE_ROW_SOURCE,
) -> str:
    """One Studio reply, in the bridge's own order, returning its scope.

    Mirrors the real Studio construction order: ``AIAgent(...)`` is built
    first with a fresh physical id, then the bridge persists the row, then
    the scope is resolved against the now-landed row.  Constructing the
    agent before persistence is the part the test exists to pin — the
    resolution path must read the row at resolve time, not assume it was
    already there when the agent was instantiated.
    """
    session_id = _runtime_session_id(room=room, name=name)
    # AIAgent first — same order as the Studio bridge pool.
    agent = _bridge_agent(session_db, session_id, declared_key=declared_key)
    # Then the bridge writes the row, which is when the scope's row reads
    # finally have something to find.
    session_db.create_session(
        session_id=session_id, source=row_source, model="test/model"
    )
    return resolve_prompt_cache_scope(agent)


class TestStudioBridgeAffinity:
    def test_the_undeclared_bridge_shape_rekeys_every_reply(self, db):
        """The reproduction, stated as a permanent negative control.

        Hermes must not infer the conversation from the id's syntax — that is
        #79017's collision class — so a bridge that declares nothing keeps a
        per-reply scope by design.  This stays green before and after the host
        adopts the key, and it is why the adoption has to happen host-side.
        """
        scopes = {_reply_scope(db, declared_key=None) for _ in range(3)}

        assert len(scopes) == 3

    def test_a_declared_conversation_holds_one_affinity_across_replies(self, db):
        """Consecutive replies keep one affinity though their ids differ."""
        key = _bridge_session_key()

        scopes = {_reply_scope(db, declared_key=key) for _ in range(3)}

        assert len(scopes) == 1
        assert scopes.pop().startswith("gwk_")

    def test_the_scope_never_carries_the_room_or_member(self, db):
        """The scope leaves the process verbatim as a provider routing key."""
        scope = _reply_scope(db, declared_key=_bridge_session_key())

        assert ROOM not in scope
        assert "Reviewer" not in scope
        assert len(scope) <= 64

    def test_a_different_member_or_room_gets_a_different_affinity(self, db):
        """Two members of one room, and two rooms, never share a bucket."""
        reviewer = _reply_scope(db, declared_key=_bridge_session_key(name="Reviewer"))
        planner = _reply_scope(
            db, declared_key=_bridge_session_key(name="Planner"), name="Planner"
        )
        other_room = _reply_scope(
            db, declared_key=_bridge_session_key(room="room-99ab"), room="room-99ab"
        )

        assert len({reviewer, planner, other_room}) == 3

    def test_a_new_session_seed_rotates_the_affinity(self, db):
        """Studio's own conversation boundary is the room-owned sessionSeed.

        A new conversation in the room mints a new seed, and the seed is part
        of the declared key, so the next conversation starts on a cold bucket
        without Hermes having to observe a reset of its own.
        """
        first = _reply_scope(db, declared_key=_bridge_session_key(seed="0"))
        second = _reply_scope(db, declared_key=_bridge_session_key(seed="1"))

        assert first != second

    def test_the_row_source_is_the_authority_not_the_platform(self, db):
        """Equal keys under different sources never share one scope.

        The declared key is caller-supplied and may legally repeat across
        hosts sharing one database, and this value leaves the process as a
        provider routing key, so the carrier is qualified by the source the
        row actually landed under.
        """
        key = _bridge_session_key()

        as_api_server = _reply_scope(db, declared_key=key, row_source="api_server")
        as_other = _reply_scope(db, declared_key=key, row_source="studio_native")

        assert as_api_server != as_other

    def test_a_tool_child_on_the_same_declaration_keeps_its_own_scope(self, db):
        """#79161 survives this construction path.

        A tool child inherits nothing from the bridge today, but were a host
        ever to hand one the room's declared key, the row's own fork marker
        still keeps it off the conversation's bucket.
        """
        key = _bridge_session_key()
        conversation = _reply_scope(db, declared_key=key)

        child_id = _runtime_session_id()
        child = _bridge_agent(db, child_id, declared_key=key)
        db.create_session(session_id=child_id, source="tool", model="test/model")

        assert resolve_prompt_cache_scope(child) == child_id
        assert child_id != conversation

    def test_a_background_review_fork_never_borrows_the_declaration(self, db):
        """Background review clones the live runtime, declared key included."""
        key = _bridge_session_key()
        session_id = _runtime_session_id()
        agent = _bridge_agent(db, session_id, declared_key=key)
        agent._persist_disabled = True
        db.create_session(
            session_id=session_id, source=BRIDGE_ROW_SOURCE, model="test/model"
        )

        assert resolve_prompt_cache_scope(agent) == session_id


class TestUndeclaredBridgeIsUnchanged:
    """Byte-identical behaviour for a bridge that declares nothing."""

    def test_the_undeclared_shape_resolves_to_the_physical_id(self, db):
        session_id = _runtime_session_id()
        agent = _bridge_agent(db, session_id, declared_key=None)
        db.create_session(
            session_id=session_id, source=BRIDGE_ROW_SOURCE, model="test/model"
        )

        assert resolve_prompt_cache_scope(agent) == session_id

    def test_an_undeclared_compression_rotation_still_walks_its_lineage(self, db):
        root = _runtime_session_id()
        db.create_session(session_id=root, source=BRIDGE_ROW_SOURCE, model="test/model")
        db.end_session(root, "compression")
        rotated = _runtime_session_id()
        db.create_session(
            session_id=rotated,
            source=BRIDGE_ROW_SOURCE,
            model="test/model",
            parent_session_id=root,
        )
        agent = _bridge_agent(db, rotated, declared_key=None)

        assert resolve_prompt_cache_scope(agent) == root
