"""Group carry-over: an approved DM synopsis is posted to a group and forks its session.

Real path: a real ``SessionStore`` + ``state.db`` under the per-test HERMES_HOME, the real
``carry_to_group`` tool, and ``carry_into_group`` running on a separate "gateway" loop thread the
way the live gateway runs it. Only the platform adapter is a fake (it records what was posted).
"""

from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.group_carryover import CarryoverRequest, carry_into_group
from gateway.session import AsyncSessionStore, SessionSource, SessionStore
from gateway.session_context import clear_session_vars, set_session_vars
from tools import group_carryover_tool as tool

GROUP_ID = "chat-family-group"
DM_ID = "chat-jim-dm"
JIM = "+15550001"
RENEE = "+15550002"
SYNOPSIS = "Plan: Saturday is the beach, not the zoo. Renee books parking; Jim packs lunch."


class _FakeAdapter:
    def __init__(self, ok: bool = True):
        self.ok, self.sent = ok, []

    async def send(self, chat_id, content, metadata=None):
        self.sent.append((chat_id, content, metadata))
        return SimpleNamespace(success=self.ok, message_id="m1", error=None if self.ok else "boom")


class _Runner:
    """The slice of GatewayRunner that carry_into_group uses (the rest is optional via getattr)."""

    def __init__(self, store: SessionStore):
        self.session_store = store
        self.async_session_store = AsyncSessionStore(store)
        self.evicted, self.running = [], set()

    def _session_key_for_source(self, source):
        return self.session_store._generate_session_key(source)

    def _is_session_running(self, key):
        return key in self.running

    def _evict_cached_agent(self, key):
        self.evicted.append(key)


def _src(chat_id, chat_type, user_id, name=None):
    return SessionSource(platform=Platform.BLUEBUBBLES, chat_id=chat_id, chat_type=chat_type,
                         user_id=user_id, user_name=user_id, chat_name=name)


@pytest.fixture
def store(tmp_path):
    s = SessionStore(tmp_path / "sessions", GatewayConfig(sessions_dir=tmp_path / "sessions",
                                                         write_sessions_json=False))
    yield s
    s.close_all_db_handles()


def _seed_group(store):
    """Existing group history: Jim's lane has stale info, Renee has her own lane."""
    jim = store.get_or_create_session(_src(GROUP_ID, "group", JIM, "Family"))
    store.append_to_transcript(jim.session_id, {"role": "user", "content": "Are we doing the zoo Saturday?"})
    store.append_to_transcript(jim.session_id, {"role": "assistant", "content": "Yes, the zoo."})
    renee = store.get_or_create_session(_src(GROUP_ID, "group", RENEE, "Family"))
    store.append_to_transcript(renee.session_id, {"role": "user", "content": "hi"})
    store.append_to_transcript(renee.session_id, {"role": "assistant", "content": "hello"})
    dm = store.get_or_create_session(_src(DM_ID, "dm", JIM))
    store.append_to_transcript(dm.session_id, {"role": "user", "content": "let's do the beach instead"})
    return jim, renee, dm


def _request(store, **over):
    from hermes_state_registry import acquire, release_or_close
    db = acquire()
    try:
        facts = db.gateway_chat_facts(platform="bluebubbles", chat_id=GROUP_ID, user_ids=(JIM,))
    finally:
        release_or_close(db)
    kw = dict(platform="bluebubbles", chat_id=GROUP_ID, chat_name="Family", chat_type="group",
              user_id=JIM, user_name="Jim", synopsis=SYNOPSIS, live_sessions=facts["live_sessions"])
    kw.update(over)
    return CarryoverRequest(**kw)


def test_fork_seeds_group_and_leaves_private_session_alone(store):
    jim, renee, dm = _seed_group(store)
    dm_before = store.load_transcript(dm.session_id)
    runner, adapter = _Runner(store), _FakeAdapter()

    result = asyncio.run(carry_into_group(runner, adapter, _request(store)))

    assert result["success"] and result["posted"]
    (chat_id, posted, _), = adapter.sent
    assert chat_id == GROUP_ID and SYNOPSIS in posted
    # Jim's group lane is a NEW session whose whole history is the seeded pair (fresh system prompt
    # on the next turn; strict user→assistant alternation).
    key = runner._session_key_for_source(_src(GROUP_ID, "group", JIM))
    new_sid = store._entries[key].session_id
    assert new_sid == result["group_session_id"] != jim.session_id
    seeded = store.load_transcript(new_sid)
    assert [m["role"] for m in seeded] == ["user", "assistant"]
    assert SYNOPSIS in seeded[0]["content"] and seeded[1]["content"] == posted
    assert key in runner.evicted
    # The old group history is untouched (forked, not rewritten).
    assert [m["content"] for m in store.load_transcript(jim.session_id)] == [
        "Are we doing the zoo Saturday?", "Yes, the zoo."]
    # Renee's lane keeps its history and gains the synopsis as an appended user turn.
    renee_msgs = store.load_transcript(renee.session_id)
    assert [m["content"] for m in renee_msgs[:2]] == ["hi", "hello"]
    assert renee_msgs[-1]["role"] == "user" and SYNOPSIS in renee_msgs[-1]["content"]
    assert result["other_lanes_mirrored"] == 1
    # The private DM session is byte-for-byte unchanged.
    assert store.load_transcript(dm.session_id) == dm_before


def test_failed_post_forks_nothing(store):
    jim, _, _ = _seed_group(store)
    runner = _Runner(store)
    result = asyncio.run(carry_into_group(runner, _FakeAdapter(ok=False), _request(store)))
    assert "error" in result and not result.get("posted")
    key = runner._session_key_for_source(_src(GROUP_ID, "group", JIM))
    assert store._entries[key].session_id == jim.session_id


def test_busy_group_is_refused_before_posting(store):
    _seed_group(store)
    runner, adapter = _Runner(store), _FakeAdapter()
    runner.running.add(runner._session_key_for_source(_src(GROUP_ID, "group", JIM)))
    result = asyncio.run(carry_into_group(runner, adapter, _request(store)))
    assert "error" in result and adapter.sent == []


# --- the model tool: DM-only, membership, explicit approval, then the gateway loop ---------------

@pytest.fixture
def in_dm():
    tokens = set_session_vars(platform="bluebubbles", chat_id=DM_ID, chat_type="dm", user_id=JIM,
                              user_name="Jim", session_key="agent:main:bluebubbles:dm:" + DM_ID)
    yield
    clear_session_vars(tokens)


@pytest.fixture
def gateway_loop(store, monkeypatch):
    """A running loop on its own thread, standing in for the gateway's."""
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    runner, adapter = _Runner(store), _FakeAdapter()
    runner._gateway_loop = loop
    monkeypatch.setattr("tools.send_message_senders._live_adapter", lambda platform, **_: (runner, adapter))
    yield runner, adapter
    loop.call_soon_threadsafe(loop.stop)
    t.join(timeout=5)


def _call(callback, **args):
    return json.loads(tool.carry_to_group_tool(callback=callback, **args))


def test_tool_posts_only_after_explicit_approval(store, in_dm, gateway_loop):
    _seed_group(store)
    runner, adapter = gateway_loop
    shown = []

    def approve(question, choices):
        shown.append((question, choices))
        return tool.CONFIRM_CHOICE

    out = _call(approve, target=GROUP_ID, synopsis=SYNOPSIS)
    assert out["success"] is True, out
    (question, choices), = shown
    assert SYNOPSIS in question and choices == [tool.CONFIRM_CHOICE, tool.CANCEL_CHOICE]
    assert len(adapter.sent) == 1 and adapter.sent[0][0] == GROUP_ID


@pytest.mark.parametrize(("answer", "status"), [
    (tool.CANCEL_CHOICE, "cancelled"),
    ("drop the parking bit", "revise"),
    ("[user did not respond within 5m]", "not_shared"),
])
def test_tool_never_posts_without_approval(store, in_dm, gateway_loop, answer, status):
    _seed_group(store)
    _, adapter = gateway_loop
    out = _call(lambda q, c: answer, target=GROUP_ID, synopsis=SYNOPSIS)
    assert out["status"] == status and adapter.sent == []
    if status == "revise":
        assert out["user_feedback"] == answer


def test_tool_refuses_groups_the_user_is_not_in(store, in_dm, gateway_loop):
    store.get_or_create_session(_src("chat-other-group", "group", RENEE, "Book club"))
    _, adapter = gateway_loop
    asked = []
    out = _call(lambda q, c: asked.append(q) or tool.CONFIRM_CHOICE,
                target="chat-other-group", synopsis=SYNOPSIS)
    assert "error" in out and asked == [] and adapter.sent == []


def test_tool_refuses_outside_a_dm(store, gateway_loop):
    _seed_group(store)
    tokens = set_session_vars(platform="bluebubbles", chat_id=GROUP_ID, chat_type="group", user_id=JIM)
    try:
        out = _call(lambda q, c: tool.CONFIRM_CHOICE, target=GROUP_ID, synopsis=SYNOPSIS)
    finally:
        clear_session_vars(tokens)
    assert "error" in out and gateway_loop[1].sent == []


def test_list_groups_shows_only_the_users_groups(store, in_dm, monkeypatch):
    _seed_group(store)
    store.get_or_create_session(_src("chat-other-group", "group", RENEE, "Book club"))
    directory = {"platforms": {"bluebubbles": [
        {"id": GROUP_ID, "name": "Family", "type": "group"},
        {"id": "chat-other-group", "name": "Book club", "type": "group"},
        {"id": DM_ID, "name": "Jim", "type": "dm"}]}}
    monkeypatch.setattr("gateway.channel_directory.load_directory", lambda: directory)
    out = _call(None, action="list_groups")
    assert out == {"groups": [{"name": "Family", "target": GROUP_ID}]}


def test_toolset_is_off_unless_opted_in():
    from hermes_cli.tools_config import _get_platform_tools
    assert "group_carryover" not in _get_platform_tools({}, "bluebubbles")
    opted_in = {"platform_toolsets": {"bluebubbles": ["hermes-bluebubbles", "group_carryover"]}}
    assert "group_carryover" in _get_platform_tools(opted_in, "bluebubbles")
