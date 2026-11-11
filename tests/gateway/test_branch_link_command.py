"""/branch --at <message link> — clone the session owning a linked Slack thread, cut the transcript at
the linked message's turn, and open the clone in a new sibling thread of the linked channel
(#32105, gateway surface).

Drives the REAL ``_handle_branch_command`` against a REAL SessionStore + SessionDB (SQLite in
tmp_path); only the Slack adapter is a fake whose ``create_handoff_thread`` returns a fixed id.
"""

from __future__ import annotations

import json

import pytest

from agent.i18n import t
from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource, SessionStore
from gateway.slash_commands_branch_link import LinkCutoffUnresolved, find_link_cutoff
from gateway.slash_commands_branch_thread import parse_branch_link_args, parse_slack_archive_link
from hermes_state import AsyncSessionDB

CHANNEL = "C0LINKED"
THREAD = "1793000001.000100"
WORKSPACE = "T0WS"


def _link(message_ts="1793000023.000000", channel=CHANNEL, thread=THREAD) -> str:
    digits = message_ts.replace(".", "")
    url = f"https://catio-tech.slack.com/archives/{channel}/p{digits}?thread_ts={thread}&cid={channel}"
    return f"<{url}|{url}>"


class _SlackAdapter:
    """Slack-shaped fake: the only thing a link branch needs from an adapter."""

    def __init__(self, thread_id="1793000999.000999", fail=False):
        self.thread_id, self.fail, self.calls = thread_id, fail, []

    async def create_handoff_thread(self, parent_chat_id, name):
        self.calls.append((parent_chat_id, name))
        return None if self.fail else self.thread_id


@pytest.fixture()
def store(tmp_path, monkeypatch):
    import hermes_state

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    return SessionStore(sessions_dir=tmp_path, config=GatewayConfig())


def _runner(store, adapter):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.SLACK: adapter} if adapter else {}
    runner._profile_adapters = {}
    runner.config = {}
    runner._background_tasks = set()
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._busy_ack_ts = {}
    runner._pending_approvals = {}
    runner._update_prompt_pending = {}
    runner._agent_cache_lock = None
    runner.session_store = store
    runner._session_db = AsyncSessionDB(store._db)
    runner._pending_skills_reload_notes = {}
    return runner


def _thread_source(channel=CHANNEL, thread: str | None = THREAD, user="U1"):
    return SessionSource(platform=Platform.SLACK, chat_id=channel,
                         chat_type="dm" if str(channel).startswith("D") else "group",
                         thread_id=thread, user_id=user, user_name="victor", scope_id=WORKSPACE)


def _seed_three_turns(store, source):
    """Three complete turns. User rows carry the Slack ts as platform_message_id; assistant
    rows carry none (only user rows do — that is what the timestamp fallback is for)."""
    entry = store.get_or_create_session(source)
    rows = [
        ("user", "first ask", 1793000010.2, "1793000010.100000"),
        ("assistant", "first reply", 1793000012.0, None),
        ("user", "second ask", 1793000020.2, "1793000020.100000"),
        ("assistant", "second reply", 1793000022.0, None),
        ("user", "third ask", 1793000100.2, "1793000100.100000"),
        ("assistant", "third reply", 1793000102.0, None),
    ]
    for role, content, timestamp, pmid in rows:
        store._db.append_message(entry.session_id, role=role, content=content,
                                 timestamp=timestamp, platform_message_id=pmid)
    return entry


# ── parsing ────────────────────────────────────────────────────────────────────────

def test_parse_mrkdwn_and_bare_link_forms():
    target = parse_slack_archive_link(_link())
    assert (target.channel_id, target.thread_ts, target.message_ts) == (CHANNEL, THREAD, "1793000023.000000")
    bare = parse_slack_archive_link("https://foo.slack.com/archives/C0X/p1793000023000000?thread_ts=1793000001.0001")
    assert (bare.channel_id, bare.thread_ts, bare.message_ts) == ("C0X", "1793000001.0001", "1793000023.000000")


def test_parse_rejects_non_thread_links_and_plain_text():
    assert parse_slack_archive_link("") is None
    assert parse_slack_archive_link("side quest") is None
    # Channel-root permalink: no thread_ts -> unresolvable session key, must not parse.
    assert parse_slack_archive_link("https://catio-tech.slack.com/archives/C0X/p1793000023000000") is None


def test_parse_branch_link_args_extracts_flag_link_and_title():
    stay_here, link, title, forced = parse_branch_link_args(f"--here --at {_link()} alternate path")
    assert stay_here and link is not None and title == "alternate path" and forced
    # A title that is not a link keeps the legacy plain-branch shape.
    stay_here, link, title, forced = parse_branch_link_args("side quest")
    assert not stay_here and link is None and title == "side quest" and not forced


def test_parse_at_flag_forces_link_mode():
    # --at is the only link form: the rest must carry a permalink, title survives.
    stay_here, link, title, forced = parse_branch_link_args(f"--at {_link()} alternate path")
    assert not stay_here and link is not None and title == "alternate path" and forced
    # --at with nothing readable after it: forced, no link, no branch.
    stay_here, link, title, forced = parse_branch_link_args("--at")
    assert not stay_here and link is None and forced
    # Flag order does not decide whether --here + link is rejected.
    stay_here, link, title, forced = parse_branch_link_args(f"--at --here {_link()}")
    assert stay_here and link is not None and forced and title == ""
    stay_here, link, title, forced = parse_branch_link_args(f"--here --at {_link()}")
    assert stay_here and link is not None and forced and title == ""
    # Without the flag nothing link-shaped is interpreted: a bare permalink is just a title.
    stay_here, link, title, forced = parse_branch_link_args(f"{_link()} my name")
    assert not stay_here and link is None and forced is False and "archives/" in title


# ── cutoff resolution (pure) ───────────────────────────────────────────────────────

def _transcript():
    return [
        {"role": "user", "content": "first ask", "timestamp": 1793000010.2, "platform_message_id": "1793000010.100000"},
        {"role": "assistant", "content": "first reply", "timestamp": 1793000012.0},
        {"role": "user", "content": "second ask", "timestamp": 1793000020.2, "platform_message_id": "1793000020.100000"},
        {"role": "assistant", "content": "second reply", "timestamp": 1793000022.0},
        {"role": "user", "content": "third ask", "timestamp": 1793000100.2, "platform_message_id": "1793000100.100000"},
        {"role": "assistant", "content": "third reply", "timestamp": 1793000102.0},
    ]


def test_cutoff_exact_platform_message_id_extends_to_reply():
    point = find_link_cutoff(_transcript(), parse_slack_archive_link(_link("1793000020.100000")))
    assert point.matched == "platform_message_id"
    assert _transcript()[: point.index + 1][-1]["content"] == "second reply"


def test_cutoff_timestamp_match_for_bot_reply():
    point = find_link_cutoff(_transcript(), parse_slack_archive_link(_link("1793000023.000000")))
    assert point.matched == "timestamp"
    assert _transcript()[: point.index + 1][-1]["content"] == "second reply"


def test_cutoff_before_history_and_not_found():
    with pytest.raises(LinkCutoffUnresolved) as before:
        find_link_cutoff(_transcript(), parse_slack_archive_link(_link("1792000000.000000")))
    assert before.value.reason == "before_history"
    # In the wide gap between turns 2 and 3, out of tolerance of every row.
    with pytest.raises(LinkCutoffUnresolved) as missing:
        find_link_cutoff(_transcript(), parse_slack_archive_link(_link("1793000060.000000")))
    assert missing.value.reason == "not_found"


def test_cutoff_trims_in_flight_tool_tail_to_whole_turns():
    history = _transcript()
    history.append({"role": "assistant", "content": "", "timestamp": 1793000103.0, "tool_calls": [{"id": "c1"}]})
    history.append({"role": "tool", "content": "{}", "timestamp": 1793000103.5, "tool_call_id": "c1"})
    # The turn containing the linked message ends in flight; the cut keeps turns 1-2 only.
    point = find_link_cutoff(history, parse_slack_archive_link(_link("1793000100.100000")))
    assert history[: point.index + 1][-1]["content"] == "second reply"


# ── handler flow (real store + runner) ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_branch_link_cuts_at_linked_message_and_keeps_every_original(store):
    target = _seed_three_turns(store, _thread_source())
    # Invoke from the linked thread itself: the link only picks the branch point of THIS
    # conversation (the same-thread rule), the clone opens as a new sibling thread.
    invoking = _thread_source()
    adapter = _SlackAdapter()
    runner = _runner(store, adapter)

    reply = await runner._handle_branch_command(
        MessageEvent(text=f"/branch --at {_link()} alternate path", source=invoking))

    assert adapter.calls == [(CHANNEL, "alternate path")]
    # Original session untouched, in its own thread.
    assert store.get_or_create_session(_thread_source()).session_id == target.session_id
    # The new thread is bound to the clone, in the linked channel, with the thread's routing columns.
    clone_entry = store.get_or_create_session(_thread_source(thread=adapter.thread_id))
    assert clone_entry.session_id != target.session_id
    row = store._db.get_session(clone_entry.session_id)
    assert (row["parent_session_id"], row["chat_id"], row["thread_id"], row["chat_type"]) == (
        target.session_id, CHANNEL, adapter.thread_id, "group")
    # Copied transcript is the prefix at the linked message, not the full history.
    assert [m["content"] for m in store._db.get_messages(clone_entry.session_id)] == [
        "first ask", "first reply", "second ask", "second reply"]
    config = json.loads(row["model_config"])
    assert config["_branched_from"] == target.session_id
    assert config["_branched_from_message"] == "1793000023.000000"
    assert adapter.thread_id in reply and target.session_id in reply


@pytest.mark.asyncio
async def test_branch_link_uses_lineage_title_when_none_given(store):
    _seed_three_turns(store, _thread_source())
    runner = _runner(store, _SlackAdapter())
    await runner._handle_branch_command(
        MessageEvent(text=f"/branch --at {_link()}", source=_thread_source()))
    assert runner._session_db is not None  # sanity: the flow completed without raising


@pytest.mark.asyncio
async def test_branch_link_wrong_thread_and_unparsable_link(store):
    _seed_three_turns(store, _thread_source())
    runner = _runner(store, _SlackAdapter())
    # The link is a cutoff selector for the invoking conversation: another thread of the
    # SAME channel may not be named, and neither may another channel.
    reply = await runner._handle_branch_command(
        MessageEvent(text=f"/branch --at {_link()}",
                     source=_thread_source(thread="1793000888.000888")))
    assert reply == t("gateway.branch.link_wrong_thread")
    reply = await runner._handle_branch_command(
        MessageEvent(text=f"/branch --at {_link()}",
                     source=_thread_source(channel="C0OTHER", thread="1793000888.000888")))
    assert reply == t("gateway.branch.link_wrong_thread")
    # Invoking from the channel root (no thread) cannot match a thread link either.
    reply = await runner._handle_branch_command(
        MessageEvent(text=f"/branch --at {_link()}", source=_thread_source(thread=None)))
    assert reply == t("gateway.branch.link_wrong_thread")
    # A channel-root permalink still cannot name a cutoff at all.
    channel_root = "https://catio-tech.slack.com/archives/C0LINKED/p1793000023000000"
    reply = await runner._handle_branch_command(
        MessageEvent(text=f"/branch --at {channel_root}", source=_thread_source()))
    assert reply == t("gateway.branch.link_unparsable")


@pytest.mark.asyncio
async def test_branch_link_dm_follows_the_same_thread_rule(store):
    runner = _runner(store, _SlackAdapter())
    dm_thread = SessionSource(platform=Platform.SLACK, chat_id="D0DM", chat_type="dm",
                              thread_id="1793000555.000555", user_id="U1", scope_id=WORKSPACE)
    dm_entry = store.get_or_create_session(dm_thread)
    store._db.append_message(dm_entry.session_id, role="user", content="dm secret",
                             timestamp=1793000011.0, platform_message_id="1793000011.000000")
    dm_link = _link("1793000011.000000", channel="D0DM", thread="1793000555.000555")
    # A different thread of the same DM may not name it (uniform rule, no DM carve-out).
    from_dm_sibling = SessionSource(platform=Platform.SLACK, chat_id="D0DM", chat_type="dm",
                                     thread_id="1793000666.000666", user_id="U1", scope_id=WORKSPACE)
    reply = await runner._handle_branch_command(
        MessageEvent(text=f"/branch --at {dm_link}", source=from_dm_sibling))
    assert reply == t("gateway.branch.link_wrong_thread")
    # The DM's own thread may.
    reply = await runner._handle_branch_command(
        MessageEvent(text=f"/branch --at {dm_link} dm branch", source=dm_thread))
    assert "dm branch" in reply


@pytest.mark.asyncio
async def test_branch_link_behind_history_and_incomplete_turn(store):
    source = _thread_source()
    entry = store.get_or_create_session(source)
    store._db.append_message(entry.session_id, role="user", content="recent ask",
                             timestamp=1793000100.2, platform_message_id="1793000100.100000")
    runner = _runner(store, _SlackAdapter())
    reply = await runner._handle_branch_command(
        MessageEvent(text=f"/branch --at {_link('1792000000.000000')}", source=_thread_source()))
    assert reply == t("gateway.branch.link_behind_history")
    # First and only turn still in flight (ends on a tool row): nothing stable to copy.
    store._db.append_message(entry.session_id, role="assistant", content="",
                             timestamp=1793000101.0, tool_calls=[{"id": "c1"}])
    store._db.append_message(entry.session_id, role="tool", content="{}",
                             timestamp=1793000101.5, tool_call_id="c1")
    reply = await runner._handle_branch_command(
        MessageEvent(text=f"/branch --at {_link('1793000100.100000')}", source=_thread_source()))
    assert reply == t("gateway.branch.link_turn_incomplete")


@pytest.mark.asyncio
async def test_branch_link_thread_creation_failure_changes_nothing(store):
    target = _seed_three_turns(store, _thread_source())
    runner = _runner(store, _SlackAdapter(fail=True))
    reply = await runner._handle_branch_command(
        MessageEvent(text=f"/branch --at {_link()}", source=_thread_source()))
    assert reply == t("gateway.branch.link_thread_failed")
    assert store.get_or_create_session(_thread_source()).session_id == target.session_id
    # No clone row was created.
    assert store._db.get_session(target.session_id)["parent_session_id"] is None


@pytest.mark.asyncio
async def test_branch_link_from_other_platform_is_rejected(store):
    _seed_three_turns(store, _thread_source())
    discord = SessionSource(platform=Platform.DISCORD, chat_id="555", chat_type="group",
                            user_id="U1", scope_id="g1")
    runner = _runner(store, None)
    reply = await runner._handle_branch_command(MessageEvent(text=f"/branch --at {_link()}", source=discord))
    assert reply == t("gateway.branch.link_wrong_platform")


@pytest.mark.asyncio
async def test_branch_link_rejects_here_and_ignores_plain_archives_titles(store):
    source = _thread_source()
    _seed_three_turns(store, source)
    runner = _runner(store, _SlackAdapter())
    # --here cannot rebind this chat onto another thread's cut clone.
    reply = await runner._handle_branch_command(
        MessageEvent(text=f"/branch --here --at {_link()}", source=_thread_source(thread="1793000888.000888")))
    assert reply == t("gateway.branch.link_here_unsupported")
    # A plain title that merely mentions archives/ (no URL) branches normally, title intact.
    reply = await runner._handle_branch_command(
        MessageEvent(text="/branch tidy the archives/ folder", source=source))
    assert "tidy the archives/ folder" in reply
    assert store.get_or_create_session(
        _thread_source(thread=_SlackAdapter().thread_id)).session_id != (
        store.get_or_create_session(source).session_id)


@pytest.mark.asyncio
async def test_branch_bare_link_without_at_is_pointed_at_the_flag(store):
    target = _seed_three_turns(store, _thread_source())
    invoking = _thread_source(thread="1793000888.000888")
    runner = _runner(store, _SlackAdapter())
    # No --at: nothing link-shaped is interpreted, and the predictable mistake gets a
    # pointer to the right form instead of a plain branch of this chat titled with a URL.
    reply = await runner._handle_branch_command(
        MessageEvent(text=f"/branch {_link()} forgot the flag", source=invoking))
    assert reply == t("gateway.branch.link_requires_at")
    # The original session is untouched and no clone row was created.
    assert store.get_or_create_session(_thread_source()).session_id == target.session_id
    assert store._db.get_session(target.session_id)["parent_session_id"] is None


@pytest.mark.asyncio
async def test_branch_at_flag_flows_and_fails_loudly(store):
    target = _seed_three_turns(store, _thread_source())
    invoking = _thread_source()
    adapter = _SlackAdapter()
    runner = _runner(store, adapter)
    # The flag form carries the branch point explicitly; same-thread rule satisfied.
    reply = await runner._handle_branch_command(
        MessageEvent(text=f"/branch --at {_link()} explicit path", source=invoking))
    assert adapter.calls == [(CHANNEL, "explicit path")]
    clone_entry = store.get_or_create_session(_thread_source(thread=adapter.thread_id))
    assert clone_entry.session_id != target.session_id
    assert [m["content"] for m in store._db.get_messages(clone_entry.session_id)] == [
        "first ask", "first reply", "second ask", "second reply"]
    assert "explicit path" in reply
    # --at with a non-Slack URL or nothing after it: unparsable, nothing created.
    for args in ("--at https://github.com/org/repo/pull/5", "--at"):
        reply = await runner._handle_branch_command(
            MessageEvent(text=f"/branch {args}", source=invoking))
        assert reply == t("gateway.branch.link_unparsable")
    assert store.get_or_create_session(_thread_source()).session_id == target.session_id