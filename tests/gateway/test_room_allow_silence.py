"""Room opt-in reaches live, queued and crash-left delivery without crossing profiles."""

import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig
from gateway.session_identity import RoutingIdentity
from tests.gateway.test_gateway_silence_tokens import _event, _runner, _source
from tests.gateway.test_active_turn_recovery import _close_store_db, _db_runner, _entry_for


def _configure(runner, tmp_path, *, setting=True, other_profile=False):
    home = tmp_path / "served"
    home.mkdir()
    room = {} if setting is None else {"allow_silence": None if setting == "null" else setting}
    raw = {"platforms": {"telegram": {"channel_overrides": {
        "-1001": room, "parent": {"allow_silence": True},
    }}}}
    if setting in ("thread", "parent"):
        del raw["platforms"]["telegram"]["channel_overrides"]["-1001"]
        raw["platforms"]["telegram"]["channel_overrides"][setting] = {"allow_silence": True}
    (home / "config.yaml").write_text(json.dumps(raw), encoding="utf-8")
    loaded = GatewayConfig.from_dict(raw)
    roundtrip = loaded.to_dict()["platforms"]["telegram"]["channel_overrides"]
    expected = {} if setting == "null" else room
    if setting not in ("thread", "parent"):
        assert roundtrip["-1001"] == expected
    # The launch snapshot deliberately disagrees with the served profile.
    runner.config = GatewayConfig.from_dict({"platforms": {"telegram": {
        "channel_overrides": {"-1001": {"allow_silence": bool(other_profile) or not bool(setting)}}}}})
    runner.config.multiplex_profiles = True
    source = _source()
    source.parent_chat_id = "parent"
    source.thread_id = "thread"
    if other_profile:
        home = tmp_path / "other"
        home.mkdir()
        (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    source._identity = RoutingIdentity("default", "served", tmp_path, home)
    return source


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["normal", "queued", "recovery"])
@pytest.mark.parametrize("setting,other_profile,reply_expected,internal,silent", [
    (True, False, True, False, True),
    (True, False, None, False, True),
    (False, False, True, False, False),
    (None, False, None, False, False),
    ("null", False, True, False, False),
    ("thread", False, True, False, True),
    ("parent", False, True, False, True),
    (True, True, True, False, False),
    (False, False, False, False, True),
    (False, False, True, True, True),
])
async def test_room_config_controls_delivery(monkeypatch, tmp_path, lane, setting,
                                            other_profile, reply_expected, internal, silent):
    runner = _runner(monkeypatch, tmp_path)
    source = _configure(runner, tmp_path, setting=setting, other_profile=other_profile)
    event = _event(internal=internal, reply_expected=reply_expected)
    event.source = source
    result = {"final_response": "NO_REPLY", "failed": False, "api_calls": 1,
              "messages": [{"role": "user", "content": "question",
                            "display_kind": "internal_notification" if internal else None,
                            "display_metadata": {"reply_expected": reply_expected}},
                           {"role": "assistant", "content": "NO_REPLY"}]}
    if lane == "normal":
        adapter = SimpleNamespace(send=AsyncMock())
        runner._delivery_adapter_for = lambda _: adapter
        if setting is True and reply_expected is None:
            result["already_sent"] = True
        runner._run_agent = AsyncMock(return_value=result)
        # Even a streaming result must not create a usage-only final.
        runner._hmwa_runtime_footer_line = lambda *_: "usage footer"
        response = await runner._handle_message_with_agent(event, source, "lane", 1)
        assert bool(response) is not silent
        assert all(call.args[1] != "usage footer" for call in adapter.send.await_args_list)
        rows = [c.args[1] for c in runner.session_store.append_to_transcript.call_args_list]
        assert any(r.get("content") == "NO_REPLY" for r in rows)
    elif lane == "queued":
        runner._deliver_queued_first_response = AsyncMock(return_value=True)
        ctx = SimpleNamespace(session_key="lane", source=source, stream_consumer_holder=[None],
                              mute_notification_reply=False, persist_user_display_kind=(
                                  "internal_notification" if internal else None),
                              reply_expected=reply_expected, _status_thread_metadata=None,
                              event_message_id=None, inbound_message_id="42", run_generation=1)
        await runner._run_agent_deliver_first_response(ctx, None, result, result, None)
        assert runner._deliver_queued_first_response.called is not silent
    else:
        from gateway.delivery_ledger import sweep_recoverable
        _, store = _db_runner(tmp_path)
        runner.session_store = store
        result["agent_persisted"] = False
        runner._run_agent = AsyncMock(return_value=result)
        try:
            # Finish the real normal turn, then crash before the adapter sends or clears its token.
            await runner._handle_message_with_agent(event, source, store._generate_session_key(source), 1)
            # Re-open the routing index and DB: the verdict must survive a process restart.
            _close_store_db(store)
            recovered, store = _db_runner(tmp_path)
            assert await recovered._recover_unclean_sessions() == (0, 0 if silent else 1)
            assert bool(sweep_recoverable(deliverable_platforms={"telegram"})) is not silent
            assert _entry_for(store, source).active_turn_token is None
        finally:
            _close_store_db(store)


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["normal", "queued", "recovery"])
@pytest.mark.parametrize("changes", [
    {"failed": True}, {"partial": True}, {"interrupted": True},
    {"completed": False}, {"error": "provider error"}, {"final_response": ""},
    {"final_response": "An ordinary reply."},
])
async def test_opt_in_does_not_hide_unhealthy_or_empty_results(monkeypatch, tmp_path, lane, changes):
    runner = _runner(monkeypatch, tmp_path)
    source = _configure(runner, tmp_path)
    result = {"final_response": "NO_REPLY", "failed": False, "api_calls": 1, "messages": [], **changes}
    if lane == "normal":
        runner._run_agent = AsyncMock(return_value=result)
        assert await runner._handle_message_with_agent(_event(reply_expected=True), source, "lane", 1)
    elif lane == "queued":
        from gateway.run import _normalize_empty_agent_response
        result = {**result, "final_response": _normalize_empty_agent_response(result, result["final_response"])}
        runner._deliver_queued_first_response = AsyncMock(return_value=True)
        ctx = SimpleNamespace(session_key="lane", source=source, stream_consumer_holder=[None],
                              mute_notification_reply=False, persist_user_display_kind=None,
                              reply_expected=True, _status_thread_metadata=None,
                              event_message_id=None, inbound_message_id="42", run_generation=1)
        await runner._run_agent_deliver_first_response(ctx, None, result, result, None)
        runner._deliver_queued_first_response.assert_awaited_once()
    else:
        from gateway.delivery_ledger import sweep_recoverable
        recovered, store = _db_runner(tmp_path)
        recovered.config = runner.config
        entry = store.get_or_create_session(source)
        store.mark_turn_active(entry.session_key)
        store.append_to_transcript(entry.session_id, {"role": "user", "content": "question"})
        store.append_to_transcript(entry.session_id, {"role": "assistant", "content": result["final_response"]})
        try:
            _, intentional, _ = await recovered._hmwa_shape_agent_response(
                result, source, [], entry, entry.session_key, entry.session_key, 1,
                entry.session_id, "telegram", time.time(), reply_expected=True)
            await recovered._record_room_silence_decision(entry.session_key, result, intentional)
            resumed, ledgered = await recovered._recover_unclean_sessions()
            assert resumed + ledgered == 1
            if result["final_response"]:
                assert sweep_recoverable(deliverable_platforms={"telegram"})
        finally:
            _close_store_db(store)


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_opted_in", [False, True])
async def test_queued_terminal_source_owns_policy(monkeypatch, tmp_path, terminal_opted_in):
    runner = _runner(monkeypatch, tmp_path)
    opted = _configure(runner, tmp_path)
    from gateway.session_identity import replace_source
    guarded_home = tmp_path / "guarded"
    guarded_home.mkdir()
    (guarded_home / "config.yaml").write_text("{}\n", encoding="utf-8")
    guarded = replace_source(opted, profile="guarded")
    guarded._identity = RoutingIdentity("default", "guarded", tmp_path, guarded_home)
    opener, terminal = (guarded, opted) if terminal_opted_in else (opted, guarded)
    runner._run_agent = AsyncMock(return_value={
        "final_response": "NO_REPLY", "api_calls": 1, "messages": [],
        "queued_terminal_inbound_id": "43", "queued_terminal_source": terminal,
        "queued_terminal_display_kind": None, "queued_terminal_reply_expected": True,
    })
    response = await runner._handle_message_with_agent(_event(), opener, "lane", 1)
    assert bool(response) is not terminal_opted_in


@pytest.mark.asyncio
async def test_agent_persisted_final_survives_trailing_session_metadata(monkeypatch, tmp_path):
    from gateway.delivery_ledger import sweep_recoverable
    runner = _runner(monkeypatch, tmp_path)
    source = _configure(runner, tmp_path)
    _, store = _db_runner(tmp_path)
    runner.session_store = store
    event = _event(reply_expected=True)
    event.source = source
    messages = [{"role": "user", "content": "question", "display_metadata": {"reply_expected": True}},
                {"role": "assistant", "content": "NO_REPLY"}]

    async def persist_agent_result(*_args, **_kwargs):
        entry = store.get_or_create_session(source)
        for message in messages:
            store.append_to_transcript(entry.session_id, message)
        return {"final_response": "NO_REPLY", "failed": False, "api_calls": 1,
                "agent_persisted": True, "messages": messages}

    runner._run_agent = AsyncMock(side_effect=persist_agent_result)
    try:
        assert await runner._handle_message_with_agent(event, source, store._generate_session_key(source), 1) == ""
        entry = _entry_for(store, source)
        assert store.load_transcript(entry.session_id)[-1]["role"] == "session_meta"
        _close_store_db(store)
        recovered, store = _db_runner(tmp_path)
        assert await recovered._recover_unclean_sessions() == (0, 0)
        assert not sweep_recoverable(deliverable_platforms={"telegram"})
        assert _entry_for(store, source).active_turn_token is None
    finally:
        _close_store_db(store)


@pytest.mark.asyncio
@pytest.mark.parametrize("mismatch", ["missing", "token", "timestamp"])
async def test_recovery_keeps_guard_without_exact_completed_decision(tmp_path, mismatch):
    from gateway.delivery_ledger import sweep_recoverable
    runner, store = _db_runner(tmp_path)
    source = _source()
    entry = store.get_or_create_session(source)
    token = store.mark_turn_active(entry.session_key)
    store.append_to_transcript(entry.session_id, {"role": "user", "content": "question"})
    store.append_to_transcript(entry.session_id, {"role": "assistant", "content": "NO_REPLY"})
    try:
        if mismatch != "missing":
            history = store.load_transcript(entry.session_id)
            store.set_session_metadata(entry.session_key, "room_silence_decision", {
                "token": "old-token" if mismatch == "token" else token,
                "timestamp": 0 if mismatch == "timestamp" else history[-1]["timestamp"],
                "silent": True,
            })
        assert await runner._recover_unclean_sessions() == (0, 1)
        assert sweep_recoverable(deliverable_platforms={"telegram"})
    finally:
        _close_store_db(store)
