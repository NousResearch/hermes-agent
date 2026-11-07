"""Profile-wide ``gateway.allow_silence``: the fallback below a room's own setting, never for DMs."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, load_gateway_config
from gateway.session_identity import RoutingIdentity
from tests.gateway.test_active_turn_recovery import _close_store_db, _db_runner, _entry_for
from tests.gateway.test_gateway_silence_tokens import _event, _runner, _source

_HEALTHY = {"final_response": "NO_REPLY", "failed": False, "api_calls": 1}


def _served(runner, tmp_path, raw, *, chat_type="group", launch_opt_in=False):
    """A turn routed to the ``served`` profile, whose config.yaml is ``raw``; the launch snapshot
    holds ``launch_opt_in`` so a leak in either direction flips the verdict."""
    home = tmp_path / "served"
    home.mkdir(exist_ok=True)
    (home / "config.yaml").write_text(json.dumps(raw), encoding="utf-8")
    runner.config = GatewayConfig.from_dict({"allow_silence": launch_opt_in})
    runner.config.multiplex_profiles = True
    source = _source()
    source.chat_type = chat_type
    source._identity = RoutingIdentity("default", "served", tmp_path, home)
    return source


_ON = {"gateway": {"allow_silence": True}}


@pytest.mark.parametrize("raw,chat_type,display_kind,reply_expected,result,silent", [
    # The opt-in covers addressed and unknown-addressing human turns in shared chats.
    (_ON, "group", None, True, _HEALTHY, True),
    (_ON, "channel", None, None, _HEALTHY, True),
    ({"allow_silence": True}, "thread", None, True, _HEALTHY, True),
    # The loader's precedence: the top-level form wins by presence over ``gateway.allow_silence``.
    ({"allow_silence": False, **_ON}, "group", None, True, _HEALTHY, False),
    # A direct question must never vanish (#110952), whatever the profile says.
    (_ON, "dm", None, True, _HEALTHY, False),
    # Off, absent or unparseable keeps the visible fallback.
    ({"gateway": {"allow_silence": False}}, "group", None, True, _HEALTHY, False),
    ({}, "group", None, True, _HEALTHY, False),
    ({"gateway": {"allow_silence": "maybe"}}, "group", None, True, _HEALTHY, False),
    # A room's explicit false wins over the profile's true.
    ({**_ON, "platforms": {"telegram": {"channel_overrides": {"-1001": {"allow_silence": False}}}}},
     "group", None, True, _HEALTHY, False),
    # A room entry that sets only other keys inherits the profile value.
    ({**_ON, "platforms": {"telegram": {"channel_overrides": {"-1001": {"model": "m"}}}}},
     "group", None, True, _HEALTHY, True),
    # The opt-in requires a healthy, completed result on an addressed turn...
    (_ON, "group", None, True, {**_HEALTHY, "partial": True}, False),
    (_ON, "group", None, True, {**_HEALTHY, "completed": False}, False),
    # ...but never adds that gate to a turn that could already stay silent.
    (_ON, "group", None, False, {**_HEALTHY, "partial": True}, True),
    (_ON, "group", None, False, {**_HEALTHY, "completed": False, "api_calls": 3}, True),
    (_ON, "dm", "internal_notification", True, {**_HEALTHY, "interrupted": True}, True),
], ids=[
    "addressed-group", "unknown-channel", "top-level-form", "top-level-wins", "dm-excluded", "off", "absent",
    "non-bool", "room-false-wins", "model-only-room-inherits", "partial-addressed",
    "incomplete-addressed", "partial-unaddressed", "incomplete-unaddressed", "internal-unhealthy",
])
def test_profile_opt_in_truth_table(monkeypatch, tmp_path, raw, chat_type, display_kind,
                                    reply_expected, result, silent):
    runner = _runner(monkeypatch, tmp_path)
    source = _served(runner, tmp_path, raw, chat_type=chat_type)

    assert runner._silence_allowed_for_source(source, display_kind, reply_expected, result) is silent


@pytest.mark.parametrize("launch_opt_in,served_raw,silent", [
    (True, {}, False),
    (False, _ON, True),
], ids=["launch-on-served-off", "launch-off-served-on"])
def test_routed_profile_reads_its_own_setting(monkeypatch, tmp_path, launch_opt_in, served_raw, silent):
    runner = _runner(monkeypatch, tmp_path)
    source = _served(runner, tmp_path, served_raw, launch_opt_in=launch_opt_in)

    assert runner._silence_allowed_for_source(source, None, True, _HEALTHY) is silent


@pytest.mark.parametrize("runtime,silent", [("primary", True), ("served", False)])
def test_primary_profile_uses_the_supplied_config(monkeypatch, tmp_path, runtime, silent):
    runner = _runner(monkeypatch, tmp_path)
    runner._primary_profile_name = "primary"
    # A ``gateway --config`` file (runner.config) opts in; the profile home's file does not.
    runner.config = GatewayConfig.from_dict({"allow_silence": True})
    home = tmp_path / runtime
    home.mkdir()
    (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    source = _source()
    source._identity = RoutingIdentity("primary", runtime, tmp_path, home, multiplexed=False)

    assert runner._silence_allowed_for_source(source, None, True, _HEALTHY) is silent


@pytest.mark.asyncio
@pytest.mark.parametrize("final,silent", [("An ordinary reply.", False), ("NO_REPLY", True)])
async def test_profile_setting_is_read_only_for_a_marker(monkeypatch, tmp_path, final, silent):
    """An ordinary reply never opens the routed profile's config.yaml (no per-turn lookup)."""
    import gateway.run as gateway_run
    runner = _runner(monkeypatch, tmp_path)
    source = _served(runner, tmp_path, _ON)
    real_load = gateway_run._load_gateway_config
    loads = []
    monkeypatch.setattr(gateway_run, "_load_gateway_config",
                        lambda path=None: loads.append(path) or real_load(path))
    runner._run_agent = AsyncMock(return_value={**_HEALTHY, "final_response": final, "messages": []})

    response = await runner._handle_message_with_agent(_event(reply_expected=True), source, "lane", 1)

    served_reads = [p for p in loads if p == tmp_path / "served" / "config.yaml"]
    assert (not response) if silent else (final in response)
    assert bool(served_reads) is silent


def test_failed_profile_lookup_keeps_the_reply_rule(monkeypatch, tmp_path):
    from gateway.profile_routing import ProfileRouteRejected
    runner = _runner(monkeypatch, tmp_path)
    source = _source()
    source.profile = "gone"

    def _reject(_source):
        raise ProfileRouteRejected("gone")

    monkeypatch.setattr(runner, "_resolve_profile_home_for_source", _reject)

    assert runner._profile_allow_silence(source) is None
    assert runner._silence_allowed_for_source(source, None, True, _HEALTHY) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["normal", "queued", "recovery"])
@pytest.mark.parametrize("chat_type,silent", [("group", True), ("dm", False)])
async def test_profile_opt_in_reaches_every_delivery_lane(monkeypatch, tmp_path, lane, chat_type, silent):
    runner = _runner(monkeypatch, tmp_path)
    source = _served(runner, tmp_path, _ON, chat_type=chat_type)
    event = _event(reply_expected=True)
    event.source = source
    result = {**_HEALTHY, "messages": [
        {"role": "user", "content": "question", "display_metadata": {"reply_expected": True}},
        {"role": "assistant", "content": "NO_REPLY"}]}
    if lane == "normal":
        adapter = SimpleNamespace(send=AsyncMock())
        runner._delivery_adapter_for = lambda _: adapter
        runner._run_agent = AsyncMock(return_value=result)
        runner._hmwa_runtime_footer_line = lambda *_: "usage footer"
        response = await runner._handle_message_with_agent(event, source, "lane", 1)
        assert bool(response) is not silent
        assert all(call.args[1] != "usage footer" for call in adapter.send.await_args_list)
        rows = [c.args[1] for c in runner.session_store.append_to_transcript.call_args_list]
        # The silent turn stays a persisted human/assistant exchange.
        assert any(r.get("content") == "NO_REPLY" for r in rows)
    elif lane == "queued":
        runner._deliver_queued_first_response = AsyncMock(return_value=True)
        ctx = SimpleNamespace(session_key="lane", source=source, stream_consumer_holder=[None],
                              mute_notification_reply=False, persist_user_display_kind=None,
                              reply_expected=True, _status_thread_metadata=None,
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
            # Finish the real turn, then crash before the adapter sends or clears its token.
            await runner._handle_message_with_agent(event, source, store._generate_session_key(source), 1)
            _close_store_db(store)
            recovered, store = _db_runner(tmp_path)
            assert await recovered._recover_unclean_sessions() == (0, 0 if silent else 1)
            assert bool(sweep_recoverable(deliverable_platforms={"telegram"})) is not silent
            assert _entry_for(store, source).active_turn_token is None
        finally:
            _close_store_db(store)


@pytest.mark.parametrize("yaml_text,expected", [
    ("gateway:\n  allow_silence: true\n", True),
    ("allow_silence: true\n", True),
    ("allow_silence: false\ngateway:\n  allow_silence: true\n", False),
    ("gateway:\n  allow_silence: 'off'\n", False),
    ("{}\n", False),
], ids=["nested", "top-level", "top-level-wins", "string-off", "absent"])
def test_load_gateway_config_reads_the_profile_setting(tmp_path, monkeypatch, yaml_text, expected):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(yaml_text, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    config = load_gateway_config()

    assert config.allow_silence is expected
    assert GatewayConfig.from_dict(config.to_dict()).allow_silence is expected


def test_config_cli_knows_both_spellings():
    from hermes_cli.config import _validate_config_key
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["gateway"]["allow_silence"] is False
    assert _validate_config_key("gateway.allow_silence") == (True, None)
    assert _validate_config_key("allow_silence") == (True, None)
