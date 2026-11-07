"""``display.suppress_warning_notifications`` also hides the unexpected-silence notice.

On a human turn the notice is the reply text itself, so the adapter's warning boundary never sees
it. With suppression on for the turn's platform (globally or per platform), a bare silence marker
on a healthy completed turn stays silent instead, on the live, queued and crash-recovery paths.
Failed, partial, interrupted, error and empty results keep their presentation. Driven through the
real turn path with the policy read from the profile's config.yaml.
"""

import logging

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run_turn import _unexpected_silence_reply
from gateway.session_identity import RoutingIdentity
from hermes_constants import get_hermes_home
from tests.gateway.test_gateway_silence_tokens import _runner
from tests.gateway.test_platform_allow_silence import (
    CHANNEL_ID, THREAD, _event, _live, _queued, _recovered, _result, _slack_source, _wa_source, _write,
)

GLOBAL = {"display": {"suppress_warning_notifications": True}}


def _per_platform(platform, value=True, **display):
    return {"display": {**display, "platforms": {platform: {"suppress_warning_notifications": value}}}}


def _profile(raw):
    _write(get_hermes_home(), raw)


def _source(platform):
    return _slack_source() if platform == "slack" else _wa_source()


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", ["slack", "whatsapp"])
@pytest.mark.parametrize("scope", ["global", "per-platform"])
@pytest.mark.parametrize("reply_expected", [None, True])
async def test_suppression_keeps_a_healthy_marker_silent(monkeypatch, tmp_path, platform, scope, reply_expected):
    _profile(GLOBAL if scope == "global" else _per_platform(platform))
    runner = _runner(monkeypatch, tmp_path)

    assert await _live(runner, _event(_source(platform), reply_expected=reply_expected)) == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("raw,platform", [
    ({}, "slack"),
    ({"display": {"suppress_warning_notifications": False}}, "whatsapp"),
    ({"display": {"suppress_warning_notifications": "maybe"}}, "slack"),
    (_per_platform("slack"), "whatsapp"),
    (_per_platform("whatsapp", False, suppress_warning_notifications=True), "whatsapp"),
], ids=["unset", "false", "invalid-value", "other-platform-only", "platform-false-beats-global"])
async def test_notice_stays_where_warnings_are_shown(monkeypatch, tmp_path, raw, platform):
    _profile(raw)
    runner = _runner(monkeypatch, tmp_path)

    assert await _live(runner, _event(_source(platform))) == _unexpected_silence_reply()


@pytest.mark.asyncio
@pytest.mark.parametrize("raw,sent", [(GLOBAL, None), ({}, "notice")])
async def test_queued_first_response_follows_suppression(monkeypatch, tmp_path, raw, sent):
    _profile(raw)
    runner = _runner(monkeypatch, tmp_path)

    text = await _queued(runner, _wa_source(), reply_expected=True)

    assert text == (None if sent is None else _unexpected_silence_reply())


@pytest.mark.asyncio
@pytest.mark.parametrize("raw,owed", [(_per_platform("slack"), False), ({}, True)])
async def test_crash_recovery_follows_the_live_decision_under_suppression(monkeypatch, tmp_path, raw, owed):
    _profile(raw)
    runner = _runner(monkeypatch, tmp_path)

    assert bool(await _recovered(runner, _event(_slack_source()), tmp_path, "slack")) is owed


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["normal", "queued"])
@pytest.mark.parametrize("changes", [
    {"failed": True}, {"partial": True}, {"interrupted": True},
    {"completed": False}, {"error": "provider error"}, {"final_response": ""},
])
async def test_suppression_does_not_change_unhealthy_or_empty_results(monkeypatch, tmp_path, lane, changes):
    """Whatever main presents for these results, it presents the same under suppression."""
    from gateway.run import _normalize_empty_agent_response

    async def presented(raw, home):
        monkeypatch.setenv("HERMES_HOME", str(home))
        _write(home, raw)
        runner = _runner(monkeypatch, tmp_path)
        result = _result(**changes)
        if lane == "normal":
            return await _live(runner, _event(_slack_source(), reply_expected=True), result)
        result = {**result, "final_response": _normalize_empty_agent_response(result, result["final_response"])}
        return await _queued(runner, _slack_source(), result, reply_expected=True)

    shown = await presented({}, tmp_path / "shown")
    hidden = await presented(GLOBAL, tmp_path / "hidden")

    assert shown and hidden == shown


@pytest.mark.asyncio
async def test_relay_fronted_slack_follows_the_slack_display_override(monkeypatch, tmp_path):
    from gateway.relay.ws_transport import _event_from_wire
    _profile(_per_platform("slack"))
    event = _event_from_wire({"text": "lol same", "message_id": "m9", "source": {
        "platform": "slack", "chat_id": CHANNEL_ID, "chat_type": "group", "thread_id": THREAD}})
    assert event.source.platform is Platform.SLACK and event.reply_expected is None
    runner = _runner(monkeypatch, tmp_path)

    assert await _live(runner, event) == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("served_hides,launch_hides,silent", [(True, False, True), (False, True, False)])
async def test_routed_profile_uses_its_own_display_policy(
        monkeypatch, tmp_path, served_hides, launch_hides, silent):
    _profile(GLOBAL if launch_hides else {})
    runner = _runner(monkeypatch, tmp_path)
    runner.config = GatewayConfig()
    runner.config.multiplex_profiles = True
    served = tmp_path / "served"
    _write(served, GLOBAL if served_hides else {})
    source = _slack_source()
    source._identity = RoutingIdentity("default", "served", tmp_path, served)

    response = await _live(runner, _event(source))

    assert response == ("" if silent else _unexpected_silence_reply())


@pytest.mark.asyncio
@pytest.mark.parametrize("platform_block", [
    {"channel_overrides": {CHANNEL_ID: {"allow_silence": False}}},
    {"allow_silence": False},
    {"allow_silence": True, "channel_overrides": {CHANNEL_ID: {"allow_silence": False}}},
], ids=["room-false", "platform-false", "platform-true-room-false"])
async def test_an_explicit_allow_silence_false_wins_over_suppression(monkeypatch, tmp_path, platform_block):
    """allow_silence: false is the operator saying 'always show the notice here'; the global
    suppression flag only decides where allow_silence is unset."""
    _profile({**GLOBAL, "platforms": {"slack": platform_block}})
    from gateway.config import load_gateway_config
    runner = _runner(monkeypatch, tmp_path)
    runner.config = load_gateway_config()

    assert await _live(runner, _event(_slack_source())) == _unexpected_silence_reply()


@pytest.mark.asyncio
async def test_suppression_silences_a_room_entry_that_leaves_allow_silence_unset(monkeypatch, tmp_path):
    _profile({**GLOBAL, "platforms": {"slack": {"channel_overrides": {CHANNEL_ID: {"model": "openai/gpt-5-mini"}}}}})
    from gateway.config import load_gateway_config
    runner = _runner(monkeypatch, tmp_path)
    runner.config = load_gateway_config()

    assert await _live(runner, _event(_slack_source())) == ""


@pytest.mark.asyncio
async def test_suppressed_notice_is_still_logged(monkeypatch, tmp_path, caplog):
    _profile(GLOBAL)
    runner = _runner(monkeypatch, tmp_path)

    with caplog.at_level(logging.INFO, logger="gateway.run"):
        assert await _live(runner, _event(_wa_source())) == ""

    assert any("warning notifications are suppressed" in r.getMessage() for r in caplog.records)

