"""Tests for the Helm L2 outbound secret simulation guard (LOCAL, DEFAULT-OFF).

These tests exercise the simulation block path end to end without any live
send: a recording adapter asserts `.send()` is never called when a message is
blocked, and the redacted incident report / repair payload are checked to never
contain the raw sentinel.
"""

import json
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.delivery import DeliveryRouter, DeliveryTarget
from gateway.platforms.base import SendResult
import gateway.outbound_secret_simulation as sim
from gateway.outbound_secret_simulation import (
    HERMES_L2_FAKE_SECRET_SENTINEL,
    SIMULATION_ENV_FLAG,
    evaluate_outbound,
    is_simulation_enabled,
)


class RecordingAdapter:
    def __init__(self):
        self.calls = []

    async def send(self, chat_id, content, metadata=None):
        self.calls.append({"chat_id": chat_id, "content": content})
        return SendResult(success=True, message_id="sent")


class KwargsRecordingAdapter:
    """Adapter that records keyword send calls (matches run/stream signatures)."""

    platform = None

    def __init__(self):
        self.calls = []
        self.draft_calls = []

    async def send(self, chat_id=None, content=None, **kwargs):
        self.calls.append({"chat_id": chat_id, "content": content, **kwargs})
        return SendResult(success=True, message_id="sent")

    async def send_draft(self, **kwargs):
        self.draft_calls.append(kwargs)
        return SendResult(success=True, message_id="draft")


@pytest.fixture
def incidents_home(tmp_path, monkeypatch):
    """Point incident reports at a throwaway profile-aware home."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


# --- pure evaluator -------------------------------------------------------


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv(SIMULATION_ENV_FLAG, raising=False)
    assert is_simulation_enabled() is False
    verdict = evaluate_outbound(f"hello {HERMES_L2_FAKE_SECRET_SENTINEL} world")
    assert verdict.blocked is False


def test_enabled_but_no_sentinel_is_not_blocked(monkeypatch):
    monkeypatch.setenv(SIMULATION_ENV_FLAG, "1")
    verdict = evaluate_outbound("a perfectly ordinary message")
    assert verdict.blocked is False


@pytest.mark.parametrize("flag", ["1", "true", "YES", "on"])
def test_truthy_flags_arm_the_gate(monkeypatch, flag):
    monkeypatch.setenv(SIMULATION_ENV_FLAG, flag)
    assert is_simulation_enabled() is True


@pytest.mark.parametrize("flag", ["0", "false", "off", "", "no"])
def test_falsey_flags_keep_gate_disabled(monkeypatch, flag):
    monkeypatch.setenv(SIMULATION_ENV_FLAG, flag)
    assert is_simulation_enabled() is False


def test_enabled_blocks_on_sentinel_and_redacts(monkeypatch, incidents_home):
    monkeypatch.setenv(SIMULATION_ENV_FLAG, "1")
    content = f"please send {HERMES_L2_FAKE_SECRET_SENTINEL} now"
    verdict = evaluate_outbound(content, platform="discord", target="chan-1")

    assert verdict.blocked is True
    # raw sentinel never surfaces in any returned field
    assert HERMES_L2_FAKE_SECRET_SENTINEL not in verdict.redacted_content
    assert HERMES_L2_FAKE_SECRET_SENTINEL not in json.dumps(verdict.repair_payload)
    assert "[REDACTED_L2_FAKE_SENTINEL]" in verdict.redacted_content


def test_incident_report_written_and_redacted(monkeypatch, incidents_home):
    monkeypatch.setenv(SIMULATION_ENV_FLAG, "1")
    content = f"leak {HERMES_L2_FAKE_SECRET_SENTINEL}"
    verdict = evaluate_outbound(content, platform="telegram", target="123")

    assert verdict.incident_path is not None
    report_text = open(verdict.incident_path).read()
    # report lives under the profile-aware security/incidents dir
    assert str(incidents_home / "security" / "incidents") in verdict.incident_path
    # raw sentinel never written to disk
    assert HERMES_L2_FAKE_SECRET_SENTINEL not in report_text
    report = json.loads(report_text)
    assert report["simulation"] is True
    assert report["sentinel_present"] is True


# --- delivery seam --------------------------------------------------------


@pytest.mark.asyncio
async def test_delivery_disabled_default_sends_normally(monkeypatch):
    monkeypatch.delenv(SIMULATION_ENV_FLAG, raising=False)
    adapter = RecordingAdapter()
    router = DeliveryRouter(GatewayConfig(), adapters={Platform.TELEGRAM: adapter})
    target = DeliveryTarget.parse("telegram:999")

    result = await router._deliver_to_platform(
        target, f"contains {HERMES_L2_FAKE_SECRET_SENTINEL}", metadata=None
    )

    # disabled gate preserves existing behavior: adapter IS called
    assert len(adapter.calls) == 1
    assert getattr(result, "success", None) is True


@pytest.mark.asyncio
async def test_delivery_blocked_when_enabled_adapter_not_called(monkeypatch, incidents_home):
    monkeypatch.setenv(SIMULATION_ENV_FLAG, "1")
    adapter = RecordingAdapter()
    router = DeliveryRouter(GatewayConfig(), adapters={Platform.TELEGRAM: adapter})
    target = DeliveryTarget.parse("telegram:999")

    result = await router._deliver_to_platform(
        target, f"contains {HERMES_L2_FAKE_SECRET_SENTINEL}", metadata=None
    )

    # blocked before send: adapter.send never invoked
    assert adapter.calls == []
    assert result["blocked"] == "l2_secret_simulation"
    assert result["delivered"] is False
    assert HERMES_L2_FAKE_SECRET_SENTINEL not in json.dumps(result)


@pytest.mark.asyncio
async def test_delivery_enabled_clean_message_still_sends(monkeypatch, incidents_home):
    monkeypatch.setenv(SIMULATION_ENV_FLAG, "1")
    adapter = RecordingAdapter()
    router = DeliveryRouter(GatewayConfig(), adapters={Platform.TELEGRAM: adapter})
    target = DeliveryTarget.parse("telegram:999")

    result = await router._deliver_to_platform(target, "ordinary text", metadata=None)

    assert len(adapter.calls) == 1
    assert getattr(result, "success", None) is True


# --- send_message_tool seam ----------------------------------------------


def test_send_tool_blocks_before_live_send(monkeypatch, incidents_home):
    monkeypatch.setenv(SIMULATION_ENV_FLAG, "1")
    import tools.send_message_tool as smt

    called = {"sent": False}

    def _boom(*a, **k):  # would indicate a live send slipped through
        called["sent"] = True
        raise AssertionError("live send must not be reached when blocked")

    # If the guard fails to short-circuit, _run_async(_send_to_platform(...))
    # would fire; trip a sentinel instead of contacting any platform.
    monkeypatch.setattr(smt, "_send_to_platform", _boom, raising=False)

    args = {
        "action": "send",
        "target": "telegram:123",
        "message": f"exfil {HERMES_L2_FAKE_SECRET_SENTINEL}",
    }
    out = smt.send_message_tool(args)
    payload = json.loads(out)

    assert called["sent"] is False
    assert payload["blocked"] == "l2_secret_simulation"
    assert payload["delivered"] is False
    assert HERMES_L2_FAKE_SECRET_SENTINEL not in out


# --- gateway/run.py direct status send seam ------------------------------


@pytest.mark.asyncio
async def test_run_status_send_blocked_when_enabled(monkeypatch, incidents_home):
    monkeypatch.setenv(SIMULATION_ENV_FLAG, "1")
    from gateway.run import _send_or_update_status_coro

    adapter = KwargsRecordingAdapter()
    result = await _send_or_update_status_coro(
        adapter,
        "999",
        "status-key",
        f"status {HERMES_L2_FAKE_SECRET_SENTINEL}",
        metadata=None,
    )

    # blocked before any send: neither send nor send_or_update_status invoked
    assert adapter.calls == []
    assert result.success is False
    assert HERMES_L2_FAKE_SECRET_SENTINEL not in (result.error or "")


@pytest.mark.asyncio
async def test_run_status_send_disabled_default_sends(monkeypatch):
    monkeypatch.delenv(SIMULATION_ENV_FLAG, raising=False)
    from gateway.run import _send_or_update_status_coro

    adapter = KwargsRecordingAdapter()
    result = await _send_or_update_status_coro(
        adapter, "999", "status-key", f"status {HERMES_L2_FAKE_SECRET_SENTINEL}", metadata=None
    )

    # disabled gate preserves existing behavior: adapter IS called
    assert len(adapter.calls) == 1
    assert result.success is True


@pytest.mark.asyncio
async def test_run_status_send_enabled_clean_sends(monkeypatch, incidents_home):
    monkeypatch.setenv(SIMULATION_ENV_FLAG, "1")
    from gateway.run import _send_or_update_status_coro

    adapter = KwargsRecordingAdapter()
    result = await _send_or_update_status_coro(
        adapter, "999", "status-key", "ordinary status text", metadata=None
    )

    assert len(adapter.calls) == 1
    assert result.success is True


# --- gateway/stream_consumer.py direct send seam -------------------------


def _make_consumer(adapter):
    from gateway.stream_consumer import GatewayStreamConsumer

    return GatewayStreamConsumer(adapter, "999", metadata=None)


@pytest.mark.asyncio
async def test_stream_send_blocked_when_enabled_adapter_not_called(monkeypatch, incidents_home):
    monkeypatch.setenv(SIMULATION_ENV_FLAG, "1")
    adapter = KwargsRecordingAdapter()
    consumer = _make_consumer(adapter)

    result = await consumer._send_new_chunk(
        f"leak {HERMES_L2_FAKE_SECRET_SENTINEL}", reply_to_id=None
    )

    # blocked before send: adapter.send never invoked
    assert adapter.calls == []
    # _send_new_chunk swallows the failed result and returns the reply anchor
    assert result is None
    assert consumer.message_id is None


@pytest.mark.asyncio
async def test_stream_send_disabled_default_sends(monkeypatch):
    monkeypatch.delenv(SIMULATION_ENV_FLAG, raising=False)
    adapter = KwargsRecordingAdapter()
    consumer = _make_consumer(adapter)

    result = await consumer._send_new_chunk(
        f"contains {HERMES_L2_FAKE_SECRET_SENTINEL}", reply_to_id=None
    )

    # disabled gate preserves existing behavior: adapter IS called
    assert len(adapter.calls) == 1
    assert result == "sent"


@pytest.mark.asyncio
async def test_stream_send_enabled_clean_sends(monkeypatch, incidents_home):
    monkeypatch.setenv(SIMULATION_ENV_FLAG, "1")
    adapter = KwargsRecordingAdapter()
    consumer = _make_consumer(adapter)

    result = await consumer._send_new_chunk("ordinary streamed text", reply_to_id=None)

    assert len(adapter.calls) == 1
    assert result == "sent"


@pytest.mark.asyncio
async def test_stream_draft_frame_blocked_when_enabled(monkeypatch, incidents_home):
    monkeypatch.setenv(SIMULATION_ENV_FLAG, "1")
    adapter = KwargsRecordingAdapter()
    consumer = _make_consumer(adapter)
    consumer._draft_id = 1

    ok = await consumer._send_draft_frame(f"draft {HERMES_L2_FAKE_SECRET_SENTINEL}")

    # blocked: send_draft never invoked, draft transport disabled for the run
    assert adapter.draft_calls == []
    assert ok is False


# --- gateway/run.py voice transcript echo seam ---------------------------
#
# The voice-channel transcript echo sends directly to a live Discord channel
# object (channel.send), not through an adapter send seam, so it can't use
# _adapter_send_with_l2_guard. These tests prove the shared guard still gates
# that direct path: a fake recording channel asserts .send() is skipped when the
# gate is armed and the sentinel is present, and is called otherwise.


class _RecordingChannel:
    def __init__(self):
        self.sends = []

    async def send(self, content):
        self.sends.append(content)
        return SimpleNamespace(id="msg")


def _make_voice_handler(channel):
    """Bind the real _handle_voice_channel_input to a minimal fake runner/self."""
    from gateway.run import GatewayRunner

    async def _noop_handle_message(event):
        return None

    adapter = SimpleNamespace(
        _voice_text_channels={7: 555},
        _voice_sources={},
        _client=SimpleNamespace(get_channel=lambda cid: channel),
        handle_message=_noop_handle_message,
    )
    fake_self = SimpleNamespace(
        adapters={Platform.DISCORD: adapter},
        _is_user_authorized=lambda source: True,
        _is_duplicate_voice_transcript=lambda guild_id, user_id, transcript: False,
    )
    return GatewayRunner._handle_voice_channel_input, fake_self


@pytest.mark.asyncio
async def test_voice_echo_blocked_when_enabled(monkeypatch, incidents_home):
    monkeypatch.setenv(SIMULATION_ENV_FLAG, "1")
    channel = _RecordingChannel()
    handler, fake_self = _make_voice_handler(channel)

    await handler(fake_self, 7, 42, f"hey {HERMES_L2_FAKE_SECRET_SENTINEL}")

    # blocked before the direct channel.send: nothing echoed to the channel
    assert channel.sends == []


@pytest.mark.asyncio
async def test_voice_echo_disabled_default_sends(monkeypatch):
    monkeypatch.delenv(SIMULATION_ENV_FLAG, raising=False)
    channel = _RecordingChannel()
    handler, fake_self = _make_voice_handler(channel)

    await handler(fake_self, 7, 42, f"hey {HERMES_L2_FAKE_SECRET_SENTINEL}")

    # disabled gate preserves existing behavior: transcript IS echoed
    assert len(channel.sends) == 1
    assert "<@42>" in channel.sends[0]


@pytest.mark.asyncio
async def test_voice_echo_enabled_clean_sends(monkeypatch, incidents_home):
    monkeypatch.setenv(SIMULATION_ENV_FLAG, "1")
    channel = _RecordingChannel()
    handler, fake_self = _make_voice_handler(channel)

    await handler(fake_self, 7, 42, "ordinary spoken text")

    assert len(channel.sends) == 1


# --- tools/send_message_tool._send_via_adapter deepest seam ---------------
#
# The normal send_message_tool path is guarded earlier, but _send_via_adapter
# is the deepest direct adapter.send helper. Guard it too so future callers
# reaching it directly cannot bypass the chokepoint. Result shape (dict with
# success/error) is preserved.


def _patch_runner_with_adapter(monkeypatch, adapter):
    fake_runner = SimpleNamespace(adapters={Platform.TELEGRAM: adapter})
    monkeypatch.setattr(
        "gateway.run._gateway_runner_ref", lambda: fake_runner, raising=False
    )


@pytest.mark.asyncio
async def test_send_via_adapter_blocked_when_enabled(monkeypatch, incidents_home):
    monkeypatch.setenv(SIMULATION_ENV_FLAG, "1")
    import tools.send_message_tool as smt

    adapter = KwargsRecordingAdapter()
    _patch_runner_with_adapter(monkeypatch, adapter)

    result = await smt._send_via_adapter(
        Platform.TELEGRAM, None, "123", f"exfil {HERMES_L2_FAKE_SECRET_SENTINEL}"
    )

    # blocked before adapter.send: never invoked, redacted repair payload returned
    assert adapter.calls == []
    assert result["blocked"] == "l2_secret_simulation"
    assert result["delivered"] is False
    assert HERMES_L2_FAKE_SECRET_SENTINEL not in json.dumps(result)


@pytest.mark.asyncio
async def test_send_via_adapter_disabled_default_sends(monkeypatch):
    monkeypatch.delenv(SIMULATION_ENV_FLAG, raising=False)
    import tools.send_message_tool as smt

    adapter = KwargsRecordingAdapter()
    _patch_runner_with_adapter(monkeypatch, adapter)

    result = await smt._send_via_adapter(
        Platform.TELEGRAM, None, "123", f"contains {HERMES_L2_FAKE_SECRET_SENTINEL}"
    )

    # disabled gate preserves existing behavior: adapter.send IS called
    assert len(adapter.calls) == 1
    assert result["success"] is True


@pytest.mark.asyncio
async def test_send_via_adapter_enabled_clean_sends(monkeypatch, incidents_home):
    monkeypatch.setenv(SIMULATION_ENV_FLAG, "1")
    import tools.send_message_tool as smt

    adapter = KwargsRecordingAdapter()
    _patch_runner_with_adapter(monkeypatch, adapter)

    result = await smt._send_via_adapter(Platform.TELEGRAM, None, "123", "ordinary text")

    assert len(adapter.calls) == 1
    assert result["success"] is True
