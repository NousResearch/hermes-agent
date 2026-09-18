"""Acceptance: image-processing feedback line (red team, t_3e017358).

Real full-turn seam ``GatewayRunner._handle_message_with_agent``; boundary doubles only.
Config-marker injection is REAL: conftest points HERMES_HOME at tmp, genuine ``load_config()``
deep-merges the config.yaml written in ``_runner``, and ``gateway_run._hermes_home`` is patched
to the same dir for the gateway face — both faces carry ``cfg-default-marker-model``.
"""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import gateway.run as gateway_run
from agent.image_routing import ImageFeedbackStatus, begin_image_feedback, build_image_feedback_line, pop_image_feedback, record_image_described
from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionEntry, SessionSource
from hermes_cli.config import load_config

KEY = "agent:main:telegram:group:-1001:12345"
RUNTIME_MODEL = "rt-sentinel-9f31-model"
CFG_MARKER = "cfg-default-marker-model"
BODY = "The photos show a cat on a mat."
S1_LINE = "🖼️ Image handling: 2/2 attached image(s) were described by an auxiliary vision model — rt-sentinel-9f31-model received the text description(s), not the original image(s)."
S4_LINE = "🖼️ Image handling: 1 attached image(s) could not be described this turn — rt-sentinel-9f31-model received no image content."


def _source():
    return SessionSource(platform=Platform.TELEGRAM, chat_id="-1001", chat_type="group", user_id="12345")


def _plain_event(text="hello", message_id="msg-42"):
    return MessageEvent(text=text, source=_source(), message_id=message_id)


def _image_event(tmp_path, count):
    urls = []
    for i in range(count):
        p = tmp_path / f"img-{i}.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\n")
        urls.append(str(p))
    return MessageEvent(
        text="what is in these photos?", message_type=MessageType.PHOTO, source=_source(), message_id="msg-42", media_urls=urls, media_types=["image/png"] * count,
    )


def _agent_result(final_response):
    return {
        "final_response": final_response, "model": RUNTIME_MODEL,
        "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": final_response}],
        "tools": [], "history_offset": 0, "last_prompt_tokens": 0, "api_calls": 1, "failed": False,
    }


def _vision(success):
    payload = {"success": success, "analysis" if success else "error": "a photo of a cat" if success else "vision unavailable"}
    return patch("tools.vision_tools.vision_analyze_tool", new=AsyncMock(return_value=json.dumps(payload)))


def _runner(monkeypatch, tmp_path, *, spy_adapter=False):
    # Canonical full-turn harness (tests/gateway/test_gateway_silence_tokens.py).
    runner = gateway_run.GatewayRunner(GatewayConfig())
    # [auto-fix] lease release lives in the OUTER _handle_message finally; this harness calls the
    # inner handler, so disable the registry (acquire() short-circuits on None) for multi-turn tests.
    runner._turn_leases = None
    runner.adapters = {}
    runner._running_agents = {}; runner._running_agents_ts = {}
    runner._pending_messages = {}; runner._pending_approvals = {}
    runner._is_user_authorized = lambda _s: True; runner._set_session_env = lambda _c: None
    runner._handle_active_session_busy_message = AsyncMock(return_value=False); runner._session_db = MagicMock()
    runner._recover_telegram_topic_thread_id = lambda _source: None; runner._cache_session_source = lambda _key, _source: None
    runner._is_session_run_current = lambda _key, _gen: True; runner._reply_anchor_for_event = lambda _event: None
    runner._get_guild_id = lambda _event: None; runner._should_send_voice_reply = lambda *_a, **_kw: False
    runner.hooks = MagicMock(); runner.hooks.emit = AsyncMock()
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key=KEY, session_id="sess-image-feedback", created_at=datetime.now(),
        updated_at=datetime.now(), platform=Platform.TELEGRAM, chat_type="group",
    )
    runner.session_store.load_transcript.return_value = []
    runner.session_store.append_to_transcript = MagicMock(); runner.session_store.update_session = MagicMock()
    home = tmp_path / "hermes_test"  # same dir conftest autouse put in HERMES_HOME
    (home / "config.yaml").write_text(f"model:\n  default: {CFG_MARKER}\n", encoding="utf-8")
    monkeypatch.setattr(gateway_run, "_hermes_home", home)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})
    monkeypatch.setattr("agent.model_metadata.get_model_context_length", lambda *_a, **_kw: 100_000)
    if spy_adapter:
        spy = MagicMock(); spy.send = AsyncMock(return_value=True)
        runner.adapters = {Platform.TELEGRAM: spy}
    return runner


@pytest.mark.asyncio
async def test_s1_text_routed_images_prepend_feedback_line_naming_runtime_model(monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    runner._decide_image_input_mode = lambda **_: "text"
    runner._run_agent = AsyncMock(return_value=_agent_result(BODY))
    with _vision(True):
        response = await runner._handle_message_with_agent(_image_event(tmp_path, 2), _source(), KEY, 1)
    first_line = response.split("\n\n", 1)[0]
    assert first_line.startswith("🖼️ Image handling: "), f"s1p1: {response!r}"
    assert "were described by an auxiliary vision model" in first_line, f"s1p1: {response!r}"
    assert RUNTIME_MODEL in first_line, f"s1p2: {response!r}"
    assert load_config()["model"]["default"] == CFG_MARKER, "s1p2: config injection not live"
    assert CFG_MARKER not in response, f"s1p2: {response!r}"
    assert BODY in response, f"s1p3: {response!r}"
    assert response.index(BODY) > response.index("🖼️"), f"s1p3: {response!r}"
    assert first_line == S1_LINE, f"s1 design template verbatim: {first_line!r}"
    assert "\n" not in first_line, f"s1 single-line contract: {first_line!r}"


@pytest.mark.asyncio
async def test_s2_no_image_turn_is_verbatim_record_free_and_send_count_neutral(monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path, spy_adapter=True)
    runner._run_agent = AsyncMock(return_value=_agent_result("plain answer"))
    await runner._handle_message_with_agent(_plain_event(), _source(), KEY, 1)
    baseline = runner.adapters[Platform.TELEGRAM].send.await_count
    response = await runner._handle_message_with_agent(_plain_event("hello again", "msg-43"), _source(), KEY, 1)
    assert response == "plain answer", f"s2p1: {response!r}"
    assert "Image handling" not in response, f"s2p1: {response!r}"
    assert pop_image_feedback(KEY) is None, "s2p2: record survived a no-image turn"
    # s2p3 relation: per-turn send DELTAS equal (orthogonal infra cadence cancels; any
    # feedback-line extra send breaks the equality). Absolute value unfrozen.
    after = runner.adapters[Platform.TELEGRAM].send.await_count
    assert after - baseline == baseline, "s2p3: send count drifted from no-image baseline"


@pytest.mark.asyncio
async def test_s3_native_mode_image_turn_stays_verbatim(monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    runner._decide_image_input_mode = lambda **_: "native"
    runner._run_agent = AsyncMock(return_value=_agent_result("native saw the image"))
    with _vision(True):
        response = await runner._handle_message_with_agent(_image_event(tmp_path, 1), _source(), KEY, 1)
    assert response == "native saw the image", f"s3p1: {response!r}"
    assert "Image handling" not in response, f"s3p1: {response!r}"


@pytest.mark.asyncio
async def test_s4_all_descriptions_failed_prepends_failure_line(monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    runner._decide_image_input_mode = lambda **_: "text"
    runner._run_agent = AsyncMock(return_value=_agent_result("I only have your caption text."))
    with _vision(False):
        response = await runner._handle_message_with_agent(_image_event(tmp_path, 1), _source(), KEY, 1)
    first_line = response.split("\n\n", 1)[0]
    assert first_line.startswith("🖼️ Image handling: "), f"s4p1: {response!r}"
    assert "could not be described" in first_line, f"s4p1: {response!r}"
    assert RUNTIME_MODEL in response, f"s4p1: {response!r}"
    assert "timeout" not in response, f"s4p2: {response!r}"
    assert "rate limit" not in response, f"s4p2: {response!r}"
    assert first_line == S4_LINE, f"s4 design template verbatim: {first_line!r}"


@pytest.mark.asyncio
async def test_s5_image_turn_does_not_leak_into_next_no_image_turn(monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    runner._decide_image_input_mode = lambda **_: "text"
    runner._run_agent = AsyncMock(side_effect=[_agent_result("turn one answer"), _agent_result("turn two answer")])
    with _vision(True):
        await runner._handle_message_with_agent(_image_event(tmp_path, 1), _source(), KEY, 1)
    second = await runner._handle_message_with_agent(_plain_event("follow up", "msg-43"), _source(), KEY, 1)
    assert second == "turn two answer", f"s5p1: {second!r}"
    assert "Image handling" not in second, f"s5p1: {second!r}"


def test_s6_build_line_empty_status_and_blank_model_fallback():
    assert build_image_feedback_line(None, "m") == "", "s6p1: None status"
    assert build_image_feedback_line(ImageFeedbackStatus(0, 0), "m") == "", "s6p1: total<=0 status"
    blank = build_image_feedback_line(ImageFeedbackStatus(1, 1), "   ")
    assert "the main model" in blank, f"s6p2: {blank!r}"
    assert "  " not in blank, f"s6p2 empty placeholder: {blank!r}"
    failed = build_image_feedback_line(ImageFeedbackStatus(2, 0), "")
    assert "the main model" in failed, f"s6p2: {failed!r}"
    assert "  " not in failed, f"s6p2 empty placeholder: {failed!r}"


def test_s6_registry_defense_arms_and_pop_idempotency():
    for falsy in ("", None):
        begin_image_feedback(falsy, 3)
        record_image_described(falsy)
        assert pop_image_feedback(falsy) is None, f"s6p3: key={falsy!r}"
    begin_image_feedback("s6p3-valid", 0)
    assert pop_image_feedback("s6p3-valid") is None, "s6p3: total<=0 defense arm"
    begin_image_feedback("s6p4-key", 2)
    record_image_described("s6p4-key"); record_image_described("s6p4-key")
    first = pop_image_feedback("s6p4-key")
    assert first is not None and first.described_images == first.total_images == 2, f"s6p4 counts: {first}"
    assert pop_image_feedback("s6p4-key") is None, "s6p4: idempotent second pop"
