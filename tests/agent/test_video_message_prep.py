"""Behavior-contract tests for native video handling across message prep, recovery, persistence, and traces."""

from __future__ import annotations

from unittest.mock import patch

from agent.client_lifecycle import ClientLifecycleMixin
from agent.tool_dispatch_helpers import _trajectory_normalize_msg
from agent.trace_upload import _part_to_block
from agent.vision_message_prep import VisionMessagePrepMixin
from session_rules.turns import _durable_content


class _StubAgent(VisionMessagePrepMixin):
    def __init__(self, provider: str, model: str) -> None:
        self.provider = provider
        self.model = model
        self._anthropic_image_fallback_cache: dict[str, str] = {}
        self._no_list_tool_content_models: set[tuple[str, str]] = set()


def _video_envelope() -> dict:
    return {
        "_multimodal": True,
        "content": [
            {"type": "text", "text": "Video loaded into your context."},
            {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}},
        ],
        "text_summary": "Video attached natively for the main model (1.0 MB).",
        "meta": {"native_video": True},
    }


def test_tool_result_video_passes_for_video_capable_gemini_and_degrades_for_image_only_claude():
    gemini_agent = _StubAgent("gemini", "gemini-3.8-flash")
    claude_agent = _StubAgent("anthropic", "claude-sonnet-4-6")
    envelope = _video_envelope()

    with (
        patch.object(gemini_agent, "_model_supports_video", return_value=True),
        patch.object(gemini_agent, "_provider_supports_vision_tool_messages", return_value=True),
    ):
        gemini_content = gemini_agent._tool_result_content_for_active_model("video_analyze", envelope)
    assert isinstance(gemini_content, list)
    assert gemini_content[1]["type"] == "video_url"

    with (
        patch.object(claude_agent, "_model_supports_vision", return_value=True),
        patch.object(claude_agent, "_model_supports_video", return_value=False),
        patch.object(claude_agent, "_provider_supports_vision_tool_messages", return_value=True),
    ):
        claude_content = claude_agent._tool_result_content_for_active_model("video_analyze", envelope)
    assert claude_content == envelope["text_summary"]


def test_mid_session_fallback_from_gemini_to_claude_strips_video_on_send_path_only():
    """Switching mid-session to an image-only model strips historical video_url on the outbound copy without mutating canonical history."""
    claude_agent = _StubAgent("anthropic", "claude-sonnet-4-6")
    canonical = [
        {"role": "user", "content": "Analyze this clip"},
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": [
                {"type": "text", "text": "Video loaded natively"},
                {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}},
            ],
        },
    ]

    with (
        patch.object(claude_agent, "_model_supports_vision", return_value=True),
        patch.object(claude_agent, "_model_supports_video", return_value=False),
    ):
        outbound = claude_agent._prepare_messages_for_non_vision_model(canonical)

    assert outbound is not canonical
    assert canonical[1]["content"][1]["type"] == "video_url"
    # Stripped tool-role message is flattened to a string (safe for strict OpenAI-compat providers).
    assert isinstance(outbound[1]["content"], str)
    assert "video" in outbound[1]["content"].lower()
    assert "Video loaded natively" in outbound[1]["content"]


def test_video_token_estimation_compaction_bedrock_and_sanitization():
    """Base64 video parts never inflate token estimates, are evicted on compaction, sanitize cleanly, and stay safe on Bedrock & traces."""
    from agent.bedrock_adapter import _convert_content_to_converse
    from agent.chat_completion_helpers import estimate_request_context_tokens
    from agent.context_compressor import _strip_historical_media
    from agent.message_sanitization import _messages_carry_images, _strip_images_from_messages
    from agent.model_metadata import estimate_messages_tokens_rough
    from agent.trace_upload import _tool_result_message

    huge_b64 = "A" * (2 * 1024 * 1024)
    video_msg = {
        "role": "tool",
        "tool_call_id": "call_v1",
        "content": [
            {"type": "text", "text": "Video loaded natively"},
            {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{huge_b64}"}},
        ],
    }

    # Bounded token estimates (never len(base64) // 4 ~= 500,000 tokens)
    assert estimate_messages_tokens_rough([video_msg]) < 10_000
    assert estimate_request_context_tokens({"messages": [video_msg]}) < 10_000

    # Context compressor strips older video tool results with a [video ...] marker
    history = [
        {"role": "user", "content": "first"},
        video_msg,
        {"role": "user", "content": "second"},
        {
            "role": "tool",
            "tool_call_id": "call_v2",
            "content": [
                {"type": "text", "text": "Newer video"},
                {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,BBBB"}},
            ],
        },
    ]
    stripped_history = _strip_historical_media(history)
    assert any(
        isinstance(p, dict) and "video" in str(p.get("text", "")).lower() and "screenshot" not in str(p.get("text", "")).lower()
        for p in stripped_history[1]["content"]
    )

    # Bedrock Converse placeholder (never drops silently or dumps base64)
    converse_blocks = _convert_content_to_converse(video_msg["content"])
    assert any("video" in b.get("text", "").lower() for b in converse_blocks)
    assert all(huge_b64 not in str(b) for b in converse_blocks)

    # Corrupt-image stripping preserves healthy video; broad all-media strip removes video cleanly
    from agent.message_sanitization import _strip_all_media_from_messages

    san_copy = [dict(video_msg, content=list(video_msg["content"]))]
    assert _messages_carry_images(san_copy) is True
    assert _strip_images_from_messages(san_copy) is False
    assert _strip_all_media_from_messages(san_copy) is True
    assert all(p.get("type") != "video_url" for p in san_copy[0]["content"] if isinstance(p, dict))

    # Trace upload _tool_result_message never dumps base64 for in-memory list content
    trace_msg = _tool_result_message(video_msg, "gemini-3.8-flash", redact=False)
    assert huge_b64 not in str(trace_msg)
    assert {"type": "text", "text": "[video omitted]"} in trace_msg["content"][0]["content"]


def test_evict_stale_outbound_tool_videos_and_bedrock_kwargs_and_auxiliary_mode_preservation(tmp_path, monkeypatch):
    """Outbound eviction labels _multimodal video envelopes as [video removed], Bedrock kwargs strips videos, and video_input_mode=auxiliary preserves historical videos on Gemini 3+."""
    from unittest.mock import MagicMock

    from agent import models_dev
    from agent.chat_completion_helpers import _build_bedrock_kwargs
    from agent.context_compressor import _strip_images_from_tool_msg, evict_stale_outbound_tool_images
    from agent.runtime_scope import bind_profile_scope

    # 1. _strip_images_from_tool_msg labels _multimodal video envelopes as [video removed], not [screenshot removed]
    vid_env_msg = {"role": "tool", "tool_call_id": "call_env", "content": _video_envelope()}
    stripped_env = _strip_images_from_tool_msg(vid_env_msg)
    assert stripped_env is not None
    assert stripped_env["content"].startswith("[video removed]")
    assert "[screenshot" not in stripped_env["content"]

    # 2. evict_stale_outbound_tool_images retires older 15 MB videos under the 24 MB budget
    blob = "A" * (15 * 1024 * 1024)
    api_msgs = [
        {"role": "tool", "tool_call_id": "v1", "content": [{"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{blob}"}}]},
        {"role": "tool", "tool_call_id": "v2", "content": [{"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{blob}"}}]},
    ]
    pruned = evict_stale_outbound_tool_images(api_msgs)
    assert pruned == 1
    assert api_msgs[0]["content"] == [{"type": "text", "text": "[video removed to save context]"}]

    # 3. _build_bedrock_kwargs strips video_url & "video" parts via _prepare_messages_for_non_vision_model
    bedrock_agent = _StubAgent("bedrock", "us.anthropic.claude-sonnet-4-6")
    bedrock_agent.max_tokens = 1024
    captured: dict = {}
    fake_transport = MagicMock()
    fake_transport.build_kwargs.side_effect = lambda **kw: captured.update(kw) or kw
    bedrock_agent._get_transport = lambda: fake_transport
    raw_msgs = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "video", "video": {"url": "data:video/mp4;base64,AAAA"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": [{"type": "text", "text": "t"}, {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}}]},
    ]
    with (
        patch.object(bedrock_agent, "_model_supports_vision", return_value=True),
        patch.object(bedrock_agent, "_model_supports_video", return_value=False),
    ):
        _build_bedrock_kwargs(bedrock_agent, raw_msgs, [], None)
    assert isinstance(captured["messages"][1]["content"], str)
    assert all(p.get("type") != "video" for p in captured["messages"][0]["content"])

    # 4. _model_supports_video stays True on Gemini 3.8 even when agent.video_input_mode is auxiliary
    home = tmp_path / "aux_home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "model:\n  provider: gemini\n  default: gemini-3.8-flash\nagent:\n  video_input_mode: auxiliary\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        models_dev,
        "_models_dev_cache",
        {"google": {"models": {"gemini-3.8-flash": {"modalities": {"input": ["text", "image", "video"], "output": ["text"]}, "tool_call": True}}}},
    )
    monkeypatch.setattr(models_dev, "_models_dev_cache_time", float("inf"))
    gem_agent = _StubAgent("gemini", "gemini-3.8-flash")
    with bind_profile_scope(home):
        assert gem_agent._model_supports_video() is True




def test_413_recovery_strips_video_tool_parts_without_blacklisting_model():
    gemini_agent = _StubAgent("gemini", "gemini-3.8-flash")
    api_messages = [
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": [
                {"type": "text", "text": "Video loaded natively"},
                {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}},
            ],
        }
    ]
    assert ClientLifecycleMixin._api_kwargs_have_image_parts({"messages": api_messages}) is True
    changed = gemini_agent._try_strip_image_parts_from_tool_messages(api_messages, remember_model=False)
    assert changed is True
    assert api_messages[0]["content"] == "Video loaded natively"
    assert ("gemini", "gemini-3.8-flash") not in gemini_agent._no_list_tool_content_models


def test_persistence_trajectory_and_trace_upload_never_leak_base64_video_or_screenshot_marker():
    content = [
        {"type": "text", "text": "Look at this"},
        {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,HUGEBASE64"}},
    ]

    durable = _durable_content(content)
    assert durable == "Look at this\n[video]"
    assert "[screenshot]" not in durable
    assert "HUGEBASE64" not in durable

    traj_msg = _trajectory_normalize_msg({"role": "tool", "content": content})
    assert traj_msg["content"] == [
        {"type": "text", "text": "Look at this"},
        {"type": "text", "text": "[video]"},
    ]

    trace_block = _part_to_block(content[1], redact=False)
    assert trace_block == {"type": "text", "text": "[video omitted]"}


def test_video_hardening_no_base64_leaks_ema_isolation_and_rejection_isolation():
    """Verify Anthropic/Bedrock no-leak guards, EMA isolation, video rejection isolation, api_mode gating, and models_dev sparse overrides."""
    import base64
    import struct
    from types import SimpleNamespace
    from agent.anthropic_message_convert import _tool_result_content
    from agent.bedrock_adapter import convert_messages_to_converse
    from agent.context_compressor import evict_stale_outbound_tool_images
    from agent.image_routing import supports_video_in_tool_results
    from agent.message_sanitization import record_media_rejection, strip_images_for_rejecting_model
    from agent.model_metadata import _count_image_tokens, estimate_messages_tokens_rough
    from agent.models_dev import _merge_catalog_entry_with_override, get_model_capabilities

    huge_b64 = "V" * 16384
    video_list_tool = {
        "role": "tool",
        "tool_call_id": "call_vid_leak",
        "content": [
            {"type": "text", "text": "Clip attached"},
            {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{huge_b64}"}},
        ],
    }

    # 1. Anthropic list-typed tool_result never falls through to json.dumps(content)
    ant_out = _tool_result_content(video_list_tool)
    assert huge_b64 not in str(ant_out)
    assert "Clip attached" in str(ant_out)

    # 2. Bedrock Converse tool_result (both list and _multimodal dict) never leaks base64
    converse_msgs, _ = convert_messages_to_converse([video_list_tool])
    assert huge_b64 not in str(converse_msgs)
    mm_tool = {"role": "tool", "tool_call_id": "call_mm", "content": _video_envelope()}
    converse_mm, _ = convert_messages_to_converse([mm_tool])
    assert "AAAA" not in str(converse_mm)

    # 3. _count_image_tokens returns 0 for video (never poisons image_token_costs.json EMA);
    #    string api_content sidecar zeroes non-wire video estimate
    assert _count_image_tokens(video_list_tool, 1000) == 0
    assert estimate_messages_tokens_rough([video_list_tool]) >= 1_500
    sidecar_msg = {
        "role": "user",
        "content": [{"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{huge_b64}"}}],
        "api_content": "text replacement only",
    }
    assert estimate_messages_tokens_rough([sidecar_msg]) < 100

    # 4. Video rejection (explicit & generic) never poisons _image_rejecting_models
    agent = SimpleNamespace(
        provider="gemini", model="gemini-3.8-flash", _image_rejecting_models=set(),
    )
    mixed = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,BBBB"}},
        ]},
    ]
    assert record_media_rejection(agent, "unexpected item type in content", mixed) == "video"
    assert ("gemini", "gemini-3.8-flash") not in agent._image_rejecting_models
    assert strip_images_for_rejecting_model(agent, mixed) is True
    assert [p["type"] for p in mixed[0]["content"]] == ["image_url"]

    # 5. api_mode gating blocks non-native transports even if provider/model look Gemini-like
    assert supports_video_in_tool_results("gemini", "gemini-3.8-flash", {}, api_mode="anthropic_messages") is False
    assert supports_video_in_tool_results("gemini", "gemini-3.8-flash", {}, api_mode="bedrock_converse") is False
    assert supports_video_in_tool_results("gemini", "gemini-3.8-flash", {}, api_mode="codex_responses") is False
    assert supports_video_in_tool_results("solstice", "gemini-3.8-flash", {}) is True
    assert supports_video_in_tool_results(
        "gemini", "gemini-3.8-flash", {}, base_url="https://generativelanguage.googleapis.com/v1",
    ) is True

    # 6. YouTube video URL alongside small PNGs never falsely disables many_image_safe
    png_100 = b"\x89PNG\r\n\x1a\n" + struct.pack(">I4sII", 13, b"IHDR", 100, 100) + b"\x08\x02\x00\x00\x00" + b"\x00" * 4
    png_url = "data:image/png;base64," + base64.b64encode(png_100).decode("ascii")
    msgs = [
        {"role": "tool", "tool_call_id": f"img_{i}", "content": [{"type": "image_url", "image_url": {"url": png_url}}]}
        for i in range(21)
    ]
    msgs.append({
        "role": "tool", "tool_call_id": "yt_1",
        "content": [{"type": "video_url", "video_url": {"url": "https://www.youtube.com/watch?v=jNQXAC9IVRw"}}],
    })
    assert evict_stale_outbound_tool_images(msgs) == 0

    # 7. Catalog-miss supports_video override preserves supports_vision=None and strict bool coercion
    merged = _merge_catalog_entry_with_override({}, {"supports_video": True})
    assert "modalities" not in merged
    assert merged["_modality_overrides"] == {"video": True}
    from unittest.mock import patch

    overrides = {
        "custom-unknown-provider": {
            "custom-vid-model": {"supports_video": True},
            "custom-vid-model-2": {"supports_vision": "false"},
        }
    }
    with patch("agent.models_dev._load_model_overrides", return_value=overrides), patch(
        "agent.models_dev.fetch_models_dev", return_value={}
    ):
        caps = get_model_capabilities("custom-unknown-provider", "custom-vid-model", allow_network=False)
        caps_false = get_model_capabilities("custom-unknown-provider", "custom-vid-model-2", allow_network=False)
    assert caps is not None
    assert caps.supports_video is True
    assert caps.supports_vision is None
    assert caps_false is not None
    assert caps_false.supports_vision is False
