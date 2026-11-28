"""Contract tests for native Audio & Document (PDF/Office) support across agent/wire adapters."""
from __future__ import annotations

import base64
import importlib
import sys
from unittest.mock import patch

from agent.anthropic_message_convert import (
    _convert_content_part_to_anthropic,
    convert_messages_to_anthropic,
)
from agent.bedrock_adapter import convert_messages_to_converse
from agent.codex_responses_adapter import _chat_content_to_responses_parts
from agent.context_compressor import (
    _strip_images_from_tool_msg,
    evict_stale_outbound_tool_images,
)
from plugins.model_providers.google.native_adapter import (
    _build_gemini_contents,
    _inline_data_part,
    _translate_tool_result_parts_to_gemini,
)
from agent.image_routing import (
    decide_audio_input_mode,
    decide_pdf_input_mode,
    supports_audio_in_tool_results,
    supports_pdf_in_tool_results,
)
from agent.message_sanitization import record_media_rejection, strip_images_for_rejecting_model
from agent.model_metadata import _count_image_tokens, _estimate_message_tokens_without_images
from agent.models_dev import ModelCapabilities, _merge_catalog_entry_with_override
from agent.trace_upload import _part_to_block
from agent.vision_message_prep import VisionMessagePrepMixin
from providers import get_provider_profile
from session_rules.turns import _durable_content, is_content_blank

_TINY_PDF = (
    b"%PDF-1.4\n1 0 obj << /Type /Catalog >> endobj\ntrailer << /Root 1 0 R >>\n%%EOF\n"
)
_TINY_PDF_B64 = base64.b64encode(_TINY_PDF).decode("ascii")
_TINY_WAV_B64 = base64.b64encode(b"RIFF\x24\x00\x00\x00WAVEfmt ").decode("ascii")
# Valid 1x1 PNG header so image_eviction_policy.images_all_within_many_image_limit reads dimensions (1x1 <= 2000).
_TINY_1X1_PNG_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


class _DummyAgent(VisionMessagePrepMixin):
    def __init__(
        self,
        *,
        vision=False,
        video=False,
        audio=False,
        doc=False,
        provider="openai",
        model="gpt-4o",
        api_mode="chat_completions",
    ):
        self._v = vision
        self._vid = video
        self._aud = audio
        self._doc = doc
        self.provider = provider
        self.model = model
        self.api_mode = api_mode
        self.base_url = ""
        self.requested_provider = provider
        self._no_list_tool_content_models = set()
        self._image_rejecting_models = set()

    def _model_supports_vision(self) -> bool:
        return self._v

    def _model_supports_video(self) -> bool:
        return self._vid

    def _model_supports_audio(self) -> bool:
        return self._aud

    def _model_supports_document(self) -> bool:
        return self._doc


def test_session_rules_audio_and_document_persistence_and_non_blank():
    audio_msg = [{"type": "input_audio", "input_audio": {"data": _TINY_WAV_B64, "format": "wav"}}]
    doc_msg = [{"type": "document", "mime_type": "application/pdf", "data": _TINY_PDF_B64, "filename": "spec.pdf"}]
    assert is_content_blank(audio_msg) is False
    assert is_content_blank(doc_msg) is False
    assert _durable_content(audio_msg) == "[audio]"
    assert _durable_content(doc_msg) == "[document: spec.pdf]"


def test_models_dev_modality_overrides_generalized():
    merged = _merge_catalog_entry_with_override(
        {"modalities": {"input": ["text", "image"]}},
        {"supports_audio": True, "supports_pdf": "yes", "supports_video": True},
    )
    assert set(merged["modalities"]["input"]) == {"text", "image", "audio", "pdf", "video"}
    caps = ModelCapabilities(input_modalities=frozenset({"text", "audio", "pdf"}))
    assert caps.supports_audio is True
    assert caps.supports_pdf is True
    assert caps.supports_modality("video") is False


def test_image_routing_audio_and_pdf_modes_and_solstice_auto():
    cfg_native = {"agent": {"audio_input_mode": "native", "pdf_input_mode": "native"}}
    cfg_text = {"agent": {"audio_input_mode": "text", "pdf_input_mode": "auxiliary"}}
    assert decide_audio_input_mode("gemini", "gemini-4-neon", cfg_native, allow_network=False) == "native"
    assert decide_pdf_input_mode("gemini", "gemini-4-neon", cfg_native, allow_network=False) == "native"
    assert decide_audio_input_mode("gemini", "gemini-4-neon", cfg_text, allow_network=False) == "auxiliary"
    assert decide_pdf_input_mode("gemini", "gemini-4-neon", cfg_text, allow_network=False) == "text"
    assert supports_audio_in_tool_results("gemini", "gemini-3.5-flash", {}) is True
    assert supports_pdf_in_tool_results("gemini", "gemini-3.5-flash", {}) is True
    assert supports_pdf_in_tool_results("solstice", "gemini-4-neon", {}) is True
    assert supports_pdf_in_tool_results("gemini", "gemini-2.5-pro", {}) is False

    # Empty config {} -> default "auto" mode activates native audio & PDF on solstice/gemini-4-neon
    assert decide_audio_input_mode("solstice", "gemini-4-neon", {}, allow_network=False) == "native"
    assert decide_pdf_input_mode("solstice", "gemini-4-neon", {}, allow_network=False) == "native"
    override_cfg = {"model_overrides": {"solstice": {"gemini-4-neon": {"supports_pdf": False, "supports_audio": False}}}}
    assert decide_audio_input_mode("solstice", "gemini-4-neon", override_cfg, allow_network=False) == "auxiliary"
    assert decide_pdf_input_mode("solstice", "gemini-4-neon", override_cfg, allow_network=False) == "text"


def test_gemini_native_adapter_audio_pdf_x_wav_and_empty_turn_alternation():
    # OpenAI input_audio raw base64 + format
    assert _inline_data_part({"type": "input_audio", "input_audio": {"data": _TINY_WAV_B64, "format": "wav"}}) == {
        "inlineData": {"mimeType": "audio/wav", "data": _TINY_WAV_B64}
    }
    # audio/x-wav normalized to audio/wav on native Gemini generateContent too
    assert _inline_data_part({"type": "audio", "mime_type": "audio/x-wav", "data": _TINY_WAV_B64}) == {
        "inlineData": {"mimeType": "audio/wav", "data": _TINY_WAV_B64}
    }
    assert _inline_data_part({"type": "audio_url", "audio_url": {"url": f"data:audio/x-wav;base64,{_TINY_WAV_B64}"}}) == {
        "inlineData": {"mimeType": "audio/wav", "data": _TINY_WAV_B64}
    }
    # Direct document and OpenAI file shapes
    assert _inline_data_part({"type": "document", "mime_type": "application/pdf", "data": _TINY_PDF_B64}) == {
        "inlineData": {"mimeType": "application/pdf", "data": _TINY_PDF_B64}
    }
    assert _inline_data_part({"type": "file", "file": {"file_data": f"data:application/pdf;base64,{_TINY_PDF_B64}"}}) == {
        "inlineData": {"mimeType": "application/pdf", "data": _TINY_PDF_B64}
    }
    assert _inline_data_part({"type": "document", "mime_type": "application/x-msdownload", "data": _TINY_PDF_B64}) is None

    # Gemini 3+ tool result preserves inlineData; Gemini 2.x drops it
    tool_msg = {
        "role": "tool",
        "tool_call_id": "c1",
        "name": "read_file",
        "content": [{"type": "text", "text": "Attached PDF"}, {"type": "document", "mime_type": "application/pdf", "data": _TINY_PDF_B64}],
    }
    assert "parts" in _translate_tool_result_parts_to_gemini(tool_msg, {"c1": "read_file"}, is_gemini3=True)[0]["functionResponse"]
    assert "parts" not in _translate_tool_result_parts_to_gemini(tool_msg, {"c1": "read_file"}, is_gemini3=False)[0]["functionResponse"]

    # Empty-turn alternation safety: a user turn whose only media parts fail validation emits a fallback text part
    contents, _ = _build_gemini_contents([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": [{"type": "document", "mime_type": "application/x-msdownload", "data": "!!!"}]},
    ])
    assert [c["role"] for c in contents] == ["user", "model", "user"]
    assert contents[-1]["parts"] == [{"text": "[media omitted: invalid payload]"}]


def test_interactions_audio_document_x_wav_and_empty_turn_alternation():
    profile = get_provider_profile("gemini")
    assert profile is not None
    pkg = sys.modules[type(profile).__module__].__package__
    interactions = importlib.import_module(f"{pkg}.interactions")

    assert interactions._audio_content(
        {"type": "input_audio", "input_audio": {"data": _TINY_WAV_B64, "format": "mp3"}}
    ) == {"type": "audio", "mime_type": "audio/mpeg", "data": _TINY_WAV_B64}
    assert interactions._audio_content(
        {"type": "audio", "mime_type": "audio/x-wav", "data": _TINY_WAV_B64}
    ) == {"type": "audio", "mime_type": "audio/wav", "data": _TINY_WAV_B64}
    docx_mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    assert interactions._document_content(
        {"type": "document", "mime_type": docx_mime, "data": _TINY_PDF_B64}
    ) == {"type": "document", "mime_type": docx_mime, "data": _TINY_PDF_B64}
    assert interactions._document_content(
        {"type": "document", "mime_type": "application/octet-stream", "data": _TINY_PDF_B64}
    ) is None

    step = interactions._function_result(
        {
            "role": "tool",
            "tool_call_id": "c1",
            "name": "read_file",
            "content": [
                {"type": "text", "text": "ok"},
                {"type": "document", "mime_type": "application/pdf", "data": _TINY_PDF_B64},
                {"type": "input_audio", "input_audio": {"data": _TINY_WAV_B64, "format": "wav"}},
            ],
        },
        {"c1": "read_file"},
    )
    assert [b["type"] for b in step["result"]] == ["text", "document", "audio"]

    steps, _ = interactions.build_input_steps([
        {"role": "user", "content": [{"type": "document", "mime_type": "application/octet-stream", "data": "bad"}]},
    ])
    assert steps == [{"type": "user_input", "content": [{"type": "text", "text": "[media omitted: invalid payload]"}]}]


def test_cross_provider_stripping_anthropic_codex_bedrock_and_user_vs_tool_gating():
    # DeepSeek (non-multimodal) strips tool-result PDFs and flattens role: "tool"
    ds = _DummyAgent(vision=True, video=False, audio=False, doc=False, provider="deepseek", model="deepseek-chat")
    env = {
        "_multimodal": True,
        "content": [
            {"type": "text", "text": "Attached PDF"},
            {"type": "document", "mime_type": "application/pdf", "data": _TINY_PDF_B64},
        ],
        "text_summary": "PDF summary fallback",
    }
    assert ds._tool_result_content_for_active_model("read_file", env) == "PDF summary fallback"
    msgs = [
        {
            "role": "tool",
            "tool_call_id": "t1",
            "content": [
                {"type": "text", "text": "Read PDF"},
                {"type": "document", "mime_type": "application/pdf", "data": _TINY_PDF_B64},
            ],
        }
    ]
    out = ds._prepare_messages_for_non_vision_model(msgs)
    assert isinstance(out[0]["content"], str) and "omitted" in out[0]["content"].lower()
    assert isinstance(msgs[0]["content"], list)

    # Anthropic user-turn PDF pass-through vs tool-result PDF downgrade (real VisionMessagePrepMixin methods)
    class _AnthropicAgent(VisionMessagePrepMixin):
        provider = "anthropic"
        model = "claude-sonnet-4-5"
        api_mode = "anthropic_messages"
        base_url = ""
        requested_provider = "anthropic"
        _no_list_tool_content_models = set()

    ant = _AnthropicAgent()
    with patch("hermes_cli.config.load_config", return_value={}):
        assert ant._model_supports_document() is True
        assert ant._model_supports_document_in_tool_results() is False
        history = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Summarize"},
                    {"type": "document", "mime_type": "application/pdf", "data": _TINY_PDF_B64},
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "t1",
                "content": [
                    {"type": "text", "text": "From tool"},
                    {"type": "document", "mime_type": "application/pdf", "data": _TINY_PDF_B64},
                ],
            },
        ]
        prepared = ant._prepare_messages_for_non_vision_model(history)
        assert prepared[0]["content"][1]["type"] == "document"
        assert isinstance(prepared[1]["content"], str) and "omitted" in prepared[1]["content"].lower()
        _, ant_msgs = convert_messages_to_anthropic([prepared[0]])
        assert ant_msgs[0]["content"][1]["type"] == "document"
        assert ant_msgs[0]["content"][1]["source"]["media_type"] == "application/pdf"

    # Gemini 2.5 Pro with models.dev audio+pdf support: keeps user-turn audio/PDF, downgrades tool results
    class _Gemini25Agent(VisionMessagePrepMixin):
        provider = "gemini"
        model = "gemini-2.5-pro"
        api_mode = "chat_completions"
        base_url = ""
        requested_provider = "gemini"
        _no_list_tool_content_models = set()

    g25 = _Gemini25Agent()
    caps_25 = ModelCapabilities(supports_vision=True, input_modalities=frozenset({"text", "image", "audio", "pdf"}))
    with (
        patch("hermes_cli.config.load_config", return_value={}),
        patch("agent.models_dev.get_model_capabilities", return_value=caps_25),
    ):
        assert g25._model_supports_audio() is True
        assert g25._model_supports_document() is True
        assert g25._model_supports_audio_in_tool_results() is False
        assert g25._model_supports_document_in_tool_results() is False

    # Anthropic strips whitespace in PDF base64 and never passes raw input_audio
    assert _convert_content_part_to_anthropic({"type": "input_audio", "input_audio": {"data": _TINY_WAV_B64, "format": "wav"}}) == {
        "type": "text", "text": "[audio omitted — unsupported by Anthropic]"
    }
    pdf_blk = _convert_content_part_to_anthropic(
        {"type": "document", "mime_type": "application/pdf", "data": f"{_TINY_PDF_B64[:8]}\n{_TINY_PDF_B64[8:]}"}
    )
    assert pdf_blk["type"] == "document" and "\n" not in pdf_blk["source"]["data"]

    # Codex Responses converts audio/document to explicit omission notes instead of silently dropping
    codex_parts = _chat_content_to_responses_parts([
        {"type": "input_audio", "input_audio": {"data": _TINY_WAV_B64, "format": "wav"}},
        {"type": "document", "mime_type": "application/pdf", "data": _TINY_PDF_B64},
    ])
    assert codex_parts == [
        {"type": "input_text", "text": "[audio omitted — unsupported by Codex Responses]"},
        {"type": "input_text", "text": "[document omitted — unsupported by Codex Responses]"},
    ]

    # Bedrock role: "tool" never dumps raw base64 JSON
    _, converse = convert_messages_to_converse(msgs)
    tool_text = converse[0]["content"][0]["toolResult"]["content"][0]["text"]
    assert _TINY_PDF_B64 not in tool_text and "[document]" in tool_text


def test_per_modality_rejection_never_poisons_vision():
    agent = _DummyAgent(vision=True, video=False, audio=False, doc=True, provider="custom", model="m1")
    pdf_only_msgs = [
        {"role": "user", "content": [{"type": "text", "text": "Read"}, {"type": "document", "mime_type": "application/pdf", "data": _TINY_PDF_B64}]}
    ]
    # Generic "unexpected item type in content" on a PDF-only message must mark "document", NEVER _image_rejecting_models
    assert record_media_rejection(agent, "unexpected item type in content", pdf_only_msgs) == "document"
    assert agent._image_rejecting_models == set()

    # Audio-specific rejection marks "audio" only
    audio_msgs = [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": _TINY_WAV_B64, "format": "wav"}}]}]
    assert record_media_rejection(agent, "audio input is not supported", audio_msgs) == "audio"
    assert agent._image_rejecting_models == set()

    # Subsequent mixed message keeps images intact while stripping rejected PDF/audio
    mixed = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "check"},
            {"type": "image_url", "image_url": {"url": _TINY_1X1_PNG_URL}},
            {"type": "document", "mime_type": "application/pdf", "data": _TINY_PDF_B64},
            {"type": "input_audio", "input_audio": {"data": _TINY_WAV_B64, "format": "wav"}},
        ],
    }]
    assert strip_images_for_rejecting_model(agent, mixed) is True
    assert [p["type"] for p in mixed[0]["content"]] == ["text", "image_url"]


def test_token_estimation_traces_and_many_image_safe_eviction():
    big_b64 = "A" * 400_000
    msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Check this"},
            {"type": "document", "mime_type": "application/pdf", "data": big_b64, "filename": "huge.pdf"},
        ],
    }
    # Stripped in wire shadow (not 100K fake tokens) + conservative non-EMA floor (>= 1500)
    est = _estimate_message_tokens_without_images(msg)
    assert 1_500 <= est < 5_000
    # _count_image_tokens stays strictly image/video so image_token_cost EMA is never poisoned
    assert _count_image_tokens(msg, 1) == 0

    assert _part_to_block({"type": "document", "data": big_b64}, False) == {"type": "text", "text": "[document omitted]"}
    assert _part_to_block({"type": "input_audio", "data": big_b64}, False) == {"type": "text", "text": "[audio omitted]"}

    stripped = _strip_images_from_tool_msg({
        "role": "tool",
        "tool_call_id": "t1",
        "content": {
            "_multimodal": True,
            "meta": {"native_document": True},
            "content": [{"type": "document", "mime_type": "application/pdf", "data": _TINY_PDF_B64}],
            "text_summary": "Attached PDF",
        },
    })
    assert stripped is not None and "[document removed]" in stripped["content"]

    # >20 small valid PNGs + bare-base64 PDF must keep many_image_safe=True (cap 100, so 21 images are NOT evicted)
    api_messages = [
        {
            "role": "user",
            "content": [{"type": "document", "mime_type": "application/pdf", "data": _TINY_PDF_B64}],
        },
        *[
            {
                "role": "tool",
                "tool_call_id": f"c{i}",
                "content": [{"type": "image_url", "image_url": {"url": _TINY_1X1_PNG_URL}}],
            }
            for i in range(21)
        ],
    ]
    assert evict_stale_outbound_tool_images(api_messages) == 0


def test_bedrock_user_turn_omission_anthropic_gemini_parity_and_user_turn_helpers():
    from agent.image_routing import supports_audio_in_user_turns, supports_pdf_in_user_turns

    # User-turn helpers allow Anthropic (PDF) and Gemini 2.5 (audio & PDF) while tool-result gating stays Gemini 3+
    assert supports_pdf_in_user_turns("anthropic", "claude-sonnet-4-5", {}, api_mode="anthropic_messages") is True
    assert supports_pdf_in_tool_results("anthropic", "claude-sonnet-4-5", {}) is False
    assert supports_pdf_in_user_turns("gemini", "gemini-2.5-pro", {}) is True
    assert supports_audio_in_user_turns("gemini", "gemini-2.5-pro", {}) is True
    assert supports_audio_in_tool_results("gemini", "gemini-2.5-pro", {}) is False

    # Bedrock Converse user turns emit explicit omission text for audio/document (never silently dropped to "(empty)")
    _, converse_user = convert_messages_to_converse([{
        "role": "user",
        "content": [
            {"type": "input_audio", "input_audio": {"data": _TINY_WAV_B64, "format": "wav"}},
            {"type": "document", "mime_type": "application/pdf", "data": _TINY_PDF_B64},
        ],
    }])
    converse_texts = [b["text"] for b in converse_user[0]["content"]]
    assert any("audio" in t.lower() and "omitted" in t.lower() for t in converse_texts)
    assert any("document" in t.lower() and "omitted" in t.lower() for t in converse_texts)

    # Bedrock Converse VisionMessagePrep strips user-turn PDFs even if models.dev advertises pdf support
    class _BedrockConverseAgent(VisionMessagePrepMixin):
        provider = "bedrock"
        model = "anthropic.claude-sonnet-4-5"
        api_mode = "bedrock_converse"
        base_url = ""
        requested_provider = "bedrock"
        _no_list_tool_content_models = set()

    bc = _BedrockConverseAgent()
    with patch("hermes_cli.config.load_config", return_value={}):
        assert bc._model_supports_document() is False
        assert bc._model_supports_audio() is False

    # Anthropic converter rejects malformed base64 PDFs instead of passing invalid payloads
    assert _convert_content_part_to_anthropic(
        {"type": "document", "mime_type": "application/pdf", "data": "not!valid!b64"}
    ) == {"type": "text", "text": "[document omitted — unsupported by Anthropic]"}

    # Gemini & Interactions adapters both accept Anthropic-shaped {type: "document", source: {type: "base64", ...}}
    anthropic_doc = {
        "type": "document",
        "source": {"type": "base64", "media_type": "application/pdf", "data": _TINY_PDF_B64},
    }
    assert _inline_data_part(anthropic_doc) == {"inlineData": {"mimeType": "application/pdf", "data": _TINY_PDF_B64}}
    profile = get_provider_profile("gemini")
    pkg = sys.modules[type(profile).__module__].__package__
    interactions = importlib.import_module(f"{pkg}.interactions")
    assert interactions._document_content(anthropic_doc) == {
        "type": "document", "mime_type": "application/pdf", "data": _TINY_PDF_B64,
    }

    # Per-modality rejection preserves media-only user turns as a placeholder instead of deleting them
    agent = _DummyAgent(vision=True, video=False, audio=False, doc=True, provider="custom", model="m-user-only")
    pdf_only = [{"role": "user", "content": [{"type": "document", "mime_type": "application/pdf", "data": _TINY_PDF_B64}]}]
    assert record_media_rejection(agent, "pdf input is not supported", pdf_only) == "document"
    assert strip_images_for_rejecting_model(agent, pdf_only) is True
    assert len(pdf_only) == 1 and pdf_only[0]["role"] == "user"
    assert isinstance(pdf_only[0]["content"], str) and "document content removed" in pdf_only[0]["content"].lower()


def test_chat_completions_transport_gating_mixed_rejection_and_corrupt_image_isolation():
    from agent.message_sanitization import _strip_images_from_messages

    # 1. Chat Completions route whose models.dev entry declares audio+pdf MUST still strip raw Gemini-shaped
    #    {type: "audio"|"document", mime_type, data} parts (Chat Completions does not accept that wire schema).
    class _OpenAIChatAgent(VisionMessagePrepMixin):
        provider = "openai"
        model = "gpt-4o-audio-preview"
        api_mode = "chat_completions"
        base_url = "https://api.openai.com/v1"
        requested_provider = "openai"
        _no_list_tool_content_models = set()

    oa = _OpenAIChatAgent()
    caps_all = ModelCapabilities(supports_vision=True, input_modalities=frozenset({"text", "image", "audio", "pdf"}))
    with (
        patch("hermes_cli.config.load_config", return_value={}),
        patch("agent.models_dev.get_model_capabilities", return_value=caps_all),
    ):
        assert oa._model_supports_audio() is False
        assert oa._model_supports_document() is False
        raw_user = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Inspect these"},
                {"type": "audio", "mime_type": "audio/wav", "data": _TINY_WAV_B64},
                {"type": "document", "mime_type": "application/pdf", "data": _TINY_PDF_B64},
            ],
        }]
        prepared = oa._prepare_messages_for_non_vision_model(raw_user)
        part_types = [p["type"] for p in prepared[0]["content"]]
        assert part_types == ["text", "text", "text"]
        assert "audio" in prepared[0]["content"][1]["text"].lower()
        assert "document" in prepared[0]["content"][2]["text"].lower()

    # 2. Generic "unexpected item type in content" on mixed image+audio+document must NOT poison vision
    #    on the first failure: strip audio/document first, keeping image intact for retry.
    mixed_agent = _DummyAgent(vision=True, video=False, audio=True, doc=True, provider="dashscope", model="qwen-vl")
    mixed_msgs = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "check"},
            {"type": "image_url", "image_url": {"url": _TINY_1X1_PNG_URL}},
            {"type": "audio", "mime_type": "audio/wav", "data": _TINY_WAV_B64},
            {"type": "document", "mime_type": "application/pdf", "data": _TINY_PDF_B64},
        ],
    }]
    assert record_media_rejection(mixed_agent, "unexpected item type in content", mixed_msgs) == "audio/document"
    assert mixed_agent._image_rejecting_models == set()
    retry_copy = [dict(mixed_msgs[0], content=list(mixed_msgs[0]["content"]))]
    assert strip_images_for_rejecting_model(mixed_agent, retry_copy) is True
    assert [p["type"] for p in retry_copy[0]["content"]] == ["text", "image_url"]
    # If the image-only retry STILL fails with the same generic error, then mark image-rejecting
    assert record_media_rejection(mixed_agent, "unexpected item type in content", retry_copy) == "image"
    assert ("dashscope", "qwen-vl") in mixed_agent._image_rejecting_models

    # 3. _strip_images_from_messages (used for corrupt-image retry) removes ONLY images, keeping healthy media intact
    corrupt_retry = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "mixed"},
            {"type": "image_url", "image_url": {"url": _TINY_1X1_PNG_URL}},
            {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}},
            {"type": "audio", "mime_type": "audio/wav", "data": _TINY_WAV_B64},
            {"type": "document", "mime_type": "application/pdf", "data": _TINY_PDF_B64},
        ],
    }]
    assert _strip_images_from_messages(corrupt_retry) is True
    assert [p["type"] for p in corrupt_retry[0]["content"]] == ["text", "video_url", "audio", "document"]


def test_tool_result_base64_leak_guard_anthropic_and_bedrock():
    big_pdf = _TINY_PDF_B64 + "A" * 2000
    # 1. Anthropic list-typed tool result with document/audio/video (no image) never falls through to json.dumps
    for part, expected in (
        ({"type": "document", "mime_type": "application/pdf", "data": big_pdf}, "document omitted"),
        ({"type": "audio", "mime_type": "audio/mpeg", "data": big_pdf}, "audio omitted"),
        ({"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{big_pdf}"}}, "video omitted"),
    ):
        _, am = convert_messages_to_anthropic([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": [{"type": "text", "text": "Header"}, part]},
        ])
        tr_content = am[-1]["content"][0]["content"]
        assert isinstance(tr_content, str)
        assert big_pdf not in tr_content
        assert "Header" in tr_content and expected in tr_content.lower()

    # 2. Bedrock Converse tool result with _multimodal dict uses _durable_content instead of json.dumps
    _, bm = convert_messages_to_converse([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": {
            "_multimodal": True,
            "text_summary": "Attached PDF summary",
            "content": [{"type": "document", "mime_type": "application/pdf", "data": big_pdf}],
        }},
    ])
    tr_text = bm[-1]["content"][0]["toolResult"]["content"][0]["text"]
    assert big_pdf not in tr_text and "Attached PDF summary" in tr_text


def test_gemini_input_image_param_mime_youtube_guard_and_interactions_string_image_url():
    # Responses-shape input_image (string image_url) converts cleanly on both generateContent & Interactions
    resp_img = {"type": "input_image", "image_url": _TINY_1X1_PNG_URL}
    inline = _inline_data_part(resp_img)
    assert inline is not None and inline["inlineData"]["mimeType"] == "image/png"

    profile = get_provider_profile("gemini")
    assert profile is not None
    pkg = sys.modules[type(profile).__module__].__package__
    interactions = importlib.import_module(f"{pkg}.interactions")
    steps, _ = interactions.build_input_steps([{"role": "user", "content": [resp_img]}])
    assert steps == [{"type": "user_input", "content": [{"type": "image", "mime_type": "image/png", "data": _TINY_1X1_PNG_URL.split(",", 1)[1]}]}]

    # Parameterized document MIME (e.g. "application/pdf; charset=binary") succeeds on both paths
    param_doc = {"type": "document", "mime_type": "application/pdf; charset=binary", "data": _TINY_PDF_B64}
    assert _inline_data_part(param_doc) == {"inlineData": {"mimeType": "application/pdf", "data": _TINY_PDF_B64}}
    assert interactions._document_content(param_doc) == {"type": "document", "mime_type": "application/pdf", "data": _TINY_PDF_B64}

    # Non-video part pointing at a YouTube URL is NOT rewritten into video/mp4 fileData
    assert _inline_data_part({"type": "image_url", "image_url": {"url": "https://youtu.be/dQw4w9WgXcQ"}}) is None


def test_models_dev_sparse_modality_override_and_bool_string_coercion():
    from agent.image_routing import _lookup_supports_audio, _lookup_supports_pdf
    from agent.models_dev import get_model_capabilities

    # Setting supports_audio=True on a catalog-miss model leaves vision/video/pdf as None (unknown), not False
    cfg_aud = {"model_overrides": {"custom": {"unk-model": {"supports_audio": True}}}}
    caps = get_model_capabilities("custom", "unk-model", allow_network=False, config=cfg_aud)
    assert caps is not None
    assert caps.supports_audio is True
    assert caps.supports_vision is None
    assert caps.supports_video is None
    assert caps.supports_pdf is None

    # Setting supports_vision=True on an unindexed Gemini model preserves Gemini audio/PDF fallbacks
    cfg_vis = {"model_overrides": {"gemini": {"gemini-99-unk": {"supports_vision": True}}}}
    assert _lookup_supports_audio("gemini", "gemini-99-unk", cfg_vis, allow_network=False) is True
    assert _lookup_supports_pdf("gemini", "gemini-99-unk", cfg_vis, allow_network=False) is True

    # String "false" for supports_vision coerces to False (not bool("false") == True)
    cfg_false = {"model_overrides": {"custom": {"unk-model": {"supports_vision": "false"}}}}
    caps_f = get_model_capabilities("custom", "unk-model", allow_network=False, config=cfg_false)
    assert caps_f is not None and caps_f.supports_vision is False


def test_anthropic_pdf_gating_native_google_v1_and_token_accounting_fixes():
    from types import SimpleNamespace
    from agent.chat_completion_helpers import estimate_request_context_tokens
    from agent.context_compressor_media import extract_media_payloads
    from agent.image_routing import (
        is_native_gemini_video_route,
        supports_audio_in_user_turns,
        supports_pdf_in_user_turns,
    )
    from agent.model_metadata_token_memo import _b64_payload_len, estimate_non_image_media_tokens
    from agent.turn_recovery import recover_before_classification

    # Non-Anthropic anthropic_messages providers (MiniMax, custom proxy) do NOT advertise PDF support
    assert supports_pdf_in_user_turns("minimax", "MiniMax-M2.5", {}, api_mode="anthropic_messages") is False
    assert supports_pdf_in_user_turns("bedrock", "anthropic.claude-sonnet-4-6", {}, api_mode="anthropic_messages") is True
    assert supports_pdf_in_tool_results("gemini", "gemini-3-pro", {}, api_mode="anthropic_messages") is False
    assert supports_audio_in_tool_results("gemini", "gemini-3-pro", {}, api_mode="anthropic_messages") is False

    class _MiniMaxAgent(VisionMessagePrepMixin):
        provider = "minimax"
        model = "MiniMax-M2.5-unk"
        api_mode = "anthropic_messages"
        base_url = "https://api.minimax.io/anthropic"
        requested_provider = "minimax"

    with patch("hermes_cli.config.load_config", return_value={}):
        assert _MiniMaxAgent()._model_supports_document() is False

    # Google native /v1 Gemini endpoint is accepted (not falsely rejected as OpenAI-compat /v1)
    v1_url = "https://generativelanguage.googleapis.com/v1"
    assert is_native_gemini_video_route("gemini", "gemini-3-pro", {}, base_url=v1_url) is True
    assert supports_audio_in_user_turns("gemini", "gemini-2.5-pro", {}, base_url=v1_url) is True
    assert supports_pdf_in_user_turns("gemini", "gemini-2.5-pro", {}, base_url=v1_url) is True

    # api_content sidecar wins -> estimate_non_image_media_tokens returns 0
    sidecar_msg = {
        "role": "user",
        "api_content": "short text",
        "content": [{"type": "document", "mime_type": "application/pdf", "data": "A" * 400_000}],
    }
    assert estimate_non_image_media_tokens(sidecar_msg) == 0

    # Flat input_file file_data is measured by both token memo and eviction payload extractor
    flat_file = {"type": "input_file", "file_data": f"data:application/pdf;base64,{'A' * 1000}"}
    assert _b64_payload_len(flat_file) == 1000
    assert extract_media_payloads([flat_file]) == [flat_file["file_data"]]

    # Watchdog payload char estimator strips audio/document base64
    big_pdf_payload = {"messages": [{"role": "user", "content": [{"type": "document", "mime_type": "application/pdf", "data": "A" * 4_000_000}]}]}
    assert estimate_request_context_tokens(big_pdf_payload) < 10_000

    # turn_recovery checks sent wire messages (api_kwargs["messages"]) instead of un-stripped api_messages
    ag = SimpleNamespace(
        provider="deepseek", model="deepseek-chat", api_mode="chat_completions",
        base_url="https://api.deepseek.com/v1", log_prefix="",
        _image_rejecting_models=set(), _rejected_media_modalities={},
        _rejected_reasoning_fields={}, _unicode_sanitization_passes=0,
        _vprint=lambda *a, **k: None,
    )
    api_msgs = [{"role": "user", "content": [{"type": "text", "text": "q"}, {"type": "document", "mime_type": "application/pdf", "data": _TINY_PDF_B64}]}]
    sent_kwargs = {"messages": [{"role": "user", "content": [{"type": "text", "text": "q"}, {"type": "text", "text": "[Document omitted]"}]}]}
    err = type("E", (Exception,), {"status_code": 400, "body": "echoed 'pdf is not supported'"})()
    retry_now, _ = recover_before_classification(ag, err, messages=api_msgs, api_messages=api_msgs, api_kwargs=sent_kwargs, active_system_prompt="")
    assert retry_now is False and ag._rejected_media_modalities == {}
