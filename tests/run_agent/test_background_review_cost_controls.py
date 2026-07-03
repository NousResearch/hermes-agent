"""Unit coverage for the background-review aux-model selector + routed digest.

Covers the two behaviors this change adds:
  • _resolve_review_runtime — auto/same-model → not routed (main model, warm
    cache); a configured different model → routed with resolved credentials.
  • _digest_history — compact replay used ONLY on the routed path (recent tail
    verbatim + a digest of older turns), preserving role alternation.

Pure-function / config-driven; no live model calls.
"""
from unittest.mock import patch

from agent import background_review as br


def _msg(role, content, tool_calls=None):
    m = {"role": role, "content": content}
    if tool_calls:
        m["tool_calls"] = tool_calls
    return m


# ---------------------------------------------------------------------------
# _resolve_review_runtime — the aux-model selector
# ---------------------------------------------------------------------------

class _FakeAgent:
    def __init__(self, provider="openai-codex", model="gpt-5.5"):
        self.provider = provider
        self.model = model

    def _current_main_runtime(self):
        return {
            "api_key": "parent-key",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_mode": "codex_app_server",
        }


def test_routing_auto_inherits_parent_and_downgrades_codex_app_server():
    agent = _FakeAgent()
    cfg = {"auxiliary": {"background_review": {"provider": "auto", "model": ""}}}
    with patch("hermes_cli.config.load_config", return_value=cfg):
        rt = br._resolve_review_runtime(agent)
    assert rt["routed"] is False
    assert rt["provider"] == "openai-codex"
    assert rt["model"] == "gpt-5.5"
    assert rt["api_mode"] == "codex_responses"  # downgraded so agent-loop tools dispatch


def test_routing_to_different_model_marks_routed_and_resolves_credentials():
    agent = _FakeAgent()
    cfg = {"auxiliary": {"background_review": {
        "provider": "openrouter", "model": "google/gemini-3-flash-preview",
    }}}
    fake_rp = {
        "provider": "openrouter", "api_key": "or-key",
        "base_url": "https://openrouter.ai/api/v1", "api_mode": "chat_completions",
    }
    with patch("hermes_cli.config.load_config", return_value=cfg), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=fake_rp):
        rt = br._resolve_review_runtime(agent)
    assert rt["routed"] is True
    assert rt["provider"] == "openrouter"
    assert rt["model"] == "google/gemini-3-flash-preview"
    assert rt["api_key"] == "or-key"


def test_routing_same_model_as_parent_is_not_routed():
    agent = _FakeAgent(provider="openrouter", model="anthropic/claude-opus-4.8")
    cfg = {"auxiliary": {"background_review": {
        "provider": "openrouter", "model": "anthropic/claude-opus-4.8",
    }}}
    with patch("hermes_cli.config.load_config", return_value=cfg):
        rt = br._resolve_review_runtime(agent)
    assert rt["routed"] is False  # same model/provider → keep full-replay path


def test_routing_resolution_failure_falls_back_to_parent():
    agent = _FakeAgent()
    cfg = {"auxiliary": {"background_review": {
        "provider": "openrouter", "model": "google/gemini-3-flash-preview",
    }}}
    with patch("hermes_cli.config.load_config", return_value=cfg), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider",
               side_effect=RuntimeError("boom")):
        rt = br._resolve_review_runtime(agent)
    assert rt["routed"] is False
    assert rt["provider"] == "openai-codex"


# ---------------------------------------------------------------------------
# _digest_history — routed-path compact replay
# ---------------------------------------------------------------------------

def test_digest_under_tail_returns_full():
    msgs = [_msg("user", "hi"), _msg("assistant", "hello")]
    assert br._digest_history(msgs, tail=24) == msgs


def test_digest_collapses_old_keeps_tail_verbatim():
    msgs = []
    for i in range(60):
        msgs.append(_msg("user", f"u{i} " + "x" * 50))
        msgs.append(_msg("assistant", f"a{i} " + "y" * 50))
    out = br._digest_history(msgs, tail=10)
    # First message is the synthetic digest (user role → alternation preserved).
    assert out[0]["role"] == "user"
    assert out[0]["content"].startswith("[Earlier conversation digest")
    # Recent tail preserved verbatim.
    assert out[-1] == msgs[-1]
    assert len(out) == 11  # 1 digest + 10 tail


def test_digest_does_not_open_tail_on_a_tool_message():
    msgs = []
    for i in range(40):
        msgs.append(_msg("user", "u" + "x" * 50))
        msgs.append(_msg("assistant", "", tool_calls=[
            {"function": {"name": "terminal", "arguments": "{}"}}]))
        msgs.append({"role": "tool", "content": "result " + "w" * 50})
    out = br._digest_history(msgs, tail=2)
    # The verbatim tail (after the digest) must not begin on a bare tool message.
    assert out[1]["role"] != "tool"


def test_digest_records_tool_names_in_arc():
    old = [
        _msg("user", "do the thing"),
        _msg("assistant", "", tool_calls=[
            {"function": {"name": "skill_view", "arguments": "{}"}},
            {"function": {"name": "patch", "arguments": "{}"}}]),
    ]
    msgs = old + [_msg("user", f"tail{i}") for i in range(30)]
    out = br._digest_history(msgs, tail=10)
    digest = out[0]["content"]
    assert "USER: do the thing" in digest
    assert "tools: skill_view, patch" in digest


# ---------------------------------------------------------------------------
# media_replay=text_only — strip media from background-review private snapshot
# ---------------------------------------------------------------------------

def test_background_review_media_replay_defaults_to_full():
    with patch("hermes_cli.config.load_config", return_value={}):
        assert br._background_review_media_replay_mode() == "full"


def test_background_review_media_replay_accepts_documented_text_only_alias():
    cfg = {"auxiliary": {"background_review": {"media_replay": "text-only"}}}
    with patch("hermes_cli.config.load_config", return_value=cfg):
        assert br._background_review_media_replay_mode() == "text_only"


def test_background_review_media_replay_unknown_values_fall_back_to_full():
    cfg = {"auxiliary": {"background_review": {"media_replay": "strip-images"}}}
    with patch("hermes_cli.config.load_config", return_value=cfg):
        assert br._background_review_media_replay_mode() == "full"


def test_strip_media_parts_preserves_text_and_omits_images_without_mutating_input():
    original = [
        {
            "role": "tool",
            "content": [
                {"type": "text", "text": "Image loaded. Question: what is visible?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
        },
        {"role": "assistant", "content": "The screenshot shows Excel with B2 selected."},
    ]

    out = br._background_review_strip_media_parts(original)

    assert out is not original
    assert original[0]["content"][1]["type"] == "image_url"  # input untouched
    assert out[0]["content"][0] == {"type": "text", "text": "Image loaded. Question: what is visible?"}
    assert all(part.get("type") != "image_url" for part in out[0]["content"])
    assert "Image omitted from background review" in out[0]["content"][-1]["text"]
    assert out[1] == original[1]


def test_strip_media_parts_handles_multimodal_envelopes():
    original = [{
        "role": "tool",
        "content": {
            "_multimodal": True,
            "text_summary": "capture mode=vision 100x100",
            "content": [
                {"type": "text", "text": "capture mode=vision 100x100"},
                {"type": "input_image", "image_url": {"url": "data:image/png;base64,BBBB"}},
            ],
        },
    }]

    out = br._background_review_strip_media_parts(original)
    content = out[0]["content"]

    assert isinstance(content, str)
    assert "capture mode=vision" in content
    assert "data:image" not in content
    assert "input_image" not in content
    assert "Image omitted from background review" in content


def test_strip_media_parts_leaves_text_only_multimodal_envelopes_unchanged():
    original = [{
        "role": "tool",
        "content": {
            "_multimodal": True,
            "text_summary": "plain text capture metadata",
            "content": [{"type": "text", "text": "plain text capture metadata"}],
        },
    }]

    out = br._background_review_strip_media_parts(original)

    assert out == original
    assert out is not original
    assert out[0]["content"] is not original[0]["content"]
    assert "Image omitted" not in str(out[0]["content"])
