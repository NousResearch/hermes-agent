"""Reasoning effort vs. reasoning VISIBILITY separation for Gemini/Vertex.

Regression tests for the Gemini thought-summary leak: with
``display.show_reasoning: false`` (and every platform override off), Hermes
still asked Vertex for thought summaries (``include_thoughts: true`` was
hardcoded whenever thinking was enabled) and Vertex's OpenAI-compat surface
returned them as UNTAGGED plain text inside ``message.content`` — so users
saw blocks like::

    **Recalling User Details**
    I'm recalling...

before (or instead of) the actual answer, and no display gate could catch
them.

The fix has two halves:

* Request: ``reasoning_config["include_thoughts"]`` (resolved from
  display.show_reasoning / per-platform overrides / ``/reasoning show|hide``)
  maps onto ``thinking_config.include_thoughts`` — independent of
  ``thinking_level``, which still comes from ``agent.reasoning_effort``.
* Response (display ON only): ``extra_body.google.thought_tag_marker`` makes
  the compat layer wrap summaries in ``<think>`` tags so the existing
  structured pipeline (extract_reasoning → msg["reasoning"],
  strip_think_blocks at the storage boundary, stream-consumer suppression)
  handles them — no natural-language heuristics.
"""

import pytest
from types import SimpleNamespace

from agent.transports import get_transport
from providers import get_provider_profile


@pytest.fixture
def transport():
    import agent.transports.chat_completions  # noqa: F401
    return get_transport("chat_completions")


def _vertex_kwargs(transport, reasoning_config, model="google/gemini-3.1-pro-preview"):
    profile = get_provider_profile("vertex")
    return transport.build_kwargs(
        model=model,
        messages=[{"role": "user", "content": "Hi"}],
        tools=[],
        reasoning_config=reasoning_config,
        supports_reasoning=True,
        provider_profile=profile,
        provider_name="vertex",
        base_url="https://aiplatform.googleapis.com/v1beta1/projects/p/locations/global/endpoints/openapi",
    )


def _google_extra(kw):
    return kw["extra_body"]["extra_body"]["google"]


class TestVertexRequestVisibility:
    """Request-body assertions for the Vertex OpenAI-compat surface."""

    def test_medium_effort_display_off_keeps_thinking_hides_thoughts(self, transport):
        kw = _vertex_kwargs(
            transport,
            {"enabled": True, "effort": "medium", "include_thoughts": False},
        )
        google = _google_extra(kw)
        # Thinking stays ON at the effort-mapped level (medium → low for
        # Gemini 3.x Pro); only the summary return is suppressed.
        assert google["thinking_config"] == {
            "include_thoughts": False,
            "thinking_level": "low",
        }
        # No tags needed when no thoughts are returned.
        assert "thought_tag_marker" not in google

    def test_high_effort_display_off_preserves_high_level(self, transport):
        kw = _vertex_kwargs(
            transport,
            {"enabled": True, "effort": "high", "include_thoughts": False},
        )
        assert _google_extra(kw)["thinking_config"] == {
            "include_thoughts": False,
            "thinking_level": "high",
        }

    def test_display_on_requests_thoughts_with_tag_marker(self, transport):
        kw = _vertex_kwargs(
            transport,
            {"enabled": True, "effort": "medium", "include_thoughts": True},
        )
        google = _google_extra(kw)
        assert google["thinking_config"] == {
            "include_thoughts": True,
            "thinking_level": "low",
        }
        # Structured separation: without the marker the compat layer returns
        # summaries as untagged content text (Vertex docs: "If not specified,
        # no tags will be returned around the model's thoughts").
        assert google["thought_tag_marker"] == "think"

    def test_missing_visibility_key_keeps_legacy_behavior(self, transport):
        """Callers that don't resolve visibility (old configs, third-party
        embedders) keep the pre-fix request shape."""
        kw = _vertex_kwargs(transport, {"enabled": True, "effort": "medium"})
        google = _google_extra(kw)
        assert google["thinking_config"]["include_thoughts"] is True
        assert google["thought_tag_marker"] == "think"

    def test_reasoning_disabled_still_omits_thoughts(self, transport):
        kw = _vertex_kwargs(
            transport, {"enabled": False, "include_thoughts": True}
        )
        assert _google_extra(kw)["thinking_config"] == {"include_thoughts": False}

    def test_visibility_does_not_leak_into_openrouter_reasoning(self, transport):
        """Non-Gemini providers must not receive the Hermes-internal key."""
        profile = get_provider_profile("openrouter")
        kw = transport.build_kwargs(
            model="deepseek/deepseek-r1",
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
            reasoning_config={"enabled": True, "effort": "high", "include_thoughts": False},
            supports_reasoning=True,
            provider_profile=profile,
            provider_name="openrouter",
            base_url=profile.base_url,
        )
        assert kw["extra_body"]["reasoning"] == {"enabled": True, "effort": "high"}

    def test_visibility_does_not_leak_into_nous_reasoning(self, transport):
        profile = get_provider_profile("nous")
        kw = transport.build_kwargs(
            model="hermes-4-405b",
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
            reasoning_config={"enabled": True, "effort": "medium", "include_thoughts": False},
            supports_reasoning=True,
            provider_profile=profile,
            provider_name="nous",
            base_url=profile.base_url,
        )
        assert kw["extra_body"]["reasoning"] == {"enabled": True, "effort": "medium"}


class TestGeminiProviderVisibility:
    """The AI-Studio ``gemini`` profile shares the same builders."""

    def _kwargs(self, transport, reasoning_config, base_url):
        profile = get_provider_profile("gemini")
        return transport.build_kwargs(
            model="gemini-3.1-pro-preview",
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
            reasoning_config=reasoning_config,
            supports_reasoning=True,
            provider_profile=profile,
            provider_name="gemini",
            base_url=base_url,
        )

    def test_openai_compat_display_off(self, transport):
        kw = self._kwargs(
            transport,
            {"enabled": True, "effort": "high", "include_thoughts": False},
            "https://generativelanguage.googleapis.com/v1beta/openai",
        )
        google = kw["extra_body"]["extra_body"]["google"]
        assert google["thinking_config"] == {
            "include_thoughts": False,
            "thinking_level": "high",
        }
        assert "thought_tag_marker" not in google

    def test_openai_compat_display_on_adds_marker(self, transport):
        kw = self._kwargs(
            transport,
            {"enabled": True, "effort": "high", "include_thoughts": True},
            "https://generativelanguage.googleapis.com/v1beta/openai",
        )
        google = kw["extra_body"]["extra_body"]["google"]
        assert google["thinking_config"]["include_thoughts"] is True
        assert google["thought_tag_marker"] == "think"

    def test_native_path_display_off_no_marker_key(self, transport):
        """The native REST client marks thought parts structurally
        (``part.thought``); it gets includeThoughts=False and never a marker."""
        kw = self._kwargs(
            transport,
            {"enabled": True, "effort": "medium", "include_thoughts": False},
            "https://generativelanguage.googleapis.com/v1beta",
        )
        assert kw["extra_body"]["thinking_config"] == {
            "includeThoughts": False,
            "thinkingLevel": "low",
        }
        assert "thought_tag_marker" not in kw["extra_body"]


class TestThoughtSignaturesUnaffected:
    """Hiding summaries must not break Gemini tool-call signature replay."""

    def test_normalize_response_preserves_tool_call_signature(self, transport):
        tc = SimpleNamespace(
            id="call_1",
            type="function",
            function=SimpleNamespace(name="t", arguments="{}"),
            extra_content={"google": {"thought_signature": "SIG_999"}},
        )
        msg = SimpleNamespace(content=None, tool_calls=[tc], reasoning=None)
        resp = SimpleNamespace(
            choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")],
            usage=None,
        )
        norm = transport.normalize_response(resp)
        assert norm.tool_calls[0].provider_data == {
            "extra_content": {"google": {"thought_signature": "SIG_999"}}
        }

    def test_convert_messages_replays_signature_for_gemini_target(self, transport):
        msgs = [{
            "role": "assistant", "content": "ok",
            "tool_calls": [{
                "id": "call_1", "type": "function",
                "extra_content": {"google": {"thought_signature": "SIG_999"}},
                "function": {"name": "t", "arguments": "{}"},
            }],
        }]
        out = transport.convert_messages(msgs, model="google/gemini-3.1-pro-preview")
        assert out[0]["tool_calls"][0]["extra_content"] == {
            "google": {"thought_signature": "SIG_999"}
        }


# ── Response-side regression fixture ─────────────────────────────────────
# What the leak looked like in production: repeated near-identical summary
# blocks. With thought_tag_marker set, the compat layer wraps each in
# <think> tags — the structured signal the pipeline keys on.
_TAGGED_SUMMARIES = (
    "<think>**Recalling User Details**\nI'm recalling...</think>"
    "<think>**Synthesizing User Data**\nI'm synthesizing...</think>"
)
_LEGIT_ANSWER = (
    "**Victor Chun**\n\nVictor is a Berkeley student who builds AI systems.\n\n"
    "**Highlights**\n- NutriScan\n- RushOS"
)


class _FakeAgent:
    """Minimal agent stub for build_assistant_message."""
    verbose_logging = False
    reasoning_callback = None
    stream_delta_callback = None
    _stream_callback = None

    def _extract_reasoning(self, assistant_message):
        from agent.agent_runtime_helpers import extract_reasoning
        return extract_reasoning(self, assistant_message)

    def _strip_think_blocks(self, content):
        from agent.agent_runtime_helpers import strip_think_blocks
        return strip_think_blocks(self, content)


class TestThoughtContentSeparation:
    def _build(self, content):
        from agent.chat_completion_helpers import build_assistant_message
        msg = SimpleNamespace(
            content=content, tool_calls=None,
            reasoning=None, reasoning_content=None, reasoning_details=None,
        )
        return build_assistant_message(_FakeAgent(), msg, "stop")

    def test_tagged_summaries_removed_from_content_captured_once_as_reasoning(self):
        built = self._build(_TAGGED_SUMMARIES + _LEGIT_ANSWER)
        # Structured thought content is gone from user-visible content …
        assert "Recalling User Details" not in built["content"]
        assert "Synthesizing User Data" not in built["content"]
        # … the legitimate bold-markdown answer is untouched …
        assert built["content"] == _LEGIT_ANSWER
        # … and the reasoning is captured exactly once for the display gate.
        assert built["reasoning"].count("Recalling User Details") == 1
        assert built["reasoning"].count("Synthesizing User Data") == 1

    def test_untagged_bold_paragraphs_are_never_stripped(self):
        """Guard against natural-language heuristics: a legitimate answer that
        *looks* like a thought summary (leading bold headers) must survive
        verbatim when there is no structured thought marker."""
        lookalike = (
            "**Crafting Victor's Description**\n\n"
            "Here is the requested description, formatted as asked.\n\n"
            + _LEGIT_ANSWER
        )
        built = self._build(lookalike)
        assert built["content"] == lookalike
        assert built["reasoning"] is None

    def test_reasoning_only_content_yields_empty_visible_text(self):
        """Dozens of summaries with no answer → nothing user-visible leaks;
        the loop's empty-response recovery takes over instead."""
        many = "".join(
            f"<think>**Recalling User Details {i}**\nI'm recalling...</think>"
            for i in range(24)
        )
        built = self._build(many)
        assert built["content"].strip() == ""
        assert "Recalling User Details 0" in built["reasoning"]


class TestStreamingSuppression:
    """Streaming path: tagged summaries never reach the platform buffer."""

    def _consumer(self):
        from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
        return GatewayStreamConsumer(
            adapter=SimpleNamespace(),
            chat_id="c1",
            config=StreamConsumerConfig(),
        )

    def test_tagged_summary_fixture_suppressed_mid_stream(self):
        c = self._consumer()
        c._filter_and_accumulate(
            "<think>**Recalling User Details**\nI'm recalling...</think>"
        )
        c._filter_and_accumulate(_LEGIT_ANSWER)
        assert "Recalling User Details" not in c._accumulated
        assert c._accumulated == _LEGIT_ANSWER

    def test_tag_split_across_deltas_still_suppressed(self):
        c = self._consumer()
        for delta in ("<thi", "nk>**Synthesizing User Data**", " I'm synthesizing...", "</think>", "Final answer."):
            c._filter_and_accumulate(delta)
        assert "Synthesizing" not in c._accumulated
        assert c._accumulated == "Final answer."
