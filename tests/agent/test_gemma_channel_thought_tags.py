"""Gemma 4 channel-thought tag handling across every strip/extract layer.

Gemma 4 embeds reasoning in assistant content using an asymmetric tag
pair (model card, "Thinking Mode Configuration"):

    <|channel>thought\n[Internal reasoning]<channel|>Final answer

Before this change the pair was unknown to every suppression layer, so
the raw tags and the full reasoning text leaked into the visible reply
instead of being folded into the Thinking section like Qwen's
<think>…</think> blocks are.

These tests cover the non-streaming layers (the streaming scrubber is
covered in test_think_scrubber.py::TestGemmaChannelThought):

* ``agent.agent_runtime_helpers.strip_think_blocks`` — storage-boundary
  strip that cleans persisted assistant content.
* ``agent.agent_runtime_helpers.extract_reasoning`` — inline fallback
  that lifts the reasoning into the ``reasoning`` field, which is what
  GUIs render as the collapsible Thinking block.
* ``cli._strip_reasoning_tags`` — the CLI's standalone display strip.
* ``gateway.stream_consumer.GatewayStreamConsumer`` tag tuples — the
  progressive-edit filter's open/close lists must include the pair.
"""

from __future__ import annotations

from types import SimpleNamespace

from agent.agent_runtime_helpers import extract_reasoning, strip_think_blocks
from cli import _strip_reasoning_tags


GEMMA_OUTPUT = (
    "<|channel>thought\n"
    "The user wants me to write a Hello World program in Go. "
    "I have already written the file hello.go using write_file. "
    "However, the go version command failed earlier."
    "<channel|>"
    "Go言語は現在インストールされていないようです。"
)


def _agent() -> SimpleNamespace:
    """Minimal agent stand-in for the helper signatures."""
    return SimpleNamespace(verbose_logging=False)


class TestStripThinkBlocks:
    def test_closed_pair_stripped(self) -> None:
        assert (
            strip_think_blocks(_agent(), GEMMA_OUTPUT)
            == "Go言語は現在インストールされていないようです。"
        )

    def test_empty_thought_block_thinking_disabled(self) -> None:
        content = "<|channel>thought\n<channel|>Final answer"
        assert strip_think_blocks(_agent(), content) == "Final answer"

    def test_unterminated_open_at_start_stripped_to_end(self) -> None:
        content = "<|channel>thought\nreasoning that never closes"
        assert strip_think_blocks(_agent(), content).strip() == ""

    def test_unterminated_open_after_newline_keeps_prior_text(self) -> None:
        content = "Visible text.\n<|channel>thought\nleaked reasoning"
        assert strip_think_blocks(_agent(), content).strip() == "Visible text."

    def test_orphan_close_tag_stripped(self) -> None:
        content = "Hello <channel|>world"
        assert strip_think_blocks(_agent(), content) == "Hello world"

    def test_case_insensitive(self) -> None:
        content = "<|CHANNEL>Thought\nx<CHANNEL|>Answer"
        assert strip_think_blocks(_agent(), content) == "Answer"

    def test_symmetric_variants_unaffected(self) -> None:
        content = "<think>a</think>Answer"
        assert strip_think_blocks(_agent(), content) == "Answer"


class TestExtractReasoning:
    def test_inline_channel_block_lifted_into_reasoning(self) -> None:
        msg = SimpleNamespace(content=GEMMA_OUTPUT)
        reasoning = extract_reasoning(_agent(), msg)
        assert reasoning is not None
        assert reasoning.startswith("The user wants me to write")
        assert "<|channel>" not in reasoning
        assert "<channel|>" not in reasoning

    def test_structured_reasoning_takes_priority(self) -> None:
        msg = SimpleNamespace(
            content=GEMMA_OUTPUT, reasoning="structured wins",
        )
        assert extract_reasoning(_agent(), msg) == "structured wins"

    def test_empty_thought_block_yields_none(self) -> None:
        msg = SimpleNamespace(content="<|channel>thought\n<channel|>Answer")
        assert extract_reasoning(_agent(), msg) is None

    def test_no_tags_yields_none(self) -> None:
        msg = SimpleNamespace(content="Plain answer")
        assert extract_reasoning(_agent(), msg) is None


class TestCliStripReasoningTags:
    def test_closed_pair_stripped(self) -> None:
        assert (
            _strip_reasoning_tags(GEMMA_OUTPUT)
            == "Go言語は現在インストールされていないようです。"
        )

    def test_unterminated_open_stripped(self) -> None:
        assert _strip_reasoning_tags("<|channel>thought\nleaked") == ""

    def test_orphan_tags_stripped(self) -> None:
        assert _strip_reasoning_tags("A <channel|>B") == "A B"


class TestGatewayTagTuples:
    def test_channel_pair_present_and_aligned(self) -> None:
        from gateway.stream_consumer import GatewayStreamConsumer as C

        assert "<|channel>thought" in C._OPEN_THINK_TAGS
        assert "<channel|>" in C._CLOSE_THINK_TAGS
        # The open/close tuples are zipped as pairs by matching index in
        # the pairing consumers — the channel pair must stay aligned.
        open_idx = C._OPEN_THINK_TAGS.index("<|channel>thought")
        close_idx = C._CLOSE_THINK_TAGS.index("<channel|>")
        assert open_idx == close_idx


class TestGluedOpenTag:
    """Regression: real Gemma output glues the open tag to the reasoning.

    The model card shows ``<|channel>thought\\n`` but observed output is
    ``<|channel>thoughtThe user wants…`` — no newline, no separator.  A
    ``\\b`` word-boundary after "thought" silently fails on that input
    (word-char → word-char), so the patterns must not use one.
    """

    GLUED = (
        "<|channel>thoughtThe user wants me to write a Hello World "
        "program in Go.<channel|>"
        "Go言語は現在インストールされていないようです。"
    )

    def test_strip_think_blocks_handles_glued_tag(self) -> None:
        assert (
            strip_think_blocks(_agent(), self.GLUED)
            == "Go言語は現在インストールされていないようです。"
        )

    def test_extract_reasoning_handles_glued_tag(self) -> None:
        msg = SimpleNamespace(content=self.GLUED)
        reasoning = extract_reasoning(_agent(), msg)
        assert reasoning is not None
        assert reasoning.startswith("The user wants me")

    def test_cli_strip_handles_glued_tag(self) -> None:
        assert (
            _strip_reasoning_tags(self.GLUED)
            == "Go言語は現在インストールされていないようです。"
        )


class TestAuxiliaryExtractContentOrReasoning:
    """auxiliary_client.extract_content_or_reasoning — same asymmetric-tag gap.

    Title generation and other auxiliary LLM calls resolve response text
    through this helper, which mirrors _strip_think_blocks with its own
    inline regexes.  It gained <thought>…</thought> in #8562 but not the
    asymmetric Gemma 4 channel pair; with a Gemma auxiliary model the
    reasoning leaked into generated titles/summaries.
    """

    @staticmethod
    def _resp(content, **fields):
        from types import SimpleNamespace

        msg = SimpleNamespace(
            content=content,
            reasoning=fields.get("reasoning"),
            reasoning_content=fields.get("reasoning_content"),
            reasoning_details=fields.get("reasoning_details"),
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    def test_channel_pair_stripped_glued(self) -> None:
        from agent.auxiliary_client import extract_content_or_reasoning

        out = extract_content_or_reasoning(self._resp(
            "<|channel>thoughtLet me think of a title.<channel|>Go setup help"
        ))
        assert out == "Go setup help"

    def test_channel_pair_stripped_with_newline(self) -> None:
        from agent.auxiliary_client import extract_content_or_reasoning

        out = extract_content_or_reasoning(self._resp(
            "<|channel>thought\nreasoning here\n<channel|>Title text"
        ))
        assert out == "Title text"

    def test_symmetric_thought_still_stripped(self) -> None:
        from agent.auxiliary_client import extract_content_or_reasoning

        out = extract_content_or_reasoning(self._resp(
            "<thought>hmm</thought>Plain title"
        ))
        assert out == "Plain title"

    def test_reasoning_only_content_falls_back_to_structured(self) -> None:
        """All-reasoning content must fall through to structured fields."""
        from agent.auxiliary_client import extract_content_or_reasoning

        out = extract_content_or_reasoning(self._resp(
            "<|channel>thoughtonly reasoning, no answer<channel|>",
            reasoning="Structured fallback title",
        ))
        assert out == "Structured fallback title"

    def test_untagged_content_untouched(self) -> None:
        from agent.auxiliary_client import extract_content_or_reasoning

        assert extract_content_or_reasoning(self._resp("As-is")) == "As-is"
