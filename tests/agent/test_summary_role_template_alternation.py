"""Regression coverage: the compaction summary role must alternate against
TEMPLATE-VISIBLE neighbours, not literal list neighbours.

Mistral-family chat templates (Devstral, Mistral Small 3.x, Magistral)
enforce user/assistant alternation at render time but exempt the tool flow
from the check: ``tool`` results and assistant messages carrying
``tool_calls`` are skipped. The summary-role selection previously keyed off
the LITERAL last head message (``compressed[-1]``), producing this captured
failure (Hermes Desktop v0.19.0 against llama.cpp-served Devstral):

    [0] system
    [1] user            <- original first user turn, protected
    [2] assistant       (tool_calls, exempt)
    [3] tool            (exempt)
    [4] user            <- COMPACTION SUMMARY, pinned to "user" because the
                           literal previous role was "tool"
    [5..] assistant(tool_calls)/tool pairs

The template counts only [1] and [4]: user -> user, and the backend rejects
the ENTIRE request with a Jinja alternation error (HTTP 500):

    "After the optional system message, conversation roles must alternate
     user and assistant roles except for tool calls and results"

Because the summary persists in the stored conversation, every retry
replays the same poisoned history: the session is permanently broken, on
EVERY compaction, overflow or not.

These tests pin the fix: neighbour roles for summary-role selection are
computed by ``_template_visible_role`` (skipping the exempt tool flow), and
the assembled output always satisfies the Mistral alternation check. The
existing #52160 / #58753 forced-user guards must keep winning: their forced
shapes (summary-user followed only by exempt messages) are alternation-safe.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.context_compressor import (
    COMPRESSED_SUMMARY_METADATA_KEY,
    SUMMARY_PREFIX,
    _template_visible_role,
)


@pytest.fixture()
def compressor():
    from agent.context_compressor import ContextCompressor

    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=100_000,
    ):
        c = ContextCompressor(
            model="test/model",
            threshold_percent=0.50,
            protect_first_n=3,
            protect_last_n=20,
            quiet_mode=True,
        )
        c.tail_token_budget = 40
        return c


def _tool_turns(start: int, n: int, payload: str = "x" * 300) -> list[dict]:
    out: list[dict] = []
    for i in range(start, start + n):
        out.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"c{i}",
                        "function": {"name": "terminal", "arguments": "{}"},
                    }
                ],
            }
        )
        out.append({"role": "tool", "content": payload, "tool_call_id": f"c{i}"})
    return out


def _summary_rows(messages: list[dict]) -> list[dict]:
    return [
        m
        for m in messages
        if isinstance(m, dict) and m.get(COMPRESSED_SUMMARY_METADATA_KEY)
    ]


def _mistral_alternation_ok(messages: list[dict]) -> bool:
    """Replay the Mistral template's pre-flight alternation check: count
    only user and assistant-without-tool_calls messages; the counted
    sequence must go user, assistant, user, assistant, ..."""
    expect = "user"
    for m in messages:
        role = _template_visible_role(m)
        if role in (None, "system"):
            continue
        if role != expect:
            return False
        expect = "assistant" if expect == "user" else "user"
    return True


class TestTemplateVisibleRoleHelper:
    def test_tool_flow_is_exempt(self):
        assert _template_visible_role({"role": "tool", "content": "r"}) is None
        assert (
            _template_visible_role(
                {"role": "assistant", "content": "", "tool_calls": [{"id": "c"}]}
            )
            is None
        )

    def test_counted_roles_pass_through(self):
        assert _template_visible_role({"role": "user", "content": "q"}) == "user"
        assert (
            _template_visible_role({"role": "assistant", "content": "a"})
            == "assistant"
        )
        assert _template_visible_role({"role": "system", "content": "s"}) == "system"

    def test_non_dict_is_exempt(self):
        assert _template_visible_role(None) is None
        assert _template_visible_role("not a message") is None


class TestSummaryRoleAlternatesAgainstVisibleNeighbours:
    def test_captured_devstral_shape_emits_assistant_summary(self, compressor):
        """The byte-captured poisoning shape: protected head ends
        ``[user, assistant(tool_calls), tool]``, tail is all tool flow.
        The literal previous role is ``tool`` (which used to pin the
        summary to "user"); the template-visible previous role is
        ``user``, so the summary must be emitted as ``assistant``."""
        c = compressor
        messages = [{"role": "user", "content": "run a full systems diagnostic"}]
        messages += _tool_turns(0, 30)

        mocked = f"{SUMMARY_PREFIX}\nrolled-up summary of the tool work"
        with patch.object(c, "_generate_summary", return_value=mocked):
            out = c.compress(messages, current_tokens=90_000)

        rows = _summary_rows(out)
        assert len(rows) == 1
        assert rows[0].get("role") == "assistant", (
            "REGRESSION: compaction summary emitted as role=user directly "
            "after a template-visible user turn (only exempt tool-flow "
            "messages between). Mistral-strict templates reject the whole "
            "request with a Jinja alternation 500 and the stored session is "
            f"poisoned permanently. Got role={rows[0].get('role')!r}."
        )

    def test_captured_shape_passes_mistral_alternation(self, compressor):
        c = compressor
        messages = [{"role": "user", "content": "run a full systems diagnostic"}]
        messages += _tool_turns(0, 30)

        mocked = f"{SUMMARY_PREFIX}\nrolled-up summary of the tool work"
        with patch.object(c, "_generate_summary", return_value=mocked):
            out = c.compress(messages, current_tokens=90_000)

        assert _mistral_alternation_ok(out), (
            "Compressed transcript fails the Mistral template alternation "
            "check. Visible-role sequence: "
            f"{[_template_visible_role(m) for m in out if _template_visible_role(m)]}"
        )

    def test_visible_head_assistant_visible_tail_user_merges(self, compressor):
        """When the visible head ends ``assistant`` and the visible tail
        opens ``user``, NO standalone role can satisfy alternation (user
        collides with the tail, assistant with the head -- and the
        intervening tool flow is template-exempt, so literal separation
        does not help). The existing merge-into-tail fallback must fire.
        The literal-role logic used to emit a standalone role="user"
        summary here (literal previous role was ``tool``), which is a
        second poisoning shape: visible user(summary) -> user(tail)."""
        c = compressor
        messages = [
            {"role": "user", "content": "question " + "q" * 200},
            {"role": "assistant", "content": "answer " + "a" * 200},
        ]
        messages += _tool_turns(0, 30)
        # A recent user turn preserved in the tail keeps the zero-user
        # guard out of the picture.
        messages += [
            {"role": "user", "content": "latest question"},
            {"role": "assistant", "content": "on it"},
        ]

        mocked = f"{SUMMARY_PREFIX}\nsummary body"
        with patch.object(c, "_generate_summary", return_value=mocked):
            out = c.compress(messages, current_tokens=90_000)

        rows = _summary_rows(out)
        assert len(rows) == 1
        # Merged into a template-exempt tail message: invisible to the
        # alternation check, summary content still delivered.
        assert _template_visible_role(rows[0]) is None
        assert _mistral_alternation_ok(out)


class TestForcedUserGuardsStillWin:
    def test_zero_user_guard_still_forces_user(self, compressor):
        """#58753: when no genuine user turn survives, the summary must
        still be pinned to role=user (and that shape is alternation-safe
        because everything after it is template-exempt)."""
        c = compressor
        c.compression_count = 1  # protect_first_n decays -> no head
        messages = [{"role": "user", "content": "work kanban task 42"}]
        messages += _tool_turns(0, 12)

        mocked = f"{SUMMARY_PREFIX}\nsummary body"
        with patch.object(c, "_generate_summary", return_value=mocked):
            out = c.compress(messages, current_tokens=90_000)

        rows = _summary_rows(out)
        assert len(rows) == 1
        assert rows[0].get("role") == "user"
        assert _mistral_alternation_ok(out)

    def test_no_literal_consecutive_user_roles(self, compressor):
        """The pre-existing literal invariant still holds alongside the
        template-visible one."""
        c = compressor
        messages = [{"role": "user", "content": "run diagnostics"}]
        messages += _tool_turns(0, 30)

        mocked = f"{SUMMARY_PREFIX}\nsummary body"
        with patch.object(c, "_generate_summary", return_value=mocked):
            out = c.compress(messages, current_tokens=90_000)

        for prev, cur in zip(out, out[1:]):
            assert not (
                prev.get("role") == "user" and cur.get("role") == "user"
            ), "compression introduced literal consecutive user-role messages"
