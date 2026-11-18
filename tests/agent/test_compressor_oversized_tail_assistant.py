"""Regression coverage: the lean tail must be able to shed a runaway assistant message.

A model that degenerates into repetition writes one enormous assistant message, and two
mechanisms pin it inside the tail: ``_find_tail_cut_by_tokens`` keeps a message-count floor
regardless of how many tokens those messages carry, and ``_ensure_last_assistant_message_in_tail``
anchors the newest assistant reply on purpose (#29824). The token budget never binds against
either, so compaction reclaims almost nothing and every later turn re-sends the wall of junk —
which is itself degeneration-inducing input.

Tool results already have this defense (``_demote_stale_tail_tools``); assistant content did not.
Only ``content`` is cut, so ``tool_calls`` and their pairing survive and the full text stays in
session history behind the footer's ``session_search`` pointer.

The cut is opt-in (``compression.tail_assistant_max_chars``, default 0): it rewrites text the
model really emitted, and a legitimate 20K-char report must not be truncated by default.
"""

from __future__ import annotations

import types
from unittest.mock import patch

import pytest

from agent.context_compressor import ContextCompressor

CAP = 16_000


def _stub(max_chars: int = CAP):
    """The method reads only ``_session_id``, ``quiet_mode`` and the limit off ``self``."""
    return types.SimpleNamespace(
        _session_id="s1", quiet_mode=True, tail_assistant_max_chars=max_chars
    )


def _cut(messages, tail_start=0, max_chars=CAP):
    return ContextCompressor._truncate_oversized_tail_assistants(
        _stub(max_chars), messages, tail_start
    )


@pytest.fixture()
def junk() -> str:
    return "tick response\n" * 40_000


def test_runaway_assistant_message_is_cut_and_the_footer_names_the_original(junk):
    """Without the cut the tail keeps ~114K tokens the budget cannot evict; the footer is how
    the agent gets the full text back, so it must name the real size and the session."""
    out = _cut([{"role": "assistant", "content": junk}])

    assert CAP <= len(out[0]["content"]) < CAP + 500
    assert f"truncated at compaction — {len(junk):,} chars" in out[0]["content"]
    assert "session_search(query=..., session_id='s1')" in out[0]["content"]


def test_tool_calls_and_the_api_content_sidecar(junk):
    """``tool_calls`` must survive (pairing) and ``api_content`` must not (it would replay the
    exact text the cut just removed)."""
    tool_calls = [{"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}}]
    messages = [
        {"role": "assistant", "content": junk, "tool_calls": tool_calls, "api_content": junk}
    ]

    out = _cut(messages)

    assert out[0]["tool_calls"] == tool_calls
    assert "api_content" not in out[0]


def test_only_oversized_tail_assistants_are_touched(junk):
    """Rows before ``tail_start``, non-assistant rows and short replies are shared, not copied,
    and the caller's list is never mutated."""
    messages = [
        {"role": "assistant", "content": junk},               # head: outside the tail
        {"role": "user", "content": "go"},
        {"role": "tool", "content": junk, "tool_name": "t"},   # tool demotion's job, not ours
        {"role": "assistant", "content": "short reply"},
        {"role": "assistant", "content": junk},
    ]

    out = _cut(messages, tail_start=1)

    assert all(out[i] is messages[i] for i in (0, 1, 2, 3))
    assert len(out[4]["content"]) < len(junk)
    assert len(messages[4]["content"]) == len(junk)


def test_second_pass_at_the_same_cap_is_a_no_op(junk):
    """Re-cutting at an unchanged cap would restate the size against the already-cut text."""
    once = _cut([{"role": "assistant", "content": junk}])

    assert _cut(once)[0] is once[0]


def test_lowering_the_cap_re_cuts_and_still_names_the_original_size(junk):
    """An operator who lowers the cap on a live session must see the tail actually shrink; the
    footer keeps pointing at the size that was really emitted, not at the earlier cut's size."""
    once = _cut([{"role": "assistant", "content": junk}])

    twice = _cut(once, max_chars=2_000)

    assert 2_000 <= len(twice[0]["content"]) < 2_500
    assert f"truncated at compaction — {len(junk):,} chars" in twice[0]["content"]
    assert twice[0]["content"].startswith(junk[:2_000])


def test_a_message_that_only_quotes_the_marker_is_still_cut(junk):
    """The exemption is for our own terminal footer, not for the marker text: a reply that
    quotes it (an agent discussing its own compacted history) and then runs away must not
    escape the cap."""
    quoted = "[assistant message truncated at compaction — 10 chars preserved in session history.]\n" + junk

    out = _cut([{"role": "assistant", "content": quoted}])

    assert CAP <= len(out[0]["content"]) < CAP + 500
    assert f"truncated at compaction — {len(quoted):,} chars" in out[0]["content"]


def test_zero_disables_the_cut(junk):
    """The default. ``0`` must leave a long legitimate reply exactly as the model wrote it."""
    messages = [{"role": "assistant", "content": junk}]

    assert _cut(messages, max_chars=0) is messages


@pytest.mark.parametrize(
    "configured, expected",
    [(None, 0), (4_000, 4_000), (0, 0), (-5, 0), ("junk", 0)],
)
def test_compression_tail_assistant_max_chars_is_configurable(configured, expected):
    """Disabled unless an operator opts in, and a garbage value must not silently enable it."""
    from agent.agent_init import _parse_compression_config

    agent = types.SimpleNamespace(model="m", api_mode="chat_completions", provider="p", base_url="")
    cfg = {} if configured is None else {"compression": {"tail_assistant_max_chars": configured}}

    assert _parse_compression_config(agent, cfg).tail_assistant_max_chars == expected


# ---------------------------------------------------------------------------
# End-to-end through compress(): the cut must run inside the lean-mode branch.
# ---------------------------------------------------------------------------


@pytest.fixture()
def compressor():
    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        c = ContextCompressor(
            model="test/model", threshold_percent=0.85, protect_first_n=2, protect_last_n=2,
            quiet_mode=True, tail_assistant_max_chars=CAP,
        )
    c.tail_token_budget = 50
    return c


def _transcript(junk: str) -> list[dict]:
    return (
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "initial"}]
        + [
            {"role": "user", "content": f"middle q{i}"} if i % 2 == 0
            else {"role": "assistant", "content": f"middle reply {i}"}
            for i in range(12)
        ]
        + [{"role": "user", "content": "last question"}, {"role": "assistant", "content": junk}]
    )


def _compress(c, messages):
    from agent.context_compressor import SUMMARY_PREFIX
    with patch.object(c, "_generate_summary", return_value=f"{SUMMARY_PREFIX}\nrolled-up middle"):
        return c.compress(messages, current_tokens=90_000)


def test_compress_sheds_the_runaway_message_the_anchor_pins(compressor, junk):
    """The whole point: the newest assistant reply is anchored into the tail by design, so a
    full compaction pass leaves the junk behind unless the cut runs. Post-cut the transcript
    carries the footer instead of the wall of repetition."""
    result = _compress(compressor, _transcript(junk))

    joined = "\n".join(m["content"] for m in result if isinstance(m.get("content"), str))
    assert "[assistant message truncated at compaction" in joined
    assert len(joined) < len(junk) // 10


def test_compress_in_legacy_tail_mode_leaves_assistant_content_alone(compressor, junk):
    """The cut is a lean-tail defense and hangs off the lean branch next to
    ``_demote_stale_tail_tools``; legacy mode keeps its old behaviour."""
    compressor.tail_mode = "legacy"

    result = _compress(compressor, _transcript(junk))

    assert any(m.get("content") == junk for m in result)
