"""Tests for harmony reasoning-leak scrubbing (agent/harmony_scrub.py).

Covers the complete-string stripper (strip_harmony_leak) and the streaming gate
(via StreamingThinkScrubber), including the over-strip regressions an adversarial
review found in the first cut of this fix, and that benign prose is untouched.
"""

import pytest

from agent.harmony_scrub import strip_harmony_leak
from agent.think_scrubber import StreamingThinkScrubber


def _stream(deltas):
    """Feed deltas through a fresh scrubber and return the assembled output."""
    s = StreamingThinkScrubber()
    out = [s.feed(d) for d in deltas]
    out.append(s.flush())
    return "".join(out)


# ── strip_harmony_leak: real leak shapes are stripped ───────────────────────
@pytest.mark.parametrize("inp,exp", [
    # degraded shape (a): bare channel word + reasoning + lone <channel|> + answer
    ("thought\nThe user wants X. My plan is Y.<channel|>Here is the answer.",
     "Here is the answer."),
    ("analysis\nreasoning here<channel|>final<|message|>The reply.", "The reply."),
    # canonical shape (b): full control tokens, answer in the final channel
    ("<|channel|>analysis<|message|>plan plan<|channel|>final<|message|>The answer.",
     "The answer."),
    ("<|start|>assistant<|channel|>analysis<|message|>think<|end|>"
     "<|start|>assistant<|channel|>final<|message|>Done.<|return|>", "Done."),
    # commentary channel is handled too
    ("<|channel|>commentary<|message|>hmm<|channel|>final<|message|>Yes.", "Yes."),
    # multiple analysis blocks → keep only after the LAST final marker
    ("<|channel|>analysis<|message|>a<|channel|>final<|message|>x"
     "<|channel|>final<|message|>real", "real"),
])
def test_strip_harmony_leak_strips_real_leaks(inp, exp):
    assert strip_harmony_leak(inp) == exp


# ── over-strip regressions (what the review caught) ─────────────────────────
def test_final_word_of_answer_is_preserved():
    # Degraded separator must NOT consume a following "Final"/"Finally".
    assert strip_harmony_leak("analysis\nx<channel|>Final answer: 42") == "Final answer: 42"
    assert strip_harmony_leak("thought\ny<channel|>Finally, done.") == "Finally, done."


def test_canonical_is_not_glued_together():
    # The token-only strip must not leave reasoning glued to the answer.
    out = strip_harmony_leak("<|channel|>analysis<|message|>plan plan<|channel|>final<|message|>Answer.")
    assert out == "Answer."
    assert "plan" not in out


# ── benign prose is never touched ───────────────────────────────────────────
@pytest.mark.parametrize("text", [
    "The computer, eh? Let me help with that.",
    "Analysis of Q2 revenue: we grew 12%.",          # word followed by prose
    "Thought experiments are a useful tool.",         # word followed by prose
    "Analysis:\nRevenue grew 12% this quarter.",      # standalone word but no channel token
    "In harmony format, the <|channel|> token separates channels.",  # quotes a token mid-msg
    "Here is how <|channel|>final<|message|> works in the spec.",     # quotes final marker mid-msg
    # Combined-condition false positive (both a heading AND a quoted token) — the
    # heading uses a COLON, so it must not be treated as a bare channel-name head.
    "Analysis:\nThe harmony format uses a <|channel|> token to separate sections. Done.",
    "Analysis:\nModels emit <|channel|>final<|message|> before the answer. Note that.",
    "thought:\nremember the <|channel|> control token exists.",
    "",
])
def test_strip_harmony_leak_leaves_benign_prose(text):
    assert strip_harmony_leak(text) == text


# ── streaming: leaks suppressed even split across deltas ─────────────────────
def test_stream_bare_word_leak_suppressed():
    assert _stream(["thou", "ght\nplan plan ", "more<chan", "nel|>", "Hi ", "there."]) == "Hi there."


def test_stream_canonical_leak_split_tokens():
    assert _stream(["<|chan", "nel|>analysis<|mess", "age|>plan<|channel|>fi",
                    "nal<|message|>Real ", "answer."]) == "Real answer."


def test_stream_final_word_preserved():
    assert _stream(["analysis\n", "x<channel|>", "Final answer: 42"]) == "Final answer: 42"


# ── streaming: benign output is byte-for-byte unchanged ─────────────────────
@pytest.mark.parametrize("deltas,exp", [
    (["Analysis of ", "Q2 revenue: ", "we grew 12%."], "Analysis of Q2 revenue: we grew 12%."),
    (["Thought ", "experiments ", "are fun."], "Thought experiments are fun."),
    (["Just ", "a normal ", "reply."], "Just a normal reply."),
    (["The answer ", "is 42."], "The answer is 42."),
])
def test_stream_benign_unchanged(deltas, exp):
    assert _stream(deltas) == exp


def test_stream_still_strips_think_tags():
    # The pre-existing tag machine must keep working behind the harmony gate.
    assert _stream(["<think>secret ", "reasoning</think>", "Visible ", "answer."]) == "Visible answer."
    assert _stream(["prose <think>x</think> more"]) == "prose  more"


def test_stream_benign_analysis_colon_not_suppressed():
    # "Analysis:\n…" is a benign heading (colon, not the leak's bare-word+newline
    # shape) — it streams straight through, even when it later quotes a token.
    assert _stream(["Analysis:\n", "Revenue grew ", "12%."]) == "Analysis:\nRevenue grew 12%."
    assert _stream(["Analysis:\n", "The <|channel|> ", "token sep."]) == "Analysis:\nThe <|channel|> token sep."


# ── analysis-only leak (no final channel) must be DISCARDED, never laundered ──
@pytest.mark.parametrize("inp", [
    # A control-token head with no final channel: the model emitted only
    # reasoning (truncated / tool-called / hit max_tokens). Peeling just the
    # delimiters would ship AND persist the chain-of-thought de-tokenised.
    "<|channel|>analysis<|message|>SECRET step-by-step plan about the user.",
    "<|start|>assistant<|channel|>analysis<|message|>SECRET plan<|end|>",
    "<|channel|>commentary<|message|>SECRET musing with no answer channel.",
])
def test_analysis_only_control_head_is_discarded(inp):
    out = strip_harmony_leak(inp)
    assert out == ""
    assert "SECRET" not in out


def test_stream_analysis_only_control_head_is_discarded():
    out = _stream(["<|channel|>analysis<|mess", "age|>SECRET plan with ", "no final channel."])
    assert out == ""
    assert "SECRET" not in out


# ── case-sensitivity: capitalised headings are prose, never a channel head ────
@pytest.mark.parametrize("text", [
    # Capitalised heading that also quotes a canonical final marker (would strip
    # via the finals path if the grammar were case-insensitive).
    "Analysis\nModels emit <|channel|>final<|message|> before the answer. Note that.",
    # Capitalised heading + bare channel token later in the body.
    "Commentary\nRunning the tests now; the <|channel|> token is next.",
    "ANALYSIS\nSHOUTY HEADING about <|channel|> tokens.",
    # Title-case word that opens a sentence.
    "Thought\nI had earlier turned out to be right.",
])
def test_capitalised_heading_is_not_a_channel_head(text):
    assert strip_harmony_leak(text) == text


def test_stream_capitalised_heading_not_suppressed():
    deltas = ["Analysis\n", "Models emit <|channel|>final<|message|> ", "before the answer."]
    assert _stream(deltas) == "".join(deltas)


# ── streaming gate re-arms after flush (intra-turn retry) ─────────────────────
def test_stream_gate_rearms_after_flush():
    # Thinking-only prefill / empty-response retries flush the scrubber then
    # stream again WITHOUT reset(); the harmony gate must re-arm or the second
    # stream's leak sails through (mirrors upstream a569226f8 for the tag flag).
    s = StreamingThinkScrubber()
    s.feed("A normal first answer.")
    s.flush()
    out = s.feed("analysis\nSECRET plan for round two<channel|>Second answer.") + s.flush()
    assert out == "Second answer."
    assert "SECRET" not in out


# ── wiring: each hooked call site actually invokes the scrubber ───────────────
_LEAK = "<|channel|>analysis<|message|>hidden plan<|channel|>final<|message|>Visible answer."


def test_strip_think_blocks_call_site_is_wired():
    # agent_runtime_helpers.strip_think_blocks (the post-hoc / persisted path,
    # also reached via run_agent._strip_think_blocks) must strip harmony leaks.
    from agent.agent_runtime_helpers import strip_think_blocks
    assert strip_think_blocks(None, _LEAK) == "Visible answer."


def test_cli_strip_reasoning_tags_call_site_is_wired():
    # cli._strip_reasoning_tags (the CLI display / copy path) must strip too.
    from cli import _strip_reasoning_tags
    assert _strip_reasoning_tags(_LEAK) == "Visible answer."
