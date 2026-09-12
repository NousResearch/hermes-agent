"""Region-scoping tripwire: the summarizer must only see the compacted region.

Builds a transcript with sentinel strings planted in (a) the protected head,
(b) the middle (to-be-compacted) region, and (c) the tail, mocks call_llm to
capture the prompt, and asserts head/tail sentinels never reach the
summarizer while the middle sentinel does. Runs for both legacy and lean
modes, and asserts the lean deterministic sections (anchors, verbatim users)
also carry only middle-region content.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from agent.context_compressor import ContextCompressor  # noqa: E402

HEAD_SENTINEL = "HEADSENTINEL_zq81"
MID_SENTINEL = "MIDSENTINEL_kv93"
TAIL_SENTINEL = "TAILSENTINEL_pw27"

# Anchorable identifiers planted in each region so we can verify the anchor
# index carries ONLY middle-region identifiers. The patterns match
# _ANCHOR_PATTERNS (PRs, commits, files, errors).
HEAD_PR = "#99001"
MID_PR = "#99002"
TAIL_PR = "#99003"
HEAD_SHA = "abc1234567890def"
MID_SHA = "fed0987654321cba"
TAIL_SHA = "1234567890abcdef"
HEAD_FILE = "tests/head_guard.py"
MID_FILE = "tests/mid_guard.py"
TAIL_FILE = "tests/tail_guard.py"
HEAD_ERROR = "HeadError: boom head"
MID_ERROR = "MidError: boom mid"
TAIL_ERROR = "TailError: boom tail"


def _mk_transcript():
    msgs = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": f"first user message {HEAD_SENTINEL} {HEAD_PR} {HEAD_FILE}"},
        {"role": "assistant", "content": f"ack {HEAD_ERROR}"},
    ]
    for i in range(40):
        marker = f" {MID_SENTINEL}-{i}" if i % 5 == 0 else ""
        # Every 10th turn plants a REAL user message in the middle region
        # carrying a PR, SHA, file path, and error string — the deterministic
        # verbatim-user and anchor sections MUST carry these, and ONLY these.
        if i % 10 == 0 and i > 0:
            msgs.append({
                "role": "user",
                "content": (
                    f"middle user instruction {MID_SENTINEL}-user-{i} "
                    f"{MID_PR} sha={MID_SHA} file={MID_FILE} err={MID_ERROR}"
                ),
            })
        msgs.append({
            "role": "assistant",
            "content": f"mid step {i}{marker}",
            "tool_calls": [{"id": f"m{i}", "function": {"name": "terminal", "arguments": "{}"}}],
        })
        msgs.append({"role": "tool", "tool_call_id": f"m{i}",
                     "content": (f"mid tool output {i} " * 300) + marker})
    for i in range(6):
        msgs.append({
            "role": "assistant",
            "content": f"tail step {i} {TAIL_SENTINEL}-{i} {TAIL_PR} {TAIL_FILE}",
            "tool_calls": [{"id": f"t{i}", "function": {"name": "terminal", "arguments": "{}"}}],
        })
        msgs.append({"role": "tool", "tool_call_id": f"t{i}",
                     "content": f"tail output {i} {TAIL_SENTINEL}-{i} {TAIL_ERROR}"})
    msgs.append({"role": "user", "content": f"latest user question {TAIL_SENTINEL}-u"})
    msgs.append({"role": "assistant", "content": "final answer in tail"})
    return msgs


def run_mode(tail_mode: str):
    captured = []

    def fake_call_llm(messages=None, **kw):
        captured.append(messages[0]["content"] if messages else "")
        resp = MagicMock()
        resp.choices[0].message.content = "## Active Task\nsummarized"
        return resp

    comp = ContextCompressor(model="anthropic/claude-fable-5", quiet_mode=True,
                             tail_mode=tail_mode)
    comp.tail_token_budget = 3_000  # force a real middle on the small fixture
    comp._session_id = "scope-test"
    msgs = _mk_transcript()
    with patch("agent.context_compressor.call_llm", side_effect=fake_call_llm), \
         patch("agent.auxiliary_client.call_llm", side_effect=fake_call_llm):
        out = comp.compress(msgs, current_tokens=200_000, force=True)

    all_prompts = "\n".join(captured)
    assert captured, f"[{tail_mode}] summarizer never called"
    assert MID_SENTINEL in all_prompts, f"[{tail_mode}] middle region missing from summarizer input"
    # Head/tail user messages MAY appear inside the FOCUS TOPIC steering block
    # (intentional: tells the summarizer what the user currently cares about).
    # They must NOT appear in the serialized TURNS body being summarized.
    for p in captured:
        body = p.split("FOCUS TOPIC:")[0]
        assert TAIL_SENTINEL not in body, f"[{tail_mode}] TAIL leaked into summarized turns"
        assert HEAD_SENTINEL not in body, f"[{tail_mode}] protected HEAD leaked into summarized turns"

    # The tail must survive verbatim; the head user message must survive.
    out_text = "\n".join(str(m.get("content")) for m in out)
    assert f"{TAIL_SENTINEL}-u" in out_text, f"[{tail_mode}] latest user message lost"
    assert HEAD_SENTINEL in out_text, f"[{tail_mode}] head lost"

    if tail_mode == "lean":
        # Lean compaction makes exactly ONE auxiliary LLM call per attempt
        # (the main summary request — there is no per-chunk digest loop).
        assert len(captured) == 1, (
            f"[lean] expected exactly 1 summarizer call, got {len(captured)}"
        )

        # The summary MUST contain the deterministic lean sections.
        summary_msg = next(
            (str(m.get("content")) for m in out
             if isinstance(m.get("content"), str) and "Anchor Index" in m["content"]),
            "",
        )
        assert summary_msg, "[lean] Anchor Index section missing from summary"
        assert _has_middle_user_msgs(summary_msg), (
            "[lean] verbatim user messages section missing from summary"
        )

        # Anchor index carries ONLY middle-region identifiers.
        assert MID_PR in summary_msg, f"[lean] middle PR {MID_PR} missing from anchor index"
        assert MID_SHA in summary_msg, f"[lean] middle SHA {MID_SHA} missing from anchor index"
        assert MID_FILE in summary_msg, f"[lean] middle file {MID_FILE} missing from anchor index"
        assert HEAD_PR not in summary_msg, f"[lean] head PR {HEAD_PR} leaked into anchor index"
        assert HEAD_SHA not in summary_msg, f"[lean] head SHA {HEAD_SHA} leaked into anchor index"
        assert HEAD_FILE not in summary_msg, f"[lean] head file {HEAD_FILE} leaked into anchor index"
        assert TAIL_PR not in summary_msg, f"[lean] tail PR {TAIL_PR} leaked into anchor index"
        assert TAIL_SHA not in summary_msg, f"[lean] tail SHA {TAIL_SHA} leaked into anchor index"
        assert TAIL_FILE not in summary_msg, f"[lean] tail file {TAIL_FILE} leaked into anchor index"

        # Verbatim user section carries ONLY middle-region user messages.
        head_frag = summary_msg.split("## User Messages")
        assert head_frag, "[lean] User Messages section missing"
        # Head user message MAY appear in FOCUS TOPIC (intentional) but NOT
        # inside the verbatim user quote block.
        for qblock in _extract_quote_blocks(summary_msg):
            assert HEAD_SENTINEL not in qblock, "[lean] head user msg leaked into verbatim quotes"
            assert TAIL_SENTINEL not in qblock, "[lean] tail user msg leaked into verbatim quotes"

        # Tail must NOT leak into any part of the summary sections.
        assert TAIL_SENTINEL not in summary_msg.split("END OF CONTEXT SUMMARY")[0], \
            "[lean] tail content leaked into summary sections"
    print(f"  {tail_mode}: OK ({len(captured)} summarizer call(s), "
          f"{len(out)} msgs out)")


def _has_middle_user_msgs(summary_msg: str) -> bool:
    return "MIDSENTINEL_kv93-user" in summary_msg


def _extract_quote_blocks(summary_msg: str) -> list[str]:
    """Return each '> '-prefixed quote block from the verbatim user section."""
    blocks: list[str] = []
    current: list[str] = []
    for line in summary_msg.split("\n"):
        if line.startswith("> "):
            current.append(line)
        elif current:
            blocks.append("\n".join(current))
            current = []
    if current:
        blocks.append("\n".join(current))
    return blocks


if __name__ == "__main__":
    for mode in ("legacy", "lean"):
        run_mode(mode)
    print("scoping tripwire: ALL PASS")
