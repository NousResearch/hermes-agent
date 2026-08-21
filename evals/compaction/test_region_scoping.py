"""Region-scoping tripwire: the summarizer must only see the compacted region.

Builds a transcript with sentinel strings planted in (a) the protected head,
(b) the middle (to-be-compacted) region, and (c) the tail, mocks call_llm to
capture the prompt, and asserts head/tail sentinels never reach the
summarizer while the middle sentinel does. Runs for both legacy and lean
modes, and asserts the lean deterministic sections (anchors, verbatim users)
also carry only middle-region content.
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from agent.context_compressor import ContextCompressor  # noqa: E402

HEAD_SENTINEL = "HEADSENTINEL_zq81"
MID_SENTINEL = "MIDSENTINEL_kv93"
TAIL_SENTINEL = "TAILSENTINEL_pw27"


def _mk_transcript():
    msgs = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": f"first user message {HEAD_SENTINEL}"},
        {"role": "assistant", "content": "ack"},
    ]
    for i in range(40):
        marker = f" {MID_SENTINEL}-{i}" if i % 5 == 0 else ""
        msgs.append({
            "role": "assistant", "content": f"mid step {i}{marker}",
            "tool_calls": [{"id": f"m{i}", "function": {"name": "terminal", "arguments": "{}"}}],
        })
        msgs.append({"role": "tool", "tool_call_id": f"m{i}",
                     "content": (f"mid tool output {i} " * 300) + marker})
    for i in range(6):
        msgs.append({"role": "assistant", "content": f"tail step {i} {TAIL_SENTINEL}-{i}",
                     "tool_calls": [{"id": f"t{i}", "function": {"name": "terminal", "arguments": "{}"}}]})
        msgs.append({"role": "tool", "tool_call_id": f"t{i}", "content": f"tail output {i} {TAIL_SENTINEL}-{i}"})
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
        summary_msg = next(
            (str(m.get("content")) for m in out
             if isinstance(m.get("content"), str) and "Anchor Index" in m["content"]),
            "",
        )
        if summary_msg:
            assert TAIL_SENTINEL not in summary_msg.split("END OF CONTEXT SUMMARY")[0], \
                "[lean] tail content leaked into summary sections"
    print(f"  {tail_mode}: OK ({len(captured)} summarizer call(s), "
          f"{len(out)} msgs out)")


if __name__ == "__main__":
    for mode in ("legacy", "lean"):
        run_mode(mode)
    print("scoping tripwire: ALL PASS")
