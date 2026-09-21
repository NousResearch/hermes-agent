# content_engine/tests/test_educational_prompt.py
from llm_generate import build_educational_prompt, _specificity_ok


def test_prompt_has_context_and_teach_frame():
    p = build_educational_prompt(
        "sahil_twitter", pillar="Harness Tuning", platform="twitter",
        signal={"summary":"tune model routing"},
        context="switched to cheap-first fallback chain across 13 gateways",
        kb_snippets=["I always route cheap-first then fall back to a stronger model"])
    assert "teach" in p["system"].lower() or "educat" in p["system"].lower()
    assert "cheap-first" in p["user"]
    assert "Harness Tuning" in p["user"] or "harness" in p["user"].lower()

def test_specificity_rejects_generic():
    ctx = "switched to cheap-first fallback chain across 13 gateways"
    assert not _specificity_ok("AI is changing everything. Exciting times ahead.", ctx)
    assert _specificity_ok("I moved my agent to a cheap-first fallback chain. Here's why.", ctx)
