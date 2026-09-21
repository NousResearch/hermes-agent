"""Tests for article_generator — long-form ArticleDraft assembly.

Mimics the build_educational_prompt path (single structured LLM call via
_llm_configs()). The generator reuses llm_generate.gate_post so the same
slop / em-dash / specificity / data-integrity rules apply.
"""
import article_generator as ag


def _sig(sid, prio=8, stype="harness_change", summary=None):
    return {
        "signal_id": sid,
        "signal_type": stype,
        "priority": prio,
        "summary": summary or f"signal {sid}",
        "repo": "KenseiAgent",
        "sha": "abc123",
        "variables": {"summary": summary or f"signal {sid}"},
    }


def test_build_article_prompt_includes_voice_context_and_kb(monkeypatch):
    monkeypatch.setattr(ag, "_load_voice_skill",
                        lambda brand: "SENTINEL_VOICE_BLOCK")
    plan = {"mode": "deep_dive", "signals": [_sig("a")], "pillar": "agent_build_notes",
            "title_hint": "harness tuning"}
    prompts = ag.build_article_prompt(
        "sahil_twitter", plan, context_blob="CTX_BLOB", kb_snippets=["prior-take-1"],
    )
    assert "SENTINEL_VOICE_BLOCK" in prompts["system"]
    assert "CTX_BLOB" in prompts["user"]
    assert "prior-take-1" in prompts["user"]
    # Article-specific structure rules.
    assert "## H2" in prompts["system"] or "##" in prompts["system"]
    # British English rule, no em-dash rule.
    assert "British English" in prompts["system"]
    assert "em-dash" in prompts["system"].lower()
    # Takeaway section rule (in the system prompt structure block).
    assert "try next" in prompts["system"].lower() or "takeaway" in prompts["system"].lower()


def test_x_prompt_uses_x_reach_block(monkeypatch):
    """sahil_twitter gets the X For-You reach guidance and the 'X Article' label."""
    monkeypatch.setattr(ag, "_load_voice_skill", lambda brand: "VOICE")
    plan = {"mode": "deep_dive", "signals": [_sig("a")], "pillar": "p", "title_hint": "t"}
    sysp = ag.build_article_prompt("sahil_twitter", plan, "CTX", ["k"])["system"]
    assert "X Article" in sysp
    assert "For You algorithm" in sysp
    assert "see more" not in sysp  # LinkedIn-only phrasing absent


def test_linkedin_prompt_uses_linkedin_reach_block(monkeypatch):
    """sahil_linkedin gets the LinkedIn reach guidance and the 'LinkedIn article' label."""
    monkeypatch.setattr(ag, "_load_voice_skill", lambda brand: "VOICE")
    plan = {"mode": "deep_dive", "signals": [_sig("a")], "pillar": "p", "title_hint": "t"}
    sysp = ag.build_article_prompt("sahil_linkedin", plan, "CTX", ["k"])["system"]
    assert "LinkedIn article" in sysp
    assert "earns reach on LinkedIn" in sysp
    assert "For You algorithm" not in sysp  # X-only phrasing absent
    # Depth contract is shared across both platforms.
    assert "depth and value contract" in sysp


def test_platform_for_brand():
    assert ag.platform_for("sahil_linkedin") == "linkedin"
    assert ag.platform_for("sahil_twitter") == "twitter"
    assert ag.platform_for("unknown") == "twitter"


def test_write_returns_none_when_llm_chain_dead(monkeypatch):
    monkeypatch.setattr(ag, "_call_llm_first", lambda sys_p, usr_p: None)
    monkeypatch.setattr(ag, "_load_voice_skill", lambda brand: "")
    plan = {"mode": "deep_dive", "signals": [_sig("a")], "pillar": "x", "title_hint": "t"}
    assert ag.write(plan, brand="sahil_twitter") is None


_LONG_BODY = (
    "# How I tuned the routing in the KenseiAgent content engine for real\n\n"
    "Lede paragraph that goes on long enough to be considered a real article, not a fragment. "
    "The author wants a substantive piece so this needs to be a real paragraph with context.\n\n"
    "## First section explains the problem in detail\n\n"
    "Body one with enough words to feel like a real article. I need to keep going so the word count "
    "is high enough to pass the gate. The problem is that the model picks the wrong brand voice when "
    "the prompt is short. We need to give it more context about Sahil's actual voice and the signal "
    "type. This is the kind of long-form building I do every day in the harness.\n\n"
    "## Second section shows the fix that actually worked\n\n"
    "Body two explains the fix. We restructured the prompt to inject the voice skill verbatim from "
    "the on-disk SKILL.md file instead of relying on the in-code fallback. The change took a few "
    "iterations to settle and we still tune it weekly. Numbers and tools are quoted from the real "
    "context, not made up. Real workflow, real numbers, real building.\n\n"
    "## Third section covers the data integrity check\n\n"
    "Body three covers the data integrity check. We added a regex pass that flags any number in the "
    "article body that has no counterpart in the enriched context blob. The redactor also strips API "
    "keys, bearer tokens, and `.env` style `KEY=value` patterns before anything touches the disk.\n\n"
    "## What I'd try next\n\n"
    "Next I want to thread the voice skill through to the digest mode and tune the per-section image "
    "density so the article stays under the per-month budget. A two-pass outline-then-expand step "
    "is also on the table for the quality work that comes after the live cutover.\n"
)


def test_write_returns_draft_with_required_sections(monkeypatch):
    body = _LONG_BODY
    monkeypatch.setattr(ag, "_load_voice_skill", lambda brand: "voice")
    monkeypatch.setattr(ag, "_call_llm_first",
                        lambda sys_p, usr_p: body)
    monkeypatch.setattr(ag, "enrich_signal", lambda s: "context-blob")
    monkeypatch.setattr(ag, "retrieve_kb", lambda topic, limit=3: ["kb-1"])
    plan = {"mode": "deep_dive", "signals": [_sig("a")], "pillar": "x",
            "title_hint": "How I tuned the routing"}
    out = ag.write(plan, brand="sahil_twitter")
    assert out is not None
    for key in ("title", "body_md", "mode", "pillar", "slug", "signals", "context"):
        assert key in out, f"missing key: {key}"
    assert out["title"] == "How I tuned the routing in the KenseiAgent content engine for real"
    assert "## What I'd try next" in out["body_md"]
    assert out["slug"]  # non-empty kebab-case


def test_article_uses_voice_skill(monkeypatch):
    """The system prompt embeds the brand voice skill verbatim."""
    seen = {}
    monkeypatch.setattr(ag, "_load_voice_skill",
                        lambda brand: seen.setdefault("v", "VOICE_OK"))
    body = _LONG_BODY
    monkeypatch.setattr(ag, "_call_llm_first", lambda s, u: body)
    monkeypatch.setattr(ag, "enrich_signal", lambda s: "ctx")
    monkeypatch.setattr(ag, "retrieve_kb", lambda t, limit=3: [])
    plan = {"mode": "deep_dive", "signals": [_sig("a")], "pillar": "x", "title_hint": "t"}
    out = ag.write(plan, brand="sahil_twitter")
    assert out is not None
    assert seen.get("v") == "VOICE_OK"


def test_article_dry_run_emits_marker_file(tmp_path, monkeypatch):
    """dry_run=True writes a dryrun marker to a tmp dir; no LLM call."""
    monkeypatch.setattr(ag, "_load_voice_skill", lambda brand: "v")
    called = {"n": 0}
    def fake_call(s, u):
        called["n"] += 1
        return "# Body\n\nLede.\n\n## A\n\nx.\n\n## B\n\ny.\n\n## C\n\nz.\n\n## What I'd try next\n\nt.\n"
    monkeypatch.setattr(ag, "_call_llm_first", fake_call)
    monkeypatch.setattr(ag, "enrich_signal", lambda s: "ctx")
    monkeypatch.setattr(ag, "retrieve_kb", lambda t, limit=3: [])
    plan = {"mode": "deep_dive", "signals": [_sig("a")], "pillar": "x", "title_hint": "t"}
    out = ag.write(plan, brand="sahil_twitter", dry_run=True, dryrun_dir=tmp_path)
    assert out is not None
    assert out.get("dryrun") is True
    # Marker file path returned.
    assert "marker" in out or out.get("dryrun_path")
    # No LLM calls were made.
    assert called["n"] == 0
