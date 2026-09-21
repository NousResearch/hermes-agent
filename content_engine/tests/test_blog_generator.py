"""Tests for blog.blog_generator — stream-aware long-form draft generation.

Reuses article_generator building blocks (build_article_prompt shape,
_call_llm_first, _extract_title, enrich_signal, retrieve_kb) but injects
the stream voice, word_target, and section_target into the system prompt.
Output is a draft dict with the blog frontmatter fields set from the stream.
"""
import blog.blog_generator as bg
from blog.blog_streams import STREAMS


def test_wiki_home_defaults_to_canonical_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("WIKI_PATH", raising=False)

    assert bg._wiki_home() == tmp_path / "docs" / "wiki"


def test_wiki_home_override_wins(monkeypatch, tmp_path):
    override = tmp_path / "override-wiki"
    monkeypatch.setenv("WIKI_PATH", str(override))

    assert bg._wiki_home() == override


_FAKE_BODY = """# Token-Maxing at the Edge

A counterintuitive claim grounded in concrete figures.

## The mechanism

The numbers tell a different story than the hype.

## Worked example

Here is the code that changed everything.

## The trade-offs

Every optimisation has a cost.

## What breaks

The failure modes are instructive.

## How to apply this

Step by step so you can do it too.

## What I'd try next

The takeaway is simple.
"""


def test_build_blog_prompt_includes_stream_voice(monkeypatch):
    """The stream voice is injected into the system prompt."""
    plan = {"topic_id": "t1", "title_hint": "token-maxing",
            "tags": ["ai-adoption"], "source": "research-paper",
            "signals": [{"signal_id": "t1", "summary": "token-maxing"}]}
    prompts = bg.build_blog_prompt("ai", plan, context_blob="CTX", kb_snippets=["k1"])
    assert STREAMS["ai"]["voice"][:30] in prompts["system"]
    assert "CTX" in prompts["user"]
    assert "k1" in prompts["user"]


def test_build_blog_prompt_includes_house_style_without_ritual_ending():
    """The shared style is primary and the old fixed ending is not mandatory."""
    plan = {"topic_id": "t1", "title_hint": "technical PM angle",
            "tags": [], "source": "manual",
            "signals": [{"signal_id": "t1", "summary": "a real decision"}]}
    prompts = bg.build_blog_prompt("ai", plan, context_blob="", kb_snippets=[])
    assert "## SahilBlog house style (primary editorial contract)" in prompts["system"]
    assert "The writing can discuss APIs, webhooks, queues" in prompts["system"]
    assert "One final `## What I'd try next` section" not in prompts["system"]
    assert "let the article form follow the material" in prompts["system"]


def test_build_blog_prompt_includes_approved_idea_editorial_brief():
    """A purpose-led intake brief reaches the writer instead of dying in routing."""
    plan = {
        "topic_id": "t1", "title_hint": "token-maxing", "tags": [],
        "source": "research-paper", "signals": [],
        "editorial_brief": {
            "post_thesis": "The thesis", "concrete_takeaway": "Do this",
            "evidence_anchor": "PR #42", "gap_claim": "Different angle",
            "stream_format_rationale": "AI essay",
        },
    }
    prompts = bg.build_blog_prompt("ai", plan, context_blob="", kb_snippets=[])
    for value in plan["editorial_brief"].values():
        assert value in prompts["user"]


def test_build_blog_prompt_includes_word_and_section_target(monkeypatch):
    """The stream word_target and section_target are in the system prompt."""
    plan = {"topic_id": "t1", "title_hint": "t", "tags": [], "source": "manual",
            "signals": [{"signal_id": "t1", "summary": "t"}]}
    prompts = bg.build_blog_prompt("builder", plan, "", [])
    assert str(STREAMS["builder"]["word_target"]) in prompts["system"]
    assert str(STREAMS["builder"]["section_target"]) in prompts["system"]


def test_write_returns_draft_with_stream_fields(monkeypatch):
    """write() returns a draft with tier, tags, source, format set from the stream."""
    plan = {"topic_id": "t1", "title_hint": "Token-maxing at the edge",
            "tags": ["ai-adoption"], "source": "research-paper",
            "signals": [{"signal_id": "t1", "summary": "token-maxing"}]}
    monkeypatch.setattr(bg, "_call_llm_first", lambda sys, usr: _FAKE_BODY)
    monkeypatch.setattr(bg, "_load_voice_skill", lambda brand: "VOICE")
    monkeypatch.setattr(bg, "enrich_signal", lambda s: "CTX")
    monkeypatch.setattr(bg, "retrieve_kb", lambda t, limit=3: [])
    draft = bg.write(plan, stream="ai")
    assert draft is not None
    assert draft["title"] == "Token-Maxing at the Edge"
    assert draft["tier"] == "ai"  # from STREAMS["ai"], 3-tier system
    assert "ai" in draft["tags"]  # base tag
    assert "ai-adoption" in draft["tags"]  # topic tag
    assert draft["source"] == "research-paper"
    assert draft["format"] == "essay"
    assert draft["stream"] == "ai"
    assert draft["description"]  # non-empty deck


def test_write_has_enough_h2_sections(monkeypatch):
    """The body has >= section_target H2 sections (the illustrator keys off these)."""
    plan = {"topic_id": "t1", "title_hint": "t", "tags": [], "source": "manual",
            "signals": [{"signal_id": "t1", "summary": "t"}]}
    monkeypatch.setattr(bg, "_call_llm_first", lambda sys, usr: _FAKE_BODY)
    monkeypatch.setattr(bg, "_load_voice_skill", lambda brand: "VOICE")
    monkeypatch.setattr(bg, "enrich_signal", lambda s: "CTX")
    monkeypatch.setattr(bg, "retrieve_kb", lambda t, limit=3: [])
    draft = bg.write(plan, stream="ai")
    h2_count = sum(1 for line in draft["body_md"].splitlines() if line.startswith("## "))
    assert h2_count >= STREAMS["ai"]["section_target"]


def test_write_returns_none_on_llm_dead(monkeypatch):
    """write() returns None when the LLM chain is dead (no body)."""
    plan = {"topic_id": "t1", "title_hint": "t", "tags": [], "source": "manual",
            "signals": [{"signal_id": "t1", "summary": "t"}]}
    monkeypatch.setattr(bg, "_call_llm_first", lambda sys, usr: None)
    monkeypatch.setattr(bg, "_load_voice_skill", lambda brand: "VOICE")
    monkeypatch.setattr(bg, "enrich_signal", lambda s: "CTX")
    monkeypatch.setattr(bg, "retrieve_kb", lambda t, limit=3: [])
    assert bg.write(plan, stream="ai") is None


def test_write_returns_none_when_plan_has_no_signals(monkeypatch):
    plan = {"topic_id": "t1", "title_hint": "t", "tags": [], "source": "manual",
            "signals": []}
    assert bg.write(plan, stream="ai") is None


def test_write_extracts_title_from_body(monkeypatch):
    """The title comes from the first # H1 in the body."""
    plan = {"topic_id": "t1", "title_hint": "WRONG HINT", "tags": [],
            "source": "manual", "signals": [{"signal_id": "t1", "summary": "s"}]}
    monkeypatch.setattr(bg, "_call_llm_first", lambda sys, usr: _FAKE_BODY)
    monkeypatch.setattr(bg, "_load_voice_skill", lambda brand: "VOICE")
    monkeypatch.setattr(bg, "enrich_signal", lambda s: "CTX")
    monkeypatch.setattr(bg, "retrieve_kb", lambda t, limit=3: [])
    draft = bg.write(plan, stream="pm")
    assert draft["title"] == "Token-Maxing at the Edge"  # from body, not title_hint


def test_write_generates_description(monkeypatch):
    """The draft includes a one-line description (deck) from the lede."""
    plan = {"topic_id": "t1", "title_hint": "t", "tags": [], "source": "manual",
            "signals": [{"signal_id": "t1", "summary": "s"}]}
    body = _FAKE_BODY.replace("A counterintuitive claim grounded in concrete figures.",
                              "After six months tuning KENSEI, the real wins came from cutting prompts not adding agents.")
    monkeypatch.setattr(bg, "_call_llm_first", lambda sys, usr: body)
    monkeypatch.setattr(bg, "_load_voice_skill", lambda brand: "VOICE")
    monkeypatch.setattr(bg, "enrich_signal", lambda s: "CTX")
    monkeypatch.setattr(bg, "retrieve_kb", lambda t, limit=3: [])
    draft = bg.write(plan, stream="builder")
    assert draft["description"]
    assert len(draft["description"]) <= 200


def test_write_threads_retry_feedback_into_prompt(monkeypatch):
    """When retry_feedback is set, it appears in the system prompt sent to the LLM."""
    plan = {"topic_id": "t1", "title_hint": "t", "tags": [], "source": "manual",
            "signals": [{"signal_id": "t1", "summary": "s"}]}
    captured = []
    def capture_llm(sys_prompt, usr_prompt):
        captured.append((sys_prompt, usr_prompt))
        return _FAKE_BODY
    monkeypatch.setattr(bg, "_call_llm_first", capture_llm)
    monkeypatch.setattr(bg, "_load_voice_skill", lambda brand: "VOICE")
    monkeypatch.setattr(bg, "enrich_signal", lambda s: "CTX")
    monkeypatch.setattr(bg, "retrieve_kb", lambda t, limit=3: [])
    bg.write(plan, stream="ai", retry_feedback="too short, needs 1700 words")
    assert len(captured) == 1
    assert "too short, needs 1700 words" in captured[0][0], \
        "retry_feedback must be threaded into the system prompt"


def test_write_with_gate_retry_uses_feedback_not_double_spend(monkeypatch):
    """On gate failure, write_with_gate retries once with feedback in a single
    LLM call, not two calls with the feedback discarded."""
    plan = {"topic_id": "t1", "title_hint": "t", "tags": [], "source": "manual",
            "signals": [{"signal_id": "t1", "summary": "s"}]}
    call_count = {"n": 0}
    prompts_seen = []
    def fake_llm(sys_prompt, usr_prompt):
        call_count["n"] += 1
        prompts_seen.append(sys_prompt)
        return _FAKE_BODY
    monkeypatch.setattr(bg, "_call_llm_first", fake_llm)
    monkeypatch.setattr(bg, "_load_voice_skill", lambda brand: "VOICE")
    monkeypatch.setattr(bg, "enrich_signal", lambda s: "CTX")
    monkeypatch.setattr(bg, "retrieve_kb", lambda t, limit=3: [])

    # Mock the reviewer to always pass so it does not interfere with the
    # deterministic-gate retry test.
    import blog.blog_reviewer as br
    monkeypatch.setattr(br, "_call_review_llm", lambda *a, **k: None)

    # Force gate to fail first time, pass second time.
    fail_first = {"fail": True}
    import article_gates as ag
    real_check = ag.check
    def fake_check(draft):
        if fail_first["fail"]:
            fail_first["fail"] = False
            return ag.GateResult(passed=False, issues=["too short"],
                                redacted_body=draft.get("body_md", ""),
                                redacted_context="", slop_score=8)
        return real_check(draft)
    monkeypatch.setattr(ag, "check", fake_check)
    monkeypatch.setattr(bg, "gate_check",
                        lambda d: (("ok" if not fail_first["fail"] else "fail"),
                                   ["too short"] if fail_first["fail"] else []))

    result = bg.write_with_gate(plan, stream="ai")
    # Only 2 LLM calls: one for the initial draft, one for the retry with feedback.
    assert call_count["n"] == 2, f"expected 2 LLM calls, got {call_count['n']}"
    # The second call's prompt must contain the feedback.
    assert "too short" in prompts_seen[1], \
        "retry prompt must contain the gate feedback"


def test_write_with_gate_populates_diagnostics_on_permanent_gate_failure(monkeypatch):
    """When the retry also fails gate/review, diagnostics carries reason + issues
    so the caller (blog_pipeline) can track and surface why, instead of a bare None."""
    plan = {"topic_id": "t1", "title_hint": "t", "tags": [], "source": "manual",
            "signals": [{"signal_id": "t1", "summary": "s"}]}
    monkeypatch.setattr(bg, "_call_llm_first", lambda sp, up: _FAKE_BODY)
    monkeypatch.setattr(bg, "_load_voice_skill", lambda brand: "VOICE")
    monkeypatch.setattr(bg, "enrich_signal", lambda s: "CTX")
    monkeypatch.setattr(bg, "retrieve_kb", lambda t, limit=3: [])

    import blog.blog_reviewer as br
    monkeypatch.setattr(br, "_call_review_llm", lambda *a, **k: None)

    # Gate fails on every attempt (initial and retry).
    monkeypatch.setattr(bg, "gate_check", lambda d: ("fail", ["persistent slop issue"]))

    diagnostics: dict = {}
    result = bg.write_with_gate(plan, stream="ai", diagnostics=diagnostics)
    assert result is None
    assert diagnostics.get("reason"), "diagnostics must explain why write_with_gate gave up"
    assert "persistent slop issue" in diagnostics.get("issues", [])


def test_write_with_gate_runs_reviewer_and_retries(monkeypatch):
    """write_with_gate runs the editorial reviewer and retries on its issues."""
    plan = {"topic_id": "t1", "title_hint": "t", "tags": [], "source": "manual",
            "signals": [{"signal_id": "t1", "summary": "s"}]}
    call_count = {"n": 0}
    def fake_llm(sys_prompt, usr_prompt):
        call_count["n"] += 1
        return _FAKE_BODY
    monkeypatch.setattr(bg, "_call_llm_first", fake_llm)
    monkeypatch.setattr(bg, "_load_voice_skill", lambda brand: "VOICE")
    monkeypatch.setattr(bg, "enrich_signal", lambda s: "CTX")
    monkeypatch.setattr(bg, "retrieve_kb", lambda t, limit=3: [])

    # Mock the deterministic gate to always pass so we isolate the reviewer.
    monkeypatch.setattr(bg, "gate_check", lambda d: ("ok", []))

    # Mock _redact_draft to avoid article_gates re-checking the short body.
    monkeypatch.setattr(bg, "_redact_draft", lambda d: None)

    # Reviewer: fail first attempt, pass on retry.
    import blog.blog_reviewer as br
    import json as _json
    review_calls = {"n": 0}
    def fake_review_llm(*a, **k):
        review_calls["n"] += 1
        if review_calls["n"] == 1:
            return _json.dumps({
                "score": 3, "passed": False,
                "issues": ["Contains an em-dash on line 4"],
                "claims_to_verify": [],
                "rubric": {"accuracy_risk": 5, "voice_fidelity": 4,
                           "secret_sauce_leakage": 8, "hype_honesty": 5,
                           "structure": 3},
            })
        return _json.dumps({
            "score": 8, "passed": True, "issues": [], "claims_to_verify": [],
            "rubric": {"accuracy_risk": 8, "voice_fidelity": 8,
                       "secret_sauce_leakage": 10, "hype_honesty": 8,
                       "structure": 8},
        })
    monkeypatch.setattr(br, "_call_review_llm", fake_review_llm)

    result = bg.write_with_gate(plan, stream="ai")
    assert result is not None, "should pass on retry"
    assert call_count["n"] == 2, "should make 2 LLM writer calls"
    assert review_calls["n"] == 2, "should make 2 reviewer calls"


def test_write_with_gate_strict_raises_on_degraded(monkeypatch):
    """strict_review=True + degraded reviewer -> raises ReviewUnavailable."""
    plan = {"topic_id": "t1", "title_hint": "t", "tags": [], "source": "manual",
            "signals": [{"signal_id": "t1", "summary": "s"}]}
    monkeypatch.setattr(bg, "_call_llm_first", lambda sys, usr: _FAKE_BODY)
    monkeypatch.setattr(bg, "_load_voice_skill", lambda brand: "VOICE")
    monkeypatch.setattr(bg, "enrich_signal", lambda s: "CTX")
    monkeypatch.setattr(bg, "retrieve_kb", lambda t, limit=3: [])
    monkeypatch.setattr(bg, "gate_check", lambda d: ("ok", []))
    monkeypatch.setattr(bg, "_redact_draft", lambda d: None)

    import blog.blog_reviewer as br
    monkeypatch.setattr(br, "_call_review_llm", lambda *a, **k: None)

    import pytest
    with pytest.raises(bg.ReviewUnavailable, match="degraded"):
        bg.write_with_gate(plan, stream="ai", strict_review=True)


def test_write_with_gate_non_strict_returns_draft_on_degraded(monkeypatch):
    """strict_review=False + degraded reviewer -> returns draft (graceful)."""
    plan = {"topic_id": "t1", "title_hint": "t", "tags": [], "source": "manual",
            "signals": [{"signal_id": "t1", "summary": "s"}]}
    monkeypatch.setattr(bg, "_call_llm_first", lambda sys, usr: _FAKE_BODY)
    monkeypatch.setattr(bg, "_load_voice_skill", lambda brand: "VOICE")
    monkeypatch.setattr(bg, "enrich_signal", lambda s: "CTX")
    monkeypatch.setattr(bg, "retrieve_kb", lambda t, limit=3: [])
    monkeypatch.setattr(bg, "gate_check", lambda d: ("ok", []))
    monkeypatch.setattr(bg, "_redact_draft", lambda d: None)

    import blog.blog_reviewer as br
    monkeypatch.setattr(br, "_call_review_llm", lambda *a, **k: None)

    result = bg.write_with_gate(plan, stream="ai", strict_review=False)
    assert result is not None, "graceful mode should return the draft"


def test_write_with_gate_strict_returns_draft_on_genuine_pass(monkeypatch):
    """strict_review=True + genuine reviewer pass -> returns draft."""
    plan = {"topic_id": "t1", "title_hint": "t", "tags": [], "source": "manual",
            "signals": [{"signal_id": "t1", "summary": "s"}]}
    monkeypatch.setattr(bg, "_call_llm_first", lambda sys, usr: _FAKE_BODY)
    monkeypatch.setattr(bg, "_load_voice_skill", lambda brand: "VOICE")
    monkeypatch.setattr(bg, "enrich_signal", lambda s: "CTX")
    monkeypatch.setattr(bg, "retrieve_kb", lambda t, limit=3: [])
    monkeypatch.setattr(bg, "gate_check", lambda d: ("ok", []))
    monkeypatch.setattr(bg, "_redact_draft", lambda d: None)

    import blog.blog_reviewer as br
    import json as _json
    monkeypatch.setattr(br, "_call_review_llm", lambda *a, **k: _json.dumps({
        "score": 8, "passed": True, "issues": [], "claims_to_verify": [],
        "rubric": {"accuracy_risk": 8, "voice_fidelity": 8,
                   "secret_sauce_leakage": 10, "hype_honesty": 8,
                   "structure": 8},
    }))

    result = bg.write_with_gate(plan, stream="ai", strict_review=True)
    assert result is not None, "strict mode + genuine pass should return draft"


# -- Approach A research-roundup prompt ---------------------------------------

def test_research_build_blog_prompt_is_roundup_shaped():
    """The research stream emits a roundup-shaped prompt, not the essay skeleton."""
    plan = {"topic_id": "t1", "title_hint": "agent memory roundup",
            "tags": ["research"], "source": "curated-roundup",
            "signals": [{"signal_id": "t1", "summary": "memory systems"}]}
    prompts = bg.build_blog_prompt("research", plan, context_blob="CTX", kb_snippets=["k1"])
    assert "curated-research roundup" in prompts["system"]
    assert "Approach A lane" in prompts["system"]
    # Roundup shape, not essay shape.
    assert "roundup shape, NOT essay shape" in prompts["system"]
    # Numbered entries target threaded from STREAMS config.
    assert "Exactly 5 numbered" in prompts["system"]
    # The mandatory recurring container is present.
    assert "(a) the finding in plain English" in prompts["system"]
    assert "(e) honest limitations" in prompts["system"]
    # Takeaways + try-next sections mandated.
    assert "## Takeaways" in prompts["system"]
    assert "## What I'd try next" in prompts["system"]
    # Article-plus-social packaging rule present.
    assert "self-contained X/LinkedIn post" in prompts["system"]
    # Reuses voice, depth contract, and context.
    assert STREAMS["research"]["voice"][:30] in prompts["system"]
    assert "CTX" in prompts["user"]
    assert "k1" in prompts["user"]


def test_research_build_blog_prompt_uses_stream_word_target():
    """The research prompt threads the stream's word_target and section_target."""
    plan = {"topic_id": "t1", "title_hint": "t", "tags": [], "source": "manual",
            "signals": [{"signal_id": "t1", "summary": "s"}]}
    prompts = bg.build_blog_prompt("research", plan, "", [])
    assert str(STREAMS["research"]["word_target"]) in prompts["system"]
    assert str(STREAMS["research"]["section_target"]) in prompts["system"]


def test_research_build_blog_prompt_injects_verification_warning():
    """An unverified claim is reframed in the research user prompt."""
    plan = {"topic_id": "t1", "title_hint": "t", "tags": [], "source": "manual",
            "signals": [{"signal_id": "t1", "summary": "s"}]}
    prompts = bg.build_blog_prompt(
        "research", plan, "", [],
        verification={"query": "ACME acquisition", "verified": False},
    )
    assert "UNVERIFIED" in prompts["user"]
    assert "ACME acquisition" in prompts["user"]


def test_research_build_blog_prompt_threads_retry_feedback():
    """Retry feedback reaches the research user prompt."""
    plan = {"topic_id": "t1", "title_hint": "t", "tags": [], "source": "manual",
            "signals": [{"signal_id": "t1", "summary": "s"}]}
    prompts = bg.build_blog_prompt(
        "research", plan, "", [], retry_feedback="too short, needs 1500 words")
    assert "too short, needs 1500 words" in prompts["user"]


def test_research_write_sets_roundup_frontmatter(monkeypatch):
    """write() on the research stream emits tier/source/format from STREAMS."""
    plan = {"topic_id": "t1", "title_hint": "Agent memory roundup",
            "tags": ["research"], "source": "curated-roundup",
            "signals": [{"signal_id": "t1", "summary": "memory systems"}]}
    monkeypatch.setattr(bg, "_call_llm_first", lambda sys, usr: _FAKE_BODY)
    monkeypatch.setattr(bg, "_load_voice_skill", lambda brand: "VOICE")
    monkeypatch.setattr(bg, "enrich_signal", lambda s: "CTX")
    monkeypatch.setattr(bg, "retrieve_kb", lambda t, limit=3: [])
    monkeypatch.setattr(bg, "_wiki_context_for", lambda t, max_results=2: [])
    draft = bg.write(plan, stream="research")
    assert draft is not None
    assert draft["tier"] == "research"          # engine-level value (clamped downstream)
    assert draft["source"] == "curated-roundup"
    assert draft["format"] == "roundup"
    assert draft["stream"] == "research"
    assert "research" in draft["tags"]