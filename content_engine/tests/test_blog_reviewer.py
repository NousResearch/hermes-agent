"""Tests for blog.blog_reviewer — editorial review layer.

The reviewer is a second opinion LLM call, independent from the writer,
that scores a draft against a strict rubric and returns claims to verify.
On model unavailability it degrades to a neutral pass (never blocks cron).
"""

import json

import blog.blog_reviewer as br


def test_review_degrades_to_pass_when_model_unavailable(monkeypatch):
    """When the LLM is unreachable, reviewer returns a neutral pass."""
    monkeypatch.setattr(br, "_call_review_llm", lambda *a, **k: None)
    draft = {
        "title": "Test post",
        "body_md": "## Section\n\nSome content here that is fine.",
        "stream": "ai",
    }
    r = br.review(draft, "ai")
    assert r["passed"] is True
    assert r["score"] == 5  # neutral midpoint
    assert r["issues"] == []
    assert r["degraded"] is True


def test_reviewer_prompt_includes_house_style_and_flow_dimensions():
    """The independent reviewer receives the same style contract as the writer."""
    prompts = br._build_rubric_prompt(
        {"title": "A technical PM article", "body_md": "## Section\n\nText."},
        "pm",
    )
    assert "## SahilBlog house style (primary editorial contract)" in prompts["system"]
    assert "flow:" in prompts["system"]
    assert "technical_pm_lens:" in prompts["system"]
    assert "material_integrity:" in prompts["system"]
    assert "formulaicness:" in prompts["system"]


def test_review_parses_valid_json(monkeypatch):
    """When the LLM returns valid JSON, the reviewer parses it."""
    raw = json.dumps({
        "score": 8,
        "passed": True,
        "issues": [],
        "claims_to_verify": [],
        "rubric": {
            "accuracy_risk": 2,
            "voice_fidelity": 8,
            "secret_sauce_leakage": 10,
            "hype_honesty": 7,
            "structure": 9,
        },
    })
    monkeypatch.setattr(br, "_call_review_llm", lambda *a, **k: raw)
    draft = {
        "title": "Test",
        "body_md": "## Section\n\nContent.",
        "stream": "ai",
    }
    r = br.review(draft, "ai")
    assert r["passed"] is True
    assert r["score"] == 8
    assert r["issues"] == []
    assert r["degraded"] is False


def test_review_flags_issues_and_claims(monkeypatch):
    """When the LLM finds issues, they flow through."""
    raw = json.dumps({
        "score": 3,
        "passed": False,
        "issues": ["Contains an em-dash on line 4", "Voice drifts mid-paragraph"],
        "claims_to_verify": ["OpenAI acquired Cursor in 2025"],
        "rubric": {
            "accuracy_risk": 7,
            "voice_fidelity": 4,
            "secret_sauce_leakage": 9,
            "hype_honesty": 5,
            "structure": 3,
        },
    })
    monkeypatch.setattr(br, "_call_review_llm", lambda *a, **k: raw)
    draft = {
        "title": "Test",
        "body_md": "## Section\n\nContent with issues.",
        "stream": "ai",
    }
    r = br.review(draft, "ai")
    assert r["passed"] is False
    assert r["score"] == 3
    assert len(r["issues"]) == 2
    assert len(r["claims_to_verify"]) == 1


def test_review_malformed_json_degrades_to_pass(monkeypatch):
    """Malformed LLM output degrades to neutral pass, never blocks."""
    monkeypatch.setattr(br, "_call_review_llm", lambda *a, **k: "not json at all")
    draft = {
        "title": "Test",
        "body_md": "## Section\n\nContent.",
        "stream": "pm",
    }
    r = br.review(draft, "pm")
    assert r["passed"] is True
    assert r["score"] == 5
    assert r["degraded"] is True


def test_review_builder_stream_checks_secret_sauce(monkeypatch):
    """Builder stream rubric includes secret-sauce leakage check."""
    raw = json.dumps({
        "score": 4,
        "passed": False,
        "issues": ["Exposes internal API key structure"],
        "claims_to_verify": [],
        "rubric": {
            "accuracy_risk": 3,
            "voice_fidelity": 6,
            "secret_sauce_leakage": 2,
            "hype_honesty": 5,
            "structure": 7,
        },
    })
    monkeypatch.setattr(br, "_call_review_llm", lambda *a, **k: raw)
    draft = {
        "title": "Builder post",
        "body_md": "## Section\n\nContent.",
        "stream": "builder",
    }
    r = br.review(draft, "builder")
    assert r["passed"] is False
    assert any("secret" in i.lower() or "internal" in i.lower() or "api" in i.lower()
               for i in r["issues"])


def test_review_filters_no_claims_placeholder(monkeypatch):
    """'No specific factual claims requiring verification' is treated as no claims.

    Models that follow the rubric but return prose instead of an empty list
    trigger a needless (and potentially failing) news-verify round-trip. The
    reviewer must normalise those placeholder strings away.
    """
    raw = json.dumps({
        "score": 8,
        "passed": True,
        "issues": [],
        "claims_to_verify": ["No specific factual claims requiring verification"],
        "rubric": {
            "accuracy_risk": 2,
            "voice_fidelity": 8,
            "secret_sauce_leakage": 10,
            "hype_honesty": 7,
            "structure": 9,
        },
    })
    monkeypatch.setattr(br, "_call_review_llm", lambda *a, **k: raw)
    draft = {
        "title": "Test",
        "body_md": "## Section\n\nContent.",
        "stream": "pm",
    }
    r = br.review(draft, "pm")
    assert r["passed"] is True
    assert r["claims_to_verify"] == []


def test_review_keeps_real_claims_with_no_prefix(monkeypatch):
    """Real claims are NOT filtered by the placeholder guard."""
    raw = json.dumps({
        "score": 7,
        "passed": True,
        "issues": [],
        "claims_to_verify": ["The 2026 AI index reports 47% enterprise adoption"],
        "rubric": {
            "accuracy_risk": 3,
            "voice_fidelity": 8,
            "secret_sauce_leakage": 10,
            "hype_honesty": 7,
            "structure": 9,
        },
    })
    monkeypatch.setattr(br, "_call_review_llm", lambda *a, **k: raw)
    draft = {
        "title": "Test",
        "body_md": "## Section\n\nContent.",
        "stream": "pm",
    }
    r = br.review(draft, "pm")
    assert len(r["claims_to_verify"]) == 1
    assert "47% enterprise adoption" in r["claims_to_verify"][0]
