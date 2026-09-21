"""Tests for the reviewer-evidence wiring.

- _verify_claims_detailed returns warnings for unverified claims AND
  evidence lines (claim + source) for verified ones.
- _build_rubric_prompt exposes grounding context and verified sources to
  the reviewer so material_integrity judgements can be made fairly.
"""
import blog.news_verify as nv
from blog.blog_generator import _verify_claims, _verify_claims_detailed
from blog.blog_reviewer import _build_rubric_prompt


def test_verify_claims_detailed_mix(monkeypatch):
    def fake_verify(claim):
        if "verified-fact" in claim:
            return {"verified": True,
                    "snippets": [{"title": "Docs", "url": "https://docs.example/x"}]}
        return {"verified": False, "snippets": []}

    monkeypatch.setattr(nv, "verify_event", fake_verify)
    warnings, evidence = _verify_claims_detailed(
        ["verified-fact about retries", "wild unverified claim"])
    assert len(warnings) == 1 and "UNVERIFIED" in warnings[0]
    assert len(evidence) == 1 and "https://docs.example/x" in evidence[0]
    # Back-compat wrapper returns only warnings.
    assert _verify_claims(["verified-fact about retries"]) == []


def test_rubric_prompt_includes_grounding():
    draft = {
        "title": "T", "body_md": "Body text.", "stream": "ai",
        "context": "Topic context blob with mechanisms.",
        "kb_snippets": ["kb one", "kb two"],
        "signals": [{"summary": "signal summary here"}],
        "verified_sources": ["'claim X' — verified via https://src.example/y"],
    }
    prompts = _build_rubric_prompt(draft, "ai")
    user = prompts["user"]
    assert "Supplied grounding context" in user
    assert "Topic context blob" in user
    assert "kb one" in user
    assert "signal summary here" in user
    assert "Web-verified sources" in user
    assert "https://src.example/y" in user
    # And the body itself is still present.
    assert "Body text." in user


def test_rubric_prompt_without_context_is_unchanged_shape():
    draft = {"title": "T", "body_md": "Body only.", "stream": "ai"}
    prompts = _build_rubric_prompt(draft, "ai")
    assert "Supplied grounding context" not in prompts["user"]
    assert "Body only." in prompts["user"]


def test_write_with_gate_re_reviews_same_draft_when_all_claims_verify(monkeypatch):
    import blog.blog_generator as G
    import blog.blog_reviewer as R

    draft = {"title": "T", "body_md": "Body.", "slug": "t",
             "context": "ctx", "kb_snippets": [], "signals": [{"summary": "s"}]}

    monkeypatch.setattr(G, "write", lambda *a, **k: dict(draft))
    monkeypatch.setattr(G, "gate_check", lambda d: ("ok", []))
    monkeypatch.setattr(G, "ground_post",
                        lambda d, stream="ai": {"body_md": d.get("body_md", ""),
                                                "dead_links": []})
    monkeypatch.setattr(G, "_verify_claims_detailed",
                        lambda claims: ([], ["'claim X' — verified via https://src.example/y"]))
    monkeypatch.setattr(G, "_redact_draft", lambda d: None)

    calls = {"n": 0}

    def fake_review(d, stream):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"passed": False, "score": 7,
                    "issues": ["Named event unverified"],
                    "claims_to_verify": ["claim X"], "degraded": False}
        assert d.get("verified_sources"), "evidence must be attached to re-review"
        return {"passed": True, "score": 8, "issues": [],
                "claims_to_verify": [], "degraded": False}

    monkeypatch.setattr(R, "review", fake_review)

    out = G.write_with_gate({"topic_id": "t", "signals": [{"summary": "s"}]},
                            stream="ai", case_study_exempt=True)
    assert out is not None and out.get("title") == "T"
    assert calls["n"] == 2, "expected exactly one re-review (no regeneration)"
