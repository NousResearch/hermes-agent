"""Tests for article_gates — slop + structure + data-integrity + secret strip."""
import article_gates as ag


def _draft(body=None, context="real context with concrete numbers and tool names: 73% growth in routing, model x, ten lines of code", title="How I tuned routing in KenseiAgent"):
    body = body or (
        f"# {title}\n\n"
        "Lede paragraph that goes on long enough to be considered a real article.\n\n"
        "## First section explains the problem in detail\n\n"
        "Body one with enough words. The 73% growth in routing model fallback drove me to look at this. "
        "We use model x for the primary and the model y tier is a free fallback. The numbers and tools "
        "are quoted from the real context, not made up. Real workflow, real numbers, real building.\n\n"
        "## Second section shows the fix that actually worked\n\n"
        "Body two explains the fix. We restructured the prompt to inject the voice skill verbatim. "
        "The change took a few iterations to settle. The 73% improvement is the headline number, and we "
        "kept the ten lines of code that made the biggest difference. Real workflow, real numbers, real building.\n\n"
        "## Third section covers the data integrity check\n\n"
        "Body three covers the data integrity check. We added a regex pass that flags any number in the "
        "article body that has no counterpart in the enriched context blob. The redactor also strips API "
        "keys, bearer tokens, and `.env` style `KEY=value` patterns before anything touches the disk.\n\n"
        "## What I'd try next\n\n"
        "Next I want to thread the voice skill through to the digest mode and tune the per-section image "
        "density so the article stays under the per-month budget. A two-pass outline-then-expand step "
        "is also on the table for the quality work that comes after the live cutover.\n"
    )
    return {
        "title": title, "body_md": body, "mode": "deep_dive",
        "pillar": "harness_tuning", "slug": "how-i-tuned-routing",
        "signals": [], "context": context,
    }


def test_check_passes_valid_draft():
    res = ag.check(_draft())
    assert res.passed, f"unexpected issues: {res.issues}"


def test_check_rejects_short_article():
    short = "# T\n\nLede one two three.\n\n## A\n\nx\n\n## B\n\ny\n\n## C\n\nz\n"
    res = ag.check(_draft(body=short))
    assert not res.passed
    assert any("short" in i.lower() or "word" in i.lower() or "min" in i.lower() for i in res.issues)


def test_check_rejects_few_sections():
    body = (
        "# T\n\nLede.\n\n## A\n\nbody body body body body body body body body body body body body body body body body body body body body body body body body body body.\n"
        "## What I'd try next\n\nTry the thing and explain in many more words to keep the length up over the min words threshold and so on.\n"
    )
    res = ag.check(_draft(body=body))
    assert not res.passed
    assert any("section" in i.lower() or "h2" in i.lower() for i in res.issues)


def test_check_rejects_no_takeaway():
    body = (
        "# T\n\nLede. Lede. Lede.\n\n"
        "## A\n\n" + ("body " * 30) + "\n\n"
        "## B\n\n" + ("body " * 30) + "\n\n"
        "## C\n\n" + ("body " * 30) + "\n"
    )
    res = ag.check(_draft(body=body))
    assert not res.passed
    assert any("meaningful closing section" in i.lower() for i in res.issues)


def test_check_accepts_decision_as_closing_section():
    body = (
        "# T\n\nLede. Lede. Lede.\n\n"
        "## A\n\n" + ("body " * 30) + "\n\n"
        "## B\n\n" + ("body " * 30) + "\n\n"
        "## C\n\n" + ("body " * 30) + "\n\n"
        "## Decision\n\n" + ("Choose the smaller change first. " * 20) + "\n"
    )
    res = ag.check(_draft(body=body))
    assert not any("takeaway" in issue.lower() for issue in res.issues)


def test_data_integrity_flags_fabricated_number():
    """Body says '99% increase' but context has no '99'. The gate flags the mismatch."""
    body = (
        "# T\n\nLede. Lede.\n\n"
        "## A\n\nThere was a 99% increase in routing errors last week.\n"
        "## B\n\n" + ("body " * 30) + "\n\n"
        "## C\n\n" + ("body " * 30) + "\n\n"
        "## What I'd try next\n\n" + ("body " * 20) + "\n"
    )
    res = ag.check(_draft(body=body, context="73% growth in routing, model x"))
    assert not res.passed
    assert any("99" in i or "data" in i.lower() or "integrity" in i.lower() for i in res.issues)


def test_data_integrity_ignores_illustrative_integers():
    """Small bare integers used rhetorically ('12 agents', '53 lines', '2am') are
    not statistics, so the data-integrity check must not flag them."""
    body = (
        "# T\n\nLede. Lede.\n\n"
        "## A\n\nIf you have 12 agents and 1 of them crashes at 2am, the 53 lines "
        "of guard code still hold. " + ("word " * 260) + "\n\n"
        "## B\n\n" + ("word " * 260) + "\n\n"
        "## C\n\n" + ("word " * 260) + "\n\n"
        "## What I'd try next\n\n" + ("word " * 200) + "\n"
    )
    res = ag.check(_draft(body=body, context="73% growth in routing, model x"))
    assert res.passed, f"unexpected issues: {res.issues}"
    assert not any("integrity" in i.lower() for i in res.issues)


def test_data_integrity_flags_fabricated_large_figure():
    """A large metric-like figure ('20000 stars') absent from context is flagged."""
    body = (
        "# T\n\nLede. Lede.\n\n"
        "## A\n\nThe repo hit 20000 stars overnight, which surprised everyone.\n"
        "## B\n\n" + ("body " * 30) + "\n\n"
        "## C\n\n" + ("body " * 30) + "\n\n"
        "## What I'd try next\n\n" + ("body " * 20) + "\n"
    )
    res = ag.check(_draft(body=body, context="73% growth in routing, model x"))
    assert not res.passed
    assert any("20000" in i or "integrity" in i.lower() for i in res.issues)


def test_secret_scan_redacts_api_key():
    """Body / context with a planted sk- token is rewritten to ***REDACTED*** and the gate flags a secret_stripped issue."""
    body = (
        "# T\n\nLede. Lede.\n\n"
        "## A\n\nThe agent was set up with the key sk-1234567890abcdefghij in env.\n"
        "## B\n\n" + ("body " * 30) + "\n\n"
        "## C\n\n" + ("body " * 30) + "\n\n"
        "## What I'd try next\n\n" + ("body " * 20) + "\n"
    )
    res = ag.check(_draft(body=body))
    assert any("secret" in i.lower() for i in res.issues)
    assert "sk-1234567890abcdefghij" not in res.redacted_body
    assert "***REDACTED***" in res.redacted_body


def test_secret_scan_does_not_redact_task_specific_words():
    """The sk- detector must not corrupt normal words like task-specific."""
    body = (
        "# T\n\nLede. Lede.\n\n"
        "## A\n\nA task-specific context packet is not a credential.\n"
        "## B\n\n" + ("body " * 30) + "\n\n"
        "## C\n\n" + ("body " * 30) + "\n\n"
        "## What I'd try next\n\n" + ("body " * 20) + "\n"
    )
    res = ag.check(_draft(body=body))
    assert "task-specific" in res.redacted_body
    assert "***REDACTED***" not in res.redacted_body


def test_check_inherits_slop_gate():
    """Known slop phrase fails on the slop_score carried over from gate_post."""
    body = (
        "# T\n\nLede.\n\n"
        "## A\n\nLet's dive in. In today's world, great question, it's worth noting.\n"
        "## B\n\n" + ("body " * 30) + "\n\n"
        "## C\n\n" + ("body " * 30) + "\n\n"
        "## What I'd try next\n\n" + ("body " * 20) + "\n"
    )
    res = ag.check(_draft(body=body))
    assert not res.passed
    assert res.issues  # at least one issue raised


def test_check_rejects_em_dash():
    body = (
        "# T\n\nLede. Lede.\n\n"
        "## A\n\nThis section has a hard\u2014em dash inside it.\n"
        "## B\n\n" + ("body " * 30) + "\n\n"
        "## C\n\n" + ("body " * 30) + "\n\n"
        "## What I'd try next\n\n" + ("body " * 20) + "\n"
    )
    res = ag.check(_draft(body=body))
    assert not res.passed
    assert any("em-dash" in i.lower() for i in res.issues)
