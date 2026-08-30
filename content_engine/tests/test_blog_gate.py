"""Tests for blog.blog_gate — adhoc quality gate for manual/adhoc blog posts.

Covers:
  - article_gates.check integration (slop, em-dash (ASCII), length, secrets)
  - blog_reviewer.review integration (voice, accuracy, hype, structure)
  - Em-dash Unicode scan (U+2014, U+2013)
  - Minimum word count check per stream target
  - Required section check ("What I'd try next" or "Takeaway")
  - External link presence check (AI/PM streams only; builder exempt)
"""
import re
from unittest.mock import patch, MagicMock

import pytest

from blog.blog_gate import (
    adhoc_check,
    _has_em_dash,
    _word_count,
    _has_required_section,
    _has_external_link,
)


# -- Test drafts ------------------------------------------------------------

def _make_draft(body, title="Test Post", stream="ai", **kwargs):
    """Build a minimal draft dict for adhoc_check."""
    d = {
        "title": title,
        "description": "A test post.",
        "body_md": body,
        "slug": "test-post",
        "tier": "pm",
        "format": "essay",
        "source": "manual",
        "stream": stream,
    }
    d.update(kwargs)
    return d


# Helper: generate enough padding to exceed 70% of word_target.
# AI stream: word_target=1700, 70% = 1190. Need ~1200 words.
# PM stream: word_target=1500, 70% = 1050. Need ~1100 words.
# Builder stream: word_target=1400, 70% = 980. Need ~1000 words.
_PADDING = (
    "This is substantive content that discusses the practical implications of "
    "the approach in detail. When you consider the trade-offs between different "
    "strategies, it becomes clear that context matters enormously. The research "
    "literature consistently shows that practitioners who take the time to "
    "measure before optimising achieve better long-term results. This is not "
    "surprising when you consider the underlying mechanics of the system. "
    "Every decision involves trade-offs between latency, throughput, and "
    "cost. The optimal balance depends on your specific workload and the "
    "constraints of your deployment environment. You should always benchmark "
    "with realistic data before committing to a particular configuration. "
    "The difference between theoretical and practical performance is often "
    "larger than people expect. This is why empirical testing matters so "
    "much in this field. "
) * 25  # ~25 repetitions, each ~80 words = ~2000 words


# A long enough body with a takeaway section and an external link.
_GOOD_BODY_AI = """\
# Token-Maxing at the Edge

A counterintuitive claim grounded in concrete figures and real source material
that can be verified through the referenced papers and articles found at
https://arxiv.org/abs/2301.12345 which covers the core mechanism.

## The mechanism

The numbers tell a story about how token-maxing works at the edge of the
distribution. When you push the context window to its limit, the model starts
to degrade in specific predictable ways. This is important for practitioners
who need to understand the trade-offs involved in long-context generation.
The research shows that performance drops measurably beyond 80 percent of
context.

## Worked example

Here is the code that illustrates the concept. The implementation uses a
simple sliding window approach that can be adapted to different model sizes
and context lengths. See https://example.com for more details on the approach.

""" + _PADDING + """

## What I'd try next

The takeaway is that you should always measure before you optimise. Start
with a baseline, then push the limits and observe where degradation begins.
"""

# Builder equivalent (no external link needed).
_GOOD_BODY_BUILDER = """\
# Building a Local Inference Server

A practical guide to running models locally without cloud dependencies.
This covers the setup, pitfalls, and real performance numbers from my own
experiments running llama.cpp on consumer hardware over the past month.

## The setup

I installed llama.cpp on my Ubuntu machine with an RTX 3090 and 64GB RAM.
The build process was straightforward but required some specific flags
to get optimal performance. I tested with models ranging from 7B to 70B
parameters and measured tokens per second at each tier.

## Reality check

The hype says local inference is easy. In practice, getting good throughput
requires careful tuning of context length, batch size, and thread count.
The reality is that most guides gloss over the hard parts.

""" + _PADDING + """

## Takeaway

Start small, measure everything, and do not assume cloud-grade performance
on consumer hardware. The gap between expectations and reality is real.
"""

# Short body (below 70% of word target).
_SHORT_BODY = """\
# Too Short

This is way too short.

## What I'd try next

Done.
"""

# Body missing the required takeaway section.
_NO_TAKEAWAY_BODY = """\
# Missing Takeaway Section

This is a long enough body that passes the word count check but does not have
the required takeaway section at the end. It needs to be at least 1050 words
for the AI stream which has a word target of 1700 and a 70 percent floor.
""" + _PADDING + """

## Something Else

This section is not the takeaway section.
"""

# Body with no external link (for AI stream).
_NO_LINK_BODY = """\
# No External Links Here

This is a long enough body with a takeaway section but no external links
anywhere in the text. It references a paper by name but does not include
a clickable link to it. This should fail the AI stream check but pass
for builder stream where links are not required.
""" + _PADDING + """

## What I'd try next

The takeaway is here.
"""


# -- _has_em_dash tests -----------------------------------------------------

def test_em_dash_detects_u2014():
    """Unicode em-dash (U+2014) is detected."""
    issues = _has_em_dash("this has an \u2014 em-dash")
    assert len(issues) == 1
    assert "U+2014" in issues[0]


def test_em_dash_detects_u2013():
    """Unicode en-dash (U+2013) is detected."""
    issues = _has_em_dash("this has an \u2013 en-dash")
    assert len(issues) == 1
    assert "U+2013" in issues[0]


def test_em_dash_clean_text_has_no_issues():
    """Clean text with no Unicode dashes produces no issues."""
    issues = _has_em_dash("this has a hyphen - but no em-dash")
    assert issues == []


# -- _word_count tests ------------------------------------------------------

def test_word_count_basic():
    assert _word_count("hello world") == 2
    assert _word_count("") == 0
    assert _word_count("one-two-three") == 3


# -- _has_required_section tests --------------------------------------------

def test_has_required_section_try_next():
    assert _has_required_section("## What I'd try next\nsome text")


def test_has_required_section_takeaway():
    assert _has_required_section("## Takeaway\nsome text")


def test_has_required_section_accepts_reflection_as_closing_move():
    assert _has_required_section("## Mechanism\nbody\n## Reflection\nsome text")


def test_has_required_section_missing():
    assert not _has_required_section("## Something else\nsome text")


# -- _has_external_link tests -----------------------------------------------

def test_has_external_link_markdown_link():
    assert _has_external_link("see [paper](https://arxiv.org/abs/123)")


def test_has_external_link_bare_url():
    assert _has_external_link("see https://example.com for more")


def test_has_external_link_no_link():
    assert not _has_external_link("just plain text with no link")


# -- adhoc_check integration tests (mocked reviewer) -------------------------

@patch("blog.blog_gate._blog_review")
@patch("blog.blog_gate._article_check")
def test_adhoc_check_passes_good_ai_post(mock_gate, mock_review):
    """A well-formed AI post passes adhoc_check."""
    from article_gates import GateResult
    mock_gate.return_value = GateResult(passed=True, issues=[],
                                        redacted_body=_GOOD_BODY_AI,
                                        redacted_context="", slop_score=0)
    mock_review.return_value = {"passed": True, "score": 8, "issues": [],
                                "claims_to_verify": [], "degraded": False}
    draft = _make_draft(_GOOD_BODY_AI, stream="ai")
    status, issues = adhoc_check(draft, "ai")
    assert status == "ok", f"Expected ok, got fail with issues: {issues}"


@patch("blog.blog_gate._blog_review")
@patch("blog.blog_gate._article_check")
def test_adhoc_check_fails_on_short_post(mock_gate, mock_review):
    """Post below 70 percent of word target fails."""
    from article_gates import GateResult
    mock_gate.return_value = GateResult(passed=True, issues=[],
                                        redacted_body=_SHORT_BODY,
                                        redacted_context="", slop_score=0)
    mock_review.return_value = {"passed": True, "score": 8, "issues": [],
                                "claims_to_verify": [], "degraded": False}
    draft = _make_draft(_SHORT_BODY, stream="ai")
    status, issues = adhoc_check(draft, "ai")
    assert status == "fail"
    assert any("too short" in i.lower() for i in issues)


@patch("blog.blog_gate._blog_review")
@patch("blog.blog_gate._article_check")
def test_adhoc_check_fails_on_unicode_em_dash(mock_gate, mock_review):
    """Unicode em-dash in body fails the gate."""
    from article_gates import GateResult
    body = _GOOD_BODY_AI.replace("always measure", "always\u2014measure")  # inject Unicode em-dash
    mock_gate.return_value = GateResult(passed=True, issues=[],
                                        redacted_body=body,
                                        redacted_context="", slop_score=0)
    mock_review.return_value = {"passed": True, "score": 8, "issues": [],
                                "claims_to_verify": [], "degraded": False}
    draft = _make_draft(body, stream="ai")
    status, issues = adhoc_check(draft, "ai")
    assert status == "fail"
    assert any("U+2014" in i for i in issues)


@patch("blog.blog_gate._blog_review")
@patch("blog.blog_gate._article_check")
def test_adhoc_check_fails_ai_no_external_link(mock_gate, mock_review):
    """AI stream post with no external link fails."""
    from article_gates import GateResult
    mock_gate.return_value = GateResult(passed=True, issues=[],
                                        redacted_body=_NO_LINK_BODY,
                                        redacted_context="", slop_score=0)
    mock_review.return_value = {"passed": True, "score": 8, "issues": [],
                                "claims_to_verify": [], "degraded": False}
    draft = _make_draft(_NO_LINK_BODY, stream="ai")
    status, issues = adhoc_check(draft, "ai")
    assert status == "fail"
    assert any("external link" in i.lower() for i in issues)


@patch("blog.blog_gate._blog_review")
@patch("blog.blog_gate._article_check")
def test_adhoc_check_builder_exempt_from_link_requirement(mock_gate, mock_review):
    """Builder stream does not require external links."""
    from article_gates import GateResult
    mock_gate.return_value = GateResult(passed=True, issues=[],
                                        redacted_body=_GOOD_BODY_BUILDER,
                                        redacted_context="", slop_score=0)
    mock_review.return_value = {"passed": True, "score": 8, "issues": [],
                                "claims_to_verify": [], "degraded": False}
    draft = _make_draft(_GOOD_BODY_BUILDER, stream="builder")
    status, issues = adhoc_check(draft, "builder")
    assert status == "ok", f"Expected ok, got fail with issues: {issues}"


@patch("blog.blog_gate._blog_review")
@patch("blog.blog_gate._article_check")
def test_adhoc_check_fails_on_missing_takeaway(mock_gate, mock_review):
    """Post without 'What I'd try next' or 'Takeaway' section fails."""
    from article_gates import GateResult
    mock_gate.return_value = GateResult(passed=True, issues=[],
                                        redacted_body=_NO_TAKEAWAY_BODY,
                                        redacted_context="", slop_score=0)
    mock_review.return_value = {"passed": True, "score": 8, "issues": [],
                                "claims_to_verify": [], "degraded": False}
    draft = _make_draft(_NO_TAKEAWAY_BODY, stream="ai")
    status, issues = adhoc_check(draft, "ai")
    assert status == "fail"
    assert any("meaningful closing section" in i.lower() for i in issues)


@patch("blog.blog_gate._blog_review")
@patch("blog.blog_gate._article_check")
def test_adhoc_check_fails_on_article_gate_issues(mock_gate, mock_review):
    """Article gate failures propagate to adhoc_check."""
    from article_gates import GateResult
    mock_gate.return_value = GateResult(
        passed=False, issues=["Slop detected: 'Let's dive in'"],
        redacted_body=_GOOD_BODY_AI, redacted_context="", slop_score=7,
    )
    mock_review.return_value = {"passed": True, "score": 8, "issues": [],
                                "claims_to_verify": [], "degraded": False}
    draft = _make_draft(_GOOD_BODY_AI, stream="ai")
    status, issues = adhoc_check(draft, "ai")
    assert status == "fail"
    assert any("Slop" in i for i in issues)


@patch("blog.blog_gate._blog_review")
@patch("blog.blog_gate._article_check")
def test_adhoc_check_fails_on_reviewer_issues(mock_gate, mock_review):
    """Reviewer failures propagate to adhoc_check."""
    from article_gates import GateResult
    mock_gate.return_value = GateResult(passed=True, issues=[],
                                        redacted_body=_GOOD_BODY_AI,
                                        redacted_context="", slop_score=0)
    mock_review.return_value = {
        "passed": False, "score": 4,
        "issues": ["Voice does not match stream"],
        "claims_to_verify": [], "degraded": False,
    }
    draft = _make_draft(_GOOD_BODY_AI, stream="ai")
    status, issues = adhoc_check(draft, "ai")
    assert status == "fail"
    assert any("Voice" in i for i in issues)


@patch("blog.blog_gate._blog_review")
@patch("blog.blog_gate._article_check")
def test_adhoc_check_reviewer_degraded_is_neutral_pass(mock_gate, mock_review):
    """When reviewer LLM degrades, it does not block the gate."""
    from article_gates import GateResult
    mock_gate.return_value = GateResult(passed=True, issues=[],
                                        redacted_body=_GOOD_BODY_AI,
                                        redacted_context="", slop_score=0)
    mock_review.return_value = {"passed": True, "score": 5, "issues": [],
                                "claims_to_verify": [], "degraded": True}
    draft = _make_draft(_GOOD_BODY_AI, stream="ai")
    status, issues = adhoc_check(draft, "ai")
    assert status == "ok", f"Expected ok, got fail with issues: {issues}"


# -- Approach A research-roundup external-link gate ---------------------------

@patch("blog.blog_gate._blog_review")
@patch("blog.blog_gate._article_check")
@patch("blog.blog_gate._verify_links")
def test_adhoc_check_research_requires_external_link(mock_verify, mock_gate, mock_review):
    """Research stream posts must cite at least one primary source."""
    from article_gates import GateResult
    mock_gate.return_value = GateResult(passed=True, issues=[],
                                        redacted_body=_NO_LINK_BODY,
                                        redacted_context="", slop_score=0)
    mock_review.return_value = {"passed": True, "score": 8, "issues": [],
                                "claims_to_verify": [], "degraded": False}
    mock_verify.return_value = []  # no dead links (no links at all)
    draft = _make_draft(_NO_LINK_BODY, stream="research")
    status, issues = adhoc_check(draft, "research")
    assert status == "fail"
    assert any("external link" in i.lower() for i in issues)


@patch("blog.blog_gate._blog_review")
@patch("blog.blog_gate._article_check")
@patch("blog.blog_gate._verify_links")
def test_adhoc_check_research_passes_with_live_links(mock_verify, mock_gate, mock_review):
    """A research post with a live external link passes the link gate."""
    from article_gates import GateResult
    mock_gate.return_value = GateResult(passed=True, issues=[],
                                        redacted_body=_GOOD_BODY_AI,
                                        redacted_context="", slop_score=0)
    mock_review.return_value = {"passed": True, "score": 8, "issues": [],
                                "claims_to_verify": [], "degraded": False}
    mock_verify.return_value = []  # links all alive
    draft = _make_draft(_GOOD_BODY_AI, stream="research")
    status, issues = adhoc_check(draft, "research")
    assert status == "ok", f"Expected ok, got fail with issues: {issues}"


@patch("blog.blog_gate._blog_review")
@patch("blog.blog_gate._article_check")
@patch("blog.blog_gate._verify_links")
def test_adhoc_check_research_flags_dead_links(mock_verify, mock_gate, mock_review):
    """Dead links in a research post fail the gate (multi-source integrity)."""
    from article_gates import GateResult
    mock_gate.return_value = GateResult(passed=True, issues=[],
                                        redacted_body=_GOOD_BODY_AI,
                                        redacted_context="", slop_score=0)
    mock_review.return_value = {"passed": True, "score": 8, "issues": [],
                                "claims_to_verify": [], "degraded": False}
    mock_verify.return_value = ["https://example.com/dead"]
    draft = _make_draft(_GOOD_BODY_AI, stream="research")
    status, issues = adhoc_check(draft, "research")
    assert status == "fail"
    assert any("dead link" in i.lower() for i in issues)