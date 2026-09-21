"""Tests for the company case study gate (Block 5).

AI-stream posts must include at least one named company/product with a
specific number. PM/builder posts get a warning only. Opinion posts exempt.
"""
import pytest

from blog.blog_gate import case_study_check


def test_case_study_pass_with_company_and_number():
    """AI post with a named company and a specific number passes."""
    body = """# How OpenAI Ships GPT-4

OpenAI reported 1.8 trillion parameters in GPT-4. The training cost was
$63 million. Anthropic used 400 GPU-hours for Claude.

## Takeaway
Numbers matter."""
    status, issues = case_study_check(body, stream="ai")
    assert status == "ok"
    assert issues == []


def test_case_study_fail_no_company():
    """AI post with numbers but no named company fails."""
    body = """# Big Models Are Expensive

Training a large model costs $100 million. It needs 500 GPUs and 30 days.

## Takeaway
Expensive."""
    status, issues = case_study_check(body, stream="ai")
    assert "company" in " ".join(issues).lower()


def test_case_study_fail_no_number():
    """AI post with a company but no specific number fails."""
    body = """# OpenAI's Approach

OpenAI built GPT-4. Anthropic built Claude. Google built Gemini.

## Takeaway
Many companies."""
    status, issues = case_study_check(body, stream="ai")
    assert "number" in " ".join(issues).lower()


def test_case_study_fail_no_company_no_number():
    """AI post with neither company nor number fails."""
    body = """# The State of AI

Models are getting bigger. Training is expensive. Inference is cheaper now.

## Takeaway
Trends."""
    status, issues = case_study_check(body, stream="ai")
    assert len(issues) >= 1


def test_case_study_pm_warning_only():
    """PM post with neither company nor number gets a warning, not a block."""
    body = """# Prioritisation Frameworks

Just ship it. Use RICE or ICE. Stakeholder alignment matters.

## Takeaway
Prioritise."""
    status, issues = case_study_check(body, stream="pm")
    # PM is warning-only, so status should be ok with issues as warnings.
    assert status == "ok"
    assert len(issues) >= 1


def test_case_study_builder_warning_only():
    """Builder post with neither company nor number gets a warning, not a block."""
    body = """# Shipping Fast

Deploy often. Use CI. Automate everything.

## Takeaway
Ship."""
    status, issues = case_study_check(body, stream="builder")
    assert status == "ok"
    assert len(issues) >= 1


def test_case_study_exempt_opinion():
    """Opinion posts skip the check entirely."""
    body = """# My Thoughts on AI

I think AI is good. Models are nice. Numbers are irrelevant.

## Takeaway
My opinion."""
    status, issues = case_study_check(body, stream="ai", is_opinion=True)
    assert status == "ok"
    assert issues == []


def test_case_study_detects_pct_numbers():
    """Numbers in percentage form count as specific numbers."""
    body = """# Anthropic's Claude

Anthropic reported 47% fewer hallucinations with Claude 3 vs GPT-4.

## Takeaway
Better."""
    status, issues = case_study_check(body, stream="ai")
    assert status == "ok"
    assert issues == []


def test_case_study_detects_currency_numbers():
    """Currency amounts count as specific numbers."""
    body = """# Mistral's Funding

Mistral raised $415 million in Series B. Revenue is £2.5M ARR.

## Takeaway
Funded."""
    status, issues = case_study_check(body, stream="ai")
    assert status == "ok"
    assert issues == []