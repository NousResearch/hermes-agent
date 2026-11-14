"""Regression: streamed/interim reconstruction glued markdown block boundaries.

Live 2026-08-19 session 20260819_223427_6bc23a persisted a Grok answer
whose OmniRoute artifact still had newlines (```txt fences, ---, ##, table)
while Hermes state.db stored the whitespace-collapsed form. Inline markup
survived; block separators did not.
"""

from __future__ import annotations

from agent.chat_completion_helpers import _repair_glued_markdown_block_boundaries


LIVE_GLUED = (
    "Include every sender you actually use:\n"
    "```txtv=spf1 include:_spf.mx.cloudflare.net include:_spf.google.com ~all```"
    "**DMARC** (TXT on `_dmarc`), start permissive:\n"
    "```txtv=DMARC1; p=none; rua=mailto:you@yourdomain.com```"
    "Tighten to `p=quarantine` later once reports look clean.\n"
    "---## The paid “just works” option**Google Workspace** on the custom domain.\n"
    "---## Things that will *not* do what you asked| Idea | Why it fails |\n"
    "|---|---|| Gmail “plus alias” (`you+domain@gmail.com`) | Still `@gmail.com` |"
)


def test_live_spf_dmarc_specimen_restores_fences_headings_and_table():
    repaired = _repair_glued_markdown_block_boundaries(LIVE_GLUED)

    assert "```txt\nv=spf1 include:_spf.mx.cloudflare.net include:_spf.google.com ~all\n```" in repaired
    assert "```txt\nv=DMARC1; p=none; rua=mailto:you@yourdomain.com\n```" in repaired
    assert "\n\n## The paid" in repaired
    assert "\n\n## Things that will *not* do what you asked\n\n| Idea | Why it fails |" in repaired
    assert "|---|---|\n| Gmail" in repaired

    assert "```txtv=spf1" not in repaired
    assert "```**DMARC**" not in repaired
    assert "---## The paid" not in repaired
    assert "asked| Idea" not in repaired
    assert "|---|---|| Gmail" not in repaired


def test_already_correct_markdown_is_stable():
    clean = (
        "Use:\n\n"
        "```txt\n"
        "v=spf1 include:_spf.google.com ~all\n"
        "```\n\n"
        "**DMARC** next.\n\n"
        "---\n\n"
        "## Things that will *not* do what you asked\n\n"
        "| Idea | Why it fails |\n"
        "|---|---|\n"
        "| Gmail plus alias | Still gmail |\n"
    )
    assert _repair_glued_markdown_block_boundaries(clean) == clean


def test_does_not_rewrite_fence_body():
    body = "```txt\nline##not-a-heading| still code ---\n```"
    assert _repair_glued_markdown_block_boundaries(body) == body
