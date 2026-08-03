"""Regression: non-retryable API failures must not leak raw HTML pages.

A scheduled cron job fell back to the Codex (``chatgpt.com``) provider, which
returned a Cloudflare *challenge* page (HTTP 403) instead of a normal API
response.  The conversation loop classified this as a non-retryable client
error and returned the failure dict — but the ``error`` field carried
``str(api_error)``, i.e. the entire ~60 KB Cloudflare HTML page.  The cron
scheduler then delivered that verbatim to Discord, where it was split into
~31 messages (the reporter's "31 part discord message which is cloudflares
challenge page").

The sibling "max retries exhausted" path already summarized the error via
``_summarize_api_error`` (which collapses HTML pages to a one-liner); the
non-retryable path did not.  These tests lock the contract: whichever
terminal path is taken, ``result['error']`` is a short, HTML-free summary.
"""

from unittest.mock import MagicMock, patch

import run_agent
from run_agent import AIAgent


# A representative Cloudflare "managed challenge" body, matching the shape the
# Codex backend returned in the field report (no <title>, large inline
# ``_cf_chl_opt`` script).  Padded so length-based assertions are meaningful.
_CLOUDFLARE_CHALLENGE_HTML = (
    "<!DOCTYPE html>\n<html>\n  <head>\n"
    '    <meta http-equiv="refresh" content="360"></head>\n'
    "  <body>\n    <div class=\"data\"><noscript>"
    "Enable JavaScript and cookies to continue</noscript>"
    "<script>(function(){window._cf_chl_opt = {cRay: 'a0ca002c4f91769c',"
    "cZone: 'chatgpt.com', cType: 'managed', "
    + ("md: '" + "x" * 4000 + "',")
    + "};})();</script></div>\n  </body>\n</html>\n"
)


def _make_403_html_error() -> Exception:
    """An exception mimicking a Codex 403 whose body is a Cloudflare page."""
    err = Exception(_CLOUDFLARE_CHALLENGE_HTML)
    err.status_code = 403
    return err


def _make_agent() -> AIAgent:
    # Drive the standard chat-completions path with a concrete model so the
    # turn actually reaches ``client.chat.completions.create`` — that is where
    # the mocked 403 is raised.  The non-retryable abort being exercised lives
    # in the shared conversation loop and is provider-agnostic; a Cloudflare
    # "managed challenge" 403 can surface on any provider sitting behind
    # Cloudflare (it was first reported on the Codex backend).  Pinning
    # ``api_mode`` + ``model`` here avoids the earlier abort the previous
    # revision hit: an empty model on the Codex Responses path raised a
    # validation ``ValueError`` *before* any API call, so the test passed
    # without ever touching the 403 summarization path.
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://api.openai.com/v1",
            provider="openai",
            api_mode="chat_completions",
            model="gpt-5.5",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    a.client = MagicMock()
    a._cached_system_prompt = "You are helpful."
    a._use_prompt_caching = False
    a.tool_delay = 0
    a.compression_enabled = False
    a.save_trajectories = False
    return a


def test_summarize_collapses_cloudflare_challenge_page():
    """``_summarize_api_error`` must never echo the raw HTML body."""
    summary = AIAgent._summarize_api_error(_make_403_html_error())

    assert "<html" not in summary.lower()
    assert "<!doctype" not in summary.lower()
    assert "_cf_chl_opt" not in summary
    # A one-liner, not a multi-kilobyte page.
    assert len(summary) < 200
    # Still informative: the HTTP status survives.
    assert "403" in summary


def test_non_retryable_failure_error_is_summarized_not_raw_html():
    """The terminal non-retryable dict must carry a short, HTML-free error.

    This is the exact field path: a 403 Cloudflare challenge with no fallback
    configured aborts as a non-retryable client error.  Before the fix the
    returned ``error`` was the full ~60 KB page.

    The mocked 403 is the *only* failure the turn can hit — the agent reaches
    ``client.chat.completions.create`` (asserted below), so the test cannot
    pass vacuously by aborting on some earlier, unrelated error.
    """
    agent = _make_agent()
    agent.client.chat.completions.create.side_effect = _make_403_html_error()

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("daily briefing please")

    # Guard against a vacuous pass: the mocked 403 must actually be the
    # failure that aborted the turn.  (The previous revision never reached
    # this call and still "passed".)
    assert agent.client.chat.completions.create.called
    assert result.get("failed") is True
    error = result.get("error") or ""
    # The whole point of the fix: no raw HTML / Cloudflare markup leaks.
    assert "<html" not in error.lower()
    assert "<!doctype" not in error.lower()
    assert "_cf_chl_opt" not in error
    # Still informative: the summarized 403 status survives into the field
    # delivered downstream.
    assert "403" in error
    # The original page was tens of kilobytes; a summary is short.
    assert len(error) < 500
    assert len(error) < len(_CLOUDFLARE_CHALLENGE_HTML)
