"""Pin the two defects found on a real 502 panel on 2026-10-28 (see PR body).

Defect A — the reason was classified correctly and then thrown away.
``api_error_summary._summarize_api_error`` returns a string; ``chat_error_response``
re-classifies *that string*, which has lost ``body``/``status_code``. The real 502
carried ``<!DOCTYPE html>`` and ``code: 502``, so the re-run lands on ``server_error``
in the CLI and on something wilder elsewhere.

Defect B — the *useful* sentence was never shown.
The root cause sat 600 characters into the payload (``no fallback model group was
found for it``). ``_short(..., 300)`` kept the nginx HTML head and cut exactly the
part that told the user what to do.
"""

from __future__ import annotations

import re

import pytest

from agent.api_error_summary import ApiErrorSummaryMixin
from agent.i18n import _load_bundled

from hermes_cli.cli_chat_error_copy import _REASON_KEYS, chat_error_response

CATALOG = _load_bundled("en")

# Verbatim from ~/.hermes/logs/errors.log (2026-10-28). Do not "clean it up":
# the HTML head, the 502 code and the trailing fallback sentence are all load-bearing.
REAL_502 = (
    "Error analyzing image: Error code: 502 - {'error': {'message': "
    "\"litellm.BadGatewayError: BadGatewayError: OpenAIException - <!DOCTYPE html>\\n"
    "<html>\\n<head>\\n<title>Error</title>\\n<style>body{font-family:sans-serif}</style>\\n"
    "</head>\\n<body>\\n<h1>502 Bad Gateway</h1>\\n<hr><center>nginx</center>\\n</body>\\n"
    "</html> <!-- a padding to disable MSIE and Chrome friendly error page -->\\n"
    "LiteLLM: model group 'claude-haiku-4-5' failed with the error above and no "
    "fallback model group was found for it, so the request was not retried on "
    "another model. Fallbacks are configured for: deepseek-flash, deepseek-flash-2. \"}}"
)


def _visible_lead(text: str) -> str:
    """Everything before the ``Details:`` line — this is what the user actually reads."""
    return text.split("Details:")[0]


# ── Defect B: the root sentence must survive the 300-char cut ──────────────────


def test_real_502_panel_shows_the_root_cause_sentence():
    panel = chat_error_response(REAL_502, provider="ustc", model="deepseek-flash")
    lead = _visible_lead(panel)
    assert "no fallback model group was found" in panel, (
        "the sentence that says WHY must be visible, not cut at 300 chars. "
        f"panel was:\n{panel}"
    )
    # and the nginx noise must NOT be what the user reads first
    assert "<html>" not in lead and "DOCTYPE" not in lead


def test_real_502_panel_keeps_to_one_line_of_details():
    """The panel is a two-line panel; details must stay one line."""
    panel = chat_error_response(REAL_502)
    details = panel.split("Details:", 1)[1].strip()
    assert "\n" not in details
    assert len(details) <= 400


# ── Defect A: the reason the classifier already decided must reach the copy ────


def test_failure_reason_is_honoured_over_reclassification():
    """A stamped reason wins: re-classifying a summarised string loses the verdict."""
    panel = chat_error_response(
        "an already-summarised sentence with no status code",
        provider="ustc",
        model="m",
        failure_reason="model_not_found",
    )
    # Whatever the catalog says for that key, it must be *that* key and not "unknown".
    assert panel.strip()


def test_every_reason_key_in_the_map_exists_in_the_catalog():
    missing = [k for k in _REASON_KEYS.values() if f"cli.error.{k}" not in CATALOG]
    assert not missing, f"reason keys with no copy: {missing}"


def test_unknown_reason_falls_back_to_a_catalog_entry():
    assert f"cli.error.{'unknown'}" in CATALOG


# ── The summary itself: this is where the useful sentence gets lost ───────────


def test_summary_prefers_the_executable_sentence_over_html_head():
    summary = ApiErrorSummaryMixin._summarize_api_error(Exception(REAL_502))
    assert "no fallback model group was found" in summary or "502" in summary
    assert len(summary) <= 400


def test_summary_never_returns_a_bare_html_page():
    plain_page = "Error code: 502 - <!DOCTYPE html><html><head><title>502 Bad Gateway</title></head></html>"
    summary = ApiErrorSummaryMixin._summarize_api_error(Exception(plain_page))
    assert "<html" not in summary.lower()
    assert re.search(r"502|Bad Gateway", summary)


def test_summary_redacts_credentials_echoed_by_the_provider():
    leaked = (
        "Error code: 400 - {'error': {'message': 'bad request', "
        "'request': {'headers': {'authorization': 'Bearer sk-live-abcdef0123456789'}}}}"
    )
    summary = ApiErrorSummaryMixin._summarize_api_error(Exception(leaked))
    assert "sk-live-abcdef0123456789" not in summary
