"""
Regression tests for #93893 — chat `@url:` markup leaks into Electron loadURL().

Mechanism (traced on 057dcdf236):

* The renderer's title-fetch path sends ANY href to the
  ``hermes:fetchLinkTitle`` IPC without validating that it parses as an
  http(s) URL (``PrettyLink`` -> ``useLinkTitle`` -> IPC).
* In main, ``fetchLinkTitle()`` builds its cache key via
  ``canonicalTitleCacheKey()``, which — on ``new URL()`` failure — returns the
  RAW string instead of rejecting it. The value then flows to
  ``fetchHtmlTitleWithCurl`` and, when that yields no title,
  ``fetchHtmlTitleWithRenderer`` -> a hidden BrowserWindow's ``loadURL(raw)``.
* A directive-shaped string (e.g. ``@url:`https://oauth2:%s@host```) therefore
  reaches Chromium as a navigation target and produces the repeating
  ``Failed to load URL: @url:… ERR_NAME_NOT_RESOLVED`` console noise from the
  issue.

Fix contract: both boundaries refuse strings that do not parse as absolute
http/https URLs — before the curl spawn AND before the hidden-window loadURL.
Non-http schemes (mailto:, file:) keep their existing handling elsewhere and
never enter the title pipeline.
"""

from __future__ import annotations

import re

# Mirrors apps/desktop/electron/main.ts canonicalTitleCacheKey(): on a URL
# parse failure it currently returns the raw input. These tests pin the JS
# behavior via an executable spec of the guard we are adding.
MARKUP_URL = "@url:`https://oauth2:%s@example.internal.host`"
VALID_URL = "https://example.com/docs"


def _js_guard_source() -> str:
    """Load the Electron main source so tests assert the REAL guard, not a
    Python re-implementation."""
    from pathlib import Path

    main_ts = (
        Path(__file__).resolve().parents[2]
        / "apps"
        / "desktop"
        / "electron"
        / "main.ts"
    )
    return main_ts.read_text(encoding="utf-8")


def test_fetch_link_title_rejects_non_url_before_renderer_window():
    """fetchLinkTitle must bail out (empty-title promise, no window) when the
    string does not parse as http/https — the markup never reaches loadURL."""
    src = _js_guard_source()

    # The fix adds a parse check inside fetchLinkTitle itself (not only inside
    # canonicalTitleCacheKey, whose catch-all return is what let the value
    # through). Assert the function body now contains a scheme guard.
    fn_match = re.search(
        r"function fetchLinkTitle\(rawUrl\)\s*\{", src
    )
    assert fn_match, "fetchLinkTitle must exist in electron/main.ts"

    # Extract the function body up to the next top-level function.
    body_start = fn_match.start()
    next_fn = src.find("\nfunction faviconCacheKey", body_start)
    assert next_fn != -1, (
        "function-body extraction failed: faviconCacheKey marker not found — "
        "main.ts was refactored and this test must be re-anchored, not "
        "silently widened to the whole file"
    )
    body = src[body_start:next_fn]

    # Structural assertion: the guard must be a regex literal testing the
    # rawUrl parameter INSIDE fetchLinkTitle, before any use of it — not a
    # comment or an unrelated string mentioning https?:.
    guard = re.search(r"if \(!/\^https\\?:\\+/\\+//i\.test\(\w+\)\)", body) or re.search(
        r"\^https\?:\\+/\\+//i\.test", body
    )
    assert guard, (
        "pre-fix state: fetchLinkTitle performs no http(s) validation of "
        "rawUrl before spawning curl / the hidden title window (#93893)"
    )

    # The gate must run before the pipeline continues past it: the next
    # consumer is the canonicalTitleCacheKey call. (The only loadURL mention
    # inside fetchLinkTitle is a comment; the hidden window lives elsewhere —
    # matching call sites, not comments, by stripping comment lines first.)
    code_lines = [ln for ln in body.splitlines() if not ln.strip().startswith("//")]
    code_body = "\n".join(code_lines)
    first_use = re.search(r"canonicalTitleCacheKey\(", code_body)
    guard_line_idx = next((i for i, ln in enumerate(code_lines) if "/^https" in ln), -1)
    guard_code_pos = (
        len("\n".join(code_lines[:guard_line_idx])) if guard_line_idx != -1 else -1
    )

    assert first_use and 0 <= guard_code_pos < first_use.start(), (
        "the scheme guard must precede the first downstream consumer "
        "(canonicalTitleCacheKey) — a guard after the call is dead code"
    )


def test_canonical_title_cache_key_does_not_passthrough_unparseable_input():
    """canonicalTitleCacheKey's catch block returning the raw string is the
    passthrough hole; the fix narrows it to '' so unparseable input can never
    become a cache key that looks loadable."""
    src = _js_guard_source()

    fn_match = re.search(r"function canonicalTitleCacheKey\(rawUrl\)\s*\{", src)
    assert fn_match, "canonicalTitleCacheKey must exist"

    body_start = fn_match.end()
    # Body ends at the next 'function ' at column 0.
    next_fn = src.find("\nfunction ", body_start)
    body = src[body_start:next_fn]

    # Pre-fix code: `return value` inside the catch. Post-fix: return ''.
    catch_block = re.search(r"catch\s*\{([^}]*)\}", body)
    assert catch_block, "canonicalTitleCacheKey must keep its try/catch"
    assert "return ''" in catch_block.group(1), (
        "pre-fix state: unparseable input is returned as a usable cache key "
        "(and later handed to loadURL) instead of being rejected (#93893)"
    )
