"""Tests for log redaction of token-bearing URLs in the Teams adapter.

Teams Bot Framework and SharePoint download URLs frequently carry bearer
or ``tempauth`` material in the query string (``?tempauth=...``,
``?upload_session_token=...``) or path. Logging raw URLs to ``gateway.log``
leaks file-access credentials to log aggregators and anyone with read access
to the gateway logs — a replayable token until expiry (typically minutes to
hours).

These tests verify:

* ``_url_fingerprint`` strips query strings, fragments, and userinfo;
* the deterministic sha8 tag remains stable for the same URL (so two log
  lines referencing the same URL can be correlated);
* the attachment-dispatch log sites in the inbound flow no longer use
  format strings that interpolate raw URL strings.

This is the regression suite for the security review on PR #26289.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from plugins.platforms.teams.adapter import _url_fingerprint


# ---------------------------------------------------------------------------
# _url_fingerprint — pure unit tests
# ---------------------------------------------------------------------------

# Real-world-shaped tokens that MUST NOT appear in fingerprint output.
# Synthesised here, not real secrets — but the strings exercise the same
# parser paths a live token would.
_BEARER = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsInkid2lWRGoyN3UifQ.PAYLOAD.SIG"
_TEMPAUTH = "0e6f1a01-tempauth-90ab-cdef-1234567890ab&access_token=opaque-tok"


@pytest.mark.parametrize(
    "url",
    [
        # Bot Framework attachment URL (path token style).
        f"https://smba.trafficmanager.net/amer/v3/attachments/{_BEARER}/views/original",
        # SharePoint tempauth download URL (query-string token).
        f"https://contoso.sharepoint.com/sites/x/_api/v2.0/drives/abc/items/def/content?tempauth={_TEMPAUTH}",
        # OneDrive upload session URL (token in query).
        f"https://api.onedrive.com/upload/sessions/abc?upload_session_token={_BEARER}",
        # Fragment-encoded token.
        f"https://example.com/path#access_token={_BEARER}",
        # userinfo-encoded credentials.
        f"https://user:{_BEARER}@example.com/path",
        # Hosted contents Graph URL (id is sensitive — opaque bearer-ish blob).
        f"https://graph.microsoft.com/v1.0/teams/X/channels/Y/messages/Z/hostedContents/{_BEARER}/$value",
    ],
)
def test_fingerprint_never_leaks_token_material(url: str) -> None:
    """No fingerprint should contain any of the bearer/tempauth substrings."""
    fp = _url_fingerprint(url)
    assert _BEARER not in fp, f"bearer leaked into fingerprint: {fp!r}"
    assert _TEMPAUTH not in fp, f"tempauth leaked into fingerprint: {fp!r}"
    assert "?" not in fp, f"query string leaked: {fp!r}"
    assert "#access_token" not in fp, f"fragment leaked: {fp!r}"
    # userinfo (user:pass@) must not survive.
    assert "@" not in fp, f"userinfo leaked: {fp!r}"


def test_fingerprint_preserves_host_and_path_prefix() -> None:
    fp = _url_fingerprint(
        f"https://contoso.sharepoint.com/sites/x/_api/v2.0/drives/abc/items/def/content?tempauth={_TEMPAUTH}"
    )
    assert fp.startswith("contoso.sharepoint.com/sites/x/_api/v2.0/")
    # 8-char sha tag at the end after `#`.
    host_path, _, tag = fp.partition("#")
    assert len(tag) == 8 and all(c in "0123456789abcdef" for c in tag)


def test_fingerprint_is_deterministic_for_correlation() -> None:
    """Two log lines for the same URL must produce the same fingerprint
    so operators can grep correlations across multi-line attachment flows.
    """
    url = f"https://smba.trafficmanager.net/amer/v3/attachments/{_BEARER}/views/original"
    assert _url_fingerprint(url) == _url_fingerprint(url)


def test_fingerprint_differs_for_different_urls() -> None:
    a = _url_fingerprint("https://smba.trafficmanager.net/amer/v3/attachments/A/views/original")
    b = _url_fingerprint("https://smba.trafficmanager.net/amer/v3/attachments/B/views/original")
    # Same host+path prefix but different sha tag.
    assert a != b


@pytest.mark.parametrize("bad", [None, "", 0, 12345, b"https://x", object()])
def test_fingerprint_safe_for_non_string_inputs(bad) -> None:
    """The helper is called from logging hot paths — it must never raise."""
    out = _url_fingerprint(bad)  # type: ignore[arg-type]
    assert isinstance(out, str)
    assert out  # non-empty sentinel
    assert "no-url" in out or "unparseable" in out


def test_fingerprint_truncates_long_paths() -> None:
    long_path = "/a" * 500
    fp = _url_fingerprint(f"https://x.example.com{long_path}?tempauth=secret")
    assert "tempauth" not in fp
    assert "..." in fp  # truncation marker


# ---------------------------------------------------------------------------
# Source-level guard — ensure log sites never re-introduce raw URL formatting
# ---------------------------------------------------------------------------

# Format-string fragments that historically leaked Teams/SharePoint URLs.
# If somebody adds a new ``content_url=%r`` or ``download_url=%r`` log
# without going through ``_url_fingerprint`` first, this test fails.
_FORBIDDEN_LOG_PATTERNS = [
    "content_url=%r",
    "download_url=%r",
    "upload_url=%r",
]


def test_no_raw_url_format_strings_in_adapter() -> None:
    src = (
        Path(__file__).resolve().parents[4]
        / "plugins" / "platforms" / "teams" / "adapter.py"
    ).read_text(encoding="utf-8")
    for pat in _FORBIDDEN_LOG_PATTERNS:
        assert pat not in src, (
            f"adapter.py contains forbidden log format {pat!r} — "
            f"raw URLs must be wrapped in _url_fingerprint() before logging"
        )


def test_bf_get_log_uses_fingerprint() -> None:
    """The two BF GET log lines flagged in the security review must wrap
    their URL through ``_url_fingerprint``."""
    src = (
        Path(__file__).resolve().parents[4]
        / "plugins" / "platforms" / "teams" / "adapter.py"
    ).read_text(encoding="utf-8")
    # Both occurrences of ``BF GET %s`` should now feed through fingerprint.
    bf_get_lines = re.findall(r'.*"\[teams\]\[attach\] BF GET %s.*', src)
    assert bf_get_lines, "expected BF GET log lines to exist"
    for line in bf_get_lines:
        # Either the format string contains both the call (next line)
        # or the format and call are on the same line. Find the matching
        # ``logger.warning(...)`` call by widening to a small window.
        idx = src.index(line)
        window = src[idx : idx + 400]
        assert "_url_fingerprint(" in window, (
            f"BF GET log line does not wrap URL through _url_fingerprint:\n{window}"
        )
