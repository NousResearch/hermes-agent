"""Login-walled URL extraction guard + attachment-block title strip.

Regression for the keyless-extractor fabrication class (2026-09-18): for URLs
behind an interactive login wall, keyless web extractors return a *nearest-match*
document stamped with the requested URL and no error signal. That fabricated
text was expanded into context (@url expansion) and then named the session via
the LLM title path (``title_source=llm``). Two invariants:

- expansion must never hand a login-walled URL to a fetcher;
- the titler must never see machine-expanded attachment blocks.
"""

import pytest

from agent.context_references import (
    _is_login_walled_url,
    preprocess_context_references_async,
)
from agent.title_generator import (
    _summarize_user_message,
    derive_title,
    is_titleable_user_message,
)

_FABRICATED = "Turkey-Russian Economic Relations and Security-Based Strategic Partnership"
_DM_URL = "https://mail.google.com/mail/u/0/#chat/dm/3H_bhAAAAAE"


class TestLoginWalledHostDetection:
    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            (_DM_URL, True),
            ("https://accounts.google.com/o/oauth2/auth?x=1", True),
            ("https://chat.google.com/room/abc", True),
            ("https://docs.google.com/document/d/xyz/edit", True),
            ("https://drive.google.com/file/d/xyz/view", True),
            # The suffix must bind on a dot boundary — lookalikes stay public.
            ("https://evil-mail.google.com.example.com/", False),
            ("https://mail.google.com.evil.tld/", False),
            ("https://example.com/mail.google.com", False),
            ("https://github.com/emirsaffar-collab/hermes-agent", False),
            ("not a url", False),
            ("", False),
        ],
    )
    def test_host_suffix_matches_on_dot_boundary(self, url, expected):
        assert _is_login_walled_url(url) is expected


@pytest.mark.asyncio
async def test_login_walled_url_is_never_fetched(tmp_path):
    """The fetcher must not run for a login-walled URL: whatever it returns is
    fabricated nearest-match content presented as the page's actual text."""
    calls: list[str] = []

    def fetcher(url):
        calls.append(url)
        return _FABRICATED

    result = await preprocess_context_references_async(
        f"check @url:{_DM_URL}",
        cwd=tmp_path,
        context_length=100_000,
        url_fetcher=fetcher,
    )

    assert calls == []
    assert _FABRICATED not in result.message
    assert "login-walled" in result.message


@pytest.mark.asyncio
async def test_public_url_still_fetched(tmp_path):
    """The guard must not over-block: a public URL goes through the fetcher unchanged."""

    def fetcher(url):
        return "REAL CONTENT"

    result = await preprocess_context_references_async(
        "read @url:https://example.com/post",
        cwd=tmp_path,
        context_length=100_000,
        url_fetcher=fetcher,
    )

    assert "REAL CONTENT" in result.message
    assert "login-walled" not in result.message


class TestAttachmentBlockTitleStrip:
    def test_attachment_content_never_reaches_the_titler(self):
        msg = (
            "Why did this session get the wrong name?\n\n"
            "--- Attached Context ---\n\n"
            f"🌐 @url:{_DM_URL}\n"
            f"{_FABRICATED}\nMarmara University, 2020. Author: DERMAN, GİRAY SAYNUR."
        )
        assert _FABRICATED not in _summarize_user_message(msg)
        assert derive_title(msg) == "Why did this session get the wrong name?"

    def test_message_without_marker_is_unchanged(self):
        msg = "plain question about deploys"
        assert derive_title(msg) == "plain question about deploys"

    def test_attachment_only_message_is_not_titleable(self):
        """An opener that reduces to a bare attachment block carries no user signal."""
        msg = f"--- Attached Context ---\n\n🌐 @url:{_DM_URL}\n{_FABRICATED}"
        assert not is_titleable_user_message(msg)
