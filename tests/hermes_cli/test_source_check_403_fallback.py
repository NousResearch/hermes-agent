"""source_check._request falls back to anonymous on HTTP 403 (stale/suspended-account token).

Previously only 401 triggered the anonymous retry; a suspended account returns 403 and the
check failed permanently, caching the error for an hour.

Fixes: https://github.com/NousResearch/hermes-agent/issues/134335
"""

from __future__ import annotations

import urllib.error
import urllib.response
from io import BytesIO
from unittest.mock import patch

import pytest

from hermes_cli.source_check import _request


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="https://api.github.com/test",
        code=code,
        msg=f"HTTP {code}",
        hdrs={},  # type: ignore[arg-type]
        fp=BytesIO(b""),
    )


def _patch_request_with(responses: list):
    """Patch _request_with to return successive values (or raise if an exception)."""
    call_iter = iter(responses)

    def _fake_request_with(url, accept, token):
        val = next(call_iter)
        if isinstance(val, BaseException):
            raise val
        return val

    return patch("hermes_cli.source_check._request_with", side_effect=_fake_request_with)


# ---------------------------------------------------------------------------
# 401 fallback (pre-existing behaviour, must not regress)
# ---------------------------------------------------------------------------

def test_401_falls_back_to_anonymous():
    with patch("hermes_cli.github_api.github_token", return_value="tok"), \
         _patch_request_with([_http_error(401), '{"sha":"abc"}']):
        result = _request("https://api.github.com/test")
    assert result == '{"sha":"abc"}'


def test_401_with_no_token_raises():
    with patch("hermes_cli.github_api.github_token", return_value=None), \
         _patch_request_with([_http_error(401)]):
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _request("https://api.github.com/test")
    assert exc_info.value.code == 401


# ---------------------------------------------------------------------------
# 403 fallback (new behaviour)
# ---------------------------------------------------------------------------

def test_403_falls_back_to_anonymous():
    """A stale/suspended-account token returns 403; the check must retry anonymously."""
    with patch("hermes_cli.github_api.github_token", return_value="stale-tok"), \
         _patch_request_with([_http_error(403), '{"sha":"def"}']):
        result = _request("https://api.github.com/test")
    assert result == '{"sha":"def"}'


def test_403_with_no_token_raises():
    """Without a token, a 403 is a real access-denied and must not be swallowed."""
    with patch("hermes_cli.github_api.github_token", return_value=None), \
         _patch_request_with([_http_error(403)]):
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _request("https://api.github.com/test")
    assert exc_info.value.code == 403


def test_other_error_codes_are_not_swallowed():
    """A 404 or 500 must propagate regardless of token presence."""
    for code in (404, 500):
        with patch("hermes_cli.github_api.github_token", return_value="tok"), \
             _patch_request_with([_http_error(code)]):
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                _request("https://api.github.com/test")
            assert exc_info.value.code == code
