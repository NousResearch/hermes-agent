"""A2A bearer comparison must answer 401, never raise, whatever bytes a token holds (#150028).

hmac.compare_digest raises TypeError on non-ASCII str and UTF-8 encoding raises on lone
surrogates; either one used to kill the request thread with no response."""

from __future__ import annotations

from dataclasses import replace

import pytest

from plugins.platforms.a2a import security


def test_non_ascii_presented_token_is_rejected_not_raised(monkeypatch):
    """http.server decodes header bytes as latin-1, so one stray byte in the Authorization
    header arrives as a non-ASCII str."""
    monkeypatch.setenv("A2A_BEARER_TOKEN", "shared-tok")
    monkeypatch.setenv("A2A_PEER_TOKENS", "carol:tok-c")
    ctx = security.A2ASecurityContext.capture()
    assert ctx.authenticate("Bearer shared\x83tok", "1.1.1.1") is None
    assert ctx.authenticate("Bearer 토큰", "1.1.1.1") is None
    assert ctx.authenticate("Bearer shared-tok", "1.1.1.1") == "ip:1.1.1.1"
    assert ctx.authenticate("Bearer tok-c", "1.1.1.1") == "carol"


@pytest.mark.parametrize("as_peer", [False, True])
def test_surrogateescaped_configured_secret_returns_401(as_peer):
    """os.environ decodes undecodable bytes with surrogateescape, so a configured token can
    hold a lone surrogate (byte 0x83 -> \\udc83)."""
    bad_token = "secret\udc83"
    ctx = replace(
        security.A2ASecurityContext.capture(),
        bearer_token="" if as_peer else bad_token,
        peer_tokens=((bad_token, "alice"),) if as_peer else (),
    )
    assert ctx.authenticate("Bearer wrong", "1.1.1.1") is None
