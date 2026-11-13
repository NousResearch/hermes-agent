"""A2A inbound auth posture from the tokens a gateway reads at startup (#148961).

A token kept only in the profile ``.env`` must turn auth on, and a token the
operator set but that parses to no ``name:token`` pair must refuse every caller
instead of reading as "no token configured" (which serves every caller).
"""
from __future__ import annotations

from plugins.platforms.a2a import security


def test_token_in_the_profile_env_only_turns_auth_on(tmp_path, monkeypatch):
    import hermes_constants
    from gateway.platforms import _shared

    monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)
    monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
    monkeypatch.setattr(_shared, "_UNSCOPED_PROFILE_SECRETS", None)  # rebuild for this home
    home = tmp_path / "gateway-home"
    home.mkdir()
    (home / ".env").write_text("A2A_PEER_TOKENS=alice:tok-a\n")
    token = hermes_constants.set_hermes_home_override(str(home))
    try:
        ctx = security.A2ASecurityContext.capture()
        assert ctx.localhost_only() is False
        assert ctx.authenticate("Bearer tok-a", "10.0.0.8") == "alice"
        assert ctx.authenticate("Bearer wrong", "10.0.0.8") is None
        assert ctx.authenticate(None, "10.0.0.8") is None
    finally:
        hermes_constants.reset_hermes_home_override(token)


def test_peer_tokens_without_a_name_fail_closed(monkeypatch):
    # A bare token (no ``name:``) is a request for auth that parses to zero pairs.
    monkeypatch.setenv("A2A_PEER_TOKENS", "vibe_abc123")
    monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
    ctx = security.A2ASecurityContext.capture()
    assert ctx.localhost_only() is False
    assert ctx.authenticate(None, "127.0.0.1") is None
    assert ctx.authenticate("Bearer vibe_abc123", "127.0.0.1") is None
    monkeypatch.setenv("A2A_HOST", "0.0.0.0")
    assert security.A2ASecurityContext.capture().resolve_bind_host() == "127.0.0.1"
