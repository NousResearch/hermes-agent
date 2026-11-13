"""A2A inbound auth posture from the tokens a gateway reads at startup (#148961).

A token kept only in the profile ``.env`` must turn auth on, and a token the
operator set but that parses to no ``name:token`` pair must refuse every caller
instead of reading as "no token configured" (which serves every caller).
"""
from __future__ import annotations

import pytest

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


def test_unusable_peer_tokens_preserve_shared_bearer_and_warn(monkeypatch, caplog):
    """A bad per-peer entry must not disable a working shared credential or misreport it as broken."""
    import asyncio
    import logging
    import urllib.error

    from tests.plugins.test_a2a_plugin import _make_live_adapter, _post_json, _send_body

    monkeypatch.setenv("A2A_PEER_TOKENS", "vibe_abc123")
    monkeypatch.setenv("A2A_BEARER_TOKEN", "shared-secret-123")
    monkeypatch.setenv("A2A_HOST", "127.0.0.1")
    monkeypatch.delenv("A2A_TRUSTED_PEERS", raising=False)
    monkeypatch.delenv("A2A_ALLOW_ALL_USERS", raising=False)
    adapter, base = _make_live_adapter(monkeypatch)

    async def exercise():
        with caplog.at_level(logging.ERROR, logger="plugins.platforms.a2a.adapter"):
            assert await adapter.connect() is True
        try:
            def unauthenticated_status():
                try:
                    _post_json(base + "/", _send_body("without token"))
                except urllib.error.HTTPError as exc:
                    return exc.code
                return 200

            assert await asyncio.to_thread(unauthenticated_status) == 401
            response = await asyncio.to_thread(
                _post_json, base + "/", _send_body("shared bearer"),
                {"Authorization": "Bearer shared-secret-123"},
            )
            assert response["result"]["status"]["state"] == "TASK_STATE_COMPLETED"
        finally:
            await adapter.disconnect()

    asyncio.run(exercise())
    errors = [
        rec.getMessage() for rec in caplog.records
        if rec.name == "plugins.platforms.a2a.adapter" and rec.levelno >= logging.ERROR
    ]
    assert any("only the shared A2A_BEARER_TOKEN is accepted" in msg for msg in errors)
    assert not any("every request is refused" in msg for msg in errors)


def test_setup_refuses_to_save_peer_tokens_the_runtime_cannot_parse(monkeypatch):
    """The wizard must not persist a value that ``capture()`` turns into refuse-every-caller."""
    import hermes_cli.setup as setup_mod
    from plugins.platforms import a2a

    for answer, expected in (("vibe_abc123", None), ("alice:tok-a", "alice:tok-a")):
        saved: dict[str, str] = {}
        warnings: list[str] = []
        answers = iter(["", "", answer, "", ""])  # port, name, peer tokens, shared bearer, bind host
        monkeypatch.setattr(setup_mod, "prompt", lambda *_a, **_kw: next(answers))
        monkeypatch.setattr(setup_mod, "prompt_yes_no", lambda *_a, **_kw: True)
        monkeypatch.setattr(setup_mod, "get_env_value", lambda _name: "")
        monkeypatch.setattr(setup_mod, "save_env_value", lambda k, v: saved.__setitem__(k, v))
        monkeypatch.setattr(setup_mod, "print_warning", warnings.append)
        a2a.interactive_setup()
        assert saved.get("A2A_PEER_TOKENS") == expected
        assert any("name:token" in w for w in warnings) is (expected is None)


@pytest.mark.parametrize("stored", [{"A2A_PEER_TOKENS": "vibe_abc123"}, {"A2A_BEARER_TOKEN": "shared"}])
def test_setup_rerun_does_not_claim_localhost_only_over_a_stored_token(monkeypatch, stored):
    """Enter at every prompt over a stored token: it stays in .env, so the wizard can't say localhost-only."""
    import hermes_cli.setup as setup_mod
    from plugins.platforms import a2a

    warnings: list[str] = []
    monkeypatch.setattr(setup_mod, "prompt", lambda _q, default=None, **_kw: default or "")
    monkeypatch.setattr(setup_mod, "prompt_yes_no", lambda *_a, **_kw: True)
    monkeypatch.setattr(setup_mod, "get_env_value", stored.get)
    monkeypatch.setattr(setup_mod, "save_env_value", lambda *_a: None)
    monkeypatch.setattr(setup_mod, "print_warning", warnings.append)
    a2a.interactive_setup()
    assert not any("localhost-only" in w for w in warnings)
    assert any("stay in effect" in w for w in warnings)
