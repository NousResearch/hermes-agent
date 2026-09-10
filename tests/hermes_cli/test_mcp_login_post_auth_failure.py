"""Regression: `hermes mcp login` must not call a post-auth failure "auth failed".

``_reauth_oauth_server`` wrapped the whole probe in one ``except`` that
reported every exception as ``Authentication failed:``. That is wrong whenever
the OAuth round-trip actually SUCCEEDED and the failure came later, in the MCP
session itself.

Observed live with GitLab: the browser flow completed and cached a valid
``scope: mcp`` token with a refresh token, then ``POST /api/v4/mcp`` returned
an application-level 404 (the MCP endpoint is gated behind GitLab Duo settings
that only exist on group namespaces). The MCP SDK surfaces that as the generic
``Session terminated``. Hermes printed "Authentication failed: Session
terminated", so the user re-ran the browser flow repeatedly against a server
that had already issued them a perfectly good token.

The discriminator is already on disk: if a token is cached, auth is not the
failing stage.
"""

import pytest

import hermes_cli.mcp_config as mcp_config


SERVER = {"url": "https://gitlab.com/api/v4/mcp", "auth": "oauth"}


@pytest.fixture
def captured(monkeypatch):
    """Capture _error/_info/_warning/_success output instead of printing."""
    lines = {"error": [], "info": [], "warning": [], "success": []}
    for level in lines:
        monkeypatch.setattr(
            mcp_config,
            f"_{level}",
            lambda msg, _l=level: lines[_l].append(str(msg)),
        )
    monkeypatch.setattr(mcp_config, "print", lambda *a, **k: None, raising=False)

    # Neutralize the OAuth-state wipe and the interactive-OAuth context manager.
    class _NullCtx:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    import tools.mcp_oauth as mcp_oauth

    monkeypatch.setattr(mcp_oauth, "force_interactive_oauth", lambda: _NullCtx())
    return lines


def _fail_probe(*args, **kwargs):
    raise RuntimeError("Session terminated")


def test_post_auth_failure_is_not_reported_as_auth_failure(captured, monkeypatch):
    """Token on disk + probe raises => report a SESSION failure, not auth."""
    monkeypatch.setattr(mcp_config, "_probe_single_server", _fail_probe)
    monkeypatch.setattr(mcp_config, "_oauth_tokens_present", lambda name: True)

    ok = mcp_config._reauth_oauth_server("gitlab", SERVER)

    assert ok is False
    joined = " ".join(captured["error"])
    assert "Authentication failed" not in joined, (
        "a post-auth session failure must not be labelled an auth failure"
    )
    assert "Session terminated" in joined, "the real error must still be shown"
    # And the user must be told re-running login is pointless.
    guidance = " ".join(captured["info"])
    assert "SUCCEEDED" in guidance
    assert "will not help" in guidance


def test_genuine_auth_failure_still_reported_as_auth_failure(captured, monkeypatch):
    """No token on disk + probe raises => the original message is correct."""
    monkeypatch.setattr(mcp_config, "_probe_single_server", _fail_probe)
    monkeypatch.setattr(mcp_config, "_oauth_tokens_present", lambda name: False)

    ok = mcp_config._reauth_oauth_server("someserver", SERVER)

    assert ok is False
    joined = " ".join(captured["error"])
    assert "Authentication failed" in joined
    # No misleading "auth succeeded" guidance on a real auth failure.
    assert "SUCCEEDED" not in " ".join(captured["info"])


def test_token_path_hint_never_raises():
    """Display helper must degrade to a sane string, never propagate."""
    hint = mcp_config._token_path_hint("nonexistent-server-xyz")
    assert isinstance(hint, str) and hint
    assert "nonexistent-server-xyz" in hint


def test_success_path_unchanged(captured, monkeypatch):
    """A clean probe with a cached token still reports authenticated."""
    monkeypatch.setattr(
        mcp_config, "_probe_single_server", lambda *a, **k: [("t1", "desc")]
    )
    monkeypatch.setattr(mcp_config, "_oauth_tokens_present", lambda name: True)

    ok = mcp_config._reauth_oauth_server("gitlab", SERVER)

    assert ok is True
    assert "Authenticated" in " ".join(captured["success"])
    assert not captured["error"]
