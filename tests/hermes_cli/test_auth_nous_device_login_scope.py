"""Device-code login must resolve the operator's URL overrides through the profile secret scope.

The same ``NOUS_INFERENCE_BASE_URL`` that ``PooledCredential.runtime_base_url`` re-reads on
every access used to be captured at login time by a raw ``os.getenv`` and persisted as the
credential's ``inference_base_url`` — including the launch profile's process-wide value when
the login ran for a multiplexed secondary. Every other reader in ``auth_nous`` resolves through
``_nous_inference_env_override()`` / ``_nous_portal_env_override()``; the login path must not be
the one unscoped reader left (#121339).
"""
from __future__ import annotations

import contextlib

from agent import secret_scope

# Dummy fixture tokens (never real credentials).
_AT = "dummy-login-access-token"
_RT = "dummy-login-refresh-token"

_DEVICE = {
    "verification_uri_complete": "https://portal.example/verify?user_code=ABCD-EFGH",
    "user_code": "ABCD-EFGH",
    "expires_in": 300,
    "interval": 1,
    "device_code": "dummy-device-code",
}
_TOKEN = {
    "access_token": _AT,
    "refresh_token": _RT,
    "scope": "inference",
    "token_type": "Bearer",
    "expires_in": 3600,
}


def _drive_login(monkeypatch):
    """Run the device-code flow with the HTTP layer mocked; return (request_kwargs, state)."""
    import hermes_cli.auth as auth_mod
    import hermes_cli.auth_nous as auth_nous

    captured = {}

    def fake_request_device_code(**kwargs):
        captured.update(kwargs)
        return dict(_DEVICE)

    monkeypatch.setattr(auth_nous, "_nous_http_client", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(auth_mod, "_request_device_code", fake_request_device_code)
    monkeypatch.setattr(auth_mod, "_print_device_code_instructions", lambda *a, **k: None)
    monkeypatch.setattr(auth_mod, "_poll_for_token", lambda **kwargs: dict(_TOKEN))
    monkeypatch.setattr(auth_mod, "_is_remote_session", lambda: True)
    monkeypatch.setattr(
        auth_mod, "refresh_nous_oauth_from_state", lambda state, **kwargs: dict(state))
    state = auth_nous._nous_device_code_login(open_browser=False)
    return captured, state


def _pconfig():
    from hermes_cli.auth import PROVIDER_REGISTRY
    return PROVIDER_REGISTRY["nous"]


def test_single_profile_env_override_reaches_login_state(monkeypatch):
    """CLI / single-profile process: the environ override IS the profile's own value — the
    persisted state and the device-code request must keep honouring it (behaviour unchanged)."""
    monkeypatch.setenv("NOUS_INFERENCE_BASE_URL", "https://nous.mine.example/v1/")
    monkeypatch.setenv("HERMES_PORTAL_BASE_URL", "https://portal.mine.example")

    request_kwargs, state = _drive_login(monkeypatch)

    assert state["inference_base_url"] == "https://nous.mine.example/v1"
    assert state["portal_base_url"] == "https://portal.mine.example"
    assert request_kwargs["portal_base_url"] == "https://portal.mine.example"


def test_multiplex_secondary_does_not_persist_launch_profile_override(monkeypatch):
    """Multiplexed secondary whose scope carries no override: the launch profile's process-wide
    value must NOT be persisted into this credential — the resolver is simply absent and the
    provider defaults apply. A raw os.getenv here would re-introduce the exact cross-profile
    token-routing this PR exists to prevent."""
    monkeypatch.setenv("NOUS_INFERENCE_BASE_URL", "https://nous.launch.example/v1")
    monkeypatch.setenv("HERMES_PORTAL_BASE_URL", "https://portal.launch.example")
    pconfig = _pconfig()
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({"SOME_OTHER_SECRET": "x"})
    try:
        request_kwargs, state = _drive_login(monkeypatch)
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(False)

    assert state["inference_base_url"] == pconfig.inference_base_url.rstrip("/")
    assert state["portal_base_url"] == pconfig.portal_base_url.rstrip("/")
    assert request_kwargs["portal_base_url"] == pconfig.portal_base_url.rstrip("/")
    assert "launch.example" not in state["inference_base_url"]
    assert "launch.example" not in state["portal_base_url"]


def test_multiplex_scoped_override_reaches_login_state(monkeypatch):
    """Multiplexed secondary with its own scoped overrides: those are the profile's values and
    must flow into the persisted state exactly like the single-profile case."""
    monkeypatch.setenv("NOUS_INFERENCE_BASE_URL", "https://nous.launch.example/v1")
    monkeypatch.setenv("HERMES_PORTAL_BASE_URL", "https://portal.launch.example")
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({
        "NOUS_INFERENCE_BASE_URL": "https://nous.secondary.example/v1/",
        "HERMES_PORTAL_BASE_URL": "https://portal.secondary.example",
    })
    try:
        request_kwargs, state = _drive_login(monkeypatch)
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(False)

    assert state["inference_base_url"] == "https://nous.secondary.example/v1"
    assert state["portal_base_url"] == "https://portal.secondary.example"
    assert request_kwargs["portal_base_url"] == "https://portal.secondary.example"
    assert "launch.example" not in state["inference_base_url"]
