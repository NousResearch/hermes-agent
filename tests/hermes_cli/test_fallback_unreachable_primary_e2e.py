"""An unreachable primary walks ``fallback_providers`` through the real resolution path.

No resolver mocks: a Nous OAuth login with an expired invoke JWT is seeded into a temp
``HERMES_HOME`` and its portal points at a closed local port, so the real token refresh raises a
real ``httpx.ConnectError`` -- the shape of a Nous Portal outage. The fallback entry resolves from
a real ``OPENROUTER_API_KEY``.
"""

import base64
import json
import socket
import time

import pytest

from hermes_cli.auth import _auth_store_lock, _load_auth_store, _save_auth_store
from hermes_cli.fallback_config import is_transient_provider_resolve_error
from hermes_cli.runtime_provider import resolve_runtime_with_fallback

_FALLBACK_MODEL = "deepseek/deepseek-v4-flash"
_CFG = {
    "model": {"provider": "nous", "default": _FALLBACK_MODEL},
    "fallback_providers": [{"provider": "openrouter", "model": _FALLBACK_MODEL}],
}


def _expired_jwt() -> str:
    def seg(obj):
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()
    payload = {"sub": "nas_user:1", "client_id": "hermes-cli", "scope": "inference:invoke",
               "exp": int(time.time()) - 600}
    return f"{seg({'alg': 'RS256'})}.{seg(payload)}.sig"


def _closed_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def unreachable_nous_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared-store"))
    monkeypatch.setenv("HERMES_NOUS_TIMEOUT_SECONDS", "2")
    for var in ("NOUS_API_KEY", "HERMES_PORTAL_BASE_URL", "NOUS_PORTAL_BASE_URL", "HERMES_INFERENCE_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-fallback")
    expired = _expired_jwt()
    with _auth_store_lock():
        store = _load_auth_store()
        store.setdefault("providers", {})["nous"] = {
            "auth_method": "oauth", "client_id": "hermes-cli", "scope": "inference:invoke",
            "access_token": expired, "agent_key": expired, "refresh_token": "rt-test",
            "expires_at": "2000-01-01T00:00:00+00:00", "agent_key_expires_at": "2000-01-01T00:00:00+00:00",
            "portal_base_url": f"http://127.0.0.1:{_closed_port()}",
            "inference_base_url": "https://inference-api.nousresearch.com/v1",
            "tls": {"insecure": False, "ca_bundle": None},
        }
        store["active_provider"] = "nous"
        _save_auth_store(store)
    return home


def test_unreachable_nous_portal_is_a_transient_resolve_error(unreachable_nous_home):
    """Control: without a chain the primary's real failure surfaces, and it is a network error,
    not an AuthError -- the case the AuthError-only walkers used to miss."""
    with pytest.raises(Exception) as excinfo:
        resolve_runtime_with_fallback({"model": _CFG["model"]}, requested="nous")
    assert is_transient_provider_resolve_error(excinfo.value), repr(excinfo.value)


def test_unreachable_nous_portal_walks_to_openrouter(unreachable_nous_home, caplog):
    with caplog.at_level("WARNING", logger="hermes_cli.runtime_provider"):
        runtime, entry = resolve_runtime_with_fallback(_CFG, requested="nous")

    assert runtime["provider"] == "openrouter"
    assert runtime["api_key"] == "sk-or-test-fallback"
    assert entry["model"] == _FALLBACK_MODEL
    assert any("unreachable" in r.getMessage() for r in caplog.records)
