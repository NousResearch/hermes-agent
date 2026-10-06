"""Additional offline regression coverage for #123746 / #123764.

Exercise the public resolver with a pool-only auth store, never a real login.
All credential values are random per-run placeholders written only to tmp_path.
"""

import json
import socket
import uuid

import pytest

from hermes_cli.auth import AuthError, resolve_codex_runtime_credentials


@pytest.mark.parametrize("read_only", [False, True])
@pytest.mark.parametrize("has_live_row", [False, True])
@pytest.mark.parametrize("reason", ["token_revoked", "token_invalidated"])
def test_terminal_dead_row_is_never_selected(
    tmp_path, monkeypatch, read_only, has_live_row, reason
):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "empty-codex"))
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_AUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)

    def no_network(*_args, **_kwargs):
        raise AssertionError("credential resolution must remain offline")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", no_network)
    monkeypatch.setattr(socket.socket, "sendto", no_network)
    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    revoked_access, revoked_refresh = uuid.uuid4().hex, uuid.uuid4().hex
    live_access, live_refresh = uuid.uuid4().hex, uuid.uuid4().hex
    rows = [
        {
            "id": "revoked",
            "source": "manual:device_code",
            "auth_type": "oauth",
            "access_token": revoked_access,
            "refresh_token": revoked_refresh,
            "base_url": "https://revoked.invalid/v1",
            "last_status": "dead",
            "last_error_code": 401,
            "last_error_reason": reason,
            # DEAD must stay terminal even after a previous cooldown has expired.
            "last_error_reset_at": 1,
        }
    ]
    if has_live_row:
        rows.append({
            "id": "live",
            "source": "manual:device_code",
            "auth_type": "oauth",
            "access_token": live_access,
            "refresh_token": live_refresh,
            "base_url": "https://live.invalid/v1",
            "last_status": "active",
        })
    store = home / "auth.json"
    store.write_text(
        json.dumps({
            "version": 1,
            "providers": {},
            "credential_pool": {"openai-codex": rows},
        })
    )
    before = store.read_bytes()
    if has_live_row:
        result = resolve_codex_runtime_credentials(
            read_only=read_only, refresh_if_expiring=False
        )
        assert result["api_key"] == live_access
        assert result["base_url"] == "https://live.invalid/v1"
    else:
        with pytest.raises(AuthError) as exc:
            resolve_codex_runtime_credentials(
                read_only=read_only, refresh_if_expiring=False
            )
        assert exc.value.relogin_required is True
    assert store.read_bytes() == before
    if read_only:
        assert not (home / "auth.lock").exists()
