"""Native grants must not widen authentication or unattended access."""
import sys
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

import pytest

from agent.vault_backends import unlock
from agent.vault_backends.onepassword import OnePasswordLoginBackend


def test_live_native_grant_cannot_prompt_from_headless_sibling(monkeypatch):
    backend = OnePasswordLoginBackend()
    started = unlock.time.monotonic()
    unlock.store_session_token("onepassword", "", context="")
    monkeypatch.setattr(unlock.time, "monotonic", lambda: started + unlock._IDLE_TTL_S - 1)
    try:
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        with patch("agent.vault_backends.onepassword.run_cli", return_value=CompletedProcess([], 0, "[]", "")) as call:
            with pytest.raises(RuntimeError, match="interactive"):
                backend.list_items()
            call.assert_not_called()
        assert backend.is_unlocked(), "refusing the sibling must not revoke the interactive grant"
        monkeypatch.setattr(unlock.time, "monotonic", lambda: started + unlock._IDLE_TTL_S + 1)
        assert not backend.is_unlocked(), "an unattended attempt must not keep a native grant alive"
    finally:
        unlock.lock("onepassword")


def test_native_environment_cannot_adopt_connect_credentials(monkeypatch):
    from agent.secret_scope import reset_secret_scope, set_secret_scope
    scope = set_secret_scope({"OP_CONNECT_HOST": "https://connect.example.com", "OP_CONNECT_TOKEN": "fixture-connect"})
    monkeypatch.setenv("OP_SESSION_example", "fixture-ambient-session")
    try:
        backend = OnePasswordLoginBackend()
        child = backend._env("")
        assert not any(k.startswith("OP_SESSION") or k.startswith("OP_CONNECT") for k in child)
        assert "OP_SERVICE_ACCOUNT_TOKEN" not in child
    finally:
        reset_secret_scope(scope)


def test_native_grant_is_bound_to_configured_account():
    a = OnePasswordLoginBackend({"account": "one.example.1password.com"})
    b = OnePasswordLoginBackend({"account": "two.example.1password.com"})
    # Use the real unlock boundary with a metadata-only CLI response, not a forged grant.
    with patch.object(OnePasswordLoginBackend, "_op", return_value=Path(sys.executable)), \
         patch("agent.vault_backends.onepassword.run_cli", return_value=CompletedProcess([], 0, "[]", "")), \
         patch("agent.vault_backends.onepassword.run_with_stdin_secret", return_value=CompletedProcess([], 0, "", "")):
        a.unlock("")
    try:
        assert a.is_unlocked()
        assert not b.is_unlocked()
        assert a.is_unlocked(), "an account mismatch must not revoke the original grant"
    finally:
        unlock.lock("onepassword")


def test_headless_mode_is_validated_on_the_same_snapshot_that_is_used(monkeypatch):
    backend = OnePasswordLoginBackend()
    unlock.store_session_token("onepassword", "fixture-manual", context="")
    get_token = unlock.get_session_token
    reads = []

    def replace_after_read(*args, **kwargs):
        value = get_token(*args, **kwargs)
        reads.append(value)
        if len(reads) == 1:
            unlock.lock("onepassword")
            unlock.store_session_token("onepassword", "", context="")
        return value

    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    monkeypatch.setattr(unlock, "get_session_token", replace_after_read)
    monkeypatch.setattr(backend, "_op", lambda: Path(sys.executable))
    try:
        with patch("agent.vault_backends.onepassword.run_cli", return_value=CompletedProcess([], 0, "[]", "")) as call:
            assert backend.list_items() == []
        assert call.call_args.kwargs["env"].get("OP_SESSION") == "fixture-manual", \
            "a manual snapshot must never be replaced by native auth between validation and use"
    finally:
        unlock.lock("onepassword")
