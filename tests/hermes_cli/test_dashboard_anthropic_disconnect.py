"""Dashboard Anthropic disconnect semantics under HERMES_AUTH_HOME.

Disconnect uses the pinned Anthropic store lock and direct unlink, clears the
auth-store/pool entry only after the file mutation succeeded, and reports
missing files, lock timeouts, and unlink failures truthfully — it never
claims a clean disconnect while the live file remains.
"""

from __future__ import annotations

import contextlib
import functools
import json
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def residence(monkeypatch, tmp_path):
    residence = tmp_path / "auth-residence"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "runtime"))
    monkeypatch.setenv("HERMES_AUTH_HOME", str(residence))
    return residence


@pytest.fixture
def client():
    from hermes_cli.web_server import _SESSION_TOKEN, app

    return TestClient(app, headers={"X-Hermes-Session-Token": _SESSION_TOKEN})


def _seed_anthropic(residence: Path) -> Path:
    from hermes_cli.auth import _auth_store_lock, _load_auth_store, _save_auth_store

    residence.mkdir(parents=True, exist_ok=True)
    oauth_file = residence / ".anthropic_oauth.json"
    oauth_file.write_text(
        json.dumps({"accessToken": "tok", "refreshToken": "ref", "expiresAt": 1}),
        encoding="utf-8",
    )
    with _auth_store_lock():
        store = _load_auth_store()
        store.setdefault("providers", {})["anthropic"] = {"api_key": "sk-ant-live"}
        store.setdefault("credential_pool", {})["anthropic"] = [
            {
                "id": "pool-1",
                "source": "manual",
                "auth_type": "api_key",
                "label": "dashboard",
                "access_token": "sk-ant-live",
                "priority": 0,
            }
        ]
        _save_auth_store(store)
    return oauth_file


def _store_state(residence: Path) -> dict:
    auth_file = residence / "auth.json"
    if not auth_file.exists():
        return {}
    return json.loads(auth_file.read_text(encoding="utf-8"))


def test_disconnect_unlinks_file_then_clears_store(residence, client):
    oauth_file = _seed_anthropic(residence)

    response = client.delete("/api/providers/oauth/anthropic")

    assert response.status_code == 200
    assert response.json() == {"ok": True, "provider": "anthropic"}
    assert not oauth_file.exists()
    state = _store_state(residence)
    assert "anthropic" not in state.get("providers", {})
    assert "anthropic" not in state.get("credential_pool", {})


def test_disconnect_with_nothing_to_clear_reports_ok_false(residence, client):
    response = client.delete("/api/providers/oauth/anthropic")

    assert response.status_code == 200
    assert response.json() == {"ok": False, "provider": "anthropic"}


def test_endpoint_disconnect_serializes_with_the_dashboard_save(residence, client):
    """The dashboard save path and the disconnect endpoint share one lock.

    While the store lock is held, the real dashboard save (re-entrant on
    this thread) lands its grant and the endpoint's disconnect stays queued;
    after release the disconnect removes exactly that grant. Without the
    shared lock the disconnect would run first, report nothing to clear, and
    strand the freshly minted credential on disk — ``ok: true`` here is only
    reachable when the save completed before the unlink.
    """
    import hermes_cli.auth as auth
    from hermes_cli.web_server import _save_anthropic_oauth_creds

    residence.mkdir(parents=True, exist_ok=True)
    oauth_file = residence / ".anthropic_oauth.json"
    result: dict = {}

    def disconnect_call() -> None:
        result["response"] = client.delete("/api/providers/oauth/anthropic")

    with contextlib.ExitStack() as stack:
        stack.enter_context(auth.anthropic_oauth_store_lock())
        thread = threading.Thread(target=disconnect_call, daemon=True)
        thread.start()
        _save_anthropic_oauth_creds("late-access", "late-refresh", 1)
        grant = json.loads(oauth_file.read_text(encoding="utf-8"))
        assert grant["accessToken"] == "late-access"
    thread.join(timeout=30)
    assert not thread.is_alive()

    response = result["response"]
    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert not oauth_file.exists()


def test_disconnect_timeout_is_reported_and_leaves_everything_intact(
    residence, client, monkeypatch
):
    import hermes_cli.auth as auth

    oauth_file = _seed_anthropic(residence)
    monkeypatch.setattr(
        auth,
        "remove_anthropic_oauth_store",
        functools.partial(auth.remove_anthropic_oauth_store, timeout_seconds=1.2),
    )

    entered = threading.Event()
    release = threading.Event()

    def holder() -> None:
        with auth.anthropic_oauth_store_lock():
            entered.set()
            release.wait(timeout=30)

    thread = threading.Thread(target=holder, daemon=True)
    thread.start()
    assert entered.wait(timeout=10)
    try:
        response = client.delete("/api/providers/oauth/anthropic")
    finally:
        release.set()
        thread.join(timeout=10)

    assert response.status_code == 500
    assert "Timed out" in response.json()["detail"]
    assert oauth_file.exists()
    state = _store_state(residence)
    assert "anthropic" in state["providers"]
    assert "anthropic" in state["credential_pool"]


def test_disconnect_unlink_failure_is_reported_and_pool_is_untouched(
    residence, client
):
    oauth_file = _seed_anthropic(residence)
    # Replace the store with a directory: unlink raises without any chmod
    # tricks, portably.
    oauth_file.unlink()
    oauth_file.mkdir()

    response = client.delete("/api/providers/oauth/anthropic")

    assert response.status_code == 500
    assert oauth_file.exists()
    state = _store_state(residence)
    assert "anthropic" in state["providers"]
    assert "anthropic" in state["credential_pool"]


def test_managed_file_api_treats_the_residence_as_sensitive(monkeypatch, tmp_path):
    from hermes_cli.web_server import _is_sensitive_path

    residence = tmp_path / "auth-residence"
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(runtime))
    monkeypatch.setenv("HERMES_AUTH_HOME", str(residence))

    for rel in (
        "auth.json",
        "auth.lock",
        "auth.json.corrupt",
        "auth.json.tmp.4242.deadbeef",
        ".anthropic_oauth.json",
        "shared/nous_auth.json",
        "profiles/other/auth.json",
        "unclassified-note.txt",
    ):
        assert _is_sensitive_path(residence / rel), rel

    assert _is_sensitive_path(runtime / "auth.json")
    assert not _is_sensitive_path(runtime / "notes.txt")

    # Path-equal override: the runtime tree is not blanket-sensitive.
    monkeypatch.setenv("HERMES_AUTH_HOME", str(runtime))
    assert not _is_sensitive_path(runtime / "notes.txt")
    assert _is_sensitive_path(runtime / "auth.json")
