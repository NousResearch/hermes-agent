"""MiniMax OAuth source-ownership and refresh-serialization invariants."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


_WORKER = r"""
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

profile_home = Path(os.environ["HERMES_HOME"])
sync_dir = Path(os.environ["DR25_SYNC_DIR"])
worker_id = os.environ["DR25_WORKER_ID"]

import hermes_cli.auth as auth
from agent.credential_pool import load_pool


def fake_refresh(state, **_kwargs):
    marker = sync_dir / f"post-{os.getpid()}-{time.time_ns()}.json"
    marker.write_text(json.dumps({"refresh_token": state["refresh_token"]}), encoding="utf-8")
    time.sleep(0.15)
    now = datetime.now(timezone.utc)
    # MiniMax may omit refresh_token on a successful response. The existing
    # grant must survive that partial response.
    return {
        "access_token": "rotated-access",
        "obtained_at": now.isoformat(),
        "expires_at": (now + timedelta(hours=1)).isoformat(),
        "expires_in": 3600,
    }


auth.refresh_minimax_oauth_pure = fake_refresh
(sync_dir / f"ready-{worker_id}").write_text("ready", encoding="utf-8")
deadline = time.monotonic() + 10
while len(list(sync_dir.glob("ready-*"))) < 2:
    if time.monotonic() >= deadline:
        raise RuntimeError("workers did not reach the refresh barrier")
    time.sleep(0.01)

selected = load_pool("minimax-oauth").select()
(profile_home / "result.json").write_text(
    json.dumps(
        {
            "access_token": selected.access_token if selected else None,
            "refresh_token": selected.refresh_token if selected else None,
        }
    ),
    encoding="utf-8",
)
"""


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def test_minimax_cross_profile_refresh_has_one_winner_and_root_ownership(tmp_path):
    root = tmp_path / "hermes-root"
    profiles = [root / "profiles" / name for name in ("alpha", "beta")]
    sync_dir = tmp_path / "sync"
    fake_home = tmp_path / "native-home"
    sync_dir.mkdir()
    fake_home.mkdir()
    expired = datetime.now(timezone.utc) - timedelta(minutes=5)
    state = {
        "provider": "minimax-oauth",
        "portal_base_url": "https://token.invalid",
        "inference_base_url": "https://inference.invalid/anthropic",
        "client_id": "test-client",
        "access_token": "old-access",
        "refresh_token": "old-refresh",
        "expires_at": expired.isoformat(),
        "expires_in": 0,
    }
    row = {
        "id": "minimax-root",
        "label": "MiniMax OAuth",
        "auth_type": "oauth",
        "priority": 0,
        "source": "oauth",
        "access_token": "old-access",
        "refresh_token": "old-refresh",
        "expires_at": expired.isoformat(),
        "expires_at_ms": int(expired.timestamp() * 1000),
        "base_url": state["inference_base_url"],
    }
    _write_json(
        root / "auth.json",
        {
            "version": 1,
            "providers": {"minimax-oauth": state},
            "credential_pool": {"minimax-oauth": [row]},
        },
    )
    for profile in profiles:
        _write_json(
            profile / "auth.json",
            {"version": 1, "active_provider": "openrouter", "providers": {}},
        )

    processes = []
    for index, profile in enumerate(profiles):
        env = os.environ.copy()
        env.update({
            "HOME": str(fake_home),
            "HERMES_HOME": str(profile),
            "DR25_SYNC_DIR": str(sync_dir),
            "DR25_WORKER_ID": str(index),
        })
        processes.append(
            subprocess.Popen(
                [sys.executable, "-c", _WORKER],
                cwd=Path(__file__).resolve().parents[2],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )

    outputs = [process.communicate(timeout=20) for process in processes]
    assert [process.returncode for process in processes] == [0, 0], outputs
    assert len(list(sync_dir.glob("post-*.json"))) == 1

    for profile in profiles:
        assert _read_json(profile / "result.json") == {
            "access_token": "rotated-access",
            "refresh_token": "old-refresh",
        }
        profile_store = _read_json(profile / "auth.json")
        assert profile_store["active_provider"] == "openrouter"
        assert "minimax-oauth" not in profile_store.get("providers", {})
        assert "minimax-oauth" not in profile_store.get("credential_pool", {})
        serialized = json.dumps(profile_store)
        assert "old-access" not in serialized
        assert "old-refresh" not in serialized
        assert "rotated-access" not in serialized

    root_store = _read_json(root / "auth.json")
    root_state = root_store["providers"]["minimax-oauth"]
    assert root_state["access_token"] == "rotated-access"
    assert root_state["refresh_token"] == "old-refresh"
    root_row = root_store["credential_pool"]["minimax-oauth"][0]
    assert root_row["access_token"] == "rotated-access"
    assert root_row["refresh_token"] == "old-refresh"


def test_minimax_borrowed_row_is_pruned_when_root_source_disappears(
    tmp_path, monkeypatch
):
    root = tmp_path / "hermes-root"
    profile = root / "profiles" / "alpha"
    fake_home = tmp_path / "native-home"
    fake_home.mkdir()
    state = {
        "provider": "minimax-oauth",
        "portal_base_url": "https://token.invalid",
        "inference_base_url": "https://inference.invalid/anthropic",
        "client_id": "test-client",
        "access_token": "root-access",
        "refresh_token": "root-refresh",
        "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        "expires_in": 3600,
    }
    _write_json(
        root / "auth.json",
        {"version": 1, "providers": {"minimax-oauth": state}},
    )
    _write_json(
        profile / "auth.json",
        {"version": 1, "active_provider": "openrouter", "providers": {}},
    )
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("HERMES_HOME", str(profile))

    import hermes_constants
    import hermes_cli.auth as auth
    from agent.credential_pool import load_pool

    hermes_constants._default_hermes_root_memo = None
    auth._global_auth_store_cache = None
    auth._oauth_heal_clean_marks.clear()

    seeded = load_pool("minimax-oauth")
    assert len(seeded.entries()) == 1
    seeded_entry = seeded.entries()[0]
    assert seeded._borrowed_root_ids == {seeded_entry.id}
    assert "credential_pool" not in _read_json(profile / "auth.json")

    # A root process may have materialized the singleton row in its pool. Once
    # the provider block disappears, that borrowed row is stale authority.
    root_store = _read_json(root / "auth.json")
    root_store["credential_pool"] = {"minimax-oauth": [seeded_entry.to_dict()]}
    root_store["providers"].pop("minimax-oauth")
    _write_json(root / "auth.json", root_store)
    auth._global_auth_store_cache = None
    auth._oauth_heal_clean_marks.clear()

    with pytest.raises(auth.AuthError) as removed_error:
        auth._refresh_minimax_oauth_state(
            dict(state),
            source_path=root / "auth.json",
            force=True,
        )
    assert removed_error.value.code == "not_logged_in"
    assert "minimax-oauth" not in _read_json(profile / "auth.json").get("providers", {})

    reloaded = load_pool("minimax-oauth")

    assert reloaded.entries() == []
    root_after = _read_json(root / "auth.json")
    assert "minimax-oauth" not in root_after.get("credential_pool", {})
    profile_after = _read_json(profile / "auth.json")
    assert profile_after["active_provider"] == "openrouter"
    assert "credential_pool" not in profile_after

    # A terminal refresh failure quarantines the authoritative root state and
    # never changes the borrowing profile's selected provider.
    _write_json(
        root / "auth.json",
        {
            "version": 1,
            "providers": {"minimax-oauth": dict(state)},
            "credential_pool": {"minimax-oauth": [seeded_entry.to_dict()]},
        },
    )
    auth._global_auth_store_cache = None
    auth._oauth_heal_clean_marks.clear()

    def terminal_refresh(_state, **_kwargs):
        raise auth.AuthError(
            "invalid_grant",
            provider="minimax-oauth",
            code="invalid_grant",
            relogin_required=True,
        )

    monkeypatch.setattr(auth, "refresh_minimax_oauth_pure", terminal_refresh)
    terminal_pool = load_pool("minimax-oauth")
    terminal_entry = terminal_pool.entries()[0]

    assert terminal_pool._refresh_entry(terminal_entry, force=True) is None
    root_quarantined = _read_json(root / "auth.json")
    quarantined = root_quarantined["providers"]["minimax-oauth"]
    assert "access_token" not in quarantined
    assert "refresh_token" not in quarantined
    assert quarantined["last_auth_error"]["relogin_required"] is True
    assert "minimax-oauth" not in root_quarantined.get("credential_pool", {})
    profile_quarantined = _read_json(profile / "auth.json")
    assert profile_quarantined["active_provider"] == "openrouter"
    assert "minimax-oauth" not in profile_quarantined.get("providers", {})

    # If the upstream rotation succeeds but its first authoritative write
    # fails, the new bearer is never returned; a retry persists quarantine.
    _write_json(
        root / "auth.json",
        {
            "version": 1,
            "providers": {"minimax-oauth": dict(state)},
            "credential_pool": {"minimax-oauth": [seeded_entry.to_dict()]},
        },
    )
    auth._global_auth_store_cache = None
    auth._oauth_heal_clean_marks.clear()

    def rotated_refresh(_state, **_kwargs):
        return {
            "access_token": "must-not-escape",
            "refresh_token": "must-not-persist",
            "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            "expires_in": 3600,
        }

    real_save_to_source = auth._save_provider_state_to_source
    save_calls = {"count": 0}

    def fail_first_save(*args, **kwargs):
        save_calls["count"] += 1
        if save_calls["count"] == 1:
            raise OSError("simulated atomic-write failure")
        return real_save_to_source(*args, **kwargs)

    monkeypatch.setattr(auth, "refresh_minimax_oauth_pure", rotated_refresh)
    monkeypatch.setattr(auth, "_save_provider_state_to_source", fail_first_save)
    failed_write_pool = load_pool("minimax-oauth")
    failed_write_entry = failed_write_pool.entries()[0]

    assert failed_write_pool._refresh_entry(failed_write_entry, force=True) is None
    assert save_calls["count"] >= 2
    root_after_failed_write = _read_json(root / "auth.json")
    serialized_root = json.dumps(root_after_failed_write)
    assert "must-not-escape" not in serialized_root
    assert "must-not-persist" not in serialized_root
    failed_state = root_after_failed_write["providers"]["minimax-oauth"]
    assert failed_state["last_auth_error"]["code"] == "credential_persist_failed"
    assert "access_token" not in failed_state
    assert "refresh_token" not in failed_state
    assert "minimax-oauth" not in root_after_failed_write.get("credential_pool", {})
