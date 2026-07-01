import time
import json
import pytest
from agent.credential_pool import load_pool, STATUS_OK, STATUS_EXHAUSTED

def _write_auth_store(tmp_path, payload: dict) -> None:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps(payload, indent=2))

def test_model_specific_exhaustion_filters_selection(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "openrouter": [
                    {
                        "id": "cred-1",
                        "label": "primary",
                        "auth_type": "api_key",
                        "priority": 0,
                        "source": "manual",
                        "access_token": "key-1",
                        "last_status": "ok",
                    },
                    {
                        "id": "cred-2",
                        "label": "secondary",
                        "auth_type": "api_key",
                        "priority": 1,
                        "source": "manual",
                        "access_token": "key-2",
                        "last_status": "ok",
                    },
                ]
            },
        },
    )

    pool = load_pool("openrouter")
    
    # Mark cred-1 as exhausted specifically for gemini-2.5-flash
    cred1 = pool.entries()[0]
    pool._mark_exhausted(cred1, status_code=429, error_context={"reason": "rate_limit"}, model="google/gemini-2.5-flash")

    # select for gemini-2.5-flash should return cred-2
    entry_flash = pool.select(model="google/gemini-2.5-flash")
    assert entry_flash is not None
    assert entry_flash.id == "cred-2"

    # select for gemini-2.5-pro should still return cred-1
    entry_pro = pool.select(model="google/gemini-2.5-pro")
    assert entry_pro is not None
    assert entry_pro.id == "cred-1"

def test_global_exhaustion_filters_all_models(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "openrouter": [
                    {
                        "id": "cred-1",
                        "label": "primary",
                        "auth_type": "api_key",
                        "priority": 0,
                        "source": "manual",
                        "access_token": "key-1",
                        "last_status": "ok",
                    },
                    {
                        "id": "cred-2",
                        "label": "secondary",
                        "auth_type": "api_key",
                        "priority": 1,
                        "source": "manual",
                        "access_token": "key-2",
                        "last_status": "ok",
                    },
                ]
            },
        },
    )

    pool = load_pool("openrouter")
    
    # Mark cred-1 globally exhausted (no model parameter)
    cred1 = pool.entries()[0]
    pool._mark_exhausted(cred1, status_code=401, error_context={"reason": "auth_error"})

    # select for any model should return cred-2
    entry_flash = pool.select(model="google/gemini-2.5-flash")
    assert entry_flash is not None
    assert entry_flash.id == "cred-2"

    entry_pro = pool.select(model="google/gemini-2.5-pro")
    assert entry_pro is not None
    assert entry_pro.id == "cred-2"

def test_reset_statuses_clears_model_lockouts(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "openrouter": [
                    {
                        "id": "cred-1",
                        "label": "primary",
                        "auth_type": "api_key",
                        "priority": 0,
                        "source": "manual",
                        "access_token": "key-1",
                        "last_status": "ok",
                    }
                ]
            },
        },
    )

    pool = load_pool("openrouter")
    cred1 = pool.entries()[0]
    pool._mark_exhausted(cred1, status_code=429, error_context={"reason": "rate_limit"}, model="google/gemini-2.5-flash")

    # Verify model is locked out
    assert pool.select(model="google/gemini-2.5-flash") is None

    # Reset statuses
    count = pool.reset_statuses()
    assert count == 1

    # Verify model is unlocked
    entry = pool.select(model="google/gemini-2.5-flash")
    assert entry is not None
    assert entry.id == "cred-1"
    assert "exhausted_models" not in entry.extra

def test_expired_model_lockout_is_pruned(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "openrouter": [
                    {
                        "id": "cred-1",
                        "label": "primary",
                        "auth_type": "api_key",
                        "priority": 0,
                        "source": "manual",
                        "access_token": "key-1",
                        "last_status": "ok",
                        "exhausted_models": {
                            "google/gemini-2.5-flash": {
                                "last_status": "exhausted",
                                "last_status_at": time.time() - 90000,  # 25 hours ago
                                "last_error_code": 429,
                                "last_error_reason": "rate_limit",
                                "last_error_message": "Rate limit exceeded",
                                "last_error_reset_at": time.time() - 80000,
                            }
                        }
                    }
                ]
            },
        },
    )

    pool = load_pool("openrouter")
    
    # select should return cred-1 and prune the expired model lockout
    entry = pool.select(model="google/gemini-2.5-flash")
    assert entry is not None
    assert entry.id == "cred-1"
    
    # Reload and check extra
    pool2 = load_pool("openrouter")
    entry2 = pool2.select(model="google/gemini-2.5-flash")
    assert "exhausted_models" not in entry2.extra
