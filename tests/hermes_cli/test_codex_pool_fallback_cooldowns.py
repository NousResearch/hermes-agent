"""Pool-only Codex resolution must honor persisted account cooldowns (PR #43277)."""
import json
import time
from pathlib import Path

import pytest


@pytest.mark.parametrize("status,status_at,reset_at", [
    ("dead", None, None),
    ("exhausted", "recent", None),
    ("exhausted", None, None),
    ("exhausted", "recent", "future_ms"),
])
def test_pool_only_resolution_skips_unusable_accounts(tmp_path, monkeypatch, status, status_at, reset_at):
    from hermes_cli.auth import resolve_codex_runtime_credentials

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "missing-codex"))
    now = time.time()
    bad = {"source": "manual:test", "access_token": "bad", "last_status": status,
           "last_status_at": now if status_at else None, "last_error_code": 429,
           "last_error_reset_at": (now + 3600) * 1000 if reset_at else None}
    (tmp_path / "auth.json").write_text(json.dumps({
        "version": 1, "providers": {}, "credential_pool": {"openai-codex": [
            bad, {"source": "manual:healthy", "access_token": "healthy", "last_status": "ok"},
        ]},
    }))
    resolved = resolve_codex_runtime_credentials(read_only=True)
    assert resolved["api_key"] == "healthy"
    assert resolved["source"] == "credential_pool"


def test_elapsed_default_cooldown_can_reenter_pool_only_resolution(tmp_path, monkeypatch):
    from hermes_cli.auth import resolve_codex_runtime_credentials

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "auth.json").write_text(json.dumps({
        "version": 1, "providers": {}, "credential_pool": {"openai-codex": [{
            "source": "manual:test", "access_token": "recovered", "last_status": "exhausted",
            "last_status_at": time.time() - 7200, "last_error_code": 429,
        }]},
    }))
    assert resolve_codex_runtime_credentials(read_only=True)["api_key"] == "recovered"
