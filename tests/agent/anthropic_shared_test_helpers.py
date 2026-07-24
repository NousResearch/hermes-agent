"""Helpers for Anthropic shared pool tests (fixture tokens only)."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest


FIXTURE_ACCESS = "fixture-oauth-access-token-aaaaaaaaaaaaaaaa"
FIXTURE_REFRESH = "fixture-oauth-refresh-token-bbbbbbbbbbbbbbbb"


@pytest.fixture
def shared_root(tmp_path, monkeypatch):
    """Isolate both HERMES_HOME and the default root under tmp_path.

    HERMES_HOME is intentionally NOT ``Path.home()/'.hermes'`` so the pytest
    seat belt in ``_auth_file_path`` does not treat the temp store as the
    real user auth path when HOME is also redirected.
    """
    user_home = tmp_path / "home"
    user_home.mkdir()
    root = tmp_path / "hermes-root"
    root.mkdir()
    (root / "shared").mkdir()
    monkeypatch.setenv("HOME", str(user_home))
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    from agent import anthropic_shared_pool as sp

    sp.reset_startup_epoch_for_tests()
    yield root
    sp.reset_startup_epoch_for_tests()


@pytest.fixture
def profile_env(shared_root, monkeypatch):
    """Named profile under the shared root."""
    profiles = shared_root / "profiles" / "worker"
    profiles.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profiles))
    from agent import anthropic_shared_pool as sp

    sp.reset_startup_epoch_for_tests()
    return profiles


def write_root_auth(root: Path, payload: Dict[str, Any]) -> Path:
    path = root / "auth.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    path.chmod(0o600)
    return path


def make_row(
    *,
    priority: int,
    label: Optional[str] = None,
    access: str = FIXTURE_ACCESS,
    refresh: Optional[str] = None,
    generation: int = 1,
    expires_at_ms: Optional[int] = None,
    last_status=None,
    **overrides,
) -> Dict[str, Any]:
    import time
    from agent.anthropic_shared_pool import (
        ENDPOINT_PLATFORM,
        SOURCE_HERMES_PKCE,
        grant_fingerprint,
        validate_shared_row,
    )

    refresh = refresh or f"{FIXTURE_REFRESH}-{priority}"
    row = {
        "id": str(uuid.uuid4()),
        "provider": "anthropic",
        "auth_type": "oauth",
        "source": SOURCE_HERMES_PKCE,
        "label": label or f"Anthropic account {priority + 1}",
        "grant_fingerprint": grant_fingerprint(refresh),
        "token_generation": generation,
        "oauth_token_endpoint": ENDPOINT_PLATFORM,
        "access_token": f"{access}-{priority}",
        "refresh_token": refresh,
        "expires_at_ms": expires_at_ms or int(time.time() * 1000) + 3_600_000,
        "priority": priority,
        "request_count": 0,
        "last_status": last_status,
        "last_status_at": None,
        "last_error_code": None,
        "last_error_reason": None,
        "last_error_message": None,
        "last_error_reset_at": None,
        "last_refresh": None,
        "refresh_attempt": None,
    }
    row.update(overrides)
    return validate_shared_row(row, at_enrollment=True)


def stage_three(root: Path, *, attest: bool = False) -> Dict[str, Any]:
    from agent.anthropic_shared_pool import empty_shared_pool, validate_shared_pool

    entries = [make_row(priority=i) for i in range(3)]
    pool = empty_shared_pool()
    pool["entries"] = entries
    pool["revision"] = 1
    if attest:
        pool["account_distinctness_attested"] = True
        pool["account_distinctness_attested_at"] = "2026-01-01T00:00:00Z"
    pool = validate_shared_pool(pool, require_three=attest)
    write_root_auth(
        root,
        {
            "version": 1,
            "providers": {},
            "shared_credential_pools": {"anthropic": pool},
        },
    )
    return pool


def enable_marker(root: Path, epoch: str = "11111111-1111-1111-1111-111111111111") -> None:
    path = root / "shared" / "anthropic_pool_scope.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(
        json.dumps({"version": 1, "scope": "shared", "epoch": epoch}) + "\n"
    )
    path.chmod(0o600)
