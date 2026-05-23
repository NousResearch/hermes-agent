"""Tests for the unified Codex credential resolver.

The matrix below is the contract the resolver enforces:

A. env override beats Hermes auth store beats Codex CLI borrow
B. only ~/.codex/auth.json populated -> resolver borrows, compression
   entry point returns the borrowed token instead of None (the bug)
C. only ~/.hermes/auth.json populated -> resolver reads Hermes store,
   compression entry point returns the Hermes token
D. borrow path never writes to disk
E. allow_codex_cli_fallback=False rejects the borrow even when Codex
   CLI has a valid token
F. concurrent force_refresh callers serialize into a single HTTP
   refresh under _auth_store_lock + DCL window
G. (in tests/plugins/image_gen/test_openai_codex_provider.py) image-gen
   plugin sees is_available()=True when only ~/.codex/auth.json exists
H. expired Codex CLI borrow is rejected
I. env override skips expiry check entirely
J. resolve_codex_runtime_credentials shim returns the legacy dict shape
   under each source branch
"""

from __future__ import annotations

import base64
import json
import os
import threading
import time
from pathlib import Path
from typing import Optional

import pytest

from agent.auth.codex import (
    CodexCredentials,
    ENV_CODEX_ACCESS_TOKEN,
    resolve_codex_credentials,
)
from hermes_cli.auth import (
    AuthError,
    DEFAULT_CODEX_BASE_URL,
    resolve_codex_runtime_credentials,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


def _jwt_with_exp(exp_epoch: int, *, extra_claims: Optional[dict] = None) -> str:
    payload = {"exp": exp_epoch}
    if extra_claims:
        payload.update(extra_claims)
    encoded = (
        base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8"))
        .rstrip(b"=")
        .decode("utf-8")
    )
    return f"h.{encoded}.s"


def _write_hermes_store(
    hermes_home: Path,
    *,
    access_token: str = "hermes-access",
    refresh_token: str = "hermes-refresh",
    last_refresh: str = "2026-02-26T00:00:00Z",
    account_id: Optional[str] = "hermes-acct",
) -> Path:
    """Write a Codex provider state into ``$HERMES_HOME/auth.json``."""
    hermes_home.mkdir(parents=True, exist_ok=True)
    tokens = {
        "access_token": access_token,
        "refresh_token": refresh_token,
    }
    if account_id is not None:
        tokens["account_id"] = account_id
    auth_store = {
        "version": 1,
        "active_provider": "openai-codex",
        "providers": {
            "openai-codex": {
                "tokens": tokens,
                "last_refresh": last_refresh,
                "auth_mode": "chatgpt",
            },
        },
    }
    auth_file = hermes_home / "auth.json"
    auth_file.write_text(json.dumps(auth_store, indent=2))
    return auth_file


def _write_empty_hermes_store(hermes_home: Path) -> Path:
    hermes_home.mkdir(parents=True, exist_ok=True)
    auth_file = hermes_home / "auth.json"
    auth_file.write_text(json.dumps({"version": 1, "providers": {}}))
    return auth_file


def _write_codex_cli_store(
    codex_home: Path,
    *,
    access_token: Optional[str] = None,
    refresh_token: str = "cli-refresh",
    account_id: Optional[str] = "cli-acct",
) -> Path:
    """Write a Codex-CLI-shaped ``auth.json`` into ``$CODEX_HOME``."""
    if access_token is None:
        access_token = _jwt_with_exp(int(time.time()) + 3600)
    codex_home.mkdir(parents=True, exist_ok=True)
    tokens = {
        "access_token": access_token,
        "refresh_token": refresh_token,
    }
    if account_id is not None:
        tokens["account_id"] = account_id
    auth_path = codex_home / "auth.json"
    auth_path.write_text(json.dumps({"tokens": tokens}))
    return auth_path


@pytest.fixture
def stores(tmp_path, monkeypatch):
    """Return (hermes_home, codex_home) with both pointing at tempdirs.

    No content written by default — each test populates exactly the
    stores it wants exercising.
    """
    hermes_home = tmp_path / "hermes"
    codex_home = tmp_path / "codex-cli"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    # The HERMES_CODEX_ACCESS_TOKEN env override matches the _TOKEN
    # suffix so the global conftest already clears it; pin to be sure.
    monkeypatch.delenv(ENV_CODEX_ACCESS_TOKEN, raising=False)
    return hermes_home, codex_home


# ── A. precedence ──────────────────────────────────────────────────────────


def test_a_env_override_beats_hermes_store(stores, monkeypatch):
    hermes_home, codex_home = stores
    _write_hermes_store(hermes_home, access_token="from-hermes")
    _write_codex_cli_store(codex_home, access_token=_jwt_with_exp(int(time.time()) + 3600))
    monkeypatch.setenv(ENV_CODEX_ACCESS_TOKEN, "from-env")

    creds = resolve_codex_credentials()

    assert creds.source == "env"
    assert creds.access_token == "from-env"
    assert creds.last_refresh is None
    assert creds.account_id is None
    assert creds.base_url == DEFAULT_CODEX_BASE_URL


def test_a_hermes_store_beats_codex_cli_borrow(stores):
    hermes_home, codex_home = stores
    _write_hermes_store(hermes_home, access_token="from-hermes")
    _write_codex_cli_store(codex_home, access_token=_jwt_with_exp(int(time.time()) + 3600))

    creds = resolve_codex_credentials()

    assert creds.source == "hermes-auth-store"
    assert creds.access_token == "from-hermes"


def test_a_codex_cli_borrow_when_only_source(stores):
    hermes_home, codex_home = stores
    _write_empty_hermes_store(hermes_home)
    cli_token = _jwt_with_exp(int(time.time()) + 3600)
    _write_codex_cli_store(codex_home, access_token=cli_token)

    creds = resolve_codex_credentials()

    assert creds.source == "codex-cli-borrow"
    assert creds.access_token == cli_token


# ── B. compression path now succeeds with only Codex CLI store ─────────────


def test_b_only_codex_cli_main_and_compression_both_succeed(stores):
    hermes_home, codex_home = stores
    _write_empty_hermes_store(hermes_home)
    cli_token = _jwt_with_exp(int(time.time()) + 3600)
    _write_codex_cli_store(codex_home, access_token=cli_token)

    # Main path
    main_creds = resolve_codex_credentials()
    assert main_creds.source == "codex-cli-borrow"
    assert main_creds.access_token == cli_token

    # Compression entry point — the original bug surface.
    from agent.auxiliary_client import _read_codex_access_token

    compression_token = _read_codex_access_token()
    assert compression_token == cli_token


# ── C. only Hermes store — both paths succeed ──────────────────────────────


def test_c_only_hermes_store_main_and_compression_both_succeed(stores):
    hermes_home, codex_home = stores
    _write_hermes_store(hermes_home, access_token="hermes-tok")
    # Codex CLI store deliberately absent.

    main_creds = resolve_codex_credentials()
    assert main_creds.source == "hermes-auth-store"
    assert main_creds.access_token == "hermes-tok"

    from agent.auxiliary_client import _read_codex_access_token

    assert _read_codex_access_token() == "hermes-tok"


# ── D. borrow path never writes to disk ────────────────────────────────────


def test_d_borrow_path_does_not_write_to_disk(stores, monkeypatch):
    hermes_home, codex_home = stores
    _write_empty_hermes_store(hermes_home)
    cli_token = _jwt_with_exp(int(time.time()) + 3600)
    cli_auth_path = _write_codex_cli_store(codex_home, access_token=cli_token)

    save_calls = {"n": 0}

    def _spy_save_codex_tokens(*args, **kwargs):
        save_calls["n"] += 1

    monkeypatch.setattr(
        "hermes_cli.auth._save_codex_tokens", _spy_save_codex_tokens
    )

    cli_mtime_before = cli_auth_path.stat().st_mtime_ns
    hermes_auth_path = hermes_home / "auth.json"
    hermes_mtime_before = hermes_auth_path.stat().st_mtime_ns

    creds = resolve_codex_credentials()

    assert creds.source == "codex-cli-borrow"
    assert save_calls["n"] == 0, (
        "Borrow path must never call _save_codex_tokens — that would "
        "leak a Codex-CLI-owned refresh token into the Hermes store."
    )
    assert cli_auth_path.stat().st_mtime_ns == cli_mtime_before, (
        "Borrow path must not touch ~/.codex/auth.json on disk."
    )
    assert hermes_auth_path.stat().st_mtime_ns == hermes_mtime_before


# ── E. allow_codex_cli_fallback=False rejects the borrow ───────────────────


def test_e_allow_codex_cli_fallback_false_rejects_borrow(stores):
    hermes_home, codex_home = stores
    _write_empty_hermes_store(hermes_home)
    _write_codex_cli_store(codex_home, access_token=_jwt_with_exp(int(time.time()) + 3600))

    with pytest.raises(AuthError):
        resolve_codex_credentials(allow_codex_cli_fallback=False)


# ── F. concurrent refresh serialises into a single HTTP refresh ────────────


def test_f_concurrent_force_refresh_serialises_to_single_http_call(
    stores, monkeypatch
):
    hermes_home, _ = stores
    _write_hermes_store(
        hermes_home,
        access_token=_jwt_with_exp(int(time.time()) - 30),
        refresh_token="rt-initial",
        last_refresh="2026-02-26T00:00:00Z",
    )

    refresh_calls = {"n": 0}
    refresh_lock = threading.Lock()
    barrier = threading.Barrier(2)
    refreshed_access = _jwt_with_exp(int(time.time()) + 3600)

    def _fake_pure_refresh(access_token, refresh_token, *, timeout_seconds=20.0):
        with refresh_lock:
            refresh_calls["n"] += 1
        # Sleep slightly so the second thread is guaranteed to be
        # waiting on the auth-store lock when this returns.
        time.sleep(0.15)
        return {
            "access_token": refreshed_access,
            "refresh_token": "rt-rotated",
        }

    monkeypatch.setattr(
        "hermes_cli.auth.refresh_codex_oauth_pure", _fake_pure_refresh
    )

    results: dict = {}

    def _worker(tag):
        barrier.wait()
        results[tag] = resolve_codex_credentials(force_refresh=True)

    t1 = threading.Thread(target=_worker, args=("a",))
    t2 = threading.Thread(target=_worker, args=("b",))
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert refresh_calls["n"] == 1, (
        f"Concurrent force_refresh must dedupe under _auth_store_lock; "
        f"saw {refresh_calls['n']} HTTP refreshes."
    )
    assert results["a"].access_token == refreshed_access
    assert results["b"].access_token == refreshed_access


# ── H. expired Codex CLI borrow is rejected ────────────────────────────────


def test_h_expired_codex_cli_borrow_is_rejected(stores):
    hermes_home, codex_home = stores
    _write_empty_hermes_store(hermes_home)
    expired_token = _jwt_with_exp(int(time.time()) - 60)
    _write_codex_cli_store(codex_home, access_token=expired_token)

    with pytest.raises(AuthError):
        resolve_codex_credentials()


# ── I. env override skips expiry check entirely ────────────────────────────


def test_i_env_override_skips_expiry_check(stores, monkeypatch):
    hermes_home, codex_home = stores
    _write_empty_hermes_store(hermes_home)
    # Codex CLI store absent too — env override is the only source.
    expired_token = _jwt_with_exp(int(time.time()) - 60)
    monkeypatch.setenv(ENV_CODEX_ACCESS_TOKEN, expired_token)

    creds = resolve_codex_credentials()

    assert creds.source == "env"
    assert creds.access_token == expired_token
    assert creds.last_refresh is None


# ── J. shim returns legacy dict shape under each source branch ─────────────


def test_j_shim_dict_shape_for_hermes_store(stores):
    hermes_home, _ = stores
    _write_hermes_store(hermes_home, access_token="hermes-tok")

    runtime = resolve_codex_runtime_credentials()

    assert runtime["provider"] == "openai-codex"
    assert runtime["api_key"] == "hermes-tok"
    assert runtime["base_url"] == DEFAULT_CODEX_BASE_URL
    assert runtime["source"] == "hermes-auth-store"
    assert runtime["auth_mode"] == "chatgpt"
    assert "last_refresh" in runtime


def test_j_shim_dict_shape_for_codex_cli_borrow(stores):
    hermes_home, codex_home = stores
    _write_empty_hermes_store(hermes_home)
    cli_token = _jwt_with_exp(int(time.time()) + 3600)
    _write_codex_cli_store(codex_home, access_token=cli_token)

    runtime = resolve_codex_runtime_credentials()

    assert runtime["provider"] == "openai-codex"
    assert runtime["api_key"] == cli_token
    assert runtime["source"] == "codex-cli-borrow"
    assert runtime["last_refresh"] is None
    assert runtime["auth_mode"] == "chatgpt"


def test_j_shim_dict_shape_for_env_override(stores, monkeypatch):
    monkeypatch.setenv(ENV_CODEX_ACCESS_TOKEN, "env-tok")

    runtime = resolve_codex_runtime_credentials()

    assert runtime["provider"] == "openai-codex"
    assert runtime["api_key"] == "env-tok"
    assert runtime["source"] == "env"
    assert runtime["last_refresh"] is None
    assert runtime["auth_mode"] == "chatgpt"


# ── Misc: account_id propagation ───────────────────────────────────────────


def test_account_id_exposed_from_hermes_store(stores):
    hermes_home, _ = stores
    _write_hermes_store(hermes_home, account_id="acct-from-hermes")

    creds = resolve_codex_credentials()

    assert creds.account_id == "acct-from-hermes"


def test_account_id_exposed_from_codex_cli_borrow(stores):
    hermes_home, codex_home = stores
    _write_empty_hermes_store(hermes_home)
    _write_codex_cli_store(
        codex_home,
        access_token=_jwt_with_exp(int(time.time()) + 3600),
        account_id="acct-from-cli",
    )

    creds = resolve_codex_credentials()

    assert creds.account_id == "acct-from-cli"


def test_account_id_none_for_env_override(stores, monkeypatch):
    monkeypatch.setenv(ENV_CODEX_ACCESS_TOKEN, "env-tok")

    creds = resolve_codex_credentials()

    assert creds.account_id is None


# ── Refresh-failure surfaces (does not fall back to borrow silently) ───────


def test_refresh_failure_does_not_silently_borrow(stores, monkeypatch):
    """When Hermes store refresh fails, we want the user to see the
    actionable re-auth message — not a silent fallback to a borrowed
    Codex CLI token that masks the broken Hermes session.
    """
    hermes_home, codex_home = stores
    _write_hermes_store(
        hermes_home,
        access_token=_jwt_with_exp(int(time.time()) - 30),
        refresh_token="rt-doomed",
    )
    _write_codex_cli_store(
        codex_home,
        access_token=_jwt_with_exp(int(time.time()) + 3600),
    )

    def _boom(*args, **kwargs):
        raise AuthError(
            "Codex refresh token was already consumed",
            provider="openai-codex",
            code="refresh_token_reused",
            relogin_required=True,
        )

    monkeypatch.setattr(
        "hermes_cli.auth.refresh_codex_oauth_pure", _boom
    )

    with pytest.raises(AuthError) as exc:
        resolve_codex_credentials()
    assert exc.value.code == "refresh_token_reused"
