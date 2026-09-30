"""Opt-in Sign in with ChatGPT (SIWC) plan usage — provider openai-chatgpt.

SIWC routes ChatGPT plan quota to https://api.openai.com/v1 (Responses API) with a
credential stored and refreshed separately from openai-codex (Codex backend), which
must stay untouched.
"""

import json
from pathlib import Path

import pytest

from hermes_cli.auth import (
    AuthError,
    DEFAULT_CHATGPT_PLAN_BASE_URL,
    _save_siwc_tokens,
    resolve_siwc_runtime_credentials,
)
from hermes_cli.auth import resolve_codex_runtime_credentials


def _seed_auth(hermes_home: Path, providers: dict) -> Path:
    hermes_home.mkdir(parents=True, exist_ok=True)
    auth_file = hermes_home / "auth.json"
    auth_file.write_text(json.dumps({"version": 1, "providers": providers}, indent=2))
    return auth_file


def _siwc_state(**overrides):
    state = {
        "tokens": {"access_token": "siwc-access", "refresh_token": "siwc-refresh"},
        "last_refresh": "2026-02-26T00:00:00Z",
        "auth_mode": "chatgpt-plan",
        "use_direct": True,
    }
    state.update(overrides)
    return state


def _codex_state():
    return {
        "tokens": {"access_token": "codex-access", "refresh_token": "codex-refresh"},
        "last_refresh": "2026-02-26T00:00:00Z",
        "auth_mode": "chatgpt",
    }


def test_siwc_registers_and_routes_to_responses_endpoint(tmp_path, monkeypatch):
    """SIWC credential registers and routes to api.openai.com/v1 (Responses API)."""
    hermes_home = tmp_path / "hermes"
    _seed_auth(hermes_home, {"openai-chatgpt": _siwc_state()})
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    assert DEFAULT_CHATGPT_PLAN_BASE_URL == "https://api.openai.com/v1"
    creds = resolve_siwc_runtime_credentials()
    assert creds["provider"] == "openai-chatgpt"
    assert creds["base_url"] == "https://api.openai.com/v1"
    assert creds["api_key"] == "siwc-access"


def test_siwc_requires_opt_in_use_direct(tmp_path, monkeypatch):
    """Without the opt-in marker the SIWC credential must not resolve."""
    hermes_home = tmp_path / "hermes"
    state = _siwc_state()
    del state["use_direct"]
    _seed_auth(hermes_home, {"openai-chatgpt": state})
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    with pytest.raises(AuthError) as exc:
        resolve_siwc_runtime_credentials()
    assert exc.value.code == "siwc_not_opted_in"


def test_siwc_save_sets_opt_in_and_leaves_codex_alone(tmp_path, monkeypatch):
    """Saving SIWC tokens opts in and never touches the Codex credential."""
    hermes_home = tmp_path / "hermes"
    auth_file = _seed_auth(hermes_home, {"openai-codex": _codex_state()})
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    before = json.loads(auth_file.read_text())["providers"]["openai-codex"]

    _save_siwc_tokens({"access_token": "new-access", "refresh_token": "new-refresh"})

    store = json.loads(auth_file.read_text())["providers"]
    assert store["openai-codex"] == before
    assert store["openai-chatgpt"]["tokens"] == {
        "access_token": "new-access", "refresh_token": "new-refresh"}
    assert store["openai-chatgpt"]["use_direct"] is True
    assert store["openai-chatgpt"]["auth_mode"] == "chatgpt-plan"


def test_siwc_refresh_rotates_own_chain_only(tmp_path, monkeypatch):
    """SIWC refresh rotates the SIWC chain; the Codex chain is untouched."""
    import hermes_cli.auth_siwc as siwc_mod

    hermes_home = tmp_path / "hermes"
    auth_file = _seed_auth(
        hermes_home, {"openai-codex": _codex_state(), "openai-chatgpt": _siwc_state()})
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(
        siwc_mod, "refresh_siwc_oauth_pure",
        lambda access, refresh, timeout_seconds=20.0: {
            "access_token": "rotated-access", "refresh_token": "rotated-refresh",
            "last_refresh": "2026-09-30T00:00:00Z"})

    creds = resolve_siwc_runtime_credentials(force_refresh=True)
    assert creds["api_key"] == "rotated-access"

    store = json.loads(auth_file.read_text())["providers"]
    assert store["openai-chatgpt"]["tokens"]["refresh_token"] == "rotated-refresh"
    assert store["openai-codex"]["tokens"] == {
        "access_token": "codex-access", "refresh_token": "codex-refresh"}


def test_codex_untouched_by_siwc(tmp_path, monkeypatch):
    """Codex still resolves to the Codex backend when SIWC is registered."""
    hermes_home = tmp_path / "hermes"
    _seed_auth(
        hermes_home, {"openai-codex": _codex_state(), "openai-chatgpt": _siwc_state()})
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    creds = resolve_codex_runtime_credentials()
    assert creds["provider"] == "openai-codex"
    assert creds["base_url"] == "https://chatgpt.com/backend-api/codex"
    assert creds["api_key"] == "codex-access"
