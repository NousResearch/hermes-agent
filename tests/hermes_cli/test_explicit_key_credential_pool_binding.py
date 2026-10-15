"""Real-resolution regression for the SECOND drop site of #92250.

The lazy CLI binding (``_stage_and_swap_model`` when the agent is not yet built) is one half.
The other half is the *subsequent* runtime resolution: ``/model`` stages the selected key in
``_explicit_api_key``, and the first-turn ``_ensure_runtime_credentials`` re-resolves with that
explicit key. Before this change the explicit rung returned *without* ``credential_pool``, so the
new provider ran pool-less and a 429/billing/401 never rotated to the next account.

These tests drive the REAL resolver (``resolve_runtime_provider``) against a temporary auth store
and the REAL pool/recovery bookkeeping (``recover_with_credential_pool``). Only the final
client-swap callback is stubbed, and all external network access is blocked.
"""
from __future__ import annotations

import base64
import json
import os
import socket
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import runtime_provider as rp
from hermes_cli.auth import DEFAULT_CODEX_BASE_URL, PROVIDER_REGISTRY
from hermes_cli.cli_model_switch_mixin import _runtime_fields
from hermes_cli.model_switch import ModelSwitchResult

CODEX_ENDPOINT = DEFAULT_CODEX_BASE_URL
DEEPSEEK_ENDPOINT = PROVIDER_REGISTRY["deepseek"].inference_base_url
_EXHAUSTED = "exhausted"


def _jwt(claims: dict) -> str:
    def _part(payload: dict) -> str:
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return f"{_part({'alg': 'none', 'typ': 'JWT'})}.{_part(claims)}.sig"


def _codex_entry(entry_id: str, label: str, token: str, priority: int) -> dict:
    return {
        "id": entry_id, "label": label, "auth_type": "oauth", "priority": priority,
        "source": "manual:device_code", "access_token": token, "refresh_token": f"{entry_id}-refresh",
        "base_url": CODEX_ENDPOINT,
    }


def _deepseek_entry(entry_id: str, label: str, token: str, priority: int) -> dict:
    return {
        "id": entry_id, "label": label, "auth_type": "api_key", "priority": priority,
        "source": "manual", "access_token": token, "base_url": DEEPSEEK_ENDPOINT,
    }


@pytest.fixture(autouse=True)
def _block_external_network(monkeypatch):
    """These tests must resolve offline; only loopback connections may pass."""
    real_connect = socket.socket.connect

    def _guarded(self, address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) else address
        if isinstance(address, tuple) and host in {"127.0.0.1", "::1", "localhost", "0.0.0.0"}:
            return real_connect(self, address, *args, **kwargs)
        raise AssertionError(f"unexpected external network access to {address!r}")

    monkeypatch.setattr(socket.socket, "connect", _guarded)


@pytest.fixture
def auth_store(monkeypatch):
    """Temporary auth store holding two synthetic Codex device-code accounts."""
    home = Path(os.environ["HERMES_HOME"])
    home.mkdir(parents=True, exist_ok=True)
    # Never seed the pool from a developer's real ~/.codex login.
    monkeypatch.setattr("hermes_cli.auth._import_codex_cli_tokens", lambda: None, raising=False)
    monkeypatch.setattr("hermes_cli.auth_codex._import_codex_cli_tokens", lambda: None, raising=False)
    key_a = "codex-account-a." + _jwt({"exp": int(time.time()) + 86400})
    key_b = "codex-account-b." + _jwt({"exp": int(time.time()) + 86400})
    store = {
        "version": 1,
        "credential_pool": {
            "openai-codex": [
                _codex_entry("cred-a", "account-a", key_a, 0),
                _codex_entry("cred-b", "account-b", key_b, 1),
            ],
        },
    }
    (home / "auth.json").write_text(json.dumps(store, indent=2), encoding="utf-8")
    return {"home": home, "key_a": key_a, "key_b": key_b}


# ── the resolver half ───────────────────────────────────────────────────────────────────────


def test_explicit_member_key_attaches_provider_pool(auth_store):
    """A staged key that is a registered pool member must keep the pool (the #92250 drop site)."""
    runtime = rp.resolve_runtime_provider(
        requested="openai-codex",
        explicit_api_key=auth_store["key_a"],
        explicit_base_url=CODEX_ENDPOINT,
    )

    assert runtime["provider"] == "openai-codex"
    assert runtime["api_key"] == auth_store["key_a"], "the explicitly selected key must be kept"
    assert runtime["base_url"] == CODEX_ENDPOINT
    pool = runtime.get("credential_pool")
    assert pool is not None, "explicit staged key dropped the provider credential pool (#92250)"
    assert pool.provider == "openai-codex"
    assert {e.id for e in pool.entries()} == {"cred-a", "cred-b"}


def test_unregistered_explicit_key_stays_pool_less(auth_store):
    """Negative control: a one-off override key (not in the pool) must NOT pull in a pool."""
    runtime = rp.resolve_runtime_provider(
        requested="openai-codex",
        explicit_api_key="one-off-override-key",
        explicit_base_url=CODEX_ENDPOINT,
    )

    assert runtime["api_key"] == "one-off-override-key"
    assert runtime.get("credential_pool") is None, (
        "an unregistered key must stay pool-less rather than rotate away from the chosen key"
    )


def test_non_canonical_endpoint_stays_pool_less(auth_store):
    """Negative control: a member key aimed at a different endpoint must NOT pull in a pool."""
    runtime = rp.resolve_runtime_provider(
        requested="openai-codex",
        explicit_api_key=auth_store["key_a"],
        explicit_base_url="https://proxy.example.test/v1",
    )

    assert runtime["api_key"] == auth_store["key_a"]
    assert runtime.get("credential_pool") is None, (
        "a non-canonical destination must stay pool-less — the pool belongs to the canonical endpoint"
    )


def test_member_key_attaches_own_pool_and_foreign_pool_is_not_lent(auth_store):
    """Provider-generic boundary: identical key strings must never cross provider pools."""
    home = auth_store["home"]
    store = json.loads((home / "auth.json").read_text(encoding="utf-8"))
    # The SAME key string is registered under a second provider, plus a key only that provider owns.
    store["credential_pool"]["deepseek"] = [
        _deepseek_entry("ds-a", "deepseek-a", auth_store["key_a"], 0),
        _deepseek_entry("ds-only", "deepseek-only", "deepseek-only-key", 1),
    ]
    (home / "auth.json").write_text(json.dumps(store, indent=2), encoding="utf-8")

    ds_runtime = rp.resolve_runtime_provider(
        requested="deepseek", explicit_api_key=auth_store["key_a"], explicit_base_url=DEEPSEEK_ENDPOINT)
    assert ds_runtime.get("credential_pool") is not None
    assert ds_runtime["credential_pool"].provider == "deepseek", "must attach its OWN pool"

    codex_runtime = rp.resolve_runtime_provider(
        requested="openai-codex", explicit_api_key=auth_store["key_a"], explicit_base_url=CODEX_ENDPOINT)
    assert codex_runtime.get("credential_pool") is not None
    assert codex_runtime["credential_pool"].provider == "openai-codex", (
        "matching key strings must not lend a foreign provider's pool"
    )

    # A key registered ONLY under deepseek must not drag deepseek's pool into a codex resolution.
    foreign = rp.resolve_runtime_provider(
        requested="openai-codex", explicit_api_key="deepseek-only-key", explicit_base_url=CODEX_ENDPOINT)
    assert foreign.get("credential_pool") is None


# ── cold switch -> staged key -> real resolution -> real recovery ────────────────────────────


class _ColdSwitchStub:
    """Minimum attrs the real ``_stage_and_swap_model`` / snapshot helpers read on ``self``."""

    agent = None
    model = ""
    provider = ""
    requested_provider = ""
    api_key = ""
    base_url = ""
    api_mode = ""
    _explicit_api_key = None
    _explicit_base_url = None
    _credential_pool = None


def _switch_result(api_key: str, base_url: str) -> ModelSwitchResult:
    return ModelSwitchResult(
        success=True,
        new_model="gpt-5.4",
        target_provider="openai-codex",
        provider_changed=True,
        api_key=api_key,
        base_url=base_url,
        api_mode="codex_responses",
        warning_message="",
        provider_label="ChatGPT Codex",
        resolved_via_alias=False,
        capabilities=None,
        model_info=SimpleNamespace(context_window=272_000, max_output=0, has_cost_data=lambda: False,
                                   format_capabilities=lambda: ""),
        is_global=False,
    )


class _PoolAgent:
    """Stub agent wired to the REAL pool + recovery bookkeeping; only the client swap is stubbed."""

    def __init__(self, pool, api_key: str, base_url: str):
        self.provider = "openai-codex"
        self.requested_provider = "openai-codex"
        self.base_url = base_url
        self.api_key = api_key
        self._credential_pool = pool
        self._credential_pool_entry_id = None
        self.swapped = []

    def _swap_credential(self, entry):
        self.swapped.append(entry)
        self.api_key = entry.runtime_api_key


def test_cold_switch_staged_key_real_resolution_recovers_to_healthy_sibling(auth_store, monkeypatch):
    """The exact reviewer scenario: cold switch -> staged explicit key -> real resolution ->
    a usage-limit 429 rotates to the second account instead of falling through to another provider."""
    import cli as cli_mod

    monkeypatch.setattr(cli_mod, "_cprint", lambda *_a, **_k: None)

    cli = _ColdSwitchStub()
    cli.provider = "openai-codex"
    cli.model = "gpt-5.4"
    cli.requested_provider = "openai-codex"
    cli.api_mode = "codex_responses"

    assert cli_mod.HermesCLI._stage_and_swap_model(cli, _switch_result(auth_store["key_a"], CODEX_ENDPOINT),
                                                   "old-model") is True
    assert cli._explicit_api_key == auth_store["key_a"], "the switch must stage the selected key"

    # First-turn lazy re-resolution with the staged explicit key: the second drop site.
    runtime = rp.resolve_runtime_provider(
        requested=cli.requested_provider,
        explicit_api_key=cli._explicit_api_key,
        explicit_base_url=cli._explicit_base_url,
    )
    cli._credential_pool = runtime.get("credential_pool")
    pool = cli._credential_pool
    assert pool is not None, "lazy resolution with the staged key dropped the pool (#92250)"
    assert pool.provider == "openai-codex"

    from agent.agent_runtime_helpers import recover_with_credential_pool

    agent = _PoolAgent(pool, auth_store["key_a"], CODEX_ENDPOINT)
    recovered, _ = recover_with_credential_pool(
        agent,
        status_code=429,
        has_retried_429=False,
        error_context={"reason": "usage_limit_reached", "message": "usage limit reached"},
    )

    assert recovered is True, "a usage-limit 429 must rotate, not fall through to another provider"
    assert [e.id for e in agent.swapped] == ["cred-b"], "recovery must select the healthy sibling"
    assert agent.api_key == auth_store["key_b"]
    statuses = {e.id: e.last_status for e in pool.entries()}
    assert statuses["cred-a"] == _EXHAUSTED, "only the failing account is marked exhausted"
    assert statuses["cred-b"] != _EXHAUSTED


def test_pool_less_agent_cannot_rotate(auth_store):
    """Contrast: without the binding, recovery is a no-op — the pre-fix symptom."""
    from agent.agent_runtime_helpers import recover_with_credential_pool

    agent = _PoolAgent(None, auth_store["key_a"], CODEX_ENDPOINT)
    recovered, retried = recover_with_credential_pool(
        agent, status_code=429, has_retried_429=False,
        error_context={"reason": "usage_limit_reached"},
    )
    assert recovered is False
    assert agent.swapped == []


# ── the CLI pool state in the rollback snapshot ─────────────────────────────────────────────


def test_runtime_fields_include_credential_pool():
    cli = _ColdSwitchStub()
    cli._credential_pool = "pool-sentinel"
    assert _runtime_fields(cli)["_credential_pool"] == "pool-sentinel"


def test_snapshot_and_restore_carries_cli_pool_state(auth_store):
    """A one-turn override that rebinds the pool must restore the original pool afterwards."""
    import cli as cli_mod

    old_pool = object()
    cli = _ColdSwitchStub()
    cli.provider = "openai-codex"
    cli.model = "gpt-5.4"
    cli.requested_provider = "openai-codex"
    cli.api_mode = "codex_responses"
    cli._credential_pool = old_pool

    snapshot = cli_mod.HermesCLI._snapshot_model_runtime(cli)
    assert snapshot["_credential_pool"] is old_pool, "pool state must be captured in the snapshot"

    # Simulate the switch binding a different pool for the override.
    cli._credential_pool = "override-pool"
    cli_mod.HermesCLI._restore_model_runtime_snapshot(cli, snapshot)

    assert cli._credential_pool is old_pool, "the rollback snapshot must restore CLI pool state"
