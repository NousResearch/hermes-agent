"""Regression for #127917: a 365-day model_entitlement bench can leave a provider logged_in
with every model benched. ``hermes auth status`` must surface the silent-kill case so the
operator can act (reset, or ``hermes auth reset --provider X``); the ``logged_in`` semantic
keeps the structural signal so the picker-facing path does not flip on a transient bench.
"""
from __future__ import annotations

import os
import time
from types import SimpleNamespace

import pytest

from hermes_cli.auth import (
    MODEL_ENTITLEMENT_BENCH_DAYS_DEFAULT,
    _pool_bench_summary,
    get_external_process_provider_status,
)


class _FakePoolEntry:
    def __init__(self, runtime_api_key, model_cooldowns=None, last_status=None):
        self.runtime_api_key = runtime_api_key
        self.model_cooldowns = model_cooldowns or {}
        self.last_status = last_status


class _FakePool:
    def __init__(self, entries):
        self._entries = entries


def _patch_pool(monkeypatch, entries):
    from agent import credential_pool

    def fake_load_pool(provider_id):
        return _FakePool(entries)

    monkeypatch.setattr(credential_pool, "load_pool", fake_load_pool)


def _patch_registry(monkeypatch, provider="codex"):
    from hermes_cli import auth as auth_mod

    pconfig = SimpleNamespace(
        id=provider,
        name="Codex",
        auth_type="external_process",
    )

    def fake_lookup(provider_id):
        return pconfig if provider_id == provider else None

    monkeypatch.setattr(auth_mod, "_registry_lookup", fake_lookup)


def _patch_status_display_env(monkeypatch):
    """Surface ``logged_in=True`` etc. when only model cooldowns are active (#127917)."""
    monkeypatch.setattr(os, "environ", os.environ.copy())


def test_pool_bench_summary_empty_when_no_cooldowns(monkeypatch):
    _patch_pool(monkeypatch, [
        _FakePoolEntry(runtime_api_key="k1"),
    ])
    out = _pool_bench_summary("codex")
    assert out == {}


def test_pool_bench_summary_surfaces_when_every_credential_benched(monkeypatch):
    """The silent-kill case: both credentials of the pool have every model benched."""
    future = time.time() + 30 * 86400
    _patch_pool(monkeypatch, [
        _FakePoolEntry(
            runtime_api_key="k1",
            model_cooldowns={"gpt-5.6": future, "gpt-6": future},
        ),
        _FakePoolEntry(
            runtime_api_key="k2",
            model_cooldowns={"gpt-6-mini": future},
        ),
    ])
    out = _pool_bench_summary("codex")
    assert "pool_blocked_models" in out
    blocked = {row["model"] for row in out["pool_blocked_models"]}
    assert blocked == {"gpt-5.6", "gpt-6", "gpt-6-mini"}
    assert "hermes auth reset" in out["hint"]
    assert str(MODEL_ENTITLEMENT_BENCH_DAYS_DEFAULT) in out["hint"]


def test_pool_bench_summary_ignores_already_expired_cooldowns(monkeypatch):
    past = time.time() - 60
    _patch_pool(monkeypatch, [
        _FakePoolEntry(runtime_api_key="k1", model_cooldowns={"gpt-6": past}),
    ])
    out = _pool_bench_summary("codex")
    assert out == {}


def test_pool_bench_summary_ignores_structurally_dead_credentials(monkeypatch):
    """A STATUS_DEAD key (auth/billing-killed) is structural, not an entitlement silent-kill."""
    from agent.credential_pool import STATUS_DEAD
    future = time.time() + 60
    _patch_pool(monkeypatch, [
        _FakePoolEntry(
            runtime_api_key="dead-key",
            model_cooldowns={"gpt-6": future},
            last_status=STATUS_DEAD,
        ),
    ])
    out = _pool_bench_summary("codex")
    assert out == {}


def test_external_process_status_surfaces_silent_kill_fields(monkeypatch):
    """The structural logged_in stays True; the silent-kill fields are additive."""
    future = time.time() + 30 * 86400
    _patch_pool(monkeypatch, [
        _FakePoolEntry(
            runtime_api_key="k1",
            model_cooldowns={"gpt-6": future, "gpt-6-mini": future},
        ),
    ])
    _patch_registry(monkeypatch, provider="codex")

    def fake_spec(pconfig):
        return ("codex", ["--model", "gpt-6"], "https://api.openai.com/v1", "codex", ("ANTHROPIC_API_KEY",))

    def fake_evidence(provider_id, resolved_command):
        return (True, "probe")

    import hermes_cli.auth as auth_mod
    monkeypatch.setattr(auth_mod, "_external_process_spec", fake_spec)
    monkeypatch.setattr(auth_mod, "_external_process_auth_evidence", fake_evidence)

    out = get_external_process_provider_status("codex")
    assert out["logged_in"] is True, "structural signal must not flip on a bench"
    assert out["configured"] is True
    assert "pool_blocked_models" in out
    blocked = {row["model"] for row in out["pool_blocked_models"]}
    assert blocked == {"gpt-6", "gpt-6-mini"}
    assert "hermes auth reset" in out["hint"]


def test_external_process_status_without_bench_returns_no_silent_kill_fields(monkeypatch):
    """The structural happy path: empty pool_blocked_models is omitted from the dict."""
    _patch_pool(monkeypatch, [
        _FakePoolEntry(runtime_api_key="k1"),
    ])
    _patch_registry(monkeypatch, provider="codex")

    def fake_spec(pconfig):
        return ("codex", ["--model", "gpt-6"], "https://api.openai.com/v1", "codex", ("ANTHROPIC_API_KEY",))

    def fake_evidence(provider_id, resolved_command):
        return (True, "probe")

    import hermes_cli.auth as auth_mod
    monkeypatch.setattr(auth_mod, "_external_process_spec", fake_spec)
    monkeypatch.setattr(auth_mod, "_external_process_auth_evidence", fake_evidence)

    out = get_external_process_provider_status("codex")
    assert out["logged_in"] is True
    assert "pool_blocked_models" not in out
    assert "hint" not in out


def test_model_entitlement_bench_days_default_is_reasonable():
    """The reduced default reflects the new opt-in affordance — must not be the legacy 365d."""
    assert 30 <= MODEL_ENTITLEMENT_BENCH_DAYS_DEFAULT <= 180