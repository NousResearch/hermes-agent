"""Hermetic fixtures for the B-1 Knowledge Discovery canary.

All fixtures are deterministic (frozen time) and in-memory. They do NOT
touch filesystem outside `pytest tmp_path`, never open network sockets, and
never invoke subprocess. The conftest is local to the canary package and
does not modify the parent `tests/test_executive_v2/conftest.py`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from tests.test_executive_v2.canary_b1.evidence_pack import EvidencePackEngine
from tests.test_executive_v2.canary_b1.fake_providers import (
    FakeProviderSpec,
    default_gbrain_spec,
    default_obsidian_spec,
    default_reports_spec,
    empty_spec,
    make_provider_bundle,
)


# ─────────────────────────────────────────────────────────────────────
# Frozen time
# ─────────────────────────────────────────────────────────────────────


CANARY_FROZEN_TIME_UTC = "2026-07-08T20:00:00+00:00"


class _FakeMonotonic:
    """Deterministic monotonic counter for duration_ms tests."""

    def __init__(self) -> None:
        self._n = 0

    def __call__(self) -> float:
        self._n += 1
        return float(self._n)


@pytest.fixture
def frozen_time(monkeypatch):
    """Pin `_now_iso8601` and `time.monotonic` to deterministic values."""
    from tests.test_executive_v2.canary_b1 import evidence_pack as _ep

    monkeypatch.setattr(_ep, "_now_iso8601", lambda: CANARY_FROZEN_TIME_UTC)
    fake_mono = _FakeMonotonic()
    monkeypatch.setattr(_ep.time, "monotonic", fake_mono)
    yield CANARY_FROZEN_TIME_UTC


# ─────────────────────────────────────────────────────────────────────
# In-memory storage (FakeDB-like, NEVER ~/.hermes/state.db)
# ─────────────────────────────────────────────────────────────────────


class _InMemoryStorage:
    """Minimal in-memory storage matching the existing conftest's FakeDB.

    Adds a `_state_meta` dict for the canary engine's `discover/rollback`
    paths. `close()` is a no-op.
    """

    def __init__(self) -> None:
        self._state_meta: dict[str, Any] = {}

    def set_meta(self, k: str, v: Any) -> None:
        self._state_meta[k] = v

    def get_meta(self, k: str) -> Any:
        return self._state_meta.get(k)

    def delete_meta(self, k: str) -> None:
        self._state_meta.pop(k, None)

    def list_meta_keys(self, prefix: Optional[str] = None) -> list[str]:
        if prefix is None:
            return list(self._state_meta.keys())
        return [k for k in self._state_meta if k.startswith(prefix)]

    def close(self) -> None:
        pass


@pytest.fixture
def in_memory_storage():
    """Fresh in-memory storage per test (no state.db)."""
    return _InMemoryStorage()


# ─────────────────────────────────────────────────────────────────────
# Audit capture (in-memory; NEVER ~/.hermes/audit/*)
# ─────────────────────────────────────────────────────────────────────


class _AuditCapture:
    def __init__(self) -> None:
        self._events: list[dict] = []

    def emit(self, event: dict) -> None:
        self._events.append(dict(event))

    def get_events(self) -> list[dict]:
        return list(self._events)


@pytest.fixture
def audit_capture():
    return _AuditCapture()


# ─────────────────────────────────────────────────────────────────────
# Fake provider specs (one fixture per source)
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def fake_gbrain_spec() -> FakeProviderSpec:
    return default_gbrain_spec()


@pytest.fixture
def fake_obsidian_spec() -> FakeProviderSpec:
    return default_obsidian_spec()


@pytest.fixture
def fake_reports_spec(tmp_path) -> FakeProviderSpec:
    return default_reports_spec(tmp_path)


@pytest.fixture
def fake_policy_spec() -> FakeProviderSpec:
    return FakeProviderSpec(
        name="policy",
        hits=(
            {
                "hit_id": "fake-policy-decision-001",
                "title": "Policy decision: B-1 closure (canary fixture)",
                "warnings": ("policy notes for knowledge discovery",),
                "decision_fingerprint": "fpr-policy-001",
                "risk_level": "medium",
                "source_updated_at": "2026-07-08T20:00:00+00:00",
                "goal_class": "OTHER",
            },
        ),
    )


@pytest.fixture
def fake_contract_spec() -> FakeProviderSpec:
    return FakeProviderSpec(
        name="contract",
        hits=(
            {
                "hit_id": "fake-contract-001",
                "title": "Execution contract: B-1 canary (fixture)",
                "risk_score": 0.30,
                "hard_constraints": ("no external network", "no real gbrain"),
                "soft_constraints": ("hermetic only",),
                "success_criteria": ("canary asserts pass", "schema valid"),
                "source_updated_at": "2026-07-08T20:00:00+00:00",
            },
        ),
    )


# ─────────────────────────────────────────────────────────────────────
# Provider bundle
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def provider_bundle(
    fake_gbrain_spec: FakeProviderSpec,
    fake_obsidian_spec: FakeProviderSpec,
    fake_reports_spec: FakeProviderSpec,
    fake_policy_spec: FakeProviderSpec,
    fake_contract_spec: FakeProviderSpec,
) -> dict:
    return make_provider_bundle(
        gbrain_spec=fake_gbrain_spec,
        obsidian_spec=fake_obsidian_spec,
        report_spec=fake_reports_spec,
        policy_spec=fake_policy_spec,
        contract_spec=fake_contract_spec,
    )


# ─────────────────────────────────────────────────────────────────────
# Engine (the canary's central fixture)
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def hermetic_evidence_pack_engine(
    frozen_time,
    provider_bundle,
    in_memory_storage,
    audit_capture,
):
    """A fully-wired hermetic engine with all 5 fake sources registered.

    Returns ``(engine, bundle)`` for tests that want to mutate the bundle.
    """
    engine = EvidencePackEngine(
        sources=provider_bundle,
        storage=in_memory_storage,
        audit_sink=audit_capture,
    )
    return engine, provider_bundle


# ─────────────────────────────────────────────────────────────────────
# Default-off flag guard
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def default_off_flags(monkeypatch):
    """Ensure both flags are at 0; canary must run regardless."""
    monkeypatch.delenv("HERMES_EXECUTIVE_KNOWLEDGE_DISCOVERY_ENABLED", raising=False)
    monkeypatch.delenv("HERMES_EXECUTIVE_V2_ENABLED", raising=False)
    yield


# ─────────────────────────────────────────────────────────────────────
# Self-improvement guard
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def self_improvement_disabled(monkeypatch):
    monkeypatch.setenv("HERMES_DISABLE_SELF_IMPROVEMENT", "1")
    yield
