"""Preparation racing renewal, and runtime resolution honouring a configured relay."""

from types import SimpleNamespace

import pytest

import providers
from agent.credential_pool import CredentialPool, PooledCredential
from agent.credential_pool_plugin import RuntimeOverride, active_runtime_override, durable_runtime_key
from hermes_cli import runtime_provider as rp
from providers.base import PoolRuntimeCredential, ProviderProfile

SOURCE = "https://api.individual.example"
TARGET = "https://api.business.example"
RELAY = "https://relay.corp.example/v1"
GRANT = "durable-grant-0001"


class RacingProfile(ProviderProfile):
    """Its prepare hook lets a concurrent action run between the hook start and its return."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "during_prepare", None)

    def prepare_pool_entry(self, entry, api_key):
        if self.during_prepare is not None:
            self.during_prepare(entry)
        return PoolRuntimeCredential("stale-prepared-bearer", TARGET)


@pytest.fixture
def profile(monkeypatch):
    prof = RacingProfile(name="pairprov", base_url=SOURCE, credential_pool_base_urls=(SOURCE, TARGET))
    monkeypatch.setattr(providers, "get_provider_profile", lambda name: prof if name == "pairprov" else None)
    return prof


def _pool():
    row = PooledCredential(provider="pairprov", id="row-a", label="row-a", auth_type="api_key", priority=0,
                           source="manual", access_token=GRANT, base_url=SOURCE)
    return CredentialPool("pairprov", [row]), row


def test_concurrent_renewal_installed_during_prepare_is_not_overwritten(profile):
    pool, row = _pool()
    fresh = RuntimeOverride(GRANT, durable_runtime_key(row), "fresh-renewed-bearer", TARGET)

    def renew(entry):
        with pool._lock:
            object.__setattr__(entry, "_runtime_override", fresh)

    profile.during_prepare = renew
    prepared = pool.prepare_runtime_entry(row)
    assert prepared is row
    assert active_runtime_override(row) is fresh
    assert row.runtime_api_key == "fresh-renewed-bearer"


def test_grant_rotated_during_prepare_installs_nothing(profile):
    pool, row = _pool()

    def rotate(entry):
        with pool._lock:
            object.__setattr__(entry, "access_token", "rotated-grant")

    profile.during_prepare = rotate
    assert pool.prepare_runtime_entry(row) is None
    assert getattr(row, "__dict__", {}).get("_runtime_override") is None


def test_row_removed_during_prepare_installs_nothing(profile):
    pool, row = _pool()

    def remove(entry):
        with pool._lock:
            pool._entries = []

    profile.during_prepare = remove
    assert pool.prepare_runtime_entry(row) is None
    assert getattr(row, "__dict__", {}).get("_runtime_override") is None


def _overridden_entry():
    return SimpleNamespace(id="r", provider="deepseek", access_token="grant", base_url="https://api.deepseek.com/v1",
                           runtime_api_key="derived", runtime_base_url=TARGET, source="manual")


class _Pool(SimpleNamespace):
    def __init__(self, entry, prepared):
        super().__init__(provider="deepseek", entry=entry, prepared=prepared, calls=[])

    def prepare_runtime_entry(self, entry, *, source_base_url=None):
        self.calls.append(source_base_url)
        return self.prepared(source_base_url) if callable(self.prepared) else self.prepared


def test_active_override_does_not_bypass_configured_relay_admission():
    entry = PooledCredential(provider="deepseek", id="r", label="r", auth_type="api_key", priority=0, source="manual",
                             access_token="grant", base_url="https://api.deepseek.com/v1")
    object.__setattr__(entry, "_runtime_override", RuntimeOverride("grant", "grant", "derived", TARGET))
    assert entry.runtime_base_url == TARGET
    pool = _Pool(entry, lambda source: None if source is not None else entry)  # relay not admitted
    cfg = {"provider": "deepseek", "base_url": RELAY}
    result = rp._resolve_runtime_from_pool_entry(provider="deepseek", entry=entry, requested_provider="deepseek",
                                                 model_cfg=cfg, pool=pool, target_model="m")
    assert pool.calls == [None, RELAY]
    assert result is None or result["base_url"].rstrip("/") == RELAY


def test_prepared_endpoint_does_not_redirect_configured_relay():
    entry = SimpleNamespace(id="r", provider="deepseek", access_token="grant", base_url="https://api.deepseek.com/v1",
                            runtime_api_key="grant", runtime_base_url="https://api.deepseek.com/v1", source="manual")
    moved = _overridden_entry()
    pool = _Pool(entry, lambda source: moved if source is None else None)
    cfg = {"provider": "deepseek", "base_url": RELAY}
    result = rp._resolve_runtime_from_pool_entry(provider="deepseek", entry=entry, requested_provider="deepseek",
                                                 model_cfg=cfg, pool=pool, target_model="m")
    assert result is None or result["base_url"].rstrip("/") == RELAY
