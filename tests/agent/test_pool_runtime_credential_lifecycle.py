"""A pool row's derived runtime credential is ephemeral, route-checked and scoped to that row."""

from dataclasses import replace

import pytest

import providers
from agent.credential_pool import CredentialPool, PooledCredential
from providers.base import PoolRuntimeCredential, ProviderProfile

SOURCE = "https://api.individual.example"
TARGET = "https://api.business.example"
GRANT = "durable-grant-0001"
DERIVED = "derived-bearer-SECRET"


class PairProfile(ProviderProfile):
    """Mints a derived bearer and endpoint per prepared row; counts every mint."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "results", {})
        object.__setattr__(self, "calls", [])

    def prepare_pool_entry(self, entry, api_key):
        self.calls.append(("prepare", entry.id, api_key))
        result = self.results.get(("prepare", entry.id), PoolRuntimeCredential(DERIVED, TARGET))
        if isinstance(result, Exception):
            raise result
        return result

    def refresh_pool_runtime_credentials(self, entry, *, reason):
        self.calls.append(("refresh", entry.id, reason))
        result = self.results.get(("refresh", entry.id))
        if isinstance(result, Exception):
            raise result
        return result


class LegacyProfile(ProviderProfile):
    def prepare_pool_entry(self, entry, api_key):
        return {"same": api_key, "empty": ""}.get(entry.id, f"{api_key}-derived")


@pytest.fixture
def profile(monkeypatch):
    prof = PairProfile(name="pairprov", base_url=SOURCE, credential_pool_base_urls=(SOURCE, TARGET))
    monkeypatch.setattr(providers, "get_provider_profile", lambda name: prof if name == "pairprov" else None)
    return prof


def _row(row_id="row-a", token=GRANT, provider="pairprov", base_url=SOURCE, priority=0):
    return PooledCredential(provider=provider, id=row_id, label=row_id, auth_type="api_key",
                            priority=priority, source="manual", access_token=token, base_url=base_url)


def _pool(*rows, provider="pairprov"):
    return CredentialPool(provider, list(rows or [_row()]))


def _method(pool, name):
    method = getattr(pool, name, None)
    assert callable(method), f"credential pool must provide {name}()"
    return method


def _prepare(pool, entry, source=SOURCE):
    return _method(pool, "prepare_runtime_entry")(entry, source_base_url=source)


def test_prepared_pair_is_runtime_only_and_reused(profile):
    pool = _pool()
    row = pool.entries()[0]
    prepared = _prepare(pool, row)
    assert prepared is not None
    assert (prepared.id, prepared.runtime_api_key, prepared.runtime_base_url) == ("row-a", DERIVED, TARGET)
    assert prepared.access_token == GRANT and prepared.base_url == SOURCE
    persisted = prepared.to_dict()
    assert DERIVED not in repr(persisted) and DERIVED not in repr(prepared)
    assert TARGET not in repr(persisted)
    # Swap re-prepares the same row from the moved route: reuse, never re-mint.
    assert _prepare(pool, row, source=TARGET).runtime_api_key == DERIVED
    assert [c for c in profile.calls if c[0] == "prepare"] == [("prepare", "row-a", GRANT)]
    assert pool.entry_id_for_api_key(DERIVED) == "row-a"


def test_override_dies_with_durable_grant_replace_and_reload(profile):
    pool = _pool()
    row = pool.entries()[0]
    _prepare(pool, row)
    assert replace(row).runtime_api_key == GRANT  # a replaced row is a fresh durable row
    reloaded = CredentialPool("pairprov", [PooledCredential.from_dict("pairprov", row.to_dict())])
    assert reloaded.entries()[0].runtime_api_key == GRANT
    assert reloaded.entries()[0].runtime_base_url == SOURCE
    row.access_token = "rotated-grant-0002"
    assert row.runtime_api_key == "rotated-grant-0002" and row.runtime_base_url == SOURCE
    profile.results[("prepare", "row-a")] = PoolRuntimeCredential("second-bearer", TARGET)
    assert _prepare(pool, row).runtime_api_key == "second-bearer"
    assert profile.calls[-1] == ("prepare", "row-a", "rotated-grant-0002")


@pytest.mark.parametrize("bad", [
    "http://api.business.example", "https://api.business.example/v1", "https://api.business.example?x=1",
    "https://api.business.example#f", "https://u@api.business.example", "https://api.business.example:8443",
    "https://api.business.example.evil.test", "https://evil.test", "", "https://api.business.example/",
])
def test_foreign_or_unexact_result_route_is_refused_without_residue(profile, bad):
    pool = _pool()
    row = pool.entries()[0]
    profile.results[("prepare", "row-a")] = PoolRuntimeCredential(DERIVED, bad)
    assert _prepare(pool, row) is None
    assert (row.runtime_api_key, row.runtime_base_url) == (GRANT, SOURCE)


def test_unallowlisted_source_route_is_refused(profile):
    pool = _pool()
    row = pool.entries()[0]
    assert _prepare(pool, row, source="https://elsewhere.example") is None
    assert row.runtime_api_key == GRANT


def test_active_override_is_rechecked_against_source_at_swap(profile):
    pool = _pool()
    row = pool.entries()[0]
    assert _prepare(pool, row) is not None
    assert _prepare(pool, row, source="https://elsewhere.example/v1") is None


def test_rejected_reprepare_leaves_no_cached_foreign_route(profile):
    pool = _pool()
    row = pool.entries()[0]
    _prepare(pool, row)
    row.access_token = "rotated-grant-0002"
    profile.results[("prepare", "row-a")] = PoolRuntimeCredential("bad-bearer", "https://evil.test")
    assert _prepare(pool, row) is None
    assert (row.runtime_api_key, row.runtime_base_url) == ("rotated-grant-0002", SOURCE)
    profile.results[("prepare", "row-a")] = ValueError("boom " + DERIVED)
    assert _prepare(pool, row) is None


def test_pair_without_endpoint_keeps_route_and_invalid_results_refuse(profile):
    pool = _pool()
    row = pool.entries()[0]
    profile.results[("prepare", "row-a")] = PoolRuntimeCredential("key-only-bearer")
    prepared = _prepare(pool, row, source="https://api.individual.example")
    assert (prepared.runtime_api_key, prepared.runtime_base_url) == ("key-only-bearer", SOURCE)
    for result in ({"api_key": "x", "base_url": TARGET}, PoolRuntimeCredential("", TARGET), 42):
        other = _pool(_row("row-b"))
        profile.results[("prepare", "row-b")] = result
        assert _prepare(other, other.entries()[0]) is None


def test_legacy_string_hook_keeps_endpoint_and_needs_no_allowlist(monkeypatch):
    prof = LegacyProfile(name="legacy", base_url="http://localhost:8080/v1")
    monkeypatch.setattr(providers, "get_provider_profile", lambda name: prof if name == "legacy" else None)
    rows = [_row(i, provider="legacy", base_url="http://localhost:8080/v1") for i in ("derive", "same", "empty")]
    pool = _pool(*rows, provider="legacy")
    derive, same, empty = pool.entries()
    prepared = _prepare(pool, derive, source="http://localhost:8080/v1")
    assert (prepared.runtime_api_key, prepared.runtime_base_url) == (f"{GRANT}-derived", "http://localhost:8080/v1")
    assert prepared.to_dict()["access_token"] == GRANT
    assert _prepare(pool, same, source=None).runtime_api_key == GRANT
    assert _prepare(pool, empty, source=None) is None


def test_provider_without_profile_returns_usable_row_unchanged(monkeypatch):
    monkeypatch.setattr(providers, "get_provider_profile", lambda name: None)
    pool = _pool(_row(provider="plainprov"), provider="plainprov")
    row = pool.entries()[0]
    assert _prepare(pool, row, source=SOURCE + "/") is row
    blank = _pool(_row(token="", provider="plainprov"), provider="plainprov")
    assert _prepare(blank, blank.entries()[0], source=None) is None


def test_unprepared_row_for_another_endpoint_is_never_adopted(monkeypatch):
    monkeypatch.setattr(providers, "get_provider_profile", lambda name: None)
    azure = "https://res.cognitiveservices.azure.com/openai/v1"
    pool = _pool(_row("az", provider="openai", base_url=azure),
                 _row("pub", provider="openai", base_url="https://api.openai.com/v1"), provider="openai")
    az, pub = pool.entries()
    assert _prepare(pool, pub, source=azure) is None
    assert _prepare(pool, az, source=azure) is az


def test_key_only_override_keeps_the_endpoint_veto(profile):
    pool = _pool()
    row = pool.entries()[0]
    profile.results[("prepare", "row-a")] = PoolRuntimeCredential("key-only-bearer")
    assert _prepare(pool, row, source=TARGET) is None
    assert _prepare(pool, row) is not None
    assert _prepare(pool, row, source=TARGET) is None


def test_scoped_main_auth_failure_never_rotates_on_mismatched_identity(profile):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from agent.credential_auth_recovery import recover_auth_failure

    pool = _pool(_row("row-a"), _row("row-b", token="other-grant-0003", priority=1))
    _prepare(pool, pool.entries()[0])
    rotate = Mock(return_value=True)
    agent = SimpleNamespace(provider="pairprov", _swap_credential=Mock(), _is_entitlement_failure=Mock(return_value=False))
    recovered, _ = recover_auth_failure(agent, pool, status_code=401, has_retried_429=False,
                                        error_context={}, api_key_hint=DERIVED,
                                        credential_id="row-b", rotate_and_swap=rotate)
    assert not recovered
    rotate.assert_not_called()
    assert not [c for c in profile.calls if c[0] == "refresh"]


def test_scoped_refresh_renews_exact_row_only(profile):
    pool = _pool(_row("row-a"), _row("row-b", token="other-grant-0003", priority=1))
    a, b = pool.entries()
    _prepare(pool, a)
    profile.results[("refresh", "row-a")] = PoolRuntimeCredential("renewed-bearer", TARGET)
    refresh = _method(pool, "try_refresh_runtime_matching")
    renewed = refresh(credential_id="row-a", api_key_hint=DERIVED, reason="unauthorized")
    assert renewed is not None and renewed.id == "row-a"
    assert (renewed.runtime_api_key, renewed.runtime_base_url) == ("renewed-bearer", TARGET)
    assert renewed.access_token == GRANT and pool.current().id == "row-a"
    assert b.runtime_api_key == "other-grant-0003"
    assert _prepare(pool, a, source=TARGET).runtime_api_key == "renewed-bearer"


def test_scoped_refresh_never_falls_back_to_ambient_rows(profile):
    pool = _pool(_row("row-a"), _row("row-b", token="other-grant-0003", priority=1))
    refresh = _method(pool, "try_refresh_runtime_matching")
    profile.results[("refresh", "row-b")] = PoolRuntimeCredential("wrong", TARGET)
    assert refresh(credential_id="missing", api_key_hint=None, reason="unauthorized") is None
    assert refresh(credential_id=None, api_key_hint=None, reason="unauthorized") is None
    assert refresh(credential_id=None, api_key_hint="unknown", reason="stale") is None
    assert refresh(credential_id="row-b", api_key_hint=GRANT, reason="stale") is None
    assert not [c for c in profile.calls if c[0] == "refresh"]
    assert refresh(credential_id=None, api_key_hint="other-grant-0003", reason="stale") is None
    assert refresh(credential_id="row-b", api_key_hint=None, reason="unauthorized") is None
    _prepare(pool, pool.entries()[1])
    assert refresh(credential_id="row-b", api_key_hint="other-grant-0003", reason="unauthorized") is None
    assert not [c for c in profile.calls if c[0] == "refresh"]


def test_scoped_refresh_decline_bad_route_or_error_changes_nothing(profile):
    pool = _pool()
    row = pool.entries()[0]
    _prepare(pool, row)
    refresh = _method(pool, "try_refresh_runtime_matching")
    for result in (None, PoolRuntimeCredential("evil", "https://evil.test"), RuntimeError(DERIVED)):
        profile.results[("refresh", "row-a")] = result
        assert refresh(credential_id="row-a", api_key_hint=DERIVED, reason="stale") is None
        assert (row.runtime_api_key, row.runtime_base_url) == (DERIVED, TARGET)
        assert row.last_status is None  # a declined stale repair never benches the account


def test_runtime_refresh_support_follows_explicit_hook(profile, monkeypatch):
    assert _method(_pool(), "supports_runtime_credential_refresh")() is True
    monkeypatch.setattr(providers, "get_provider_profile", lambda name: LegacyProfile(name="legacy"))
    assert _method(_pool(provider="legacy"), "supports_runtime_credential_refresh")() is False
