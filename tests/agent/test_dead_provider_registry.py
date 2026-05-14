"""Tests for DeadProviderRegistry — persistent dead-provider tracking."""

from __future__ import annotations

import json
import os
import tempfile
import time

import pytest

from agent.dead_provider_registry import (
    DEFAULT_DEAD_TTL_SECONDS,
    DeadProviderRecord,
    DeadProviderRegistry,
    HealthCheckProbe,
)


# ── DeadProviderRecord ────────────────────────────────────────────────────


class TestDeadProviderRecord:
    def test_create_record(self):
        """A record stores provider, model, and reason."""
        rec = DeadProviderRecord(
            provider="openai",
            model="gpt-4",
            reason="5 consecutive 500 errors",
        )
        assert rec.provider == "openai"
        assert rec.model == "gpt-4"
        assert rec.reason == "5 consecutive 500 errors"
        assert rec.marked_at > 0
        assert rec.ttl_seconds == DEFAULT_DEAD_TTL_SECONDS

    def test_is_dead_within_ttl(self):
        """A record is dead until its TTL expires."""
        rec = DeadProviderRecord(provider="p", model="m", reason="test")
        assert rec.is_dead() is True

    def test_is_dead_expired_ttl(self):
        """A record with an expired TTL is no longer dead."""
        rec = DeadProviderRecord(
            provider="p", model="m", reason="test",
            marked_at=time.monotonic() - 100,
            ttl_seconds=10,
        )
        assert rec.is_dead() is False

    def test_is_dead_zero_ttl(self):
        """A record with zero TTL is immediately expired."""
        rec = DeadProviderRecord(
            provider="p", model="m", reason="test",
            ttl_seconds=0,
        )
        assert rec.is_dead() is False

    def test_to_dict_round_trip(self):
        """to_dict() and from_dict() preserve all fields."""
        rec = DeadProviderRecord(
            provider="anthropic",
            model="claude-sonnet-4",
            reason="timeout on 3 consecutive requests",
            marked_at=12345.0,
            ttl_seconds=300,
        )
        d = rec.to_dict()
        restored = DeadProviderRecord.from_dict(d)
        assert restored.provider == rec.provider
        assert restored.model == rec.model
        assert restored.reason == rec.reason
        assert restored.marked_at == rec.marked_at
        assert restored.ttl_seconds == rec.ttl_seconds

    def test_from_dict_defaults(self):
        """from_dict fills in sensible defaults for missing fields."""
        d = {"provider": "deepseek", "model": "deepseek-chat", "reason": "n/a"}
        rec = DeadProviderRecord.from_dict(d)
        assert rec.provider == "deepseek"
        assert rec.marked_at > 0
        assert rec.ttl_seconds == DEFAULT_DEAD_TTL_SECONDS

    def test_repr(self):
        """__repr__ includes provider, model, and reason."""
        rec = DeadProviderRecord(provider="x", model="y", reason="z")
        r = repr(rec)
        assert "x" in r
        assert "y" in r
        assert "z" in r


# ── Shared fixture ────────────────────────────────────────────────────────


@pytest.fixture
def _db_path():
    """Temporary SQLite database path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


# ── DeadProviderRegistry ──────────────────────────────────────────────────


class TestDeadProviderRegistry:
    @pytest.fixture
    def db_path(self, _db_path):
        return _db_path

    def test_init_creates_table(self, db_path):
        """The registry creates the SQLite table on init."""
        reg = DeadProviderRegistry(db_path=db_path)
        reg._ensure_db()
        # Verify table exists by querying
        with reg._connect() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='dead_providers'"
            ).fetchall()
            assert len(rows) == 1

    def test_mark_dead_adds_entry(self, db_path):
        """mark_provider_dead adds a record that is_dead returns True for."""
        reg = DeadProviderRegistry(db_path=db_path)
        reg.mark_provider_dead("openai", "gpt-4", "test failure")
        assert reg.is_provider_dead("openai", "gpt-4") is True

    def test_mark_dead_updates_existing(self, db_path):
        """Marking the same provider/model again updates the timestamp and reason."""
        reg = DeadProviderRegistry(db_path=db_path)
        reg.mark_provider_dead("openai", "gpt-4", "first failure")
        reg.mark_provider_dead("openai", "gpt-4", "more severe failure")
        assert reg.is_provider_dead("openai", "gpt-4") is True
        entry = reg.get_dead_entry("openai", "gpt-4")
        assert entry is not None
        assert entry.reason == "more severe failure"

    def test_mark_alive_removes_entry(self, db_path):
        """mark_provider_alive removes the entry entirely."""
        reg = DeadProviderRegistry(db_path=db_path)
        reg.mark_provider_dead("openai", "gpt-4", "test")
        assert reg.is_provider_dead("openai", "gpt-4") is True
        reg.mark_provider_alive("openai", "gpt-4")
        assert reg.is_provider_dead("openai", "gpt-4") is False

    def test_mark_alive_idempotent(self, db_path):
        """mark_provider_alive on a non-dead entry does not error."""
        reg = DeadProviderRegistry(db_path=db_path)
        reg.mark_provider_alive("nonexistent", "model")  # no error

    def test_list_dead_providers(self, db_path):
        """list_dead_providers returns all non-expired dead entries."""
        reg = DeadProviderRegistry(db_path=db_path)
        reg.mark_provider_dead("a", "m1", "err1")
        reg.mark_provider_dead("b", "m2", "err2")
        dead = reg.list_dead_providers()
        assert len(dead) == 2
        assert any(e.provider == "a" and e.model == "m1" for e in dead)
        assert any(e.provider == "b" and e.model == "m2" for e in dead)

    def test_list_dead_providers_excludes_expired(self, db_path):
        """list_dead_providers skips entries whose TTL has expired."""
        reg = DeadProviderRegistry(db_path=db_path)
        reg._add_raw(
            DeadProviderRecord(
                provider="expired", model="x",
                reason="old", marked_at=time.monotonic() - 1000, ttl_seconds=10,
            )
        )
        reg.mark_provider_dead("alive", "y", "current")
        dead = reg.list_dead_providers()
        assert len(dead) == 1
        assert dead[0].provider == "alive"

    def test_skip_dead_providers(self, db_path):
        """A list of providers can be filtered to skip dead ones."""
        reg = DeadProviderRegistry(db_path=db_path)
        reg.mark_provider_dead("openai", "gpt-4", "dead")
        candidates = [
            ("openai", "gpt-4"),
            ("anthropic", "claude-sonnet-4"),
            ("deepseek", "deepseek-chat"),
        ]
        filtered = reg.filter_alive(candidates)
        assert len(filtered) == 2
        assert ("openai", "gpt-4") not in filtered
        assert ("anthropic", "claude-sonnet-4") in filtered

    def test_clear_all(self, db_path):
        """clear clears the entire dead provider table."""
        reg = DeadProviderRegistry(db_path=db_path)
        reg.mark_provider_dead("a", "m1", "err")
        reg.mark_provider_dead("b", "m2", "err")
        count = reg.clear()
        assert count == 2
        assert len(reg.list_dead_providers()) == 0

    def test_get_dead_entry_nonexistent(self, db_path):
        """get_dead_entry returns None for a live provider."""
        reg = DeadProviderRegistry(db_path=db_path)
        assert reg.get_dead_entry("nope", "x") is None

    def test_get_dead_entry_expired(self, db_path):
        """get_dead_entry returns None for an expired entry."""
        reg = DeadProviderRegistry(db_path=db_path)
        reg._add_raw(
            DeadProviderRecord(
                provider="exp", model="m",
                reason="old", marked_at=time.monotonic() - 1000, ttl_seconds=10,
            )
        )
        # Should be expired and not returned
        entry = reg.get_dead_entry("exp", "m")
        assert entry is None

    def test_dead_providers_count_updates(self, db_path):
        """The count of dead providers changes as entries are added/removed."""
        reg = DeadProviderRegistry(db_path=db_path)
        assert reg.dead_count() == 0
        reg.mark_provider_dead("x", "y", "z")
        assert reg.dead_count() == 1
        reg.mark_provider_alive("x", "y")
        assert reg.dead_count() == 0

    def test_ttl_isolation(self, db_path):
        """Each dead entry can have its own TTL."""
        reg = DeadProviderRegistry(db_path=db_path)
        reg.mark_provider_dead("a", "m1", "err", ttl_seconds=10)
        reg.mark_provider_dead("b", "m2", "err", ttl_seconds=3600)
        entry_a = reg.get_dead_entry("a", "m1")
        entry_b = reg.get_dead_entry("b", "m2")
        assert entry_a is not None
        assert entry_b is not None
        assert entry_a.ttl_seconds == 10
        assert entry_b.ttl_seconds == 3600

    def test_thread_safety(self, db_path):
        """Multiple concurrent marks don't corrupt the database."""
        import concurrent.futures

        reg = DeadProviderRegistry(db_path=db_path)
        providers = [(f"p{i}", f"m{i}", f"reason{i}") for i in range(50)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(lambda args: reg.mark_provider_dead(*args), providers))
        assert reg.dead_count() == 50

    def test_context_manager(self, db_path):
        """The registry works as a context manager (__enter__/__exit__)."""
        with DeadProviderRegistry(db_path=db_path) as reg:
            reg.mark_provider_dead("ctx", "mgr", "test")
            assert reg.is_provider_dead("ctx", "mgr") is True
        # After exit, the DB file persists
        assert os.path.isfile(db_path)

    def test_default_db_path(self):
        """The default db_path is under get_hermes_home()."""
        reg = DeadProviderRegistry()
        assert "dead_providers.db" in str(reg.db_path)


# ── HealthCheckProbe ──────────────────────────────────────────────────────


class _FakeProvider:
    """Minimal provider stub for health-check testing."""

    def __init__(self, provider: str, model: str, health_check_ok: bool = True):
        self.provider = provider
        self.model = model
        self.health_check_ok = health_check_ok
        self.last_health_check_args: tuple | None = None

    def health_check(self, provider: str, model: str) -> bool:
        self.last_health_check_args = (provider, model)
        return self.health_check_ok


class _FakeRegistry:
    """Minimal registry stub for health-check testing."""

    def __init__(self):
        self.marked_alive: list[tuple[str, str]] = []
        self.dead_entries: list[DeadProviderRecord] = []

    def list_dead_providers(self) -> list[DeadProviderRecord]:
        return self.dead_entries

    def mark_provider_alive(self, provider: str, model: str) -> None:
        self.marked_alive.append((provider, model))


class TestHealthCheckProbe:
    def test_revives_responsive_provider(self):
        """A health check that succeeds marks the provider alive."""
        registry = _FakeRegistry()
        provider = _FakeProvider("openai", "gpt-4", health_check_ok=True)
        probe = HealthCheckProbe(registry, provider)
        registry.dead_entries = [
            DeadProviderRecord("openai", "gpt-4", "test dead"),
        ]
        result = probe.check_once("openai", "gpt-4")
        assert result is True
        assert ("openai", "gpt-4") in registry.marked_alive

    def test_leaves_unresponsive_provider_dead(self):
        """A health check that fails leaves the provider dead."""
        registry = _FakeRegistry()
        provider = _FakeProvider("openai", "gpt-4", health_check_ok=False)
        probe = HealthCheckProbe(registry, provider)
        registry.dead_entries = [
            DeadProviderRecord("openai", "gpt-4", "test dead"),
        ]
        result = probe.check_once("openai", "gpt-4")
        assert result is False
        assert len(registry.marked_alive) == 0

    def test_checks_all_dead_providers(self):
        """check_all processes every entry from list_dead_providers."""
        registry = _FakeRegistry()
        provider = _FakeProvider("a", "m1", health_check_ok=True)
        probe = HealthCheckProbe(registry, provider)
        registry.dead_entries = [
            DeadProviderRecord("a", "m1", "err1"),
            DeadProviderRecord("b", "m2", "err2"),
        ]
        results = probe.check_all()
        assert len(results) == 2
        assert all(r is True for r in results)

    def test_check_all_no_dead_providers(self):
        """check_all with no dead entries returns an empty list."""
        registry = _FakeRegistry()
        provider = _FakeProvider("a", "m1")
        probe = HealthCheckProbe(registry, provider)
        results = probe.check_all()
        assert results == []

    def test_logs_check_results(self, caplog):
        """Health check results are logged at INFO level."""
        import logging
        caplog.set_level(logging.INFO)

        registry = _FakeRegistry()
        provider = _FakeProvider("openai", "gpt-4", health_check_ok=True)
        probe = HealthCheckProbe(registry, provider)
        registry.dead_entries = [
            DeadProviderRecord("openai", "gpt-4", "test dead"),
        ]
        probe.check_all()
        assert "openai/gpt-4" in caplog.text
        assert "revived" in caplog.text or "alive" in caplog.text

    def test_logs_still_dead(self, caplog):
        """Health check logs when a provider remains dead."""
        import logging
        caplog.set_level(logging.INFO)

        registry = _FakeRegistry()
        provider = _FakeProvider("openai", "gpt-4", health_check_ok=False)
        probe = HealthCheckProbe(registry, provider)
        registry.dead_entries = [
            DeadProviderRecord("openai", "gpt-4", "test dead"),
        ]
        probe.check_all()
        assert "still dead" in caplog.text or "unreachable" in caplog.text or "failed" in caplog.text

    def test_health_check_against_provider_model_pairs(self, _db_path):
        """Integration: full probe cycle with real registry."""
        reg = DeadProviderRegistry(db_path=_db_path)
        reg.mark_provider_dead("openai", "gpt-4", "simulated dead")

        class _LiveChecker(_FakeProvider):
            def health_check(self, provider, model):
                return True

        checker = _LiveChecker("openai", "gpt-4")
        probe = HealthCheckProbe(reg, checker)
        probe.check_all()
        # Should have been revived
        assert reg.is_provider_dead("openai", "gpt-4") is False

    def test_health_check_twice_revive_then_stay_dead(self, _db_path):
        """A provider revived on first check stays revived on second."""
        reg = DeadProviderRegistry(db_path=_db_path)
        reg.mark_provider_dead("openai", "gpt-4", "simulated dead")

        call_count = 0

        class _Checker(_FakeProvider):
            def health_check(self, provider, model):
                nonlocal call_count
                call_count += 1
                return True

        checker = _Checker("openai", "gpt-4")
        probe = HealthCheckProbe(reg, checker)
        probe.check_all()
        assert call_count == 1  # only checked while dead

        probe.check_all()
        assert call_count == 1  # not checked again — already alive

    def test_skip_provider_not_in_registry(self, _db_path):
        """Health check only checks providers in the dead registry."""
        reg = DeadProviderRegistry(db_path=_db_path)
        # Nothing marked dead

        class _Checker(_FakeProvider):
            def health_check(self, provider, model):
                raise AssertionError("should not be called")

        checker = _Checker("openai", "gpt-4")
        probe = HealthCheckProbe(reg, checker)
        probe.check_all()  # no error