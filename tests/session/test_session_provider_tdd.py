"""TDD harness for the pluggable SessionDB provider (RFC #23717, Phase 1).

Contracts pinned here, before the implementation exists:

- ``SessionDBProvider`` is a strict ABC (MemoryProvider pattern, ``agent/memory_provider.py``):
  direct instantiation and incomplete subclasses are rejected, and the shipped SQLite engine
  (``SessionDB``) satisfies every abstract contract with a compatible signature.
- ``get_session_db_provider()`` is the single construction seam: SQLite by default, loud
  failure on an unknown ``sessiondb.provider`` config value.
- Payload concurrency: WAL's single writer must serialize many concurrent multi-MB appends
  with zero loss and zero corruption.
- Crash recovery: a mid-transaction crash (the testing-only ``_test_force_sigkill`` seam,
  fired from a ``pre_commit`` hook) must lose ONLY the in-flight write — committed history
  survives, the crashed payload is fully absent, and the provider recovers in place.
"""

import hashlib
import inspect
import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from hermes_state import SessionDB
from hermes_state_provider import SessionDBProvider, get_session_db_provider


def _json_payload(target_bytes: int, seed: int) -> str:
    """A JSON blob of ~target_bytes, deterministic per seed."""
    skeleton = json.dumps({"seed": seed, "data": ""})
    return json.dumps({"seed": seed, "data": "x" * max(target_bytes - len(skeleton), 0)})


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.fixture
def provider(tmp_path):
    db = get_session_db_provider(db_path=tmp_path / "state.db")
    yield db
    db.close()


class TestInterfaceEnforcement:
    def test_provider_cannot_be_instantiated_directly(self):
        with pytest.raises(TypeError):
            SessionDBProvider()

    def test_incomplete_subclass_cannot_be_instantiated(self):
        class _Partial(SessionDBProvider):
            pass

        with pytest.raises(TypeError):
            _Partial()

    def test_sqlite_provider_satisfies_contract(self, provider):
        assert isinstance(provider, SessionDBProvider)
        assert provider.name == "sqlite"
        assert provider.is_available()
        for member in SessionDBProvider.__abstractmethods__:
            resolved = getattr(provider, member)
            if not callable(resolved):  # abstract property (e.g. name) resolves to a value
                assert isinstance(getattr(type(provider), member), property), member

    def test_abstract_signatures_match_sqlite_implementation(self):
        """Every explicit contract parameter must exist on SessionDB's implementation
        (or pass through its **kwargs) — a relationship between the two, not a snapshot."""
        for member in sorted(SessionDBProvider.__abstractmethods__):
            contract_attr = getattr(SessionDBProvider, member)
            contract_fn = contract_attr.fget if isinstance(contract_attr, property) else contract_attr
            contract = inspect.signature(contract_fn)
            impl_attr = getattr(SessionDB, member)
            impl_fn = impl_attr.fget if isinstance(impl_attr, property) else impl_attr
            impl = inspect.signature(impl_fn)
            impl_params = set(impl.parameters) - {"self"}
            impl_accepts_kwargs = any(
                p.kind is inspect.Parameter.VAR_KEYWORD for p in impl.parameters.values()
            )
            for pname, param in contract.parameters.items():
                if pname == "self" or param.kind is inspect.Parameter.VAR_KEYWORD:
                    continue
                assert pname in impl_params or impl_accepts_kwargs, (
                    f"{member}: contract param '{pname}' missing from SessionDB"
                )


class TestFactory:
    def test_default_provider_is_sqlite(self, tmp_path):
        db = get_session_db_provider(db_path=tmp_path / "state.db")
        try:
            assert isinstance(db, SessionDB)
            assert db.name == "sqlite"
        finally:
            db.close()

    def test_explicit_sqlite_config(self, tmp_path):
        db = get_session_db_provider(
            db_path=tmp_path / "state.db", config={"sessiondb": {"provider": "sqlite"}}
        )
        try:
            assert isinstance(db, SessionDB)
        finally:
            db.close()

    def test_unknown_provider_fails_loudly(self, tmp_path):
        with pytest.raises(ValueError, match="does-not-exist"):
            get_session_db_provider(
                db_path=tmp_path / "state.db",
                config={"sessiondb": {"provider": "does-not-exist"}},
            )


class TestPayloadConcurrency:
    WORKERS = 4
    PAYLOAD_BYTES = 10 * 1024 * 1024

    def test_concurrent_large_payload_appends(self, provider):
        provider.create_session("s1", source="cli")
        payloads = {i: _json_payload(self.PAYLOAD_BYTES, seed=i) for i in range(self.WORKERS)}
        expected = {_sha(p) for p in payloads.values()}
        barrier = threading.Barrier(self.WORKERS)

        def _worker(i: int) -> None:
            barrier.wait(timeout=30)
            provider.append_message("s1", role="user", content=payloads[i])

        with ThreadPoolExecutor(max_workers=self.WORKERS) as pool:
            futures = [pool.submit(_worker, i) for i in payloads]
            for fut in futures:
                fut.result(timeout=300)

        rows = provider.get_messages("s1")
        assert len(rows) == self.WORKERS
        assert {_sha(r["content"]) for r in rows} == expected
        assert provider.message_count("s1") == self.WORKERS
        # The provider stays usable after the contended burst.
        assert provider.get_session("s1")["id"] == "s1"

    def test_single_oversized_payload_roundtrip(self, provider):
        provider.create_session("s1", source="cli")
        blob = _json_payload(50 * 1024 * 1024, seed=0)
        provider.append_message("s1", role="user", content=blob)
        (row,) = provider.get_messages("s1")
        assert row["content"] == blob


class TestCrashRecovery:
    def test_mid_write_crash_is_atomic_and_recovers(self, provider):
        provider.create_session("s1", source="cli")
        provider.append_message("s1", role="user", content="before-crash")

        provider.add_pre_commit_hook(
            lambda conn: setattr(provider, "_test_force_sigkill", True)
        )
        with pytest.raises(sqlite3.OperationalError, match="simulated mid-write crash"):
            provider.append_message("s1", role="user", content="crashed-payload")
        provider.clear_write_hooks()

        # Recovery in place: the SAME handle reopens its writer and keeps serving.
        provider.append_message("s1", role="user", content="after-crash")
        contents = [r["content"] for r in provider.get_messages("s1")]
        assert contents == ["before-crash", "after-crash"]
        # Search agrees: committed history is findable, the crashed write never existed.
        assert any(
            "before-crash" in (hit.get("content") or "")
            for hit in provider.search_messages("before-crash")
        )
        assert not any(
            "crashed-payload" in (hit.get("content") or "")
            for hit in provider.search_messages("crashed-payload")
        )


class TestWriteHooks:
    def test_pre_commit_veto_rolls_back(self, provider):
        provider.create_session("s1", source="cli")

        def _veto(conn):
            raise RuntimeError("veto")

        provider.add_pre_commit_hook(_veto)
        with pytest.raises(RuntimeError, match="veto"):
            provider.append_message("s1", role="user", content="x")
        provider.clear_write_hooks()
        assert provider.get_messages("s1") == []

    def test_post_write_fires_only_after_successful_commit(self, provider):
        provider.create_session("s1", source="cli")
        calls = []
        provider.add_post_write_hook(lambda: calls.append("post"))
        provider.append_message("s1", role="user", content="x")
        assert calls == ["post"]

        provider.clear_write_hooks()
        provider.add_post_write_hook(lambda: calls.append("post"))
        provider.add_pre_commit_hook(
            lambda conn: setattr(provider, "_test_force_sigkill", True)
        )
        with pytest.raises(sqlite3.OperationalError):
            provider.append_message("s1", role="user", content="y")
        assert calls == ["post"]  # a crashed write never reaches post_write


class TestConnectionFactoryDI:
    def test_conn_factory_yields_temporary_schema(self, tmp_path):
        injected_path = tmp_path / "injected.db"
        made = []

        def _factory() -> sqlite3.Connection:
            conn = sqlite3.connect(
                str(injected_path), check_same_thread=False, timeout=1.0,
                isolation_level=None,
            )
            made.append(conn)
            return conn

        db = SessionDB(db_path=tmp_path / "ignored.db", conn_factory=_factory)
        try:
            db.create_session("s1", source="cli")
            assert made, "writer connection must come from the injected factory"
            assert injected_path.exists()
            assert not (tmp_path / "ignored.db").exists()
            assert db.get_session("s1")["id"] == "s1"
        finally:
            db.close()
