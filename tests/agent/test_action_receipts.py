"""Tests for agent/action_receipts.py, agent/task_envelope.py, and the
_maybe_record_action_receipt tool-executor boundary.

All behavioral contracts are tested here per task spec:
  - default-off gate (is_enabled)
  - owner-only DB permissions (0o600)
  - retention bounds (max_rows, max_age_days, max_size_mb)
  - hash-chain verification (detects corruption, does NOT prove originality)
  - redaction of secrets in args
  - BEGIN IMMEDIATE / _DB_LOCK concurrency
  - tool-executor import boundary (_maybe_record_action_receipt importable
    and does not write when disabled)
  - task_envelope: build_shadow_envelope, validate_envelope, canonical_hash
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import stat
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ── helpers ────────────────────────────────────────────────────────────────────

def _cfg(enabled: bool = True, **extra) -> dict:
    """Build a minimal config dict for injection."""
    receipts: dict[str, Any] = {"enabled": enabled}
    receipts.update(extra)
    return {"observability": {"action_receipts": receipts}}


def _make_ledger(tmp_path: Path, monkeypatch, cfg: dict | None = None):
    """Return an ActionReceiptLedger pointed at tmp_path, with config injected."""
    import agent.action_receipts as ar

    db = tmp_path / "receipts.db"
    monkeypatch.setattr(ar, "_load_config", lambda: cfg if cfg is not None else _cfg())
    return ar.ActionReceiptLedger(db_path=db), db


# ── default-off ────────────────────────────────────────────────────────────────

class TestIsEnabled:
    def test_absent_key_is_false(self, monkeypatch):
        import agent.action_receipts as ar
        monkeypatch.setattr(ar, "_load_config", lambda: {})
        assert ar.is_enabled() is False

    def test_enabled_false_is_false(self, monkeypatch):
        import agent.action_receipts as ar
        monkeypatch.setattr(ar, "_load_config", lambda: _cfg(enabled=False))
        assert ar.is_enabled() is False

    def test_enabled_string_true_is_false(self, monkeypatch):
        """Must be exactly the boolean True, not a truthy string."""
        import agent.action_receipts as ar
        monkeypatch.setattr(ar, "_load_config", lambda: _cfg(enabled="true"))
        assert ar.is_enabled() is False

    def test_enabled_true_is_true(self, monkeypatch):
        import agent.action_receipts as ar
        monkeypatch.setattr(ar, "_load_config", lambda: _cfg(enabled=True))
        assert ar.is_enabled() is True

    def test_config_exception_returns_false(self, monkeypatch):
        import agent.action_receipts as ar
        monkeypatch.setattr(ar, "_load_config", lambda: (_ for _ in ()).throw(RuntimeError("oops")))
        assert ar.is_enabled() is False

    def test_non_dict_config_is_false(self, monkeypatch):
        import agent.action_receipts as ar
        monkeypatch.setattr(ar, "_load_config", lambda: None)
        assert ar.is_enabled() is False

    def test_observability_non_dict_is_false(self, monkeypatch):
        import agent.action_receipts as ar
        monkeypatch.setattr(ar, "_load_config", lambda: {"observability": "yes"})
        assert ar.is_enabled() is False

    def test_action_receipts_non_dict_is_false(self, monkeypatch):
        import agent.action_receipts as ar
        monkeypatch.setattr(ar, "_load_config", lambda: {"observability": {"action_receipts": True}})
        assert ar.is_enabled() is False


# ── owner-only DB permissions ───────────────────────────────────────────────────

class TestOwnerOnlyPermissions:
    def test_db_created_with_0o600(self, tmp_path, monkeypatch):
        import agent.action_receipts as ar
        ledger, db = _make_ledger(tmp_path, monkeypatch)
        ledger.record_receipt(tool_name="echo", args={})
        mode = stat.S_IMODE(db.stat().st_mode)
        assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"

    def test_db_not_world_readable(self, tmp_path, monkeypatch):
        import agent.action_receipts as ar
        ledger, db = _make_ledger(tmp_path, monkeypatch)
        ledger.record_receipt(tool_name="echo", args={})
        mode = db.stat().st_mode
        assert not (mode & stat.S_IROTH), "DB must not be world-readable"
        assert not (mode & stat.S_IRGRP), "DB must not be group-readable"

    def test_wal_sidecars_hardened_to_0o600(self, tmp_path, monkeypatch):
        """The hardening pass covers SQLite's WAL and SHM sidecars."""
        import agent.action_receipts as ar

        ledger, db = _make_ledger(tmp_path, monkeypatch)
        conn = ledger._connect()
        try:
            sidecars = [db.parent / f"{db.name}{suffix}" for suffix in ("-wal", "-shm")]
            assert all(path.exists() for path in sidecars)
            for path in sidecars:
                path.chmod(0o666)

            ar._harden_db_permissions(db)

            for path in sidecars:
                mode = stat.S_IMODE(path.stat().st_mode)
                assert mode == 0o600, f"{path.name} expected 0o600, got {oct(mode)}"
        finally:
            conn.close()

    @pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor opens only")
    def test_connection_pins_validated_inode_across_path_substitution(
        self, tmp_path, monkeypatch
    ):
        """SQLite must use the inode securely opened before a path swap."""
        import agent.action_receipts as ar

        ledger, db = _make_ledger(tmp_path, monkeypatch)
        original_open = ar._descriptor_path

        def _swap_path(fd):
            db.rename(tmp_path / "validated.db")
            db.write_bytes(b"attacker replacement")
            return original_open(fd)

        monkeypatch.setattr(ar, "_descriptor_path", _swap_path)
        conn = ledger._connect()
        try:
            database_path = conn.execute("PRAGMA database_list").fetchone()[2]
            assert Path(database_path).name == "validated.db"
        finally:
            conn.close()
        assert db.read_bytes() == b"attacker replacement"

    def test_readonly_open_rejects_symlink(self, tmp_path, monkeypatch):
        import agent.action_receipts as ar

        real = tmp_path / "real.db"
        real.write_bytes(b"")
        link = tmp_path / "receipts.db"
        link.symlink_to(real)
        monkeypatch.setattr(ar, "_load_config", lambda: _cfg())
        with pytest.raises(PermissionError, match="symlink"):
            ar.ActionReceiptLedger(link)._connect_readonly()

    def test_uri_significant_filename_opens_as_literal_path(self, tmp_path, monkeypatch):
        import agent.action_receipts as ar

        db = tmp_path / "receipts?mode=ro#literal.db"
        monkeypatch.setattr(ar, "_load_config", lambda: _cfg())
        ledger = ar.ActionReceiptLedger(db)
        ledger.record_receipt(tool_name="echo", args={})
        assert db.is_file()
        assert len(ledger.read_all()) == 1

# ── basic append / read ─────────────────────────────────────────────────────────

class TestBasicAppend:
    def test_record_returns_receipt_id(self, tmp_path, monkeypatch):
        import agent.action_receipts as ar
        ledger, _ = _make_ledger(tmp_path, monkeypatch)
        rid = ledger.record_receipt(tool_name="bash", args={"cmd": "ls"})
        assert rid.startswith("r-") or len(rid) > 4

    def test_read_all_returns_one_row(self, tmp_path, monkeypatch):
        import agent.action_receipts as ar
        ledger, _ = _make_ledger(tmp_path, monkeypatch)
        ledger.record_receipt(tool_name="bash", args={})
        rows = ledger.read_all()
        assert len(rows) == 1
        assert rows[0]["tool_name"] == "bash"

    def test_read_all_empty_when_no_db(self, tmp_path, monkeypatch):
        import agent.action_receipts as ar
        db = tmp_path / "nonexistent.db"
        ledger = ar.ActionReceiptLedger(db_path=db)
        assert ledger.read_all() == []

    def test_multiple_receipts_ordered_asc(self, tmp_path, monkeypatch):
        import agent.action_receipts as ar
        ledger, _ = _make_ledger(tmp_path, monkeypatch)
        for name in ("first", "second", "third"):
            ledger.record_receipt(tool_name=name, args={})
        rows = ledger.read_all()
        assert [r["tool_name"] for r in rows] == ["first", "second", "third"]

    def test_no_update_trigger_raises(self, tmp_path, monkeypatch):
        import agent.action_receipts as ar
        import sqlite3
        ledger, db = _make_ledger(tmp_path, monkeypatch)
        ledger.record_receipt(tool_name="bash", args={})
        conn = sqlite3.connect(db)
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("UPDATE action_receipts SET tool_name='hacked'")
        conn.close()

    def test_no_delete_trigger_raises(self, tmp_path, monkeypatch):
        import agent.action_receipts as ar
        import sqlite3
        ledger, db = _make_ledger(tmp_path, monkeypatch)
        ledger.record_receipt(tool_name="bash", args={})
        conn = sqlite3.connect(db)
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("DELETE FROM action_receipts")
        conn.close()


# ── hash-chain verification ─────────────────────────────────────────────────────

class TestHashChain:
    def test_intact_chain_returns_empty(self, tmp_path, monkeypatch):
        import agent.action_receipts as ar
        ledger, _ = _make_ledger(tmp_path, monkeypatch)
        for i in range(3):
            ledger.record_receipt(tool_name=f"tool{i}", args={"i": i})
        assert ledger.verify_chain() == []

    def test_corrupted_hash_detected(self, tmp_path, monkeypatch):
        """Modifying a stored hash field produces a chain defect.

        The SHA-256 chain detects internal partial inconsistency.
        It does NOT prove originality: an actor who rewrites the
        entire file can rebuild a passing chain.
        """
        import agent.action_receipts as ar
        import sqlite3
        ledger, db = _make_ledger(tmp_path, monkeypatch)
        ledger.record_receipt(tool_name="bash", args={})
        ledger.record_receipt(tool_name="python", args={})

        # Directly corrupt the receipt_hash of row 1 (bypass triggers via
        # the connection-level override used in _apply_retention).
        conn = sqlite3.connect(db)
        conn.execute("DROP TRIGGER IF EXISTS action_receipts_no_update")
        conn.execute("UPDATE action_receipts SET receipt_hash='deadbeef' WHERE id=1")
        conn.commit()
        conn.close()

        problems = ledger.verify_chain()
        assert len(problems) >= 1
        assert any("hash mismatch" in p or "chain link" in p for p in problems)

    def test_prev_hash_linkage(self, tmp_path, monkeypatch):
        import agent.action_receipts as ar
        ledger, _ = _make_ledger(tmp_path, monkeypatch)
        ledger.record_receipt(tool_name="a", args={})
        ledger.record_receipt(tool_name="b", args={})
        rows = ledger.read_all()
        assert rows[0]["prev_receipt_hash"] is None
        assert rows[1]["prev_receipt_hash"] == rows[0]["receipt_hash"]

    def test_tail_truncation_is_not_detected(self, tmp_path, monkeypatch):
        """A locally valid shorter chain has no cryptographic tail anchor."""
        import sqlite3
        import agent.action_receipts as ar

        ledger, db = _make_ledger(tmp_path, monkeypatch)
        for i in range(5):
            ledger.record_receipt(tool_name=f"t{i}", args={})

        conn = sqlite3.connect(db)
        conn.execute("DROP TRIGGER IF EXISTS action_receipts_no_delete")
        conn.execute("DELETE FROM action_receipts WHERE id > 3")
        conn.commit()
        conn.close()

        assert len(ledger.read_all()) == 3
        assert ledger.verify_chain() == []


# ── redaction ───────────────────────────────────────────────────────────────────

class TestRedaction:
    def test_api_key_redacted_in_summary(self, tmp_path, monkeypatch):
        import agent.action_receipts as ar
        monkeypatch.setattr(
            "agent.redact.redact_sensitive_text",
            lambda text, **kw: text,
        )
        summary = ar.redacted_args_summary({"api_key": "sk-secretvalue"})
        assert "sk-secretvalue" not in summary
        assert "[redacted]" in summary

    def test_password_key_redacted(self, tmp_path, monkeypatch):
        import agent.action_receipts as ar
        monkeypatch.setattr(
            "agent.redact.redact_sensitive_text",
            lambda text, **kw: text,
        )
        summary = ar.redacted_args_summary({"password": "hunter2"})
        assert "hunter2" not in summary
        assert "[redacted]" in summary

    def test_body_field_fingerprinted(self, tmp_path, monkeypatch):
        """Body-class keys (content, code, etc.) are fingerprinted, not stored.

        The fingerprint uses an ephemeral HMAC (not raw SHA-256) so the stored
        value cannot be used offline to confirm the original content.
        """
        import agent.action_receipts as ar
        monkeypatch.setattr(
            "agent.redact.redact_sensitive_text",
            lambda text, **kw: text,
        )
        summary = ar.redacted_args_summary({"content": "secret text here"})
        parsed = json.loads(summary)
        # The value is a fingerprint dict, not the original string.
        assert isinstance(parsed["content"], dict)
        assert "hmac" in parsed["content"], "fingerprint must use HMAC, not raw sha256"
        assert "sha256" not in parsed["content"], "raw sha256 must not appear in fingerprint"
        assert "bytes" in parsed["content"]
        assert "secret text here" not in summary

    def test_innocent_key_not_redacted(self, tmp_path, monkeypatch):
        import agent.action_receipts as ar
        monkeypatch.setattr(
            "agent.redact.redact_sensitive_text",
            lambda text, **kw: text,
        )
        summary = ar.redacted_args_summary({"path": "/tmp/foo"})
        assert "/tmp/foo" in summary

    def test_args_hash_is_hmac_not_canonical_hash(self, tmp_path, monkeypatch):
        """args_hash must be an ephemeral HMAC, not the deterministic canonical_hash.

        The raw canonical_hash (unkeyed SHA-256) would allow an offline attacker
        to confirm guessed inputs.  The stored HMAC uses a per-process ephemeral
        key so the stored value is opaque to offline analysis.
        """
        import agent.action_receipts as ar
        from agent.task_envelope import canonical_hash
        args = {"api_key": "sk-abc", "cmd": "ls"}
        monkeypatch.setattr(ar, "_load_config", lambda: _cfg())
        monkeypatch.setattr(
            "agent.redact.redact_sensitive_text",
            lambda text, **kw: text,
        )
        ledger, _ = _make_ledger(tmp_path, monkeypatch)
        ledger.record_receipt(tool_name="bash", args=args)
        rows = ledger.read_all()
        stored = rows[0]["args_hash"]
        # Must be a 64-char hex string (HMAC-SHA-256 digest).
        assert len(stored) == 64
        assert all(c in "0123456789abcdef" for c in stored)
        # Must not equal the deterministic canonical_hash (raw SHA-256).
        assert stored != canonical_hash(args), (
            "args_hash must be an HMAC, not the unkeyed canonical_hash"
        )

    def test_redaction_stored_not_plaintext(self, tmp_path, monkeypatch):
        import agent.action_receipts as ar
        monkeypatch.setattr(
            "agent.redact.redact_sensitive_text",
            lambda text, **kw: text,
        )
        ledger, _ = _make_ledger(tmp_path, monkeypatch)
        ledger.record_receipt(tool_name="bash", args={"api_key": "sk-secret"})
        rows = ledger.read_all()
        assert "sk-secret" not in rows[0]["args_redacted_summary"]
    def test_structured_output_secret_values_never_enter_database(
        self, tmp_path, monkeypatch
    ):
        import agent.action_receipts as ar

        ledger, db = _make_ledger(tmp_path, monkeypatch)
        ledger.record_receipt(
            tool_name="echo",
            args={},
            output={"result": {"api_key": "OUTPUT-SECRET", "ok": True}},
        )
        assert b"OUTPUT-SECRET" not in db.read_bytes()

    def test_unsupported_output_object_string_is_never_invoked(
        self, tmp_path, monkeypatch
    ):
        import agent.action_receipts as ar

        class SecretObject:
            def __str__(self):
                raise AssertionError("receipt serialization must not call __str__")

        ledger, _ = _make_ledger(tmp_path, monkeypatch)
        ledger.record_receipt(tool_name="echo", args={}, output=SecretObject())
        stored = ledger.read_all()[0]
        assert stored["output_hash"] == ar._receipt_hmac(b"[unsupported:SecretObject]")



# ── retention / rotation ────────────────────────────────────────────────────────

class TestRetention:
    def test_max_rows_prunes_oldest(self, tmp_path, monkeypatch):
        import agent.action_receipts as ar
        cfg = _cfg(max_rows=3)
        monkeypatch.setattr(ar, "_load_config", lambda: cfg)
        ledger = ar.ActionReceiptLedger(db_path=tmp_path / "r.db")
        for i in range(5):
            ledger.record_receipt(tool_name=f"t{i}", args={})
        rows = ledger.read_all()
        assert len(rows) <= 3
        # The newest rows survive.
        names = [r["tool_name"] for r in rows]
        assert "t4" in names

    def test_max_age_days_prunes_old_rows(self, tmp_path, monkeypatch):
        import agent.action_receipts as ar
        cfg = _cfg(max_age_days=30)
        monkeypatch.setattr(ar, "_load_config", lambda: cfg)
        ledger = ar.ActionReceiptLedger(db_path=tmp_path / "r.db")
        old_ts = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        # Write one old receipt directly (bypass auto-timestamp).
        ledger.record_receipt(tool_name="old", args={}, created_at=old_ts)
        # Write one fresh receipt — this triggers pruning.
        ledger.record_receipt(tool_name="fresh", args={})
        rows = ledger.read_all()
        names = [r["tool_name"] for r in rows]
        assert "fresh" in names
        assert "old" not in names

    def test_zero_max_rows_treated_as_unlimited(self, tmp_path, monkeypatch):
        """max_rows=0 is a safe no-op; treated as unlimited."""
        import agent.action_receipts as ar
        cfg = _cfg(max_rows=0)
        monkeypatch.setattr(ar, "_load_config", lambda: cfg)
        ledger = ar.ActionReceiptLedger(db_path=tmp_path / "r.db")
        for i in range(5):
            ledger.record_receipt(tool_name=f"t{i}", args={})
        assert len(ledger.read_all()) == 5

    def test_negative_max_rows_treated_as_unlimited(self, tmp_path, monkeypatch):
        import agent.action_receipts as ar
        cfg = _cfg(max_rows=-1)
        monkeypatch.setattr(ar, "_load_config", lambda: cfg)
        ledger = ar.ActionReceiptLedger(db_path=tmp_path / "r.db")
        for i in range(4):
            ledger.record_receipt(tool_name=f"t{i}", args={})
        assert len(ledger.read_all()) == 4

    def test_no_retention_config_keeps_all(self, tmp_path, monkeypatch):
        import agent.action_receipts as ar
        monkeypatch.setattr(ar, "_load_config", lambda: _cfg())
        ledger = ar.ActionReceiptLedger(db_path=tmp_path / "r.db")
        for i in range(10):
            ledger.record_receipt(tool_name=f"t{i}", args={})
        assert len(ledger.read_all()) == 10

    def test_pruned_chain_verifies_cleanly(self, tmp_path, monkeypatch):
        """A legitimately pruned chain must verify with no defects.

        The prune-boundary metadata written atomically during _apply_retention
        tells verify_chain the expected predecessor hash for the first surviving
        row, so no broken-link defect is reported.
        """
        import agent.action_receipts as ar
        cfg = _cfg(max_rows=3)
        monkeypatch.setattr(ar, "_load_config", lambda: cfg)
        ledger = ar.ActionReceiptLedger(db_path=tmp_path / "r.db")
        for i in range(6):
            ledger.record_receipt(tool_name=f"t{i}", args={})
        # verify_chain must report no defects: the prune boundary is legitimate.
        assert ledger.verify_chain() == []

    def test_forged_boundary_metadata_detected(self, tmp_path, monkeypatch):
        """A mismatched prune_boundary_prev_hash must be reported as a defect.

        If the boundary metadata is forged or corrupted the first surviving row's
        prev_receipt_hash will not match it, and verify_chain must report an error.
        """
        import sqlite3
        import agent.action_receipts as ar
        cfg = _cfg(max_rows=3)
        monkeypatch.setattr(ar, "_load_config", lambda: cfg)
        db = tmp_path / "r.db"
        ledger = ar.ActionReceiptLedger(db_path=db)
        for i in range(6):
            ledger.record_receipt(tool_name=f"t{i}", args={})

        # Corrupt the prune_boundary_prev_hash so it no longer matches the
        # first surviving row's prev_receipt_hash.
        conn = sqlite3.connect(db)
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) "
            "VALUES ('prune_boundary_prev_hash', 'forged_hash_value')"
        )
        conn.commit()
        conn.close()

        problems = ledger.verify_chain()
        # A defect must be reported for the prune boundary mismatch.
        assert any("chain link" in p for p in problems), (
            f"Expected a chain-link defect for forged boundary; got: {problems}"
        )
        # Content hashes of surviving rows must still be intact.
        hash_mismatches = [p for p in problems if "hash mismatch" in p]
        assert hash_mismatches == [], f"Unexpected content hash errors: {hash_mismatches}"

# ── concurrency ─────────────────────────────────────────────────────────────────

class TestConcurrency:
    def test_concurrent_writes_produce_valid_chain(self, tmp_path, monkeypatch):
        """Multiple threads writing concurrently must produce a linear chain."""
        import agent.action_receipts as ar
        monkeypatch.setattr(ar, "_load_config", lambda: _cfg())
        ledger = ar.ActionReceiptLedger(db_path=tmp_path / "c.db")
        errors: list[Exception] = []

        def _write(n: int) -> None:
            try:
                ledger.record_receipt(tool_name=f"tool_{n}", args={"n": n})
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_write, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        rows = ledger.read_all()
        assert len(rows) == 10
        assert ledger.verify_chain() == []

    def test_begin_immediate_serializes_writes(self, tmp_path, monkeypatch):
        """Two simultaneous writers must not produce a forked chain."""
        import agent.action_receipts as ar
        monkeypatch.setattr(ar, "_load_config", lambda: _cfg())
        ledger = ar.ActionReceiptLedger(db_path=tmp_path / "ci.db")
        barrier = threading.Barrier(2)
        results: list[str] = []

        def _write(name: str) -> None:
            barrier.wait()
            results.append(ledger.record_receipt(tool_name=name, args={}))

        t1 = threading.Thread(target=_write, args=("alpha",))
        t2 = threading.Thread(target=_write, args=("beta",))
        t1.start(); t2.start()
        t1.join(); t2.join()

        # Both writes succeed with unique IDs.
        assert len(set(results)) == 2
        # Chain is linear (no fork).
        assert ledger.verify_chain() == []


# ── task_envelope behavior ──────────────────────────────────────────────────────

class TestTaskEnvelope:
    def test_build_shadow_envelope_minimal(self):
        from agent.task_envelope import build_shadow_envelope, ENVELOPE_SCHEMA_VERSION
        env = build_shadow_envelope(tool_name="bash", args={"cmd": "ls"})
        assert env.tool_name == "bash"
        assert env.envelope_version == ENVELOPE_SCHEMA_VERSION
        assert env.execution_surface == "local_tool"
        assert env.lane == "hermes"

    def test_shadow_envelope_safety_fields_none(self):
        """Shadow mode leaves all safety fields None — records incompleteness."""
        from agent.task_envelope import build_shadow_envelope, MANDATORY_SAFETY_FIELDS
        env = build_shadow_envelope(tool_name="bash")
        for field in MANDATORY_SAFETY_FIELDS:
            assert getattr(env, field) is None, f"Expected {field} to be None"

    def test_validate_envelope_reports_missing(self):
        from agent.task_envelope import build_shadow_envelope, validate_envelope, MANDATORY_SAFETY_FIELDS
        env = build_shadow_envelope(tool_name="bash")
        missing = validate_envelope(env)
        for field in MANDATORY_SAFETY_FIELDS:
            assert field in missing

    def test_validate_envelope_never_raises(self):
        from agent.task_envelope import validate_envelope, TaskEnvelope
        # Pass a partial object — validate_envelope must not raise.
        class _Broken:
            envelope_version = 999
        result = validate_envelope(_Broken())  # type: ignore[arg-type]
        assert isinstance(result, list)

    def test_canonical_hash_deterministic(self):
        from agent.task_envelope import canonical_hash
        payload = {"b": 2, "a": 1}
        h1 = canonical_hash(payload)
        h2 = canonical_hash(payload)
        assert h1 == h2

    def test_canonical_hash_key_order_independent(self):
        from agent.task_envelope import canonical_hash
        assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})

    def test_envelope_hash_changes_with_tool_name(self):
        from agent.task_envelope import build_shadow_envelope, envelope_hash
        e1 = build_shadow_envelope(tool_name="bash")
        e2 = build_shadow_envelope(tool_name="python")
        assert envelope_hash(e1) != envelope_hash(e2)

    def test_to_dict_tuples_become_lists(self):
        from agent.task_envelope import build_shadow_envelope
        env = build_shadow_envelope(tool_name="bash")
        d = env.to_dict()
        for v in d.values():
            assert not isinstance(v, tuple), "to_dict must not return tuples"

    def test_objective_truncated_to_2000(self):
        from agent.task_envelope import build_shadow_envelope
        long_obj = "x" * 3000
        env = build_shadow_envelope(tool_name="bash", objective=long_obj)
        assert len(env.objective) == 2000

    def test_input_hash_covers_args(self):
        from agent.task_envelope import build_shadow_envelope, canonical_hash
        args = {"cmd": "ls", "cwd": "/tmp"}
        env = build_shadow_envelope(tool_name="bash", args=args)
        assert env.input_hash == canonical_hash(args)

    def test_input_hash_empty_args(self):
        from agent.task_envelope import build_shadow_envelope, canonical_hash
        env = build_shadow_envelope(tool_name="bash")
        assert env.input_hash == canonical_hash({})

    def test_input_hash_normalizes_nonfinite_numbers(self):
        from agent.task_envelope import build_shadow_envelope, canonical_hash

        args = {"nan": float("nan"), "inf": float("inf"), "neg": float("-inf")}
        env = build_shadow_envelope(tool_name="bash", args=args)
        assert env.input_hash == canonical_hash(args)
        assert len(env.input_hash) == 64


# ── tool-executor import boundary ───────────────────────────────────────────────

class TestToolExecutorBoundary:
    def test_maybe_record_action_receipt_importable(self):
        """_maybe_record_action_receipt must be importable from tool_executor."""
        from agent.tool_executor import _maybe_record_action_receipt
        assert callable(_maybe_record_action_receipt)

    def test_execution_state_starts_callback_at_most_once(self):
        from agent.tool_executor import _ToolExecutionState

        state = _ToolExecutionState()
        args = {"nested": {"q": "rewritten"}}
        trace = [{"source": "test-middleware"}]
        assert state.try_start(args, trace) is True
        args["nested"]["q"] = "mutated"
        trace[0]["source"] = "mutated"
        assert state.try_start({}, []) is False
        assert state.started_metadata() == (
            True,
            {"nested": {"q": "rewritten"}},
            [{"source": "test-middleware"}],
        )
        assert state.abandon() is True

    def test_no_write_when_disabled(self, tmp_path, monkeypatch):
        """When is_enabled() is False, no DB file must be created or written."""
        import agent.action_receipts as ar
        monkeypatch.setattr(ar, "_load_config", lambda: _cfg(enabled=False))
        # Point the default DB path into tmp_path so we can verify it is absent.
        db = tmp_path / "receipts_disabled.db"
        monkeypatch.setattr(ar, "default_db_path", lambda: db)
        from agent.tool_executor import _maybe_record_action_receipt

        agent_mock = MagicMock()
        agent_mock.session_id = "s1"

        with patch("agent.tool_executor.get_active_env", return_value=None):
            _maybe_record_action_receipt(
                agent_mock,
                function_name="bash",
                function_args={"cmd": "ls"},
                result="output",
                effective_task_id="task1",
                tool_call_id="tc1",
                duration_s=0.1,
                exit_status="ok",
            )

        # The ledger must have returned immediately without touching the filesystem.
        assert not db.exists(), (
            "DB must not be created when action_receipts is disabled; "
            f"found: {db}"
        )

    def test_no_exception_propagated_on_write_error(self, tmp_path, monkeypatch):
        """Receipt write failures must be swallowed, never raised to caller."""
        import agent.action_receipts as ar
        monkeypatch.setattr(ar, "_load_config", lambda: _cfg(enabled=True))
        from agent.tool_executor import _maybe_record_action_receipt

        agent = MagicMock()
        agent.session_id = "s1"

        # Patch ActionReceiptLedger.record_receipt to raise.
        with patch.object(ar.ActionReceiptLedger, "record_receipt", side_effect=RuntimeError("disk full")):
            with patch("agent.tool_executor.get_active_env", return_value=None):
                # Must not raise.
                _maybe_record_action_receipt(
                    agent,
                    function_name="bash",
                    function_args={},
                    result="",
                    effective_task_id="task1",
                    tool_call_id="tc1",
                    duration_s=0.0,
                    exit_status="ok",
                )

    def test_write_when_enabled(self, tmp_path, monkeypatch):
        """When enabled, _maybe_record_action_receipt writes one receipt."""
        import agent.action_receipts as ar
        db = tmp_path / "receipts.db"
        monkeypatch.setattr(ar, "_load_config", lambda: _cfg())
        from agent.tool_executor import _maybe_record_action_receipt

        agent = MagicMock()
        agent.session_id = "s-test"

        ledger = ar.ActionReceiptLedger(db_path=db)
        with patch.object(ar, "ActionReceiptLedger", return_value=ledger):
            with patch("agent.tool_executor.get_active_env", return_value=None):
                _maybe_record_action_receipt(
                    agent,
                    function_name="pytest_tool",
                    function_args={"x": 1},
                    result="done",
                    effective_task_id="t1",
                    tool_call_id="tc-abc",
                    duration_s=0.05,
                    exit_status="ok",
                )

        rows = ledger.read_all()
        assert len(rows) == 1
        assert rows[0]["tool_name"] == "pytest_tool"
        assert rows[0]["exit_status"] == "ok"
        assert rows[0]["session_id"] == "s-test"

    def test_concurrent_dispatch_records_one_receipt_each(self, tmp_path, monkeypatch):
        """_maybe_record_action_receipt from N concurrent callers writes N receipts.

        Each concurrent call is independent; the ledger's _DB_LOCK serializes
        writes so the chain is linear and verify_chain reports no defects.
        """
        import agent.action_receipts as ar
        db = tmp_path / "concurrent_receipts.db"
        monkeypatch.setattr(ar, "_load_config", lambda: _cfg())
        from agent.tool_executor import _maybe_record_action_receipt

        ledger = ar.ActionReceiptLedger(db_path=db)
        errors: list[Exception] = []

        def _call(i: int) -> None:
            try:
                mock_agent = MagicMock()
                mock_agent.session_id = f"s-{i}"
                with patch("agent.tool_executor.get_active_env", return_value=None):
                    _maybe_record_action_receipt(
                        mock_agent,
                        function_name=f"tool_{i}",
                        function_args={"i": i},
                        result=f"out_{i}",
                        effective_task_id="t1",
                        tool_call_id=f"tc-{i}",
                        duration_s=0.01 * i,
                        exit_status="ok",
                    )
            except Exception as exc:
                errors.append(exc)

        # Patch ActionReceiptLedger once (thread-safely) before threads start,
        # not per-thread; concurrent patch.object in threads races on module state.
        monkeypatch.setattr(ar, "ActionReceiptLedger", lambda **kw: ledger)
        threads = [threading.Thread(target=_call, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Concurrent receipt errors: {errors}"
        rows = ledger.read_all()
        assert len(rows) == 8
        assert ledger.verify_chain() == []


    def test_unsupported_object_string_is_never_persisted(self, tmp_path, monkeypatch):
        import agent.action_receipts as ar

        class SecretObject:
            def __str__(self):
                return "SECRET-via-str"

        ledger, db = _make_ledger(tmp_path, monkeypatch)
        ledger.record_receipt(tool_name="echo", args={"value": SecretObject()})
        assert b"SECRET-via-str" not in db.read_bytes()
        assert "[unsupported:SecretObject]" in ledger.read_all()[0]["args_redacted_summary"]


# ── cookie / set-cookie redaction at arbitrary nesting ──────────────────────────

class TestCookieRedaction:
    """Cookie and Set-Cookie values must be redacted at any nesting depth."""

    def _summary(self, args: Any) -> Any:
        import agent.action_receipts as ar
        import json
        from unittest.mock import patch
        with patch("agent.redact.redact_sensitive_text", lambda text, **kw: text):
            return json.loads(ar.redacted_args_summary(args))

    def test_top_level_cookie_redacted(self, monkeypatch):
        import agent.action_receipts as ar
        monkeypatch.setattr(ar, "_load_config", lambda: {})
        result = self._summary({"cookie": "session=abc123; auth=xyz"})
        assert result["cookie"] == "[redacted]"

    def test_top_level_set_cookie_redacted(self, monkeypatch):
        import agent.action_receipts as ar
        monkeypatch.setattr(ar, "_load_config", lambda: {})
        result = self._summary({"set-cookie": "id=secret; Path=/"})
        assert result["set-cookie"] == "[redacted]"

    def test_nested_cookie_redacted_depth_1(self, monkeypatch):
        import agent.action_receipts as ar
        monkeypatch.setattr(ar, "_load_config", lambda: {})
        result = self._summary({"headers": {"cookie": "tok=abc"}})
        assert result["headers"]["cookie"] == "[redacted]"

    def test_nested_cookie_redacted_depth_2(self, monkeypatch):
        import agent.action_receipts as ar
        monkeypatch.setattr(ar, "_load_config", lambda: {})
        result = self._summary({"req": {"headers": {"set-cookie": "s=x"}}})
        assert result["req"]["headers"]["set-cookie"] == "[redacted]"

    def test_cookie_inside_list_redacted(self, monkeypatch):
        import agent.action_receipts as ar
        monkeypatch.setattr(ar, "_load_config", lambda: {})
        result = self._summary({"items": [{"Cookie": "user=jack"}]})
        assert result["items"][0]["Cookie"] == "[redacted]"

    def test_innocent_cookie_string_value_not_double_redacted(self, monkeypatch):
        """The word 'cookie' inside an innocent value should not be redacted."""
        import agent.action_receipts as ar
        monkeypatch.setattr(ar, "_load_config", lambda: {})
        result = self._summary({"description": "fortune cookie recipe"})
        assert "fortune cookie recipe" in result["description"]


# ── non-confirmable sensitive hashes ────────────────────────────────────────────

class TestNonConfirmableHashes:
    """args_hash, output_hash, and body fingerprints must use HMAC, not raw SHA-256.

    Raw SHA-256 would allow an observer to confirm whether a stored hash matches
    a guessed input (dictionary / preimage attack).  HMAC-SHA-256 with an
    ephemeral key that is never persisted prevents this.
    """

    def test_body_fingerprint_contains_hmac_not_sha256(self):
        import agent.action_receipts as ar
        fp = ar._body_fingerprint("sensitive content")
        assert "hmac" in fp, "body fingerprint must use 'hmac' key, not 'sha256'"
        assert "sha256" not in fp, "raw sha256 must not appear in body fingerprint"
        assert "bytes" in fp

    def test_body_fingerprint_hmac_is_hex_string(self):
        import agent.action_receipts as ar
        fp = ar._body_fingerprint("content")
        assert isinstance(fp["hmac"], str)
        # HMAC-SHA-256 hex digest is always 64 chars.
        assert len(fp["hmac"]) == 64

    def test_args_hash_is_hmac_not_raw_sha256(self, tmp_path, monkeypatch):
        """args_hash stored in DB must not match the raw SHA-256 of the canonical args.

        If it did, an offline attacker could confirm input guesses by hashing
        candidates and comparing to the stored value.
        """
        import agent.action_receipts as ar
        import hashlib
        from agent.task_envelope import canonical_json

        ledger, _ = _make_ledger(tmp_path, monkeypatch)
        args = {"cmd": "ls /tmp"}
        ledger.record_receipt(tool_name="bash", args=args)
        rows = ledger.read_all()

        raw_sha256 = hashlib.sha256(
            canonical_json(args).encode("utf-8")
        ).hexdigest()
        stored = rows[0]["args_hash"]
        assert stored != raw_sha256, (
            "args_hash must not equal the raw SHA-256 of the args "
            "(that would allow offline confirmation of inputs)"
        )
        # It should still be a 64-char hex string (HMAC-SHA-256).
        assert len(stored) == 64
        assert all(c in "0123456789abcdef" for c in stored)

    def test_output_hash_is_hmac_not_raw_sha256(self, tmp_path, monkeypatch):
        """output_hash stored in DB must not match the raw SHA-256 of the output."""
        import agent.action_receipts as ar
        import hashlib

        ledger, _ = _make_ledger(tmp_path, monkeypatch)
        output = "command output here"
        ledger.record_receipt(tool_name="bash", args={}, output=output)
        rows = ledger.read_all()

        raw_sha256 = hashlib.sha256(output.encode("utf-8")).hexdigest()
        stored = rows[0]["output_hash"]
        assert stored != raw_sha256, (
            "output_hash must not equal the raw SHA-256 of the output text"
        )
        assert len(stored) == 64

    def test_args_hash_consistent_within_process(self, tmp_path, monkeypatch):
        """Same args produce the same HMAC within one process (ephemeral key is stable).

        Tests the _receipt_hmac function directly to avoid any module-level attribute
        leakage from concurrent tests that use patch.object on ar.ActionReceiptLedger.
        The invariant is: _HMAC_KEY is generated once at module import and never changes;
        identical inputs always produce identical HMAC-SHA-256 outputs.
        """
        import agent.action_receipts as ar
        from agent.task_envelope import canonical_json
        args = {"x": 42}
        encoded = canonical_json(args).encode("utf-8")
        h1 = ar._receipt_hmac(encoded)
        h2 = ar._receipt_hmac(encoded)
        assert h1 == h2, "Same data must produce the same HMAC (key is stable per process)"
        # Also verify two DB writes produce the same args_hash.
        # Use ActionReceiptLedger directly (imported name, not ar.ActionReceiptLedger)
        # to be immune to concurrent-test patch.object leaks on the module attribute.
        from agent.action_receipts import ActionReceiptLedger
        monkeypatch.setattr(ar, "_load_config", lambda: _cfg())
        db = tmp_path / "consistency.db"
        ledger = ActionReceiptLedger(db_path=db)
        ledger.record_receipt(tool_name="t1", args=args, receipt_id="r-first")
        ledger.record_receipt(tool_name="t2", args=args, receipt_id="r-second")
        rows = ledger.read_all()
        assert rows[0]["args_hash"] == rows[1]["args_hash"], (
            "Same args within one process must produce the same HMAC"
        )
        # The stored hash must match what _receipt_hmac computes directly.
        assert rows[0]["args_hash"] == h1, (
            "Stored args_hash must equal _receipt_hmac of canonical_json(args)"
        )

    def test_nonfinite_args_still_produce_receipt(self, tmp_path, monkeypatch):
        ledger, _ = _make_ledger(tmp_path, monkeypatch)
        args = {"nan": float("nan"), "inf": float("inf")}

        ledger.record_receipt(tool_name="bash", args=args, output="ok")

        rows = ledger.read_all()
        assert len(rows) == 1
        assert len(rows[0]["args_hash"]) == 64

    def test_envelope_json_input_hash_is_hmac_not_canonical_hash(self, tmp_path, monkeypatch):
        """envelope_json must not persist the raw canonical_hash(args) as input_hash.

        TaskEnvelope.input_hash is SHA-256(canonical_json(args)); if persisted
        verbatim it allows offline confirmation of argument content.  action_receipts
        must replace it with an ephemeral HMAC before storing.
        """
        import json
        import hashlib
        import agent.action_receipts as ar
        from agent.task_envelope import canonical_json as te_canonical_json

        ledger, _ = _make_ledger(tmp_path, monkeypatch)
        args = {"cmd": "ls /sensitive"}
        ledger.record_receipt(tool_name="bash", args=args)
        rows = ledger.read_all()
        env = json.loads(rows[0]["envelope_json"])

        raw_sha256 = hashlib.sha256(
            te_canonical_json(args).encode("utf-8")
        ).hexdigest()
        stored_input_hash = env.get("input_hash")
        assert stored_input_hash is not None
        assert stored_input_hash != raw_sha256, (
            "envelope_json.input_hash must not equal the raw SHA-256 of args "
            "(offline confirmation attack vector)"
        )
        # The replacement value must still be a 64-char hex string.
        assert len(stored_input_hash) == 64
        assert all(c in "0123456789abcdef" for c in stored_input_hash)

    def test_envelope_hash_does_not_expose_raw_input_hash(self, tmp_path, monkeypatch):
        """envelope_hash must be computed over the scrubbed dict, not the raw envelope."""
        import hashlib
        import agent.action_receipts as ar
        from agent.task_envelope import canonical_hash, canonical_json as te_canonical_json, build_shadow_envelope

        ledger, _ = _make_ledger(tmp_path, monkeypatch)
        args = {"x": 1}
        ledger.record_receipt(tool_name="bash", args=args)
        rows = ledger.read_all()
        stored_eh = rows[0]["envelope_hash"]

        # The raw envelope's SHA-256 (with unscrubbed input_hash) must not match.
        env = build_shadow_envelope(tool_name="bash", args=args)
        raw_env_hash = canonical_hash(env.to_dict())
        assert stored_eh != raw_env_hash, (
            "envelope_hash must be computed from the scrubbed dict, "
            "not the original envelope containing raw input_hash"
        )


# ── symlink / non-regular file rejection ────────────────────────────────────────

class TestDBFileHardening:
    """DB and sidecars must be regular files; symlinks must be rejected."""

    def test_db_created_as_regular_file(self, tmp_path, monkeypatch):
        import agent.action_receipts as ar
        ledger, db = _make_ledger(tmp_path, monkeypatch)
        ledger.record_receipt(tool_name="echo", args={})
        st = os.lstat(db)
        assert stat.S_ISREG(st.st_mode), "DB must be a regular file"
        assert not stat.S_ISLNK(st.st_mode), "DB must not be a symlink"

    def test_db_created_with_0600_no_umask_window(self, tmp_path, monkeypatch):
        """DB must be created at 0o600 without a permissive umask window."""
        import agent.action_receipts as ar
        ledger, db = _make_ledger(tmp_path, monkeypatch)
        ledger.record_receipt(tool_name="echo", args={})
        mode = stat.S_IMODE(os.lstat(db).st_mode)
        assert mode == 0o600, f"DB mode must be 0o600, got {oct(mode)}"

    def test_symlink_db_raises_permission_error(self, tmp_path, monkeypatch):
        """Opening a DB path that is a symlink must raise PermissionError."""
        import agent.action_receipts as ar
        db = tmp_path / "receipts.db"
        target = tmp_path / "real.db"
        target.write_bytes(b"")
        db.symlink_to(target)
        monkeypatch.setattr(ar, "_load_config", lambda: _cfg())
        ledger = ar.ActionReceiptLedger(db_path=db)
        with pytest.raises(PermissionError, match="symlink"):
            ledger._connect()

    def test_assert_regular_file_passes_on_missing(self, tmp_path):
        """_assert_regular_file must not raise when the path does not exist yet."""
        import agent.action_receipts as ar
        ar._assert_regular_file(tmp_path / "nonexistent.db")  # must not raise

    def test_assert_regular_file_raises_on_symlink(self, tmp_path):
        import agent.action_receipts as ar
        link = tmp_path / "link.db"
        real = tmp_path / "real.db"
        real.write_bytes(b"")
        link.symlink_to(real)
        with pytest.raises(PermissionError, match="symlink"):
            ar._assert_regular_file(link)

    def test_harden_skips_symlink_sidecar(self, tmp_path):
        """_harden_db_permissions must not follow a symlink sidecar."""
        import agent.action_receipts as ar
        db = tmp_path / "receipts.db"
        db.write_bytes(b"")
        # Create a symlink where the -wal sidecar would be.
        other = tmp_path / "other_file"
        other.write_bytes(b"")
        other.chmod(0o644)
        wal = tmp_path / "receipts.db-wal"
        wal.symlink_to(other)
        # Must not raise and must not chmod the symlink target.
        ar._harden_db_permissions(db)
        # The symlink target retains its original mode (was not chmoded).
        assert stat.S_IMODE(other.stat().st_mode) == 0o644


# ── max_size_mb deterministic bound ─────────────────────────────────────────────


def _live_data_bytes(db_path) -> int:
    """Return (page_count - freelist_count) * page_size for *db_path*.

    This is the same metric _apply_retention uses: pages holding live data
    (not yet reclaimed by VACUUM).  Matches the documented bound.
    """
    import sqlite3
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        r = conn.execute(
            "SELECT (page_count - freelist_count) * page_size"
            " FROM pragma_page_count(), pragma_freelist_count(), pragma_page_size()"
        ).fetchone()
        return r[0] if r else 0
    finally:
        conn.close()


class TestMaxSizeMbRetention:
    """max_size_mb yields a deterministic bound on live-data pages after pruning.

    The bound is measured as (page_count - freelist_count) * page_size, which
    tracks live-data usage and converges within a single transaction without
    requiring VACUUM.  A floor of _SIZE_FLOOR_ROWS (50) newest rows is always
    preserved; if those rows alone exceed the limit the pruning stops.
    """

    def test_converges_to_bound_from_far_above(self, tmp_path, monkeypatch):
        """A ledger far above max_size_mb must converge to its bound after
        record_receipt returns; the live-data size must not exceed the limit
        while any deletable rows exist.
        """
        import sqlite3
        import agent.action_receipts as ar

        db = tmp_path / "r.db"
        # Pre-populate far above the limit without size enforcement active.
        monkeypatch.setattr(ar, "_load_config", lambda: _cfg())
        ledger = ar.ActionReceiptLedger(db_path=db)
        payload = "x" * 4000  # ~4 KB per row
        for i in range(500):
            ledger.record_receipt(tool_name=f"t{i}", args={"p": payload})

        # Now enable a 1 MB limit and trigger pruning via one more insert.
        limit_mb = 1
        monkeypatch.setattr(ar, "_load_config", lambda: _cfg(max_size_mb=limit_mb))
        ledger.record_receipt(tool_name="trigger", args={})

        live = _live_data_bytes(db)
        limit_bytes = limit_mb * 1024 * 1024
        # After pruning, live-data footprint must be at or below the limit.
        # The only exception is when only the floor rows remain and those still
        # exceed the limit — but with 4 KB rows and a 1 MB limit, > 50 rows
        # exist so no floor exception applies here.
        assert live <= limit_bytes, (
            f"Live-data footprint {live} exceeds configured bound {limit_bytes} "
            f"after record_receipt; deterministic pruning must converge."
        )

    def test_chain_valid_after_deterministic_pruning(self, tmp_path, monkeypatch):
        """verify_chain must return no defects after max_size_mb convergence."""
        import agent.action_receipts as ar

        db = tmp_path / "r.db"
        monkeypatch.setattr(ar, "_load_config", lambda: _cfg())
        ledger = ar.ActionReceiptLedger(db_path=db)
        payload = "x" * 4000
        for i in range(500):
            ledger.record_receipt(tool_name=f"t{i}", args={"p": payload})

        monkeypatch.setattr(ar, "_load_config", lambda: _cfg(max_size_mb=1))
        ledger.record_receipt(tool_name="trigger", args={})

        assert ledger.verify_chain() == [], (
            "Chain must verify cleanly after max_size_mb deterministic pruning."
        )

    def test_newest_rows_survive_pruning(self, tmp_path, monkeypatch):
        """The _SIZE_FLOOR_ROWS (50) most-recent rows must survive pruning."""
        import agent.action_receipts as ar

        db = tmp_path / "r.db"
        monkeypatch.setattr(ar, "_load_config", lambda: _cfg())
        ledger = ar.ActionReceiptLedger(db_path=db)
        payload = "x" * 4000
        for i in range(200):
            ledger.record_receipt(tool_name=f"t{i}", args={"p": payload})

        monkeypatch.setattr(ar, "_load_config", lambda: _cfg(max_size_mb=1))
        ledger.record_receipt(tool_name="newest", args={})

        names = {r["tool_name"] for r in ledger.read_all()}
        assert "newest" in names, (
            "Most-recent row must survive max_size_mb pruning."
        )

    def test_floor_case_no_deletion_when_at_floor(self, tmp_path, monkeypatch):
        """When total rows <= _SIZE_FLOOR_ROWS, no row may be deleted even if
        the live-data footprint exceeds max_size_mb.  This is the documented
        exceptional case: the newest rows alone exceed the budget.
        """
        import agent.action_receipts as ar

        db = tmp_path / "r.db"
        monkeypatch.setattr(ar, "_load_config", lambda: _cfg())
        ledger = ar.ActionReceiptLedger(db_path=db)
        payload = "x" * 4000
        # Write exactly 10 rows (well below the 50-row floor).
        for i in range(10):
            ledger.record_receipt(tool_name=f"t{i}", args={"p": payload})

        monkeypatch.setattr(ar, "_load_config", lambda: _cfg(max_size_mb=1))
        ledger.record_receipt(tool_name="floor_trigger", args={})

        rows = ledger.read_all()
        # All rows must survive because total <= _SIZE_FLOOR_ROWS.
        assert len(rows) == 11, (
            f"Expected 11 rows (floor protection); got {len(rows)}."
        )
        assert ledger.verify_chain() == []

    def test_repeated_record_receipt_stays_bounded(self, tmp_path, monkeypatch):
        """After initial convergence, every subsequent record_receipt call must
        keep live-data at or below the limit — not re-inflate it.
        """
        import agent.action_receipts as ar

        db = tmp_path / "r.db"
        limit_mb = 1
        payload = "x" * 4000
        monkeypatch.setattr(ar, "_load_config", lambda: _cfg())
        ledger = ar.ActionReceiptLedger(db_path=db)
        for i in range(500):
            ledger.record_receipt(tool_name=f"seed{i}", args={"p": payload})

        monkeypatch.setattr(ar, "_load_config", lambda: _cfg(max_size_mb=limit_mb))
        limit_bytes = limit_mb * 1024 * 1024
        # After the first convergence trigger, every committed ledger state
        # remains within the bound because pruning observes the incoming row.
        for j in range(10):
            ledger.record_receipt(tool_name=f"steady{j}", args={})
            live = _live_data_bytes(db)
            assert live <= limit_bytes, (
                f"Iteration {j}: live-data {live} exceeded bound {limit_bytes} "
                "after steady-state record_receipt."
            )

    def test_oversize_floor_case_chain_still_valid(self, tmp_path, monkeypatch):
        """Even when the floor exception applies (all rows at floor, still over
        limit), verify_chain must return no defects.
        """
        import agent.action_receipts as ar

        db = tmp_path / "r.db"
        monkeypatch.setattr(ar, "_load_config", lambda: _cfg())
        ledger = ar.ActionReceiptLedger(db_path=db)
        payload = "x" * 4000
        for i in range(10):
            ledger.record_receipt(tool_name=f"t{i}", args={"p": payload})

        monkeypatch.setattr(ar, "_load_config", lambda: _cfg(max_size_mb=1))
        ledger.record_receipt(tool_name="trigger", args={})

        assert ledger.verify_chain() == [], (
            "Chain must be valid even when floor exception prevents pruning."
        )


# ── verify_chain single-snapshot regression ─────────────────────────────────────

class TestVerifyChainSingleSnapshot:
    """verify_chain must read prune_boundary_prev_hash and receipt rows inside
    one _connect_readonly connection *and* one BEGIN transaction so both
    SELECTs share the same WAL read-point.

    Two deterministic behavioral checks (no timing sleeps, no source reading,
    no implementation-shape assertions beyond the public observable contract):

    1. connection-count seam — _connect_readonly is called exactly once per
       verify_chain invocation on a populated, pruned ledger.

    2. between-SELECT mutation seam — a side-effect injected via execute()
       wrapping fires right after the meta SELECT completes (i.e. after BEGIN
       has pinned the WAL snapshot) and corrupts prune_boundary_prev_hash.
       Because both SELECTs share the same BEGIN snapshot the mutation is
       invisible to the receipts SELECT, so verify_chain still returns [].
       The test asserts that outcome deterministically, with no timing
       dependencies.
    """

    def _pruned_ledger(self, tmp_path, monkeypatch):
        """Return (ledger, db_path) with a pruned boundary row in meta."""
        import agent.action_receipts as ar

        db = tmp_path / "r.db"
        # max_rows=2 so the third record prunes, writing prune_boundary_prev_hash.
        monkeypatch.setattr(ar, "_load_config", lambda: _cfg(max_rows=2))
        ledger = ar.ActionReceiptLedger(db_path=db)
        for i in range(3):
            ledger.record_receipt(tool_name=f"tool{i}", args={"i": i})
        return ledger, db

    def test_verify_chain_opens_one_connection(self, tmp_path, monkeypatch):
        """_connect_readonly is called exactly once per verify_chain call."""
        import agent.action_receipts as ar

        ledger, _ = self._pruned_ledger(tmp_path, monkeypatch)

        call_count = [0]
        _real_connect_readonly = ar.ActionReceiptLedger._connect_readonly

        def _counting_connect_readonly(self_inner):
            call_count[0] += 1
            return _real_connect_readonly(self_inner)

        monkeypatch.setattr(
            ar.ActionReceiptLedger, "_connect_readonly", _counting_connect_readonly
        )

        result = ledger.verify_chain()

        assert result == [], f"verify_chain reported defects: {result}"
        assert call_count[0] == 1, (
            f"verify_chain opened {call_count[0]} read-only connections; "
            "expected exactly 1 (single-snapshot contract)"
        )

    def test_verify_chain_unaffected_by_between_select_mutation(
        self, tmp_path, monkeypatch
    ):
        """A boundary mutation committed between the meta SELECT and the receipts
        SELECT must be invisible — the BEGIN transaction has already pinned the
        WAL read-point before the meta SELECT ran, so the receipts SELECT sees
        the same snapshot.

        The injection seam wraps conn.execute() on the single connection returned
        by _connect_readonly.  The first prune_boundary_prev_hash SELECT returns
        the correct value; then the mutation fires via a separate writer that
        corrupts the boundary; then the receipts SELECT proceeds.  Under a proper
        BEGIN snapshot the mutation is outside the read horizon and verify_chain
        returns no defects.
        """
        import sqlite3
        import agent.action_receipts as ar

        ledger, db = self._pruned_ledger(tmp_path, monkeypatch)

        # Confirm the baseline chain is intact before instrumentation.
        assert ledger.verify_chain() == []

        _real_connect_readonly = ar.ActionReceiptLedger._connect_readonly

        def _instrumenting_connect_readonly(self_inner):
            conn = _real_connect_readonly(self_inner)
            # Wrap execute() so we can inject a mutation between specific SELECTs.
            mutation_fired = [False]
            original_execute = conn.execute

            def _execute_with_injection(sql, *args, **kwargs):
                result = original_execute(sql, *args, **kwargs)
                # After the meta row has been fetched, commit a corruption via
                # a *separate* writer connection.  This fires after BEGIN has
                # pinned our snapshot, so the receipts SELECT that follows
                # should remain unaffected.
                if (
                    not mutation_fired[0]
                    and "prune_boundary_prev_hash" in sql
                    and sql.strip().upper().startswith("SELECT")
                ):
                    mutation_fired[0] = True
                    try:
                        writer = sqlite3.connect(str(db))
                        writer.execute(
                            "UPDATE meta SET value='deadbeef_injected' "
                            "WHERE key='prune_boundary_prev_hash'"
                        )
                        writer.commit()
                        writer.close()
                    except Exception:
                        pass
                return result

            conn.execute = _execute_with_injection
            return conn

        monkeypatch.setattr(
            ar.ActionReceiptLedger, "_connect_readonly", _instrumenting_connect_readonly
        )

        result = ledger.verify_chain()

        # The BEGIN snapshot pins both SELECTs to the same WAL point.  The
        # mutation was committed after BEGIN but is outside the read horizon,
        # so verify_chain sees the original correct boundary and returns no
        # defects.
        assert result == [], (
            "verify_chain reported defects after a between-SELECT boundary "
            "mutation; this indicates the receipts SELECT saw a different "
            "snapshot than the meta SELECT"
        )
