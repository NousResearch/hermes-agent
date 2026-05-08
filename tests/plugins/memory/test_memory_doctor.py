"""Tests for plugins.memory.holographic.doctor.check_memory_health.

The doctor is read-only and structured: every check returns
``{"name", "status", "detail"}`` and the report rolls up to a single
top-level status. These tests exercise:

  - healthy synthetic DB → status="ok", every check ok
  - tampered hrr_vector byte-length → vector_shape=error
  - encoding_version downgrade → encoding_version=warn
  - missing schema_version → schema_version=error
  - <5s budget on a synthetic 200-fact corpus
  - missing DB → db_exists=error (no exception)
"""

from __future__ import annotations

import os
import sqlite3
import tempfile

import pytest

from plugins.memory.holographic.doctor import check_memory_health
from plugins.memory.holographic.store import (
    MemoryStore,
    _CURRENT_ENCODING_VERSION,
    _CURRENT_SCHEMA_VERSION,
)


DIM = 2048


@pytest.fixture
def healthy_db():
    """A small healthy synthetic store. Yields the path."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    store = MemoryStore(db_path=path, hrr_dim=DIM, default_trust=0.85)
    try:
        store.add_fact("Walker Anderson is a person.", category="identity")
        store.add_fact("Apollo Energy Group runs LNG logistics.", category="general")
        store.add_fact("L-Charge needs generator maintenance.", category="general")
    finally:
        store.close()
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


def _by_name(report: dict, name: str) -> dict:
    for c in report["checks"]:
        if c["name"] == name:
            return c
    raise KeyError(name)


def test_healthy_db_reports_ok(healthy_db):
    report = check_memory_health(db_path=healthy_db, hrr_dim=DIM)
    assert report["status"] == "ok", report
    for c in report["checks"]:
        assert c["status"] == "ok", c


def test_missing_db_returns_structured_error():
    report = check_memory_health(db_path="/nonexistent/path/x.db", hrr_dim=DIM)
    assert report["status"] == "error"
    assert _by_name(report, "db_exists")["status"] == "error"


def test_wrong_byte_length_flags_vector_shape(healthy_db):
    """Corrupt one fact's hrr_vector byte-length; expect vector_shape=error."""
    conn = sqlite3.connect(healthy_db)
    conn.execute(
        "UPDATE facts SET hrr_vector = ? WHERE fact_id = "
        "(SELECT MIN(fact_id) FROM facts WHERE hrr_vector IS NOT NULL)",
        (b"\x00" * 16,),  # 16 bytes — wildly wrong
    )
    conn.commit()
    conn.close()

    report = check_memory_health(db_path=healthy_db, hrr_dim=DIM)
    shape = _by_name(report, "vector_shape")
    assert shape["status"] == "error", report
    assert shape["bad_count"] >= 1


def test_old_encoding_version_flags_warn(healthy_db):
    """Force a row's encoding_version below current; expect warn."""
    conn = sqlite3.connect(healthy_db)
    conn.execute(
        "UPDATE facts SET encoding_version = ? WHERE fact_id = "
        "(SELECT MIN(fact_id) FROM facts WHERE hrr_vector IS NOT NULL)",
        (max(0, _CURRENT_ENCODING_VERSION - 1),),
    )
    conn.commit()
    conn.close()

    report = check_memory_health(db_path=healthy_db, hrr_dim=DIM)
    enc = _by_name(report, "encoding_version")
    assert enc["status"] == "warn", report
    assert enc["stale"] >= 1


def test_missing_schema_version_flags_error(healthy_db):
    """Drop the schema_version row; expect schema_version=error."""
    conn = sqlite3.connect(healthy_db)
    conn.execute("DELETE FROM schema_version")
    conn.commit()
    conn.close()

    report = check_memory_health(db_path=healthy_db, hrr_dim=DIM)
    sv = _by_name(report, "schema_version")
    assert sv["status"] == "error", report


def test_smoke_probe_meets_budget_on_200_facts():
    """Synthetic 200-fact corpus must complete all checks well under 5s."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    store = MemoryStore(db_path=path, hrr_dim=DIM, default_trust=0.85)
    try:
        for i in range(200):
            store.add_fact(
                f"Fact number {i} mentions Walker Anderson and Apollo Energy.",
                category="general" if i % 2 else "identity",
            )
    finally:
        store.close()

    try:
        report = check_memory_health(db_path=path, hrr_dim=DIM,
                                     smoke_entity="Walker Anderson")
        assert report["elapsed_ms"] < 5000, (
            f"doctor took {report['elapsed_ms']}ms over 200 facts"
        )
        assert report["status"] in ("ok", "warn"), report
        smoke = _by_name(report, "smoke_probe")
        # Walker Anderson appears in every fact's content; even though entity
        # extraction (multi-word capitalized regex) catches it, signal must
        # exceed the noise floor of 0.10 — and must do so quickly.
        assert smoke["status"] == "ok", smoke
        assert smoke["best_sim"] > 0.10
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def test_schema_version_value_in_detail(healthy_db):
    report = check_memory_health(db_path=healthy_db, hrr_dim=DIM)
    sv = _by_name(report, "schema_version")
    assert f"v{_CURRENT_SCHEMA_VERSION}" in sv["detail"]


# ---------------------------------------------------------------------------
# trust_signal_writers (ADR-001 invariant)
# ---------------------------------------------------------------------------
#
# helpful_count is the ranking multiplier (_reinforced_trust at retrieval.py).
# ADR-001 closed the _reinforce_facts write path that used to inflate it on
# every probe. record_feedback is now the sole writer. The check below scans
# plugin source for SQL writes outside that allowed function.


def test_trust_signal_writers_passes_against_real_plugin():
    """The live plugin code must satisfy ADR-001."""
    from plugins.memory.holographic.doctor import _check_trust_signal_writers
    result = _check_trust_signal_writers()
    assert result["status"] == "ok", result
    assert "record_feedback" in result["detail"]


def test_trust_signal_writers_appears_in_full_report(healthy_db):
    """Wiring check: the new check shows up in check_memory_health output."""
    report = check_memory_health(db_path=healthy_db, hrr_dim=DIM)
    tsw = _by_name(report, "trust_signal_writers")
    assert tsw["status"] == "ok"


def test_trust_signal_writers_flags_synthetic_violator(tmp_path):
    """A non-record_feedback function writing helpful_count must error."""
    from plugins.memory.holographic.doctor import _check_trust_signal_writers

    (tmp_path / "store.py").write_text(
        'def record_feedback(fact_id, helpful):\n'
        '    sql = """\n'
        '    UPDATE facts\n'
        '    SET trust_score = ?,\n'
        '        helpful_count = helpful_count + 1\n'
        '    WHERE fact_id = ?\n'
        '    """\n'
    )
    (tmp_path / "evil.py").write_text(
        'def sneaky_writer(fact_ids):\n'
        '    sql = "UPDATE facts SET helpful_count = helpful_count + 1"\n'
    )

    result = _check_trust_signal_writers(plugin_dir=tmp_path)
    assert result["status"] == "error"
    assert len(result["violations"]) == 1
    v = result["violations"][0]
    assert v["file"] == "evil.py"
    assert v["function"] == "sneaky_writer"


def test_trust_signal_writers_ignores_select_and_schema(tmp_path):
    """SELECTs of helpful_count and CREATE TABLE column declarations are reads,
    not writes — they must not be flagged."""
    from plugins.memory.holographic.doctor import _check_trust_signal_writers

    (tmp_path / "reader.py").write_text(
        'def list_facts():\n'
        '    sql = """\n'
        '    CREATE TABLE facts (helpful_count INTEGER DEFAULT 0);\n'
        '    SELECT fact_id, trust_score, helpful_count FROM facts;\n'
        '    """\n'
    )

    result = _check_trust_signal_writers(plugin_dir=tmp_path)
    assert result["status"] == "ok", result


def test_trust_signal_writers_handles_unparseable_source(tmp_path):
    """Syntax error in a plugin file must surface as error, not crash."""
    from plugins.memory.holographic.doctor import _check_trust_signal_writers

    (tmp_path / "broken.py").write_text("def oops(:\n  pass\n")
    result = _check_trust_signal_writers(plugin_dir=tmp_path)
    assert result["status"] == "error"
    assert "broken.py" in result["detail"]
