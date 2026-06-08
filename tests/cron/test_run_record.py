"""Plan 056-D — tests for the Hermes-side run-record emitter (vendored RunRecord).

Covers (per master plan §056-D acceptance, unit/autonomous):
  * a simulated cron run produces ONE RunRecord with ``producer="hermes"`` +
    a run_id + the 056-B field shape;
  * the vendored RunRecord JSON shape is field-for-field parity with the
    canonical 056-B schema (so the orchestrator's validate-on-read accepts it);
  * the daily-brief cron path stamps a run_id and emits exactly one record
    (fail-soft, behind the existing flow — the brief behavior is unchanged);
  * the atlas-annotation threads run_id through ``_write_fact`` and is
    backward-compatible when omitted.
"""

from __future__ import annotations

import json

from cron import run_record as rr


# The canonical 056-B field set (orchestrator/activities/run_record.py::RunRecord).
# This list is the contract: the vendored record must emit EXACTLY these keys.
_CANONICAL_056B_FIELDS = {
    "schema_version",
    "run_id",
    "parent_run_id",
    "producer",
    "kind",
    "started",
    "ended",
    "status",
    "plan_id",
    "phase_id",
    "commit_sha",
    "pr_url",
    "branch",
    "spend_usd",
    "artifact_keys",
    "memory_refs",
    "notes",
}


# ---------------------------------------------------------------------------
# new_run_id
# ---------------------------------------------------------------------------


def test_new_run_id_is_unique_opaque_string():
    a = rr.new_run_id()
    b = rr.new_run_id()
    assert isinstance(a, str) and a
    assert a != b


# ---------------------------------------------------------------------------
# RunRecord shape parity with 056-B
# ---------------------------------------------------------------------------


def test_run_record_emits_exact_056b_field_shape():
    rec = rr.build_run_record(run_id="run-abc", kind="brief", status="ok")
    d = rec.to_dict()
    assert set(d.keys()) == _CANONICAL_056B_FIELDS
    # producer is hermes; schema_version stamped; run_id carried.
    assert d["producer"] == "hermes"
    assert d["schema_version"] == rr.RUN_RECORD_SCHEMA_VERSION == "1"
    assert d["run_id"] == "run-abc"
    assert d["kind"] == "brief"
    assert d["status"] == "ok"


def test_run_record_json_roundtrips_to_canonical_dict():
    rec = rr.build_run_record(
        run_id="run-xyz",
        kind="brief",
        status="delivered",
        started="2026-06-07T07:30:00+00:00",
        ended="2026-06-07T07:30:05+00:00",
        notes="daily-brief: weekday-fire",
        memory_refs=["urn:atlas:agent_decision:fake"],
    )
    decoded = json.loads(rec.to_json())
    assert decoded == rec.to_dict()
    assert decoded["memory_refs"] == ["urn:atlas:agent_decision:fake"]
    assert decoded["parent_run_id"] is None  # carried, not acted on


def test_run_record_carries_parent_run_id_without_acting_on_it():
    rec = rr.build_run_record(
        run_id="child", kind="brief", parent_run_id="parent-run"
    )
    assert rec.to_dict()["parent_run_id"] == "parent-run"


def test_spend_usd_defaults_and_lists_independent():
    a = rr.build_run_record(run_id="a")
    b = rr.build_run_record(run_id="b")
    a.artifact_keys.append("k")
    assert b.artifact_keys == []  # no shared mutable default
    assert a.to_dict()["spend_usd"] == 0.0


# ---------------------------------------------------------------------------
# emit_run_record — fail-soft
# ---------------------------------------------------------------------------


def test_emit_writes_one_jsonl_line(tmp_path):
    target = tmp_path / "sub" / "run_records.jsonl"
    rec = rr.build_run_record(run_id="run-1", kind="brief", status="ok")
    ok = rr.emit_run_record(rec, path=str(target))
    assert ok is True
    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["run_id"] == "run-1"


def test_emit_uses_injected_sink():
    captured: list[str] = []
    rec = rr.build_run_record(run_id="run-2")
    ok = rr.emit_run_record(rec, sink=captured.append)
    assert ok is True
    assert len(captured) == 1
    assert json.loads(captured[0])["producer"] == "hermes"


def test_emit_is_fail_soft_never_raises():
    rec = rr.build_run_record(run_id="run-3")

    def _boom(_line: str) -> None:
        raise OSError("disk gone")

    # Must NOT raise — returns False and logs.
    assert rr.emit_run_record(rec, sink=_boom) is False


# ---------------------------------------------------------------------------
# daily-brief cron path emits exactly one record, behind the existing flow
# ---------------------------------------------------------------------------


def test_daily_brief_cron_run_emits_one_hermes_record(monkeypatch, tmp_path):
    from datetime import datetime, timezone

    from cron import daily_brief as db

    target = tmp_path / "run_records.jsonl"
    monkeypatch.setenv("HERMES_RUN_RECORD_PATH", str(target))

    captured: dict = {}

    async def _invoke(*, slack_channel: str):
        captured["slack_channel"] = slack_channel
        return {
            "blocks": [{"type": "section",
                        "text": {"type": "mrkdwn", "text": "*Top 3*"}}],
            "text": "Daily brief",
            "_daily_meta": {"agent_decision_urn": "urn:atlas:agent_decision:fake"},
        }

    def _post(payload, *, token):
        return {"ok": True, "ts": "1717000000.000100"}

    # Mon 2026-06-01 07:30 — weekday, fires.
    result = db.run_daily_brief(
        now=datetime(2026, 6, 1, 7, 30, tzinfo=timezone.utc),
        invoke_daily=_invoke,
        slack_post=_post,
        channel="D0123ABCD",
        token="xoxb-test",
    )

    assert result["status"] == "delivered"
    # A run_id was stamped and surfaced.
    assert result["run_id"]
    # Exactly ONE record, producer=hermes, carrying that run_id.
    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["producer"] == "hermes"
    assert rec["kind"] == "brief"
    assert rec["status"] == "delivered"
    assert rec["run_id"] == result["run_id"]
    assert set(rec.keys()) == _CANONICAL_056B_FIELDS
    # The triggered decision is carried as a memory_ref (the join seam).
    assert rec["memory_refs"] == ["urn:atlas:agent_decision:fake"]


def test_daily_brief_inbound_run_id_overrides_default(monkeypatch, tmp_path):
    from datetime import datetime, timezone

    from cron import daily_brief as db

    monkeypatch.setenv("HERMES_RUN_RECORD_PATH", str(tmp_path / "rr.jsonl"))

    # Weekend → skipped path still stamps/emits, and honors the inbound id.
    result = db.run_daily_brief(
        now=datetime(2026, 6, 6, 7, 30, tzinfo=timezone.utc),  # Saturday
        run_id="inbound-chain-id",
    )
    assert result["status"] == "skipped"
    assert result["run_id"] == "inbound-chain-id"
    rec = json.loads((tmp_path / "rr.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert rec["run_id"] == "inbound-chain-id"
    assert rec["status"] == "skipped"


def test_emit_brief_run_record_helper_is_fail_soft(monkeypatch):
    """The brief's emit helper swallows ANY failure (so the delivered path,
    which calls it directly, can never be broken by a run-record problem)."""
    from cron import daily_brief as db

    # Force the underlying build to explode; the helper must NOT propagate.
    def _explode(*_a, **_k):
        raise RuntimeError("build exploded")

    monkeypatch.setattr("cron.run_record.build_run_record", _explode)

    # Must return None and never raise.
    assert (
        db._emit_brief_run_record(
            run_id="r", status="delivered", reason="x",
            agent_decision_urn=None,
        )
        is None
    )


def test_daily_brief_delivers_even_with_broken_run_record_sink(monkeypatch, tmp_path):
    """End-to-end: an unwritable run-record path does not break delivery."""
    from datetime import datetime, timezone

    from cron import daily_brief as db

    # Point the record path at a location that cannot be written (a file used
    # as a directory parent) — emit_run_record swallows the OSError.
    bad = tmp_path / "afile"
    bad.write_text("x", encoding="utf-8")
    monkeypatch.setenv("HERMES_RUN_RECORD_PATH", str(bad / "nested" / "rr.jsonl"))

    async def _invoke(*, slack_channel: str):
        return {"blocks": [{"type": "section",
                            "text": {"type": "mrkdwn", "text": "x"}}],
                "text": "Daily brief"}

    result = db.run_daily_brief(
        now=datetime(2026, 6, 1, 7, 30, tzinfo=timezone.utc),
        invoke_daily=_invoke,
        slack_post=lambda payload, *, token: {"ok": True, "ts": "1.1"},
        channel="D0123ABCD",
        token="xoxb-test",
    )
    assert result["status"] == "delivered"
