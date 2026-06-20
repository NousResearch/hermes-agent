"""Unit + integration tests for the deterministic delivery override.

A cron job may declare ``deliver_file`` (and optional ``deliver_markers``).
When it does, a SUCCESSFUL run delivers that file's contents instead of the
model's free final answer — but ONLY when the file is fresh (written during the
run), non-empty, and marker-complete. Every guard falls back to the model's
final answer, so the override can only make delivery more reliable, never break
an otherwise-working job. Jobs without ``deliver_file`` are unaffected.
"""
import cron.scheduler as s


# --- helpers ---------------------------------------------------------------

def test_job_deliver_markers_normalizes_forms():
    assert s._job_deliver_markers({"deliver_markers": ["A", "B"]}) == ["A", "B"]
    assert s._job_deliver_markers({"deliver_markers": "A"}) == ["A"]
    assert s._job_deliver_markers({}) == []
    assert s._job_deliver_markers({"deliver_markers": None}) == []
    # blank/whitespace markers are dropped
    assert s._job_deliver_markers({"deliver_markers": ["A", "", "  "]}) == ["A"]


def test_missing_markers():
    assert s._missing_markers("has A and B", ["A", "B"]) == []
    assert s._missing_markers("has A only", ["A", "B"]) == ["B"]
    assert s._missing_markers("", ["A"]) == ["A"]
    assert s._missing_markers(None, ["A"]) == ["A"]


# --- apply_canonical_delivery_override -------------------------------------

def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_no_deliver_file_returns_model_answer():
    out = s.apply_canonical_delivery_override({"id": "j"}, "model answer", 0.0)
    assert out == "model answer"


def test_fresh_complete_file_overrides_model_answer(tmp_path):
    f = _write(tmp_path / "report.txt", "CANON header\nbody\nTOTAL: 3")
    job = {"id": "j", "deliver_file": f, "deliver_markers": ["CANON", "TOTAL"]}
    # run started before the file write → fresh
    out = s.apply_canonical_delivery_override(job, "truncated recap", run_started_at=0.0)
    assert out == "CANON header\nbody\nTOTAL: 3"


def test_file_matching_model_answer_delivers_file_bytes(tmp_path):
    body = "CANON\nTOTAL: 1"
    f = _write(tmp_path / "report.txt", body)
    job = {"id": "j", "deliver_file": f, "deliver_markers": ["CANON"]}
    out = s.apply_canonical_delivery_override(job, body, run_started_at=0.0)
    assert out == body


def test_stale_file_is_not_delivered(tmp_path, monkeypatch):
    f = _write(tmp_path / "report.txt", "CANON\nTOTAL: 1")
    job = {"id": "j", "deliver_file": f, "deliver_markers": ["CANON"]}
    # run started far AFTER the file's mtime → stale → fall back to model answer
    future_start = (tmp_path / "report.txt").stat().st_mtime + 3600.0
    out = s.apply_canonical_delivery_override(job, "model answer", run_started_at=future_start)
    assert out == "model answer"


def test_incomplete_file_missing_markers_is_not_delivered(tmp_path):
    f = _write(tmp_path / "report.txt", "CANON only, truncated")
    job = {"id": "j", "deliver_file": f, "deliver_markers": ["CANON", "TOTAL"]}
    out = s.apply_canonical_delivery_override(job, "model answer", run_started_at=0.0)
    assert out == "model answer"


def test_absent_file_returns_model_answer(tmp_path):
    job = {"id": "j", "deliver_file": str(tmp_path / "nope.txt")}
    out = s.apply_canonical_delivery_override(job, "model answer", run_started_at=0.0)
    assert out == "model answer"


def test_empty_file_returns_model_answer(tmp_path):
    f = _write(tmp_path / "report.txt", "   \n  ")
    job = {"id": "j", "deliver_file": f}
    out = s.apply_canonical_delivery_override(job, "model answer", run_started_at=0.0)
    assert out == "model answer"


def test_none_run_started_at_skips_freshness_check(tmp_path):
    f = _write(tmp_path / "report.txt", "CANON\nTOTAL: 1")
    job = {"id": "j", "deliver_file": f, "deliver_markers": ["CANON"]}
    out = s.apply_canonical_delivery_override(job, "model answer", run_started_at=None)
    assert out == "CANON\nTOTAL: 1"


def test_override_never_raises_on_bad_input():
    # A non-path deliver_file value must not crash — it falls back gracefully.
    job = {"id": "j", "deliver_file": object()}
    out = s.apply_canonical_delivery_override(job, "model answer", run_started_at=0.0)
    assert out == "model answer"


# --- integration through run_one_job ---------------------------------------

def test_run_one_job_delivers_canonical_file(tmp_path, monkeypatch):
    """A successful job with a fresh deliver_file delivers the FILE body, not
    the model's (truncated) final answer."""
    f = _write(tmp_path / "report.txt", "CANON report\nTOTAL: 42")
    delivered = {}

    monkeypatch.setattr(s, "run_job", lambda job: (True, "out", "truncated recap", None))
    monkeypatch.setattr(s, "save_job_output", lambda jid, out: f"/tmp/{jid}.txt")
    monkeypatch.setattr(s, "mark_job_run", lambda *a, **k: None)

    def fake_deliver(job, content, adapters=None, loop=None):
        delivered["content"] = content
        return None

    monkeypatch.setattr(s, "_deliver_result", fake_deliver)

    job = {"id": "j", "name": "t", "deliver_file": f, "deliver_markers": ["CANON", "TOTAL"]}
    ok = s.run_one_job(job)

    assert ok is True
    assert delivered["content"] == "CANON report\nTOTAL: 42"


def test_run_one_job_without_deliver_file_delivers_model_answer(monkeypatch):
    """A job with no deliver_file is unaffected — delivers the model answer."""
    delivered = {}
    monkeypatch.setattr(s, "run_job", lambda job: (True, "out", "model answer", None))
    monkeypatch.setattr(s, "save_job_output", lambda jid, out: f"/tmp/{jid}.txt")
    monkeypatch.setattr(s, "mark_job_run", lambda *a, **k: None)
    monkeypatch.setattr(
        s, "_deliver_result",
        lambda job, content, adapters=None, loop=None: delivered.update(content=content),
    )

    s.run_one_job({"id": "j", "name": "t"})
    assert delivered["content"] == "model answer"
