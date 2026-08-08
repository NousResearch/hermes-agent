"""Tests for the [FAILURE: reason] content-level failure marker.

Covers the parser contract (_parse_cron_failure_marker) and the
run_one_job integration: status flip, escalation delivery, and
escalation-failure resilience.
"""
import cron.scheduler as s


# ---------------------------------------------------------------------------
# Parser contract
# ---------------------------------------------------------------------------

def test_parse_failure_marker_with_reason():
    parse = s._parse_cron_failure_marker
    assert parse("[FAILURE: skill not found]") == "skill not found"


def test_parse_failure_marker_without_reason():
    parse = s._parse_cron_failure_marker
    assert parse("[FAILURE]") == "agent reported task failure"


def test_parse_failure_marker_case_insensitive():
    parse = s._parse_cron_failure_marker
    assert parse("[failure: timeout]") == "timeout"
    assert parse("[Failure: API error]") == "API error"


def test_parse_failure_marker_first_line_only():
    parse = s._parse_cron_failure_marker
    assert parse("[FAILURE: oops]\nSome detail follows") == "oops"


def test_parse_failure_marker_rejects_mid_sentence():
    """A [FAILURE] mention mid-sentence must not trigger — only prefix."""
    parse = s._parse_cron_failure_marker
    assert parse("The task said [FAILURE: x] in its log") is None


def test_parse_failure_marker_rejects_normal_response():
    parse = s._parse_cron_failure_marker
    assert parse("Everything went fine, report attached.") is None


def test_parse_failure_marker_rejects_empty_and_none():
    parse = s._parse_cron_failure_marker
    assert parse("") is None
    assert parse("   ") is None
    assert parse(None) is None


def test_parse_failure_marker_rejects_silent():
    """[SILENT] must not be confused with [FAILURE]."""
    parse = s._parse_cron_failure_marker
    assert parse("[SILENT]") is None


# ---------------------------------------------------------------------------
# Pipeline integration (run_one_job)
# ---------------------------------------------------------------------------

def _patch_pipeline(monkeypatch, *, success=True, output="out", final="final response",
                    error=None, config=None):
    """Patch the job pipeline primitives and record calls.

    ``config`` is the return value of ``load_config()`` — set it to
    ``{"cron": {"failure_escalation_channel": "..."}}`` to enable escalation.
    """
    calls = []

    def fake_run_job(job, *, defer_agent_teardown=None):
        calls.append(("run_job", job["id"]))
        return (success, output, final, error)

    def fake_save(jid, out):
        calls.append(("save", jid))
        return f"/tmp/{jid}.txt"

    def fake_deliver(job, content, adapters=None, loop=None):
        calls.append(("deliver", job.get("deliver", "default"), content))
        return None

    def fake_mark(jid, ok, err=None, delivery_error=None):
        calls.append(("mark", jid, ok, err))

    monkeypatch.setattr(s, "run_job", fake_run_job)
    monkeypatch.setattr(s, "save_job_output", fake_save)
    monkeypatch.setattr(s, "_deliver_result", fake_deliver)
    monkeypatch.setattr(s, "mark_job_run", fake_mark)
    monkeypatch.setattr(s, "load_config", lambda: config)
    return calls


def test_failure_marker_flips_status_to_failed(monkeypatch):
    """[FAILURE: reason] must flip success→False and record the reason."""
    calls = _patch_pipeline(monkeypatch, final="[FAILURE: skill not found]\nDetails here")

    ok = s.run_one_job({"id": "f1", "name": "t"})

    assert ok is True  # run_one_job returns True (job processed, not crashed)
    mark = [c for c in calls if c[0] == "mark"][0]
    assert mark[2] is False  # success=False
    assert "skill not found" in mark[3]  # error contains reason


def test_failure_marker_escalation_delivery(monkeypatch):
    """When failure_escalation_channel is configured, the response is also
    delivered to the escalation channel."""
    calls = _patch_pipeline(
        monkeypatch,
        final="[FAILURE: tool error]\nCould not run",
        config={"cron": {"failure_escalation_channel": "discord:escalation-room"}},
    )

    s.run_one_job({"id": "f2", "name": "t"})

    delivers = [c for c in calls if c[0] == "deliver"]
    # Two deliveries: escalation channel + main channel
    assert len(delivers) == 2
    esc = [d for d in delivers if d[1] == "discord:escalation-room"]
    assert len(esc) == 1
    assert "[FAILURE: tool error]" in esc[0][2]


def test_failure_marker_no_escalation_without_config(monkeypatch):
    """Without failure_escalation_channel, only the main delivery fires."""
    calls = _patch_pipeline(monkeypatch, final="[FAILURE: oops]", config={})

    s.run_one_job({"id": "f3", "name": "t"})

    delivers = [c for c in calls if c[0] == "deliver"]
    assert len(delivers) == 1  # main delivery only


def test_failure_marker_escalation_error_does_not_crash(monkeypatch):
    """If escalation delivery raises, the main pipeline still completes."""
    calls = []

    def fake_run_job(job, *, defer_agent_teardown=None):
        calls.append(("run_job",))
        return (True, "out", "[FAILURE: boom]", None)

    deliver_count = {"n": 0}

    def fake_deliver(job, content, adapters=None, loop=None):
        deliver_count["n"] += 1
        if job.get("deliver") == "discord:esc":
            raise RuntimeError("escalation channel down")
        calls.append(("deliver-main",))
        return None

    monkeypatch.setattr(s, "run_job", fake_run_job)
    monkeypatch.setattr(s, "save_job_output", lambda jid, out: f"/tmp/{jid}.txt")
    monkeypatch.setattr(s, "_deliver_result", fake_deliver)
    monkeypatch.setattr(s, "mark_job_run", lambda *a, **k: calls.append(("mark",)))
    monkeypatch.setattr(s, "load_config",
                        lambda: {"cron": {"failure_escalation_channel": "discord:esc"}})

    ok = s.run_one_job({"id": "f4", "name": "t"})

    assert ok is True
    assert ("deliver-main",) in calls
    assert ("mark",) in calls
    assert deliver_count["n"] == 2  # escalation attempted + main delivered


def test_normal_response_not_affected_by_failure_marker(monkeypatch):
    """A normal (non-failure) response must not trigger failure logic."""
    calls = _patch_pipeline(monkeypatch, final="All tasks completed successfully.")

    s.run_one_job({"id": "f5", "name": "t"})

    mark = [c for c in calls if c[0] == "mark"][0]
    assert mark[2] is True  # success=True
    assert mark[3] is None  # no error
