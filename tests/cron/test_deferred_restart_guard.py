"""G3 (gateway-deferred-restart spec): `hermes gateway defer-restart` is forbidden to cron.

Thin, cron-scoped shape (PR slim decision, 2026-10-18), all funneling through
`check_deferred_restart_job` so the layers cannot drift:

  L1  definition paths — create / edit / import / re-enable
      (`cron/jobs.py`, `cron/job_definition.py`),
  L2  fail-closed dispatch preflight at the REAL dispatch entry
      (`cron/scheduler.py` `_run_one_job_body`), INCLUDING pre-existing jobs,
  LB  the CLI entrypoint's trusted-cron-provenance refusal (tests live in
      `tests/hermes_cli/test_gateway_defer_restart.py`) — the backstop for what
      lexical layers cannot see (runtime assembly like `hermes gateway "$left-$right"`).

Mention-level strictness (gate 2, G2-R1/R2): NO inert-heredoc masking for the deferred
detector, and carriers (prompt / monitor source / script) are scanned independently
through one detector, then concatenated so cross-carrier splits are caught too.

The last section pins the 15 fact-finding G1 probe verdicts byte-for-byte: adding the
deferred channel must not shift a single existing guard decision.
"""

from __future__ import annotations

import pytest

from cron import jobs
from cron.job_definition import merge_job_definition
from cron.lifecycle_guard import (
    DeferredRestartBlocked,
    check_deferred_restart_job,
    check_deferred_restart_payload,
    contains_deferred_restart_command,
    contains_gateway_lifecycle_command,
    scan_deferred_restart,
)

# --- detector: what counts as "referencing the entrypoint" -----------------------

DEFERRED_REFERENCES = [
    "hermes gateway defer-restart",
    "hermes gateway defer-restart --max-wait 5",
    "hermes -p work gateway defer-restart",
    "hermes --profile=work gateway defer-restart",
    'hermes gateway defer-"restart"',  # quote splice — closed by the token pass
    "hermes gateway \\\n  defer-restart",  # line continuation between tokens
    'subprocess.run(["hermes", "gateway", "defer-restart"])',  # argv-list punctuation pass
    "python -m hermes_cli.main gateway defer-restart",
    "sleep 30 && hermes gateway defer-restart",
    # Mention-level on purpose (spec wording: payloads *referencing* the entrypoint),
    # which is stricter than G1's data-argument exemption — fail-closed in a payload.
    "echo 'hermes gateway defer-restart' >> notes.md",
    "bash <<'EOF'\nhermes gateway defer-restart\nEOF",  # executed heredoc body
    # G2-R1 counterexample B: a data heredoc that GENERATES a script which is then
    # executed. No masking for this detector — the literal mention is refused.
    "cat > child.sh <<'EOF'\nhermes gateway defer-restart\nEOF\nbash child.sh",
    "cat <<'EOF'\ngateway defer-restart\nEOF",
]

NOT_DEFERRED_REFERENCES = [
    "",
    "hermes gateway restart",
    "hermes gateway restart --deferred 30",
    "deploy gateway; restart services",
]


@pytest.mark.parametrize("text", DEFERRED_REFERENCES)
def test_detector_flags_deferred_restart_references(text):
    assert contains_deferred_restart_command(text) is True


@pytest.mark.parametrize("text", NOT_DEFERRED_REFERENCES)
def test_detector_ignores_other_shapes(text):
    assert contains_deferred_restart_command(text) is False


def test_data_heredoc_masking_still_applies_to_g1_only():
    """One rule set, one口径: G1 keeps its data-argument exemption, G3 refuses mentions."""
    assert contains_gateway_lifecycle_command("cat <<'EOF'\ngateway defer-restart\nEOF") is False
    assert contains_deferred_restart_command("cat <<'EOF'\ngateway defer-restart\nEOF") is True


def test_referenced_script_walk_catches_nested_deferred_restart(tmp_path):
    """The walk is shared with G1 — a nested script's reference is found, not just the root."""
    inner = tmp_path / "inner.sh"
    inner.write_text("#!/bin/sh\nhermes gateway defer-restart\n")
    outer = tmp_path / "outer.sh"
    outer.write_text(f"#!/bin/sh\nbash {inner}\n")
    unsafe, refusal = scan_deferred_restart(f"bash {outer}", cwd=str(tmp_path))
    assert unsafe is True
    assert refusal is None


# --- payload checks (the L1/L2 chokepoint) -------------------------------------

def test_payload_prompt_reference_is_rejected():
    with pytest.raises(DeferredRestartBlocked):
        check_deferred_restart_payload("do the thing: hermes gateway defer-restart")


def test_payload_script_reference_is_rejected(tmp_path):
    script = tmp_path / "job.sh"
    script.write_text("#!/bin/sh\nhermes gateway defer-restart\n")
    with pytest.raises(DeferredRestartBlocked):
        check_deferred_restart_payload("run it", str(script))


def test_payload_split_across_prompt_and_script_is_rejected(tmp_path):
    """The combined pass catches a command split across carriers."""
    script = tmp_path / "job.sh"
    script.write_text("#!/bin/sh\ngateway defer-restart\n")
    with pytest.raises(DeferredRestartBlocked):
        check_deferred_restart_payload("exec hermes", str(script))


def test_cross_field_heredoc_cannot_mask_the_script_carrier(tmp_path):
    """G2-R2 counterexample chain: a prompt ending in an unclosed data heredoc must not
    swallow the script body when carriers are merged. Per-carrier scanning closes it."""
    script = tmp_path / "job.sh"
    script.write_text("#!/bin/sh\nhermes gateway defer-restart\ntrue\n")
    with pytest.raises(DeferredRestartBlocked):
        check_deferred_restart_payload("dump this:\ncat <<'true'", str(script))


def test_each_carrier_is_scanned_independently():
    """A shell-syntax trick in one carrier never hides another carrier's mention."""
    with pytest.raises(DeferredRestartBlocked):
        check_deferred_restart_job({
            "prompt": "intro <<'X'",
            "script": None,
            "monitor_script": "hermes gateway defer-restart",
        })
    with pytest.raises(DeferredRestartBlocked):
        check_deferred_restart_job({
            "prompt": "hermes gateway defer-restart",
            "script": None,
            "monitor_script": "exit 0\n<<'X'",
        })


def test_clean_payload_passes(tmp_path):
    script = tmp_path / "job.sh"
    script.write_text("#!/bin/sh\necho hello\n")
    check_deferred_restart_payload("greet the world", str(script))


def test_unscannable_script_fails_closed_with_named_reason(monkeypatch):
    """A script the guard cannot scan refuses with the NAMED reason (fail-closed).
    (Missing/unreadable paths intentionally stay empty so scheduler validation reports
    them — parity with check_gateway_lifecycle; the refusal route is the unscannable one.)"""
    from cron import lifecycle_guard

    monkeypatch.setattr(
        lifecycle_guard, "_read_script_for_scanning",
        lambda p: ("", "unit-test refusal"),
    )
    with pytest.raises(DeferredRestartBlocked) as exc:
        check_deferred_restart_payload("run it", "/tmp/anything.sh")
    assert "unit-test refusal" in str(exc.value)


def test_scan_budget_exhaustion_fails_closed(monkeypatch):
    from cron import lifecycle_guard

    monkeypatch.setattr(lifecycle_guard, "_MAX_LIFECYCLE_SCAN_LINES", 2)
    with pytest.raises(DeferredRestartBlocked):
        check_deferred_restart_payload("line1\nline2\nline3\nline4")


def test_job_chokepoint_covers_monitor_source():
    """Monitor source runs FIRST each tick — it is part of the executable payload."""
    with pytest.raises(DeferredRestartBlocked):
        check_deferred_restart_job({
            "prompt": "clean",
            "script": None,
            "monitor_script": "hermes gateway defer-restart",
        })
    check_deferred_restart_job({"prompt": "clean", "script": None})


# --- L1: definition paths -------------------------------------------------------

def test_create_job_rejects_deferred_restart_prompt(tmp_path):
    with jobs.use_cron_store(tmp_path):
        with pytest.raises(DeferredRestartBlocked):
            jobs.create_job(
                prompt="run hermes gateway defer-restart",
                schedule="every 5m", deliver="local",
            )


def test_create_job_accepts_clean_prompt(tmp_path):
    with jobs.use_cron_store(tmp_path):
        job = jobs.create_job(prompt="say hi", schedule="every 5m", deliver="local")
        assert job["prompt"] == "say hi"


def test_update_job_edit_rejects_deferred_restart_prompt(tmp_path):
    with jobs.use_cron_store(tmp_path):
        job = jobs.create_job(prompt="say hi", schedule="every 5m", deliver="local")
        with pytest.raises(DeferredRestartBlocked):
            jobs.update_job(job["id"], {"prompt": "now hermes gateway defer-restart"})


def test_re_enable_rejects_legacy_deferred_restart_payload():
    """Re-activating a paused legacy record re-checks the merged payload."""
    old = {"prompt": "bad hermes gateway defer-restart", "enabled": False}
    merged = {**old, "enabled": True}
    with pytest.raises(DeferredRestartBlocked):
        jobs._reject_deferred_restart_payload(old, merged, {"enabled": True, "state": "scheduled"})


def test_unrelated_bookkeeping_update_of_legacy_record_still_loads():
    """Scoped to payload edits/re-enables — legacy records must keep loading (update layer).

    The dispatch preflight (L2) is what re-checks such records fail-closed before execution.
    """
    old = {"prompt": "bad hermes gateway defer-restart", "enabled": True}
    merged = {**old, "name": "renamed"}
    jobs._reject_deferred_restart_payload(old, merged, {"name": "renamed"})  # no raise
    with pytest.raises(DeferredRestartBlocked):
        check_deferred_restart_job(merged)  # ...but L2 refuses it at dispatch


def test_import_path_rejects_deferred_restart_record():
    with pytest.raises(DeferredRestartBlocked):
        merge_job_definition({}, {
            "id": "imported",
            "prompt": "run hermes gateway defer-restart",
            "schedule": "every 5m",
        })


# --- L2 at the REAL dispatch entry ----------------------------------------------
# Gate 2 (G2-R2) demanded real-entry proof: a refused job entering `_run_one_job_body`
# must reach ZERO user-code/model side effects and book the execution failed with the
# named reason. `run_job` is THE execution entry for agent turns, user scripts and
# monitor scripts alike — a counter there is the side-effect sentinel.

def _dispatch(job_dict, monkeypatch):
    """Run the real dispatch body with side-effect sentinels. Returns (result, seen)."""
    from cron import executions, scheduler

    seen = {"run_job": 0, "finish": None}

    def _forbidden_run(job, **kwargs):  # pragma: no cover - must never run
        seen["run_job"] += 1
        return True, "", "", None

    def _finish(execution_id, *, success, error=None, **kwargs):
        seen["finish"] = {"execution_id": execution_id, "success": success, "error": error}

    monkeypatch.setattr(scheduler, "run_job", _forbidden_run)
    monkeypatch.setattr(scheduler, "claim_dispatch", lambda job_id: True)
    monkeypatch.setattr(scheduler, "finish_execution", _finish)
    # `_start_owned_run` CASes the execution row (claimed → running); feed it a
    # REAL row instead of stubbing the ownership check away.
    fire = executions.create_execution(job_dict["id"], source="test")
    result = scheduler._run_one_job_body(dict(job_dict, execution_id=fire["id"]))
    return result, seen


def _assert_refused_before_any_side_effect(result, seen):
    assert result is True  # dispatch handled the fire: booked failed, never re-queued
    assert seen["run_job"] == 0  # zero model launches / user-script or monitor runs
    assert seen["finish"] is not None
    assert seen["finish"]["success"] is False
    assert "Pre-execution guard refused this job" in seen["finish"]["error"]
    assert "hermes gateway defer-restart" in seen["finish"]["error"]


def test_real_dispatch_refuses_agent_shape(tmp_path, monkeypatch):
    with jobs.use_cron_store(tmp_path / "cron"):
        result, seen = _dispatch(
            {"id": "legacy-a", "prompt": "run hermes gateway defer-restart",
             "enabled": True, "schedule": "every 1h"},
            monkeypatch)
    _assert_refused_before_any_side_effect(result, seen)


def test_real_dispatch_refuses_script_shape(tmp_path, monkeypatch):
    script = tmp_path / "job.sh"
    script.write_text("#!/bin/sh\nhermes gateway defer-restart\n")
    with jobs.use_cron_store(tmp_path / "cron"):
        result, seen = _dispatch(
            {"id": "legacy-s", "prompt": "clean", "script": str(script),
             "enabled": True, "schedule": "every 1h"},
            monkeypatch)
    _assert_refused_before_any_side_effect(result, seen)


def test_real_dispatch_refuses_monitor_shape(tmp_path, monkeypatch):
    with jobs.use_cron_store(tmp_path / "cron"):
        result, seen = _dispatch(
            {"id": "legacy-m", "prompt": "clean", "monitor_script": "hermes gateway defer-restart",
             "enabled": True, "schedule": "every 1h"},
            monkeypatch)
    _assert_refused_before_any_side_effect(result, seen)


def test_real_dispatch_refuses_heredoc_write_then_run(tmp_path, monkeypatch):
    """G2-R1 counterexample B through the real entry: the payload GENERATES its command
    via a data heredoc and executes it. Mention-level detection refuses it at dispatch —
    before the generator ever runs."""
    script = tmp_path / "job.sh"
    script.write_text(
        "#!/bin/sh\ncat > child.sh <<'EOF'\nhermes gateway defer-restart\nEOF\nbash child.sh\n")
    with jobs.use_cron_store(tmp_path / "cron"):
        result, seen = _dispatch(
            {"id": "legacy-h", "prompt": "clean", "script": str(script),
             "enabled": True, "schedule": "every 1h"},
            monkeypatch)
    _assert_refused_before_any_side_effect(result, seen)


def test_script_resolution_follows_the_active_home(tmp_path, monkeypatch):
    """G2-R2 path binding: the referenced script resolves under the ACTIVE home binding
    (the routed fire's profile home), never a caller-default home. Same-named script in
    another home must be invisible to the scan."""
    from cron import lifecycle_guard

    active_home = tmp_path / "active-home"
    (active_home / "scripts").mkdir(parents=True)
    (active_home / "scripts" / "job.sh").write_text("#!/bin/sh\necho safe\n")
    other_home = tmp_path / "other-home"
    (other_home / "scripts").mkdir(parents=True)
    (other_home / "scripts" / "job.sh").write_text("#!/bin/sh\nhermes gateway defer-restart\n")

    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: active_home)
    check_deferred_restart_job({"prompt": "clean", "script": "job.sh"})  # scans ACTIVE home

    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: other_home)
    with pytest.raises(DeferredRestartBlocked):
        check_deferred_restart_job({"prompt": "clean", "script": "job.sh"})  # active = other → refused


# --- G1 regression: the 15 fact-finding probes, verdicts byte-identical ----------
# Source of truth: the pre-change probe run (probe_guard.py, 2026-10-17) — these are the
# exact expectations recorded BEFORE the deferred channel existed.

G1_PROBES = [
    ("hermes gateway restart", True),
    ("hermes gateway restart --deferred 30", True),
    ('hermes gateway re"start"', True),
    ("hermes gateway stop", True),
    ("hermes gateway defer-restart", False),
    ("hermes gateway defer-restart --max-wait 5", False),
    ("hermes -p work gateway defer-restart", False),
    ("hermes --profile work gateway defer-restart", False),
    ('hermes gateway defer-"restart"', False),
    ("hermes gateway defer-restart; hermes gateway restart", True),
    ("hermes gateway defer-restart && launchctl kickstart ai.hermes.gateway", True),
    ("sleep 30 && hermes gateway defer-restart", False),
    ("launchctl submit -l ai.hermes.gateway-x -- /bin/sh helper.sh", True),
    ("pkill -f hermes.*gateway", True),
    ("echo 'hermes gateway defer-restart' >> notes.md", False),
]


@pytest.mark.parametrize("text,want", G1_PROBES)
def test_g1_probe_verdict_unchanged(text, want):
    assert contains_gateway_lifecycle_command(text) is want
