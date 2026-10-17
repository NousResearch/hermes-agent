"""G3 (gateway-deferred-restart spec): `hermes gateway defer-restart` is forbidden to cron.

Three layers, all funnelling through `check_deferred_restart_job` so they cannot drift:

  L1  definition paths — create / edit / import / re-enable
      (`cron/jobs.py`, `cron/job_definition.py`),
  L2  fail-closed dispatch preflight, INCLUDING pre-existing jobs
      (`cron/scheduler.py` `_run_one_job_body`),
  L3  runtime refusal inside a cron execution context, provenance = the in-process
      cron execution ContextVar + session `HERMES_CRON_SESSION` marker
      (`tools/terminal_tool_guards.gateway_lifecycle_block`,
       `tools/code_execution_tool.py` mirror).

The last test module pins the 15 fact-finding G1 probe verdicts byte-for-byte: adding the
deferred channel must not shift a single existing guard decision.
"""

from __future__ import annotations

import types

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
]

NOT_DEFERRED_REFERENCES = [
    "",
    "hermes gateway restart",
    "hermes gateway restart --deferred 30",
    "deploy gateway; restart services",
    "cat <<'EOF'\ngateway defer-restart\nEOF",  # inert data heredoc is masked (G1 parity)
]


@pytest.mark.parametrize("text", DEFERRED_REFERENCES)
def test_detector_flags_deferred_restart_references(text):
    assert contains_deferred_restart_command(text) is True


@pytest.mark.parametrize("text", NOT_DEFERRED_REFERENCES)
def test_detector_ignores_other_shapes(text):
    assert contains_deferred_restart_command(text) is False


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
    script = tmp_path / "job.sh"
    script.write_text("#!/bin/sh\ngateway defer-restart\n")
    with pytest.raises(DeferredRestartBlocked):
        check_deferred_restart_payload("exec hermes", str(script))


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


# --- L3: runtime refusal in cron execution context ------------------------------

def _patch_runtime(monkeypatch, *, in_cron: bool):
    monkeypatch.setattr("tools.process_registry._is_supervised_gateway_process", lambda: True)
    monkeypatch.setattr("tools.terminal_tool.get_session_cwd", lambda key: None)
    monkeypatch.setattr("tools.terminal_tool._resolve_command_cwd", lambda **kw: "/tmp")
    monkeypatch.setattr(
        "cron.execution_identity.current_cron_execution",
        lambda: object() if in_cron else None,
    )
    monkeypatch.setattr("tools.approval_context._is_cron_approval_context", lambda: False)
    return types.SimpleNamespace(cwd=None)


def _block(command: str, *, in_cron: bool, monkeypatch):
    from tools.terminal_tool_guards import gateway_lifecycle_block

    env = _patch_runtime(monkeypatch, in_cron=in_cron)
    return gateway_lifecycle_block(
        command=command, env=env, env_type="local", cwd="/tmp",
        workdir=None, session_key="s",
    )


def test_cron_context_refuses_deferred_restart(monkeypatch):
    out = _block("hermes gateway defer-restart", in_cron=True, monkeypatch=monkeypatch)
    assert out is not None
    assert "cron execution context" in out


def test_cron_context_refuses_session_env_marker(monkeypatch):
    """The session HERMES_CRON_SESSION marker is an independent trusted provenance source."""
    from tools.terminal_tool_guards import gateway_lifecycle_block

    env = _patch_runtime(monkeypatch, in_cron=False)
    monkeypatch.setattr("tools.approval_context._is_cron_approval_context", lambda: True)
    out = gateway_lifecycle_block(
        command="hermes gateway defer-restart", env=env, env_type="local", cwd="/tmp",
        workdir=None, session_key="s",
    )
    assert out is not None and "cron execution context" in out


def test_ordinary_turn_still_allows_deferred_restart(monkeypatch):
    assert _block("hermes gateway defer-restart", in_cron=False, monkeypatch=monkeypatch) is None


def test_cron_context_clean_command_passes(monkeypatch):
    assert _block("ls -la", in_cron=True, monkeypatch=monkeypatch) is None


def test_existing_g1_block_message_unchanged_in_cron_context(monkeypatch):
    out = _block("hermes gateway restart", in_cron=True, monkeypatch=monkeypatch)
    assert out is not None
    assert "cannot restart, stop, or uninstall the gateway" in out


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
