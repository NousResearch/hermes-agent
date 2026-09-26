"""Regression: Windows gateway pause/resume must feed the #91277 Phase 2
plan-vs-execution reconciliation, not report a correctly-relaunched Windows
gateway as "unaccounted".

``_pause_windows_gateways_for_update`` / ``_resume_windows_gateways_after_update``
are Windows's own gateway restart mechanism — separate from the
systemd/launchd restart phase in ``_cmd_update_impl`` that populates
``restarted_services`` / ``relaunched_profiles`` / ``killed_pids`` /
``externally_supervised_profiles``. Before this fix, a Windows gateway that
was correctly paused and relaunched left no trace in that bookkeeping, so
``match_runtime_outcomes`` classified it "unaccounted" — the plan saw it and
NO bookkeeping mentions it — and ``report_unaccounted_runtimes`` escalated
that into ``sys.exit(1)`` even though the update (and the restart) succeeded.

``_resume_windows_gateways_after_update`` now writes the profiles it
successfully relaunched onto ``token["relaunched_profiles"]``; the update
command merges that into the shared ``relaunched_profiles`` list before
reconciliation runs (mirrored here directly, since driving the full
``_cmd_update_impl`` end to end is impractical).
"""

from unittest.mock import patch

import pytest

import hermes_cli.gateway_windows as gateway_windows
import hermes_cli.main as hm


@pytest.fixture(autouse=True)
def _stub_post_relaunch_liveness(monkeypatch):
    """The resume path now verifies a stable gateway process actually exists
    before vouching for the relaunch (#48820 3rd/4th repro — a parent Job
    Object killing the respawned gateway made '✓ Restarting' a lie). These
    reconciliation tests exercise the token bookkeeping, not the liveness
    poll, so stub it as 'gateway came up'."""
    monkeypatch.setattr(
        gateway_windows, "_wait_for_gateway_ready", lambda **_kw: [4242]
    )
    monkeypatch.setattr(
        gateway_windows, "_write_start_attestation", lambda *_a, **_kw: None
    )


def test_merge_helper_reads_token_keys_into_restart_outcome(monkeypatch):
    """Drive the real merge helper (not a mirror): the Windows resume token's
    ``relaunched_profiles`` / ``restarted_services`` / ``service_profiles`` /
    ``services`` keys must land in the shared restart bookkeeping."""
    from hermes_cli import update_cmd

    monkeypatch.setattr(hm, "_resume_windows_gateways_after_update", lambda token: None)
    outcome = update_cmd._GatewayRestartOutcome(
        incomplete=False,
        phase_errors=[],
        pre_restart_gateway_pids=[],
        restarted_services=["hermes-gateway"],
        failed_or_stale_units=[],
        relaunched_profiles=[],
        externally_supervised_profiles=[],
        killed_pids=set(),
    )
    token = {
        "resume_needed": False,
        "relaunched_profiles": ["p1"],
        "restarted_services": ["svc"],
        "service_profiles": {"svc": "p2", "pending": "p3"},
        "services": ["pending"],
    }
    with patch("hermes_cli.update_receipt.record_gateway_restart", lambda **kw: None):
        update_cmd._resume_windows_gateways_and_merge_outcome(outcome, token, False)

    assert outcome.relaunched_profiles == ["p1", "p2"]
    assert outcome.restarted_services == ["hermes-gateway", "svc"]
    assert outcome.failed_or_stale_units == ["p3"]
    assert outcome.incomplete is False


# ---------------------------------------------------------------------------
# #115563: symmetric resume-failure handling + atexit double-fire
# ---------------------------------------------------------------------------

def test_resume_unregisters_its_own_atexit_fallback_before_running(monkeypatch):
    """Every foreground call site registers this same function via atexit as a dead-process
    safety net. Once execution actually reaches here it must disarm that fallback immediately
    -- otherwise a failure below (or the foreground caller dying right after return) replays
    the identical error a second time at interpreter teardown (#115563)."""
    import atexit

    from hermes_cli import update_cmd_windows

    calls = []
    monkeypatch.setattr(
        atexit, "unregister",
        lambda fn: calls.append(fn) or None,
    )
    monkeypatch.setattr(hm, "_is_windows", lambda: False)

    token = {"resume_needed": True}
    update_cmd_windows._resume_windows_gateways_after_update(token)

    assert calls == [update_cmd_windows._resume_windows_gateways_after_update]
    assert token["resume_needed"] is False
