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

import hermes_cli.gateway as gateway
import hermes_cli.gateway_windows as gateway_windows
import hermes_cli.main as hm
import hermes_cli.main_install_repair as main_install_repair
from hermes_cli.update_cmd import _resume_windows_gateways_after_update
from hermes_cli.update_inventory import (
    RuntimeRecord,
    UpdatePlan,
    match_runtime_outcomes,
    report_unaccounted_runtimes,
)


def _token(profiles: dict) -> dict:
    return {
        "resume_needed": True,
        "profiles": profiles,
        "unmapped_pids": [],
        "unmapped": [],
    }


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


def test_resume_records_successfully_relaunched_profiles_on_the_token(monkeypatch):
    monkeypatch.setattr(hm, "_is_windows", lambda: True)
    monkeypatch.setattr(main_install_repair, "_is_windows", lambda: True)
    monkeypatch.setattr(hm, "_refresh_windows_gateway_launchers", lambda: None)
    monkeypatch.setattr(
        gateway, "launch_detached_profile_gateway_restart", lambda *_a: True
    )
    monkeypatch.setattr(
        gateway, "launch_detached_gateway_restart_by_cmdline", lambda *_a: True
    )

    token = _token({"default": 1111, "work": 2222})
    with patch("builtins.print"):
        _resume_windows_gateways_after_update(token)

    assert sorted(token["relaunched_profiles"]) == ["default", "work"]


def test_resume_omits_profiles_whose_relaunch_failed(monkeypatch):
    """A profile whose relaunch genuinely fails must NOT be marked
    'relaunched' — it needs to keep surfacing as unaccounted so the user is
    told to restart it manually (Windows has no watcher to recover it)."""
    monkeypatch.setattr(hm, "_is_windows", lambda: True)
    monkeypatch.setattr(main_install_repair, "_is_windows", lambda: True)
    monkeypatch.setattr(hm, "_refresh_windows_gateway_launchers", lambda: None)

    def _relaunch(profile, _old_pid):
        return profile == "default"  # "work" fails to relaunch

    monkeypatch.setattr(
        gateway, "launch_detached_profile_gateway_restart", _relaunch
    )
    monkeypatch.setattr(
        gateway, "launch_detached_gateway_restart_by_cmdline", lambda *_a: True
    )

    token = _token({"default": 1111, "work": 2222})
    with patch("builtins.print"):
        # Fail-closed contract: a profile whose relaunch failed
        # raises so the update is marked incomplete (the caller catches,
        # records the phase error, and exits 1 in gateway mode).
        with pytest.raises(RuntimeError, match="Could not restart every paused"):
            _resume_windows_gateways_after_update(token)

    assert token["relaunched_profiles"] == ["default"]


def test_merged_windows_relaunch_resolves_as_restarted_not_unaccounted(monkeypatch):
    """End-to-end shape of the actual fix: the token's relaunched_profiles,
    merged into the shared list _cmd_update_impl passes to
    match_runtime_outcomes, must turn a Windows gateway's plan row from
    'unaccounted' (loud warning + exit 1) into 'restarted' (clean)."""
    monkeypatch.setattr(hm, "_is_windows", lambda: True)
    monkeypatch.setattr(main_install_repair, "_is_windows", lambda: True)
    monkeypatch.setattr(hm, "_refresh_windows_gateway_launchers", lambda: None)
    monkeypatch.setattr(
        gateway, "launch_detached_profile_gateway_restart", lambda *_a: True
    )
    monkeypatch.setattr(
        gateway, "launch_detached_gateway_restart_by_cmdline", lambda *_a: True
    )

    old_pid = 4242
    token = _token({"default": old_pid})
    with patch("builtins.print"):
        _resume_windows_gateways_after_update(token)

    # collect_runtime_inventory() would have recorded the pre-update PID —
    # the plan is built BEFORE the pause/relaunch, so it still names the old
    # pid even though a fresh process now owns the port.
    plan = UpdatePlan()
    plan.runtimes = [
        RuntimeRecord(
            kind="gateway", profile="default", pid=old_pid, supervisor="manual",
            restart_via="manual",
        )
    ]

    # Without the merge (the pre-fix state): unaccounted, escalates.
    pre_fix_outcomes = match_runtime_outcomes(
        plan, restarted_services=[], relaunched_profiles=[],
        externally_supervised_profiles=[], killed_pids=set(), failed_units=[],
    )
    assert pre_fix_outcomes[0]["outcome"] == "unaccounted"
    assert report_unaccounted_runtimes(pre_fix_outcomes) is True

    # With the merge _cmd_update_impl now performs: restarted, clean.
    relaunched_profiles: list = []
    for profile in token.get("relaunched_profiles") or []:
        if profile not in relaunched_profiles:
            relaunched_profiles.append(profile)

    post_fix_outcomes = match_runtime_outcomes(
        plan, restarted_services=[], relaunched_profiles=relaunched_profiles,
        externally_supervised_profiles=[], killed_pids=set(), failed_units=[],
    )
    assert post_fix_outcomes[0]["outcome"] == "restarted"
    assert report_unaccounted_runtimes(post_fix_outcomes) is False


def test_resume_with_no_relaunched_profiles_key_does_not_crash_the_merge():
    """A token from an early-return path (e.g. cold-start, no profiles) may
    never gain a 'relaunched_profiles' key at all — the merge in
    _cmd_update_impl must tolerate that (.get(...) or [])."""
    token = {"resume_needed": False}
    relaunched_profiles: list = []
    for profile in token.get("relaunched_profiles") or []:
        relaunched_profiles.append(profile)
    assert relaunched_profiles == []


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
