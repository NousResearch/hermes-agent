"""Regression coverage for stale ``hermes serve`` generations after an abort.

Issue #92145: ``hermes update`` moves the checkout from generation N to N+1
while the updater still holds the pre-pull module graph.  When the in-process
restart phase raises, a fresh process retries the *gateway* profiles — but
``hermes serve`` is not a gateway profile.  It hosts ``tui_gateway.server``,
systemd lists ``hermes-gateway.service`` before ``hermes-serve.service`` (so
the serve unit is the one an abort typically never reaches), and the final
read-back (``collect_fleet_versions``) only inspects gateway state.  A serve
process left on generation N is therefore invisible to every check, and the
update reports a clean recovery while every chat turn fails with an
``ImportError`` for a symbol that imports fine on disk.

These tests pin the closure predicate: no runtime from the pre-update
inventory may still be the same process once recovery reports success, and
nothing without a verified relaunch authority may be killed to get there.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from hermes_cli import update_cmd
from hermes_cli import update_restart_recovery as recovery


class _Completed:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _runtime(kind: str, profile: str, supervisor: str, pid):
    return SimpleNamespace(kind=kind, profile=profile, supervisor=supervisor, pid=pid)


def _plan(*runtimes):
    return SimpleNamespace(runtimes=list(runtimes))


def _identity_module():
    return __import__("hermes_cli.process_identity", fromlist=["ledger_entries"])


# ---------------------------------------------------------------------------
# The closure predicate itself
# ---------------------------------------------------------------------------


def _gateway_only_success():
    return {
        "requested": ["default"],
        "verified": ["default"],
        "relaunch_attempted": [],
        "failed": [],
        "skipped": [],
        "serve_units": {"verified": [], "failed": []},
    }


def test_full_gateway_recovery_does_not_clear_the_flag_while_serve_is_stale():
    """The reported bug as a predicate: gateway green, serve still generation N."""
    assert (
        update_cmd._abort_recovery_is_complete(
            planned_gateway_profiles={"default"},
            covered_gateway_profiles={"default"},
            recovery_result=_gateway_only_success(),
            stale_runtime_rows=[
                {
                    "pid": 4242,
                    "kind": "serve",
                    "profile": "default",
                    "supervisor": "manual-serve",
                }
            ],
        )
        is False
    )


def test_recovery_is_complete_when_no_runtime_survived():
    assert (
        update_cmd._abort_recovery_is_complete(
            planned_gateway_profiles={"default"},
            covered_gateway_profiles={"default"},
            recovery_result=_gateway_only_success(),
            stale_runtime_rows=[],
        )
        is True
    )


def test_failed_serve_unit_blocks_completion():
    result = _gateway_only_success()
    result["serve_units"] = {"verified": [], "failed": ["hermes-serve"]}
    assert (
        update_cmd._abort_recovery_is_complete(
            planned_gateway_profiles={"default"},
            covered_gateway_profiles={"default"},
            recovery_result=result,
            stale_runtime_rows=[],
        )
        is False
    )


@pytest.mark.parametrize("blocking", ["failed", "relaunch_attempted"])
def test_unverified_gateway_outcomes_still_block_completion(blocking):
    result = _gateway_only_success()
    result[blocking] = ["default"]
    assert (
        update_cmd._abort_recovery_is_complete(
            planned_gateway_profiles={"default"},
            covered_gateway_profiles={"default"},
            recovery_result=result,
            stale_runtime_rows=[],
        )
        is False
    )


def test_uncovered_gateway_profile_blocks_completion():
    assert (
        update_cmd._abort_recovery_is_complete(
            planned_gateway_profiles={"default", "coder"},
            covered_gateway_profiles={"default"},
            recovery_result=_gateway_only_success(),
            stale_runtime_rows=[],
        )
        is False
    )


def test_no_planned_gateway_leg_is_not_completeness():
    """A serve-only fleet must fall through to the caller's fail-closed path."""
    assert (
        update_cmd._abort_recovery_is_complete(
            planned_gateway_profiles=set(),
            covered_gateway_profiles=set(),
            recovery_result=_gateway_only_success(),
            stale_runtime_rows=[],
        )
        is False
    )


def test_legacy_recovery_result_without_serve_key_still_evaluates():
    """Forward/backward compatibility: a pre-#92145 shape must not crash."""
    legacy = {
        "requested": ["default"],
        "verified": ["default"],
        "relaunch_attempted": [],
        "failed": [],
        "skipped": [],
    }
    assert (
        update_cmd._abort_recovery_is_complete(
            planned_gateway_profiles={"default"},
            covered_gateway_profiles={"default"},
            recovery_result=legacy,
            stale_runtime_rows=[],
        )
        is True
    )


# ---------------------------------------------------------------------------
# Survivor probe — identity, not PID guessing
# ---------------------------------------------------------------------------


def test_serve_process_that_never_restarted_is_reported_as_stale(monkeypatch):
    monkeypatch.setattr(
        _identity_module(),
        "ledger_entries",
        lambda *a, **k: [{"pid": 4242, "purpose": "serve"}],
    )
    rows = update_cmd._surviving_pre_update_serve_runtimes(
        _plan(_runtime("serve", "default", "manual-serve", 4242))
    )
    assert rows == [
        {
            "pid": 4242,
            "kind": "serve",
            "profile": "default",
            "supervisor": "manual-serve",
        }
    ]


def test_restarted_serve_leaves_no_survivor(monkeypatch):
    """A restarted unit re-registers under a NEW pid; the old row is pruned."""
    monkeypatch.setattr(
        _identity_module(),
        "ledger_entries",
        lambda *a, **k: [{"pid": 9999, "purpose": "serve"}],
    )
    assert (
        update_cmd._surviving_pre_update_serve_runtimes(
            _plan(_runtime("serve", "default", "manual-serve", 4242))
        )
        == []
    )


def test_gateway_runtimes_are_not_counted_as_serve_survivors(monkeypatch):
    monkeypatch.setattr(
        _identity_module(),
        "ledger_entries",
        lambda *a, **k: [{"pid": 4242, "purpose": "gateway"}],
    )
    assert (
        update_cmd._surviving_pre_update_serve_runtimes(
            _plan(_runtime("gateway", "default", "systemd", 4242))
        )
        == []
    )


def test_dashboard_survivors_count_too(monkeypatch):
    monkeypatch.setattr(
        _identity_module(),
        "ledger_entries",
        lambda *a, **k: [{"pid": 77, "purpose": "dashboard"}],
    )
    rows = update_cmd._surviving_pre_update_serve_runtimes(
        _plan(_runtime("dashboard", "default", "manual-serve", 77))
    )
    assert [row["kind"] for row in rows] == ["dashboard"]


def test_unreadable_ledger_fails_closed(monkeypatch):
    """Unprovable state is unsafe state — never a silent all-clear."""

    def boom(*a, **k):
        raise OSError("ledger unreadable")

    monkeypatch.setattr(_identity_module(), "ledger_entries", boom)
    rows = update_cmd._surviving_pre_update_serve_runtimes(
        _plan(_runtime("serve", "default", "manual-serve", 4242))
    )
    assert [row["pid"] for row in rows] == [4242]


def test_no_inventoried_serve_runtime_is_a_clean_no_op(monkeypatch):
    def boom(*a, **k):  # must not even be consulted
        raise AssertionError("ledger probed with nothing to check")

    monkeypatch.setattr(_identity_module(), "ledger_entries", boom)
    assert (
        update_cmd._surviving_pre_update_serve_runtimes(
            _plan(_runtime("gateway", "default", "systemd", 111))
        )
        == []
    )


def test_runtimes_without_a_usable_pid_are_ignored(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("ledger probed with nothing to check")

    monkeypatch.setattr(_identity_module(), "ledger_entries", boom)
    assert (
        update_cmd._surviving_pre_update_serve_runtimes(
            _plan(
                _runtime("serve", "default", "manual-serve", 0),
                _runtime("serve", "other", "manual-serve", None),
            )
        )
        == []
    )


# ---------------------------------------------------------------------------
# restart_serve_units — what may claim "verified"
# ---------------------------------------------------------------------------


class _Systemctl:
    """Scriptable systemctl double: unit table + MainPID progression."""

    def __init__(self, listed, active, main_pids, restart_rc=0):
        self.listed = listed
        self.active = dict(active)
        self.main_pids = dict(main_pids)
        self.restart_rc = restart_rc
        self.restarted: list[str] = []

    def __call__(self, argv, **kwargs):
        if "list-units" in argv:
            if "--user" not in argv:
                return _Completed(0, stdout="")
            body = "\n".join(
                f"{unit} loaded active running Hermes" for unit in self.listed
            )
            return _Completed(0, stdout=body)
        if "is-active" in argv:
            unit = argv[-1]
            return _Completed(
                0, stdout="active" if self.active.get(unit) else "inactive"
            )
        if "show" in argv:
            unit = argv[argv.index("show") + 1]
            return _Completed(0, stdout=str(self.main_pids.get(unit, 0)))
        if "restart" in argv:
            unit = argv[-1]
            self.restarted.append(unit)
            if self.restart_rc == 0:
                self.main_pids[unit] = self.main_pids.get(unit, 0) + 1000
            return _Completed(self.restart_rc)
        raise AssertionError(f"unexpected systemctl call: {argv}")


@pytest.fixture
def linux_systemctl(monkeypatch):
    monkeypatch.setattr(recovery.sys, "platform", "linux")
    monkeypatch.setattr(recovery.shutil, "which", lambda name: "/usr/bin/systemctl")


def test_serve_unit_verified_when_main_pid_changes(linux_systemctl):
    fake = _Systemctl(
        listed=["hermes-serve.service"],
        active={"hermes-serve.service": True},
        main_pids={"hermes-serve.service": 4242},
    )
    out = recovery.restart_serve_units(run=fake, sleep=lambda _: None)
    assert out == {"verified": ["hermes-serve"], "failed": []}
    assert fake.restarted == ["hermes-serve.service"]


def test_unchanged_main_pid_is_not_verified(linux_systemctl):
    """rc 0 with the same main process means the stale interpreter survived."""
    fake = _Systemctl(
        listed=["hermes-serve.service"],
        active={"hermes-serve.service": True},
        main_pids={"hermes-serve.service": 4242},
    )
    original_call = fake.__call__

    def frozen(argv, **kwargs):
        if "restart" in argv:
            fake.restarted.append(argv[-1])
            return _Completed(0)  # rc 0, MainPID deliberately untouched
        return original_call(argv, **kwargs)

    out = recovery.restart_serve_units(run=frozen, sleep=lambda _: None)
    assert out == {"verified": [], "failed": ["hermes-serve"]}


def test_unit_that_does_not_come_back_active_is_failed(linux_systemctl):
    fake = _Systemctl(
        listed=["hermes-serve.service"],
        active={"hermes-serve.service": True},
        main_pids={"hermes-serve.service": 4242},
    )
    original_call = fake.__call__

    def dies(argv, **kwargs):
        if "restart" in argv:
            fake.restarted.append(argv[-1])
            fake.active["hermes-serve.service"] = False
            return _Completed(0)
        return original_call(argv, **kwargs)

    out = recovery.restart_serve_units(run=dies, sleep=lambda _: None)
    assert out == {"verified": [], "failed": ["hermes-serve"]}


def test_nonzero_restart_is_failed_not_silently_skipped(linux_systemctl):
    """The unprivileged system-scope case must read as failure, not absence."""
    fake = _Systemctl(
        listed=["hermes-serve.service"],
        active={"hermes-serve.service": True},
        main_pids={"hermes-serve.service": 4242},
        restart_rc=1,
    )
    out = recovery.restart_serve_units(run=fake, sleep=lambda _: None)
    assert out == {"verified": [], "failed": ["hermes-serve"]}


def test_restart_timeout_is_failed(linux_systemctl):
    fake = _Systemctl(
        listed=["hermes-serve.service"],
        active={"hermes-serve.service": True},
        main_pids={"hermes-serve.service": 4242},
    )
    original_call = fake.__call__

    def timing_out(argv, **kwargs):
        if "restart" in argv:
            raise subprocess.TimeoutExpired(cmd=argv, timeout=60)
        return original_call(argv, **kwargs)

    out = recovery.restart_serve_units(run=timing_out, sleep=lambda _: None)
    assert out == {"verified": [], "failed": ["hermes-serve"]}


def test_inactive_serve_unit_is_left_alone(linux_systemctl):
    """Nothing inactive can be serving a stale generation."""
    fake = _Systemctl(
        listed=["hermes-serve.service"],
        active={"hermes-serve.service": False},
        main_pids={"hermes-serve.service": 0},
    )
    out = recovery.restart_serve_units(run=fake, sleep=lambda _: None)
    assert out == {"verified": [], "failed": []}
    assert fake.restarted == []


def test_units_already_restarted_by_the_phase_are_skipped(linux_systemctl):
    fake = _Systemctl(
        listed=["hermes-serve.service", "hermes-serve-work.service"],
        active={"hermes-serve.service": True, "hermes-serve-work.service": True},
        main_pids={"hermes-serve.service": 1, "hermes-serve-work.service": 2},
    )
    out = recovery.restart_serve_units(
        skip_units=["hermes-serve"], run=fake, sleep=lambda _: None
    )
    assert fake.restarted == ["hermes-serve-work.service"]
    assert out == {"verified": ["hermes-serve-work"], "failed": []}


def test_unrelated_hermes_server_service_is_never_touched(linux_systemctl):
    """``hermes-serve*`` is a systemd glob, not a name gate (review on #83595)."""
    fake = _Systemctl(
        listed=["hermes-server.service", "hermes-serve.service"],
        active={"hermes-server.service": True, "hermes-serve.service": True},
        main_pids={"hermes-server.service": 7, "hermes-serve.service": 8},
    )
    out = recovery.restart_serve_units(run=fake, sleep=lambda _: None)
    assert fake.restarted == ["hermes-serve.service"]
    assert out["verified"] == ["hermes-serve"]


def test_gateway_units_are_not_restarted_by_the_serve_pass(linux_systemctl):
    """Gateway profiles have their own per-profile relaunch command."""
    fake = _Systemctl(
        listed=["hermes-gateway.service"],
        active={"hermes-gateway.service": True},
        main_pids={"hermes-gateway.service": 5},
    )
    out = recovery.restart_serve_units(run=fake, sleep=lambda _: None)
    assert fake.restarted == []
    assert out == {"verified": [], "failed": []}


def test_no_systemctl_means_no_serve_pass(monkeypatch):
    monkeypatch.setattr(recovery.shutil, "which", lambda name: None)

    def unreachable(*a, **k):
        raise AssertionError("systemctl invoked without a systemctl binary")

    assert recovery.restart_serve_units(run=unreachable) == {
        "verified": [],
        "failed": [],
    }


def test_non_linux_hosts_do_not_run_the_serve_pass(monkeypatch):
    monkeypatch.setattr(recovery.sys, "platform", "win32")
    monkeypatch.setattr(recovery.shutil, "which", lambda name: "systemctl")

    def unreachable(*a, **k):
        raise AssertionError("serve unit pass ran off Linux")

    assert recovery.restart_serve_units(run=unreachable) == {
        "verified": [],
        "failed": [],
    }


def test_active_unit_with_no_readable_main_pid_is_failed(linux_systemctl):
    """No observable main process means no observable replacement."""
    fake = _Systemctl(
        listed=["hermes-serve.service"],
        active={"hermes-serve.service": True},
        main_pids={"hermes-serve.service": 0},
    )
    out = recovery.restart_serve_units(run=fake, sleep=lambda _: None)
    assert out == {"verified": [], "failed": ["hermes-serve"]}
    assert fake.restarted == []


def test_same_unit_name_in_both_scopes_is_proven_in_both(linux_systemctl):
    """User and system scope are two different processes, not one."""
    calls: list[tuple[str, list]] = []

    def dual_scope(argv, **kwargs):
        scope = "user" if "--user" in argv else "system"
        calls.append((scope, list(argv)))
        if "list-units" in argv:
            return _Completed(0, stdout="hermes-serve.service loaded active running x")
        if "is-active" in argv:
            return _Completed(0, stdout="active")
        if "show" in argv:
            return _Completed(0, stdout="10" if scope == "user" else "20")
        if "restart" in argv:
            # The user scope relaunches cleanly; the system scope is refused
            # (no privilege) and must not be masked by the user-scope success.
            return _Completed(0 if scope == "user" else 1)
        raise AssertionError(argv)

    out = recovery.restart_serve_units(run=dual_scope, sleep=lambda _: None)
    restarts = [scope for scope, argv in calls if "restart" in argv]
    assert restarts == ["user", "system"]
    assert out == {"verified": [], "failed": ["hermes-serve"]}


def test_restart_is_invoked_without_an_interactive_auth_prompt(linux_systemctl):
    """A polkit prompt inside a captured subprocess hangs the whole update."""
    seen = []
    fake = _Systemctl(
        listed=["hermes-serve.service"],
        active={"hermes-serve.service": True},
        main_pids={"hermes-serve.service": 4242},
    )
    original_call = fake.__call__

    def recording(argv, **kwargs):
        if "restart" in argv:
            seen.append(list(argv))
        return original_call(argv, **kwargs)

    recovery.restart_serve_units(run=recording, sleep=lambda _: None)
    assert seen and "--no-ask-password" in seen[0]


# ---------------------------------------------------------------------------
# Payload boundary
# ---------------------------------------------------------------------------


def test_skip_unit_names_are_filtered_to_the_serve_family():
    _, _, recover_serve, skip = recovery._parse_payload(
        io.StringIO(
            json.dumps(
                {
                    "profiles": [],
                    "serve_units": {
                        "recover": True,
                        "skip": [
                            "hermes-serve",
                            "hermes-serve-work.service",
                            "hermes-server",
                            "../../etc/systemd/evil",
                            "hermes-serve; rm -rf /",
                            "hermes-gateway",
                        ],
                    },
                }
            )
        )
    )
    assert recover_serve is True
    assert skip == ["hermes-serve", "hermes-serve-work.service"]


def test_malformed_serve_block_is_rejected():
    with pytest.raises(ValueError, match="serve_units"):
        recovery._parse_payload(
            io.StringIO(json.dumps({"profiles": [], "serve_units": ["nope"]}))
        )


def test_non_string_skip_entries_are_rejected():
    with pytest.raises(ValueError, match="skip list"):
        recovery._parse_payload(
            io.StringIO(
                json.dumps(
                    {"profiles": [], "serve_units": {"recover": True, "skip": [7]}}
                )
            )
        )


def test_absent_serve_block_defaults_to_no_serve_recovery():
    _, _, recover_serve, skip = recovery._parse_payload(
        io.StringIO(json.dumps({"profiles": []}))
    )
    assert recover_serve is False
    assert skip == []


# ---------------------------------------------------------------------------
# Driver: the fresh child is still launched for a serve-only fleet
# ---------------------------------------------------------------------------


def test_serve_only_fleet_still_spawns_the_recovery_child(monkeypatch):
    """Before #92145 the driver returned early whenever no gateway profile was
    supervised, so a stale serve unit was never even attempted."""
    monkeypatch.setattr(update_cmd, "_serve_unit_recovery_available", lambda: True)
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return _Completed(
            0,
            stdout=json.dumps(
                {
                    "verified": [],
                    "relaunch_attempted": [],
                    "failed": [],
                    "serve_units": {"verified": ["hermes-serve"], "failed": []},
                }
            ),
        )

    monkeypatch.setattr(update_cmd.subprocess, "run", fake_run)
    result = update_cmd._recover_gateway_restart_after_abort(
        _plan(_runtime("serve", "default", "manual-serve", 4242)),
        gateway_mode=False,
    )
    assert len(calls) == 1
    payload = json.loads(calls[0][1]["input"])
    assert payload["profiles"] == []
    assert payload["serve_units"]["recover"] is True
    assert result["serve_units"] == {"verified": ["hermes-serve"], "failed": []}


def test_child_timeout_budget_covers_the_serve_pass(monkeypatch):
    """A serve-only recovery must not be killed before its settle window."""
    monkeypatch.setattr(update_cmd, "_serve_unit_recovery_available", lambda: True)
    captured = {}

    def fake_run(argv, **kwargs):
        captured["timeout"] = kwargs["timeout"]
        return _Completed(
            0,
            stdout=json.dumps(
                {
                    "verified": [],
                    "relaunch_attempted": [],
                    "failed": [],
                    "serve_units": {"verified": [], "failed": []},
                }
            ),
        )

    monkeypatch.setattr(update_cmd.subprocess, "run", fake_run)
    update_cmd._recover_gateway_restart_after_abort(
        _plan(_runtime("serve", "default", "manual-serve", 4242)),
        gateway_mode=False,
    )
    assert captured["timeout"] >= 180


def test_verified_serve_units_do_not_enter_gateway_restart_vocabulary(monkeypatch):
    """Serve coverage must not silently widen the gateway fleet-probe gate."""
    monkeypatch.setattr(update_cmd, "_serve_unit_recovery_available", lambda: True)

    def fake_run(argv, **kwargs):
        return _Completed(
            0,
            stdout=json.dumps(
                {
                    "verified": ["default"],
                    "relaunch_attempted": [],
                    "failed": [],
                    "serve_units": {"verified": ["hermes-serve"], "failed": []},
                }
            ),
        )

    monkeypatch.setattr(update_cmd.subprocess, "run", fake_run)
    result = update_cmd._recover_gateway_restart_after_abort(
        _plan(_runtime("gateway", "default", "systemd", 111)),
        gateway_mode=False,
    )
    assert result["verified"] == ["default"]
    assert "hermes-serve" not in result["verified"]


def test_no_serve_authority_and_no_gateway_profile_spawns_nothing(monkeypatch):
    monkeypatch.setattr(update_cmd, "_serve_unit_recovery_available", lambda: False)

    def unreachable(*a, **k):
        raise AssertionError("recovery child spawned with nothing to recover")

    monkeypatch.setattr(update_cmd.subprocess, "run", unreachable)
    result = update_cmd._recover_gateway_restart_after_abort(
        _plan(_runtime("serve", "default", "manual-serve", 4242)),
        gateway_mode=False,
    )
    assert result["requested"] == []
    assert result["serve_units"] == {"verified": [], "failed": []}


def test_already_restarted_units_are_forwarded_as_skips(monkeypatch):
    monkeypatch.setattr(update_cmd, "_serve_unit_recovery_available", lambda: True)
    captured = {}

    def fake_run(argv, **kwargs):
        captured["payload"] = json.loads(kwargs["input"])
        return _Completed(
            0,
            stdout=json.dumps(
                {
                    "verified": ["default"],
                    "relaunch_attempted": [],
                    "failed": [],
                    "serve_units": {"verified": [], "failed": []},
                }
            ),
        )

    monkeypatch.setattr(update_cmd.subprocess, "run", fake_run)
    update_cmd._recover_gateway_restart_after_abort(
        _plan(_runtime("gateway", "default", "systemd", 111)),
        gateway_mode=False,
        skip_units={"hermes-serve"},
    )
    assert captured["payload"]["serve_units"]["skip"] == ["hermes-serve"]


def test_unreadable_serve_block_from_the_child_reads_as_failure(monkeypatch):
    """A child that answers with a broken serve block has proven nothing."""
    monkeypatch.setattr(update_cmd, "_serve_unit_recovery_available", lambda: True)

    def fake_run(argv, **kwargs):
        return _Completed(
            0,
            stdout=json.dumps(
                {
                    "verified": ["default"],
                    "relaunch_attempted": [],
                    "failed": [],
                    "serve_units": {"verified": "not-a-list", "failed": []},
                }
            ),
        )

    monkeypatch.setattr(update_cmd.subprocess, "run", fake_run)
    result = update_cmd._recover_gateway_restart_after_abort(
        _plan(_runtime("gateway", "default", "systemd", 111)),
        gateway_mode=False,
    )
    assert result["serve_units"]["failed"] == ["<unreadable>"]
    assert (
        update_cmd._abort_recovery_is_complete(
            planned_gateway_profiles={"default"},
            covered_gateway_profiles={"default"},
            recovery_result=result,
            stale_runtime_rows=[],
        )
        is False
    )


def test_missing_serve_block_off_linux_is_not_a_failure(monkeypatch):
    """Hosts with no serve authority must not manufacture a phantom failure."""
    monkeypatch.setattr(update_cmd, "_serve_unit_recovery_available", lambda: False)

    def fake_run(argv, **kwargs):
        return _Completed(
            0,
            stdout=json.dumps(
                {
                    "verified": ["default"],
                    "relaunch_attempted": [],
                    "failed": [],
                }
            ),
        )

    monkeypatch.setattr(update_cmd.subprocess, "run", fake_run)
    result = update_cmd._recover_gateway_restart_after_abort(
        _plan(_runtime("gateway", "default", "systemd", 111)),
        gateway_mode=False,
    )
    assert result["serve_units"] == {"verified": [], "failed": []}


def test_spawn_failure_reports_empty_serve_coverage(monkeypatch):
    monkeypatch.setattr(update_cmd, "_serve_unit_recovery_available", lambda: True)

    def boom(*a, **k):
        raise OSError("no fork available")

    monkeypatch.setattr(update_cmd.subprocess, "run", boom)
    result = update_cmd._recover_gateway_restart_after_abort(
        _plan(_runtime("gateway", "default", "systemd", 111)),
        gateway_mode=False,
    )
    assert result["failed"] == ["default"]
    assert result["serve_units"] == {"verified": [], "failed": []}


# ---------------------------------------------------------------------------
# Receipt persistence
# ---------------------------------------------------------------------------


def test_serve_coverage_reaches_the_persisted_receipt(tmp_path, monkeypatch):
    from hermes_cli import update_receipt

    monkeypatch.setattr(update_receipt, "_receipt_dir", lambda: tmp_path)
    update_receipt.begin_update_receipt()
    update_receipt.record_gateway_restart(
        restarted_services=[],
        incomplete=True,
        phase_error="cannot import name 'line_input' from 'hermes_cli.cli_output'",
        fresh_recovery={
            "requested": ["default"],
            "verified": ["default"],
            "relaunch_attempted": [],
            "failed": [],
            "skipped": [],
            "serve_units": {
                "verified": ["hermes-serve"],
                "failed": ["hermes-serve-work"],
            },
            "stale_runtimes": [
                {
                    "pid": 4242,
                    "kind": "serve",
                    "profile": "default",
                    "supervisor": "manual-serve",
                }
            ],
        },
    )
    path = update_receipt.finalize_update_receipt("partial")
    assert path is not None
    persisted = json.loads(path.read_text(encoding="utf-8"))["gateway_restart"][
        "fresh_recovery"
    ]
    assert persisted["serve_units"] == {
        "verified": ["hermes-serve"],
        "failed": ["hermes-serve-work"],
    }
    assert persisted["stale_runtimes"] == [
        {
            "pid": 4242,
            "kind": "serve",
            "profile": "default",
            "supervisor": "manual-serve",
        }
    ]


# ---------------------------------------------------------------------------
# Operator-facing output
# ---------------------------------------------------------------------------


def test_stale_serve_warning_names_the_process_and_the_fix(capsys):
    update_cmd._warn_stale_serve_runtimes(
        [
            {
                "pid": 4242,
                "kind": "serve",
                "profile": "default",
                "supervisor": "manual-serve",
            }
        ]
    )
    out = capsys.readouterr().out
    assert "4242" in out
    assert "serve" in out
    assert "systemctl --user restart hermes-serve.service" in out


def test_no_survivors_prints_nothing(capsys):
    update_cmd._warn_stale_serve_runtimes([])
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# Real fresh interpreter
# ---------------------------------------------------------------------------


def test_recovery_module_reports_serve_units_in_a_real_process():
    """The protocol survives a genuine subprocess round-trip."""
    result = subprocess.run(
        [sys.executable, "-m", "hermes_cli.update_restart_recovery", "--stdin"],
        input=json.dumps(
            {"profiles": [], "serve_units": {"recover": True, "skip": []}}
        ),
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert set(payload["serve_units"]) == {"verified", "failed"}


# ---------------------------------------------------------------------------
# The managed-dashboard fallback must not short-circuit the serve scan
# ---------------------------------------------------------------------------


def _dashboard_main_stub(scan_calls, *, restart_result=True):
    """A ``_m()`` stand-in for the dashboard-cleanup path."""
    return SimpleNamespace(
        _DASHBOARD_SYSTEMD_UNIT="hermes-dashboard.service",
        _restart_managed_dashboard_service=lambda reason, *a, **k: restart_result,
        _find_stale_dashboard_pids=lambda **kwargs: scan_calls.append(kwargs) or [],
    )


def test_managed_dashboard_restart_still_scans_for_serve_backends(monkeypatch):
    """A restarted dashboard unit may not end the pass (#92145).

    The reporter's host runs ``hermes-dashboard.service`` AND
    ``hermes-serve.service``.  Returning as soon as the dashboard unit was
    restarted meant the serve backend hosting ``tui_gateway`` was never even
    looked for, so it kept serving the pre-update generation.
    """
    from hermes_cli import dashboard_procs

    scan_calls: list[dict] = []
    monkeypatch.setattr(
        dashboard_procs, "_m", lambda: _dashboard_main_stub(scan_calls)
    )
    monkeypatch.setattr(dashboard_procs, "_lock_owned_serve_pids", lambda: set())

    dashboard_procs._kill_stale_dashboard_processes(restart_managed=True)

    assert scan_calls, "serve/dashboard scan never ran after the dashboard restart"


def test_restarted_dashboard_unit_is_not_killed_by_the_continued_scan(monkeypatch):
    """Continuing the scan must not undo the restart it just performed."""
    from hermes_cli import dashboard_procs

    killed: list[int] = []
    main = SimpleNamespace(
        _DASHBOARD_SYSTEMD_UNIT="hermes-dashboard.service",
        _restart_managed_dashboard_service=lambda reason, *a, **k: True,
        _find_stale_dashboard_pids=lambda **kwargs: [4242],
        _get_pid_cgroup_path=lambda pid: None,
        _get_systemd_service_for_pid=lambda pid: "hermes-dashboard.service",
        _dashboard_cmdline_for_pid=lambda pid: None,
    )
    monkeypatch.setattr(dashboard_procs, "_m", lambda: main)
    monkeypatch.setattr(dashboard_procs, "_lock_owned_serve_pids", lambda: set())
    monkeypatch.setattr(dashboard_procs.sys, "platform", "linux")
    monkeypatch.setattr(
        dashboard_procs.os, "kill", lambda pid, sig: killed.append(pid)
    )

    result = dashboard_procs._kill_stale_dashboard_processes(restart_managed=True)

    assert killed == []
    assert result["killed"] == []


def test_serve_backend_survives_selection_when_the_dashboard_unit_restarts(monkeypatch):
    """A serve PID owned by a DIFFERENT unit is still selected for recovery."""
    from hermes_cli import dashboard_procs

    signalled: list[int] = []
    restarted: list[str] = []
    main = SimpleNamespace(
        _DASHBOARD_SYSTEMD_UNIT="hermes-dashboard.service",
        _restart_managed_dashboard_service=lambda reason, *a, **k: True,
        _find_stale_dashboard_pids=lambda **kwargs: [7001],
        _get_pid_cgroup_path=lambda pid: "/user.slice/hermes-serve.service",
        _get_systemd_service_for_pid=lambda pid: "hermes-serve.service",
        _dashboard_cmdline_for_pid=lambda pid: None,
        _try_restart_systemd_service=lambda svc, cg: restarted.append(svc) or True,
        _respawn_dashboard_processes=lambda cmds: [],
    )
    monkeypatch.setattr(dashboard_procs, "_m", lambda: main)
    monkeypatch.setattr(dashboard_procs, "_lock_owned_serve_pids", lambda: set())
    monkeypatch.setattr(dashboard_procs.sys, "platform", "linux")

    def _fake_kill(pid, sig):
        signalled.append(pid)
        raise ProcessLookupError

    monkeypatch.setattr(dashboard_procs.os, "kill", _fake_kill)

    result = dashboard_procs._kill_stale_dashboard_processes(restart_managed=True)

    assert signalled == [7001]
    assert restarted == ["hermes-serve.service"]
    assert result["killed"] == [7001]
