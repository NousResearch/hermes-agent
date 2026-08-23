"""Plan-vs-execution reconciliation (#91277 Phase 2: restart via declared mechanism).

Pins:
- _restart_mechanism returns machine-readable ids; describe_restart_mechanism
  derives display strings (policy table is data, not prose).
- match_runtime_outcomes classifies every planned runtime against the restart
  phase's bookkeeping: restarted / stopped / failed / unaccounted.
- report_unaccounted_runtimes escalates (returns True) ONLY on unaccounted
  rows — the silent-miss tripwire.
"""

from hermes_cli.update_inventory import (
    RuntimeRecord,
    UpdatePlan,
    _restart_mechanism,
    describe_restart_mechanism,
    match_runtime_outcomes,
    report_unaccounted_runtimes,
)


def _plan(*runtimes: RuntimeRecord) -> UpdatePlan:
    plan = UpdatePlan()
    plan.runtimes = list(runtimes)
    return plan


def _rt(profile: str, pid: int, supervisor: str = "manual") -> RuntimeRecord:
    return RuntimeRecord(
        kind="gateway",
        profile=profile,
        pid=pid,
        supervisor=supervisor,
        restart_via=_restart_mechanism(supervisor, profile),
    )


def test_mechanism_ids_are_machine_readable_and_described():
    assert _restart_mechanism("systemd", "default") == "systemd"
    assert _restart_mechanism("launchd", "work") == "launchd"
    assert _restart_mechanism("desktop", "default") == "desktop"
    assert _restart_mechanism("manual", "work") == "manual"
    # display derives FROM the id
    assert "systemctl" in describe_restart_mechanism("systemd", "default")
    assert "kickstart" in describe_restart_mechanism("launchd", "work")
    assert "-p work" in describe_restart_mechanism("manual", "work")
    assert describe_restart_mechanism("manual", "default") == "hermes gateway restart"


def test_relaunched_profile_is_restarted():
    outcomes = match_runtime_outcomes(
        _plan(_rt("default", 100)),
        restarted_services=[], relaunched_profiles=["default"],
        externally_supervised_profiles=[], killed_pids=set(), failed_units=[],
    )
    assert outcomes == [
        {"kind": "gateway", "profile": "default", "pid": 100,
         "mechanism": "manual", "outcome": "restarted"}
    ]
    assert report_unaccounted_runtimes(outcomes) is False


def test_killed_pid_is_stopped():
    outcomes = match_runtime_outcomes(
        _plan(_rt("work", 200)),
        restarted_services=[], relaunched_profiles=[],
        externally_supervised_profiles=[], killed_pids={200}, failed_units=[],
    )
    assert outcomes[0]["outcome"] == "stopped"


def test_failed_unit_is_failed():
    outcomes = match_runtime_outcomes(
        _plan(_rt("work", 300, supervisor="systemd")),
        restarted_services=[], relaunched_profiles=[],
        externally_supervised_profiles=[], killed_pids=set(),
        failed_units=["hermes-gateway-work.service"],
    )
    assert outcomes[0]["outcome"] == "failed"


def test_restarted_service_unit_matches_profile():
    outcomes = match_runtime_outcomes(
        _plan(_rt("default", 400, supervisor="systemd")),
        restarted_services=["hermes-gateway.service"], relaunched_profiles=[],
        externally_supervised_profiles=[], killed_pids=set(), failed_units=[],
    )
    assert outcomes[0]["outcome"] == "restarted"


def test_untouched_runtime_is_unaccounted_and_escalates(capsys):
    """The tripwire: plan saw it, NO bookkeeping mentions it."""
    outcomes = match_runtime_outcomes(
        _plan(_rt("coder", 500)),
        restarted_services=["hermes-gateway.service"],
        relaunched_profiles=["default"],
        externally_supervised_profiles=[], killed_pids={123}, failed_units=[],
    )
    assert outcomes[0]["outcome"] == "unaccounted"
    assert report_unaccounted_runtimes(outcomes) is True
    out = capsys.readouterr().out
    assert "never touched" in out
    assert "coder" in out and "500" in out
    assert "hermes -p <profile> gateway restart" in out


def test_external_supervisor_counts_as_restarted():
    outcomes = match_runtime_outcomes(
        _plan(_rt("default", 600, supervisor="desktop")),
        restarted_services=[], relaunched_profiles=[],
        externally_supervised_profiles=["default"], killed_pids=set(),
        failed_units=[],
    )
    assert outcomes[0]["outcome"] == "restarted"


def test_mixed_fleet_only_the_missed_one_escalates(capsys):
    outcomes = match_runtime_outcomes(
        _plan(
            _rt("default", 700, supervisor="systemd"),
            _rt("work", 701),
            _rt("ghost", 702),
        ),
        restarted_services=["hermes-gateway.service"],
        relaunched_profiles=["work"],
        externally_supervised_profiles=[], killed_pids=set(), failed_units=[],
    )
    by_profile = {o["profile"]: o["outcome"] for o in outcomes}
    assert by_profile == {
        "default": "restarted", "work": "restarted", "ghost": "unaccounted"
    }
    assert report_unaccounted_runtimes(outcomes) is True
    out = capsys.readouterr().out
    missed_block = out.split("never touched")[1]
    assert "ghost" in missed_block
    assert "[default]" not in missed_block
    assert "[work]" not in missed_block
