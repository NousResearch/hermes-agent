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
    assert _restart_mechanism("windows-service", "default") == "windows-service"
    # display derives FROM the id
    assert "systemctl" in describe_restart_mechanism("systemd", "default")
    assert "kickstart" in describe_restart_mechanism("launchd", "work")
    assert "-p work" in describe_restart_mechanism("manual", "work")
    assert describe_restart_mechanism("manual", "default") == "hermes gateway restart"
    assert "sc.exe" in describe_restart_mechanism("windows-service", "default")


def test_windows_service_supervisor_classification():
    from hermes_cli.update_inventory import _detect_supervisor_for_pid

    # An SCM-owned gateway PID classifies as windows-service even when the
    # generic service-PID probe also knows the pid.
    assert (
        _detect_supervisor_for_pid(41, set(), {41}) == "windows-service"
    )
    assert (
        _detect_supervisor_for_pid(41, {41}, {41}) == "windows-service"
    )
    # Without SCM ownership the existing classification is untouched.
    assert _detect_supervisor_for_pid(42, set(), set()) == "manual"
    assert _detect_supervisor_for_pid(42, set(), None) == "manual"


def test_windows_service_runtime_reconciles_via_service_profiles():
    # The update path merges the pause token's service_profiles into
    # relaunched_profiles after sc.exe start — a restarted SCM gateway
    # must not trip the unaccounted tripwire.
    outcomes = match_runtime_outcomes(
        _plan(_rt("default", 500, supervisor="windows-service")),
        restarted_services=["hermes-gateway"], relaunched_profiles=["default"],
        externally_supervised_profiles=[], killed_pids=set(), failed_units=[],
    )
    assert outcomes == [
        {"kind": "gateway", "profile": "default", "pid": 500,
         "mechanism": "windows-service", "outcome": "restarted"}
    ]
    assert report_unaccounted_runtimes(outcomes) is False


def test_windows_service_runtime_unaccounted_when_restart_fails():
    outcomes = match_runtime_outcomes(
        _plan(_rt("work", 501, supervisor="windows-service")),
        restarted_services=[], relaunched_profiles=[],
        externally_supervised_profiles=[], killed_pids=set(), failed_units=[],
    )
    assert outcomes[0]["mechanism"] == "windows-service"
    assert outcomes[0]["outcome"] == "unaccounted"
    assert report_unaccounted_runtimes(outcomes) is True


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


def test_unmanaged_serve_runtime_under_default_profile_is_unaccounted():
    """#100479: an sshd-spawned `serve --isolated` has no systemd unit and
    shares the default profile with the gateway. A gateway-only restart
    must not be read as covering it — it must trip the tripwire instead."""
    serve_runtime = RuntimeRecord(
        kind="serve",
        profile="default",
        pid=900,
        supervisor="manual-serve",
        restart_via=_restart_mechanism("manual-serve", "default"),
    )
    outcomes = match_runtime_outcomes(
        _plan(_rt("default", 100, supervisor="systemd"), serve_runtime),
        restarted_services=["hermes-gateway"], relaunched_profiles=[],
        externally_supervised_profiles=[], killed_pids=set(), failed_units=[],
    )
    by_pid = {o["pid"]: o["outcome"] for o in outcomes}
    assert by_pid[100] == "restarted"
    assert by_pid[900] == "unaccounted"
    assert report_unaccounted_runtimes(outcomes) is True


def _serve(profile: str, pid: int, kind: str = "serve") -> RuntimeRecord:
    return RuntimeRecord(
        kind=kind,
        profile=profile,
        pid=pid,
        supervisor="manual-serve",
        restart_via=_restart_mechanism("manual-serve", profile),
    )


def test_serve_never_borrows_relaunched_or_external_gateway_profile():
    """Sibling site of #100479: the relaunched_profiles / external-supervisor
    bookkeeping is gateway vocabulary too. A manual gateway relaunch under
    ``default`` (or a named profile) says nothing about a serve that shares
    the profile name."""
    outcomes = match_runtime_outcomes(
        _plan(_rt("default", 100), _serve("default", 900),
              _rt("work", 101), _serve("work", 901, kind="dashboard")),
        restarted_services=[], relaunched_profiles=["default"],
        externally_supervised_profiles=["work"], killed_pids=set(), failed_units=[],
    )
    by_pid = {o["pid"]: o["outcome"] for o in outcomes}
    assert by_pid == {
        100: "restarted", 900: "unaccounted", 101: "restarted", 901: "unaccounted"
    }


def test_named_profile_serve_does_not_match_gateway_profile_unit():
    """``hermes-gateway-work.service`` restarted must not credit the ``work``
    serve — the old substring match (``"work" in unit``) did exactly that."""
    outcomes = match_runtime_outcomes(
        _plan(_rt("work", 101, supervisor="systemd"), _serve("work", 901)),
        restarted_services=["hermes-gateway-work.service"], relaunched_profiles=[],
        externally_supervised_profiles=[], killed_pids=set(), failed_units=[],
    )
    by_pid = {o["pid"]: o["outcome"] for o in outcomes}
    assert by_pid == {101: "restarted", 901: "unaccounted"}


def test_serve_reconciles_against_its_own_unit_vocabulary():
    """A serve IS covered when a ``hermes-serve*`` unit for its profile was
    restarted (or failed) — scope-qualified identities included."""
    outcomes = match_runtime_outcomes(
        _plan(_serve("default", 900), _serve("work", 901),
              _serve("ops", 902, kind="dashboard"), _serve("qa", 903)),
        restarted_services=["hermes-gateway", "user/hermes-serve",
                            "hermes-serve-work.service", "hermes-dashboard-ops"],
        relaunched_profiles=[], externally_supervised_profiles=[],
        killed_pids=set(), failed_units=["hermes-serve-qa.service"],
    )
    by_pid = {o["pid"]: o["outcome"] for o in outcomes}
    assert by_pid == {900: "restarted", 901: "restarted", 902: "restarted", 903: "failed"}
    # exact names: ``work`` must not claim ``hermes-serve-workbench``
    outcomes = match_runtime_outcomes(
        _plan(_serve("work", 901)),
        restarted_services=["hermes-serve-workbench.service"], relaunched_profiles=[],
        externally_supervised_profiles=[], killed_pids=set(), failed_units=[],
    )
    assert outcomes[0]["outcome"] == "unaccounted"


def test_serve_outcome_follows_incarnation_probe_when_provided():
    """With the (pid, create_time) survivor probe result, liveness decides:
    a pre-update serve that is gone was replaced (restarted); one still
    alive is unaccounted — even when a hermes-serve unit was restarted."""
    plan = _plan(_serve("default", 900), _serve("default", 901, kind="dashboard"))
    outcomes = match_runtime_outcomes(
        plan, restarted_services=["hermes-serve.service"], relaunched_profiles=[],
        externally_supervised_profiles=[], killed_pids=set(), failed_units=[],
        stale_serve_pids={900},
    )
    by_pid = {o["pid"]: o["outcome"] for o in outcomes}
    assert by_pid == {900: "unaccounted", 901: "restarted"}
    # killed pid still wins as "stopped"; probe None => fail closed
    outcomes = match_runtime_outcomes(
        plan, restarted_services=[], relaunched_profiles=[],
        externally_supervised_profiles=[], killed_pids={901}, failed_units=[],
        stale_serve_pids=None,
    )
    by_pid = {o["pid"]: o["outcome"] for o in outcomes}
    assert by_pid == {900: "unaccounted", 901: "stopped"}


def test_unaccounted_serve_report_names_serve_remedy_not_gateway_restart(capsys):
    outcomes = match_runtime_outcomes(
        _plan(_serve("default", 900)),
        restarted_services=["hermes-gateway"], relaunched_profiles=[],
        externally_supervised_profiles=[], killed_pids=set(), failed_units=[],
    )
    assert report_unaccounted_runtimes(outcomes) is True
    out = capsys.readouterr().out
    assert "serve [default] pid 900" in out
    assert "hermes-serve.service" in out
    assert "hermes gateway restart" not in out


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
