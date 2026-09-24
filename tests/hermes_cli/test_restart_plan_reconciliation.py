"""Plan-vs-execution reconciliation (#91277 Phase 2: restart via declared mechanism).

Pins:
- match_runtime_outcomes classifies every planned runtime against the restart
  phase's bookkeeping: restarted / stopped / failed / unaccounted.
- report_unaccounted_runtimes escalates (returns True) ONLY on unaccounted
  rows — the silent-miss tripwire.
"""

from hermes_cli.update_inventory import (
    RuntimeRecord,
    UpdatePlan,
    _restart_mechanism,
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


def test_launchd_default_gateway_restarted_via_ai_hermes_label():
    """macOS restart bookkeeping records ``ai.hermes.gateway``, which does not
    contain the substring ``hermes-gateway``. The default-profile gateway must
    still count as restarted — otherwise every Desktop update on launchd
    exits 1 after a successful kickstart (receipt outcome=partial, tripwire
    'never touched')."""
    outcomes = match_runtime_outcomes(
        _plan(_rt("default", 400, supervisor="launchd")),
        restarted_services=["ai.hermes.gateway"], relaunched_profiles=[],
        externally_supervised_profiles=[], killed_pids=set(), failed_units=[],
    )
    assert outcomes[0]["outcome"] == "restarted"
    assert report_unaccounted_runtimes(outcomes) is False


def test_launchd_named_profile_and_failed_label():
    restarted = match_runtime_outcomes(
        _plan(_rt("work", 401, supervisor="launchd")),
        restarted_services=["ai.hermes.gateway-work"], relaunched_profiles=[],
        externally_supervised_profiles=[], killed_pids=set(), failed_units=[],
    )
    assert restarted[0]["outcome"] == "restarted"
    failed = match_runtime_outcomes(
        _plan(_rt("default", 402, supervisor="launchd")),
        restarted_services=[], relaunched_profiles=[],
        externally_supervised_profiles=[], killed_pids=set(),
        failed_units=["ai.hermes.gateway"],
    )
    assert failed[0]["outcome"] == "failed"


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


def test_desktop_serve_deferral_requires_a_verified_alive_incarnation():
    """Desktop may defer only a serve the survivor probe confirmed alive."""
    desktop_serve = _serve("default", 900)
    desktop_serve.supervisor = "desktop"
    desktop_serve.restart_via = _restart_mechanism("desktop", "default")

    unknown = match_runtime_outcomes(
        _plan(desktop_serve), restarted_services=["hermes-serve.service"],
        relaunched_profiles=[], externally_supervised_profiles=[], killed_pids=set(),
        failed_units=[], stale_serve_pids=None,
    )
    assert unknown[0]["outcome"] == "unaccounted"

    alive = match_runtime_outcomes(
        _plan(desktop_serve), restarted_services=[], relaunched_profiles=[],
        externally_supervised_profiles=[], killed_pids=set(), failed_units=[],
        stale_serve_pids={900},
    )
    assert alive[0]["outcome"] == "deferred"

    gone = match_runtime_outcomes(
        _plan(desktop_serve), restarted_services=[], relaunched_profiles=[],
        externally_supervised_profiles=[], killed_pids=set(), failed_units=[],
        stale_serve_pids=set(),
    )
    assert gone[0]["outcome"] == "restarted"


def test_unaccounted_serve_report_names_serve_remedy_not_gateway_restart(capsys):
    outcomes = match_runtime_outcomes(
        _plan(_serve("default", 900)),
        restarted_services=["hermes-gateway"], relaunched_profiles=[],
        externally_supervised_profiles=[], killed_pids=set(), failed_units=[],
    )
    assert report_unaccounted_runtimes(outcomes) is True
    out = capsys.readouterr().out
    assert "serve [default] pid 900" in out
    assert "relaunch `hermes serve`" in out
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


def test_gateway_credited_by_successor_incarnation_not_service_name():
    """A service can serve a profile its own name does not encode: with a sticky
    active profile, the root-home ``ai.hermes.gateway`` LaunchAgent supervises the
    ``coder`` gateway, so the plan row (profile ``coder``) and the restarted label
    never match by name. The planned PID being gone while a gateway answers for the
    same profile is the evidence that has to credit it — otherwise every update
    exits 1 (receipt outcome=partial) after a successful launchd restart."""
    outcomes = match_runtime_outcomes(
        _plan(_rt("coder", 76508, supervisor="launchd")),
        restarted_services=["ai.hermes.gateway"], relaunched_profiles=[],
        externally_supervised_profiles=[], killed_pids=set(), failed_units=[],
        live_gateway_pids={"coder": {76796}},
    )
    assert outcomes[0]["outcome"] == "restarted"
    assert report_unaccounted_runtimes(outcomes) is False


def test_gateway_successor_credit_requires_a_live_replacement():
    """The tripwire keeps its teeth: no successor (the gateway was stopped and
    nothing replaced it) or the planned PID still answering (never restarted)
    stays unaccounted and escalates."""
    no_successor = match_runtime_outcomes(
        _plan(_rt("coder", 76508, supervisor="launchd")),
        restarted_services=["ai.hermes.gateway"], relaunched_profiles=[],
        externally_supervised_profiles=[], killed_pids=set(), failed_units=[],
        live_gateway_pids={},
    )
    assert no_successor[0]["outcome"] == "unaccounted"
    assert report_unaccounted_runtimes(no_successor) is True

    never_restarted = match_runtime_outcomes(
        _plan(_rt("coder", 76508, supervisor="launchd")),
        restarted_services=[], relaunched_profiles=[],
        externally_supervised_profiles=[], killed_pids=set(), failed_units=[],
        live_gateway_pids={"coder": {76508}},
    )
    assert never_restarted[0]["outcome"] == "unaccounted"

    other_profile_only = match_runtime_outcomes(
        _plan(_rt("coder", 76508, supervisor="launchd")),
        restarted_services=[], relaunched_profiles=[],
        externally_supervised_profiles=[], killed_pids=set(), failed_units=[],
        live_gateway_pids={"default": {76796}},
    )
    assert other_profile_only[0]["outcome"] == "unaccounted"

    empty_evidence = match_runtime_outcomes(
        _plan(_rt("coder", 76508, supervisor="launchd")),
        restarted_services=["ai.hermes.gateway"], relaunched_profiles=[],
        externally_supervised_profiles=[], killed_pids=set(), failed_units=[],
        live_gateway_pids={"coder": set()},
    )
    assert empty_evidence[0]["outcome"] == "unaccounted"


def test_missing_successor_evidence_is_logged(caplog):
    """A profile the fleet probe has no row for logs why reconciliation stayed on
    the name-matching path: the tripwire output reads identically whether the
    evidence was missing or the restart was genuinely missed."""
    import logging

    with caplog.at_level(logging.DEBUG, logger="hermes_cli.update_inventory"):
        outcomes = match_runtime_outcomes(
            _plan(_rt("coder", 76508, supervisor="launchd")),
            restarted_services=["ai.hermes.gateway"], relaunched_profiles=[],
            externally_supervised_profiles=[], killed_pids=set(), failed_units=[],
            live_gateway_pids={},
        )
    assert outcomes[0]["outcome"] == "unaccounted"
    assert "No post-restart gateway evidence for profile 'coder'" in caplog.text


def test_successor_evidence_never_outranks_stopped_or_failed():
    """Bookkeeping verdicts stay authoritative when the incarnation evidence is
    also passed: the successor branch is reached only after them, so a killed or
    name-failed gateway cannot be promoted to ``restarted``."""
    stopped = match_runtime_outcomes(
        _plan(_rt("coder", 76508, supervisor="launchd")),
        restarted_services=[], relaunched_profiles=[],
        externally_supervised_profiles=[], killed_pids={76508}, failed_units=[],
        live_gateway_pids={"coder": {76796}},
    )
    assert stopped[0]["outcome"] == "stopped"

    failed = match_runtime_outcomes(
        _plan(_rt("coder", 76508, supervisor="launchd")),
        restarted_services=[], relaunched_profiles=[],
        externally_supervised_profiles=[], killed_pids=set(),
        failed_units=["ai.hermes.gateway-coder"],
        live_gateway_pids={"coder": {76796}},
    )
    assert failed[0]["outcome"] == "failed"


def test_live_gateway_pids_from_fleet_skips_down_and_unusable_rows():
    """The fleet snapshot is the successor evidence. A ``down`` row carries the
    PRE-restart PID (nothing replaced it) so it must never count as a successor;
    rows without a usable profile/PID are skipped; a ``stale`` row is a live
    successor (the fleet matrix escalates staleness on its own)."""
    from hermes_cli.update_cmd_fleet import _live_gateway_pids_from_fleet

    rows = [
        {"profile": "coder", "pid": 76796, "state": "current"},
        {"profile": "coder", "pid": 76508, "state": "down"},
        {"profile": "default", "pid": 4242, "state": "stale"},
        {"profile": "default", "pid": None, "state": "unknown"},
        {"profile": "", "pid": 7, "state": "current"},
        {"profile": "researcher", "pid": "not-a-pid", "state": "current"},
    ]
    assert _live_gateway_pids_from_fleet(rows) == {"coder": {76796}, "default": {4242}}
    assert _live_gateway_pids_from_fleet([]) == {}


def test_one_successor_cannot_credit_two_planned_runtimes_same_profile():
    """The fleet probe publishes at most one row per profile, so one successor cannot
    say which of two planned same-profile gateways it replaced. Neither may be
    credited — otherwise an untouched sibling disappears behind a replacement that
    can only have replaced one of them, and the tripwire is suppressed."""
    outcomes = match_runtime_outcomes(
        _plan(_rt("coder", 76508, supervisor="launchd"), _rt("coder", 76795, supervisor="launchd")),
        restarted_services=[], relaunched_profiles=[], externally_supervised_profiles=[],
        killed_pids=set(), failed_units=[], live_gateway_pids={"coder": {76796}},
    )
    assert [o["outcome"] for o in outcomes] == ["unaccounted", "unaccounted"]
    assert report_unaccounted_runtimes(outcomes) is True


def test_resolved_orphan_sibling_does_not_block_successor_credit():
    """An orphan row the restart phase already killed has its verdict, so it must not
    make the surviving same-profile row's successor evidence look ambiguous — both
    runtimes are accounted for (one stopped, one restarted) and the wire stays quiet."""
    outcomes = match_runtime_outcomes(
        _plan(_rt("coder", 100, supervisor="launchd"), _rt("coder", 101, supervisor="launchd")),
        restarted_services=[], relaunched_profiles=[], externally_supervised_profiles=[],
        killed_pids={100}, failed_units=[], live_gateway_pids={"coder": {102}},
    )
    by_pid = {o["pid"]: o["outcome"] for o in outcomes}
    assert by_pid == {100: "stopped", 101: "restarted"}
    assert report_unaccounted_runtimes(outcomes) is False


def test_successor_credit_stays_per_profile_with_several_planned_runtimes():
    """The ambiguity guard is per profile: two profiles with one planned runtime each
    are both credited from their own successor."""
    outcomes = match_runtime_outcomes(
        _plan(_rt("coder", 76508, supervisor="launchd"), _rt("work", 401, supervisor="launchd")),
        restarted_services=[], relaunched_profiles=[], externally_supervised_profiles=[],
        killed_pids=set(), failed_units=[],
        live_gateway_pids={"coder": {76796}, "work": {402}},
    )
    assert [o["outcome"] for o in outcomes] == ["restarted", "restarted"]


def test_successor_evidence_is_gateway_only_and_stays_optional():
    """Serve/dashboard rows reconcile in their own vocabulary, and callers that
    pass no ``live_gateway_pids`` keep the bookkeeping-only verdict. Two non-gateway
    rows for the same profile must not make the gateway's evidence look ambiguous:
    the baseline count is per gateway kind."""
    outcomes = match_runtime_outcomes(
        _plan(_rt("coder", 76508, supervisor="launchd"),
              _serve("coder", 900), _serve("coder", 901, kind="dashboard")),
        restarted_services=[], relaunched_profiles=[],
        externally_supervised_profiles=[], killed_pids=set(), failed_units=[],
        live_gateway_pids={"coder": {76796}},
    )
    by_pid = {o["pid"]: o["outcome"] for o in outcomes}
    assert by_pid == {76508: "restarted", 900: "unaccounted", 901: "unaccounted"}

    without_probe = match_runtime_outcomes(
        _plan(_rt("coder", 76508, supervisor="launchd")),
        restarted_services=["ai.hermes.gateway"], relaunched_profiles=[],
        externally_supervised_profiles=[], killed_pids=set(), failed_units=[],
    )
    assert without_probe[0]["outcome"] == "unaccounted"
