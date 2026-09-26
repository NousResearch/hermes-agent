"""Regression coverage for #123370: network fetches precede Windows gateway pause."""

from types import SimpleNamespace

import pytest

from hermes_cli import main, update_cmd


def _result(returncode=0, stderr=""):
    return SimpleNamespace(returncode=returncode, stdout="", stderr=stderr)


def test_fetch_retry_uses_exponential_backoff_for_transient_failures(monkeypatch):
    results = iter([
        _result(128, "fatal: unable to access: Could not resolve host: github.com"),
        _result(128, "fatal: unable to access: HTTP 503"),
        _result(0),
    ])
    calls = []
    sleeps = []

    def run(git_cmd, args, **kwargs):
        calls.append((list(git_cmd), list(args), kwargs))
        return next(results)

    monkeypatch.setattr(update_cmd, "_git_run", run)
    monkeypatch.setattr(update_cmd._time, "sleep", sleeps.append)

    result = update_cmd._fetch_updates_with_retry(["git"], ["fetch", "origin", "main"])

    assert result.returncode == 0
    assert len(calls) == 3
    assert all(call[2] == {"network": True} for call in calls)
    assert sleeps == [1.0, 2.0]


def test_fetch_retry_leaves_bounded_timeout_to_transport_owner(monkeypatch):
    """rc=124 has a dedicated HTTP/1.1 recovery PR; do not duplicate it here."""
    calls, sleeps = [], []
    monkeypatch.setattr(
        update_cmd,
        "_git_run",
        lambda *a, **k: (
            calls.append((a, k))
            or _result(124, "git fetch timed out after 300s with no response from the remote")
        ),
    )
    monkeypatch.setattr(update_cmd._time, "sleep", sleeps.append)

    result = update_cmd._fetch_updates_with_retry(["git"], ["fetch", "origin", "main"])

    assert result.returncode == 124
    assert len(calls) == 1
    assert sleeps == []


def test_fetch_retry_does_not_retry_rate_limits_or_auth(monkeypatch):
    for stderr in (
        "error: RPC failed; HTTP 429 curl 22 The requested URL returned error: 429",
        "fatal: Authentication failed for 'https://github.com/private/repo.git/'",
    ):
        calls = []
        sleeps = []
        monkeypatch.setattr(
            update_cmd, "_git_run",
            lambda *a, _calls=calls, _stderr=stderr, **k: (
                _calls.append((a, k)) or _result(128, _stderr)
            ),
        )
        monkeypatch.setattr(update_cmd._time, "sleep", sleeps.append)

        result = update_cmd._fetch_updates_with_retry(["git"], ["fetch", "origin", "main"])

        assert result.returncode == 128
        assert len(calls) == 1
        assert sleeps == []


def _patch_update_preflight(monkeypatch, tmp_path, events, fetch_results):
    opts = update_cmd._UpdateOptions(
        pre_update_version=None,
        gw_input_fn=None,
        assume_yes=True,
        keep_stash=False,
        switch_branch=False,
        discard_local_changes=False,
    )
    monkeypatch.setattr(update_cmd, "_resolve_update_options", lambda *_: opts)
    monkeypatch.setattr(update_cmd, "_begin_update_receipt_and_plan", lambda *_: None)
    monkeypatch.setattr(update_cmd, "_record_pre_update_backup_outcome", lambda *_: None)
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(main, "_run_pre_update_backup", lambda *_: None)
    monkeypatch.setattr(main, "_desktop_packaged_executable", lambda *_: None)
    monkeypatch.setattr(main, "_desktop_dist_exists", lambda *_: False)
    monkeypatch.setattr(main, "_resolve_update_branch", lambda *_: "main")
    monkeypatch.setattr(main, "_warn_orphaned_update_autostashes", lambda *_: None)
    monkeypatch.setattr(update_cmd, "_source_update_channel", lambda *_: "main")
    monkeypatch.setattr(update_cmd, "_prepare_git_command", lambda: (False, ["git"], False))
    monkeypatch.setattr(
        update_cmd,
        "_source_completion_request",
        lambda opts, plan, snapshot, windows, desktop, gateway: {"windows_resume": windows},
    )
    monkeypatch.setattr(update_cmd._time, "sleep", lambda seconds: events.append(("sleep", seconds)))

    import hermes_cli.gitlock as gitlock

    monkeypatch.setattr(gitlock, "clear_stale_git_locks", lambda *_: [])
    monkeypatch.setattr(gitlock, "clear_stale_tmp_packs", lambda *_: [])
    monkeypatch.setattr(gitlock, "repair_broken_shallow_boundaries", lambda *_: 0)
    monkeypatch.setattr(gitlock, "prune_stale_shallow_grafts", lambda *_: 0)

    results = iter(fetch_results)

    def git_run(git_cmd, args, **kwargs):
        assert args[:1] == ["fetch"], args
        events.append(("fetch", len([e for e in events if e[0] == "fetch"]) + 1))
        return next(results)

    monkeypatch.setattr(update_cmd, "_git_run", git_run)


def test_failed_fetch_never_pauses_windows_gateway(monkeypatch, tmp_path):
    events = []
    _patch_update_preflight(
        monkeypatch,
        tmp_path,
        events,
        [
            _result(128, "fatal: unable to access: Could not resolve host: github.com"),
            _result(128, "fatal: unable to access: Could not resolve host: github.com"),
            _result(128, "fatal: unable to access: Could not resolve host: github.com"),
        ],
    )
    monkeypatch.setattr(
        main,
        "_pause_windows_gateways_for_update",
        lambda: events.append(("pause", None)) or {"resume_needed": True},
    )
    resumed = []
    monkeypatch.setattr(main, "_resume_windows_gateways_after_update", resumed.append)

    args = SimpleNamespace(branch="main", channel=None)
    with pytest.raises(SystemExit) as exc:
        update_cmd._cmd_update_impl(args, False)

    assert exc.value.code == 1
    assert [event[0] for event in events] == ["fetch", "sleep", "fetch", "sleep", "fetch"]
    assert resumed == []


def test_branch_probe_failure_still_never_pauses_windows_gateway(monkeypatch, tmp_path):
    """A read-only local probe failure must not turn into gateway downtime."""
    events = []
    _patch_update_preflight(monkeypatch, tmp_path, events, [_result(0)])
    monkeypatch.setattr(
        main,
        "_pause_windows_gateways_for_update",
        lambda: events.append(("pause", None)) or {"resume_needed": True},
    )

    class BranchProbeFailed(BaseException):
        pass

    def fail_branch_probe(*_args, **_kwargs):
        events.append(("branch-probe", None))
        raise BranchProbeFailed

    monkeypatch.setattr(update_cmd, "_current_branch_name", fail_branch_probe)
    args = SimpleNamespace(branch="main", channel=None)
    with pytest.raises(BranchProbeFailed):
        update_cmd._cmd_update_impl(args, False)

    assert events == [("fetch", 1), ("branch-probe", None)]


def test_gateway_pauses_only_after_fetch_succeeds_and_token_reaches_mutation(
    monkeypatch, tmp_path,
):
    events = []
    _patch_update_preflight(
        monkeypatch,
        tmp_path,
        events,
        [
            _result(128, "fatal: unable to access: Could not resolve host: github.com"),
            _result(0),
        ],
    )
    token = {"resume_needed": True}
    monkeypatch.setattr(
        main,
        "_pause_windows_gateways_for_update",
        lambda: events.append(("pause", None)) or token,
    )
    registrations = []
    monkeypatch.setattr("atexit.register", lambda fn, value: registrations.append((fn, value)))
    monkeypatch.setattr(update_cmd, "_current_branch_name", lambda *_a, **_k: "main")

    def prepare(*_args, **kwargs):
        events.append(("prepare", None))
        assert kwargs["_windows_gateway_resume"] is token
        return SimpleNamespace(commit_count=0)

    monkeypatch.setattr(update_cmd, "_prepare_checkout_for_update", prepare)
    completed = []

    def finish(_git, _branch, _current, _plan, *, gw_input_fn, completion_request):
        completed.append(dict(completion_request))

    monkeypatch.setattr(update_cmd, "_finish_already_up_to_date", finish)

    args = SimpleNamespace(branch="main", channel=None)
    update_cmd._cmd_update_impl(args, False)

    assert events == [
        ("fetch", 1),
        ("sleep", 1.0),
        ("fetch", 2),
        ("pause", None),
        ("prepare", None),
    ]
    assert completed == [{"windows_resume": token, "branch": "main"}]
    assert registrations and registrations[0][1] is token
