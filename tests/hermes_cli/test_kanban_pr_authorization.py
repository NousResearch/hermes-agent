"""Operator-scoped same-PR remediation authorization (``authorize-existing-pr``).

The ``active_pr`` respawn guard must stay exactly as it was unless an unexpired,
task-scoped grant names the exact PR; the grant is spent by the first claimed run.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban as kcli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_pr_authorization as pra

REPO = "acme/backend"
BRANCH = "ticket/p3-packaging-adapter"
PR = f"https://github.com/{REPO}/pull/100"


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    import hermes_cli.config as cfgmod
    import hermes_cli.profiles as profmod
    monkeypatch.setattr(profmod, "profile_exists", lambda name: True)
    monkeypatch.setattr(cfgmod, "load_config", lambda *a, **k: {})
    kb.init_db()
    return home


def _pr(state="open", merged=False, ref=BRANCH, repo=REPO, head_repo=None):
    return {"state": state, "merged": merged,
            "head": {"ref": ref, "repo": {"full_name": head_repo or repo}},
            "base": {"ref": "main", "repo": {"full_name": repo}}}


def _lookup(pr):
    calls = []

    def lookup(repo, number):
        calls.append((repo, number))
        return pr
    lookup.calls = calls
    return lookup


def _task_with_pr(conn, *, assignee="dev", url=PR):
    tid = kb.create_task(conn, title="remediate review", assignee=assignee)
    kb.add_comment(conn, tid, author=assignee, body=f"Opened {url} for review.")
    return tid


def _authorize(conn, tid, **over):
    kw = dict(pr_url=PR, repo=REPO, branch=BRANCH, reason="Authorized same-PR review remediation",
              operator="gelo", pr_lookup=_lookup(_pr()))
    kw.update(over)
    return pra.authorize_existing_pr(conn, tid, **kw)


def _state(conn, tid):
    task = tuple(conn.execute("SELECT * FROM tasks WHERE id = ?", (tid,)).fetchone())
    events = [tuple(r) for r in conn.execute("SELECT * FROM task_events WHERE task_id = ?", (tid,))]
    return task, events


def _grants(conn, tid):
    return [r for r in conn.execute(
        "SELECT * FROM task_events WHERE task_id = ? AND kind = ?", (tid, pra.AUTH_EVENT))]


def test_open_pr_without_authorization_stays_guarded(kanban_home):
    with kbc.connect() as conn:
        tid = _task_with_pr(conn)
        assert kbd.check_respawn_guard(conn, tid) == "active_pr"
        res = kbd.dispatch_once(conn, dry_run=True)
        assert dict(res.respawn_guarded).get(tid) == "active_pr"
        assert tid not in [s[0] for s in res.spawned]


def test_exact_authorization_permits_one_dispatch_then_expires(kanban_home):
    spawned = []
    with kbc.connect() as conn:
        tid = _task_with_pr(conn)
        _authorize(conn, tid)
        assert kbd.check_respawn_guard(conn, tid) is None

        res = kbd.dispatch_once(conn, spawn_fn=lambda task, ws, *a: spawned.append(task.id) or None)
        assert spawned == [tid]
        assert tid in [s[0] for s in res.spawned]
        run_id = kb.get_task(conn, tid).current_run_id
        consumed = conn.execute(
            "SELECT run_id, payload FROM task_events WHERE task_id = ? AND kind = ?",
            (tid, pra.CONSUMED_EVENT)).fetchall()
        assert [r["run_id"] for r in consumed] == [run_id]

        # The run ends (crash -> ready): the spent grant must not lift the guard again.
        assert kb.reclaim_task(conn, tid, reason="worker died")
        assert kb.get_task(conn, tid).status == "ready"
        assert pra.active_authorization(conn, tid) is None
        assert kbd.check_respawn_guard(conn, tid) == "active_pr"
        res = kbd.dispatch_once(conn, spawn_fn=lambda task, ws, *a: spawned.append(task.id) or None)
        assert spawned == [tid]
        assert dict(res.respawn_guarded).get(tid) == "active_pr"


def test_grant_is_spent_by_one_run_even_when_claimed_outside_the_dispatcher(kanban_home):
    with kbc.connect() as conn:
        tid = _task_with_pr(conn)
        _authorize(conn, tid)
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        assert kb.reclaim_task(conn, tid, reason="released")
        assert kbd.check_respawn_guard(conn, tid) == "active_pr"


def test_concurrent_claims_yield_a_single_dispatch(kanban_home):
    """Two dispatchers both pass the guard; only one claim CAS wins, and the
    winning run alone expires the grant."""
    with kbc.connect() as a, kbc.connect() as b:
        tid = _task_with_pr(a)
        _authorize(a, tid)
        assert kbd.check_respawn_guard(a, tid) is None
        assert kbd.check_respawn_guard(b, tid) is None
        first = kb.claim_task(a, tid)
        second = kb.claim_task(b, tid)
        assert (first is None) != (second is None)
        assert pra.active_authorization(a, tid) is None
        assert pra.active_authorization(b, tid) is None


@pytest.mark.parametrize("override, message", [
    ({"pr_url": "https://github.com/acme/backend/pull/101"}, "never referenced"),
    ({"repo": "acme/frontend"}, "does not match PR URL"),
    ({"pr_url": "https://github.com/acme/frontend/pull/100", "repo": "acme/frontend"}, "never referenced"),
    ({"branch": "ticket/other"}, "head branch"),
    ({"pr_lookup": _lookup(_pr(state="closed"))}, "must be open"),
    ({"pr_lookup": _lookup(_pr(state="closed", merged=True))}, "must be open"),
    ({"pr_lookup": _lookup(_pr(head_repo="fork/backend"))}, "head comes from"),
    ({"pr_lookup": _lookup(_pr(repo="acme/other"))}, "base repository"),
])
def test_mismatched_or_closed_pr_is_refused_without_state_change(kanban_home, override, message):
    with kbc.connect() as conn:
        tid = _task_with_pr(conn)
        before = _state(conn, tid)
        with pytest.raises(pra.AuthorizationError, match=message):
            _authorize(conn, tid, **override)
        assert _state(conn, tid) == before
        assert kbd.check_respawn_guard(conn, tid) == "active_pr"


def test_task_branch_must_match_pr_branch(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="wt", assignee="dev", workspace_kind="worktree",
                             branch_name="wt/elsewhere")
        kb.add_comment(conn, tid, author="dev", body=f"PR: {PR}")
        before = _state(conn, tid)
        with pytest.raises(pra.AuthorizationError, match="task branch"):
            _authorize(conn, tid)
        assert _state(conn, tid) == before


def test_lookup_failure_fails_closed(kanban_home):
    def boom(repo, number):
        raise subprocess.CalledProcessError(1, ["gh"])
    with kbc.connect() as conn:
        tid = _task_with_pr(conn)
        before = _state(conn, tid)
        with pytest.raises(pra.AuthorizationError, match="could not verify"):
            _authorize(conn, tid, pr_lookup=boom)
        assert _state(conn, tid) == before


def test_authorization_cannot_be_reused_by_another_task(kanban_home):
    with kbc.connect() as conn:
        owner = _task_with_pr(conn)
        other = _task_with_pr(conn)
        _authorize(conn, owner)
        assert kbd.check_respawn_guard(conn, other) == "active_pr"
        assert pra.active_authorization(conn, other) is None
        res = kbd.dispatch_once(conn, dry_run=True)
        assert dict(res.respawn_guarded).get(other) == "active_pr"
        assert owner in [s[0] for s in res.spawned]


def test_a_different_pr_url_in_the_window_keeps_the_guard(kanban_home):
    with kbc.connect() as conn:
        tid = _task_with_pr(conn)
        _authorize(conn, tid)
        kb.add_comment(conn, tid, author="dev", body="Also opened https://github.com/acme/backend/pull/102")
        assert kbd.check_respawn_guard(conn, tid) == "active_pr"


def test_audit_event_records_every_required_field(kanban_home):
    with kbc.connect() as conn:
        tid = _task_with_pr(conn)
        lookup = _lookup(_pr())
        _authorize(conn, tid, pr_lookup=lookup)
        assert lookup.calls == [(REPO, 100)]
        (row,) = _grants(conn, tid)
        payload = kb._json_or(row["payload"], {})
        assert payload["operator"] == "gelo"
        assert payload["task_id"] == tid
        assert payload["pr_url"] == PR
        assert payload["repo"] == REPO
        assert payload["branch"] == BRANCH
        assert payload["reason"] == "Authorized same-PR review remediation"
        assert isinstance(payload["authorized_at"], int) and payload["authorized_at"] > 0
        assert row["created_at"] > 0


@pytest.mark.parametrize("task_id, override", [
    ("", {}),
    (" padded ", {}),
    ("t_missing", {}),
    (None, {"pr_url": ""}),
    (None, {"pr_url": "https://github.com/acme/backend/pull/100/files"}),
    (None, {"pr_url": "http://github.com/acme/backend/pull/100"}),
    (None, {"pr_url": "https://github.com/acme/backend/pull/0"}),
    (None, {"pr_url": "https://gitlab.com/acme/backend/pull/100"}),
    (None, {"repo": ""}),
    (None, {"repo": "acme"}),
    (None, {"branch": ""}),
    (None, {"branch": "-x"}),
    (None, {"branch": "a b"}),
    (None, {"branch": "a..b"}),
    (None, {"reason": "  "}),
    (None, {"operator": ""}),
])
def test_invalid_or_missing_identifiers_fail_safely(kanban_home, task_id, override):
    with kbc.connect() as conn:
        tid = _task_with_pr(conn)
        lookup = _lookup(_pr())
        before = _state(conn, tid)
        with pytest.raises(pra.AuthorizationError):
            _authorize(conn, tid if task_id is None else task_id, pr_lookup=lookup, **override)
        assert lookup.calls == []
        assert _state(conn, tid) == before
        assert kbd.check_respawn_guard(conn, tid) == "active_pr"


def test_only_a_read_only_pr_lookup_reaches_github(kanban_home, monkeypatch):
    """No duplicate PR, no close/merge/rebase/retarget: the default lookup is one GET."""
    seen = []

    class _Done:
        stdout = '{"state": "open", "merged": false, "head": {"ref": "%s", "repo": {"full_name": "%s"}},' \
                 ' "base": {"ref": "main", "repo": {"full_name": "%s"}}}' % (BRANCH, REPO, REPO)

    def fake_run(cmd, *a, **k):
        seen.append(list(cmd))
        return _Done()
    monkeypatch.setattr(subprocess, "run", fake_run)
    with kbc.connect() as conn:
        tid = _task_with_pr(conn)
        _authorize(conn, tid, pr_lookup=None)
    assert seen == [["gh", "api", f"repos/{REPO}/pulls/100", "--hostname", "github.com"]]


def test_merge_and_completion_gates_are_untouched(kanban_home):
    """The grant writes one event and nothing on the task row: completion
    contract, status and review routing behave exactly as without it."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="gated", assignee="dev", completion_contract=PR)
        kb.add_comment(conn, tid, author="dev", body=f"Opened {PR}")
        task_before = _state(conn, tid)[0]
        _authorize(conn, tid)
        assert _state(conn, tid)[0] == task_before
        assert kb.get_task(conn, tid).completion_contract == PR
        # Review lane is unaffected by the grant: it never applied active_pr.
        assert kbd.check_respawn_guard(conn, tid, lane="review") is None


def test_cli_is_operator_only_and_reports_refusals(kanban_home, monkeypatch, capsys):
    with kbc.connect() as conn:
        tid = _task_with_pr(conn)
    args = argparse.Namespace(task_id=tid, pr=PR, repo=REPO, branch=BRANCH, reason="fix review")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker")
    assert kcli._cmd_authorize_existing_pr(args) == 1
    assert "operator-only" in capsys.readouterr().err
    monkeypatch.delenv("HERMES_KANBAN_TASK")
    monkeypatch.setattr(pra, "github_pr_lookup", lambda repo, number: _pr(state="closed"))
    assert kcli._cmd_authorize_existing_pr(args) == 1
    assert "must be open" in capsys.readouterr().err
    monkeypatch.setattr(pra, "github_pr_lookup", lambda repo, number: _pr())
    assert kcli._cmd_authorize_existing_pr(args) == 0
    with kbc.connect() as conn:
        (row,) = _grants(conn, tid)
        assert kbd.check_respawn_guard(conn, tid) is None
