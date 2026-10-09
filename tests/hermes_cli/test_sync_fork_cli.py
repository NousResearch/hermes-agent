"""Tests for ``hermes sync-fork``'s CLI surface.

The merge logic is tested in test_sync_fork.py. What is pinned here is the part a
script depends on and a human reads: exit codes, whether stdout stays
machine-parsable under ``--json``, and that ``--dry-run`` cannot merge.

Exit codes carry meaning beyond success. ``--check`` returns 1 when a sync is
*available* — not when something failed — so a cron can branch on it; a plain run
returns 1 only when the sync itself did not complete. Confusing those two makes a
scheduled job either never act or act on nothing, so both are asserted
explicitly.

``sync_fork`` is the single seam. No test builds a repository or runs git; the
module's own suite covers that.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from hermes_cli import sync_fork
from hermes_cli import sync_fork_cli


def _args(**over):
    base = dict(repo="/tmp/whatever", upstream_ref="upstream/main",
                check=False, dry_run=False, json=False, ui=False)
    base.update(over)
    return SimpleNamespace(**base)


def _state(behind=0, ahead=0, dirty=False):
    return sync_fork.ForkState(behind=behind, ahead=ahead,
                               diverged=bool(behind and ahead), dirty=dirty)


def _plan(steps=None, prone=None):
    return sync_fork.SyncPlan(steps=steps or [], conflict_prone=prone or [])


def _patch(monkeypatch, state, plan=None, result=None):
    monkeypatch.setattr(sync_fork, "inspect", lambda *a, **k: state)
    monkeypatch.setattr(sync_fork, "plan", lambda *a, **k: plan or _plan())
    calls = []

    def fake_sync(cwd, upstream_ref=sync_fork.DEFAULT_UPSTREAM_REF, dry_run=False):
        calls.append({"dry_run": dry_run, "upstream_ref": upstream_ref})
        return result or sync_fork.SyncResult(ok=True, reason="Merged upstream in 1 step(s).")

    monkeypatch.setattr(sync_fork, "sync", fake_sync)
    return calls


# ---------------------------------------------------------------------------
# --json: the contract a script consumes
# ---------------------------------------------------------------------------

def test_json_prints_one_parsable_object_and_nothing_else(monkeypatch, capsys):
    _patch(monkeypatch, _state(behind=42, ahead=3), _plan(["a", "upstream/main"], ["a"]))
    rc = sync_fork_cli.run(_args(json=True))
    out = capsys.readouterr().out.strip()
    assert rc == 0
    assert len(out.splitlines()) == 1, "extra lines break a caller doing json.loads"
    payload = json.loads(out)
    assert payload == {"behind": 42, "ahead": 3, "diverged": True, "dirty": False,
                       "steps": 2, "conflict_prone": 1}


def test_json_never_merges(monkeypatch, capsys):
    """--json is a status read. If it could merge, a monitoring poll would
    mutate the checkout it was only supposed to observe."""
    calls = _patch(monkeypatch, _state(behind=42, ahead=3), _plan(["upstream/main"]))
    sync_fork_cli.run(_args(json=True))
    capsys.readouterr()
    assert calls == []


# ---------------------------------------------------------------------------
# --check: exit code as a signal, not as an error
# ---------------------------------------------------------------------------

def test_check_exits_1_when_a_sync_is_available(monkeypatch, capsys):
    _patch(monkeypatch, _state(behind=7, ahead=1), _plan(["upstream/main"]))
    rc = sync_fork_cli.run(_args(check=True))
    capsys.readouterr()
    assert rc == 1, "cron branches on this; 0 here means the job never acts"


def test_check_exits_0_when_current(monkeypatch, capsys):
    _patch(monkeypatch, _state())
    rc = sync_fork_cli.run(_args(check=True))
    capsys.readouterr()
    assert rc == 0


def test_check_never_merges_even_when_behind(monkeypatch, capsys):
    calls = _patch(monkeypatch, _state(behind=99, ahead=2), _plan(["upstream/main"]))
    sync_fork_cli.run(_args(check=True))
    capsys.readouterr()
    assert calls == [], "--check must be read-only"


def test_check_reports_availability_not_dirtiness(monkeypatch, capsys):
    """A dirty worktree with commits behind is still 'a sync is available' —
    the sync itself refuses later. Returning 0 here would hide the backlog."""
    _patch(monkeypatch, _state(behind=5, ahead=1, dirty=True), _plan(["upstream/main"]))
    rc = sync_fork_cli.run(_args(check=True))
    assert rc == 1
    assert "uncommitted" in capsys.readouterr().out.lower()


# ---------------------------------------------------------------------------
# --dry-run
# ---------------------------------------------------------------------------

def test_dry_run_passes_the_flag_through(monkeypatch, capsys):
    calls = _patch(monkeypatch, _state(behind=3, ahead=1), _plan(["upstream/main"]),
                   sync_fork.SyncResult(ok=True, reason="3 commit(s) behind; would merge in 1 step(s)."))
    rc = sync_fork_cli.run(_args(dry_run=True))
    capsys.readouterr()
    assert rc == 0
    assert calls and calls[0]["dry_run"] is True


def test_a_plain_run_does_not_pass_dry_run(monkeypatch, capsys):
    calls = _patch(monkeypatch, _state(behind=3, ahead=1), _plan(["upstream/main"]))
    sync_fork_cli.run(_args())
    capsys.readouterr()
    assert calls and calls[0]["dry_run"] is False


# ---------------------------------------------------------------------------
# a plain run
# ---------------------------------------------------------------------------

def test_up_to_date_returns_0_without_calling_sync(monkeypatch, capsys):
    calls = _patch(monkeypatch, _state())
    rc = sync_fork_cli.run(_args())
    assert rc == 0
    assert calls == []
    assert "up to date" in capsys.readouterr().out.lower()


def test_failed_sync_returns_1_and_lists_conflicted_paths(monkeypatch, capsys):
    """A conflict must be visibly a conflict: exit 1, and the files named on
    their own lines so an operator can scan them."""
    _patch(monkeypatch, _state(behind=9, ahead=2), _plan(["abc", "upstream/main"]),
           sync_fork.SyncResult(
               ok=False,
               reason="Conflict merging abc. The fork was restored to 44c9871f8.",
               conflicted_paths=["tools/delegate_tool.py", "agent/conversation_loop.py"]))
    rc = sync_fork_cli.run(_args())
    out = capsys.readouterr().out
    assert rc == 1
    assert "restored" in out
    assert "conflicted:" in out, "the paths appear with no label saying what they are"
    assert "tools/delegate_tool.py" in out
    assert "agent/conversation_loop.py" in out
    # Indented under their header, so a reader sees a list rather than prose.
    assert "    tools/delegate_tool.py" in out


def test_successful_sync_returns_0(monkeypatch, capsys):
    _patch(monkeypatch, _state(behind=2, ahead=1), _plan(["upstream/main"]),
           sync_fork.SyncResult(ok=True, reason="Merged upstream in 1 step(s)."))
    rc = sync_fork_cli.run(_args())
    capsys.readouterr()
    assert rc == 0


def test_the_upstream_ref_reaches_sync(monkeypatch, capsys):
    """A caller pointing at a different remote must not silently get the
    default; that would merge from somewhere they did not ask for."""
    calls = _patch(monkeypatch, _state(behind=1, ahead=1), _plan(["fork/main"]))
    sync_fork_cli.run(_args(upstream_ref="fork/main"))
    capsys.readouterr()
    assert calls and calls[0]["upstream_ref"] == "fork/main"


# ---------------------------------------------------------------------------
# the human-readable block
# ---------------------------------------------------------------------------

def test_a_diverged_fork_is_named_as_diverged(monkeypatch, capsys):
    """This is the state `hermes update` walks away from, so the output has to
    say so rather than leaving the operator to infer it."""
    _patch(monkeypatch, _state(behind=671, ahead=38), _plan(["a", "b", "upstream/main"], ["a", "b"]))
    sync_fork_cli.run(_args(check=True))
    out = capsys.readouterr().out
    assert "671" in out and "38" in out
    assert "diverged" in out.lower()
    assert "fast-forward" in out.lower(), "the reason update skips this is not explained"


def test_risky_steps_are_called_out_when_present(monkeypatch, capsys):
    _patch(monkeypatch, _state(behind=42, ahead=3), _plan(["a", "b", "upstream/main"], ["a", "b"]))
    sync_fork_cli.run(_args(check=True))
    out = capsys.readouterr().out
    assert "3 step" in out
    assert "2 upstream commit" in out


def test_no_risky_steps_says_so_rather_than_staying_silent(monkeypatch, capsys):
    _patch(monkeypatch, _state(behind=4, ahead=1), _plan(["upstream/main"], []))
    sync_fork_cli.run(_args(check=True))
    assert "no upstream commit touches" in capsys.readouterr().out.lower()


# ---------------------------------------------------------------------------
# --ui without a terminal
# ---------------------------------------------------------------------------

def test_ui_degrades_to_plain_output_without_a_terminal(monkeypatch, capsys):
    """--ui over a pipe must print status, not raise. A cron that inherits the
    flag from a shell alias would otherwise fail on a curses init error."""
    _patch(monkeypatch, _state(behind=5, ahead=1), _plan(["upstream/main"]))

    import curses

    def boom(_fn):
        raise curses.error("no terminal")

    monkeypatch.setattr(curses, "wrapper", boom)
    rc = sync_fork_cli.run(_args(ui=True))
    out = capsys.readouterr().out
    assert rc == 0
    assert "no usable terminal" in out
    assert "5 commit" in out, "the status the user asked for is missing"
# ---------------------------------------------------------------------------
# an unreadable state must not reach anything that branches on `behind`
# ---------------------------------------------------------------------------


def _unreadable(dirty=False):
    return sync_fork.ForkState(behind=0, ahead=0, diverged=False, dirty=dirty,
                               error="upstream ref 'upstream/main' does not resolve")


def test_json_reports_an_unreadable_state_as_an_error_object(monkeypatch, capsys):
    """The documented contract: an unresolvable ref answers {"error": ...}.

    It was documented in this command's own description and consumed by the
    fork-keeper cron, which extracts `.error` to decide whether to bail — but
    nothing ever produced the key, so that guard was dead code and the cron read
    behind=0 as "nothing to do".
    """
    monkeypatch.setattr(sync_fork, "inspect", lambda *a, **k: _unreadable())
    rc = sync_fork_cli.run(_args(json=True))
    payload = json.loads(capsys.readouterr().out.strip())

    assert rc == 2
    assert "does not resolve" in payload["error"]
    # No commit count at all: a consumer must not be able to read a number here.
    assert "behind" not in payload


def test_check_does_not_report_current_when_the_ref_is_unresolvable(monkeypatch, capsys):
    """--check returns 0 for "current" and 1 for "sync available". Neither is true
    when the position could not be read, so it must return the third value."""
    monkeypatch.setattr(sync_fork, "inspect", lambda *a, **k: _unreadable())
    rc = sync_fork_cli.run(_args(check=True))
    capsys.readouterr()
    assert rc == 2


def test_an_unreadable_state_never_reaches_the_merge(monkeypatch, capsys):
    """The action path branches on `behind`, which is 0 here for the wrong reason."""
    monkeypatch.setattr(sync_fork, "inspect", lambda *a, **k: _unreadable())

    def _boom(*a, **k):
        raise AssertionError("sync() was called with a state that could not be read")

    monkeypatch.setattr(sync_fork, "sync", _boom)
    monkeypatch.setattr(sync_fork, "plan", _boom)
    rc = sync_fork_cli.run(_args())
    capsys.readouterr()
    assert rc == 2


def test_the_human_headline_is_the_error_not_up_to_date(monkeypatch):
    """_render_state is shared with the curses screen, which calls inspect itself
    and so never passes the guard in _run. It has to refuse on its own."""
    lines = sync_fork_cli._render_state(_unreadable(), _plan())
    assert lines and "does not resolve" in lines[0]
    assert not any("Up to date" in line for line in lines)
