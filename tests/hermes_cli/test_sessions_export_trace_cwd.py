"""``sessions export --format trace`` must describe the session, not the shell.

The trace format is Claude Code JSONL for the HF Agent Trace Viewer, and every
record carries ``cwd`` and ``gitBranch``. Those describe the workspace the
session ran in — which the session row already records — so reading them off
the exporting process mislabels the transcript.
"""

import json
import os
import sys

import pytest


SESSION_CWD = "/w/some-project"
SESSION_BRANCH = "feature/trace-cwd"


@pytest.fixture()
def exported_session(tmp_path):
    """A real session row with a recorded cwd + branch, and its export path."""
    from hermes_state import SessionDB

    sid = "20260817_090000_tracec"
    db = SessionDB()
    try:
        db.create_session(sid, source="cli", cwd=SESSION_CWD)
        db.append_message(sid, "user", "where am I")
        db.append_message(sid, "assistant", "in the project")
        db.update_session_cwd(
            sid,
            SESSION_CWD,
            git_branch=SESSION_BRANCH,
            git_repo_root=SESSION_CWD,
        )
    finally:
        db.close()
    return sid, tmp_path / "trace.jsonl"


def _export(monkeypatch, sid, out, extra=()):
    import hermes_cli.main as main_mod

    monkeypatch.setattr(
        sys, "argv",
        ["hermes", "sessions", "export", str(out),
         "--format", "trace", "--session-id", sid, *extra],
    )
    main_mod.main()
    assert out.exists(), "trace export wrote no file"
    return [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_trace_records_carry_the_session_cwd(monkeypatch, exported_session):
    sid, out = exported_session
    records = _export(monkeypatch, sid, out)

    assert records, "no trace records emitted"
    stamped = {r["cwd"] for r in records}
    assert stamped == {SESSION_CWD}, (
        f"trace stamped {stamped} instead of the session's cwd {SESSION_CWD!r} — "
        "the viewer attributes the transcript to the wrong project"
    )
    assert os.getcwd() not in stamped, (
        "the exporting process's directory leaked into another session's trace"
    )


def test_trace_records_carry_the_recorded_branch(monkeypatch, exported_session):
    sid, out = exported_session
    records = _export(monkeypatch, sid, out)

    assert {r["gitBranch"] for r in records} == {SESSION_BRANCH}, (
        "gitBranch was not taken from the session row; it is recorded at "
        "session start precisely because the checkout's branch moves on"
    )


def test_bulk_export_does_not_probe_git_per_session(monkeypatch, exported_session):
    """The recorded branch is authoritative, so no subprocess should run.

    Probing would be both slower (one `git rev-parse` per exported session)
    and wrong (it reports the branch checked out now, not the one the session
    ran on).
    """
    from agent import trace_upload

    def _forbidden(cwd):
        raise AssertionError(f"git branch probe ran for {cwd!r} during export")

    monkeypatch.setattr(trace_upload, "_probe_git_branch", _forbidden)

    sid, out = exported_session
    records = _export(monkeypatch, sid, out)
    assert records


def test_a_session_without_a_recorded_branch_exports_an_empty_one(
    monkeypatch, tmp_path
):
    """No branch on the row is reported as unknown, not guessed from the shell."""
    from agent import trace_upload
    from hermes_state import SessionDB

    sid = "20260817_090100_nobranch"
    db = SessionDB()
    try:
        db.create_session(sid, source="cli", cwd=SESSION_CWD)
        db.append_message(sid, "user", "hi")
    finally:
        db.close()

    monkeypatch.setattr(
        trace_upload, "_probe_git_branch",
        lambda cwd: (_ for _ in ()).throw(AssertionError("probed anyway")),
    )
    records = _export(monkeypatch, sid, tmp_path / "t.jsonl")

    assert {r["gitBranch"] for r in records} == {""}
    assert {r["cwd"] for r in records} == {SESSION_CWD}
