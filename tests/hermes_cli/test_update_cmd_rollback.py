"""`hermes update --rollback`: undo the last committed update from its receipt."""
import pytest

import hermes_cli.update_cmd_rollback as rollback_mod
from hermes_cli.update_cmd_rollback import (
    ROLLBACK_NOTHING,
    ROLLBACK_OK,
    ROLLBACK_REFUSED,
    _cmd_update_rollback,
    _rollback_target,
)

PRE_SHA = "a" * 40
OTHER_SHA = "b" * 40


def _record(outcome="success", sha=PRE_SHA, snapshot="20261020-100000-pre-update", **extra):
    steps = [{"name": "pre_update_backup", "ok": True, "detail": f"snapshot={snapshot}"}] if snapshot else []
    record = {
        "outcome": outcome,
        "started_at": "2026-10-20T10:00:00+00:00",
        "pre_update": {"sha": sha, "short_sha": sha[:12]},
        "steps": steps,
    }
    record.update(extra)
    return record


def test_no_receipt_refuses_with_the_reason(monkeypatch):
    import hermes_cli.update_receipt as receipt_mod

    monkeypatch.setattr(receipt_mod, "read_latest_receipt", lambda: None)
    with pytest.raises(RuntimeError, match="no update receipt yet"):
        _rollback_target()


def test_only_committed_updates_are_rollback_targets(monkeypatch):
    import hermes_cli.update_receipt as receipt_mod

    monkeypatch.setattr(receipt_mod, "read_latest_receipt", lambda: _record(outcome="failed"))
    with pytest.raises(RuntimeError, match="that committed"):
        _rollback_target()


def test_target_carries_pre_sha_and_snapshot_id(monkeypatch):
    import hermes_cli.update_receipt as receipt_mod

    monkeypatch.setattr(receipt_mod, "read_latest_receipt", lambda: _record())
    record, sha, snapshot_id = _rollback_target()
    assert sha == PRE_SHA
    assert snapshot_id == "20261020-100000-pre-update"
    assert record["outcome"] == "success"


def test_missing_snapshot_id_still_targets_the_commit(monkeypatch):
    import hermes_cli.update_receipt as receipt_mod

    monkeypatch.setattr(receipt_mod, "read_latest_receipt", lambda: _record(snapshot=None))
    _, sha, snapshot_id = _rollback_target()
    assert sha == PRE_SHA
    assert snapshot_id is None


def _patch_git(monkeypatch, *, head, dirty=""):
    calls = []

    class _Result:
        def __init__(self, returncode=0, stdout=""):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = ""

    def fake_git(root, *args):
        calls.append(args)
        if args[:2] == ("rev-parse", "HEAD"):
            return _Result(0, head + "\n")
        if args[:2] == ("status", "--porcelain"):
            return _Result(0, dirty)
        if args[:2] == ("reset", "--hard"):
            return _Result(0, "")
        if args == ("rev-parse", "--git-dir"):
            return _Result(0, ".git\n")
        return _Result(0, "")

    monkeypatch.setattr(rollback_mod, "_git", fake_git)
    return calls


def test_dirty_tree_refuses_before_touching_anything(tmp_path, monkeypatch, capsys):
    import hermes_cli.update_receipt as receipt_mod

    monkeypatch.setattr(receipt_mod, "read_latest_receipt", lambda: _record())
    (tmp_path / ".git").mkdir()  # look like a checkout; rev-parse covered by the fake
    _patch_git(monkeypatch, head=OTHER_SHA, dirty=" M hermes_cli/main.py\n")
    assert _cmd_update_rollback(tmp_path) == ROLLBACK_REFUSED
    assert "uncommitted changes" in capsys.readouterr().out


def test_already_on_pre_commit_is_a_noop(tmp_path, monkeypatch, capsys):
    import hermes_cli.update_receipt as receipt_mod

    monkeypatch.setattr(receipt_mod, "read_latest_receipt", lambda: _record())
    _patch_git(monkeypatch, head=PRE_SHA)
    assert _cmd_update_rollback(tmp_path) == ROLLBACK_NOTHING
    assert "nothing to roll back" in capsys.readouterr().out


def test_happy_path_resets_restores_and_resyncs(tmp_path, monkeypatch, capsys):
    import hermes_cli.backup as backup_mod
    import hermes_cli.update_cmd as cmd_mod
    import hermes_cli.venv_sync as venv_mod
    import hermes_cli.update_receipt as receipt_mod

    monkeypatch.setattr(receipt_mod, "read_latest_receipt", lambda: _record())
    git_calls = _patch_git(monkeypatch, head=OTHER_SHA)
    restored, synced, recorded = {}, {}, {}

    monkeypatch.setattr(backup_mod, "restore_quick_snapshot", lambda sid: restored.setdefault("sid", sid) or True)
    def fake_sync(root):
        synced["root"] = root
        return {"ok": True}

    monkeypatch.setattr(venv_mod, "sync", fake_sync)
    monkeypatch.setattr(cmd_mod, "_record_update_step", lambda *a: recorded.setdefault("args", a))

    assert _cmd_update_rollback(tmp_path) == ROLLBACK_OK
    assert ("reset", "--hard", PRE_SHA) in git_calls
    assert restored["sid"] == "20261020-100000-pre-update"
    assert synced["root"] == tmp_path
    out = capsys.readouterr().out
    assert "Rolled back" in out
