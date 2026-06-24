"""Unit tests for scripts/improvement_queue.py — the approval gate.

Covers the full proposal lifecycle: create -> pending_review, list/show,
approve (writes target + an audit backup), and reject (target untouched).
"""

from pathlib import Path

import pytest

import scripts.improvement_queue as queue


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return home


def _make_proposal(tmp_path, *, target_text, proposed_text):
    target = tmp_path / "PREFERENCES.md"
    target.write_text(target_text, encoding="utf-8")
    proposed = tmp_path / "draft.md"
    proposed.write_text(proposed_text, encoding="utf-8")
    rc = queue.main([
        "create", "--target", str(target), "--proposed-file", str(proposed),
        "--source", "identity-reflection", "--summary", "prefer terse replies",
    ])
    return rc, target


def test_create_lands_pending_review(hermes_home, tmp_path, capsys):
    rc, _ = _make_proposal(tmp_path, target_text="old\n", proposed_text="old\nnew line\n")
    assert rc == 0

    capsys.readouterr()
    assert queue.main(["list", "--status", "pending_review"]) == 0
    out = capsys.readouterr().out
    assert "0001" in out
    assert "identity-reflection" in out


def test_digest_is_silent_when_no_pending_proposals(hermes_home, capsys):
    assert queue.main(["digest"]) == 0
    assert capsys.readouterr().out == ""


def test_digest_surfaces_pending_identity_proposals(hermes_home, tmp_path, capsys):
    rc, _ = _make_proposal(
        tmp_path, target_text="old\n", proposed_text="old\nearned: terse\n")
    assert rc == 0

    capsys.readouterr()
    assert queue.main(["digest"]) == 0
    out = capsys.readouterr().out
    assert "Verdict — 1 identity proposal(s) need review." in out
    assert "0001" in out
    assert "prefer terse replies" in out
    assert "approve 0001" in out


def test_create_identical_content_is_refused(hermes_home, tmp_path):
    rc, _ = _make_proposal(tmp_path, target_text="same\n", proposed_text="same\n")
    assert rc == 2


def test_approve_writes_target_and_audit_backup(hermes_home, tmp_path):
    rc, target = _make_proposal(
        tmp_path, target_text="old\n", proposed_text="old\nearned: terse\n")
    assert rc == 0

    assert queue.main(["approve", "0001"]) == 0
    assert "earned: terse" in target.read_text(encoding="utf-8")

    backups = list((hermes_home / "identity" / "audit-backups").glob("*.bak"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "old\n"
    manifest = (hermes_home / "identity" / "audit-backups" / "MANIFEST.log").read_text()
    assert "proposal=0001" in manifest


def test_approve_twice_is_refused(hermes_home, tmp_path):
    _make_proposal(tmp_path, target_text="old\n", proposed_text="old\nx\n")
    assert queue.main(["approve", "0001"]) == 0
    assert queue.main(["approve", "0001"]) == 2  # no longer pending


def test_reject_leaves_target_untouched(hermes_home, tmp_path):
    rc, target = _make_proposal(
        tmp_path, target_text="original\n", proposed_text="original\nrejected\n")
    assert rc == 0

    assert queue.main(["reject", "0001", "--reason", "not earned"]) == 0
    assert target.read_text(encoding="utf-8") == "original\n"
    # No audit backup should exist for a rejection.
    assert not list((hermes_home / "identity" / "audit-backups").glob("*.bak"))


def test_ids_increment(hermes_home, tmp_path):
    _make_proposal(tmp_path, target_text="a\n", proposed_text="a\nb\n")
    # Second proposal against a different target gets the next id.
    t2 = tmp_path / "SOUL.md"
    t2.write_text("soul\n", encoding="utf-8")
    d2 = tmp_path / "draft2.md"
    d2.write_text("soul\nclause\n", encoding="utf-8")
    assert queue.main([
        "create", "--target", str(t2), "--proposed-file", str(d2),
    ]) == 0
    ids = sorted(p.name for p in (hermes_home / "identity" / "queue").iterdir())
    assert ids == ["0001", "0002"]
