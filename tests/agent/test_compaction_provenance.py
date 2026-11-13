"""The compaction provenance block: harness facts, refreshed, never invented."""

from datetime import datetime, timedelta, timezone

import pytest

from agent import compaction_provenance as prov
from agent.compaction_provenance import (
    PROVENANCE_HEADING,
    append_provenance_section,
    detect_git_state,
    strip_provenance_section,
)

_STAMP = datetime(2026, 11, 9, 18, 4, 12, tzinfo=timezone(timedelta(hours=-8)))


@pytest.fixture(autouse=True)
def _isolated_git_cache():
    prov._clear_git_cache()
    yield
    prov._clear_git_cache()


def _fixed_git(monkeypatch, branch, head):
    monkeypatch.setattr(prov, "detect_git_state", lambda cwd: (branch, head))


def test_appends_block_with_workspace_facts(monkeypatch):
    _fixed_git(monkeypatch, "main", "abc1234")

    out = append_provenance_section("## Goal\nFix the flaky gateway timeout.", cwd="/work/repo", now=_STAMP)

    assert out.startswith("## Goal\nFix the flaky gateway timeout.")
    assert PROVENANCE_HEADING in out
    assert "2026-11-09T18:04:12-08:00" in out
    assert "- Working directory: /work/repo" in out
    assert "- Git branch: main (HEAD abc1234)" in out
    # The block must not claim more than it can prove.
    assert "may be stale" in out


def test_non_git_workspace_is_named_not_invented(monkeypatch):
    _fixed_git(monkeypatch, None, None)

    out = append_provenance_section("## Goal\nWrite the doc.", cwd="/work/notes", now=_STAMP)

    assert "- Git: not a git repository" in out
    assert "- Git branch:" not in out


def test_missing_workspace_degrades_to_unknown(monkeypatch):
    _fixed_git(monkeypatch, None, None)

    out = append_provenance_section("## Goal\nRecover the thread.", cwd="", now=_STAMP)

    assert "- Working directory: unknown" in out


def test_copied_block_is_refreshed_not_duplicated(monkeypatch):
    """The iterative path feeds the previous summary back, so the model may copy an old block forward."""
    _fixed_git(monkeypatch, "feature/x", "deadbee")
    stale = append_provenance_section("## Goal\nOlder work.", cwd="/work/repo", now=datetime(2026, 11, 1, 9, 0, tzinfo=timezone.utc))
    carried = f"## Goal\nOlder work.\n\n{stale.split(PROVENANCE_HEADING, 1)[1].join([PROVENANCE_HEADING, ''])}"
    carried = carried.rstrip() + "\n\n## Next Steps\n1. Re-run the suite.\n"

    out = append_provenance_section(carried, cwd="/work/repo", now=_STAMP)

    assert out.count(PROVENANCE_HEADING) == 1
    assert "2026-11-01" not in out
    assert "2026-11-09T18:04:12-08:00" in out
    assert "feature/x" in out
    # Content after the old block survives, and the fresh block is the last thing in the summary.
    assert "## Next Steps" in out
    assert out.rstrip().endswith("may be stale. Re-check with a tool call before relying on them.")


def test_strip_is_a_no_op_when_no_block_present():
    text = "## Goal\nNothing to strip."
    assert strip_provenance_section(text) == text


def test_detect_git_state_reads_a_real_repository(tmp_path):
    import shutil
    import subprocess

    if not shutil.which("git"):
        pytest.skip("git not installed")
    repo = tmp_path / "repo"
    repo.mkdir()
    run = lambda *a: subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=True)
    run("init")
    run("-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "--allow-empty", "-m", "one")

    branch, head = detect_git_state(str(repo))

    assert branch and branch != "HEAD"
    assert head and len(head) >= 7


def test_detect_git_state_outside_a_repository(tmp_path):
    assert detect_git_state(str(tmp_path)) == (None, None)


def test_provenance_reaches_the_summary_in_legacy_tail_mode(monkeypatch):
    """Provenance is a property of the summary, not of the lean tail policy, so legacy mode gets it too."""
    from agent.context_compressor import ContextCompressor

    _fixed_git(monkeypatch, "main", "abc1234")
    compressor = object.__new__(ContextCompressor)
    compressor.tail_mode = "legacy"

    out = compressor._augment_summary_lean("## Goal\nLegacy summary.", [])

    assert PROVENANCE_HEADING in out


def test_failure_inside_the_block_leaves_the_summary_intact(monkeypatch):
    def _boom(_summary, **_kwargs):
        raise RuntimeError("git exploded")

    _fixed_git(monkeypatch, "main", "abc1234")
    monkeypatch.setattr(prov, "_format_section", _boom)

    out = append_provenance_section("## Goal\nKeep me.", cwd="/work/repo", now=_STAMP)

    assert out == "## Goal\nKeep me."
