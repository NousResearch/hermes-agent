"""The interrupted-update finish banner must not claim unmoved code.

Issue #139518: a lazy finish (``venv_sync`` -> ``source_completion
--finish-update``) prints "✓ Code updated!" unconditionally, even when it
runs against the exact tree the last completed install/update stamped —
only the tail (dependencies, products, maintenance) is owed, the checkout
never moved. Together with the maintenance line's checkout version, users
parsed the banner pair as a downgrade to an older release. ``hermes
update``'s own HEAD gate refuses the same claim against a stale tree
(#79678); these tests pin the finish path to it.
"""

from hermes_cli.source_completion import _finish_announcement

MOVED = "\n✓ Code updated!"
UNMOVED = "\n✓ Finished an interrupted source update."
HEAD = "a" * 40


def _stamp(commit, **extra):
    return {"commit": commit, "dirty": False, **extra}


def _patch_state(monkeypatch, stamp, head=HEAD):
    monkeypatch.setattr("hermes_cli.steward.read_install_stamp", lambda root: stamp)
    monkeypatch.setattr(
        "hermes_cli.version_info._run_git", lambda repo_dir, *args: head
    )


def test_same_completed_tree_reports_finished_update(monkeypatch, tmp_path):
    """The stamped tree IS the current tree: the tail finished, the code did not move."""
    _patch_state(monkeypatch, _stamp(HEAD))
    assert _finish_announcement(tmp_path) == UNMOVED


def test_moved_tree_still_claims_code_updated(monkeypatch, tmp_path):
    """HEAD is past the stamped commit: the finish really did land new code."""
    _patch_state(monkeypatch, _stamp("b" * 40))
    assert _finish_announcement(tmp_path) == MOVED


def test_stampless_checkout_still_claims_code_updated(monkeypatch, tmp_path):
    """No completed tree on record: assume movement rather than understate it."""
    _patch_state(monkeypatch, {})
    assert _finish_announcement(tmp_path) == MOVED


def test_adopted_stamp_is_no_completion_evidence(monkeypatch, tmp_path):
    """Boot-time adoption stamps identity without building: never read as completed."""
    _patch_state(monkeypatch, _stamp(HEAD, adoptedAt="2026-10-16T00:00:00+00:00"))
    assert _finish_announcement(tmp_path) == MOVED


def test_dirty_stamped_tree_is_no_completion_evidence(monkeypatch, tmp_path):
    """A stamp taken on a dirty tree did not record a reproducible completed state."""
    _patch_state(monkeypatch, _stamp(HEAD, dirty=True))
    assert _finish_announcement(tmp_path) == MOVED


def test_unreadable_head_still_claims_code_updated(monkeypatch, tmp_path):
    """git failing to name HEAD must not understate a possibly moved tree."""
    _patch_state(monkeypatch, _stamp(HEAD), head=None)
    assert _finish_announcement(tmp_path) == MOVED
