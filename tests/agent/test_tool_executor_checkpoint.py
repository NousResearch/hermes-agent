"""Regression tests for pre-mutation checkpointing of file-editing tools.

Focuses on the V4A multi-file patch gap: a ``patch(mode="patch", patch=...)``
call carries its real targets in the patch BODY, not the ``path`` arg, so a
checkpoint guard that keyed off ``path`` alone silently skipped the snapshot for
the single highest-blast-radius operation the agent has (a multi-file patch that
can overwrite or ``*** Delete File:`` many files with nothing to roll back to).

These tests exercise the shared helper directly with a fake checkpoint manager,
so they never touch a real git repo or ``~/.hermes``.
"""

from __future__ import annotations

from agent.tool_executor import _checkpoint_before_file_mutation


class _FakeCheckpointMgr:
    """Records ensure_checkpoint calls; maps a path to its working dir by parent."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.checkpoints: list[tuple[str, str]] = []  # (working_dir, reason)

    def get_working_dir_for_path(self, file_path: str) -> str:
        # Strip the filename so two files in the same dir collapse to one dir,
        # mirroring the real resolver's dir-level granularity.
        return file_path.rsplit("/", 1)[0] if "/" in file_path else "."

    def ensure_checkpoint(self, working_dir: str, reason: str = "auto") -> bool:
        # Mirror the real dedupe: one snapshot per working dir.
        if any(wd == working_dir for wd, _ in self.checkpoints):
            return False
        self.checkpoints.append((working_dir, reason))
        return True


class _FakeAgent:
    def __init__(self, enabled: bool = True):
        self._checkpoint_mgr = _FakeCheckpointMgr(enabled=enabled)


_PATCH_BODY = (
    "*** Begin Patch\n"
    "*** Update File: pkg/alpha.py\n"
    "@@\n"
    "-old\n"
    "+new\n"
    "*** Delete File: pkg/beta.py\n"
    "*** Add File: other/gamma.py\n"
    "+hello\n"
    "*** End Patch\n"
)


def test_v4a_patch_without_path_arg_still_checkpoints_each_target():
    """The regression: a V4A patch has no ``path`` arg, but every body target
    (across distinct working dirs) must still be checkpointed before it runs."""
    agent = _FakeAgent()
    args = {"mode": "patch", "patch": _PATCH_BODY}  # note: no "path" key at all

    _checkpoint_before_file_mutation(agent, "patch", args)

    dirs = {wd for wd, _ in agent._checkpoint_mgr.checkpoints}
    # pkg/alpha.py + pkg/beta.py collapse to "pkg"; other/gamma.py -> "other".
    assert dirs == {"pkg", "other"}
    assert all(reason == "before patch" for _, reason in agent._checkpoint_mgr.checkpoints)


def test_v4a_patch_single_repo_snapshots_once():
    """A multi-file patch confined to one working dir takes exactly one snapshot
    (ensure_checkpoint dedupes per dir)."""
    agent = _FakeAgent()
    body = (
        "*** Begin Patch\n"
        "*** Update File: pkg/a.py\n+x\n"
        "*** Update File: pkg/b.py\n+y\n"
        "*** End Patch\n"
    )
    _checkpoint_before_file_mutation(agent, "patch", {"mode": "patch", "patch": body})

    assert agent._checkpoint_mgr.checkpoints == [("pkg", "before patch")]


def test_replace_mode_patch_checkpoints_path():
    agent = _FakeAgent()
    _checkpoint_before_file_mutation(
        agent, "patch", {"mode": "replace", "path": "pkg/a.py"}
    )
    assert agent._checkpoint_mgr.checkpoints == [("pkg", "before patch")]


def test_write_file_checkpoints_path():
    agent = _FakeAgent()
    _checkpoint_before_file_mutation(agent, "write_file", {"path": "pkg/a.py"})
    assert agent._checkpoint_mgr.checkpoints == [("pkg", "before write_file")]


def test_patch_with_unparseable_body_takes_no_checkpoint():
    """No targets -> no snapshot, and definitely no crash."""
    agent = _FakeAgent()
    _checkpoint_before_file_mutation(agent, "patch", {"mode": "patch", "patch": ""})
    assert agent._checkpoint_mgr.checkpoints == []


def test_checkpoint_failure_never_propagates():
    """ensure_checkpoint raising must not bubble up and break tool execution."""

    class _BoomMgr(_FakeCheckpointMgr):
        def ensure_checkpoint(self, working_dir: str, reason: str = "auto") -> bool:
            raise RuntimeError("git exploded")

    agent = _FakeAgent()
    agent._checkpoint_mgr = _BoomMgr()
    # Must not raise.
    _checkpoint_before_file_mutation(agent, "patch", {"mode": "patch", "patch": _PATCH_BODY})
