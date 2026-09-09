from __future__ import annotations

import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest


_SPEC = importlib.util.spec_from_file_location(
    "hermes_pr_bridge", Path(__file__).parents[2] / "scripts/ci/hermes_pr_bridge.py"
)
assert _SPEC and _SPEC.loader
bridge = importlib.util.module_from_spec(_SPEC)
sys.modules["hermes_pr_bridge"] = bridge
_SPEC.loader.exec_module(bridge)


SHA = "a" * 40
BASE = "b" * 40
STALE = "c" * 40


class RecordingRunner:
    def __init__(self, archive: Path | None = None, pr: dict[str, object] | None = None):
        self.calls: list[tuple[str, ...]] = []
        self.archive = archive
        self.pr = pr

    def run(self, argv: tuple[str, ...]) -> bridge.CommandResult:
        self.calls.append(argv)
        if argv[0:2] == ("gh", "api") and argv[2].startswith("repos/test/repo/pulls/"):
            return bridge.CommandResult(0, json.dumps(self.pr or _pr_payload()), "")
        if argv[:3] == ("gh", "api", "--output"):
            assert self.archive is not None
            Path(argv[3]).write_bytes(self.archive.read_bytes())
            return bridge.CommandResult(0, "", "")
        if argv[0] == "hermes":
            return bridge.CommandResult(0, json.dumps({"id": "task-1"}), "")
        raise AssertionError(argv)


def _pr_payload(*, head_sha: str = SHA) -> dict[str, object]:
    return {
        "state": "open",
        "base": {"sha": BASE, "repo": {"full_name": "test/repo"}},
        "head": {"sha": head_sha, "repo": {"full_name": "test/repo"}},
        "labels": [{"name": "hermes:review"}],
    }


def _archive(path: Path, member_name: str = "repo/file.txt") -> Path:
    with tarfile.open(path, "w:gz") as handle:
        info = tarfile.TarInfo(member_name)
        content = b"case file\n"
        info.size = len(content)
        handle.addfile(info, io.BytesIO(content))
    return path


def test_rejects_stale_workflow_head_before_snapshot_or_task(tmp_path: Path) -> None:
    runner = RecordingRunner()
    event = {"workflow_run": {"event": "pull_request", "head_sha": STALE, "pull_requests": [{"number": 7, "head_sha": STALE}]}}
    with pytest.raises(bridge.BridgeError, match="stale"):
        bridge.build_case(
            event=event, repository="test/repo", board="repairs", assignee="reviewer",
            snapshot_root=tmp_path, runner=runner,
        )
    assert len(runner.calls) == 1


def test_creates_exact_idempotent_blocked_case(tmp_path: Path) -> None:
    archive = _archive(tmp_path / "source.tar.gz")
    runner = RecordingRunner(archive)
    event = {"workflow_run": {"event": "pull_request", "head_sha": SHA, "pull_requests": [{"number": 7, "head_sha": SHA}]}}
    result = bridge.build_case(
        event=event, repository="test/repo", board="repairs", assignee="reviewer",
        snapshot_root=tmp_path, runner=runner,
    )
    assert result["status"] == "queued"
    assert result["head_sha"] == SHA
    task_call = next(call for call in runner.calls if call[0] == "hermes")
    assert "--initial-status" in task_call and task_call[task_call.index("--initial-status") + 1] == "blocked"
    assert any("github-pr-bridge:v1:test/repo:7:" in argument for argument in task_call)
    manifest = next(tmp_path.glob("test-repo-7-*-review/.hermes-pr-bridge.json"))
    assert json.loads(manifest.read_text())["head_sha"] == SHA


def test_unlabeled_workflow_run_is_a_successful_noop(tmp_path: Path) -> None:
    archive = _archive(tmp_path / "source.tar.gz")
    unlabeled = _pr_payload()
    unlabeled["labels"] = []
    runner = RecordingRunner(archive, unlabeled)
    event = {"workflow_run": {"event": "pull_request", "head_sha": SHA, "pull_requests": [{"number": 7, "head_sha": SHA}]}}
    result = bridge.build_case(
        event=event, repository="test/repo", board="repairs", assignee="reviewer",
        snapshot_root=tmp_path, runner=runner,
    )
    assert result["status"] == "ignored"
    assert not any(call[0] == "hermes" for call in runner.calls)


def test_rejects_symlink_archive_members(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        info = tarfile.TarInfo("repo/link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        handle.addfile(info)
    with pytest.raises(bridge.BridgeError, match="unsupported"):
        bridge.extract_snapshot(archive, tmp_path / "out")
