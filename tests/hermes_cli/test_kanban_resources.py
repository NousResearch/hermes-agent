"""Behavioral tests for task/run-scoped resource leases."""

from __future__ import annotations

import json
import subprocess

from hermes_cli.kanban_resources import cleanup_owned_resources
from hermes_cli.kanban_resources import lease_for_run
from hermes_cli.kanban_resources import lease_from_path
from hermes_cli.kanban_resources import resource_environment


class DockerFake:
    def __init__(self):
        self.resources = {
            "container": {"owned-c": True, "other-c": False},
            "network": {"owned-n": True, "other-n": False},
            "volume": {"owned-v": True, "other-v": False},
        }
        self.commands: list[list[str]] = []

    def __call__(self, argv: list[str]) -> subprocess.CompletedProcess:
        self.commands.append(argv)
        kind = {
            "ps": "container",
            "network": "network",
            "volume": "volume",
        }.get(argv[1])
        if kind is not None and (
            (kind == "container" and argv[1:3] == ["ps", "-aq"])
            or (kind in {"network", "volume"} and argv[2:4] == ["ls", "-q"])
        ):
            owned = [
                name for name, is_owned in self.resources[kind].items() if is_owned
            ]
            return subprocess.CompletedProcess(
                argv, 0, "\n".join(owned) + ("\n" if owned else ""), ""
            )
        if argv[1:3] == ["rm", "-f"]:
            for name in argv[3:]:
                self.resources["container"].pop(name, None)
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[1:3] == ["network", "rm"]:
            for name in argv[3:]:
                self.resources["network"].pop(name, None)
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[1:3] == ["volume", "rm"]:
            for name in argv[3:]:
                self.resources["volume"].pop(name, None)
            return subprocess.CompletedProcess(argv, 0, "", "")
        raise AssertionError(argv)


def test_lease_writes_task_run_labels_and_round_trips(tmp_path):
    lease = lease_for_run(
        board_dir=tmp_path / "board",
        board="political-manager",
        task_id="t_123abc",
        run_id=7,
        workspace="/tmp/work",
    )

    payload = json.loads(lease.manifest_path.read_text(encoding="utf-8"))
    assert payload["task_id"] == "t_123abc"
    assert payload["run_id"] == 7
    assert payload["labels"]["task"] == "com.hermes.kanban.task_id=t_123abc"
    assert resource_environment(lease)["HERMES_KANBAN_RESOURCE_RUN_LABEL"].endswith(
        "=7"
    )
    restored = lease_from_path(lease.manifest_path)
    assert restored == lease


def test_cleanup_removes_only_exact_task_run_resources(tmp_path):
    lease = lease_for_run(
        board_dir=tmp_path / "board",
        board="political-manager",
        task_id="t_123abc",
        run_id=7,
    )
    fake = DockerFake()

    result = cleanup_owned_resources(lease, runner=fake)

    assert result["status"] == "cleaned"
    assert result["removed"] == {
        "container": ["owned-c"],
        "network": ["owned-n"],
        "volume": ["owned-v"],
    }
    assert result["residual"] == {"container": [], "network": [], "volume": []}
    assert fake.resources == {
        "container": {"other-c": False},
        "network": {"other-n": False},
        "volume": {"other-v": False},
    }
    payload = json.loads(lease.manifest_path.read_text(encoding="utf-8"))
    assert payload["cleanup"]["status"] == "cleaned"
    assert payload["cleanup"]["attempts"] == 1


def test_cleanup_records_residual_when_removal_fails(tmp_path):
    lease = lease_for_run(
        board_dir=tmp_path / "board",
        board="political-manager",
        task_id="t_123abc",
        run_id=7,
    )

    def failing_runner(argv):
        if argv[1:] == [
            "ps",
            "-aq",
            "--filter",
            "label=com.hermes.kanban.task_id=t_123abc",
            "--filter",
            "label=com.hermes.kanban.run_id=7",
        ]:
            return subprocess.CompletedProcess(argv, 0, "owned-c\n", "")
        if argv[1:3] == ["rm", "-f"]:
            return subprocess.CompletedProcess(argv, 1, "", "busy")
        return subprocess.CompletedProcess(argv, 0, "", "")

    result = cleanup_owned_resources(lease, runner=failing_runner)

    assert result["status"] == "partial"
    assert result["residual"]["container"] == ["owned-c"]
    assert result["errors"] == []
