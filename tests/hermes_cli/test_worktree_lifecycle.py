"""Focused tests for the Hermes Worktree Lifecycle Manager.

Run directly with Python's standard library so this safety gate does not depend
on the repository-wide pytest environment.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hermes_cli import worktree_lifecycle as wtl


def run(cwd: Path, *args: str) -> str:
    completed = subprocess.run(
        list(args), cwd=str(cwd), check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


class WorktreeLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.remote = self.base / "remote.git"
        self.repo = self.base / "canonical"
        self.root = self.base / "managed"
        self.registry = self.base / "registry.json"
        run(self.base, "git", "init", "--bare", str(self.remote))
        run(self.base, "git", "init", "-b", "main", str(self.repo))
        run(self.repo, "git", "config", "user.email", "test@example.com")
        run(self.repo, "git", "config", "user.name", "WTL Test")
        (self.repo / "README.md").write_text("baseline\n", encoding="utf-8")
        run(self.repo, "git", "add", "README.md")
        run(self.repo, "git", "commit", "-m", "baseline")
        run(self.repo, "git", "remote", "add", "origin", str(self.remote))
        run(self.repo, "git", "push", "-u", "origin", "main")
        self.port_patcher = mock.patch.object(wtl, "allocate_port", side_effect=range(8200, 8300))
        self.port_patcher.start()
        self.disk_patcher = mock.patch.object(
            wtl, "disk_policy",
            return_value={"percent": 20.0, "level": "normal", "message": "พื้นที่อยู่ในเกณฑ์ปกติ"},
        )
        self.disk_patcher.start()

    def tearDown(self) -> None:
        self.disk_patcher.stop()
        self.port_patcher.stop()
        self.temp.cleanup()

    def open_args(self, task: str = "TASK-1", slug: str = "first", apply: bool = True, staff: str = "staff-1") -> argparse.Namespace:
        return argparse.Namespace(
            project_id="project-1", staff_id=staff, task_id=task, slug=slug,
            repo=str(self.repo), root=str(self.root), registry=str(self.registry),
            machine_id="notebook-a", remote="origin", base_branch="main",
            lease_hours=12, allow_over_limit=False, apply=apply, as_json=True,
        )

    def task(self, task_id: str = "TASK-1") -> dict:
        return wtl.load_registry(self.registry)["tasks"][task_id]

    def test_notebook_default_root_is_under_documents(self) -> None:
        self.assertEqual(
            self.base / "Documents" / "Worktrees",
            wtl.default_worktree_root("notebook-a", self.base),
        )
        self.assertEqual(
            Path("/home/linux-nat/.worktree"),
            wtl.default_worktree_root("vps-linux-nat", self.base),
        )

    def test_dry_run_does_not_change_git_or_registry(self) -> None:
        result = wtl.command_open(self.open_args(apply=False))
        self.assertEqual("WTL_OPEN_PROPOSED", result["decision"])
        self.assertFalse(self.registry.exists())
        self.assertFalse(Path(result["task"]["worktree_path"]).exists())

    def test_open_is_idempotent_and_blocks_same_task_at_other_path(self) -> None:
        first = wtl.command_open(self.open_args())
        self.assertEqual("WTL_READY", first["decision"])
        repeated = wtl.command_open(self.open_args())
        self.assertEqual(first["task"]["worktree_path"], repeated["task"]["worktree_path"])
        with self.assertRaises(wtl.WorktreeLifecycleError):
            wtl.command_open(self.open_args(slug="different"))
        other_machine = self.open_args()
        other_machine.machine_id = "notebook-b"
        blocked = wtl.command_open(other_machine)
        self.assertEqual("WTL_BLOCKED", blocked["decision"])

    def test_three_open_worktree_limit(self) -> None:
        for number in range(1, 4):
            wtl.command_open(self.open_args("TASK-{}".format(number), "work-{}".format(number)))
        with self.assertRaisesRegex(wtl.WorktreeLifecycleError, "ครบ 3 งาน"):
            wtl.command_open(self.open_args("TASK-4", "work-4"))

    def test_two_people_and_tasks_get_distinct_paths_branches_and_runtime(self) -> None:
        one = wtl.command_open(self.open_args("TASK-1", "alpha", staff="staff-1"))["task"]
        two = wtl.command_open(self.open_args("TASK-2", "beta", staff="staff-2"))["task"]
        self.assertNotEqual(one["worktree_path"], two["worktree_path"])
        self.assertNotEqual(one["branch"], two["branch"])
        tasks = wtl.load_registry(self.registry)["tasks"]
        self.assertNotEqual(tasks["TASK-1"]["runtime"]["port"], tasks["TASK-2"]["runtime"]["port"])
        self.assertNotEqual(tasks["TASK-1"]["runtime"]["database_name"], tasks["TASK-2"]["runtime"]["database_name"])

    def test_status_blocks_branch_drift(self) -> None:
        wtl.command_open(self.open_args())
        worktree = Path(self.task()["worktree_path"])
        run(worktree, "git", "checkout", "--detach")
        result = wtl.command_status(argparse.Namespace(task_id="TASK-1", registry=str(self.registry)))
        self.assertEqual("WTL_BLOCKED", result["decision"])
        self.assertEqual("BLOCKED", self.task()["state"])

    def test_status_blocks_and_releases_expired_lease(self) -> None:
        wtl.command_open(self.open_args())
        data = wtl.load_registry(self.registry)
        data["tasks"]["TASK-1"]["lease_expires_at"] = (wtl.utcnow() - dt.timedelta(minutes=1)).isoformat()
        wtl.save_registry(self.registry, data)
        result = wtl.command_status(argparse.Namespace(task_id="TASK-1", registry=str(self.registry)))
        self.assertEqual("WTL_BLOCKED", result["decision"])
        self.assertIsNone(self.task()["lease_id"])

    def test_pause_releases_writer_lease(self) -> None:
        wtl.command_open(self.open_args())
        result = wtl.command_pause(argparse.Namespace(task_id="TASK-1", registry=str(self.registry), reason="พักทดสอบ"))
        self.assertEqual("WTL_READ_ONLY", result["decision"])
        self.assertIsNone(self.task()["lease_id"])

    def test_dirty_or_unpushed_blocks_handoff_and_close(self) -> None:
        wtl.command_open(self.open_args())
        worktree = Path(self.task()["worktree_path"])
        (worktree / "dirty.txt").write_text("dirty\n", encoding="utf-8")
        handoff = wtl.command_handoff(argparse.Namespace(task_id="TASK-1", registry=str(self.registry), to_machine="vps-a", apply=True))
        close = wtl.command_close(argparse.Namespace(task_id="TASK-1", registry=str(self.registry), merged=False, merge_sha=None))
        self.assertEqual("WTL_BLOCKED", handoff["decision"])
        self.assertEqual("WTL_BLOCKED", close["decision"])

    def test_handoff_accept_moves_single_writer_to_new_machine(self) -> None:
        wtl.command_open(self.open_args())
        first = self.task()
        worktree = Path(first["worktree_path"])
        run(worktree, "git", "push", "-u", "origin", first["branch"])
        prepared = wtl.command_handoff(argparse.Namespace(task_id="TASK-1", registry=str(self.registry), to_machine="vps-a", apply=True))
        self.assertEqual("WTL_HANDOFF_READY", prepared["decision"])
        self.assertIsNone(self.task()["lease_id"])

        receiver_repo = self.base / "receiver-canonical"
        run(self.base, "git", "clone", str(self.remote), str(receiver_repo))
        run(receiver_repo, "git", "checkout", "main")
        receiver_root = self.base / "receiver-managed"
        receiver_path = receiver_root / "project-1" / "staff-1" / "TASK-1-first"
        accepted = wtl.command_accept(argparse.Namespace(
            task_id="TASK-1", registry=str(self.registry), machine_id="vps-a",
            repo=str(receiver_repo), root=str(receiver_root), worktree_path=str(receiver_path),
            lease_hours=12, apply=True,
        ))
        self.assertEqual("WTL_READY", accepted["decision"])
        current = self.task()
        self.assertEqual("vps-a", current["machine_id"])
        self.assertIsNotNone(current["lease_id"])
        self.assertEqual(1, len(current["locations"]))

    def test_offline_owner_machine_keeps_existing_lease_but_cannot_transfer(self) -> None:
        wtl.command_open(self.open_args())
        worktree = Path(self.task()["worktree_path"])
        run(worktree, "git", "remote", "set-url", "origin", "https://offline.invalid/repo.git")
        remote_registry = "ssh://user@example.test/home/user/wtl/registry.json"
        with mock.patch.object(wtl, "get_hermes_home", return_value=self.base / "offline-home"):
            wtl.save_remote_cache(remote_registry, wtl.load_registry(self.registry))
            status = wtl.command_status(argparse.Namespace(
                task_id="TASK-1", registry=remote_registry, offline=True, machine_id="notebook-a",
            ))
            wrong_machine = wtl.command_status(argparse.Namespace(
                task_id="TASK-1", registry=remote_registry, offline=True, machine_id="notebook-b",
            ))
        self.assertEqual("WTL_READY_OFFLINE", status["decision"])
        self.assertEqual("WTL_BLOCKED", wrong_machine["decision"])
        with self.assertRaisesRegex(wtl.WorktreeLifecycleError, "ยังไม่พร้อมส่งต่อ"):
            wtl.command_accept(argparse.Namespace(
                task_id="TASK-1", registry=str(self.registry), machine_id="vps-a",
                repo=str(self.repo), root=str(self.root), worktree_path=str(worktree),
                lease_hours=12, apply=True,
            ))

    def test_remote_registry_uri_rejects_shell_metacharacters(self) -> None:
        self.assertEqual(
            ("user@example.test", "/home/user/wtl/registry.json"),
            wtl.parse_remote_registry("ssh://user@example.test/home/user/wtl/registry.json"),
        )
        with self.assertRaises(wtl.WorktreeLifecycleError):
            wtl.parse_remote_registry("ssh://user@example.test/tmp/a;touch-pwned")

    def test_cleanup_requires_six_gates_and_quarantines_before_remove(self) -> None:
        wtl.command_open(self.open_args())
        task = self.task()
        worktree = Path(task["worktree_path"])
        run(worktree, "git", "push", "-u", "origin", task["branch"])
        closed = wtl.command_close(argparse.Namespace(task_id="TASK-1", registry=str(self.registry), merged=True, merge_sha=task["base_sha"]))
        self.assertEqual("MERGED", closed["task"]["state"])
        dry = wtl.command_cleanup(argparse.Namespace(task_id="TASK-1", registry=str(self.registry), apply=False))
        self.assertEqual("WTL_CLEANUP_PROPOSED", dry["decision"])
        self.assertEqual(6, sum(bool(value) for value in dry["gates"].values()))
        quarantined = wtl.command_cleanup(argparse.Namespace(task_id="TASK-1", registry=str(self.registry), apply=True))
        self.assertEqual("WTL_CLEANUP_READY", quarantined["decision"])
        self.assertTrue(worktree.exists(), "ช่วงกักพักต้องยังไม่ลบ Worktree")

        data = wtl.load_registry(self.registry)
        data["tasks"]["TASK-1"]["cleanup"]["quarantine_until"] = (
            wtl.utcnow() - dt.timedelta(minutes=1)
        ).isoformat()
        wtl.save_registry(self.registry, data)
        second_dry = wtl.command_cleanup(argparse.Namespace(task_id="TASK-1", registry=str(self.registry), apply=False))
        self.assertEqual("WTL_CLEANUP_PROPOSED", second_dry["decision"])
        archived = wtl.command_cleanup(argparse.Namespace(task_id="TASK-1", registry=str(self.registry), apply=True))
        self.assertEqual("WTL_ARCHIVED", archived["decision"])
        self.assertFalse(worktree.exists())

    def test_active_worktree_cannot_enter_cleanup(self) -> None:
        wtl.command_open(self.open_args())
        result = wtl.command_cleanup(argparse.Namespace(task_id="TASK-1", registry=str(self.registry), apply=True))
        self.assertEqual("WTL_BLOCKED", result["decision"])
        self.assertIn("merged_or_owner_abandoned", result["reasons"])
        self.assertTrue(Path(self.task()["worktree_path"]).exists())

    def test_cleanup_apply_rejects_missing_or_stale_dry_run(self) -> None:
        wtl.command_open(self.open_args())
        task = self.task()
        worktree = Path(task["worktree_path"])
        run(worktree, "git", "push", "-u", "origin", task["branch"])
        wtl.command_close(argparse.Namespace(task_id="TASK-1", registry=str(self.registry), merged=True, merge_sha=task["base_sha"]))
        blocked = wtl.command_cleanup(argparse.Namespace(task_id="TASK-1", registry=str(self.registry), apply=True))
        self.assertEqual("WTL_BLOCKED", blocked["decision"])
        self.assertIn("dry_run_recorded", blocked["reasons"])
        self.assertTrue(worktree.exists())

    def test_disk_policy_has_70_85_90_thresholds(self) -> None:
        total = 1_000
        expected = [(699, "normal"), (700, "warning"), (850, "stop_new"), (900, "recovery")]
        self.disk_patcher.stop()
        try:
            for used, level in expected:
                with mock.patch.object(wtl.shutil, "disk_usage", return_value=wtl.shutil._ntuple_diskusage(total, used, total - used)):
                    self.assertEqual(level, wtl.disk_policy(self.base)["level"])
        finally:
            self.disk_patcher.start()

    def test_failed_open_is_recoverable_as_blocked_registry_record(self) -> None:
        original_git = wtl.git

        def fail_worktree_add(cwd, *args, **kwargs):
            if args[:2] == ("worktree", "add"):
                return subprocess.CompletedProcess(["git", *args], 1, "", "simulated interruption")
            return original_git(cwd, *args, **kwargs)

        with mock.patch.object(wtl, "git", side_effect=fail_worktree_add):
            with self.assertRaisesRegex(wtl.WorktreeLifecycleError, "simulated interruption"):
                wtl.command_open(self.open_args())
        task = self.task()
        self.assertEqual("BLOCKED", task["state"])
        self.assertIsNone(task["lease_id"])
        self.assertTrue(any(item["action"] == "open-failed" for item in task["history"]))

    def test_doctor_detects_path_drift(self) -> None:
        wtl.command_open(self.open_args())
        data = wtl.load_registry(self.registry)
        data["tasks"]["TASK-1"]["worktree_path"] = str(self.base / "missing")
        wtl.save_registry(self.registry, data)
        result = wtl.command_doctor(argparse.Namespace(registry=str(self.registry)))
        self.assertEqual("WTL_BLOCKED", result["decision"])
        self.assertTrue(any("ไม่ตรง" in reason for reason in result["reasons"]))

    def test_scan_is_read_only_and_classifies_managed_unknown(self) -> None:
        wtl.command_open(self.open_args())
        before = run(self.repo, "git", "status", "--porcelain=v1", "-b")
        result = wtl.command_scan(argparse.Namespace(repo=[str(self.repo)], registry=str(self.registry)))
        after = run(self.repo, "git", "status", "--porcelain=v1", "-b")
        self.assertEqual(before, after)
        self.assertEqual(1, result["counts"]["managed"])
        self.assertGreaterEqual(result["counts"]["unknown"], 1)  # canonical checkout
        self.assertIn("ไม่ลบ", result["message"])

    def test_import_registers_existing_worktree_paused_without_git_change(self) -> None:
        legacy = self.root / "project-1" / "staff-1" / "legacy-task"
        legacy.parent.mkdir(parents=True, exist_ok=True)
        run(self.repo, "git", "worktree", "add", "-b", "legacy/staff-1/task", str(legacy), "origin/main")
        branch_before = run(legacy, "git", "branch", "--show-current")
        result = wtl.command_import(argparse.Namespace(
            project_id="project-1", staff_id="staff-1", task_id="LEGACY-1",
            machine_id="notebook-a", repo=str(self.repo), worktree_path=str(legacy),
            root=str(self.root), remote="origin", base_branch="main",
            owner_approval="owner approved migration", registry=str(self.registry),
        ))
        self.assertEqual("WTL_IMPORTED_READ_ONLY", result["decision"])
        self.assertEqual("PAUSED", self.task("LEGACY-1")["state"])
        self.assertIsNone(self.task("LEGACY-1")["lease_id"])
        self.assertEqual(branch_before, run(legacy, "git", "branch", "--show-current"))

    def test_legacy_route_resolves_exact_task_from_wtl_registry(self) -> None:
        wtl.command_open(self.open_args())
        script = ROOT / "scripts" / "hermes_worktree_route.py"
        result = subprocess.run([
            sys.executable, str(script), "--staff-id", "staff-1", "--project", "project-1",
            "--task-id", "TASK-1", "--registry", str(self.registry), "--local",
        ], text=True, capture_output=True, check=False)
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual("wtl_registry", payload["source"])
        self.assertEqual(self.task()["worktree_path"], payload["selected"]["path"])

    def test_legacy_write_permit_cannot_bypass_wtl_writer_gate(self) -> None:
        wtl.command_open(self.open_args())
        task = self.task()
        worktree = Path(task["worktree_path"])
        script = ROOT / "scripts" / "hermes_write_permit.py"
        env = {
            **os.environ,
            "HERMES_HOME": str(self.base / "hermes-home"),
            "HERMES_WORKTREE_REGISTRY": str(self.registry),
        }
        command = [
            sys.executable, str(script), "acquire", "--cwd", str(worktree),
            "--task-id", "TASK-1", "--branch", task["branch"], "--base-sha", task["base_sha"],
            "--allowed-path", "README.md", "--approval", "owner-ok",
        ]
        ready = subprocess.run(command, env=env, text=True, capture_output=True, check=False)
        self.assertEqual(0, ready.returncode, ready.stdout + ready.stderr)
        self.assertTrue(json.loads(ready.stdout)["permit"]["wtl"]["checks"]["writer_lease"])

        data = wtl.load_registry(self.registry)
        data["tasks"]["TASK-1"]["lease_id"] = None
        wtl.save_registry(self.registry, data)
        blocked = subprocess.run(command, env=env, text=True, capture_output=True, check=False)
        self.assertEqual(2, blocked.returncode)
        self.assertIn("WTL_BLOCKED", blocked.stdout)

    def test_pdca_report_tracks_24h_and_168h_cadence(self) -> None:
        wtl.command_open(self.open_args())
        preview = wtl.command_report(argparse.Namespace(
            registry=str(self.registry), record=False, cleanup_review=False,
        ))
        self.assertEqual("WTL_PDCA_HEALTHY", preview["decision"])
        self.assertTrue(preview["report"]["cadence"]["light_check_due"])
        self.assertTrue(preview["report"]["cadence"]["cleanup_review_due"])

        recorded = wtl.command_report(argparse.Namespace(
            registry=str(self.registry), record=True, cleanup_review=True,
        ))
        self.assertTrue(recorded["recorded"])
        data = wtl.load_registry(self.registry)
        self.assertIsNotNone(data["pdca"]["last_light_check_at"])
        self.assertIsNotNone(data["pdca"]["last_cleanup_review_at"])

    def test_parser_exposes_all_lifecycle_commands(self) -> None:
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command", required=True)
        worktree = wtl.register_worktree_subparser(subparsers)
        action = next(item for item in worktree._actions if isinstance(item, argparse._SubParsersAction))
        self.assertEqual(
            {"open", "list", "status", "enter", "doctor", "scan", "import", "report", "pause", "handoff", "accept", "close", "abandon", "cleanup"},
            set(action.choices),
        )
        parsed = parser.parse_args([
            "worktree", "open", "--project-id", "project-1", "--staff-id", "staff-1",
            "--task-id", "TASK-1", "--slug", "first", "--repo", str(self.repo),
        ])
        self.assertIsNone(parsed.root)


if __name__ == "__main__":
    unittest.main(verbosity=2)
