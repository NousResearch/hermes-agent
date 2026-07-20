"""Git-based gateway deployment helper.

This module replaces the old file-by-file SSH deployment flow with a
versioned release workflow:

- require a clean local git worktree
- resolve a concrete commit/ref to deploy
- create a git bundle for that ref
- upload and checkout the commit on the remote host
- install/restart the gateway via systemd system scope
- verify readiness via systemctl + gateway status files
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import shlex
from typing import Any


DEFAULT_HOST = "root@888933.xyz"
DEFAULT_REMOTE_DIR = "/root/projects/hermes-agent"
DEFAULT_REMOTE_HERMES_HOME = "/root/.hermes"
DEFAULT_TAIL_LINES = 80
DEFAULT_READINESS_TIMEOUT = 45
DEFAULT_RELEASE_STATE_NAME = "gateway-release.json"
DEFAULT_SERVICE_NAME = "hermes-gateway"
DEFAULT_RUN_AS_USER = "root"
PRESERVED_REMOTE_DIRS = (".git", ".deploy-backups", "venv", ".venv", "node_modules")


class DeploymentError(RuntimeError):
    """Raised when deployment cannot continue safely."""


@dataclass(frozen=True)
class DeployOptions:
    host: str
    project_dir: Path
    remote_dir: str
    remote_hermes_home: str
    remote_python: str
    ref: str = "HEAD"
    tail_lines: int = DEFAULT_TAIL_LINES
    sync: bool = True
    restart: bool = True
    rollback: bool = False
    system_service: bool = True


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_stdout(result: Any) -> str:
    if isinstance(result, str):
        return result.strip()
    return str(getattr(result, "stdout", "") or "").strip()


def _extract_returncode(result: Any) -> int:
    if isinstance(result, str):
        return 0
    return int(getattr(result, "returncode", 0))


def _parse_json(raw: str | None) -> dict[str, Any] | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def parse_args(argv: list[str] | None = None) -> DeployOptions:
    parser = argparse.ArgumentParser(
        description="Deploy Hermes gateway from a git commit/ref via systemd system scope.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR)
    parser.add_argument("--remote-hermes-home", default=DEFAULT_REMOTE_HERMES_HOME)
    parser.add_argument("--remote-python", default="")
    parser.add_argument("--ref", default="HEAD", help="Git ref/commit to deploy (default: HEAD)")
    parser.add_argument("--tail", type=int, default=DEFAULT_TAIL_LINES, help="Journal lines to print during verification")
    parser.add_argument("--no-sync", action="store_true", help="Skip git bundle sync and only restart/verify current release")
    parser.add_argument("--no-restart", action="store_true", help="Sync release but skip service restart")
    parser.add_argument("--rollback", action="store_true", help="Deploy the previously recorded release commit")
    parser.add_argument("legacy_paths", nargs="*", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.legacy_paths:
        parser.error("File-based deploy arguments are no longer supported; use --ref or --rollback.")
    if args.rollback and args.no_sync:
        parser.error("--rollback requires sync; do not combine it with --no-sync.")

    project_dir = Path(__file__).resolve().parents[1]
    remote_python = args.remote_python or f"{args.remote_dir}/venv/bin/python"
    return DeployOptions(
        host=args.host,
        project_dir=project_dir,
        remote_dir=args.remote_dir,
        remote_hermes_home=args.remote_hermes_home,
        remote_python=remote_python,
        ref=args.ref,
        tail_lines=args.tail,
        sync=not args.no_sync,
        restart=not args.no_restart,
        rollback=args.rollback,
        system_service=True,
    )


class GatewayDeployer:
    def __init__(self, options: DeployOptions):
        self.options = options

    @property
    def release_state_path(self) -> str:
        return f"{self.options.remote_hermes_home}/deployments/{DEFAULT_RELEASE_STATE_NAME}"

    def run_local(
        self,
        args: list[str],
        *,
        cwd: Path | None = None,
        check: bool = True,
        timeout: int = 60,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            args,
            cwd=str(cwd or self.options.project_dir),
            capture_output=True,
            text=True,
            check=check,
            timeout=timeout,
        )

    def run_remote(
        self,
        command: str,
        *,
        check: bool = True,
        timeout: int = 90,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["ssh", self.options.host, command],
            capture_output=True,
            text=True,
            check=check,
            timeout=timeout,
        )

    def copy_to_remote(self, local_path: Path, remote_path: str, *, timeout: int = 120) -> None:
        subprocess.run(
            ["scp", str(local_path), f"{self.options.host}:{remote_path}"],
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout,
        )

    def ensure_clean_worktree(self) -> None:
        result = self.run_local(["git", "status", "--short"], cwd=self.options.project_dir)
        status = _extract_stdout(result)
        if status:
            raise DeploymentError(
                "Local git working tree is dirty. Commit or stash changes before git-based deployment."
            )

    def resolve_target_commit(self, previous_state: dict[str, Any] | None = None) -> str:
        if self.options.rollback:
            previous_commit = str((previous_state or {}).get("previous_commit") or "").strip()
            if not previous_commit:
                raise DeploymentError("No previous deployed commit is recorded for rollback.")
            return previous_commit

        result = self.run_local(
            ["git", "rev-parse", "--verify", self.options.ref],
            cwd=self.options.project_dir,
        )
        commit = _extract_stdout(result)
        if not commit:
            raise DeploymentError(f"Could not resolve git ref: {self.options.ref}")
        return commit

    def create_git_bundle(self, target_commit: str) -> Path:
        fd, raw_path = tempfile.mkstemp(prefix="hermes-gateway-", suffix=".bundle")
        bundle_path = Path(raw_path)
        bundle_path.unlink(missing_ok=True)
        try:
            self.run_local(
                ["git", "bundle", "create", str(bundle_path), target_commit],
                cwd=self.options.project_dir,
                timeout=120,
            )
        except Exception:
            bundle_path.unlink(missing_ok=True)
            raise
        return bundle_path

    def build_remote_gateway_command(self, *gateway_args: str) -> str:
        joined_args = " ".join(shlex.quote(arg) for arg in gateway_args)
        return (
            f"cd {shlex.quote(self.options.remote_dir)} && "
            f"HERMES_HOME={shlex.quote(self.options.remote_hermes_home)} "
            f"{shlex.quote(self.options.remote_python)} -m hermes_cli.main gateway {joined_args}"
        )

    def read_remote_release_state(self) -> dict[str, Any] | None:
        result = self.run_remote(
            f"test -f {shlex.quote(self.release_state_path)} && cat {shlex.quote(self.release_state_path)} || true",
            check=False,
            timeout=30,
        )
        return _parse_json(_extract_stdout(result))

    def write_remote_release_state(self, state: dict[str, Any]) -> None:
        local_tmp = Path(tempfile.mkstemp(prefix="hermes-gateway-release-", suffix=".json")[1])
        try:
            local_tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
            self.run_remote(
                f"mkdir -p {shlex.quote(str(Path(self.release_state_path).parent))}",
                timeout=30,
            )
            self.copy_to_remote(local_tmp, self.release_state_path)
        finally:
            local_tmp.unlink(missing_ok=True)

    def build_release_state(self, previous_commit: str | None, current_commit: str) -> dict[str, Any]:
        return {
            "current_commit": current_commit,
            "previous_commit": previous_commit,
            "source_ref": self.options.ref,
            "deployed_at": _now_iso(),
            "host": self.options.host,
            "remote_dir": self.options.remote_dir,
        }

    def sync_release_bundle(self, bundle_path: Path, target_commit: str) -> None:
        remote_bundle = f"/tmp/{bundle_path.name}"
        self.run_remote(
            f"mkdir -p {shlex.quote(self.options.remote_dir)} {shlex.quote(self.options.remote_hermes_home)}/deployments",
            timeout=30,
        )
        self.copy_to_remote(bundle_path, remote_bundle)

        excludes = " ".join(f"-e {shlex.quote(name)}" for name in PRESERVED_REMOTE_DIRS)
        command = (
            "set -euo pipefail; "
            f"cd {shlex.quote(self.options.remote_dir)}; "
            "if [ ! -d .git ]; then git init; fi; "
            f"git fetch --force {shlex.quote(remote_bundle)} {shlex.quote(target_commit)}:refs/hermes-deploy/target; "
            f"git checkout --force --detach {shlex.quote(target_commit)}; "
            f"git clean -fd {excludes}; "
            f"rm -f {shlex.quote(remote_bundle)}"
        )
        self.run_remote(command, timeout=180)

    def ensure_system_service(self) -> None:
        self.run_remote("command -v systemctl >/dev/null", timeout=15)
        self.run_remote(
            self.build_remote_gateway_command("install", "--system", "--run-as-user", DEFAULT_RUN_AS_USER),
            timeout=120,
        )

    def restart_system_service(self) -> None:
        self.run_remote(
            self.build_remote_gateway_command("restart", "--system"),
            timeout=120,
        )

    def _remote_service_name(self) -> str:
        result = self.run_remote(
            (
                f"cd {shlex.quote(self.options.remote_dir)} && "
                f"HERMES_HOME={shlex.quote(self.options.remote_hermes_home)} "
                f"{shlex.quote(self.options.remote_python)} - <<'PY'\n"
                "from hermes_cli.gateway import get_service_name\n"
                "print(get_service_name())\n"
                "PY"
            ),
            check=False,
            timeout=30,
        )
        return _extract_stdout(result) or DEFAULT_SERVICE_NAME

    def verify_readiness(self) -> dict[str, Any]:
        service_name = self._remote_service_name()
        deadline = time.time() + DEFAULT_READINESS_TIMEOUT
        state_path = f"{self.options.remote_hermes_home}/gateway_state.json"
        pid_path = f"{self.options.remote_hermes_home}/gateway.pid"
        last_snapshot: dict[str, Any] = {}

        while time.time() < deadline:
            is_active = self.run_remote(
                f"systemctl is-active {shlex.quote(service_name)}",
                check=False,
                timeout=15,
            )
            main_pid_result = self.run_remote(
                f"systemctl show {shlex.quote(service_name)} --property MainPID --value",
                check=False,
                timeout=15,
            )
            state_raw = self.run_remote(
                f"test -f {shlex.quote(state_path)} && cat {shlex.quote(state_path)} || true",
                check=False,
                timeout=15,
            )
            pid_raw = self.run_remote(
                f"test -f {shlex.quote(pid_path)} && cat {shlex.quote(pid_path)} || true",
                check=False,
                timeout=15,
            )

            state_data = _parse_json(_extract_stdout(state_raw)) or {}
            pid_data = _parse_json(_extract_stdout(pid_raw)) or {}
            main_pid = _extract_stdout(main_pid_result)
            service_active = _extract_stdout(is_active) == "active"
            gateway_state = str(state_data.get("gateway_state") or "").strip().lower()

            last_snapshot = {
                "service_name": service_name,
                "service_active": service_active,
                "main_pid": int(main_pid) if str(main_pid).isdigit() else 0,
                "gateway_state": gateway_state,
                "pid_file_pid": int(pid_data.get("pid", 0) or 0),
            }

            if (
                service_active
                and last_snapshot["main_pid"] > 0
                and last_snapshot["pid_file_pid"] > 0
                and gateway_state in {"running", "degraded"}
            ):
                return last_snapshot

            time.sleep(2)

        raise DeploymentError(f"Gateway readiness check failed: {json.dumps(last_snapshot, ensure_ascii=False)}")

    def tail_remote_logs(self) -> str:
        service_name = self._remote_service_name()
        result = self.run_remote(
            f"journalctl -u {shlex.quote(service_name)} -n {int(self.options.tail_lines)} --no-pager || true",
            check=False,
            timeout=30,
        )
        return _extract_stdout(result)

    def run(self) -> dict[str, Any]:
        previous_state = self.read_remote_release_state()
        previous_commit = str((previous_state or {}).get("current_commit") or "").strip() or None

        if self.options.sync:
            self.ensure_clean_worktree()
            target_commit = self.resolve_target_commit(previous_state)
            bundle_path = self.create_git_bundle(target_commit)
            try:
                self.sync_release_bundle(bundle_path, target_commit)
            finally:
                bundle_path.unlink(missing_ok=True)
        else:
            target_commit = previous_commit or ""

        self.ensure_system_service()
        if self.options.restart:
            self.restart_system_service()
        readiness = self.verify_readiness()

        if self.options.sync:
            self.write_remote_release_state(self.build_release_state(previous_commit, target_commit))

        return {"target_commit": target_commit, **readiness}


def main(argv: list[str] | None = None) -> int:
    options = parse_args(argv)
    deployer = GatewayDeployer(options)
    result = deployer.run()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    logs = deployer.tail_remote_logs()
    if logs:
        print("\n--- journal ---")
        print(logs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
