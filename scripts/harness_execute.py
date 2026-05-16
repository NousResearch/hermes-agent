#!/usr/bin/env python3
"""Phase harness executor for agent-driven implementation steps.

This is a local, repo-agnostic adaptation of jha0313/harness_framework:
- phase specs live under ``phases/<phase-dir>/``
- ``index.json`` tracks step status
- each pending step is delegated to an external coding agent command
- the agent must update the phase index to completed/blocked/error

Safety-first defaults for Hermes/Undersea Friends operations:
- refuse dangerous full-auto agent flags unless explicitly opted in
- refuse to run from a dirty worktree
- stage only phase allowlisted paths, never blanket ``git add -A``
- write redacted, bounded agent output artifacts
- keep a ``status.json`` snapshot fresh while a step is running
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

KST = timezone(timedelta(hours=9))
DEFAULT_AGENT_COMMAND = "claude -p --output-format json"
DANGEROUS_AGENT_ARGS = {
    "--dangerously-skip-permissions",
    "--full-auto",
    "--yolo",
    "--force",
}
MAX_OUTPUT_CHARS = 20_000
SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|token|secret|password|passwd)\s*[:=]\s*([^\s'\";,}]+)"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]+"),
    re.compile(r"xapp-[A-Za-z0-9-]+"),
]


class HarnessExecutor:
    """Execute pending phase steps with a coding agent command."""

    def __init__(
        self,
        root: Path | str,
        phase_dir_name: str,
        *,
        agent_command: str | None = None,
        auto_push: bool = False,
        max_retries: int = 3,
        allow_unsafe_agent: bool = False,
        require_clean: bool = True,
        status_interval: float = 60.0,
        confirm_push: bool = False,
    ):
        self.root = Path(root).resolve()
        self.phase_dir_name = phase_dir_name
        self.phases_dir = self.root / "phases"
        self.phase_dir = self.phases_dir / phase_dir_name
        self.phase_index_path = self.phase_dir / "index.json"
        self.top_index_path = self.phases_dir / "index.json"
        self.agent_command = agent_command or os.environ.get("HARNESS_AGENT_COMMAND") or DEFAULT_AGENT_COMMAND
        self.auto_push = auto_push
        self.max_retries = max_retries
        self.allow_unsafe_agent = allow_unsafe_agent
        self.require_clean = require_clean
        self.status_interval = status_interval
        self.confirm_push = confirm_push

        if self.status_interval <= 0:
            print("ERROR: --status-interval must be greater than 0", file=sys.stderr)
            raise SystemExit(2)

        if not self.phase_dir.is_dir():
            raise SystemExit(f"ERROR: {self.phase_dir} not found")
        if not self.phase_index_path.exists():
            raise SystemExit(f"ERROR: {self.phase_index_path} not found")
        self.validate_agent_command()
        self._allowed_paths = self.load_allowed_paths_from_index()

    @staticmethod
    def stamp() -> str:
        return datetime.now(KST).strftime("%Y-%m-%dT%H:%M:%S%z")

    @staticmethod
    def read_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def write_json(path: Path, data: dict[str, Any]) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    @staticmethod
    def redact_output(text: str) -> str:
        redacted = text[:MAX_OUTPUT_CHARS]
        for pattern in SECRET_PATTERNS:
            redacted = pattern.sub(lambda m: f"{m.group(1)}=<redacted>" if m.lastindex and m.lastindex >= 1 else "<redacted>", redacted)
        if len(text) > MAX_OUTPUT_CHARS:
            redacted += f"\n... <truncated {len(text) - MAX_OUTPUT_CHARS} chars>"
        return redacted

    def read_phase_index(self) -> dict[str, Any]:
        return self.read_json(self.phase_index_path)

    def write_phase_index(self, index: dict[str, Any]) -> None:
        self.write_json(self.phase_index_path, index)

    def validate_agent_command(self) -> None:
        if self.allow_unsafe_agent:
            return
        parts = shlex.split(self.agent_command)
        dangerous = sorted(set(parts) & DANGEROUS_AGENT_ARGS)
        if dangerous:
            print(
                "ERROR: unsafe harness agent flags are disabled by default: "
                f"{', '.join(dangerous)}. Use --allow-unsafe-agent only in an isolated worktree.",
                file=sys.stderr,
            )
            raise SystemExit(2)

    def load_allowed_paths_from_index(self) -> list[str]:
        index = self.read_phase_index()
        paths = index.get("allowed_paths") or [f"phases/{self.phase_dir_name}/"]
        normalized: list[str] = []
        for item in paths:
            path = str(item).strip()
            if not path or path.startswith("/") or ".." in Path(path).parts:
                raise SystemExit(f"ERROR: unsafe allowed path in phase index: {item!r}")
            normalized.append(path.replace("\\", "/"))
        phase_prefix = f"phases/{self.phase_dir_name}/"
        if phase_prefix not in normalized:
            normalized.append(phase_prefix)
        return normalized


    def allowed_paths(self) -> list[str]:
        return list(self._allowed_paths)

    def ensure_allowed_paths_unchanged(self) -> None:
        current = self.load_allowed_paths_from_index()
        if current != self._allowed_paths:
            print(
                "ERROR: phase allowed_paths changed during agent execution; refusing to stage changes.",
                file=sys.stderr,
            )
            raise SystemExit(1)

    @staticmethod
    def status_paths_from_porcelain_line(line: str) -> list[str]:
        path = line[3:] if len(line) > 3 else ""
        if " -> " in path:
            return [part.strip().strip('"') for part in path.split(" -> ", 1)]
        return [path.strip().strip('"')]

    def is_allowed_worktree_path(self, path: str) -> bool:
        if not path:
            return True
        phase_output_prefix = f"phases/{self.phase_dir_name}/step"
        if path.startswith(phase_output_prefix) and path.endswith("-output.json"):
            return True
        for allowed in self._allowed_paths:
            if allowed.endswith("/"):
                if path.startswith(allowed):
                    return True
            elif path == allowed:
                return True
        return False

    def ensure_no_out_of_scope_changes(self) -> None:
        result = self.run_git("status", "--porcelain")
        if result.returncode != 0:
            raise SystemExit(f"ERROR: git status failed: {result.stderr.strip()}")
        out_of_scope = [
            path
            for line in result.stdout.splitlines()
            for path in self.status_paths_from_porcelain_line(line)
            if not self.is_allowed_worktree_path(path)
        ]
        if out_of_scope:
            preview = "\n".join(out_of_scope[:20])
            print(
                "ERROR: agent changed files outside phase allowed_paths; refusing to stage changes.\n"
                f"Out-of-scope files:\n{preview}",
                file=sys.stderr,
            )
            raise SystemExit(1)

    def load_guardrails(self) -> str:
        """Load repo context files that should constrain every step."""
        sections: list[str] = []
        for name in ("AGENTS.md", "CLAUDE.md"):
            path = self.root / name
            if path.exists():
                sections.append(f"## {name}\n\n{path.read_text(encoding='utf-8')}")

        docs_dir = self.root / "docs"
        if docs_dir.is_dir():
            for doc in sorted(docs_dir.glob("*.md")):
                rel = doc.relative_to(self.root)
                sections.append(f"## {rel}\n\n{doc.read_text(encoding='utf-8')}")
        return "\n\n---\n\n".join(sections)

    @staticmethod
    def completed_step_context(index: dict[str, Any]) -> str:
        lines = []
        for step in index.get("steps", []):
            if step.get("status") == "completed" and step.get("summary"):
                lines.append(f"- Step {step['step']} ({step['name']}): {step['summary']}")
        if not lines:
            return ""
        return "## Previous step outputs\n\n" + "\n".join(lines) + "\n\n"

    def build_prompt(self, step: dict[str, Any], index: dict[str, Any], previous_error: str | None = None) -> str:
        step_num = step["step"]
        step_file = self.phase_dir / f"step{step_num}.md"
        if not step_file.exists():
            raise SystemExit(f"ERROR: {step_file} not found")

        project = index.get("project", "project")
        phase = index.get("phase", self.phase_dir_name)
        retry = ""
        if previous_error:
            retry = (
                "\n## Previous attempt failed\n\n"
                f"Use this failure as context and fix the cause:\n\n{previous_error}\n\n---\n\n"
            )

        return (
            f"You are implementing a step for the {project} repository.\n\n"
            f"## Phase\n\n{phase}\n\n---\n\n"
            f"{self.load_guardrails()}\n\n---\n\n"
            f"{self.completed_step_context(index)}"
            f"{retry}"
            "## Harness rules\n\n"
            "1. Only perform the work requested by this step.\n"
            "2. Preserve existing behavior and tests.\n"
            "3. Verify the acceptance criteria directly.\n"
            f"4. Update phases/{self.phase_dir_name}/index.json for this step:\n"
            "   - success: set status to \"completed\" and add a one-line summary.\n"
            "   - needs user input: set status to \"blocked\" and add blocked_reason.\n"
            "   - cannot complete: set status to \"error\" and add error_message.\n"
            "5. Do not commit, push, restart gateways, kill processes, or print secrets. "
            "The harness will stage and commit allowlisted files after verification.\n\n"
            "---\n\n"
            f"## Step {step_num}: {step.get('name', '')}\n\n"
            f"{step_file.read_text(encoding='utf-8')}"
        )

    def run_git(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["git", *args], cwd=self.root, capture_output=True, text=True)

    def ensure_clean_worktree(self) -> None:
        if not self.require_clean:
            return
        result = self.run_git("status", "--porcelain")
        if result.returncode != 0:
            raise SystemExit(f"ERROR: git status failed: {result.stderr.strip()}")
        if result.stdout.strip():
            preview = "\n".join(result.stdout.strip().splitlines()[:20])
            print(
                "ERROR: harness requires a clean worktree before agent execution. "
                "Commit/stash unrelated changes or run in a dedicated worktree.\n"
                f"Dirty files:\n{preview}",
                file=sys.stderr,
            )
            raise SystemExit(2)

    def checkout_branch(self, phase: str) -> None:
        branch = f"feat-{self.safe_branch_slug(phase)}"
        current = self.run_git("rev-parse", "--abbrev-ref", "HEAD")
        if current.returncode != 0:
            raise SystemExit(f"ERROR: git repo unavailable: {current.stderr.strip()}")
        if current.stdout.strip() == branch:
            return
        exists = self.run_git("rev-parse", "--verify", branch)
        result = self.run_git("checkout", branch) if exists.returncode == 0 else self.run_git("checkout", "-b", branch)
        if result.returncode != 0:
            raise SystemExit(f"ERROR: could not checkout {branch}: {result.stderr.strip()}")

    @staticmethod
    def safe_branch_slug(value: str) -> str:
        slug = re.sub(r"[^A-Za-z0-9._/-]+", "-", value).strip("-/.")
        if not slug or slug.startswith("-") or ".." in slug:
            raise SystemExit(f"ERROR: unsafe phase branch name: {value!r}")
        return slug[:120]

    def invoke_agent(self, prompt: str) -> subprocess.CompletedProcess[str]:
        cmd = [*shlex.split(self.agent_command), prompt]
        return subprocess.run(cmd, cwd=self.root, capture_output=True, text=True, timeout=1800)

    def commit_step(self, step_num: int, step_name: str, phase: str) -> None:
        self.ensure_allowed_paths_unchanged()
        self.ensure_no_out_of_scope_changes()
        for path in self.allowed_paths():
            add = self.run_git("add", "--", path)
            if add.returncode != 0:
                raise SystemExit(f"ERROR: git add failed for {path}: {add.stderr.strip()}")
        for output_path in self.phase_dir.glob("step*-output.json"):
            rel = output_path.relative_to(self.root).as_posix()
            self.run_git("reset", "HEAD", "--", rel)
        message = f"feat({phase}): step {step_num} — {step_name}"
        commit = self.run_git("commit", "-m", message)
        if commit.returncode != 0 and "nothing to commit" not in (commit.stdout + commit.stderr).lower():
            raise SystemExit(f"ERROR: git commit failed: {commit.stderr.strip()}")

    def update_top_index(self, status: str) -> None:
        if not self.top_index_path.exists():
            return
        top = self.read_json(self.top_index_path)
        timestamp_key = {"completed": "completed_at", "error": "failed_at", "blocked": "blocked_at"}.get(status)
        for phase in top.get("phases", []):
            if phase.get("dir") == self.phase_dir_name:
                phase["status"] = status
                if timestamp_key:
                    phase[timestamp_key] = self.stamp()
                break
        self.write_json(self.top_index_path, top)

    def write_status_snapshot(self, status: str, step: dict[str, Any] | None = None) -> None:
        index = self.read_phase_index()
        steps = index.get("steps", [])
        completed = sum(1 for item in steps if item.get("status") == "completed")
        existing: dict[str, Any] = {}
        status_path = self.phase_dir / "status.json"
        if status_path.exists():
            try:
                existing = self.read_json(status_path)
            except json.JSONDecodeError:
                existing = {}
        current_step = step.get("step") if step else existing.get("current_step")
        current_step_name = step.get("name") if step else existing.get("current_step_name")
        now = self.stamp()
        snapshot = {
            "phase": index.get("phase", self.phase_dir_name),
            "current_step": current_step,
            "current_step_name": current_step_name,
            "status": status,
            "started_at": existing.get("started_at") or now,
            "updated_at": now,
            "completed_steps": completed,
            "total_steps": len(steps),
        }
        self.write_json(status_path, snapshot)

    def start_status_ticker(self, step: dict[str, Any]) -> tuple[threading.Event, threading.Thread]:
        stop = threading.Event()

        def tick() -> None:
            while not stop.wait(self.status_interval):
                self.write_status_snapshot("running", step)

        thread = threading.Thread(target=tick, daemon=True)
        thread.start()
        return stop, thread

    def mark_step_timestamp(self, step_num: int, key: str) -> None:
        index = self.read_phase_index()
        for step in index.get("steps", []):
            if step.get("step") == step_num:
                step[key] = self.stamp()
                break
        self.write_phase_index(index)

    def run(self) -> None:
        index = self.read_phase_index()
        phase = index.get("phase", self.phase_dir_name)
        self.ensure_clean_worktree()
        self.checkout_branch(phase)
        if "created_at" not in index:
            index["created_at"] = self.stamp()
            self.write_phase_index(index)

        while True:
            index = self.read_phase_index()
            pending = next((s for s in index.get("steps", []) if s.get("status") == "pending"), None)
            if pending is None:
                index["completed_at"] = self.stamp()
                self.write_phase_index(index)
                self.write_status_snapshot("completed", None)
                self.update_top_index("completed")
                if self.auto_push:
                    if not self.confirm_push:
                        print("ERROR: --push requires --yes-push after reviewing staged history and secret scan", file=sys.stderr)
                        raise SystemExit(2)
                    branch = f"feat-{self.safe_branch_slug(phase)}"
                    pushed = self.run_git("push", "-u", "origin", branch)
                    if pushed.returncode != 0:
                        raise SystemExit(f"ERROR: git push failed: {pushed.stderr.strip()}")
                print(f"Phase '{phase}' completed")
                return

            self.execute_step(pending, phase)

    def execute_step(self, step: dict[str, Any], phase: str) -> None:
        step_num = step["step"]
        step_name = step.get("name", f"step-{step_num}")
        previous_error = None
        self.mark_step_timestamp(step_num, "started_at")
        self.write_status_snapshot("running", step)

        for attempt in range(1, self.max_retries + 1):
            index = self.read_phase_index()
            prompt = self.build_prompt(step, index, previous_error)
            stop, ticker = self.start_status_ticker(step)
            try:
                result = self.invoke_agent(prompt)
            except subprocess.TimeoutExpired as exc:
                result = subprocess.CompletedProcess(exc.cmd, 124, stdout=exc.stdout or "", stderr=str(exc))
            finally:
                stop.set()
                ticker.join(timeout=1)

            stdout_text = result.stdout.decode("utf-8", "replace") if isinstance(result.stdout, bytes) else (result.stdout or "")
            stderr_text = result.stderr.decode("utf-8", "replace") if isinstance(result.stderr, bytes) else (result.stderr or "")
            output = {
                "step": step_num,
                "name": step_name,
                "attempt": attempt,
                "exitCode": result.returncode,
                "stdout_excerpt": self.redact_output(stdout_text),
                "stderr_excerpt": self.redact_output(stderr_text),
            }
            self.write_json(self.phase_dir / f"step{step_num}-output.json", output)

            updated = self.read_phase_index()
            current = next((s for s in updated.get("steps", []) if s.get("step") == step_num), {})
            status = current.get("status", "pending")
            if status == "completed":
                self.mark_step_timestamp(step_num, "completed_at")
                self.write_status_snapshot("completed", step)
                self.commit_step(step_num, step_name, phase)
                print(f"✓ Step {step_num}: {step_name}")
                return
            if status == "blocked":
                self.mark_step_timestamp(step_num, "blocked_at")
                self.write_status_snapshot("blocked", step)
                self.update_top_index("blocked")
                raise SystemExit(2)

            previous_error = current.get("error_message") or "Agent did not update step status to completed/blocked/error"
            if attempt < self.max_retries:
                updated = self.read_phase_index()
                for item in updated.get("steps", []):
                    if item.get("step") == step_num:
                        item["status"] = "pending"
                        item.pop("error_message", None)
                self.write_phase_index(updated)
                continue

            updated = self.read_phase_index()
            for item in updated.get("steps", []):
                if item.get("step") == step_num:
                    item["status"] = "error"
                    item["error_message"] = f"[{self.max_retries} attempt(s)] {previous_error}"
                    item["failed_at"] = self.stamp()
            self.write_phase_index(updated)
            self.write_status_snapshot("error", step)
            self.update_top_index("error")
            self.commit_step(step_num, step_name, phase)
            raise SystemExit(1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute harness phase steps with a coding agent")
    parser.add_argument("phase_dir", help="Phase directory name under ./phases, e.g. 0-mvp")
    parser.add_argument("--root", default=".", help="Repository root, default: current directory")
    parser.add_argument("--agent-command", help="Command prefix used to invoke the agent; prompt is appended")
    parser.add_argument("--push", action="store_true", help="Push feat-<phase> branch after phase completion")
    parser.add_argument("--yes-push", action="store_true", help="Second confirmation required with --push after manual diff/secret review")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--allow-unsafe-agent", action="store_true", help="Allow dangerous/full-auto agent flags; use only in isolated worktrees")
    parser.add_argument("--no-clean-check", action="store_true", help="Skip clean-worktree guard; not recommended")
    parser.add_argument("--status-interval", type=float, default=60.0, help="Seconds between running status.json refreshes")
    args = parser.parse_args(argv)

    executor = HarnessExecutor(
        args.root,
        args.phase_dir,
        agent_command=args.agent_command,
        auto_push=args.push,
        max_retries=args.max_retries,
        allow_unsafe_agent=args.allow_unsafe_agent,
        require_clean=not args.no_clean_check,
        status_interval=args.status_interval,
        confirm_push=args.yes_push,
    )
    executor.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
