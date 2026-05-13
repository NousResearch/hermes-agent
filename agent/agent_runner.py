"""Trusted CLI runner execution for named Hermes agents."""

from __future__ import annotations

import json
import os
import re
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home


@dataclass(frozen=True)
class CliRunnerResult:
    success: bool
    output: str
    stderr: str
    returncode: int
    runner_name: str
    command: list[str]
    resumed: bool = False
    external_session_id: str | None = None
    error: str | None = None
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "success": self.success,
            "output": self.output,
            "stderr": self.stderr,
            "returncode": self.returncode,
            "runner": {"mode": "cli", "name": self.runner_name, "resumed": self.resumed},
            "external_session_id": self.external_session_id,
            "error": self.error,
        }
        if self.warning:
            result["warning"] = self.warning
        return result


class CliSessionStore:
    """JSON-backed mapping from Hermes session + agent + runner to CLI session id."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (get_hermes_home() / "agent-runner-sessions.json")

    @staticmethod
    def _make_key(*, parent_session_id: str, agent_name: str, runner_name: str, workdir: str | None) -> str:
        return json.dumps(
            {
                "parent_session_id": parent_session_id or "default",
                "agent_name": agent_name,
                "runner_name": runner_name,
                "workdir": str(Path(workdir).resolve()) if workdir else "",
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @contextmanager
    def _locked(self):
        """Serialize updates across Hermes processes using an advisory lock."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with lock_path.open("a+") as lock_file:
            try:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            except (ImportError, OSError):
                # Best-effort fallback for non-POSIX platforms; writes remain atomic
                # via replace(), but concurrent read-modify-write may still race.
                pass
            try:
                yield
            finally:
                try:
                    import fcntl

                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                except (ImportError, OSError):
                    pass

    def _read(self) -> dict[str, str]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def _write(self, data: dict[str, str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)

    def get(self, *, parent_session_id: str, agent_name: str, runner_name: str, workdir: str | None) -> str | None:
        value = self._read().get(self._make_key(
            parent_session_id=parent_session_id,
            agent_name=agent_name,
            runner_name=runner_name,
            workdir=workdir,
        ))
        return value if isinstance(value, str) and value else None

    def set(self, *, parent_session_id: str, agent_name: str, runner_name: str, workdir: str | None, external_session_id: str) -> None:
        if not external_session_id:
            return
        with self._locked():
            data = self._read()
            data[self._make_key(
                parent_session_id=parent_session_id,
                agent_name=agent_name,
                runner_name=runner_name,
                workdir=workdir,
            )] = external_session_id
            self._write(data)


def _load_agent_runners_config() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
    except Exception:
        cfg = {}
    runners = cfg.get("agent_runners") or {}
    return runners if isinstance(runners, dict) else {}


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("runner args must be a list of strings")
    if not all(isinstance(item, str) for item in value):
        raise ValueError("runner args must be a list of strings")
    return list(value)


def _extract_external_session_id(stdout: str) -> str | None:
    for line in stdout.splitlines():
        try:
            obj = json.loads(line.strip())
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        for key in ("session_id", "sessionId", "sessionID", "id"):
            value = obj.get(key)
            if isinstance(value, str) and value:
                return value
        session = obj.get("session")
        if isinstance(session, dict):
            for key in ("id", "session_id", "sessionId"):
                value = session.get(key)
                if isinstance(value, str) and value:
                    return value
    match = re.search(r"(?:session[_ -]?id|Session ID)[:=]\s*([A-Za-z0-9_.:-]{6,})", stdout)
    return match.group(1) if match else None


def build_cli_prompt(*, agent_name: str, agent_prompt: str, task: str, context: str | None, source: str, path: str) -> str:
    parts = [
        f"## Named agent: {agent_name}",
        f"Source: {source} — {path}",
        "",
        "## Agent instructions",
        agent_prompt,
        "",
        "## Assignment task",
        task,
    ]
    if context and context.strip():
        parts.extend(["", "## Assignment context", context.strip()])
    return "\n".join(parts)


def run_cli_agent(*, agent: Any, task: str, context: str | None, workdir: str | None, parent_session_id: str) -> CliRunnerResult:
    runner_name = getattr(agent.routing, "runner_name", None)
    if not runner_name:
        return CliRunnerResult(False, "", "", 0, "", [], error="CLI runner requires runner.name.")

    runner_cfg = _load_agent_runners_config().get(runner_name)
    if not isinstance(runner_cfg, dict):
        return CliRunnerResult(False, "", "", 0, runner_name, [], error=f"Agent runner '{runner_name}' is not configured.")
    if runner_cfg.get("type") != "cli":
        return CliRunnerResult(False, "", "", 0, runner_name, [], error=f"Agent runner '{runner_name}' is not a cli runner.")
    if getattr(agent, "source", None) == "project" and not bool(runner_cfg.get("allowed_from_project_agents", False)):
        return CliRunnerResult(False, "", "", 0, runner_name, [], error=f"Agent runner '{runner_name}' is not allowed for project-local agents.")

    command = runner_cfg.get("command")
    if not isinstance(command, str) or not command.strip():
        return CliRunnerResult(False, "", "", 0, runner_name, [], error=f"Agent runner '{runner_name}' has no command.")
    try:
        cmd = [command, *_as_str_list(runner_cfg.get("args", []))]
    except ValueError as exc:
        return CliRunnerResult(False, "", "", 0, runner_name, [], error=str(exc))

    continue_mode = getattr(agent.routing, "runner_continue", None) or "off"
    if continue_mode not in ("off", "auto", "require"):
        return CliRunnerResult(False, "", "", 0, runner_name, cmd, error=f"Unsupported runner.continue='{continue_mode}'.")

    store = CliSessionStore()
    external_session_id = None
    resumed = False
    if continue_mode in ("auto", "require"):
        external_session_id = store.get(
            parent_session_id=parent_session_id,
            agent_name=agent.name,
            runner_name=runner_name,
            workdir=workdir,
        )
        if external_session_id:
            resume_arg = runner_cfg.get("resume_arg")
            if not isinstance(resume_arg, str) or not resume_arg:
                return CliRunnerResult(False, "", "", 0, runner_name, cmd, error=f"Agent runner '{runner_name}' has no resume_arg configured.")
            cmd.extend([resume_arg, external_session_id])
            resumed = True
        elif continue_mode == "require":
            return CliRunnerResult(False, "", "", 0, runner_name, cmd, error=f"Agent '{agent.name}' requires an existing CLI session for runner '{runner_name}'.")

    prompt = build_cli_prompt(
        agent_name=agent.name,
        agent_prompt=agent.prompt,
        task=task,
        context=context,
        source=agent.source,
        path=str(agent.path),
    )
    timeout = getattr(agent.limits, "timeout_seconds", None) or runner_cfg.get("timeout_seconds") or 600
    try:
        completed = subprocess.run(
            cmd,
            input=prompt,
            text=True,
            capture_output=True,
            cwd=workdir or None,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return CliRunnerResult(False, "", "", 127, runner_name, cmd, resumed=resumed, error=f"Runner command not found: {command}")
    except subprocess.TimeoutExpired as exc:
        return CliRunnerResult(False, exc.stdout or "", exc.stderr or "", 124, runner_name, cmd, resumed=resumed, external_session_id=external_session_id, error=f"Runner '{runner_name}' timed out after {timeout} seconds.")

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    new_session_id = _extract_external_session_id(stdout) or external_session_id
    warning = None
    if new_session_id:
        try:
            store.set(
                parent_session_id=parent_session_id,
                agent_name=agent.name,
                runner_name=runner_name,
                workdir=workdir,
                external_session_id=new_session_id,
            )
        except OSError as exc:
            warning = f"Failed to persist CLI session id for runner '{runner_name}': {exc}"

    return CliRunnerResult(
        completed.returncode == 0,
        stdout,
        stderr,
        completed.returncode,
        runner_name,
        cmd,
        resumed=resumed,
        external_session_id=new_session_id,
        error=None if completed.returncode == 0 else f"Runner '{runner_name}' exited with {completed.returncode}.",
        warning=warning,
    )
