"""Lane executor interfaces and deterministic fake executor for Phase 2."""

from __future__ import annotations

import json
import re
import shutil
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from .command import CommandRunner, SubprocessCommandRunner
from .lanes import LaneRequest, LaneResult, LaneStatus
from .redaction import redact_text


_UNSAFE_PATH_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")


def _executor_redact(text: str) -> str:
    return redact_text(text).replace("***", "[REDACTED]")


def _safe_lane_dir_name(lane_id: str) -> str:
    cleaned = _UNSAFE_PATH_CHARS.sub("-", lane_id).strip(".-")
    return cleaned or "lane"


class LaneExecutor(Protocol):
    def execute(self, req: LaneRequest) -> LaneResult:
        """Execute one lane and return a structured result. Must not leak secrets."""


class FakeLaneExecutor:
    """Programmed executor used by dry-run tests; never calls real models."""

    def __init__(self, programmed: dict[str, LaneResult | LaneStatus | Exception | str]):
        self.programmed = dict(programmed)
        self.seen: list[LaneRequest] = []

    def execute(self, req: LaneRequest) -> LaneResult:
        self.seen.append(req)
        value = self.programmed.get(req.lane_id, LaneStatus.SUCCEEDED)
        if isinstance(value, Exception):
            raise value
        if isinstance(value, LaneResult):
            return value
        if isinstance(value, LaneStatus):
            if value is LaneStatus.SUCCEEDED:
                return LaneResult(req.lane_id, req.agent, value, f"fake output for {req.lane_id}", None, 0.0, 0, None)
            if value is LaneStatus.TIMED_OUT:
                return LaneResult.timed_out(req)
            if value is LaneStatus.SKIPPED:
                return LaneResult.skipped(req, "programmed skip")
            return LaneResult(req.lane_id, req.agent, value, None, value.value, 0.0, 1, None)
        return LaneResult(req.lane_id, req.agent, LaneStatus.SUCCEEDED, str(value), None, 0.0, 0, None)


class CodexExternalIsolationExecutor:
    """Run a Codex lane inside a Hermes-owned isolated workspace.

    This is an internal execution unit for the external-isolated Codex path.
    It is intentionally not wired to the gateway command surface yet.
    """

    def __init__(
        self,
        *,
        codex_path: str,
        source_dir: str | Path,
        artifact_root: str | Path,
        runner: CommandRunner | None = None,
        env: Mapping[str, str] | None = None,
    ):
        self.codex_path = str(codex_path)
        self.source_dir = Path(source_dir)
        self.artifact_root = Path(artifact_root)
        self.runner = runner or SubprocessCommandRunner()
        self.env = dict(env or {})

    def _lane_paths(self, req: LaneRequest) -> dict[str, Path]:
        lane_root = self.artifact_root / _safe_lane_dir_name(req.lane_id)
        return {
            "lane_root": lane_root,
            "workdir": lane_root / "workdir",
            "codex_home": lane_root / "codex-home",
            "prompt": lane_root / "prompt.md",
            "output": lane_root / "codex-output.md",
            "log": lane_root / "codex.log",
            "invocation": lane_root / "invocation.json",
        }

    def _prepare_workspace(self, paths: dict[str, Path]) -> None:
        if not self.source_dir.exists():
            raise FileNotFoundError(f"source_dir not found: {self.source_dir}")
        paths["lane_root"].mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            self.source_dir,
            paths["workdir"],
            ignore=shutil.ignore_patterns(".git", "venv", ".venv", "__pycache__", ".pytest_cache"),
            dirs_exist_ok=False,
        )
        paths["codex_home"].mkdir(parents=True, exist_ok=False)
        paths["codex_home"].chmod(0o700)

    def _build_argv(self, req: LaneRequest, paths: dict[str, Path]) -> list[str]:
        argv = [
            self.codex_path,
            "--ask-for-approval",
            "never",
            "--sandbox",
            "danger-full-access",
            "--cd",
            str(paths["workdir"]),
        ]
        if req.effort:
            argv.extend(["--config", f"model_reasoning_effort={req.effort}"])
        argv.extend([
            "exec",
            "--skip-git-repo-check",
            "-o",
            str(paths["output"]),
            "-",
        ])
        return argv

    def _write_process_log(self, paths: dict[str, Path], completed) -> None:
        parts: list[str] = []
        if completed.stdout:
            parts.extend(["## stdout", completed.stdout])
        if completed.stderr:
            parts.extend(["## stderr", completed.stderr])
        body = _executor_redact("\n".join(parts).strip())
        paths["log"].write_text(body + ("\n" if body else ""), encoding="utf-8")

    def _redacted_output_text(self, paths: dict[str, Path], fallback: str = "") -> str:
        if paths["output"].exists():
            text = paths["output"].read_text(encoding="utf-8")
        else:
            text = fallback
        safe = _executor_redact(text)
        if paths["output"].exists() or safe:
            paths["output"].write_text(safe, encoding="utf-8")
        return safe.strip()

    def execute(self, req: LaneRequest) -> LaneResult:
        paths = self._lane_paths(req)
        artifacts = {name: str(path) for name, path in paths.items() if name != "lane_root"}
        started = time.monotonic()
        try:
            self._prepare_workspace(paths)
            paths["prompt"].write_text(req.prompt, encoding="utf-8")
            argv = self._build_argv(req, paths)
            env = {**self.env, "CODEX_HOME": str(paths["codex_home"])}
            paths["invocation"].write_text(
                json.dumps(
                    {
                        "argv": argv,
                        "cwd": str(paths["workdir"]),
                        "timeout_s": req.timeout_s,
                        "env_keys": sorted(env),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            completed = self.runner.run(
                argv,
                req.timeout_s,
                cwd=str(paths["workdir"]),
                input_text=req.prompt,
                env=env,
            )
        except Exception as exc:  # noqa: BLE001 - executor errors become lane failures
            duration = time.monotonic() - started
            try:
                paths["log"].write_text(_executor_redact(str(exc)) + "\n", encoding="utf-8")
            except Exception:
                pass
            return LaneResult.failed(
                req,
                _executor_redact(str(exc)),
                duration_s=duration,
                exit_code=None,
                log_path=str(paths["log"]),
            )

        duration = time.monotonic() - started
        self._write_process_log(paths, completed)
        output = self._redacted_output_text(paths, completed.stdout)
        if completed.timed_out:
            return LaneResult(
                req.lane_id,
                req.agent,
                LaneStatus.TIMED_OUT,
                None,
                f"timed out after {req.timeout_s:g}s",
                duration,
                completed.returncode,
                str(paths["log"]),
                artifacts,
            )
        if completed.returncode != 0:
            detail = "\n".join(part for part in (completed.stdout, completed.stderr) if part).strip()
            return LaneResult(
                req.lane_id,
                req.agent,
                LaneStatus.FAILED,
                None,
                _executor_redact(detail or f"codex exited with {completed.returncode}"),
                duration,
                completed.returncode,
                str(paths["log"]),
                artifacts,
            )
        return LaneResult(
            req.lane_id,
            req.agent,
            LaneStatus.SUCCEEDED,
            output,
            None,
            duration,
            completed.returncode,
            str(paths["log"]),
            artifacts,
        )


class RealLaneExecutor:
    """Placeholder for Phase 3 real model/worktree execution lanes."""

    def execute(self, req: LaneRequest) -> LaneResult:  # pragma: no cover - deliberately not used in Phase 1~2
        raise NotImplementedError("Real external-agent execution is deferred to Phase 3")
