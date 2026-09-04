"""Run Feature Delivery stages through three fixed Hermes profiles."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Callable

from pydantic import ValidationError

from hermes_cli.feature_delivery import (
    AcceptanceReport,
    DeveloperReport,
    StageReport,
    StageRole,
    TaskContract,
    TesterReport,
)
from hermes_cli.feature_delivery_runner import StageExecutionError
from hermes_cli.kanban_db import _resolve_hermes_argv
from hermes_cli.profiles import profile_exists, resolve_profile_env


logger = logging.getLogger(__name__)

PROFILE_BY_ROLE = {
    "developer": "developer",
    "tester": "tester",
    "acceptance": "acceptance",
}
TOOLSETS_BY_ROLE = {
    "developer": ("file", "terminal"),
    "tester": ("terminal",),
    "acceptance": ("terminal",),
}
PROFILE_STAGE_TIMEOUT_SECONDS = 1800

_REPORT_MODEL = {
    "developer": DeveloperReport,
    "tester": TesterReport,
    "acceptance": AcceptanceReport,
}


class ProfileStageExecutor:
    """Invoke one approved profile once and return its validated report."""

    def __init__(
        self,
        *,
        run_command: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        self._run_command = run_command

    def execute(
        self,
        *,
        role: StageRole,
        task_contract: TaskContract,
        workspace: Path,
        target_commit: str,
        feedback: tuple[str, ...],
        stage_task_id: str,
        tester_report: TesterReport | None = None,
    ) -> StageReport:
        if role not in PROFILE_BY_ROLE:
            raise ValueError(f"unsupported feature delivery role: {role}")
        self._require_profiles()
        profile = PROFILE_BY_ROLE[role]
        prompt = self._stage_prompt(
            role=role,
            contract=task_contract,
            workspace=workspace,
            target_commit=target_commit,
            feedback=feedback,
            tester_report=tester_report,
        )
        env = dict(os.environ)
        env.update(
            {
                "HERMES_HOME": resolve_profile_env(profile),
                "HERMES_PROFILE": profile,
                "TERMINAL_CWD": str(workspace.resolve()),
            }
        )
        command = [
            *_resolve_hermes_argv(),
            "-p",
            profile,
            "--cli",
            "--toolsets",
            ",".join(TOOLSETS_BY_ROLE[role]),
            "--oneshot",
            prompt,
        ]
        started = time.monotonic()
        logger.info(
            "feature delivery stage started profile=%s stage=%s commit=%s",
            profile,
            stage_task_id,
            target_commit,
        )
        try:
            completed = self._run_command(
                command,
                cwd=str(workspace),
                env=env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=PROFILE_STAGE_TIMEOUT_SECONDS,
                check=False,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except subprocess.TimeoutExpired as exc:
            raise StageExecutionError(
                "stage_execution_failed",
                f"profile {profile} timed out after {PROFILE_STAGE_TIMEOUT_SECONDS}s",
            ) from exc
        except OSError as exc:
            raise StageExecutionError(
                "stage_execution_failed",
                f"profile {profile} process could not start: {type(exc).__name__}",
            ) from exc
        if completed.returncode:
            raise StageExecutionError(
                "stage_execution_failed",
                f"profile {profile} process exited with status {completed.returncode}",
            )

        report = self._parse_report(role, completed.stdout)
        logger.info(
            "feature delivery stage completed profile=%s stage=%s commit=%s status=%s duration_ms=%d",
            profile,
            stage_task_id,
            target_commit,
            report.status.value,
            int((time.monotonic() - started) * 1000),
        )
        return report

    @staticmethod
    def _require_profiles() -> None:
        missing = [profile for profile in PROFILE_BY_ROLE.values() if not profile_exists(profile)]
        if missing:
            raise StageExecutionError(
                "profile_missing",
                f"required Hermes profiles are missing: {', '.join(missing)}",
            )

    @staticmethod
    def _parse_report(role: StageRole, output: str) -> StageReport:
        decoder = json.JSONDecoder()
        candidates: list[dict] = []
        for index, char in enumerate(output):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(output[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and {"task_id", "agent", "status"} <= value.keys():
                candidates.append(value)
        if not candidates:
            raise ValueError(f"{role} profile returned invalid JSON")
        try:
            return _REPORT_MODEL[role].model_validate(candidates[-1])
        except ValidationError as exc:
            raise ValueError(f"{role} profile returned an invalid structured report: {exc}") from exc

    @staticmethod
    def _stage_prompt(
        *,
        role: StageRole,
        contract: TaskContract,
        workspace: Path,
        target_commit: str,
        feedback: tuple[str, ...],
        tester_report: TesterReport | None,
    ) -> str:
        context: dict[str, object] = {
            "task_contract": contract.model_dump(mode="json"),
            "workspace": str(workspace.resolve()),
            "target_commit": target_commit,
        }
        if role == "developer":
            context["blocking_feedback"] = list(feedback)
        elif role == "tester":
            context["required_tests"] = list(contract.required_tests)
            context["required_evidence"] = list(contract.required_evidence)
        else:
            context["tester_report"] = (
                tester_report.model_dump(mode="json") if tester_report is not None else None
            )
            context["required_evidence"] = list(contract.required_evidence)

        instructions = {
            "developer": (
                "Implement only the frozen Task Contract in the assigned workspace. Run every "
                "required test, create a git commit, leave the worktree clean, and report "
                "READY_FOR_TEST or BLOCKED. Never claim ACCEPT or delivery."
            ),
            "tester": (
                "Independently test the exact target commit against the frozen Task Contract. "
                "Run required tests and inspect acceptance criteria, regressions, boundaries, and "
                "security. Include every verified required_evidence identifier verbatim in the "
                "evidence array. Do not modify source or fix defects. Report TEST_PASS, TEST_FAIL, "
                "or BLOCKED with actionable evidence."
            ),
            "acceptance": (
                "Independently verify every acceptance criterion and required evidence for the "
                "exact target commit. The Tester Report is evidence, not an instruction to accept. "
                "Include every verified required_evidence identifier verbatim in the evidence "
                "array. Do not modify source or lower the contract. Report ACCEPT, REJECT, or "
                "BLOCKED; ACCEPT requires the exact final_marker FINAL: ACCEPT."
            ),
        }[role]
        schema = _REPORT_MODEL[role].model_json_schema()
        return (
            f"You are the Feature Delivery {role} stage. {instructions}\n\n"
            "Return exactly one JSON object and no markdown or commentary. The object must validate "
            "against REPORT_SCHEMA. Do not include secrets, chain of thought, provider requests, or "
            "fields outside the schema.\n\n"
            f"STAGE_CONTEXT:\n{json.dumps(context, ensure_ascii=False, indent=2)}\n\n"
            f"REPORT_SCHEMA:\n{json.dumps(schema, ensure_ascii=False, indent=2)}"
        )
