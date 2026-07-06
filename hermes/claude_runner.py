"""Claude Agent SDK runner scaffolding and approval-gated stages."""

from __future__ import annotations

import json
import inspect
import logging
import os
import re
import subprocess
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query,
)


ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"
DESIGN_STAGE_SYSTEM_PROMPT = """You are the Hermes design-stage agent.

Produce a build spec as a single JSON object only. Do not include markdown,
code fences, commentary, or any text outside the JSON object.

The JSON object must contain exactly these keys:
- summary: string
- files_to_change: array of strings
- steps: array of strings
- risks: array of strings
- test_plan: array of strings
"""

BUILD_STAGE_SYSTEM_PROMPT = """You are the Hermes build-stage agent.

Implement only the approved DesignSpec in the target repository. Keep the
work minimal, reviewable, and test-backed. Do not commit, do not push, do not
create pull requests, and do not alter git remotes or credentials.

Bash tool is allowed only for running tests and linters. Do not use Bash to
edit files, commit, push, install unrelated dependencies, deploy, change
permissions, delete repository content, or perform network/credential actions.
Use Read, Edit, Write, Glob, and Grep for repository inspection and edits.

When finished, summarize what changed and which tests/linters were run. The
caller will collect and send the git diff for human review.
"""

REVIEW_STAGE_SYSTEM_PROMPT = """You are the Hermes review-stage agent.

verify the diff implements the spec, flag deviations, security issues, and
missing tests. Use read-only repository inspection only. Do not edit files, do
not run commands, do not commit, and do not push.

Return exactly one JSON object with keys:
- verdict: "pass" or "fail"
- findings: array of strings
"""

BuildProgressCallback = Callable[[str], None | Awaitable[None]]

logger = logging.getLogger(__name__)


class RunStatus(StrEnum):
    """Lifecycle states for an approval-gated Claude stage run."""

    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    BUILDING = "building"
    DONE = "done"
    FAILED = "failed"


@dataclass(slots=True)
class ClaudeRunnerConfig:
    """Configuration for a future Claude Agent SDK stage runner."""

    model_name: str
    allowed_tools: list[str] = field(default_factory=list)
    permission_mode: str | None = None
    cwd: Path | None = None
    max_turns: int | None = None

    @property
    def anthropic_api_key(self) -> str | None:
        """Return the Anthropic API key from the process environment only."""

        return os.getenv(ANTHROPIC_API_KEY_ENV)


@dataclass(frozen=True, slots=True)
class DesignSpec:
    """Parsed build specification returned by the Claude design stage."""

    summary: str
    files_to_change: list[str]
    steps: list[str]
    risks: list[str]
    test_plan: list[str]


@dataclass(frozen=True, slots=True)
class BuildStageResult:
    """Result of a Claude build-stage run."""

    diff: str
    transcript: str
    cwd: Path


@dataclass(frozen=True, slots=True)
class ReviewStageResult:
    """Parsed verdict returned by the Claude review stage."""

    verdict: str
    findings: list[str]


@dataclass(frozen=True, slots=True)
class DesignRunRecord:
    """Durable record for one approval-gated design/build run."""

    run_id: str
    requirements: str
    design_spec: DesignSpec
    status: RunStatus
    approver_identity: str | None = None
    rejection_reason: str | None = None
    created_at: str = ""
    updated_at: str = ""
    approved_at: str | None = None
    rejected_at: str | None = None
    building_at: str | None = None
    done_at: str | None = None
    failed_at: str | None = None


class DesignStageParseError(RuntimeError):
    """Raised when the Claude design-stage response cannot be parsed."""


class ReviewStageParseError(RuntimeError):
    """Raised when the Claude review-stage response cannot be parsed."""


class BuildApprovalRequiredError(RuntimeError):
    """Raised when a build stage is started before design approval."""


class BuildConfigError(RuntimeError):
    """Raised when build-stage configuration is missing or invalid."""


class DesignRunNotFoundError(KeyError):
    """Raised when a requested design run record does not exist."""


class InvalidRunStatusTransitionError(RuntimeError):
    """Raised when a run status transition is not allowed."""


class DesignRunStore:
    """JSON-file store for approval-gated design/build run records."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else _default_run_store_root()
        self.root.mkdir(parents=True, exist_ok=True)

    def create_pending_run(self, requirements: str, design_spec: DesignSpec) -> DesignRunRecord:
        now = _utc_now()
        record = DesignRunRecord(
            run_id=str(uuid.uuid4()),
            requirements=requirements,
            design_spec=design_spec,
            status=RunStatus.PENDING_APPROVAL,
            created_at=now,
            updated_at=now,
        )
        self.save(record)
        return record

    def get(self, run_id: str) -> DesignRunRecord:
        path = self._path_for(run_id)
        if not path.exists():
            raise DesignRunNotFoundError(run_id)
        return _record_from_dict(json.loads(path.read_text(encoding="utf-8")))

    def save(self, record: DesignRunRecord) -> None:
        path = self._path_for(record.run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = _record_to_dict(record)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)

    def approve(self, run_id: str, approver_identity: str) -> DesignRunRecord:
        record = self.get(run_id)
        if record.status is not RunStatus.PENDING_APPROVAL:
            raise InvalidRunStatusTransitionError(
                f"Run {run_id} cannot be approved from status {record.status}"
            )
        now = _utc_now()
        updated = _replace_record(
            record,
            status=RunStatus.APPROVED,
            approver_identity=approver_identity,
            approved_at=now,
            updated_at=now,
        )
        self.save(updated)
        return updated

    def reject(self, run_id: str, approver_identity: str, reason: str) -> DesignRunRecord:
        record = self.get(run_id)
        if record.status is not RunStatus.PENDING_APPROVAL:
            raise InvalidRunStatusTransitionError(
                f"Run {run_id} cannot be rejected from status {record.status}"
            )
        reason = reason.strip()
        if not reason:
            raise ValueError("A rejection reason is required")
        now = _utc_now()
        updated = _replace_record(
            record,
            status=RunStatus.REJECTED,
            approver_identity=approver_identity,
            rejection_reason=reason,
            rejected_at=now,
            updated_at=now,
        )
        self.save(updated)
        return updated

    def mark_building(self, run_id: str) -> DesignRunRecord:
        record = self.get(run_id)
        if record.status is not RunStatus.APPROVED:
            raise BuildApprovalRequiredError(
                f"Run {run_id} must be approved before build can start; status={record.status}"
            )
        now = _utc_now()
        updated = _replace_record(
            record,
            status=RunStatus.BUILDING,
            building_at=now,
            updated_at=now,
        )
        self.save(updated)
        return updated

    def mark_done(self, run_id: str) -> DesignRunRecord:
        record = self.get(run_id)
        now = _utc_now()
        updated = _replace_record(record, status=RunStatus.DONE, done_at=now, updated_at=now)
        self.save(updated)
        return updated

    def mark_failed(self, run_id: str) -> DesignRunRecord:
        record = self.get(run_id)
        now = _utc_now()
        updated = _replace_record(record, status=RunStatus.FAILED, failed_at=now, updated_at=now)
        self.save(updated)
        return updated

    def _path_for(self, run_id: str) -> Path:
        safe_run_id = re.sub(r"[^A-Za-z0-9_.-]", "_", run_id)
        return self.root / f"{safe_run_id}.json"


class ClaudeStageRunner:
    """Runs Claude Agent SDK stages for Hermes workflows."""

    def __init__(self, run_store: DesignRunStore | None = None) -> None:
        self.run_store = run_store or DesignRunStore()

    async def run_design_stage(self, requirements: str) -> DesignSpec:
        """Run the Claude design stage and return a parsed build spec.

        The stage asks Claude for a single JSON build spec, retries once with a
        corrective prompt if parsing fails, then raises DesignStageParseError.
        """

        first_text = await self._collect_design_stage_text(requirements)
        try:
            return _parse_design_spec(first_text)
        except DesignStageParseError as first_error:
            corrective_prompt = _build_corrective_prompt(requirements, first_text, first_error)
            retry_text = await self._collect_design_stage_text(corrective_prompt)
            try:
                return _parse_design_spec(retry_text)
            except DesignStageParseError as retry_error:
                raise DesignStageParseError(
                    "Claude design stage did not return parseable DesignSpec JSON after retry"
                ) from retry_error

    async def run_design_stage_with_approval(self, requirements: str) -> DesignRunRecord:
        """Run design and persist a pending-approval run record."""

        design_spec = await self.run_design_stage(requirements)
        return self.run_store.create_pending_run(requirements, design_spec)

    def start_build_stage(self, run_id: str) -> DesignRunRecord:
        """Start the build stage only after the run has been approved."""

        return self.run_store.mark_building(run_id)

    async def run_build_stage(
        self,
        spec: DesignSpec,
        progress_callback: BuildProgressCallback | None = None,
    ) -> BuildStageResult:
        """Run the approved Claude build stage and return the resulting git diff.

        The build stage is intentionally constrained: Bash is allowed only for
        tests and linters via the system prompt, and this method never commits or
        pushes. Progress text emitted by Claude is forwarded to
        ``progress_callback`` so gateway adapters can stream it to Telegram.
        """

        cwd = _target_repo_path_from_config()
        prompt = _build_stage_prompt(spec)
        options = ClaudeAgentOptions(
            setting_sources=[],
            model="sonnet",
            allowed_tools=["Read", "Edit", "Write", "Bash", "Glob", "Grep"],
            permission_mode="acceptEdits",
            cwd=str(cwd),
            system_prompt=BUILD_STAGE_SYSTEM_PROMPT,
        )

        transcript: list[str] = []
        await _emit_progress(progress_callback, f"🏗️ Claude build stage started in `{cwd}`.")
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    text = _text_from_stream_block(block)
                    if text:
                        transcript.append(text)
                        await _emit_progress(progress_callback, text)
            elif isinstance(message, ResultMessage):
                logger.info(
                    "Claude build stage usage=%s cost_usd=%s",
                    message.usage,
                    message.total_cost_usd,
                )
                if message.is_error:
                    await _emit_progress(progress_callback, "⚠️ Claude build stage reported an error result.")

        diff = _git_diff(cwd)
        if diff.strip():
            await _emit_progress(progress_callback, _format_diff_for_review(diff))
        else:
            await _emit_progress(progress_callback, "✅ Claude build stage completed with no git diff.")
        return BuildStageResult(diff=diff, transcript="".join(transcript), cwd=cwd)

    async def run_review_stage(
        self,
        spec: DesignSpec,
        diff: str,
        progress_callback: BuildProgressCallback | None = None,
    ) -> ReviewStageResult:
        """Run a read-only Claude review over the approved spec and git diff."""

        cwd = _target_repo_path_from_config()
        prompt = _review_stage_prompt(spec, diff)
        options = ClaudeAgentOptions(
            setting_sources=[],
            model="sonnet",
            allowed_tools=["Read", "Glob", "Grep"],
            permission_mode="dontAsk",
            cwd=str(cwd),
            system_prompt=REVIEW_STAGE_SYSTEM_PROMPT,
        )

        text_blocks: list[str] = []
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    text = _text_from_stream_block(block)
                    if text:
                        text_blocks.append(text)
            elif isinstance(message, ResultMessage):
                logger.info(
                    "Claude review stage usage=%s cost_usd=%s",
                    message.usage,
                    message.total_cost_usd,
                )

        result = _parse_review_result("".join(text_blocks).strip())
        await _emit_progress(progress_callback, _format_review_for_telegram(result, diff))
        return result

    async def _collect_design_stage_text(self, prompt: str) -> str:
        options = ClaudeAgentOptions(
            setting_sources=[],
            model="opus",
            allowed_tools=["Read", "Grep", "Glob"],
            permission_mode="dontAsk",
            max_turns=15,
            system_prompt=DESIGN_STAGE_SYSTEM_PROMPT,
        )

        text_blocks: list[str] = []
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        text_blocks.append(block.text)
            elif isinstance(message, ResultMessage):
                logger.info(
                    "Claude design stage usage=%s cost_usd=%s",
                    message.usage,
                    message.total_cost_usd,
                )

        return "".join(text_blocks).strip()


def format_design_approval_message(record: DesignRunRecord) -> str:
    """Render a concise user-facing approval prompt for a design run."""

    spec = record.design_spec
    files = ", ".join(spec.files_to_change) if spec.files_to_change else "None listed"
    steps = "\n".join(f"- {step}" for step in spec.steps[:8]) or "- None listed"
    risks = "\n".join(f"- {risk}" for risk in spec.risks[:5]) or "- None listed"
    tests = "\n".join(f"- {test}" for test in spec.test_plan[:5]) or "- None listed"
    return (
        "🧭 Claude design stage complete\n\n"
        f"Run ID: {record.run_id}\n"
        f"Status: {record.status.value}\n\n"
        f"Summary: {spec.summary}\n\n"
        f"Files to change: {files}\n\n"
        f"Steps:\n{steps}\n\n"
        f"Risks:\n{risks}\n\n"
        f"Test plan:\n{tests}\n\n"
        f"Reply `approve {record.run_id}` to proceed, or `reject {record.run_id} <reason>` to stop."
    )


def approve_design_run(run_id: str, approver_identity: str, store: DesignRunStore | None = None) -> DesignRunRecord:
    return (store or DesignRunStore()).approve(run_id, approver_identity)


def reject_design_run(
    run_id: str,
    approver_identity: str,
    reason: str,
    store: DesignRunStore | None = None,
) -> DesignRunRecord:
    return (store or DesignRunStore()).reject(run_id, approver_identity, reason)


def _build_stage_prompt(spec: DesignSpec) -> str:
    return (
        "Approved DesignSpec JSON follows. Implement this spec exactly.\n\n"
        f"{json.dumps(asdict(spec), indent=2, sort_keys=True)}"
    )


def _review_stage_prompt(spec: DesignSpec, diff: str) -> str:
    return (
        "Approved DesignSpec JSON:\n"
        f"{json.dumps(asdict(spec), indent=2, sort_keys=True)}\n\n"
        "Git diff to review:\n"
        f"```diff\n{diff}\n```"
    )


async def _emit_progress(callback: BuildProgressCallback | None, message: str) -> None:
    if callback is None or not message:
        return
    result = callback(message)
    if inspect.isawaitable(result):
        await result


def _text_from_stream_block(block: Any) -> str:
    text = getattr(block, "text", None)
    if isinstance(text, str):
        return text
    return ""


def _format_diff_for_review(diff: str) -> str:
    return f"✅ Claude build stage complete. Git diff for review:\n\n```diff\n{diff}\n```"


def _load_hermes_config() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config
        config = load_config()
        return config if isinstance(config, dict) else {}
    except Exception as exc:
        raise BuildConfigError(f"Unable to load Hermes config: {exc}") from exc


def _target_repo_path_from_config() -> Path:
    config = _load_hermes_config()
    candidates: list[Any] = []
    for section_name in ("claude_runner", "claude", "agent"):
        section = config.get(section_name)
        if isinstance(section, dict):
            candidates.extend(
                section.get(key)
                for key in ("target_repo_path", "target_repo", "repo_path", "cwd")
            )
    terminal = config.get("terminal")
    if isinstance(terminal, dict):
        candidates.append(terminal.get("cwd"))

    for raw in candidates:
        if isinstance(raw, str) and raw.strip():
            path = Path(raw).expanduser().resolve()
            if not path.exists() or not path.is_dir():
                raise BuildConfigError(f"Configured Claude build target repo path is not a directory: {path}")
            return path

    raise BuildConfigError(
        "Missing Claude build target repo path in config. Set claude_runner.target_repo_path."
    )


def _git_diff(cwd: Path) -> str:
    tracked = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--"],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    if tracked.returncode != 0:
        raise RuntimeError(f"git diff failed: {tracked.stderr.strip() or tracked.stdout.strip()}")

    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    if untracked.returncode != 0:
        raise RuntimeError(f"git ls-files failed: {untracked.stderr.strip() or untracked.stdout.strip()}")

    parts = [tracked.stdout]
    for rel_path in [line for line in untracked.stdout.splitlines() if line.strip()]:
        untracked_diff = subprocess.run(
            ["git", "diff", "--no-index", "--binary", "--", "/dev/null", rel_path],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
        )
        # git diff --no-index exits 1 when differences exist; that is success.
        if untracked_diff.returncode not in (0, 1):
            raise RuntimeError(
                f"git diff for untracked file {rel_path} failed: "
                f"{untracked_diff.stderr.strip() or untracked_diff.stdout.strip()}"
            )
        parts.append(untracked_diff.stdout)
    return "".join(parts)


def _build_corrective_prompt(
    requirements: str, previous_response: str, error: DesignStageParseError
) -> str:
    return (
        "Your previous response could not be parsed as the required JSON. "
        f"Parse error: {error}\n\n"
        "Return only one valid JSON object with exactly these keys: "
        "summary, files_to_change, steps, risks, test_plan. "
        "Do not include markdown fences or explanatory text.\n\n"
        f"Original requirements:\n{requirements}\n\n"
        f"Previous response:\n{previous_response}"
    )


def _strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return stripped


def _parse_design_spec(text: str) -> DesignSpec:
    payload = _strip_markdown_fences(text)
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise DesignStageParseError(f"Invalid JSON: {exc.msg}") from exc

    if not isinstance(raw, dict):
        raise DesignStageParseError("Design spec JSON must be an object")

    required_keys = {"summary", "files_to_change", "steps", "risks", "test_plan"}
    missing_keys = required_keys - raw.keys()
    if missing_keys:
        raise DesignStageParseError(f"Design spec missing required keys: {sorted(missing_keys)}")

    return DesignSpec(
        summary=_require_string(raw, "summary"),
        files_to_change=_require_string_list(raw, "files_to_change"),
        steps=_require_string_list(raw, "steps"),
        risks=_require_string_list(raw, "risks"),
        test_plan=_require_string_list(raw, "test_plan"),
    )


def _parse_review_result(text: str) -> ReviewStageResult:
    payload = _strip_markdown_fences(text)
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ReviewStageParseError(f"Invalid JSON: {exc.msg}") from exc
    if not isinstance(raw, dict):
        raise ReviewStageParseError("Review result JSON must be an object")
    missing = {"verdict", "findings"} - raw.keys()
    if missing:
        raise ReviewStageParseError(f"Review result missing required keys: {sorted(missing)}")
    verdict = _require_review_verdict(raw)
    findings = raw["findings"]
    if not isinstance(findings, list) or not all(isinstance(item, str) for item in findings):
        raise ReviewStageParseError("Review result key 'findings' must be an array of strings")
    return ReviewStageResult(verdict=verdict, findings=findings)


def _require_review_verdict(raw: dict[str, Any]) -> str:
    verdict = raw["verdict"]
    if verdict not in {"pass", "fail"}:
        raise ReviewStageParseError("Review result key 'verdict' must be 'pass' or 'fail'")
    return verdict


def _format_review_for_telegram(result: ReviewStageResult, diff: str) -> str:
    findings = "\n".join(f"- {finding}" for finding in result.findings) or "- None"
    return (
        f"🔎 Review verdict: {result.verdict}\n\n"
        f"Findings:\n{findings}\n\n"
        "Human approval is still required before anything is committed.\n\n"
        f"Git diff for review:\n```diff\n{diff}\n```"
    )


def _require_string(raw: dict[str, Any], key: str) -> str:
    value = raw[key]
    if not isinstance(value, str):
        raise DesignStageParseError(f"Design spec key {key!r} must be a string")
    return value


def _require_string_list(raw: dict[str, Any], key: str) -> list[str]:
    value = raw[key]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise DesignStageParseError(f"Design spec key {key!r} must be an array of strings")
    return value


def _record_to_dict(record: DesignRunRecord) -> dict[str, Any]:
    payload = asdict(record)
    payload["status"] = record.status.value
    return payload


def _record_from_dict(raw: dict[str, Any]) -> DesignRunRecord:
    design_raw = raw.get("design_spec") or {}
    return DesignRunRecord(
        run_id=str(raw["run_id"]),
        requirements=str(raw.get("requirements", "")),
        design_spec=DesignSpec(
            summary=str(design_raw.get("summary", "")),
            files_to_change=list(design_raw.get("files_to_change") or []),
            steps=list(design_raw.get("steps") or []),
            risks=list(design_raw.get("risks") or []),
            test_plan=list(design_raw.get("test_plan") or []),
        ),
        status=RunStatus(str(raw.get("status", RunStatus.PENDING_APPROVAL.value))),
        approver_identity=raw.get("approver_identity"),
        rejection_reason=raw.get("rejection_reason"),
        created_at=str(raw.get("created_at", "")),
        updated_at=str(raw.get("updated_at", "")),
        approved_at=raw.get("approved_at"),
        rejected_at=raw.get("rejected_at"),
        building_at=raw.get("building_at"),
        done_at=raw.get("done_at"),
        failed_at=raw.get("failed_at"),
    )


def _replace_record(record: DesignRunRecord, **changes: Any) -> DesignRunRecord:
    payload = _record_to_dict(record)
    payload.update(changes)
    if isinstance(payload.get("status"), RunStatus):
        payload["status"] = payload["status"].value
    return _record_from_dict(payload)


def _default_run_store_root() -> Path:
    try:
        from hermes_constants import get_hermes_home
        return get_hermes_home() / "claude_runs"
    except Exception:
        return Path.home() / ".hermes" / "claude_runs"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
