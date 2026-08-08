"""Portable evaluation-task manifests for Hermes Agent.

The format is intentionally data-only. It can be produced from local session
traces, reviewed in Git, and exported to an external runner without adding a
model tool or changing the conversation loop.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import import_module
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


_TASK_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}$")
_ALLOWED_STATUSES = frozenset({"candidate", "approved", "retired"})
_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "id",
        "status",
        "instruction",
        "source",
        "environment",
        "success",
        "forbidden",
        "skills",
        "signals",
        "provenance",
    }
)
_DETERMINISTIC_CHECK_FIELDS = {
    "tool_called": "name",
    "tool_succeeded": "name",
    "final_response_contains": "value",
    "final_response_excludes": "value",
}
_CORRECTION_RE = re.compile(
    r"^\s*(?:no\b|actually\b|that(?:'s| is) (?:wrong|not right)\b|"
    r"you (?:did not|didn't|haven't|have not|missed|forgot)\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ValidationResult:
    """Structured validation result suitable for both CLI and tests."""

    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.errors

    @property
    def ready(self) -> bool:
        return self.valid and not self.warnings


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _string_list(value: Any) -> bool:
    return isinstance(value, list) and all(_nonempty_string(item) for item in value)


def _redact_trace_text(
    value: Any,
    *,
    limit: int = 2_000,
    workspace: str | None = None,
) -> str:
    """Fail-closed secret/PII/path sanitization for candidate artifacts."""

    redact_sensitive_text = import_module("agent.redact").redact_sensitive_text
    redact_for_export = import_module(
        "agent.monitoring.redaction"
    ).redact_for_export
    text = str(value or "")
    text = text.replace(str(Path.home()), "$HOME")
    if workspace:
        text = text.replace(workspace, "$WORKSPACE")
    # The file-read pass emits non-reusable sentinels for token shapes; the
    # regular pass then catches assignment-style secrets. Monitoring redaction
    # adds fail-closed PII scrubbing.
    text = redact_sensitive_text(
        text,
        force=True,
        file_read=True,
        redact_url_credentials=True,
    )
    text = redact_sensitive_text(
        text,
        force=True,
        redact_url_credentials=True,
    )
    scrubbed = redact_for_export(text)
    return str(scrubbed or "")[:limit]


def _normalize_user_instruction(content: Any) -> str | None:
    extractor = import_module(
        "agent.skill_commands"
    ).extract_user_instruction_from_skill_message
    extracted = extractor(content)
    return extracted if isinstance(extracted, str) and extracted.strip() else None


def _task_id_for_candidate(instruction: str, tools: Sequence[str]) -> str:
    material = json.dumps(
        {"instruction": instruction, "allowed_tools": sorted(tools)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]
    return f"trace-{digest}"


def _trace_digest(
    messages: Sequence[Mapping[str, Any]],
    *,
    workspace: str | None,
) -> str:
    """Return an opaque, content-derived trace digest for deduplication."""

    digest = hashlib.sha256()
    for message in messages:
        digest.update(str(message.get("role") or "").encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(message.get("tool_name") or "").encode("utf-8"))
        digest.update(b"\0")
        content = message.get("content")
        rendered = content if isinstance(content, str) else repr(content)
        digest.update(
            _redact_trace_text(
                rendered,
                limit=4096,
                workspace=workspace,
            ).encode("utf-8")
        )
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def _tool_failed(content: Any) -> bool:
    text = str(content or "").strip()
    if not text:
        return False
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        lowered = text.casefold()
        return any(
            marker in lowered
            for marker in ("traceback", "connection refused", "permission denied")
        )
    if not isinstance(payload, Mapping):
        return False
    if payload.get("success") is False:
        return True
    exit_code = payload.get("exit_code")
    return isinstance(exit_code, int) and not isinstance(exit_code, bool) and exit_code != 0


def _tool_names(messages: Sequence[Mapping[str, Any]]) -> list[str]:
    names: set[str] = set()
    for message in messages:
        tool_name = message.get("tool_name")
        if _nonempty_string(tool_name):
            names.add(str(tool_name))
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for call in tool_calls:
            if not isinstance(call, Mapping):
                continue
            function = call.get("function")
            if isinstance(function, Mapping) and _nonempty_string(function.get("name")):
                names.add(str(function["name"]))
    return sorted(names)


def build_candidate_from_trace(
    session: Mapping[str, Any],
    messages: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Convert one session trace into a sanitized, review-required task candidate.

    The compiler records evidence and proposes verifier criteria, but never
    approves its own output. Human review must replace or confirm the checks
    before changing ``status`` to ``approved``.
    """

    session_id = str(session.get("id") or "")
    if not session_id:
        raise ValueError("session id is required")

    selected = [message for message in messages if isinstance(message, Mapping)]
    workspace = str(session.get("cwd") or "") or None
    first_instruction = next(
        (
            normalized
            for message in selected
            if message.get("role") == "user"
            for normalized in [_normalize_user_instruction(message.get("content"))]
            if normalized
        ),
        None,
    )
    if first_instruction is None:
        raise ValueError("session trace has no user instruction")

    signals: list[dict[str, Any]] = []
    seen_assistant = False
    for ordinal, message in enumerate(selected):
        role = message.get("role")
        content = message.get("content")
        if isinstance(content, str) and "[OUT-OF-BAND USER MESSAGE" in content:
            signals.append(
                {
                    "kind": "midturn_user_steering",
                    "message_ordinal": ordinal,
                }
            )
        if role == "assistant":
            seen_assistant = True
        elif role == "tool" and _tool_failed(content):
            signals.append(
                {
                    "kind": "tool_failure",
                    "message_ordinal": ordinal,
                    "tool": message.get("tool_name"),
                }
            )
        elif (
            role == "user"
            and seen_assistant
            and isinstance(content, str)
            and _CORRECTION_RE.search(content)
        ):
            signals.append(
                {
                    "kind": "user_correction",
                    "message_ordinal": ordinal,
                }
            )

    instruction = _redact_trace_text(first_instruction, workspace=workspace)
    allowed_tools = _tool_names(selected)
    source: dict[str, Any] = {
        "kind": "session_trace",
        "digest": _trace_digest(selected, workspace=workspace),
        "message_count": len(selected),
        "sanitized": True,
    }

    return {
        "schema_version": 1,
        "id": _task_id_for_candidate(instruction, allowed_tools),
        "status": "candidate",
        "instruction": instruction,
        "source": source,
        "environment": {"allowed_tools": allowed_tools},
        "success": {
            "deterministic": [],
            "judged": [
                "The requested outcome is supported by real execution evidence.",
                "The final report does not claim success before verification.",
            ],
        },
        "forbidden": [
            "Invent command, API, file, or deployment results.",
            "Treat an unverified intermediate result as task completion.",
        ],
        "skills": [],
        "signals": signals,
        "provenance": {
            "source": session.get("source"),
            "message_count": len(selected),
            "tool_call_count": session.get("tool_call_count", 0),
            "has_final_response": any(
                message.get("role") == "assistant"
                and _nonempty_string(message.get("content"))
                for message in selected
            ),
        },
    }


def _successful_tool_call(call: Mapping[str, Any]) -> bool:
    result = call.get("result")
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except (json.JSONDecodeError, TypeError, ValueError):
            return False
    if not isinstance(result, Mapping):
        return False
    if result.get("success") is False:
        return False
    exit_code = result.get("exit_code")
    if isinstance(exit_code, int) and not isinstance(exit_code, bool):
        return exit_code == 0
    return result.get("success") is True


def score_run_artifact(
    manifest: Mapping[str, Any], run: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply deterministic checks to a recorded run artifact.

    Qualitative criteria are returned for a separate evaluator; this function
    deliberately does not let the task generator grade its own prose.
    """

    validation = validate_manifest(manifest)
    if validation.errors:
        raise ValueError("invalid task manifest: " + "; ".join(validation.errors))
    if not validation.ready:
        raise ValueError("task manifest is not ready: " + "; ".join(validation.warnings))
    task_id = str(manifest.get("id"))
    if run.get("task_id") != task_id:
        raise ValueError("run.task_id does not match manifest id")

    final_response = run.get("final_response")
    if not isinstance(final_response, str):
        raise ValueError("run artifact final_response must be a string")
    raw_calls = run.get("tool_calls")
    if not isinstance(raw_calls, list):
        raise ValueError("run artifact tool_calls must be a list")
    tool_calls: list[Mapping[str, Any]] = []
    for index, call in enumerate(raw_calls):
        if not isinstance(call, Mapping) or not _nonempty_string(call.get("name")):
            raise ValueError(
                f"run artifact tool_calls[{index}] must be a mapping with a non-empty name"
            )
        tool_calls.append(call)

    deterministic = manifest.get("success", {}).get("deterministic", [])
    results: list[dict[str, Any]] = []

    allowed_tools = set(manifest["environment"]["allowed_tools"])
    disallowed_tools = sorted(
        {
            str(call["name"])
            for call in tool_calls
            if str(call["name"]) not in allowed_tools
        }
    )
    results.append(
        {
            "index": "policy.allowed_tools",
            "type": "allowed_tools",
            "passed": not disallowed_tools,
            "detail": (
                "all called tools were allowed"
                if not disallowed_tools
                else "disallowed tools called: " + ", ".join(disallowed_tools)
            ),
        }
    )

    for index, check in enumerate(deterministic):
        check_type = check["type"]
        passed = False
        if check_type == "tool_called":
            passed = any(call.get("name") == check["name"] for call in tool_calls)
        elif check_type == "tool_succeeded":
            passed = any(
                call.get("name") == check["name"] and _successful_tool_call(call)
                for call in tool_calls
            )
        elif check_type == "final_response_contains":
            passed = str(check["value"]).casefold() in final_response.casefold()
        elif check_type == "final_response_excludes":
            passed = str(check["value"]).casefold() not in final_response.casefold()
        results.append(
            {
                "index": index,
                "type": check_type,
                "passed": passed,
            }
        )

    passed_count = sum(1 for result in results if result["passed"])
    deterministic_passed = passed_count == len(results)
    judge_criteria = list(manifest.get("success", {}).get("judged", []))
    judge_criteria.extend(
        f"Verify forbidden behavior did not occur: {item}"
        for item in manifest.get("forbidden", [])
    )
    if not deterministic_passed:
        status = "failed"
        passed: bool | None = False
    elif judge_criteria:
        status = "needs_judge"
        passed = None
    else:
        status = "passed"
        passed = True

    return {
        "task_id": task_id,
        "status": status,
        "passed": passed,
        "deterministic": {"passed": passed_count, "total": len(results)},
        "checks": results,
        "judge_criteria": judge_criteria,
    }


def validate_manifest(manifest: Mapping[str, Any]) -> ValidationResult:
    """Validate a Hermes evaluation-task manifest.

    Candidates may be structurally valid without being runnable. ``ready`` is
    true only for approved tasks whose deterministic or judged success checks
    have been filled in and whose source is marked sanitized.
    """

    errors: list[str] = []
    warnings: list[str] = []

    unknown_fields = sorted(set(manifest) - _TOP_LEVEL_FIELDS)
    if unknown_fields:
        errors.append("unknown top-level fields: " + ", ".join(unknown_fields))

    if manifest.get("schema_version") != 1:
        errors.append("schema_version must be 1")

    task_id = manifest.get("id")
    if not isinstance(task_id, str) or not _TASK_ID_RE.fullmatch(task_id):
        errors.append(
            "id must be 3-128 lowercase characters using letters, numbers, '.', '_' or '-'"
        )

    status = manifest.get("status")
    if status not in _ALLOWED_STATUSES:
        errors.append("status must be one of: candidate, approved, retired")
    elif status != "approved":
        warnings.append("status must be 'approved'")

    if not _nonempty_string(manifest.get("instruction")):
        errors.append("instruction must be a non-empty string")

    source = manifest.get("source")
    if not isinstance(source, Mapping):
        errors.append("source must be a mapping")
    else:
        extra_source_fields = sorted(
            set(source) - {"kind", "sanitized", "digest", "message_count"}
        )
        if extra_source_fields:
            errors.append(
                "source has unknown fields: " + ", ".join(extra_source_fields)
            )
        if not _nonempty_string(source.get("kind")):
            errors.append("source.kind must be a non-empty string")
        digest = source.get("digest")
        if digest is not None and (
            not isinstance(digest, str)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
        ):
            errors.append(
                "source.digest must be a sha256 digest for session_trace tasks"
            )
        message_count = source.get("message_count")
        if message_count is not None and (
            not isinstance(message_count, int)
            or isinstance(message_count, bool)
            or message_count <= 0
        ):
            errors.append("source.message_count must be a positive integer")
        if source.get("kind") == "session_trace":
            if digest is None:
                errors.append(
                    "source.digest must be a sha256 digest for session_trace tasks"
                )
            if message_count is None:
                errors.append("source.message_count must be a positive integer")
        if source.get("sanitized") is not True:
            errors.append("source.sanitized must be true")

    environment = manifest.get("environment")
    if not isinstance(environment, Mapping):
        errors.append("environment must be a mapping")
    else:
        extra_environment_fields = sorted(set(environment) - {"allowed_tools"})
        if extra_environment_fields:
            errors.append(
                "environment has unknown fields: "
                + ", ".join(extra_environment_fields)
            )
        allowed_tools = environment.get("allowed_tools")
        if not _string_list(allowed_tools):
            errors.append("environment.allowed_tools must be a list of non-empty strings")
        elif isinstance(allowed_tools, list) and len(allowed_tools) != len(
            set(allowed_tools)
        ):
            errors.append("environment.allowed_tools must not contain duplicates")

    success = manifest.get("success")
    if not isinstance(success, Mapping):
        errors.append("success must be a mapping")
    else:
        extra_success_fields = sorted(set(success) - {"deterministic", "judged"})
        if extra_success_fields:
            errors.append(
                "success has unknown fields: " + ", ".join(extra_success_fields)
            )
        deterministic = success.get("deterministic")
        judged = success.get("judged")
        if not isinstance(deterministic, list):
            errors.append("success.deterministic must be a list")
        else:
            for index, check in enumerate(deterministic):
                if not isinstance(check, Mapping):
                    errors.append(f"success.deterministic[{index}] must be a mapping")
                    continue
                check_type = check.get("type")
                required_field = (
                    _DETERMINISTIC_CHECK_FIELDS.get(check_type)
                    if isinstance(check_type, str)
                    else None
                )
                if required_field is None:
                    allowed = ", ".join(sorted(_DETERMINISTIC_CHECK_FIELDS))
                    errors.append(
                        f"success.deterministic[{index}].type must be one of: {allowed}"
                    )
                elif not _nonempty_string(check.get(required_field)):
                    errors.append(
                        f"success.deterministic[{index}].{required_field} must be a non-empty string"
                    )
                else:
                    extra_fields = sorted(set(check) - {"type", required_field})
                    if extra_fields:
                        errors.append(
                            f"success.deterministic[{index}] has unknown fields: "
                            + ", ".join(extra_fields)
                        )
        if not _string_list(judged):
            errors.append("success.judged must be a list of non-empty strings")
        if isinstance(deterministic, list) and isinstance(judged, list):
            if not deterministic and not judged:
                warnings.append("at least one success check is required before the task is ready")

    for field in ("forbidden", "skills"):
        if not _string_list(manifest.get(field)):
            errors.append(f"{field} must be a list of non-empty strings")

    signals = manifest.get("signals")
    if signals is not None and not (
        isinstance(signals, list)
        and all(isinstance(signal, Mapping) for signal in signals)
    ):
        errors.append("signals must be a list of mappings")

    provenance = manifest.get("provenance")
    if provenance is not None and not isinstance(provenance, Mapping):
        errors.append("provenance must be a mapping")

    return ValidationResult(tuple(errors), tuple(warnings))
