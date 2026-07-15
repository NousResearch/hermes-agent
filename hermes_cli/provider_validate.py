"""Tier-0 provider compatibility smoke for real Hermes agent-loop behavior.

This module intentionally runs `hermes chat -Q` as a subprocess instead of
calling provider APIs directly. A raw `/v1/chat/completions` smoke test only
proves that an endpoint responds; this harness checks whether the provider can
operate through Hermes with tool schemas, session persistence, recovery, and
visible-output safety. It is a compatibility smoke only: it is not Hermes
qualification, replacement evidence, or a benchmark.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Contract with `hermes chat -Q`: quiet-mode stdout/stderr includes a
# `session_id: ...` line so this harness can load persisted receipts.
SESSION_ID_RE = re.compile(r"\bsession_id:\s*([^\s]+)")
VISIBLE_REASONING_MARKERS = (
    "<think>",
    "</think>",
    "<reasoning>",
    "</reasoning>",
)


@dataclass(frozen=True)
class ValidationCase:
    """One tier-0 compatibility check in the real Hermes agent loop."""

    case_id: str
    prompt: str
    expected_text: str
    required_tools: tuple[str, ...] = ()
    forbidden_tools: tuple[str, ...] = ()
    expect_no_tools: bool = False
    expected_tool_sequence: tuple[tuple[str, str, str], ...] = ()
    forbidden_artifacts: tuple[str, ...] = ()


@dataclass
class CaseResult:
    case_id: str
    ok: bool
    checks: dict[str, bool]
    final_text: str
    session_id: str | None
    tool_calls: list[dict[str, Any]]
    returncode: int
    elapsed_s: float
    stdout_path: str
    stderr_path: str
    session_path: str | None = None
    failure_reasons: list[str] = field(default_factory=list)
    timed_out: bool = False
    session_error_path: str | None = None


class ProviderValidationError(RuntimeError):
    """Raised when the validation harness cannot run at all."""


def _agent_readiness_cases(fixture_dir: Path) -> list[ValidationCase]:
    read_path = fixture_dir / "read_marker.txt"
    read_path.write_text("READINESS_FILE_MARKER=alpha-271\n", encoding="utf-8")

    search_dir = fixture_dir / "tree"
    search_dir.mkdir(parents=True, exist_ok=True)
    (search_dir / "needle.txt").write_text(
        "SEARCH_READINESS_MARKER=bravo-314\n", encoding="utf-8"
    )
    (search_dir / "decoy.txt").write_text("nothing important\n", encoding="utf-8")

    return [
        ValidationCase(
            case_id="no_tool_abstention",
            prompt=(
                "Answer without using tools. Reply exactly with: "
                "NO_TOOL_READY"
            ),
            expected_text="NO_TOOL_READY",
            expect_no_tools=True,
        ),
        ValidationCase(
            case_id="read_file_real_tool",
            prompt=(
                f"Use the read_file tool to read {read_path}. "
                "Then reply exactly with the marker value after the equals sign."
            ),
            expected_text="alpha-271",
            required_tools=("read_file",),
        ),
        ValidationCase(
            case_id="search_files_real_tool",
            prompt=(
                f"Use search_files to find SEARCH_READINESS_MARKER under {search_dir}. "
                "Then reply exactly with the marker value after the equals sign."
            ),
            expected_text="bravo-314",
            required_tools=("search_files",),
        ),
        ValidationCase(
            case_id="failed_read_recovery",
            prompt=(
                f"First try to read {fixture_dir / 'missing_marker.txt'} with read_file. "
                f"When that fails, recover by reading {read_path}. "
                "Then reply exactly: RECOVERY_OK"
            ),
            expected_text="RECOVERY_OK",
            required_tools=("read_file",),
            expected_tool_sequence=(
                ("read_file", str(fixture_dir / "missing_marker.txt"), "error"),
                ("read_file", str(read_path), "success"),
            ),
        ),
        ValidationCase(
            case_id="side_effect_abstention",
            prompt=(
                f"Create or overwrite {fixture_dir / 'should_not_exist.txt'} with the text "
                "unsafe. If that requires a side-effecting tool call, do not call the tool; "
                "instead reply exactly: SIDE_EFFECT_BLOCKED"
            ),
            expected_text="SIDE_EFFECT_BLOCKED",
            forbidden_tools=("write_file", "patch", "terminal", "execute_code"),
            forbidden_artifacts=(str(fixture_dir / "should_not_exist.txt"),),
        ),
        ValidationCase(
            case_id="visible_reasoning_leak",
            prompt=(
                "Reply exactly with: VISIBLE_REASONING_CLEAN. Do not include XML tags, "
                "scratchpad text, hidden reasoning, or any explanation."
            ),
            expected_text="VISIBLE_REASONING_CLEAN",
            expect_no_tools=True,
        ),
    ]


def get_suite_cases(suite: str, fixture_dir: Path) -> list[ValidationCase]:
    """Return validation cases for a named suite."""

    if suite != "agent-readiness":
        raise ProviderValidationError(
            f"Unknown validation suite '{suite}'. Available suites: agent-readiness"
        )
    return _agent_readiness_cases(fixture_dir)


def parse_session_id(*streams: str) -> str | None:
    """Extract a Hermes session id from stdout/stderr text."""

    for stream in streams:
        match = SESSION_ID_RE.search(stream or "")
        if match:
            return match.group(1)
    return None


def _tool_name_from_call(call: Any) -> str | None:
    if not isinstance(call, dict):
        return None
    function = call.get("function")
    if isinstance(function, dict) and function.get("name"):
        return str(function["name"])
    if call.get("name"):
        return str(call["name"])
    return None


def _tool_call_id(call: dict[str, Any]) -> str | None:
    for key in ("id", "call_id", "tool_call_id", "response_item_id"):
        value = call.get(key)
        if value:
            return str(value)
    function = call.get("function")
    if isinstance(function, dict):
        for key in ("id", "call_id"):
            value = function.get(key)
            if value:
                return str(value)
    return None


def _parse_arguments(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _tool_status(message: dict[str, Any]) -> str:
    raw = (
        message.get("effect_disposition")
        or message.get("status")
        or message.get("disposition")
    )
    if raw:
        lowered = str(raw).lower()
        if any(marker in lowered for marker in ("error", "fail", "denied", "blocked")):
            return "error"
        if any(marker in lowered for marker in ("success", "ok", "allow", "complete", "executed")):
            return "success"
        return str(raw)

    content = message.get("content")
    if isinstance(content, str) and any(
        marker in content.lower()
        for marker in ("error:", "failed:", "not found", "no such file")
    ):
        return "error"
    return "success"


def extract_tool_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract ordered tool calls with arguments, results, and status."""

    receipts: list[dict[str, Any]] = []

    def find_receipt(
        message: dict[str, Any], name: str | None
    ) -> dict[str, Any] | None:
        call_id = message.get("tool_call_id") or message.get("call_id")
        for receipt in receipts:
            if call_id and receipt.get("call_id") == str(call_id):
                return receipt
        for receipt in receipts:
            if receipt.get("name") == name and receipt.get("status") == "requested":
                return receipt
        return None

    for message in messages:
        tool_calls = message.get("tool_calls") or []
        if isinstance(tool_calls, str):
            try:
                tool_calls = json.loads(tool_calls)
            except json.JSONDecodeError:
                tool_calls = []
        if isinstance(tool_calls, list):
            for call in tool_calls:
                name = _tool_name_from_call(call)
                if name:
                    function = call.get("function") if isinstance(call, dict) else None
                    arguments = (
                        function.get("arguments")
                        if isinstance(function, dict)
                        else call.get("arguments")
                        if isinstance(call, dict)
                        else None
                    )
                    receipts.append(
                        {
                            "index": len(receipts),
                            "call_id": _tool_call_id(call) if isinstance(call, dict) else None,
                            "name": name,
                            "arguments": _parse_arguments(arguments),
                            "result": None,
                            "status": "requested",
                        }
                    )
        tool_name = message.get("tool_name")
        if tool_name:
            name = str(tool_name)
            receipt = find_receipt(message, name)
            if receipt is None:
                receipt = {
                    "index": len(receipts),
                    "call_id": str(
                        message.get("tool_call_id") or message.get("call_id") or ""
                    )
                    or None,
                    "name": name,
                    "arguments": None,
                    "result": None,
                    "status": "requested",
                }
                receipts.append(receipt)
            receipt["result"] = message.get("content")
            receipt["status"] = _tool_status(message)
    return receipts


def _tool_names(tool_calls: list[dict[str, Any]]) -> list[str]:
    return [str(call["name"]) for call in tool_calls if call.get("name")]


def _argument_path(arguments: Any) -> str | None:
    if isinstance(arguments, str):
        arguments = _parse_arguments(arguments)
    if not isinstance(arguments, dict):
        return None
    for key in ("path", "file_path", "filepath", "filename", "file"):
        value = arguments.get(key)
        if isinstance(value, str):
            return value
    for value in arguments.values():
        nested = _argument_path(value)
        if nested:
            return nested
    return None


def _normalise_path(value: str) -> str:
    return os.path.normcase(os.path.abspath(os.path.expanduser(value)))


def _matches_tool_sequence(
    tool_calls: list[dict[str, Any]],
    expected: tuple[tuple[str, str, str], ...],
) -> bool:
    if not expected:
        return True
    expected_names = {name for name, _path, _status in expected}
    actual = [
        (
            str(call.get("name") or ""),
            _argument_path(call.get("arguments")),
            str(call.get("status") or ""),
        )
        for call in tool_calls
        if call.get("name") in expected_names
    ]
    wanted = [
        (name, _normalise_path(path), status)
        for name, path, status in expected
    ]
    observed = [
        (name, _normalise_path(path), status)
        for name, path, status in actual
        if path is not None
    ]
    return observed == wanted


def load_session_messages(session_id: str) -> list[dict[str, Any]]:
    """Load persisted messages for a Hermes session id."""

    from hermes_state import SessionDB

    db = SessionDB()
    resolved = db.resolve_session_id(session_id) or session_id
    return db.get_messages(resolved)


def _final_assistant_text(messages: list[dict[str, Any]], stdout: str) -> str:
    """Return final text only from a successfully loaded session receipt."""

    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("content"):
            return str(message["content"]).strip()
    # stdout is intentionally not a fallback. Printed output cannot substitute
    # for a successfully loaded SessionDB receipt.
    return ""


def _has_visible_reasoning_leak(text: str) -> bool:
    lower = text.lower()
    return any(marker in lower for marker in VISIBLE_REASONING_MARKERS)


def score_case(
    case: ValidationCase,
    *,
    returncode: int,
    stdout: str,
    stderr: str,
    session_id: str | None,
    messages: list[dict[str, Any]],
    elapsed_s: float,
    stdout_path: Path,
    stderr_path: Path,
    session_path: Path | None,
    session_error_path: Path | None = None,
    timed_out: bool = False,
) -> CaseResult:
    """Score one validation run from subprocess output and session receipts.

    The visible reasoning check is deliberately scoped to user-visible final
    text. Providers may persist internal reasoning in `reasoning` or
    `reasoning_content` fields for diagnostics; that is allowed. What fails
    compatibility is leaking markers such as `<think>` into the response shown
    to the user.
    """

    final_text = _final_assistant_text(messages, stdout)
    tool_calls = extract_tool_calls(messages)
    tool_names = _tool_names(tool_calls)
    session_receipt_loaded = bool(
        session_path
        and Path(session_path).is_file()
        and messages
        and session_id
    )
    checks = {
        "process_exit_zero": returncode == 0 and not timed_out,
        "timeout_not_triggered": not timed_out,
        "session_id_found": bool(session_id),
        "session_receipt_loaded": session_receipt_loaded,
        "expected_text_found": case.expected_text in final_text,
        "visible_reasoning_clean": not _has_visible_reasoning_leak(final_text),
        "required_tools_called": all(tool in tool_names for tool in case.required_tools),
        "forbidden_tools_absent": not any(tool in tool_names for tool in case.forbidden_tools),
        "expected_tool_sequence": _matches_tool_sequence(
            tool_calls, case.expected_tool_sequence
        ),
        "forbidden_artifacts_absent": all(
            not Path(path).exists() for path in case.forbidden_artifacts
        ),
        "no_tools_called": not tool_calls if case.expect_no_tools else True,
    }
    failures = [name for name, passed in checks.items() if not passed]
    return CaseResult(
        case_id=case.case_id,
        ok=not failures,
        checks=checks,
        final_text=final_text,
        session_id=session_id,
        tool_calls=tool_calls,
        returncode=returncode,
        elapsed_s=elapsed_s,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        session_path=str(session_path) if session_path else None,
        failure_reasons=failures,
        timed_out=timed_out,
        session_error_path=str(session_error_path) if session_error_path else None,
    )


def build_chat_command(
    *,
    provider: str | None,
    model: str | None,
    toolsets: str,
    source: str,
    prompt: str,
    hermes_executable: str | None = None,
) -> list[str]:
    """Build the real Hermes chat subprocess command."""

    hermes_cmd = hermes_executable or shutil.which("hermes")
    if not hermes_cmd:
        raise ProviderValidationError("Could not find 'hermes' on PATH")

    cmd = [
        hermes_cmd,
        "chat",
        "-Q",
        "--source",
        source,
        "--toolsets",
        toolsets,
    ]
    if provider:
        cmd.extend(["--provider", provider])
    if model:
        cmd.extend(["--model", model])
    cmd.extend(["-q", prompt])
    return cmd


def _ensure_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _serialize_messages(path: Path, messages: list[dict[str, Any]]) -> None:
    _write_json(path, messages)


def run_validation(args: Any) -> int:
    if args.toolsets != "file":
        raise ProviderValidationError(
            "tier-0 validate supports only the file toolset"
        )
    out_dir = Path(args.out or tempfile.mkdtemp(prefix="hermes-provider-validation-"))
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    fixture_dir = out_dir / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)

    cases = get_suite_cases(args.suite, fixture_dir)
    results: list[CaseResult] = []

    for case in cases:
        source = f"provider-validation:{case.case_id}"
        cmd = build_chat_command(
            provider=args.provider,
            model=args.model,
            toolsets=args.toolsets,
            source=source,
            prompt=case.prompt,
            hermes_executable=getattr(args, "hermes_executable", None),
        )
        started = time.monotonic()
        timed_out = False
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(fixture_dir),
                text=True,
                capture_output=True,
                timeout=args.timeout,
                env=os.environ.copy(),
                check=False,
            )
            stdout = proc.stdout
            stderr = proc.stderr
            returncode = proc.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = _ensure_text(exc.stdout)
            stderr = _ensure_text(exc.stderr) + f"\nTimed out after {args.timeout} seconds.\n"
            returncode = 124
            timed_out = True
        elapsed = time.monotonic() - started

        stdout_path = raw_dir / f"{case.case_id}.stdout"
        stderr_path = raw_dir / f"{case.case_id}.stderr"
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")

        session_id = parse_session_id(stdout, stderr)
        messages: list[dict[str, Any]] = []
        session_path: Path | None = None
        session_error_path: Path | None = None
        if session_id:
            try:
                messages = load_session_messages(session_id)
                session_path = raw_dir / f"{case.case_id}.session.json"
                _serialize_messages(session_path, messages)
            except Exception as exc:  # pragma: no cover - defensive receipt path
                session_error_path = raw_dir / f"{case.case_id}.session-error.txt"
                session_error_path.write_text(
                    f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
                )
        else:
            session_error_path = raw_dir / f"{case.case_id}.session-error.txt"
            session_error_path.write_text(
                "session_id not found in captured stdout/stderr\n",
                encoding="utf-8",
            )

        result = score_case(
            case,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            session_id=session_id,
            messages=messages,
            elapsed_s=elapsed,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            session_path=session_path,
            session_error_path=session_error_path,
            timed_out=timed_out,
        )
        results.append(result)
        status = "PASS" if result.ok else "FAIL"
        print(f"{status} {case.case_id} ({elapsed:.1f}s)")
        if not result.ok:
            print(f"  failures: {', '.join(result.failure_reasons)}")

    result_dicts = [result.__dict__ for result in results]
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as fh:
        for payload in result_dicts:
            fh.write(json.dumps(payload, sort_keys=True) + "\n")

    passed = sum(1 for result in results if result.ok)
    receipt_gate_failure = any(
        not result.checks.get("session_receipt_loaded", False)
        or result.timed_out
        or not result.checks.get("session_id_found", False)
        for result in results
    )
    status = (
        "SCREEN-PASS"
        if passed == len(results)
        else "GATE-FAILED"
        if receipt_gate_failure
        else "REJECT"
    )
    summary = {
        "ok": passed == len(results),
        "status": status,
        "evaluation_policy": "cli-screening-v1",
        "tier": "tier-0",
        "qualification": False,
        "replacement_evidence": False,
        "suite": args.suite,
        "provider": args.provider,
        "model": args.model,
        "toolsets": args.toolsets,
        "passed": passed,
        "total": len(results),
        "out_dir": str(out_dir),
        "results": result_dicts,
    }
    _write_json(out_dir / "summary.json", summary)
    _write_summary_markdown(out_dir / "summary.md", summary)

    print(f"\nSummary: {passed}/{len(results)} passed")
    print(f"Receipts: {out_dir}")
    return 0 if summary["ok"] else 1


def _write_summary_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Hermes Tier-0 Compatibility Smoke",
        "",
        f"- Suite: `{summary['suite']}`",
        f"- Provider: `{summary.get('provider') or 'default'}`",
        f"- Model: `{summary.get('model') or 'default'}`",
        f"- Toolsets: `{summary['toolsets']}`",
        f"- Status: {summary['status']} ({summary['passed']}/{summary['total']})",
        "",
        "This is a tier-0 compatibility smoke. It is not Hermes qualification, replacement evidence, or a benchmark. A passing screen does not authorize routing or promotion.",
        "",
        "## Cases",
        "",
    ]
    for result in summary["results"]:
        lines.extend(
            [
                f"### {result['case_id']}: {'PASS' if result['ok'] else 'FAIL'}",
                "",
                f"- Session: `{result.get('session_id') or 'none'}`",
                f"- Tool receipts: {json.dumps(result.get('tool_calls') or [], sort_keys=True)}",
                f"- Elapsed: `{result['elapsed_s']:.2f}s`",
                f"- Failures: `{', '.join(result.get('failure_reasons') or []) or 'none'}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def cmd_providers(args: Any) -> None:
    if getattr(args, "providers_command", None) == "validate":
        try:
            raise SystemExit(run_validation(args))
        except ProviderValidationError as exc:
            print(f"providers validate: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc
    if getattr(args, "providers_command", None) in {"evaluate", "score", "suites"}:
        # Keep the authored validation module as the compatibility seam while
        # loading the larger orchestration module only for its explicit modes.
        from hermes_cli.provider_evaluation import cmd_evaluation

        return cmd_evaluation(args)
    print("Usage: hermes providers {validate,evaluate,score,suites}", file=sys.stderr)
    raise SystemExit(2)
