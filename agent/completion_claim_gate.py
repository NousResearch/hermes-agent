"""Framework-level final-response gate for completion claims.

When enabled, Hermes can withhold a Done/completed claim unless the final
response explicitly references a verification report within configured roots
that contains a passing marker. The feature is off by default and only
activates when configured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import re

_EXPLICIT_DONE_PATTERNS = [
    r"^done(?:[.!,:-]|\s*$)",
    r"^completed(?:[.!,:-]|\s*$)",
    r"^complete(?:[.!,:-]|\s*$)",
    r"^finished(?:[.!,:-]|\s*$)",
    r"^implemented(?:[.!,:-]|\s*$)",
    r"^shipped(?:[.!,:-]|\s*$)",
    r"^i(?:'ve| have)?\s+(?:completed|finished|implemented|shipped)\b",
    r"^(?:the\s+)?(?:task|work|job|request|fix|implementation)\s+(?:is|was|has\s+been)?\s*(?:done|complete|completed|finished)\b",
    r"^已完成",
    r"^已经完成",
    r"^任务完成",
    r"^交付完成",
    r"^修复完成",
    r"^已修复",
    r"^已交付",
    r"^搞定(?:了)?",
]

_NEGATING_PATTERNS = [
    r"\bnot\s+(?:done|complete|completed|finished)\b",
    r"\b(?:isn't|is not|wasn't|was not)\s+(?:done|complete|completed|finished)\b",
    r"\b(?:pending|blocked|in progress|wip)\b",
    r"\b(?:done|complete|completed|finished)\s+(?:once|when|after|if|until)\b",
    r"未完成",
    r"尚未完成",
    r"还没完成",
    r"进行中",
    r"待验证",
    r"待处理",
]

_WITHHELD_PATTERNS = [
    r"completion claim withheld",
    r"pending verification",
    r"not verified",
    r"unverified",
    r"未验证",
    r"无法验证",
    r"阻塞",
]

_DEFAULT_ALLOWED_ROOTS = ["."]
_DEFAULT_REPORT_PATH_REGEX = (
    r"(?P<path>(?:~|/|\.)?[^\s`'\"]*(?:report|verification)[^\s`'\"]*\.(?:md|txt|json))"
)
_DEFAULT_PASS_REGEX = (
    r"(?:^\s*(?:Gate|Status|Result)\s*:\s*PASS(?:ED)?\s*$|\"(?:gate|status|result)\"\s*:\s*\"PASS(?:ED)?\")"
)


@dataclass
class CompletionClaimGateResult:
    """Result of evaluating/applying the final-response completion gate."""

    response: str
    changed: bool
    status: str
    reason: str = ""
    report_paths: list[str] = field(default_factory=list)


def _as_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}
    return default


def _normalize_allowed_roots(value: object) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return list(_DEFAULT_ALLOWED_ROOTS)
    normalized = [str(item) for item in value if str(item).strip()]
    return normalized or list(_DEFAULT_ALLOWED_ROOTS)


def config_from_mapping(mapping: dict | None) -> dict:
    """Normalize ``agent.completion_claim_gate`` config from config.yaml."""

    agent_cfg = mapping.get("agent", {}) if isinstance(mapping, dict) else {}
    if not isinstance(agent_cfg, dict):
        agent_cfg = {}
    section = agent_cfg.get("completion_claim_gate", {})
    if not isinstance(section, dict):
        section = {}

    report_path_regex = section.get("report_path_regex") or _DEFAULT_REPORT_PATH_REGEX
    pass_regex = section.get("pass_regex") or _DEFAULT_PASS_REGEX
    remediation_command = section.get("remediation_command") or ""

    return {
        "enabled": _as_bool(section.get("enabled"), False),
        "require_report_for_done": _as_bool(section.get("require_report_for_done"), True),
        "allowed_roots": _normalize_allowed_roots(section.get("allowed_roots", _DEFAULT_ALLOWED_ROOTS)),
        "report_path_regex": str(report_path_regex),
        "pass_regex": str(pass_regex),
        "remediation_command": str(remediation_command),
    }


def _candidate_segments(response: str) -> list[str]:
    visible = (response or "").strip()
    if not visible:
        return []
    window = visible[:400]
    segments = re.split(r"(?<=[.!?。！？])\s+|\n+", window)
    cleaned = [segment.strip() for segment in segments if segment.strip()]
    return cleaned[:6]


def _contains_done_claim(response: str) -> bool:
    visible = (response or "").strip()
    if not visible:
        return False
    lowered = visible.lower()
    if any(re.search(pattern, lowered, re.IGNORECASE) for pattern in _WITHHELD_PATTERNS):
        return False

    for segment in _candidate_segments(visible):
        lowered_segment = segment.lower()
        if any(re.search(pattern, lowered_segment, re.IGNORECASE) for pattern in _NEGATING_PATTERNS):
            continue
        if any(re.search(pattern, lowered_segment, re.IGNORECASE) for pattern in _EXPLICIT_DONE_PATTERNS):
            return True
    return False


def _resolve_allowed_roots(allowed_roots: list[str]) -> list[Path]:
    resolved: list[Path] = []
    for raw in allowed_roots or _DEFAULT_ALLOWED_ROOTS:
        try:
            resolved.append(Path(os.path.expanduser(raw)).resolve())
        except Exception:
            continue
    return resolved or [Path.cwd().resolve()]


def _path_is_within_roots(path: Path, allowed_roots: list[Path]) -> bool:
    for root in allowed_roots:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _extract_report_paths(response: str, report_path_regex: str, allowed_roots: list[str]) -> list[Path]:
    pattern = re.compile(report_path_regex)
    roots = _resolve_allowed_roots(allowed_roots)
    paths: list[Path] = []
    seen: set[str] = set()
    for match in pattern.finditer(response or ""):
        raw_path = str(match.group("path") or "").strip()
        if not raw_path:
            continue
        candidates: list[Path] = []
        try:
            expanded = Path(os.path.expanduser(raw_path))
        except Exception:
            continue
        if expanded.is_absolute():
            candidates.append(expanded.resolve())
        else:
            for root in roots:
                candidates.append((root / expanded).resolve())
        for path in candidates:
            if not _path_is_within_roots(path, roots):
                continue
            key = str(path)
            if key not in seen:
                paths.append(path)
                seen.add(key)
    return paths


def _report_is_passing(path: Path, pass_regex: str) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return bool(re.search(pass_regex, text, flags=re.IGNORECASE | re.MULTILINE))


def _downgrade_response(original: str, reason: str, remediation_command: str) -> str:
    lines = [
        "## Completion claim withheld pending verification",
        "",
        "Hermes detected a Done/completed claim, but could not verify a referenced report with a passing marker.",
        "",
        f"Reason: {reason}",
        "",
    ]
    if remediation_command:
        lines.extend(
            [
                "Configured remediation command:",
                f"```bash\n{remediation_command}\n```",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "To preserve a completion claim, reference a verification report file in the final reply and include a passing status marker such as `Status: PASS` or `\"status\": \"PASS\"`.",
                "",
            ]
        )
    lines.extend(
        [
            "The original completion claim was withheld from the user-facing response.",
        ]
    )
    return "\n".join(lines)


def evaluate_completion_claim_gate(
    response: str,
    *,
    enabled: bool,
    require_report_for_done: bool = True,
    allowed_roots: list[str] | None = None,
    report_path_regex: str = _DEFAULT_REPORT_PATH_REGEX,
    pass_regex: str = _DEFAULT_PASS_REGEX,
    remediation_command: str = "",
) -> CompletionClaimGateResult:
    """Evaluate whether a final response needs completion-claim downgrading."""

    original = response or ""
    if not enabled:
        return CompletionClaimGateResult(original, False, "disabled")
    if not require_report_for_done:
        return CompletionClaimGateResult(original, False, "not_required")
    if not _contains_done_claim(original):
        return CompletionClaimGateResult(original, False, "not_a_done_claim")

    try:
        re.compile(report_path_regex)
        re.compile(pass_regex)
    except re.error as exc:
        downgraded = _downgrade_response(
            original,
            f"completion-claim gate configuration is invalid: {exc}",
            remediation_command,
        )
        return CompletionClaimGateResult(
            downgraded,
            True,
            "completion_withheld",
            reason=f"invalid_gate_config: {exc}",
        )

    report_paths = _extract_report_paths(
        original,
        report_path_regex,
        _normalize_allowed_roots(allowed_roots or _DEFAULT_ALLOWED_ROOTS),
    )
    report_strings = [str(path) for path in report_paths]
    if report_paths and any(_report_is_passing(path, pass_regex) for path in report_paths):
        return CompletionClaimGateResult(
            original,
            False,
            "completion_verified",
            report_paths=report_strings,
        )

    reason = "no passing verification report was found within the allowed roots"
    if report_paths:
        reason = "referenced verification report did not contain a passing marker"

    downgraded = _downgrade_response(original, reason, remediation_command)
    return CompletionClaimGateResult(
        downgraded,
        True,
        "completion_withheld",
        reason=reason,
        report_paths=report_strings,
    )


def apply_completion_claim_gate(
    response: str,
    *,
    enabled: bool,
    require_report_for_done: bool = True,
    allowed_roots: list[str] | None = None,
    report_path_regex: str = _DEFAULT_REPORT_PATH_REGEX,
    pass_regex: str = _DEFAULT_PASS_REGEX,
    remediation_command: str = "",
) -> CompletionClaimGateResult:
    """Apply the gate and return the possibly modified final response."""

    return evaluate_completion_claim_gate(
        response,
        enabled=enabled,
        require_report_for_done=require_report_for_done,
        allowed_roots=allowed_roots,
        report_path_regex=report_path_regex,
        pass_regex=pass_regex,
        remediation_command=remediation_command,
    )
