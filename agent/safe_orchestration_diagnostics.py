"""Pure diagnostics helpers for safe-orchestration log summaries.

The helpers in this module parse already-collected log text and produce a compact,
redacted summary. They do not read files, inspect the environment, mutate state,
or talk to the gateway.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re


_PREFLIGHT_RE = re.compile(r"preflight risk summary: (?P<body>.*)")
_VERIFIER_SUMMARY_RE = re.compile(r"safe orchestration verifier summary: (?P<body>.*)")
_VERIFIER_FINDING_RE = re.compile(r"safe orchestration verifier finding: (?P<body>.*)")
_FIELD_RE_TEMPLATE = r"(?:^|\s){name}=([^\s]+)"


@dataclass(frozen=True)
class SafeOrchestrationLogSummary:
    """Redacted aggregate counts from safe-orchestration log lines."""

    total_lines: int
    preflight_levels: dict[str, int]
    preflight_signals: dict[str, int]
    verifier_codes: dict[str, int]
    verifier_tools: dict[str, int]

    def render(self) -> str:
        """Return a deterministic, argument-free diagnostic summary."""

        return (
            "safe orchestration diagnostics: "
            f"total_lines={self.total_lines} "
            f"preflight_levels={_format_counts(self.preflight_levels)} "
            f"preflight_signals={_format_counts(self.preflight_signals)} "
            f"verifier_codes={_format_counts(self.verifier_codes)} "
            f"verifier_tools={_format_counts(self.verifier_tools)}"
        )


def summarize_safe_orchestration_log(log_text: str) -> SafeOrchestrationLogSummary:
    """Summarize safe-orchestration log text without echoing raw log contents."""

    preflight_levels: Counter[str] = Counter()
    preflight_signals: Counter[str] = Counter()
    verifier_codes: Counter[str] = Counter()
    verifier_tools: Counter[str] = Counter()
    total = 0

    for raw_line in str(log_text).splitlines():
        if match := _PREFLIGHT_RE.search(raw_line):
            total += 1
            body = match.group("body")
            _increment_csv_field(preflight_levels, _field(body, "level"))
            _increment_csv_field(preflight_signals, _field(body, "signals"))
            continue

        if match := _VERIFIER_SUMMARY_RE.search(raw_line):
            total += 1
            body = match.group("body")
            _increment_count_field(verifier_codes, _field(body, "codes"))
            _increment_csv_field(verifier_tools, _field(body, "tools_seen"))
            continue

        if match := _VERIFIER_FINDING_RE.search(raw_line):
            total += 1
            body = match.group("body")
            _increment_csv_field(verifier_codes, _field(body, "code"))
            _increment_csv_field(verifier_tools, _field(body, "tool"))

    return SafeOrchestrationLogSummary(
        total_lines=total,
        preflight_levels=dict(sorted(preflight_levels.items())),
        preflight_signals=dict(sorted(preflight_signals.items())),
        verifier_codes=dict(sorted(verifier_codes.items())),
        verifier_tools=dict(sorted(verifier_tools.items())),
    )


def _field(text: str, name: str) -> str | None:
    match = re.search(_FIELD_RE_TEMPLATE.format(name=re.escape(name)), text)
    return match.group(1) if match else None


def _increment_csv_field(counter: Counter[str], value: str | None) -> None:
    if not value or value == "none":
        if value == "none":
            counter["none"] += 1
        return
    for item in value.split(","):
        key = item.strip()
        if key:
            counter[key] += 1


def _increment_count_field(counter: Counter[str], value: str | None) -> None:
    if not value or value == "none":
        return
    for item in value.split(","):
        key, sep, count_text = item.partition(":")
        key = key.strip()
        if not key:
            continue
        if sep:
            try:
                counter[key] += int(count_text)
            except ValueError:
                counter[key] += 1
        else:
            counter[key] += 1


def _format_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return ",".join(f"{key}:{counts[key]}" for key in sorted(counts))
