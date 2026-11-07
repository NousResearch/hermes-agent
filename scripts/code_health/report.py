"""Allow comments and the agent-facing report."""

from __future__ import annotations

import re
from collections import Counter

from scripts.code_health.config import ALLOW_SYNTAX, METRIC_FIX, RULES_BY_ID, TARGETS
from scripts.code_health.model import FileMeasure, Finding

_ALLOW = re.compile(r"health:\s*allow\s+([A-Za-z0-9_,\- ]+?)\s*(?:--|—|:)\s*(\S.*)$")
_ALLOW_BARE = re.compile(r"health:\s*allow\b")


def _allow_lines(finding: Finding, fm: FileMeasure) -> list[int]:
    """Where an allow for this finding may sit: its own line, or a comment-only line right
    above it. A trailing allow on the previous statement waives only that statement."""
    if finding.rule == "FILE_LINES":
        return list(range(1, 6))
    above = finding.line - 1
    return [finding.line, above] if not fm.code_line(above) else [finding.line]


def apply_allows(findings: list[Finding], head: dict[str, FileMeasure]) -> None:
    """Mark findings whose line carries ``# health: allow RULE -- why`` (in a real comment) as allowed.

    A bare allow without a reason does not count; the finding stays and says why.
    """
    for finding in findings:
        fm = head.get(finding.path)
        if fm is None:
            continue
        for line_no in _allow_lines(finding, fm):
            text = fm.comments.get(line_no, "")
            match = _ALLOW.search(text)
            # `allow BLE001 S110` and `allow BLE001, S110` both name two rules (_ALLOW takes both).
            if match and finding.rule in re.split(r"[,\s]+", match.group(1).strip()):
                finding.allowed_reason = match.group(2).strip()
                break
            if _ALLOW_BARE.search(text) and finding.rule in text:
                finding.detail += f"  [allow comment ignored: it needs a reason, {ALLOW_SYNTAX}]"


def _fix_text(rule_id: str) -> str:
    if rule_id in METRIC_FIX:
        return METRIC_FIX[rule_id]
    return RULES_BY_ID[rule_id].fix


def format_findings(findings: list[Finding]) -> str:
    out: list[str] = []
    for f in sorted(findings, key=lambda f: (f.path, f.line, f.rule)):
        if f.allowed_reason is not None:
            continue
        tag = "" if f.blocking else " (advisory)"
        out.append(f"{f.path}:{f.line}  {f.rule}{tag}  {f.scope}")
        out.append(f"    {f.detail}")
        out.append(f"    fix: {_fix_text(f.rule)}")
    allowed = [f for f in findings if f.allowed_reason is not None]
    if allowed:
        out.append("")
        out.append("Allowed by comment (reviewers: check the reasons):")
        for f in allowed:
            out.append(f"  {f.path}:{f.line}  {f.rule}  {f.scope}  -- {f.allowed_reason}")
    return "\n".join(out)


def verdict(findings: list[Finding]) -> tuple[int, int]:
    blocking = sum(1 for f in findings if f.blocking and f.allowed_reason is None)
    advisory = sum(1 for f in findings if not f.blocking and f.allowed_reason is None)
    return blocking, advisory


def summarize_tree(measures: dict[str, FileMeasure]) -> str:
    """Burn-down view of a whole tree: units over target and hits per rule."""
    over: Counter[str] = Counter()
    worst: dict[str, tuple[int, str]] = {}
    hits: Counter[str] = Counter()
    for path, fm in measures.items():
        if fm.metrics.get("FILE_LINES", 0) > TARGETS["FILE_LINES"]:
            over["FILE_LINES"] += 1
            _track(worst, "FILE_LINES", fm.metrics["FILE_LINES"], path)
        for unit in fm.units.values():
            for metric, value in unit.metrics.items():
                if value > TARGETS[metric]:
                    over[metric] += 1
                    _track(worst, metric, value, f"{path}::{unit.qualname}")
        for hit, count in fm.hits.items():
            hits[hit.rule] += count
    lines = ["Over target (units that are baselined at their current value):"]
    for metric, target in TARGETS.items():
        value, where = worst.get(metric, (0, "-"))
        lines.append(f"  {metric:<11} target {target:<5} over: {over[metric]:<5} worst: {value} {where}")
    lines.append("Pattern hits (existing ones are baselined; new ones fail):")
    for rule_id, count in sorted(hits.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {rule_id:<9} {count:>6}  {RULES_BY_ID[rule_id].title}")
    return "\n".join(lines)


def _track(worst: dict[str, tuple[int, str]], metric: str, value: int, where: str) -> None:
    if value > worst.get(metric, (0, ""))[0]:
        worst[metric] = (value, where)
