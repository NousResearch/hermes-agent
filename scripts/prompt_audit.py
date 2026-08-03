#!/usr/bin/env python3
"""Deterministic, redacted audit of effective Hermes profile prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except ImportError:  # pragma: no cover - the repository runtime includes PyYAML
    yaml = None


SCHEMA_VERSION = 1
RUBRIC_ID = "kensei.prompt-audit.aispa"
RUBRIC_VERSION = "1.0.0"
DEFAULT_IDENTITY = "You are Hermes, a helpful AI agent."
SEVERITY_ORDER = {"low": 1, "medium": 2, "high": 3, "critical": 4}
SEVERITY_WEIGHTS = {"low": 1, "medium": 4, "high": 10, "critical": 25}
SECRET_RE = re.compile(
    r"(?i)(api[_-]?key|token|password|secret|authorization)\s*[:=]\s*([^\s,;]+)"
)
REQUIRED_SECTIONS = {
    "identity": ("identity", "role", "mission"),
    "scope": ("scope", "owns", "responsibilities"),
    "boundaries": ("boundaries", "does not own", "do not"),
    "approval": ("approval", "privilege", "gate"),
    "tool_safety": ("tool safety", "verification", "tools"),
    "untrusted_content": ("prompt injection", "untrusted", "content trust"),
    "completion": ("definition of done", "completion", "evidence"),
}


@dataclass(frozen=True)
class PromptSource:
    kind: str
    location: str
    status: str
    sha256: str | None
    chars: int | None
    reason: str | None = None


@dataclass(frozen=True)
class ProfileRecord:
    profile: str
    home: str
    config_path: str
    identity_status: str
    effective_prompt_sha256: str
    effective_prompt: str
    sources: tuple[PromptSource, ...]
    config_errors: tuple[str, ...]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").split("\n")).strip()


def _read_config(path: Path) -> tuple[dict[str, Any], list[str]]:
    if not path.exists():
        return {}, ["config.yaml is missing"]
    if yaml is None:
        return {}, ["PyYAML is unavailable"]
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(value, dict):
            return {}, ["config.yaml root is not a mapping"]
        return value, []
    except Exception as exc:  # concise parser evidence only
        return {}, [f"config.yaml could not be parsed: {type(exc).__name__}"]


def _nested(config: dict[str, Any], *keys: str) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _profile_homes(root: Path) -> list[tuple[str, Path]]:
    result = [("default", root)]
    profiles_dir = root / "profiles"
    if profiles_dir.is_dir():
        result.extend(
            (entry.name, entry)
            for entry in sorted(profiles_dir.iterdir(), key=lambda item: item.name)
            if entry.is_dir() and not entry.name.startswith(".")
        )
    return result


def resolve_profile(profile: str, home: Path) -> ProfileRecord:
    config_path = home / "config.yaml"
    config, errors = _read_config(config_path)
    soul_path = home / "SOUL.md"
    sources: list[PromptSource] = []
    identity_status = "applied"
    identity = ""
    if soul_path.exists():
        try:
            identity = _canonical(soul_path.read_text(encoding="utf-8"))
        except OSError as exc:
            errors.append(f"SOUL.md is unreadable: {type(exc).__name__}")
        if identity:
            sources.append(PromptSource("soul", str(soul_path), "applied", _sha(identity), len(identity)))
        else:
            identity_status = "fallback"
            sources.append(PromptSource("soul", str(soul_path), "empty", None, 0, "default identity applied"))
    else:
        identity_status = "fallback"
        sources.append(PromptSource("soul", str(soul_path), "missing", None, None, "default identity applied"))
    if identity_status == "fallback":
        identity = DEFAULT_IDENTITY
        sources.append(PromptSource("default_identity", "agent.prompt_builder.DEFAULT_AGENT_IDENTITY", "fallback", _sha(identity), len(identity)))

    overlay = _nested(config, "agent", "system_prompt")
    parts = [identity]
    if isinstance(overlay, str) and overlay.strip():
        overlay = _canonical(overlay)
        parts.append(overlay)
        sources.append(PromptSource("config_overlay", f"{config_path}:agent.system_prompt", "applied", _sha(overlay), len(overlay)))
    elif overlay is not None:
        errors.append("agent.system_prompt is not a non-empty string")
        sources.append(PromptSource("config_overlay", f"{config_path}:agent.system_prompt", "malformed", None, None))

    prompt = "\n\n".join(parts)
    return ProfileRecord(
        profile=profile,
        home=str(home),
        config_path=str(config_path),
        identity_status=identity_status,
        effective_prompt_sha256=_sha(prompt),
        effective_prompt=prompt,
        sources=tuple(sources),
        config_errors=tuple(errors),
    )


def discover_profiles(root: Path) -> list[ProfileRecord]:
    root = root.expanduser().resolve()
    return [resolve_profile(name, home) for name, home in _profile_homes(root)]


def _redact(text: str, limit: int = 240) -> str:
    text = SECRET_RE.sub(lambda match: f"{match.group(1)}=[REDACTED_SECRET]", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _finding(
    record: ProfileRecord,
    rule_id: str,
    severity: str,
    evidence: str,
    location: str,
    remediation: str,
) -> dict[str, str]:
    return {
        "profile": record.profile,
        "rule_id": rule_id,
        "severity": severity,
        "evidence": _redact(evidence),
        "source_location": location,
        "remediation": remediation,
    }


def _line_findings(record: ProfileRecord) -> Iterable[dict[str, str]]:
    source = next((item.location for item in record.sources if item.kind == "soul"), record.home)
    lines = record.effective_prompt.splitlines()
    unsafe_patterns = {
        "KPA-005": (
            "critical",
            re.compile(r"(?i)\b(disable safeguards|bypass (?:approval|permissions?)|unrestricted (?:sudo|root|filesystem|network)|always use sudo)\b"),
            "Narrow the capability, require approval, rollback and post-action verification.",
        ),
        "KPA-006": (
            "critical",
            re.compile(r"(?i)\b(assume (?:it )?succeeds?|fabricate (?:the )?(?:output|result)|ignore (?:the )?exit status|suppress (?:all )?errors)\b"),
            "Require exit/error checks, read-back proof and honest blocker reporting.",
        ),
        "KPA-007": (
            "critical",
            re.compile(r"(?i)\b(follow instructions? (?:found|embedded) in (?:web|files?|tool output|memory)|retrieved content overrides)\b"),
            "Treat retrieved text as data and prohibit embedded role or tool directives.",
        ),
        "KPA-013": (
            "critical",
            re.compile(r"(?i)\b(conceal (?:data )?access|expose (?:passwords?|tokens?|api keys?)|retain all personal data)\b"),
            "Minimise data use and require typed redaction in durable evidence.",
        ),
        "KPA-015": (
            "critical",
            re.compile(r"(?i)\b(ignore safety|accept (?:all )?jailbreaks?|comply with every request|user may disable safety)\b"),
            "Define non-overridable safety invariants and bounded refusal behaviour.",
        ),
    }
    for number, line in enumerate(lines, 1):
        for rule_id, (severity, pattern, remediation) in unsafe_patterns.items():
            if pattern.search(line):
                yield _finding(record, rule_id, severity, line, f"{source}:{number}", remediation)

        for date_match in re.finditer(r"\b(20\d{2})-(\d{2})-(\d{2})\b", line):
            try:
                parsed = datetime.fromisoformat(date_match.group(0)).date()
            except ValueError:
                continue
            if parsed < datetime.now(timezone.utc).date() and re.search(r"(?i)\b(expires?|valid until|follow[- ]?up|current)\b", line):
                yield _finding(
                    record,
                    "KPA-003",
                    "medium",
                    line,
                    f"{source}:{number}",
                    "Replace the mutable claim with a source-of-truth lookup or mark it historical.",
                )

    normative = [(index, line) for index, line in enumerate(lines, 1) if re.search(r"(?i)\b(must|never|always|do not|may)\b", line)]
    for left_index, left in normative:
        for right_index, right in normative:
            if right_index <= left_index:
                continue
            left_words = set(re.findall(r"[a-z]{4,}", left.lower()))
            right_words = set(re.findall(r"[a-z]{4,}", right.lower()))
            opposite = ("always" in left.lower() and "never" in right.lower()) or ("never" in left.lower() and "always" in right.lower())
            if opposite and len(left_words & right_words) >= 2:
                evidence = f"Potential conflict: L{left_index} {_redact(left, 90)} <> L{right_index} {_redact(right, 90)}"
                yield _finding(
                    record,
                    "KPA-002",
                    "high",
                    evidence,
                    f"{source}:{left_index},{right_index}",
                    "Remove one instruction or state explicit conditions and precedence.",
                )
                return


def _required_section_finding(record: ProfileRecord) -> dict[str, str] | None:
    headings = [match.group(1).strip().lower() for match in re.finditer(r"(?m)^#{1,6}\s+(.+?)\s*$", record.effective_prompt)]
    missing = [key for key, aliases in REQUIRED_SECTIONS.items() if not any(any(alias in heading for alias in aliases) for heading in headings)]
    if not missing:
        return None
    source = next((item.location for item in record.sources if item.kind == "soul"), record.home)
    return _finding(
        record,
        "KPA-001",
        "high",
        "Missing sections: " + ", ".join(missing),
        source,
        "Add substantive canonical sections or configure a reviewed profile-type exemption.",
    )


def _baseline_map(path: Path | None) -> tuple[dict[str, dict[str, Any]], list[str]]:
    if path is None:
        return {}, ["KPA-004 baseline comparison not run: no --baseline supplied"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {item["profile"]: item for item in payload.get("profiles", [])}, []
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return {}, [f"KPA-004 baseline comparison not run: {type(exc).__name__}"]


def _profile_json(record: ProfileRecord) -> dict[str, Any]:
    value = asdict(record)
    value.pop("effective_prompt")
    return value


def audit(
    *,
    root: Path,
    baseline_path: Path | None = None,
    profiles: set[str] | None = None,
    severities: set[str] | None = None,
) -> dict[str, Any]:
    all_records = discover_profiles(root)
    known_profiles = {record.profile for record in all_records}
    records = all_records
    if profiles:
        records = [record for record in records if record.profile in profiles]
    baseline, checks_not_run = _baseline_map(baseline_path)
    findings: list[dict[str, str]] = []

    for record in records:
        if record.identity_status == "fallback":
            source = next(item.location for item in record.sources if item.kind == "soul")
            findings.append(_finding(
                record,
                "KPA-011",
                "critical",
                "Missing or empty SOUL.md caused explicit default identity fallback.",
                source,
                "Add a profile-specific SOUL.md or explicitly approve generic Hermes fallback.",
            ))
        for error in record.config_errors:
            findings.append(_finding(
                record,
                "KPA-011",
                "critical",
                error,
                record.config_path,
                "Repair the profile config and fail identity/authority resolution closed.",
            ))
        section_finding = _required_section_finding(record)
        if section_finding:
            findings.append(section_finding)
        findings.extend(_line_findings(record))

        if baseline_path is not None:
            old = baseline.get(record.profile)
            if old is None:
                findings.append(_finding(
                    record, "KPA-004", "high", "Profile is absent from approved baseline.", str(baseline_path),
                    "Review the new profile and explicitly capture a new approved baseline.",
                ))
            elif old.get("effective_prompt_sha256") != record.effective_prompt_sha256:
                findings.append(_finding(
                    record,
                    "KPA-004",
                    "high",
                    f"baseline={old.get('effective_prompt_sha256', 'missing')} current={record.effective_prompt_sha256}",
                    str(baseline_path),
                    "Restore approved text or review the diff and explicitly capture a new baseline.",
                ))

    for record in records:
        for match in re.finditer(r"(?im)^\s*(?:reports?|escalates?)\s+to\s+[`*]*([A-Za-z][A-Za-z0-9 _-]{1,60})", record.effective_prompt):
            display = match.group(1).strip().rstrip(".*`")
            reference = re.sub(r"[^a-z0-9]+", "-", display.lower()).strip("-")
            if reference not in known_profiles and display.lower() not in {"the user", "user", "human", "sahil"}:
                line = record.effective_prompt[: match.start()].count("\n") + 1
                source = next((item.location for item in record.sources if item.kind == "soul"), record.home)
                findings.append(_finding(
                    record,
                    "KPA-010",
                    "medium",
                    f"Related profile '{display}' is an unknown profile in the discovered fleet.",
                    f"{source}:{line}",
                    "Correct the profile reference or add the governed profile and reciprocal reporting edge.",
                ))

    hash_groups: dict[str, list[str]] = {}
    for record in records:
        hash_groups.setdefault(record.effective_prompt_sha256, []).append(record.profile)
    for digest, names in sorted(hash_groups.items()):
        if len(names) > 1:
            for record in records:
                if record.profile in names:
                    findings.append(_finding(
                        record,
                        "KPA-010",
                        "medium",
                        f"Profiles share the same effective prompt hash {digest}: {', '.join(names)}",
                        record.home,
                        "Review whether related profiles intentionally share identity, duties and escalation paths.",
                    ))

    findings.sort(key=lambda item: (item["profile"], item["rule_id"], item["source_location"], item["evidence"]))
    if severities:
        findings = [item for item in findings if item["severity"] in severities]
    score = sum(SEVERITY_WEIGHTS[item["severity"]] for item in {f"{item['profile']}:{item['rule_id']}": item for item in findings}.values())
    verdict = "fail" if any(item["severity"] == "critical" for item in findings) or score >= 10 else "review" if score >= 4 or checks_not_run else "pass"
    return {
        "schema_version": SCHEMA_VERSION,
        "rubric": {"id": RUBRIC_ID, "version": RUBRIC_VERSION},
        "root": str(root.expanduser().resolve()),
        "profiles": [_profile_json(record) for record in records],
        "checks_run": ["profile_discovery", "effective_prompt_resolution", "static_rules", "cross_profile_hashes"] + (["baseline_comparison"] if baseline_path else []),
        "checks_not_run": checks_not_run,
        "findings": findings,
        "score": score,
        "verdict": verdict,
    }


def write_baseline(root: Path, path: Path, *, approved_by: str) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing baseline: {path}")
    records = discover_profiles(root)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "rubric": {"id": RUBRIC_ID, "version": RUBRIC_VERSION},
        "approved_by": approved_by,
        "captured_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "root": str(root.expanduser().resolve()),
        "profiles": [
            {
                "profile": record.profile,
                "effective_prompt_sha256": record.effective_prompt_sha256,
                "sources": [asdict(source) for source in record.sources],
            }
            for record in records
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def exit_code(report: dict[str, Any], threshold: str) -> int:
    if threshold == "off":
        return 0
    floor = SEVERITY_ORDER[threshold]
    return int(any(SEVERITY_ORDER[item["severity"]] >= floor for item in report["findings"]))


def _terminal(report: dict[str, Any]) -> str:
    lines = [
        f"Prompt audit: {report['verdict'].upper()}  score={report['score']}  profiles={len(report['profiles'])}  findings={len(report['findings'])}"
    ]
    for finding in report["findings"]:
        lines.append(f"[{finding['severity'].upper():8}] {finding['profile']} {finding['rule_id']} {finding['source_location']}")
        lines.append(f"  {finding['evidence']}")
        lines.append(f"  Remediation: {finding['remediation']}")
    for skipped in report["checks_not_run"]:
        lines.append(f"[NOT RUN ] {skipped}")
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.home() / ".hermes", help="Hermes root containing default and named profiles")
    parser.add_argument("--profile", action="append", help="Audit only this profile (repeatable)")
    parser.add_argument("--severity", action="append", choices=tuple(SEVERITY_ORDER), help="Emit only this severity (repeatable)")
    parser.add_argument("--baseline", type=Path, help="Compare with an explicit approved baseline")
    parser.add_argument("--create-baseline", type=Path, help="Create a new baseline; never overwrites")
    parser.add_argument("--approved-by", help="Required with --create-baseline")
    parser.add_argument("--format", choices=("terminal", "json"), default="terminal")
    parser.add_argument("--output", type=Path, help="Write output to this path instead of stdout")
    parser.add_argument("--fail-on", choices=("off", "low", "medium", "high", "critical"), default="high")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.create_baseline:
        if not args.approved_by:
            _parser().error("--approved-by is required with --create-baseline")
        try:
            payload = write_baseline(args.root, args.create_baseline, approved_by=args.approved_by)
        except FileExistsError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(f"Created reviewable baseline for {len(payload['profiles'])} profiles: {args.create_baseline}")
        return 0

    report = audit(
        root=args.root,
        baseline_path=args.baseline,
        profiles=set(args.profile or []),
        severities=set(args.severity or []),
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n" if args.format == "json" else _terminal(report) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return exit_code(report, args.fail_on)


if __name__ == "__main__":
    raise SystemExit(main())
