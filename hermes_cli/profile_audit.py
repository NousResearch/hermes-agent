"""Read-only effectiveness and policy audit for Hermes profiles."""

from __future__ import annotations

import json
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import yaml

_COMPLETED_OUTCOMES = frozenset({"completed", "success"})
_ACTIVE_OUTCOMES = frozenset({None, ""})

_SOUL_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("role", ("you are", "# identity", "# role", "mission")),
    ("responsibilities", ("responsibil", "your job", "mission", "duties")),
    ("process", ("process", "workflow", "procedure", "how you work", "approach", "operating rules")),
    ("quality_standards", ("quality", "correctness", "security", "maintainab", "test", "verif")),
    ("output_or_dod", ("definition of done", "done means", "output", "deliverable", "report", "complete")),
    ("block_or_escalation", ("block", "escalat", "genuine gate", "missing access", "ambigu")),
    ("preflight", ("preflight", "prerequisite", "before starting", "before you start", "inspect the", "read first")),
)
_EXTERNAL_DEPENDENCIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Claude subscription wrapper", ("claude-sub", "claude code through", "claude subscription")),
    ("external browser automation", ("browser-use cloud", "browserbase", "stagehand cloud")),
)


def _issue(rule: str, message: str, severity: str = "warning") -> dict[str, str]:
    return {"rule": rule, "severity": severity, "message": message}


def lint_soul(text: str) -> list[dict[str, str]]:
    """Return missing effectiveness-policy requirements for one SOUL.md."""
    lowered = text.casefold()
    issues = [
        _issue(f"soul.{rule}", f"SOUL.md does not document {rule.replace('_', ' ')}")
        for rule, terms in _SOUL_RULES
        if not any(term in lowered for term in terms)
    ]
    if not ("kanban_complete" in lowered and "kanban_block" in lowered):
        issues.append(
            _issue(
                "soul.kanban_handoff",
                "SOUL.md must require Kanban workers to call kanban_complete or kanban_block",
            )
        )
    for name, indicators in _EXTERNAL_DEPENDENCIES:
        if any(indicator in lowered for indicator in indicators):
            has_preflight = any(term in lowered for term in ("preflight", "prerequisite", "check access", "verify access", "command -v"))
            has_fallback = any(term in lowered for term in ("fallback", "fall back", "if unavailable", "if it fails", "block"))
            if not (has_preflight and has_fallback):
                issues.append(
                    _issue(
                        "soul.external_dependency",
                        f"{name} is referenced without both a preflight and fallback/block policy",
                    )
                )
    return issues


def _read_mapping(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.is_file():
        return {}, None
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        return {}, f"{type(exc).__name__}: could not parse {path.name}"
    if not isinstance(value, dict):
        return {}, f"{path.name} must contain a mapping"
    return value, None


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            decoded = None
        value = decoded if isinstance(decoded, list) else [value]
    if not isinstance(value, list):
        return []
    return sorted({str(item) for item in value if str(item).strip()})


def _safe_config_inventory(config: dict[str, Any]) -> dict[str, Any]:
    model_cfg = config.get("model") if isinstance(config.get("model"), dict) else {}
    platforms = config.get("platform_toolsets")
    platform_cli = platforms.get("cli") if isinstance(platforms, dict) else None
    toolsets = _string_list(platform_cli if isinstance(platform_cli, list) else config.get("toolsets"))
    return {
        "model": str(model_cfg.get("default") or "") or None,
        "provider": str(model_cfg.get("provider") or "") or None,
        "toolsets": toolsets,
    }


def _policy_issues(inventory: dict[str, Any], policy: dict[str, Any]) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    expected_model = policy.get("model")
    if expected_model and inventory["model"] != str(expected_model):
        issues.append(_issue("config.model", f"model does not match policy (expected {expected_model!s})"))
    expected_provider = policy.get("provider")
    if expected_provider and inventory["provider"] != str(expected_provider):
        issues.append(_issue("config.provider", f"provider does not match policy (expected {expected_provider!s})"))
    actual = set(inventory["toolsets"])
    for required in _string_list(policy.get("required_toolsets")):
        if required not in actual:
            issues.append(_issue("config.required_toolset", f"required toolset is missing: {required}"))
    for forbidden in _string_list(policy.get("forbidden_toolsets")):
        if forbidden in actual:
            issues.append(_issue("config.forbidden_toolset", f"forbidden toolset is enabled: {forbidden}"))
    return issues


def _profile_directories(root: Path) -> Iterable[Path]:
    try:
        entries = sorted(root.iterdir(), key=lambda path: path.name)
    except OSError:
        return []
    return [
        path for path in entries
        if path.is_dir() and not path.name.startswith((".", "_"))
        and ((path / "config.yaml").exists() or (path / "SOUL.md").exists())
    ]


def _run_stats(db_path: Path, since_epoch: int) -> tuple[dict[str, dict[str, Any]], str | None]:
    if not db_path.is_file():
        return {}, "kanban database not found"
    try:
        uri = f"{db_path.resolve().as_uri()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as conn:
            rows = conn.execute(
                "SELECT profile, outcome FROM task_runs WHERE started_at >= ?",
                (since_epoch,),
            ).fetchall()
    except (OSError, sqlite3.Error) as exc:
        return {}, f"{type(exc).__name__}: could not read task_runs"

    grouped: dict[str, Counter[str]] = {}
    active: Counter[str] = Counter()
    for profile, outcome in rows:
        name = str(profile or "unassigned")
        if outcome in _ACTIVE_OUTCOMES:
            active[name] += 1
            continue
        grouped.setdefault(name, Counter())[str(outcome)] += 1

    result: dict[str, dict[str, Any]] = {}
    for name in sorted(set(grouped) | set(active)):
        outcomes = grouped.get(name, Counter())
        finished = sum(outcomes.values())
        completed = sum(outcomes[value] for value in _COMPLETED_OUTCOMES)
        result[name] = {
            "finished": finished,
            "active": active[name],
            "completed": completed,
            "completion_rate": round(completed / finished, 4) if finished else None,
            "outcomes": dict(sorted(outcomes.items())),
        }
    return result, None


def audit_profiles(
    profiles_root: Path,
    kanban_db: Path,
    *,
    policy: dict[str, Any] | None = None,
    since_days: int = 30,
    now: int | None = None,
) -> dict[str, Any]:
    """Build a secret-safe report without writing profiles or the board."""
    now = int(time.time()) if now is None else now
    since_epoch = 0 if since_days == 0 else now - (since_days * 86_400)
    profile_policy = (policy or {}).get("profiles", {})
    if not isinstance(profile_policy, dict):
        profile_policy = {}

    profiles: list[dict[str, Any]] = []
    for profile_dir in _profile_directories(profiles_root):
        issues: list[dict[str, str]] = []
        config, config_error = _read_mapping(profile_dir / "config.yaml")
        if config_error:
            issues.append(_issue("config.parse", config_error, "error"))
        inventory = _safe_config_inventory(config)
        soul_path = profile_dir / "SOUL.md"
        if not soul_path.is_file():
            issues.append(_issue("soul.missing", "SOUL.md is missing", "error"))
        else:
            try:
                issues.extend(lint_soul(soul_path.read_text(encoding="utf-8")))
            except OSError:
                issues.append(_issue("soul.read", "SOUL.md could not be read", "error"))
        one_policy = profile_policy.get(profile_dir.name, {})
        if isinstance(one_policy, dict):
            issues.extend(_policy_issues(inventory, one_policy))
        profiles.append({"name": profile_dir.name, "config": inventory, "issues": issues})

    stats, stats_error = _run_stats(kanban_db, since_epoch)
    total_issues = sum(len(profile["issues"]) for profile in profiles)
    report: dict[str, Any] = {
        "schema_version": 1,
        "profiles_root": str(profiles_root),
        "kanban_db": str(kanban_db),
        "since_days": since_days,
        "summary": {"profiles": len(profiles), "issues": total_issues},
        "profiles": profiles,
        "run_stats": stats,
    }
    if stats_error:
        report["run_stats_error"] = stats_error
    return report


def render_text(report: dict[str, Any]) -> str:
    """Render the JSON report as a compact human-readable summary."""
    summary = report["summary"]
    lines = [
        "Profile effectiveness audit",
        f"Profiles: {summary['profiles']}  Issues: {summary['issues']}  Window: {report['since_days']} day(s)",
        "",
    ]
    stats = report.get("run_stats", {})
    for profile in report["profiles"]:
        name = profile["name"]
        one_stats = stats.get(name, {})
        rate = one_stats.get("completion_rate")
        rate_text = "n/a" if rate is None else f"{rate * 100:.1f}%"
        lines.append(
            f"{name}: {len(profile['issues'])} issue(s); "
            f"runs {one_stats.get('completed', 0)}/{one_stats.get('finished', 0)} completed ({rate_text})"
        )
        for issue in profile["issues"]:
            lines.append(f"  - [{issue['severity']}] {issue['rule']}: {issue['message']}")
    if report.get("run_stats_error"):
        lines.extend(("", f"Run stats unavailable: {report['run_stats_error']}"))
    return "\n".join(lines)


def load_policy(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    policy, error = _read_mapping(path)
    if error:
        raise ValueError(error)
    return policy
