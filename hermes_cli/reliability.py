"""Bounded reliability checks for orchestration-first Hermes profiles."""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml

AUTH_PAT = re.compile(r"token_revoked|invalidated oauth|http\s*401|error code:\s*401", re.I)
RATE_PAT = re.compile(r"usage_limit_reached|rate[- ]?limited|http\s*429|error code:\s*429", re.I)
WS_PAT = re.compile(r"send_failed_after_response|websocket.*disconnect|ws send failed", re.I)
READY_PAT = re.compile(r"readiness probe failed|gateway readiness|dashboard readiness", re.I)

DISABLED_SKILL_POLICY = {
    "spotify",
    "teams_pipeline",
    "teams-meeting-pipeline",
    "google-meet",
    "google_meet",
    "google-workspace",
    "apple-notes",
    "apple-reminders",
    "findmy",
    "imessage",
    "himalaya",
    "heartmula",
    "touchdesigner-mcp",
    "jupyter-live-kernel",
    "weights-and-biases",
    "huggingface-hub",
    "llama-cpp",
    "serving-llms-vllm",
    "segment-anything-model",
    "maps",
    "xurl",
    "godmode",
}


def _profile_home(explicit: str | os.PathLike[str] | None = None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    env_home = os.environ.get("HERMES_HOME")
    if env_home:
        return Path(env_home).expanduser()
    return Path.home() / ".hermes"


def _root_home(profile_home: Path) -> Path:
    if profile_home.parent.name == "profiles":
        return profile_home.parent.parent
    return profile_home


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _issue(
    issue_id: str,
    severity: str,
    title: str,
    detail: str,
    *,
    repair: str = "blocked",
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": issue_id,
        "severity": severity,
        "title": title,
        "detail": detail,
        "repair": repair,
        "data": data or {},
    }


def _check_url(url: str, timeout: float = 2.0) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read(16_384)
            parsed: Any = None
            try:
                parsed = json.loads(body.decode("utf-8"))
            except Exception:
                parsed = body.decode("utf-8", errors="replace")[:500]
            return {"ok": True, "status": response.status, "body": parsed}
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        return {"ok": False, "error": str(exc)}


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    except sqlite3.Error:
        return set()


def _count_state(profile_home: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    state_path = profile_home / "state.db"
    data: dict[str, Any] = {"path": str(state_path), "exists": state_path.exists()}
    issues: list[dict[str, Any]] = []
    if not state_path.exists():
        return data, issues
    data["bytes"] = state_path.stat().st_size
    try:
        conn = sqlite3.connect(str(state_path))
        cols = _table_columns(conn, "sessions")
        if "sessions" in {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }:
            data["sessions"] = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            if "message_count" in cols:
                data["zero_message_sessions"] = conn.execute(
                    "SELECT COUNT(*) FROM sessions WHERE COALESCE(message_count, 0)=0"
                ).fetchone()[0]
                data["max_message_count"] = conn.execute(
                    "SELECT MAX(COALESCE(message_count, 0)) FROM sessions"
                ).fetchone()[0] or 0
            status_expr = "status" if "status" in cols else None
            archived_expr = "archived" if "archived" in cols else None
            if status_expr:
                open_where = "status IN ('open', 'active', 'running')"
                if archived_expr:
                    open_where += " AND COALESCE(archived, 0)=0"
                data["open_sessions"] = conn.execute(
                    f"SELECT COUNT(*) FROM sessions WHERE {open_where}"
                ).fetchone()[0]
    except sqlite3.Error as exc:
        issues.append(_issue(
            "state_db_unreadable",
            "error",
            "State database could not be inspected",
            str(exc),
        ))
    finally:
        try:
            conn.close()  # type: ignore[name-defined]
        except Exception:
            pass
    return data, issues


def _scan_logs(profile_home: Path) -> dict[str, Any]:
    logs = [
        profile_home / "logs" / "errors.log",
        profile_home / "logs" / "gui.log",
        profile_home / "logs" / "agent.log",
        profile_home / "logs" / "dashboard-supervisor.log",
        profile_home / "logs" / "gateway-supervisor.log",
    ]
    counts = {"auth_401": 0, "rate_429": 0, "websocket_disconnect": 0, "readiness": 0}
    files: dict[str, dict[str, Any]] = {}
    for path in logs:
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")[-200_000:]
        except OSError:
            continue
        files[path.name] = {"bytes": path.stat().st_size}
        counts["auth_401"] += len(AUTH_PAT.findall(text))
        counts["rate_429"] += len(RATE_PAT.findall(text))
        counts["websocket_disconnect"] += len(WS_PAT.findall(text))
        counts["readiness"] += len(READY_PAT.findall(text))
    return {"counts": counts, "files": files}


def _count_request_dumps(profile_home: Path) -> dict[str, Any]:
    count = 0
    total = 0
    for path in profile_home.rglob("request_dump_*.json"):
        try:
            count += 1
            total += path.stat().st_size
        except OSError:
            pass
    return {"count": count, "bytes": total}


def _process_snapshot() -> dict[str, Any]:
    try:
        proc = subprocess.run(
            ["ps", "-eo", "stat="],
            text=True,
            capture_output=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc)}
    zombies = sum(1 for line in proc.stdout.splitlines() if "Z" in line)
    return {"ok": proc.returncode == 0, "zombies": zombies}


def _kanban_snapshot() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    data: dict[str, Any] = {"ok": False}
    try:
        from hermes_cli import kanban_db as kb

        kb.init_db()
        with kb.connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS count FROM tasks GROUP BY status"
            ).fetchall()
            statuses = {row["status"]: int(row["count"]) for row in rows}
            data = {"ok": True, "statuses": statuses, "db": str(kb.kanban_db_path())}
            active_ready = statuses.get("ready", 0) + statuses.get("running", 0)
            waiting = statuses.get("todo", 0) + statuses.get("blocked", 0)
            if active_ready == 0 and waiting:
                issues.append(_issue(
                    "kanban_stalled_no_ready_running",
                    "warning",
                    "Kanban has waiting work but no active ready/running lane",
                    "The board contains blocked or waiting tasks but no ready or running tasks.",
                    repair="kanban_card",
                    data={"statuses": statuses},
                ))
            shared = conn.execute(
                "SELECT workspace_path, COUNT(*) AS count "
                "FROM tasks WHERE status IN ('ready','running','review') "
                "AND workspace_path IS NOT NULL AND workspace_path != '' "
                "GROUP BY workspace_path HAVING COUNT(*) > 1"
            ).fetchall()
            if shared:
                issues.append(_issue(
                    "shared_active_worktree",
                    "warning",
                    "Multiple active coding tasks share a workspace path",
                    "Future coding cards should use unique worktrees to avoid dirty-worktree collisions.",
                    repair="kanban_card",
                    data={"paths": {row["workspace_path"]: row["count"] for row in shared}},
                ))
    except Exception as exc:
        issues.append(_issue(
            "kanban_unreadable",
            "error",
            "Kanban database could not be inspected",
            str(exc),
        ))
    return data, issues


def _config_issues(config: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if not config.get("fallback_providers"):
        issues.append(_issue(
            "fallback_not_configured",
            "info",
            "Fallback providers are prepared but not configured",
            "No fallback provider is active. This pass intentionally does not add credentials.",
            repair="blocked",
        ))
    skills_cfg = config.get("skills") if isinstance(config.get("skills"), dict) else {}
    if not skills_cfg.get("guard_agent_created"):
        issues.append(_issue(
            "skills_guard_disabled",
            "error",
            "Agent-created skill guard is disabled",
            "Set skills.guard_agent_created=true so generated skills are reviewed before use.",
            repair="config",
        ))
    disabled = set()
    raw_disabled = skills_cfg.get("disabled") if isinstance(skills_cfg, dict) else []
    if isinstance(raw_disabled, list):
        disabled.update(str(item).strip() for item in raw_disabled)
    missing = sorted(s for s in DISABLED_SKILL_POLICY if s not in disabled)
    if missing:
        issues.append(_issue(
            "engineering_skill_disable_policy_incomplete",
            "warning",
            "Engineering-core skill disable policy is incomplete",
            "Irrelevant or risky skills should be absent or disabled.",
            repair="config",
            data={"missing": missing[:50]},
        ))
    known = config.get("known_plugin_toolsets")
    cli = known.get("cli") if isinstance(known, dict) else None
    if isinstance(cli, list) and "spotify" in {str(item) for item in cli}:
        issues.append(_issue(
            "spotify_toolset_registered",
            "warning",
            "Spotify toolset is still registered",
            "Remove spotify from known_plugin_toolsets.cli or disable it via agent.disabled_toolsets.",
            repair="config",
        ))
    return issues


def build_reliability_report(
    profile_home: str | os.PathLike[str] | None = None,
    *,
    dashboard_url: str | None = None,
    now: int | None = None,
) -> dict[str, Any]:
    home = _profile_home(profile_home)
    root = _root_home(home)
    ts = int(now or time.time())
    config = _read_yaml(home / "config.yaml")
    issues: list[dict[str, Any]] = []
    issues.extend(_config_issues(config))

    status_url = dashboard_url or os.environ.get(
        "HERMES_DASHBOARD_URL", "http://127.0.0.1:9120"
    )
    dashboard = {
        "status": _check_url(status_url.rstrip("/") + "/api/status"),
        "readiness": _check_url(status_url.rstrip("/") + "/api/readiness"),
    }
    if not dashboard["status"].get("ok"):
        issues.append(_issue(
            "dashboard_status_unreachable",
            "error",
            "Dashboard status endpoint is unreachable",
            str(dashboard["status"].get("error") or "unknown error"),
            repair="kanban_card",
        ))
    if not dashboard["readiness"].get("ok"):
        issues.append(_issue(
            "dashboard_readiness_unreachable",
            "warning",
            "Dashboard readiness endpoint is unreachable",
            str(dashboard["readiness"].get("error") or "unknown error"),
            repair="kanban_card",
        ))

    logs = _scan_logs(home)
    if logs["counts"]["auth_401"]:
        issues.append(_issue(
            "provider_auth_401_burst",
            "critical",
            "Recent logs contain provider authentication failures",
            "Re-authenticate the affected provider/profile before dispatching more work.",
            repair="blocked",
            data={"count": logs["counts"]["auth_401"]},
        ))
    if logs["counts"]["rate_429"]:
        issues.append(_issue(
            "provider_rate_429_burst",
            "warning",
            "Recent logs contain provider rate/quota failures",
            "Use cooldowns or approved fallback credentials before retrying large queues.",
            repair="kanban_card",
            data={"count": logs["counts"]["rate_429"]},
        ))

    state, state_issues = _count_state(home)
    issues.extend(state_issues)
    kanban, kanban_issues = _kanban_snapshot()
    issues.extend(kanban_issues)
    proc = _process_snapshot()
    if proc.get("zombies", 0) > 0:
        issues.append(_issue(
            "zombie_processes_present",
            "warning",
            "Zombie processes are present",
            "Restart the owning supervisor if the count grows or workers stop dispatching.",
            repair="kanban_card",
            data={"zombies": proc.get("zombies")},
        ))

    request_dumps = _count_request_dumps(home)
    if request_dumps["bytes"] > 250_000_000:
        issues.append(_issue(
            "request_dumps_large",
            "warning",
            "Request dumps exceed retention target",
            "Rotate or compress request dumps after preserving a backup.",
            repair="bounded",
            data=request_dumps,
        ))

    return {
        "schema_version": 1,
        "generated_at": ts,
        "profile_home": str(home),
        "root_home": str(root),
        "dashboard_url": status_url,
        "ok": not any(i["severity"] in {"error", "critical"} for i in issues),
        "issues": issues,
        "checks": {
            "dashboard": dashboard,
            "logs": logs,
            "state": state,
            "kanban": kanban,
            "processes": proc,
            "request_dumps": request_dumps,
        },
    }


def write_reliability_report(
    report: dict[str, Any],
    profile_home: str | os.PathLike[str] | None = None,
) -> Path:
    home = _profile_home(profile_home or report.get("profile_home"))
    out = home / "health" / "reliability.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def _backup_profile(home: Path) -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    backup = _root_home(home) / "backups" / f"reliability-sentinel-{stamp}"
    backup.mkdir(parents=True, exist_ok=True)
    for rel in (
        "config.yaml",
        ".env",
        "SOUL.md",
        "profile.yaml",
        "state.db",
        "cron/jobs.json",
        "logs/gui.log",
        "logs/errors.log",
        "logs/agent.log",
    ):
        src = home / rel
        if src.exists():
            dst = backup / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(src, dst)
            except OSError:
                pass
    try:
        from hermes_cli import kanban_db as kb

        db = kb.kanban_db_path()
        if db.exists():
            dst = backup / "kanban.db"
            shutil.copy2(db, dst)
    except Exception:
        pass
    return backup


def _archive_empty_sessions(home: Path) -> int:
    state_path = home / "state.db"
    if not state_path.exists():
        return 0
    try:
        conn = sqlite3.connect(str(state_path))
        cols = _table_columns(conn, "sessions")
        if "archived" not in cols or "message_count" not in cols:
            return 0
        sets = ["archived=1"]
        if "ended_at" in cols:
            sets.append(f"ended_at={int(time.time())}")
        if "end_reason" in cols:
            sets.append("end_reason='reliability_sentinel_empty_archive'")
        where = "COALESCE(archived, 0)=0 AND COALESCE(message_count, 0)=0"
        cur = conn.execute(f"UPDATE sessions SET {', '.join(sets)} WHERE {where}")
        conn.commit()
        return int(cur.rowcount or 0)
    except sqlite3.Error:
        return 0
    finally:
        try:
            conn.close()  # type: ignore[name-defined]
        except Exception:
            pass


def _create_issue_cards(report: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    try:
        from hermes_cli import kanban_db as kb

        kb.init_db()
        with kb.connect() as conn:
            for issue in report.get("issues", []):
                if issue.get("repair") not in {"kanban_card", "blocked", "bounded"}:
                    continue
                title = f"reliability: {issue.get('title', issue.get('id'))}"
                body = (
                    "Bounded reliability sentinel finding.\n\n"
                    f"```json\n{json.dumps(issue, indent=2, sort_keys=True)}\n```\n\n"
                    "Do not mutate credentials, deployments, merges, or production state "
                    "without explicit operator approval."
                )
                tid = kb.create_task(
                    conn,
                    title=title[:240],
                    body=body,
                    assignee="open-investigate",
                    idempotency_key=f"reliability:{issue.get('id')}",
                    workspace_kind="worktree",
                    workspace_path=(
                        f"/workspace/hermes-worktrees/reliability/"
                        f"{time.strftime('%Y%m%d')}-{issue.get('id')}"
                    ),
                )
                ids.append(tid)
    except Exception:
        return ids
    return ids


def apply_bounded_repairs(
    report: dict[str, Any],
    profile_home: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    home = _profile_home(profile_home or report.get("profile_home"))
    backup = _backup_profile(home)
    archived = _archive_empty_sessions(home)
    cards = _create_issue_cards(report)
    return {
        "backup_dir": str(backup),
        "archived_empty_sessions": archived,
        "kanban_cards": cards,
        "blocked_actions": [
            issue["id"]
            for issue in report.get("issues", [])
            if issue.get("repair") == "blocked"
        ],
    }


def reliability_command(args) -> int:
    sub = getattr(args, "reliability_command", None) or "check"
    if sub != "check":
        print(f"Unknown reliability command: {sub}")
        return 2
    report = build_reliability_report(
        getattr(args, "profile_home", None),
        dashboard_url=getattr(args, "dashboard_url", None),
    )
    if getattr(args, "apply_bounded", False):
        report["bounded_repairs"] = apply_bounded_repairs(
            report,
            getattr(args, "profile_home", None),
        )
    if getattr(args, "write_report", False) or getattr(args, "apply_bounded", False):
        report["report_path"] = str(write_reliability_report(
            report,
            getattr(args, "profile_home", None),
        ))
    if getattr(args, "json", False):
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        status = "OK" if report.get("ok") else "ISSUES"
        print(f"Hermes reliability: {status} ({len(report.get('issues', []))} issues)")
        for issue in report.get("issues", []):
            print(f"- [{issue['severity']}] {issue['id']}: {issue['title']}")
        if report.get("report_path"):
            print(f"Report: {report['report_path']}")
    return 0 if report.get("ok") else 1
