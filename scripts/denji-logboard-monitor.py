#!/usr/bin/env python3
"""
denji-logboard-monitor.py — Governance logboard monitor for Denji.

Scans all profile errors.log + governance logboard JSON files for patterns
requiring investigation. Outputs JSON findings to stdout, or nothing (silent)
when there are no actionable findings.

Detection rules:
  - overdue_followup: ledger entries >14 days pending
  - quality_gate_consecutive_failures: 2+ consecutive quality gate fails
  - system_health_memory: >85% RAM
  - system_health_swap: >50% swap
  - system_health_disk: >90% disk
  - error_log_patterns: recurring error patterns in errors.log (last 4h)
  - cron_failures: from system-health JSON findings
"""
import os, sys, json, glob, time, re, sqlite3
from datetime import datetime, timedelta, timezone

HERMES = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
LOGBOARD = os.path.join(HERMES, "governance", "logboard")
FOUR_HRS = 4 * 3600
SYSTEM_HEALTH_MAX_AGE_SECONDS = 36 * 3600

SEVERITY_ORDER = {"info": 0, "warning": 1, "error": 2, "critical": 3}


def _timestamp_epoch(value):
    """Return an epoch for an ISO timestamp, or None when it is untrusted."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _profile_to_domain(profile_name):
    """Map profile name to a domain for routing."""
    name = profile_name.lower()
    if any(k in name for k in ["wesker", "ops"]):
        return "ops"
    if any(k in name for k in ["remii", "research", "scanner"]):
        return "research"
    if any(k in name for k in ["octacon", "testrunner", "frontend"]):
        return "apps"
    if any(k in name for k in ["ceecee", "content", "dezzy"]):
        return "content"
    return "default"


def _dedup(findings):
    """Deduplicate by (rule, domain), keep highest severity."""
    seen = {}
    for f in findings:
        key = (f["rule"], f.get("domain", "default"))
        if key not in seen:
            seen[key] = f
        else:
            old_count = seen[key].get("count", 1)
            if SEVERITY_ORDER.get(f["severity"], 0) > SEVERITY_ORDER.get(seen[key]["severity"], 0):
                f["count"] = old_count + f.get("count", 1)
                seen[key] = f
            else:
                seen[key]["count"] = old_count + f.get("count", 1)
    return list(seen.values())


def parse_system_health():
    """Parse latest system-health JSON for memory/swap/disk/cron issues."""
    findings = []
    pattern = os.path.join(LOGBOARD, "system-health-*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return findings

    latest = files[-1]
    try:
        with open(latest) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return findings

    ts = data.get("timestamp", "")
    observed_at = _timestamp_epoch(ts)
    if observed_at is None or time.time() - observed_at > SYSTEM_HEALTH_MAX_AGE_SECONDS:
        # Historical health output proves its own run, not current state.
        # Live cron failures come from the executions/incident store below.
        return findings

    for finding in data.get("findings", []):
        priority = finding.get("priority", "P3")
        title = finding.get("title", "")
        body = finding.get("body", "")
        slug = finding.get("slug", "")

        if slug == "cron-errors":
            # Superseded by parse_cron_incidents(), which checks that the
            # authoritative latest execution still failed.
            continue

        if priority in ("P1", "P2"):
            severity = "warning"
            domain = "ops"
            if "kanban" in slug or "kanban" in title.lower():
                domain = "default"
            findings.append({
                "rule": f"system_health_{slug}",
                "severity": severity,
                "classification": "pattern",
                "domain": domain,
                "detail": f"{title}: {body[:200]}",
                "source_file": latest,
                "timestamp": ts,
                "count": 1,
            })

    # Also check raw metrics if present
    metrics = data.get("metrics", data.get("system", {}))
    mem_pct = metrics.get("memory_percent", metrics.get("memory", {}).get("percent"))
    swap_pct = metrics.get("swap_percent", metrics.get("swap", {}).get("percent"))
    disk_pct = metrics.get("disk_percent", metrics.get("disk", {}).get("percent"))

    if mem_pct and mem_pct > 85:
        findings.append({
            "rule": "system_health_memory",
            "severity": "warning",
            "classification": "pattern",
            "domain": "ops",
            "detail": f"Memory usage {mem_pct}% (>85% threshold)",
            "source_file": latest,
            "timestamp": ts,
            "count": 1,
        })
    if swap_pct and swap_pct > 50:
        findings.append({
            "rule": "system_health_swap",
            "severity": "warning",
            "classification": "pattern",
            "domain": "ops",
            "detail": f"Swap usage {swap_pct}% (>50% threshold)",
            "source_file": latest,
            "timestamp": ts,
            "count": 1,
        })
    if disk_pct and disk_pct > 90:
        findings.append({
            "rule": "system_health_disk",
            "severity": "warning",
            "classification": "pattern",
            "domain": "ops",
            "detail": f"Disk usage {disk_pct}% (>90% threshold)",
            "source_file": latest,
            "timestamp": ts,
            "count": 1,
        })

    return findings


def parse_cron_incidents():
    """Return open cron incidents whose authoritative latest run still failed."""
    db_path = os.path.join(HERMES, "cron", "executions.db")
    if not os.path.isfile(db_path):
        return []

    names = {}
    jobs_path = os.path.join(HERMES, "cron", "jobs.json")
    try:
        with open(jobs_path) as f:
            jobs_data = json.load(f)
        for job in jobs_data.get("jobs", []):
            if isinstance(job, dict) and job.get("id"):
                names[str(job["id"])] = str(job.get("name") or job["id"])
    except (OSError, json.JSONDecodeError, AttributeError):
        pass

    conn = None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if not {"cron_incidents", "executions"}.issubset(tables):
            return []

        incidents = conn.execute(
            "SELECT * FROM cron_incidents "
            "WHERE state IN ('detected','alerted') "
            "ORDER BY last_seen_at DESC, id DESC"
        ).fetchall()
        findings = []
        for incident in incidents:
            latest = conn.execute(
                "SELECT status, started_at, finished_at, error FROM executions "
                "WHERE job_id=? "
                "ORDER BY COALESCE(started_at, finished_at) DESC, id DESC "
                "LIMIT 1",
                (incident["job_id"],),
            ).fetchone()
            if latest is None or latest["status"] not in ("failed", "error"):
                # The incident row can remain open for audit/ack purposes, but
                # a later successful run makes it non-current.
                continue
            job_name = names.get(incident["job_id"], incident["job_id"])
            error = str(latest["error"] or incident["error"] or "unknown error")
            failure_type = str(incident["failure_type"] or "unknown")
            findings.append(
                {
                    "rule": f"cron_incident_{incident['id']}",
                    "severity": (
                        "error"
                        if failure_type in {"auth", "config", "delivery"}
                        else "warning"
                    ),
                    "classification": "pattern",
                    "domain": "ops",
                    "detail": f"Cron '{job_name}' latest run failed: {error[:200]}",
                    "source_file": db_path,
                    "timestamp": latest["started_at"] or incident["last_seen_at"],
                    "count": 1,
                    "incident_id": incident["id"],
                    "job_id": incident["job_id"],
                }
            )
        return findings
    except (sqlite3.Error, OSError):
        return []
    finally:
        if conn is not None:
            conn.close()


def parse_quality_gates():
    """Check for consecutive quality gate failures per profile."""
    findings = []
    pattern = os.path.join(LOGBOARD, "quality-gate-*.json")
    files = sorted(glob.glob(pattern))

    # Look at last 10 quality gate files
    recent = files[-10:] if len(files) > 10 else files
    profile_fails = {}

    for fpath in recent:
        try:
            with open(fpath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        ts = data.get("timestamp", "")
        for result in data.get("results", []):
            profile = result.get("profile", "default")
            passed = result.get("passed", result.get("status") == "pass")
            if not passed:
                if profile not in profile_fails:
                    profile_fails[profile] = {"count": 0, "last_ts": ts}
                profile_fails[profile]["count"] += 1
                profile_fails[profile]["last_ts"] = ts

    for profile, info in profile_fails.items():
        if info["count"] >= 2:
            domain = _profile_to_domain(profile)
            findings.append({
                "rule": "quality_gate_consecutive_failures",
                "severity": "warning",
                "classification": "pattern",
                "domain": domain,
                "detail": f"Profile '{profile}' failed quality gate {info['count']} times in recent checks",
                "source_file": "quality-gate-*",
                "timestamp": info["last_ts"],
                "count": info["count"],
                "escalated": info["count"] >= 4,
            })

    return findings


def parse_ledger_overdue():
    """Check profile-activity-ledger for overdue followups."""
    findings = []
    ledger_dir = os.path.join(LOGBOARD, "profile-activity-ledger")
    if not os.path.isdir(ledger_dir):
        return findings

    cutoff = time.time() - (14 * 86400)  # 14 days

    for fpath in sorted(glob.glob(os.path.join(ledger_dir, "*.jsonl")))[-7:]:  # last 7 days
        try:
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    event_type = entry.get("event_type", "")
                    if "pending" in str(entry.get("status_to", "")).lower():
                        occurred = entry.get("occurred_at", 0)
                        if occurred and occurred < cutoff:
                            profile = entry.get("actor_profile", "unknown")
                            findings.append({
                                "rule": "overdue_followup",
                                "severity": "warning",
                                "classification": "pattern",
                                "domain": _profile_to_domain(profile),
                                "detail": f"Pending followup in ledger for {entry.get('summary', '')[:100]} (>{14}d old)",
                                "source_file": fpath,
                                "timestamp": datetime.fromtimestamp(occurred).isoformat(),
                                "count": 1,
                            })
        except IOError:
            continue

    return findings


def parse_error_logs():
    """Scan all profile errors.log for recurring patterns in last 4 hours."""
    findings = []
    now = time.time()
    cutoff = now - FOUR_HRS

    # Collect all errors.log paths
    log_dirs = {"main": os.path.join(HERMES, "logs")}
    for d in glob.glob(os.path.join(HERMES, "profiles", "*", "logs")):
        name = d.split("/profiles/")[1].split("/logs")[0]
        log_dirs[name] = d

    # Pattern categories to track
    categories = {
        "resource_exhausted": {
            "regex": re.compile(r"ResourceExhausted"),
            "severity": "warning",
            "domain": "ops",
            "detail": "Provider resource exhaustion (rate limits)",
        },
        "mcp_keepalive": {
            "regex": re.compile(r"keepalive failed"),
            "severity": "warning",
            "domain": "ops",
            "detail": "MCP server keepalive failures (reconnect cycles)",
        },
        "auth_failure": {
            "regex": re.compile(r"HTTP 401|Invalid API key|AuthenticationError"),
            "severity": "warning",
            "domain": "ops",
            "detail": "Provider authentication failures (invalid API keys)",
        },
        "rate_limit_429": {
            "regex": re.compile(r"HTTP 429|Weekly usage limit|RateLimitError"),
            "severity": "warning",
            "domain": "ops",
            "detail": "Provider rate limit hits (HTTP 429)",
        },
        "iteration_limit": {
            "regex": re.compile(r"iteration limit"),
            "severity": "warning",
            "domain": "ops",
            "detail": "Cron jobs hitting iteration limits",
        },
        "mcp_outlook_error": {
            "regex": re.compile(r"mcp_outlook|MCP error.*outlook|ErrorInvalidIdMalformed"),
            "severity": "warning",
            "domain": "default",
            "detail": "Outlook MCP errors (malformed IDs, validation failures)",
        },
        "python_import_error": {
            "regex": re.compile(r"ImportError|AttributeError.*has no attribute"),
            "severity": "warning",
            "domain": "apps",
            "detail": "Python import/attribute errors in scripts",
        },
    }

    for profile, log_dir in log_dirs.items():
        errfile = os.path.join(log_dir, "errors.log")
        if not os.path.isfile(errfile):
            continue

        cat_counts = {cat: {"count": 0, "samples": []} for cat in categories}
        total_count = 0

        try:
            with open(errfile, "r", errors="replace") as f:
                for line in f:
                    if len(line) < 19 or line[4:5] != "-" or line[7:8] != "-":
                        continue
                    try:
                        dt = datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")
                        epoch = dt.timestamp()
                    except ValueError:
                        continue

                    if epoch < cutoff:
                        continue

                    total_count += 1

                    for cat, config in categories.items():
                        if config["regex"].search(line):
                            cat_counts[cat]["count"] += 1
                            if len(cat_counts[cat]["samples"]) < 2:
                                cat_counts[cat]["samples"].append(line[:200])
        except IOError:
            continue

        # Report categories with 3+ occurrences (pattern threshold)
        for cat, data in cat_counts.items():
            if data["count"] >= 3:
                config = categories[cat]
                domain = config["domain"]
                # Route outlook errors to Gojo (admin/mailbox)
                if cat == "mcp_outlook_error":
                    domain = "default"
                # Route auth/rate limit to the profile's domain if not main
                if profile != "main" and cat in ("auth_failure", "rate_limit_429"):
                    domain = _profile_to_domain(profile)

                detail = f"[{profile}] {config['detail']} — {data['count']} occurrences"
                if data["samples"]:
                    detail += f" | Sample: {data['samples'][0][:100]}"

                findings.append({
                    "rule": f"error_log_{cat}",
                    "severity": config["severity"],
                    "classification": "pattern" if data["count"] >= 5 else "emerging",
                    "domain": domain,
                    "detail": detail,
                    "source_file": errfile,
                    "timestamp": datetime.now().isoformat(),
                    "count": data["count"],
                    "escalated": data["count"] >= 10,
                    "profile": profile,
                })

    return findings


def main():
    findings = []
    findings.extend(parse_cron_incidents())
    findings.extend(parse_system_health())
    findings.extend(parse_quality_gates())
    findings.extend(parse_ledger_overdue())
    findings.extend(parse_error_logs())

    findings = _dedup(findings)

    if not findings:
        # Silent — no stdout
        return

    # Output as JSON array
    print(json.dumps(findings, indent=2))


if __name__ == "__main__":
    main()
