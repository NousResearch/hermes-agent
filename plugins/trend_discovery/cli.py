"""Operator CLI for Trend Discovery Center."""

from __future__ import annotations

import argparse
import json
import shutil
import textwrap
from pathlib import Path
import subprocess
import sys
from typing import Any

from cron.jobs import create_job
from hermes_constants import get_hermes_home

from .health import health_check, health_json
from .knowledge import build_digest, build_reliability_report, write_review_queue
from .notifications import notify, watchdog
from .scanner import TrendScanner
from .store import TrendDiscoveryStore


def register_cli(subparser: argparse.ArgumentParser) -> None:
    sub = subparser.add_subparsers(dest="trend_action")

    init_p = sub.add_parser("init", help="Initialize persistent plan, sources, and config")
    init_p.add_argument("--store-path", default="")

    status_p = sub.add_parser("status", help="Show phase and runtime status")
    status_p.add_argument("--json", action="store_true")
    status_p.add_argument("--store-path", default="")

    ops_p = sub.add_parser("ops", help="Show operator mode, schedules, sources, and control points")
    ops_p.add_argument("--json", action="store_true")
    ops_p.add_argument("--store-path", default="")

    comply_p = sub.add_parser("comply", help="Print numeric compliance table")
    comply_p.add_argument("--json", action="store_true")
    comply_p.add_argument("--store-path", default="")

    scan_p = sub.add_parser("scan", help="Run a resilient multi-source scan")
    scan_p.add_argument("--query", default="agentic workflow startup")
    scan_p.add_argument("--limit", type=int, default=25)
    scan_p.add_argument("--write-review-queue", action="store_true")
    scan_p.add_argument("--store-path", default="")

    watch_p = sub.add_parser("watchdog", help="Run overdue/source failure watchdog")
    watch_p.add_argument("--no-notify", action="store_true")
    watch_p.add_argument("--store-path", default="")

    health_p = sub.add_parser("health", help="Run local health checks")
    health_p.add_argument("--json", action="store_true")
    health_p.add_argument("--check-dashboard", action="store_true")
    health_p.add_argument("--store-path", default="")

    digest_p = sub.add_parser("digest", help="Print or write latest digest")
    digest_p.add_argument("--write-review-queue", action="store_true")
    digest_p.add_argument("--store-path", default="")

    report_p = sub.add_parser("reliability-report", help="Print source and notification reliability report")
    report_p.add_argument("--store-path", default="")

    sources_p = sub.add_parser("sources", help="List and manage source adapters")
    sources_p.add_argument("source_action", choices=["list", "add", "enable", "disable", "delete"])
    sources_p.add_argument("--name", default="")
    sources_p.add_argument("--adapter", default="")
    sources_p.add_argument("--url", default="")
    sources_p.add_argument("--priority", type=int, default=50)
    sources_p.add_argument("--timeout", type=int, default=20)
    sources_p.add_argument("--metadata", default="{}")
    sources_p.add_argument("--json", action="store_true")
    sources_p.add_argument("--store-path", default="")

    notify_p = sub.add_parser("notify", help="Send a notification through configured target")
    notify_p.add_argument("message")
    notify_p.add_argument("--target", default="")
    notify_p.add_argument("--store-path", default="")

    config_p = sub.add_parser("configure", help="Set Trend Discovery runtime config")
    config_p.add_argument("--notification-primary", default="")
    config_p.add_argument("--notification-fallback", default="")
    config_p.add_argument("--localhost-url", default="")
    config_p.add_argument("--store-path", default="")

    done_p = sub.add_parser("mark-all-complete", help="Mark all canonical issues complete after verification")
    done_p.add_argument("--evidence", default="operator-approved implementation and tests")
    done_p.add_argument("--store-path", default="")

    cron_p = sub.add_parser("install-cron", help="Install Hermes no-agent watchdog cron job")
    cron_p.add_argument("--schedule", default="every 1d")
    cron_p.add_argument("--deliver", default="local")
    cron_p.add_argument("--store-path", default="")

    launchd_p = sub.add_parser("install-launchd", help="Install macOS launchd jobs for scan and watchdog")
    launchd_p.add_argument("--scan-interval", type=int, default=86400)
    launchd_p.add_argument("--watchdog-interval", type=int, default=3600)
    launchd_p.add_argument("--query", default="agentic workflow startup")
    launchd_p.add_argument("--store-path", default="")

    launchd_status_p = sub.add_parser("launchd-status", help="Show macOS launchd job status")
    launchd_status_p.add_argument("--store-path", default="")

    launchd_uninstall_p = sub.add_parser("uninstall-launchd", help="Unload macOS launchd jobs but keep DB/logs")
    launchd_uninstall_p.add_argument("--store-path", default="")

    logs_p = sub.add_parser("logs", help="Print recent launchd stdout/stderr logs")
    logs_p.add_argument("--lines", type=int, default=80)
    logs_p.add_argument("--store-path", default="")

    subparser.set_defaults(func=trend_discovery_command)


def _store(args: argparse.Namespace) -> TrendDiscoveryStore:
    return TrendDiscoveryStore(getattr(args, "store_path", "") or None)


def _print_status(snapshot: dict) -> None:
    print(f"Store: {snapshot['store_path']}")
    print(f"Findings: {snapshot['findings_count']}")
    print("Phases:")
    for phase in snapshot["phases"]:
        print(
            f"  {phase['phase_id']} {phase['percent_complete']:3d}% "
            f"{phase['status']} - {phase['name']}"
        )


def _print_ops(store: TrendDiscoveryStore) -> None:
    snapshot = store.status_snapshot()
    launchd = _launchd_status()
    print("Trend Discovery Center")
    print(f"Store: {snapshot['store_path']}")
    print(f"Findings: {snapshot['findings_count']}")
    print("")
    print("Modes:")
    print("  scheduled-scan      launchd daily scan, writes review queue")
    print("  scheduled-watchdog  launchd hourly watchdog, sends notification only on alerts")
    print("  manual-scan         operator-triggered scan")
    print("  manual-watchdog     operator-triggered watchdog check")
    print("  compliance          numeric phase/issue completion report")
    print("  source-admin        list/add/enable/disable/delete source adapters")
    print("  knowledge-writeback digest/review-queue output")
    print("")
    print("Launchd:")
    for label, result in launchd.items():
        stdout = result.get("stdout", "")
        runs = _extract_launchd_line(stdout, "runs =")
        exit_code = _extract_launchd_line(stdout, "last exit code =")
        interval = _extract_launchd_line(stdout, "run interval =")
        print(f"  {label}: {runs or 'runs = ?'}, {exit_code or 'last exit code = ?'}, {interval or 'run interval = ?'}")
    print("")
    print("Sources:")
    for source in snapshot["sources"]:
        enabled = "enabled" if source["enabled"] else "disabled"
        print(
            f"  {source['name']} [{source['adapter']}] {enabled} "
            f"status={source['last_status'] or '-'} success={source['success_count']} failure={source['failure_count']}"
        )


def _extract_launchd_line(text: str, prefix: str) -> str:
    for line in str(text or "").splitlines():
        clean = line.strip()
        if clean.startswith(prefix):
            return clean
    return ""


def _sources_payload(store: TrendDiscoveryStore) -> list[dict[str, Any]]:
    with store.connect() as conn:
        rows = conn.execute(
            """
            SELECT name, adapter, url, enabled, priority, timeout_seconds,
                   failure_count, success_count, circuit_open_until,
                   last_status, last_error, metadata
            FROM sources
            ORDER BY priority, name
            """
        ).fetchall()
    return [dict(row) for row in rows]


def _source_required(args: argparse.Namespace) -> str:
    name = str(getattr(args, "name", "") or "").strip()
    if not name:
        raise ValueError("--name is required for this source action")
    return name


def _handle_sources(store: TrendDiscoveryStore, args: argparse.Namespace) -> int:
    action = args.source_action
    if action == "list":
        rows = _sources_payload(store)
        if getattr(args, "json", False):
            print(json.dumps(rows, indent=2, sort_keys=True))
        else:
            for source in rows:
                enabled = "enabled" if source["enabled"] else "disabled"
                print(
                    f"{source['name']} {source['adapter']} {enabled} "
                    f"priority={source['priority']} status={source['last_status'] or '-'} "
                    f"success={source['success_count']} failure={source['failure_count']} url={source['url']}"
                )
        return 0

    name = _source_required(args)
    if action == "add":
        adapter = str(args.adapter or "").strip()
        if adapter not in {"rss", "webpage", "open_crawl", "n8n", "searxng"}:
            raise ValueError("--adapter must be one of: rss, webpage, open_crawl, n8n, searxng")
        try:
            metadata = json.loads(args.metadata or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"--metadata must be valid JSON: {exc}") from exc
        with store.connect() as conn:
            conn.execute(
                """
                INSERT INTO sources
                    (name, adapter, url, enabled, priority, timeout_seconds, metadata)
                VALUES (?, ?, ?, 1, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    adapter=excluded.adapter,
                    url=excluded.url,
                    enabled=1,
                    priority=excluded.priority,
                    timeout_seconds=excluded.timeout_seconds,
                    metadata=excluded.metadata
                """,
                (
                    name,
                    adapter,
                    str(args.url or "").strip(),
                    int(args.priority),
                    int(args.timeout),
                    json.dumps(metadata, sort_keys=True),
                ),
            )
        print(json.dumps({"ok": True, "action": "add", "name": name}, indent=2, sort_keys=True))
        return 0

    if action in {"enable", "disable", "delete"}:
        with store.connect() as conn:
            existing = conn.execute("SELECT name FROM sources WHERE name=?", (name,)).fetchone()
            if existing is None:
                raise ValueError(f"source not found: {name}")
            if action == "delete":
                conn.execute("DELETE FROM sources WHERE name=?", (name,))
            else:
                conn.execute(
                    "UPDATE sources SET enabled=? WHERE name=?",
                    (1 if action == "enable" else 0, name),
                )
        print(json.dumps({"ok": True, "action": action, "name": name}, indent=2, sort_keys=True))
        return 0

    raise ValueError(f"unknown source action: {action}")
    print("Sources:")
    for source in snapshot["sources"]:
        print(
            f"  {source['name']} [{source['adapter']}] success={source['success_count']} "
            f"failure={source['failure_count']} status={source['last_status'] or '-'}"
        )


def _print_comply(rows: list[dict]) -> None:
    for row in rows:
        print(f"{row['issue_id']} {row['percent_complete']} {row['remaining_percent']}")


def _install_cron_script(store: TrendDiscoveryStore, schedule: str, deliver: str) -> dict:
    scripts_dir = get_hermes_home() / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script = scripts_dir / "trend_discovery_watchdog.py"
    script.write_text(
        textwrap.dedent(
            f"""
            from plugins.trend_discovery.notifications import watchdog
            from plugins.trend_discovery.store import TrendDiscoveryStore

            result = watchdog(TrendDiscoveryStore({str(store.path)!r}), notify_user=True)
            if result.get("ok"):
                print("[SILENT]")
            else:
                print("Trend Discovery watchdog alerts:")
                for alert in result.get("alerts", []):
                    print("- " + alert)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    job = create_job(
        prompt="Trend Discovery watchdog",
        schedule=schedule,
        name="Trend Discovery Watchdog",
        deliver=deliver,
        script=script.name,
        no_agent=True,
    )
    return {"script": str(script), "job": job}


def _launchd_dir() -> Path:
    path = Path.home() / "Library" / "LaunchAgents"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _python_executable() -> str:
    return sys.executable


def _write_plist(label: str, args: list[str], interval: int, log_name: str) -> Path:
    log_dir = get_hermes_home() / "trend-discovery" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    plist_path = _launchd_dir() / f"{label}.plist"
    program_args = "\n".join(f"    <string>{arg}</string>" for arg in args)
    plist_path.write_text(
        textwrap.dedent(
            f"""\
            <?xml version="1.0" encoding="UTF-8"?>
            <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
            <plist version="1.0">
            <dict>
              <key>Label</key>
              <string>{label}</string>
              <key>ProgramArguments</key>
              <array>
            {program_args}
              </array>
              <key>WorkingDirectory</key>
              <string>{_repo_root()}</string>
              <key>StartInterval</key>
              <integer>{interval}</integer>
              <key>RunAtLoad</key>
              <true/>
              <key>StandardOutPath</key>
              <string>{log_dir / (log_name + ".out.log")}</string>
              <key>StandardErrorPath</key>
              <string>{log_dir / (log_name + ".err.log")}</string>
            </dict>
            </plist>
            """
        ),
        encoding="utf-8",
    )
    return plist_path


def _launchctl(*args: str) -> dict:
    launchctl = shutil.which("launchctl")
    if not launchctl:
        return {"ok": False, "error": "launchctl not found"}
    result = subprocess.run(
        [launchctl, *args],
        capture_output=True,
        text=True,
        timeout=15,
    )
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _install_launchd(store: TrendDiscoveryStore, *, scan_interval: int, watchdog_interval: int, query: str) -> dict:
    uid = subprocess.run(["id", "-u"], capture_output=True, text=True, timeout=5).stdout.strip()
    domain = f"gui/{uid}" if uid else "gui"
    python = _python_executable()
    scan_label = "com.hermes.trend-discovery.scan"
    watchdog_label = "com.hermes.trend-discovery.watchdog"
    scan_plist = _write_plist(
        scan_label,
        [
            python,
            "-m",
            "plugins.trend_discovery.cli_runner",
            "scan",
            "--query",
            query,
            "--write-review-queue",
            "--store-path",
            str(store.path),
        ],
        max(300, scan_interval),
        "scan",
    )
    watchdog_plist = _write_plist(
        watchdog_label,
        [
            python,
            "-m",
            "plugins.trend_discovery.cli_runner",
            "watchdog",
            "--store-path",
            str(store.path),
        ],
        max(300, watchdog_interval),
        "watchdog",
    )
    results = {}
    for label, plist in ((scan_label, scan_plist), (watchdog_label, watchdog_plist)):
        _launchctl("bootout", domain, str(plist))
        bootstrap = _launchctl("bootstrap", domain, str(plist))
        kickstart = _launchctl("kickstart", "-k", f"{domain}/{label}")
        results[label] = {
            "plist": str(plist),
            "bootstrap": bootstrap,
            "kickstart": kickstart,
        }
    return results


def _launchd_status() -> dict:
    labels = (
        "com.hermes.trend-discovery.scan",
        "com.hermes.trend-discovery.watchdog",
    )
    uid = subprocess.run(["id", "-u"], capture_output=True, text=True, timeout=5).stdout.strip()
    domain = f"gui/{uid}" if uid else "gui"
    return {label: _launchctl("print", f"{domain}/{label}") for label in labels}


def _uninstall_launchd() -> dict:
    labels = (
        "com.hermes.trend-discovery.scan",
        "com.hermes.trend-discovery.watchdog",
    )
    uid = subprocess.run(["id", "-u"], capture_output=True, text=True, timeout=5).stdout.strip()
    domain = f"gui/{uid}" if uid else "gui"
    results = {}
    for label in labels:
        plist = _launchd_dir() / f"{label}.plist"
        bootout = _launchctl("bootout", domain, str(plist))
        results[label] = {"plist": str(plist), "bootout": bootout}
    return results


def _tail(path: Path, lines: int) -> str:
    if not path.exists():
        return ""
    data = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(data[-lines:])


def _print_logs(lines: int) -> None:
    log_dir = get_hermes_home() / "trend-discovery" / "logs"
    for name in ("scan.out.log", "scan.err.log", "watchdog.out.log", "watchdog.err.log"):
        path = log_dir / name
        print(f"== {path} ==")
        text = _tail(path, lines)
        print(text if text else "(empty)")


def trend_discovery_command(args: argparse.Namespace) -> int:
    action = getattr(args, "trend_action", None)
    if not action:
        print("Usage: hermes trend-discovery {init|status|comply|scan|watchdog|health|digest|reliability-report|notify|install-cron}")
        return 2

    store = _store(args)
    store.init()

    if action == "init":
        print(json.dumps({"ok": True, "store_path": str(store.path)}, indent=2))
        return 0
    if action == "status":
        snapshot = store.status_snapshot()
        if getattr(args, "json", False):
            print(json.dumps(snapshot, indent=2, sort_keys=True))
        else:
            _print_status(snapshot)
        return 0
    if action == "ops":
        if getattr(args, "json", False):
            print(
                json.dumps(
                    {
                        "snapshot": store.status_snapshot(),
                        "launchd": _launchd_status(),
                        "modes": [
                            "scheduled-scan",
                            "scheduled-watchdog",
                            "manual-scan",
                            "manual-watchdog",
                            "compliance",
                            "source-admin",
                            "knowledge-writeback",
                        ],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            _print_ops(store)
        return 0
    if action == "comply":
        rows = store.compliance_rows()
        if getattr(args, "json", False):
            print(json.dumps(rows, indent=2, sort_keys=True))
        else:
            _print_comply(rows)
        return 0
    if action == "scan":
        result = TrendScanner(store).scan(query=args.query, limit=args.limit)
        if getattr(args, "write_review_queue", False):
            result["review_queue_path"] = str(write_review_queue(store))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "success" else 1
    if action == "watchdog":
        print(json.dumps(watchdog(store, notify_user=not args.no_notify), indent=2, sort_keys=True))
        return 0
    if action == "health":
        result = health_check(store, check_dashboard=getattr(args, "check_dashboard", False))
        if getattr(args, "json", False):
            print(health_json(store, check_dashboard=getattr(args, "check_dashboard", False)))
        else:
            print("ok" if result["ok"] else "failed")
            for name, check in result["checks"].items():
                print(f"{name} {100 if check.get('ok') else 0} {0 if check.get('ok') else 100}")
        return 0 if result["ok"] else 1
    if action == "digest":
        if getattr(args, "write_review_queue", False):
            print(str(write_review_queue(store)))
        else:
            print(build_digest(store), end="")
        return 0
    if action == "reliability-report":
        print(build_reliability_report(store), end="")
        return 0
    if action == "sources":
        try:
            return _handle_sources(store, args)
        except ValueError as exc:
            print(str(exc))
            return 2
    if action == "notify":
        print(json.dumps(notify(store, args.message, target=args.target or None), indent=2, sort_keys=True))
        return 0
    if action == "configure":
        changed = {}
        if args.notification_primary:
            store.set_config("notification.primary", args.notification_primary)
            changed["notification.primary"] = args.notification_primary
        if args.notification_fallback:
            store.set_config("notification.fallback", args.notification_fallback)
            changed["notification.fallback"] = args.notification_fallback
        if args.localhost_url:
            store.set_config("health.localhost_url", args.localhost_url)
            changed["health.localhost_url"] = args.localhost_url
        print(json.dumps({"ok": True, "changed": changed}, indent=2, sort_keys=True))
        return 0
    if action == "mark-all-complete":
        evidence = {"evidence": args.evidence}
        for row in store.compliance_rows():
            issue_id = row["issue_id"]
            if issue_id.startswith("TD-"):
                store.set_issue_complete(issue_id, evidence=evidence)
        _print_comply(store.compliance_rows())
        return 0
    if action == "install-cron":
        if shutil.which("python") is None and shutil.which("python3") is None:
            print("Python interpreter not found on PATH")
            return 1
        print(json.dumps(_install_cron_script(store, args.schedule, args.deliver), indent=2, sort_keys=True, default=str))
        return 0
    if action == "install-launchd":
        print(
            json.dumps(
                _install_launchd(
                    store,
                    scan_interval=args.scan_interval,
                    watchdog_interval=args.watchdog_interval,
                    query=args.query,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if action == "launchd-status":
        print(json.dumps(_launchd_status(), indent=2, sort_keys=True))
        return 0
    if action == "uninstall-launchd":
        print(json.dumps(_uninstall_launchd(), indent=2, sort_keys=True))
        return 0
    if action == "logs":
        _print_logs(args.lines)
        return 0

    print(f"Unknown trend-discovery action: {action}")
    return 2


def _from_slash(raw_args: str) -> str:
    parser = argparse.ArgumentParser(prog="/trend-discovery")
    register_cli(parser)
    parts = (raw_args or "status").split()
    try:
        args = parser.parse_args(parts)
    except SystemExit:
        return "Usage: /trend-discovery status|comply|watchdog"
    store = _store(args)
    store.init()
    if getattr(args, "trend_action", "") == "comply":
        return "\n".join(
            f"{row['issue_id']} {row['percent_complete']} {row['remaining_percent']}"
            for row in store.compliance_rows()
        )
    if getattr(args, "trend_action", "") == "watchdog":
        return json.dumps(watchdog(store), indent=2, sort_keys=True)
    snapshot = store.status_snapshot()
    return "\n".join(
        [f"Trend Discovery store: {snapshot['store_path']}"]
        + [f"{p['phase_id']} {p['percent_complete']} {100 - p['percent_complete']}" for p in snapshot["phases"]]
    )


trend_discovery_command.from_slash = _from_slash  # type: ignore[attr-defined]
