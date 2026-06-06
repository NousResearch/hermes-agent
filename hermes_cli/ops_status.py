from __future__ import annotations

from dataclasses import dataclass
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional
from typing import Protocol, Tuple


EVIDENCE_ROOT = "/Users/indie/Projects/Hermes"
DEFAULT_PROOF_DIR = Path(EVIDENCE_ROOT) / "ops" / "status-proofs"
_SECRET_PATTERNS = (
    re.compile(r"xox[baprs]-[A-Za-z0-9-]+"),
    re.compile(r"sk-[A-Za-z0-9_-]+"),
    re.compile(r"AIza[0-9A-Za-z_-]+"),
    re.compile(r"/Users/indie/\.hermes/auth\.json"),
)
CommandRunner = Callable[[List[str]], str]


class OpsStatusChecks(Protocol):
    def hermes_version(self) -> str:
        ...

    def gateway_state(self) -> str:
        ...

    def slack_state(self) -> str:
        ...

    def xai_oauth_state(self) -> str:
        ...

    def cron_summary(self) -> str:
        ...

    def xurl_state(self) -> str:
        ...

    def desktop_state(self) -> str:
        ...


@dataclass(frozen=True)
class OpsStatusSnapshot:
    health: str
    hermes_version: str
    gateway: str
    slack: str
    xai_oauth: str
    cron_summary: str
    xurl: str
    desktop: str
    evidence_root: str
    blockers: Tuple[str, ...] = ()


def _health_for(
    gateway: str,
    slack: str,
    xai_oauth: str,
    cron_summary: str,
    xurl: str,
    desktop: str,
    blockers: Tuple[str, ...],
) -> str:
    if "unknown" in {gateway, slack, xai_oauth, cron_summary, xurl, desktop}:
        return "unknown"
    if gateway != "running" or slack != "connected":
        return "down"
    if blockers:
        return "partial"
    return "healthy"


def collect_ops_status(checks: OpsStatusChecks) -> OpsStatusSnapshot:
    gateway = checks.gateway_state()
    slack = checks.slack_state()
    xai_oauth = checks.xai_oauth_state()
    cron_summary = checks.cron_summary()
    xurl = checks.xurl_state()
    desktop = checks.desktop_state()
    blockers = ()
    if xurl.startswith("blocked"):
        blockers = ("native bookmarks/List reads need xurl OAuth",)
    return OpsStatusSnapshot(
        health=_health_for(gateway, slack, xai_oauth, cron_summary, xurl, desktop, blockers),
        hermes_version=checks.hermes_version(),
        gateway=gateway,
        slack=slack,
        xai_oauth=xai_oauth,
        cron_summary=cron_summary,
        xurl=xurl,
        desktop=desktop,
        evidence_root=EVIDENCE_ROOT,
        blockers=blockers,
    )


def format_ops_status_text(snapshot: OpsStatusSnapshot) -> str:
    hermes_version = _public_text(snapshot.hermes_version)
    gateway = _public_text(snapshot.gateway)
    slack = _public_text(snapshot.slack)
    xai_oauth = _public_text(snapshot.xai_oauth)
    cron_summary = _public_text(snapshot.cron_summary)
    xurl = _public_text(snapshot.xurl)
    desktop = _public_text(snapshot.desktop)
    blockers = tuple(_public_text(blocker) for blocker in snapshot.blockers)
    lines = [
        f"Hermes ops: {snapshot.health}",
        (
            f"Hermes {hermes_version} | gateway {gateway} | "
            f"Slack {slack} | xAI {xai_oauth}"
        ),
        (
            f"Cron: {cron_summary} | xurl: {xurl} | "
            f"Desktop: {desktop}"
        ),
    ]
    if blockers:
        lines.append(f"Blocked: {'; '.join(blockers)}")
    lines.append(f"Evidence: {snapshot.evidence_root}")
    return "\n".join(lines)


def _public_text(value: str) -> str:
    redacted = value
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("[redacted]", redacted)
    return redacted


def snapshot_to_public_dict(snapshot: OpsStatusSnapshot) -> dict:
    return {
        "health": snapshot.health,
        "hermes_version": _public_text(snapshot.hermes_version),
        "gateway": _public_text(snapshot.gateway),
        "slack": _public_text(snapshot.slack),
        "xai_oauth": _public_text(snapshot.xai_oauth),
        "cron_summary": _public_text(snapshot.cron_summary),
        "xurl": _public_text(snapshot.xurl),
        "desktop": _public_text(snapshot.desktop),
        "evidence_root": snapshot.evidence_root,
        "blockers": [_public_text(blocker) for blocker in snapshot.blockers],
    }


def ops_command(args) -> int:
    if getattr(args, "ops_command", None) != "status":
        print("Run: hermes ops status")
        return 2
    snapshot = collect_ops_status(DefaultOpsStatusChecks())
    if getattr(args, "proof", False):
        proof_path = write_ops_status_proof(snapshot)
        print(str(proof_path))
    elif getattr(args, "json", False):
        print(json.dumps(snapshot_to_public_dict(snapshot), indent=2, sort_keys=True))
    else:
        print(format_ops_status_text(snapshot))
    return 0


def gateway_ops_status_reply(args_text: str) -> str:
    normalized = " ".join(args_text.strip().lower().split())
    if normalized not in {"status", "status proof"}:
        return "Usage: /ops status"
    snapshot = collect_ops_status(DefaultOpsStatusChecks())
    if normalized == "status proof":
        proof_path = write_ops_status_proof(snapshot)
        return f"{format_ops_status_text(snapshot)}\nProof: {proof_path}"
    return format_ops_status_text(snapshot)


def write_ops_status_proof(
    snapshot: OpsStatusSnapshot,
    output_dir: Optional[Path] = None,
    timestamp: Optional[str] = None,
) -> Path:
    target_dir = output_dir or DEFAULT_PROOF_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    proof_path = target_dir / f"{stamp}-ops-status.md"
    proof_path.write_text(_proof_markdown(snapshot, stamp), encoding="utf-8")
    return proof_path


def _proof_markdown(snapshot: OpsStatusSnapshot, timestamp: str) -> str:
    public = snapshot_to_public_dict(snapshot)
    lines = [
        "# Hermes Ops Status Proof",
        "",
        f"Timestamp: {timestamp}",
        "",
        "## Compact Status",
        "",
        "```text",
        format_ops_status_text(snapshot),
        "```",
        "",
        "## Parsed Snapshot",
        "",
        "```json",
        json.dumps(public, indent=2, sort_keys=True),
        "```",
        "",
    ]
    return "\n".join(lines)


class DefaultOpsStatusChecks:
    def __init__(
        self,
        run: Optional[CommandRunner] = None,
        slack_log_tail: Optional[str] = None,
    ) -> None:
        self._run = run or _run_command
        self._slack_log_tail = slack_log_tail

    def hermes_version(self) -> str:
        out = self._safe_run(["hermes", "--version"])
        match = re.search(r"Hermes Agent (v[0-9][^\s]+)", out)
        return match.group(1) if match else "unknown"

    def gateway_state(self) -> str:
        out = self._safe_run(["hermes", "gateway", "status"])
        if "not loaded" in out or "not running" in out or "not loaded" in out.lower():
            return "stopped"
        if "Gateway service is loaded" in out or "PID" in out:
            return "running"
        return "unknown"

    def slack_state(self) -> str:
        tail = self._slack_log_tail
        if tail is None:
            log_path = Path.home() / ".hermes" / "logs" / "gateway.log"
            try:
                tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-80:])
            except OSError:
                tail = ""
        if "slack connected" in tail.lower() or "socket mode connected" in tail.lower():
            return "connected"
        if "slack disconnected" in tail.lower() or "slack failed" in tail.lower():
            return "disconnected"
        return "unknown"

    def xai_oauth_state(self) -> str:
        out = self._safe_run(["hermes", "auth", "status", "xai-oauth"])
        lower = out.lower()
        if "not logged in" in lower:
            return "not logged in"
        if "logged in" in lower:
            return "logged in"
        return "unknown"

    def cron_summary(self) -> str:
        out = self._safe_run(["hermes", "cron", "status"])
        if "No active jobs" in out:
            return "0 jobs -> #crons"
        match = re.search(r"(\d+)\s+active job", out)
        if match:
            return f"{match.group(1)} jobs -> #crons"
        return "unknown"

    def xurl_state(self) -> str:
        out = self._safe_run(["xurl", "auth", "status"])
        lower = out.lower()
        if "no apps registered" in lower:
            return "blocked, no app"
        if "not authenticated" in lower or "unauthenticated" in lower:
            return "blocked, not authenticated"
        if "default" in lower or "authenticated" in lower:
            return "ready"
        return "unknown"

    def desktop_state(self) -> str:
        out = self._safe_run(["hermes", "dashboard", "--status"])
        if "process(es) running" in out:
            return "running"
        if "No hermes dashboard processes running" in out:
            return "stopped"
        return "unknown"

    def _safe_run(self, command: List[str]) -> str:
        try:
            return self._run(command)
        except Exception:
            return ""


def _run_command(command: List[str]) -> str:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return "\n".join(part for part in (result.stdout, result.stderr) if part)


def rewrite_ops_message(text: str) -> Optional[str]:
    normalized = " ".join(text.strip().lower().split())
    if normalized == "ops status":
        return "/ops status"
    return None
