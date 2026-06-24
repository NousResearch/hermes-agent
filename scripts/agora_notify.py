#!/usr/bin/env python3
"""Post lightweight status/messages to the local Ágora dashboard plugin.

This helper is intentionally local and cheap: workers can call it to update
human telemetry without waking a tech-lead LLM or polling on a timer.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
try:
    from hermes_constants import get_default_hermes_root, get_hermes_home
except ImportError:
    def get_default_hermes_root() -> Path:  # type: ignore[misc]
        val = (os.environ.get("HERMES_HOME") or "").strip()
        if not val:
            return Path.home() / ".hermes"
        env_path = Path(val)
        if env_path.parent.name == "profiles":
            return env_path.parent.parent
        return env_path

    def get_hermes_home() -> Path:  # type: ignore[misc]
        val = (os.environ.get("HERMES_HOME") or "").strip()
        return Path(val) if val else Path.home() / ".hermes"

DEFAULT_BASE_URL = os.environ.get("AGORA_DASHBOARD_URL", "http://127.0.0.1:9119")
TOKEN_ENV = "HERMES_DASHBOARD_TOKEN"


def _token_file() -> Path:
    """Return the shared dashboard token file path.

    Ágora helpers may run from non-default Hermes profiles, while the local
    dashboard commonly runs as a shared process rooted at the default Hermes
    home. Reading ``<active-profile>/.dashboard-token`` makes helpers pick up a
    missing or stale token and 401 when posting back into Ágora. Use the shared
    default Hermes root so every profile looks at the live dashboard token.
    """
    return get_default_hermes_root() / ".dashboard-token"


def _read_token(args_token: str | None = None) -> str | None:
    if args_token:
        return args_token
    if os.environ.get(TOKEN_ENV):
        return os.environ[TOKEN_ENV]
    try:
        token_path = _token_file()
        if token_path.exists():
            val = token_path.read_text(encoding="utf-8").strip()
            return val or None
    except Exception:
        return None
    return None


def _request(method: str, path: str, payload: dict[str, Any] | None, *, base_url: str, token: str | None) -> Any:
    url = base_url.rstrip("/") + path
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
        headers["X-Hermes-Session-Token"] = token
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=10) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else {"ok": True}


def _resolve_profile_pid(profile: str) -> int | None:
    """Return the PID of a visible Hermes process for ``profile``, if any.

    Mirrors the discovery in ``plugins.agora.dashboard.plugin_api`` so the
    helper and the backend agree on which process represents a profile. Falls
    back to ``None`` when no tmux session or Hermes process is found.
    """
    if shutil.which("tmux"):
        try:
            proc = subprocess.run(
                ["tmux", "list-panes", "-t", profile, "-F", "#{pane_active} #{pane_pid}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=3,
                check=False,
            )
            if proc.returncode == 0:
                for line in proc.stdout.strip().splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 2 and parts[0] == "1":
                        try:
                            pid = int(parts[1])
                            if pid > 0:
                                return pid
                        except ValueError:
                            continue
        except Exception:
            pass

    try:
        import psutil

        candidates: list[tuple[int, str]] = []
        for p in psutil.process_iter(["pid", "cmdline"]):
            cmdline = p.info["cmdline"] or []
            cmd = " ".join(cmdline)
            if "hermes" in cmd and f"-p {profile}" in cmd:
                candidates.append((p.info["pid"], cmd))
        for pid, cmd in candidates:
            if "chat" in cmd:
                return pid
        if candidates:
            return candidates[0][0]
    except Exception:
        pass

    if shutil.which("pgrep"):
        try:
            proc = subprocess.run(
                ["pgrep", "-a", "-f", f"hermes -p {profile}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=3,
                check=False,
            )
            if proc.returncode == 0:
                lines = [l.strip() for l in proc.stdout.strip().splitlines() if l.strip()]
                for line in lines:
                    if "chat" in line:
                        try:
                            return int(line.split(None, 1)[0])
                        except ValueError:
                            continue
                if lines:
                    try:
                        return int(lines[0].split(None, 1)[0])
                    except ValueError:
                        pass
        except Exception:
            pass

    return None


def _hermes_executable() -> str | None:
    """Return the ``hermes`` CLI binary path, or Python module fallback."""
    exe = shutil.which("hermes")
    if exe:
        return exe
    return sys.executable


def _run_hermes_send(target: str, message: str) -> dict[str, Any]:
    """Invoke ``hermes send --to <target> <message>`` as a subprocess."""
    exe = _hermes_executable()
    if exe == sys.executable:
        cmd = [exe, "-m", "hermes_cli.main", "send", "--to", target, "--quiet", message]
    else:
        cmd = [exe, "send", "--to", target, "--quiet", message]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(_PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if proc.returncode == 0:
            return {"success": True, "stdout": proc.stdout.strip() or None}
        return {
            "success": False,
            "error": (proc.stderr.strip() or f"hermes send exited {proc.returncode}"),
        }
    except Exception as exc:
        return {"success": False, "error": f"failed to run hermes send: {exc}"}


def _wake_tech_lead(
    target: str | None,
    message: str | None,
    state: str,
    wake_on: set[str],
    enabled: bool,
) -> dict[str, Any]:
    """Prepare (and optionally deliver) a wake-up ping to the tech lead."""
    result: dict[str, Any] = {"target": target, "triggered": False}
    if not target:
        return result
    if state not in wake_on:
        result["note"] = f"state '{state}' not in wake-on list"
        return result
    result["triggered"] = True
    result["message"] = message or "Ágora wake-up ping"
    if not enabled:
        result["action"] = "would-send"
        result["note"] = "wake disabled; set AGORA_WAKE_ENABLED=1 or --wake-enabled to deliver"
        return result
    result["action"] = "send"
    delivery = _run_hermes_send(target, result["message"])
    result["delivery"] = delivery
    if not delivery.get("success"):
        result["warning"] = delivery.get("error")
    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Post status/message to Ágora dashboard plugin")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--token", default=None, help="Dashboard session token; prefer HERMES_DASHBOARD_TOKEN env")
    p.add_argument("--profile", required=True, help="Agent/profile name")
    p.add_argument("--state", default="working", choices=["idle", "deliberating", "working", "reviewing", "waiting-human", "blocked", "error"])
    p.add_argument("--step", default=None, help="Current step/status headline")
    p.add_argument("--status", default=None, help="Human-readable status text")
    p.add_argument("--task-id", default=None)
    p.add_argument("--run-id", type=int, default=None)
    p.add_argument("--pid", type=int, default=None)
    p.add_argument("--channel", default="praca", help="Ágora channel slug for the message")
    p.add_argument("--message", default=None, help="Optional message to post")
    p.add_argument("--author-type", default="agent", choices=["agent", "human", "system"])
    p.add_argument("--metadata", default=None, help="JSON metadata object")
    p.add_argument(
        "--wake-target",
        default=os.environ.get("AGORA_WAKE_TARGET"),
        help="Messaging target for wake-up ping (e.g. telegram, ntfy:topic). From AGORA_WAKE_TARGET env.",
    )
    p.add_argument(
        "--wake-message",
        default=None,
        help="Override wake-up message; defaults to --message or a short status summary.",
    )
    p.add_argument(
        "--wake-on",
        default=os.environ.get("AGORA_WAKE_ON", "blocked,error"),
        help="Comma-separated states that trigger wake-up (default: blocked,error).",
    )
    p.add_argument(
        "--wake-enabled",
        action="store_true",
        default=os.environ.get("AGORA_WAKE_ENABLED", "").lower() in {"1", "true", "yes"},
        help="Actually deliver wake-up pings via hermes send (default: prepare-only).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    token = _read_token(args.token)
    metadata = None
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError as exc:
            print(f"invalid --metadata JSON: {exc}", file=sys.stderr)
            return 2

    pid = args.pid or _resolve_profile_pid(args.profile) or os.getpid()
    status_payload = {
        "state": args.state,
        "current_task_id": args.task_id,
        "current_step": args.step,
        "status_text": args.status,
        "pid": pid,
        "run_id": args.run_id,
        "metadata": metadata,
    }
    result: dict[str, Any] = {}
    try:
        result["agent"] = _request(
            "POST",
            f"/api/plugins/agora/agents/status/{args.profile}",
            status_payload,
            base_url=args.base_url,
            token=token,
        )

        if args.message:
            msg_payload = {
                "body": args.message,
                "author_type": args.author_type,
                "author_profile": args.profile,
                "linked_task_id": args.task_id,
            }
            result["message"] = _request(
                "POST",
                f"/api/plugins/agora/channels/{args.channel}/messages",
                msg_payload,
                base_url=args.base_url,
                token=token,
            )
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {args.base_url.rstrip('/')}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"cannot reach {args.base_url}: {exc.reason}") from exc

    wake_on_set = {s.strip().lower() for s in (args.wake_on or "").split(",") if s.strip()}
    result["wake"] = _wake_tech_lead(
        target=args.wake_target,
        message=(args.wake_message or args.message or args.status or f"{args.profile}: {args.state}"),
        state=args.state,
        wake_on=wake_on_set,
        enabled=args.wake_enabled,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
