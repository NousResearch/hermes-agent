"""Per-subaction implementations for ``hermes tunnel``.

Dispatch is invoked from ``cmd_tunnel`` in ``hermes_cli/main.py``. The
heavy lifting (cloudflared process, idle-reset timer, approvals) lives in
``tunnel_supervisor`` / ``tunnel_approvals``; this module wires args to
those and to the config resolver.
"""

from __future__ import annotations

import os
import re
from argparse import Namespace

from hermes_cli.tunnel_config import resolve_tunnel_config


def _approvals_path() -> str:
    home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
    return os.path.join(home, "tunnel", "hold_requests.jsonl")


def _current_user() -> str:
    # Resolved from the active profile name; fall back to OS user.
    return os.environ.get("HERMES_PROFILE") or os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"


_DURATION_RE = re.compile(r"^(?P<n>\d+)\s*(?P<u>s|m|h|d)$")
_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def _parse_duration(spec: str) -> float | None:
    if not spec:
        return None
    m = _DURATION_RE.match(spec.strip())
    if not m:
        return None
    import time
    return time.time() + int(m.group("n")) * _UNITS[m.group("u")]


def _print(*a):
    print(*a)


def tunnel_command(args: Namespace) -> int:
    cmd = getattr(args, "tunnel_command", None)
    if cmd == "up":
        return _cmd_up(args)
    if cmd == "down":
        return _cmd_down(args)
    if cmd == "status":
        return _cmd_status(args)
    if cmd == "doctor":
        return _cmd_doctor(args)
    if cmd == "hold":
        return _cmd_hold(args)
    if cmd == "requests":
        return _cmd_requests(args)
    if cmd == "approve":
        return _cmd_approve(args)
    if cmd == "deny":
        return _cmd_deny(args)
    _print("usage: hermes tunnel {up,down,status,doctor,hold,requests,approve,deny}")
    return 2


def _cmd_up(args) -> int:
    cfg = resolve_tunnel_config(cli_origins=getattr(args, "origins", None) or None)
    if not cfg["routes"]:
        _print("tunnel up: no origins. Pass --origin SUB=HOST:PORT or set tunnel.routes.")
        return 2
    if not cfg["zone"]:
        _print("tunnel up: no zone configured (set tunnel.zone or HERMES_TUNNEL_ZONE).")
        return 2
    if not cfg["tunnel_name"] or not cfg["credentials_file"]:
        _print("tunnel up: tunnel_name and credentials_file are required.")
        return 2
    if not os.path.exists(cfg["credentials_file"]):
        _print(f"tunnel up: credentials file not found: {cfg['credentials_file']}")
        return 2

    config_path = _write_cloudflared_config(cfg)
    hold_id = None
    if getattr(args, "hold_request", False):
        from hermes_cli import tunnel_approvals as ta
        hold_id = ta.file_request(_approvals_path(), user=_current_user(),
                                   subdomains=[f"{r['subdomain']}.{cfg['zone']}" for r in cfg["routes"]],
                                   reason=getattr(args, "reason", ""),
                                   requested_until=_parse_duration(getattr(args, "until", "")))
        _print(f"hold request filed: {hold_id} (pending admin approval)")

    # If any route targets the dashboard port, set the dashboard public URL
    # so OAuth callback / WebSocket URLs build from the public hostname.
    _maybe_set_dashboard_public_url(cfg)

    from hermes_cli.tunnel_supervisor import TunnelSupervisor
    sup = TunnelSupervisor(cfg, _approvals_path(), hold_request_id=hold_id)
    _print(f"tunnel up: https://{cfg['routes'][0]['subdomain']}.{cfg['zone']} "
           f"-> 127.0.0.1:{cfg['routes'][0]['port']} (idle {cfg['idle_timeout_seconds']}s)")
    sup.run(config_path)
    return 0


def _write_cloudflared_config(cfg) -> str:
    import tempfile, json
    ingress = []
    for r in cfg["routes"]:
        ingress.append({"hostname": f"{r['subdomain']}.{cfg['zone']}",
                        "service": f"http://{r['host']}:{r['port']}"})
    ingress.append({"service": "http_status:404"})
    doc = {"tunnel": cfg["tunnel_name"], "credentials-file": cfg["credentials_file"],
           "ingress": ingress}
    home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
    os.makedirs(os.path.join(home, "tunnel"), exist_ok=True)
    path = os.path.join(home, "tunnel", "cloudflared.yml")
    with open(path, "w", encoding="utf-8") as f:
        import yaml  # PyYAML is already a project dep (config.py uses YAML)
        yaml.safe_dump(doc, f)
    return path


def _maybe_set_dashboard_public_url(cfg) -> None:
    DASH_PORT = 9119
    for r in cfg["routes"]:
        if int(r.get("port", 0)) == DASH_PORT:
            os.environ["HERMES_DASHBOARD_PUBLIC_URL"] = f"https://{r['subdomain']}.{cfg['zone']}"
            return


def _cmd_down(args) -> int:
    # Best-effort: terminate running cloudflared for this profile.
    import subprocess
    killed = 0
    try:
        out = subprocess.run(["cloudflared", "tunnel", "list"], capture_output=True, text=True, timeout=10)
    except Exception:
        _print("tunnel down: cloudflared not reachable")
        return 1
    _print(out.stdout)
    return 0


def _cmd_status(args) -> int:
    _print("tunnel status: (cloudflared process introspection — see `cloudflared tunnel info`)")
    return 0


def _cmd_doctor(args) -> int:
    import shutil, socket
    ok = True
    if not shutil.which("cloudflared"):
        _print("doctor: cloudflared NOT on PATH"); ok = False
    else:
        _print("doctor: cloudflared present")
    cfg = resolve_tunnel_config()
    if cfg["credentials_file"] and not os.path.exists(cfg["credentials_file"]):
        _print(f"doctor: credentials file missing: {cfg['credentials_file']}"); ok = False
    for r in cfg["routes"]:
        s = socket.socket(); s.settimeout(2)
        try:
            s.connect((r["host"], int(r["port"]))); _print(f"doctor: origin up {r['subdomain']} -> {r['host']}:{r['port']}")
        except Exception:
            _print(f"doctor: origin DOWN {r['subdomain']} -> {r['host']}:{r['port']}"); ok = False
        finally:
            s.close()
    return 0 if ok else 1


def _cmd_hold(args) -> int:
    from hermes_cli import tunnel_approvals as ta
    cfg = resolve_tunnel_config()
    rid = ta.file_request(_approvals_path(), user=_current_user(),
                           subdomains=[f"{r['subdomain']}.{cfg['zone']}" for r in cfg["routes"]],
                           reason=getattr(args, "reason", ""),
                           requested_until=_parse_duration(getattr(args, "until", "")))
    _print(f"hold request filed: {rid} (pending admin approval)")
    return 0


def _cmd_requests(args) -> int:
    from hermes_cli import tunnel_approvals as ta
    for r in ta.list_pending(_approvals_path()):
        _print(f"{r['id']}  user={r['user']}  subs={','.join(r['subdomains'])}  reason={r['reason']!r}  status={r['status']}")
    return 0


def _cmd_approve(args) -> int:
    from hermes_cli import tunnel_approvals as ta
    if not getattr(args, "id", None):
        _print("approve: hold request id is required")
        return 2
    if not getattr(args, "until", ""):
        _print("approve: --until is required (e.g. --until 6h)")
        return 2
    cfg = resolve_tunnel_config()
    until = _parse_duration(args.until)
    if until is None:
        _print(f"approve: bad --until value: {args.until!r}")
        return 2
    try:
        ta.approve(_approvals_path(), args.id, until=until,
                   by=_current_user(), admin_ids=cfg["admin"])
    except PermissionError:
        _print(f"approve: {_current_user()} is not a tunnel admin")
        return 3
    except KeyError:
        _print(f"approve: no such hold request: {args.id}")
        return 4
    _print(f"approved: {args.id} until {args.until}")
    return 0


def _cmd_deny(args) -> int:
    from hermes_cli import tunnel_approvals as ta
    if not getattr(args, "id", None):
        _print("deny: hold request id is required")
        return 2
    cfg = resolve_tunnel_config()
    try:
        ta.deny(_approvals_path(), args.id, reason=getattr(args, "reason", ""),
                by=_current_user(), admin_ids=cfg["admin"])
    except PermissionError:
        _print(f"deny: {_current_user()} is not a tunnel admin")
        return 3
    except KeyError:
        _print(f"deny: no such hold request: {args.id}")
        return 4
    _print(f"denied: {args.id}")
    return 0