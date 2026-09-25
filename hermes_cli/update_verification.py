"""Post-update verification for ``hermes update`` — G3 guard rail.

Before an update reports success, three checks run against the pre-update
state captured by :func:`capture_pre_state`:

* config parity — every pre-update user-data root key still exists on disk;
* gateway liveness — ``gateway_state.json`` reports ``running`` and the
  webhook listener port accepts connections (bounded polling);
* MCP fingerprint parity — each profile's MCP server shape (name, enabled,
  command/url — never secret values) is unchanged.

An update never intentionally removes keys, so ANY missing pre-state key is a
verification failure by definition (spec §5.1). Failure rolls the update back:
``git reset --hard`` to the recorded pre-update SHA, config snapshots
restored, gateway restarted best-effort.

Run standalone after a desktop update:
``<python> -m hermes_cli.update_verification [--pre-state PATH] [--timeout S]``
"""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

LIVENESS_TIMEOUT_SECONDS = 120.0
LIVENESS_POLL_INTERVAL_SECONDS = 2.0
PRE_STATE_FILENAME = "pre-state-latest.json"


def _mcp_fingerprint(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """names + enabled + command/url. Secret values (env, args, headers) never
    enter the fingerprint."""
    fingerprint: Dict[str, Dict[str, Any]] = {}
    for name, server in (config.get("mcp_servers") or {}).items():
        if not isinstance(server, dict):
            continue
        fingerprint[str(name)] = {
            "enabled": bool(server.get("enabled", True)),
            "command": server.get("command"),
            "url": server.get("url"),
        }
    return fingerprint


def _read_raw_config(config_path: Path) -> Dict[str, Any]:
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _read_gateway_state() -> Dict[str, Any]:
    from hermes_constants import get_hermes_home
    try:
        state = json.loads(
            (get_hermes_home() / "gateway_state.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return state if isinstance(state, dict) else {}


def _listener_port(state: Dict[str, Any]) -> Optional[int]:
    webhook = (state.get("platforms") or {}).get("webhook") or {}
    base = webhook.get("listener_base") or ""
    try:
        return int(str(base).rsplit(":", 1)[-1])
    except (ValueError, AttributeError):
        return None


def _port_accepts(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex((host, port)) == 0


def _head_sha() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else ""
    except OSError:
        return ""


def capture_pre_state(homes: List[Tuple[str, Path]], *, backup: bool = True) -> Dict[str, Any]:
    """Snapshot every served profile's config BEFORE the update mutates anything.

    Per profile: home path, root-key set, MCP fingerprint, and (``backup``) a
    ``backup_config`` snapshot path used by rollback restore. Never raises on
    per-profile read problems — a profile that cannot be read records empty
    state, and config parity will flag the drift later."""
    from hermes_cli.config_backups import backup_config

    state: Dict[str, Any] = {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gateway_was_running": bool(_read_gateway_state()),
        "profiles": {},
    }
    for name, home in homes:
        config_path = Path(home) / "config.yaml"
        raw = _read_raw_config(config_path)
        snapshot_path: Optional[str] = None
        if backup and config_path.exists():
            snapshot = backup_config(config_path, "pre-update-verification")
            snapshot_path = str(snapshot) if snapshot else None
        state["profiles"][str(name)] = {
            "home": str(home),
            "root_keys": sorted(raw),
            "mcp_fingerprint": _mcp_fingerprint(raw),
            "config_snapshot": snapshot_path,
        }
    return state


def _pre_state_path(path: Optional[Path] = None) -> Path:
    from hermes_constants import get_hermes_home
    return path or (get_hermes_home() / "logs" / "update_receipts" / PRE_STATE_FILENAME)


def save_pre_state(state: Dict[str, Any]) -> Path:
    path = _pre_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_pre_state(path: Optional[Path] = None) -> Dict[str, Any]:
    return json.loads(_pre_state_path(path).read_text(encoding="utf-8"))


def check_config_parity(pre_state: Dict[str, Any]) -> Tuple[bool, str]:
    missing_all: List[str] = []
    for name, snapshot in pre_state.get("profiles", {}).items():
        raw = _read_raw_config(Path(snapshot["home"]) / "config.yaml")
        missing = sorted(set(snapshot["root_keys"]) - set(raw) - {"_config_version"})
        if missing:
            missing_all.append(f"{name}: missing {missing}")
    if missing_all:
        return False, "; ".join(missing_all)
    return True, f"{len(pre_state.get('profiles', {}))} profile(s) intact"


def check_mcp_fingerprint_parity(pre_state: Dict[str, Any]) -> Tuple[bool, str]:
    diffs: List[str] = []
    for name, snapshot in pre_state.get("profiles", {}).items():
        raw = _read_raw_config(Path(snapshot["home"]) / "config.yaml")
        before = snapshot.get("mcp_fingerprint") or {}
        after = _mcp_fingerprint(raw)
        if before == after:
            continue
        removed = sorted(set(before) - set(after))
        added = sorted(set(after) - set(before))
        changed = sorted(k for k in set(before) & set(after) if before[k] != after[k])
        diffs.append(f"{name}: removed={removed} added={added} changed={changed}")
    if diffs:
        return False, "; ".join(diffs)
    return True, "MCP fingerprints match pre-state"


def check_gateway_liveness(timeout: float = LIVENESS_TIMEOUT_SECONDS) -> Tuple[bool, str]:
    """Bounded polling of gateway_state.json + listener port. Never waits
    forever: the deadline is absolute (spec §5.3)."""
    deadline = time.monotonic() + timeout
    last_state: Dict[str, Any] = {}
    while True:
        last_state = _read_gateway_state()
        if last_state.get("gateway_state") == "running":
            port = _listener_port(last_state)
            if port is None:
                return True, "gateway_state=running (no listener port recorded)"
            if _port_accepts(port):
                return True, f"gateway_state=running, port {port} accepting"
        if time.monotonic() >= deadline:
            detail = f"gateway not live within {timeout:.0f}s"
            if last_state:
                detail += f"; last gateway_state={last_state.get('gateway_state')!r}"
            return False, detail
        time.sleep(LIVENESS_POLL_INTERVAL_SECONDS)


def run_verification(pre_state: Dict[str, Any], *,
                     timeout: float = LIVENESS_TIMEOUT_SECONDS) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    ok, detail = check_config_parity(pre_state)
    results.append({"name": "config_parity", "passed": ok, "detail": detail})
    ok, detail = check_mcp_fingerprint_parity(pre_state)
    results.append({"name": "mcp_fingerprint_parity", "passed": ok, "detail": detail})
    if pre_state.get("gateway_was_running"):
        ok, detail = check_gateway_liveness(timeout)
        results.append({"name": "gateway_liveness", "passed": ok, "detail": detail})
    return results


def rollback_update(pre_state: Dict[str, Any], *, checkout: Path, pre_sha: str,
                    failed_check: str) -> Dict[str, Any]:
    """git reset --hard to the pre-update SHA, restore config snapshots from
    their backup_config copies, restart the gateway best-effort."""
    reset = subprocess.run(["git", "reset", "--hard", pre_sha], cwd=str(checkout),
                           capture_output=True, text=True, check=False)
    restored: List[str] = []
    restore_errors: List[str] = []
    for name, snapshot in pre_state.get("profiles", {}).items():
        snapshot_file = snapshot.get("config_snapshot")
        if snapshot_file and Path(snapshot_file).exists():
            try:
                (Path(snapshot["home"]) / "config.yaml").write_bytes(
                    Path(snapshot_file).read_bytes())
                restored.append(str(name))
            except OSError as exc:
                restore_errors.append(f"{name}: {exc}")
    gateway_restart = ""
    if restored or reset.returncode == 0:
        try:
            subprocess.run([sys.executable, "-m", "hermes_cli.main", "gateway", "start", "--all"],
                           capture_output=True, timeout=60, check=False)
            gateway_restart = "attempted"
        except (OSError, subprocess.TimeoutExpired):
            gateway_restart = "failed"
    return {"reset_ok": reset.returncode == 0,
            "reset_detail": (reset.stderr or reset.stdout).strip(),
            "restored_profiles": restored, "restore_errors": restore_errors,
            "gateway_restart": gateway_restart, "failed_check": failed_check}
