"""macOS launchd watchdog for the Hermes gateway.

This intentionally runs outside the gateway process.  It recovers both the
rare failure where the launchd job is removed from the gui/<uid> domain and the
silent failure where the job is still loaded but no gateway process is live.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Sequence


DEFAULT_LABEL = "ai.hermes.gateway"
DEFAULT_PLIST = "ai.hermes.gateway.plist"
LOG_NAME = "gateway-domain-watchdog.log"
ALERT_COOLDOWN_SECONDS = 30 * 60


def _account_home() -> Path:
    try:
        import pwd
        return Path(pwd.getpwuid(os.getuid()).pw_dir)
    except Exception:
        return Path.home()


def _hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME") or _account_home() / ".hermes").expanduser()


def _domain() -> str:
    return f"gui/{os.getuid()}"


def _plist_path() -> Path:
    return _account_home() / "Library" / "LaunchAgents" / DEFAULT_PLIST


def _log(message: str) -> None:
    home = _hermes_home()
    log_dir = home / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with (log_dir / LOG_NAME).open("a", encoding="utf-8") as f:
        f.write(f"{ts} {message}\n")


def _load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def _telegram_chat_id(home: Path, env: dict[str, str]) -> str | None:
    for key in ("TELEGRAM_ALERT_CHAT_ID", "TELEGRAM_ALLOWED_CHATS", "TELEGRAM_CHAT_ID"):
        raw = env.get(key) or os.environ.get(key)
        if raw:
            return raw.split(",", 1)[0].strip()

    cfg = home / "config.yaml"
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
        allowed = (data.get("telegram") or {}).get("allowed_chats")
        if isinstance(allowed, (list, tuple)) and allowed:
            return str(allowed[0])
        if isinstance(allowed, str) and allowed.strip():
            return allowed.split(",", 1)[0].strip()
    except Exception:
        return None
    return None


def _alert_state_path(home: Path) -> Path:
    return home / "state" / "gateway-domain-watchdog-alert.json"


def _alert_allowed(home: Path, now: float) -> bool:
    path = _alert_state_path(home)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        last = float(data.get("last_alert_at") or 0)
    except Exception:
        last = 0.0
    if now - last < ALERT_COOLDOWN_SECONDS:
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"last_alert_at": now}), encoding="utf-8")
    except OSError:
        pass
    return True


def _send_telegram_alert(message: str) -> bool:
    home = _hermes_home()
    env = {**_load_env(home / ".env"), **_load_env(Path.home() / ".hermes" / ".env")}
    token = os.environ.get("TELEGRAM_BOT_TOKEN") or env.get("TELEGRAM_BOT_TOKEN")
    chat_id = _telegram_chat_id(home, env)
    if not token or not chat_id:
        _log("alert skipped: missing Telegram token or chat id")
        return False
    if not _alert_allowed(home, time.time()):
        _log("alert suppressed: cooldown active")
        return False

    body = urllib.parse.urlencode({"chat_id": chat_id, "text": message}).encode("utf-8")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        with urllib.request.urlopen(url, data=body, timeout=10) as resp:  # noqa: S310 - configured Telegram API endpoint
            ok = 200 <= getattr(resp, "status", 0) < 300
            _log(f"alert sent: {ok}")
            return ok
    except Exception as exc:
        # Never log urllib exception text: it can include the bot-token URL.
        _log(f"alert failed: {type(exc).__name__}")
        return False


def _run(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=30)


def is_gateway_loaded(run: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] = _run) -> bool:
    return read_gateway_launchd_status(run).loaded


def read_gateway_launchd_status(
    run: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] = _run,
) -> "LaunchdGatewayStatus":
    result = run(["launchctl", "print", f"{_domain()}/{DEFAULT_LABEL}"])
    if result.returncode != 0:
        return LaunchdGatewayStatus(loaded=False, state=None, pid=None)
    return LaunchdGatewayStatus.from_launchctl_print(result.stdout or "")


class LaunchdGatewayStatus:
    """Small parsed view of `launchctl print` for watchdog decisions."""

    def __init__(self, *, loaded: bool, state: str | None, pid: int | None) -> None:
        self.loaded = loaded
        self.state = state
        self.pid = pid

    @classmethod
    def from_launchctl_print(cls, output: str) -> "LaunchdGatewayStatus":
        state_match = re.search(r"^\s*state\s*=\s*(.+?)\s*$", output, re.MULTILINE)
        pid_match = re.search(r"^\s*pid\s*=\s*(\d+)\s*$", output, re.MULTILINE)
        pid = int(pid_match.group(1)) if pid_match else None
        state = state_match.group(1).strip() if state_match else None
        return cls(loaded=True, state=state, pid=pid)

    @property
    def live(self) -> bool:
        if not self.loaded:
            return False
        if self.state is None and self.pid is None:
            # `launchctl print` succeeded but the text shape was not recognized.
            # Treat that as healthy so macOS output-format drift cannot cause a
            # destructive bootout/bootstrap loop on an otherwise loaded gateway.
            return True
        return self.state == "running" and self.pid is not None

    @property
    def summary(self) -> str:
        return f"loaded={self.loaded} state={self.state or 'unknown'} pid={self.pid or 'none'}"


def recover_gateway(run: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] = _run) -> bool:
    plist = _plist_path()
    if not plist.exists():
        _log(f"recover failed: missing plist {plist}")
        return False

    domain = _domain()
    run(["launchctl", "bootout", f"{domain}/{DEFAULT_LABEL}"])
    run(["launchctl", "bootstrap", domain, str(plist)])
    run(["launchctl", "kickstart", f"{domain}/{DEFAULT_LABEL}"])
    for attempt in range(5):
        if attempt:
            time.sleep(0.2)
        if read_gateway_launchd_status(run).live:
            return True
    return False


def check_once(*, alert: bool = True, run: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] = _run) -> int:
    """Return 0 when healthy/recovered, 2 when recovery failed."""
    status = read_gateway_launchd_status(run)
    if status.live:
        return 0

    if status.loaded:
        _log(f"loaded but not live: {_domain()}/{DEFAULT_LABEL} {status.summary}; attempting bootstrap")
    else:
        _log(f"domain missing: {_domain()}/{DEFAULT_LABEL}; attempting bootstrap")

    recovered = recover_gateway(run)
    if recovered:
        msg = "⚠️ Hermes default gateway was not live in launchd; watchdog re-bootstrapped it."
        _log("recovered default gateway launchd liveness")
        if alert:
            _send_telegram_alert(msg)
        return 0

    msg = "🔴 Hermes default gateway launchd watchdog failed to re-bootstrap a live ai.hermes.gateway."
    _log("recovery failed")
    if alert:
        _send_telegram_alert(msg)
    return 2


def main(argv: list[str] | None = None) -> int:
    argv = argv or []
    alert = "--no-alert" not in argv
    return check_once(alert=alert)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
