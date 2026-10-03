"""Deadman's switch: the gateway startup block verifies the transcriber stack (LAG-677).

Second layer of defense for the transcriber deployment: if the transcriber's
own watchdog (``com.alex.transcriber.watchdog``) is itself dead — it was, all
through the 2026-09 outage — the ``ai.hermes.gateway`` startup block should
notice and notify. The transcriber watchdog cannot detect its own death; only
an independent process can. The gateway is the right independent observer: it
restarts via launchd on boot and after crashes, exactly when a dead stack
should be re-checked.

Scoped to the same startup-liveness block that arms
``hermes_startup_watchdog`` (``hermes_cli.main`` — the block is shared by both
launchd gateway instances, so both callers behave identically). Runs once per
gateway start, before the heavy import graph; the module is stdlib-only so it
cannot wedge or slow that window (same import-lightness contract as
``hermes_startup_watchdog``). Never raises: a deadman failure must never
affect the startup it observes.

Signals (mirror health_check.py doctrine — the heartbeat file's mtime is the
only honest liveness signal; launchctl PIDs lie in both directions):

* fresh heartbeat — ``~/.superwhisper_transcriber_heartbeat.json`` mtime within
  the staleness threshold. A heartbeat whose ``writer`` field reads as
  ``"manual"`` (HB-12) does not prove daemon liveness: an ops script refreshed
  the file, the daemon may still be dead.
* watchdog process present — exact-label row in ``launchctl list`` for
  ``com.alex.transcriber.watchdog`` (HC-6 exact matching; the substring trap
  ``com.alex.transcriber`` ⊂ ``...watchdog`` must not confuse the two).

Healthy = fresh daemon heartbeat AND watchdog process present → silent. Any
other combination → one macOS notification per gateway start (osascript argv
pattern, ES-1/ES-2: text passed as argv, never interpolated into AppleScript
source). No cooldown is needed: gateway starts are rare (boot, crash, manual
restart).

Gates (env-only; config.yaml parsing is itself in the startup-wedge scope):

* ``HERMES_GATEWAY_DEADMAN`` — ``0``/``false``/``no``/``off`` disables.
* Absent heartbeat file → inert no-op: users without the transcriber stack are
  untouched, and the deployed gateway needs no plist change to adopt this.
* A non-expired pause sentinel (``~/.superwhisper_transcriber_watchdog.pause``,
  WD-10/WD-10-revision) means planned maintenance — stay silent.
* Overrides: ``HERMES_GATEWAY_DEADMAN_HEARTBEAT`` (path),
  ``HERMES_GATEWAY_DEADMAN_LABEL`` (launchd label),
  ``HERMES_GATEWAY_DEADMAN_STALE_S`` (staleness threshold, clamped to
  [60, 3600]).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, Optional

DEFAULT_HEARTBEAT_PATH = "~/.superwhisper_transcriber_heartbeat.json"
DEFAULT_LABEL = "com.alex.transcriber"
DEFAULT_WATCHDOG_LABEL = "com.alex.transcriber.watchdog"
DEFAULT_PAUSE_SENTINEL_PATH = "~/.superwhisper_transcriber_watchdog.pause"

# A healthy daemon refreshes its heartbeat every scan cycle (30s); 600s is
# 20x that, far enough above any transient stall to mean "the writer is gone".
DEFAULT_STALE_S = 600.0
_MIN_STALE_S = 60.0
_MAX_STALE_S = 3600.0

# Pause sentinel TTL — mirrors health_check.py's WD-10 revision constant
# (LAG-674): an older sentinel is a forgotten one, not a live maintenance.
PAUSE_TTL_S = 4 * 3600.0

# WD-7 parity: every spawned subprocess gets a timeout and an absolute path
# (WD-8: an absolute launchctl path cannot be shimmed via PATH).
_LAUNCHCTL_BIN = "/bin/launchctl"
_SUBPROCESS_TIMEOUT = 15.0

ENV_DEADMAN = "HERMES_GATEWAY_DEADMAN"
ENV_DEADMAN_HEARTBEAT = "HERMES_GATEWAY_DEADMAN_HEARTBEAT"
ENV_DEADMAN_LABEL = "HERMES_GATEWAY_DEADMAN_LABEL"
ENV_DEADMAN_STALE_S = "HERMES_GATEWAY_DEADMAN_STALE_S"

_FALSEY = frozenset({"0", "false", "no", "off"})

# Duplicated from health_check.py on purpose: that module lives in another
# repository and must stay importable under /usr/bin/python3 without this one.
# Post-HB-12 writers always set ``writer``; a missing/non-string field reads as
# the daemon (pre-HB-12 heartbeats carried no field), never the reverse — the
# permissive default cannot hide a manual run.
_DAEMON_WRITER = "daemon"


def _env_path(name: str, default: str) -> str:
    raw = os.environ.get(name, "").strip()
    return raw if raw else default


def _stale_threshold() -> float:
    """Env override, clamped to [_MIN_STALE_S, _MAX_STALE_S]; default on garbage."""
    raw = os.environ.get(ENV_DEADMAN_STALE_S, "").strip()
    if not raw:
        return DEFAULT_STALE_S
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_STALE_S
    if value <= 0:
        return DEFAULT_STALE_S
    return min(max(value, _MIN_STALE_S), _MAX_STALE_S)


def heartbeat_age_s(path: str, *, now: Optional[float] = None) -> Optional[float]:
    """Seconds since the heartbeat file's mtime; None when missing/unreadable.

    The mtime is the liveness signal (HC-1); an unreadable file proves no
    daemon write either way, so both cases collapse to None.
    """
    try:
        mtime = os.stat(os.path.expanduser(path)).st_mtime
    except OSError:
        return None
    current = time.time() if now is None else now
    return current - mtime


def heartbeat_writer(payload: Any) -> str:
    """Which process wrote this heartbeat (HB-12). Pure; mirrors health_check.py."""
    if not isinstance(payload, dict):
        return _DAEMON_WRITER
    writer = payload.get("writer")
    return writer if isinstance(writer, str) else _DAEMON_WRITER


def read_heartbeat_writer(path: str) -> str:
    """Writer field of the heartbeat payload; "daemon" when unreadable/absent.

    An unreadable file is already a deadman trigger via the mtime being None,
    so the writer here only refines the reason string.
    """
    try:
        with open(os.path.expanduser(path), encoding="utf-8") as handle:
            return heartbeat_writer(json.load(handle))
    except (OSError, ValueError):
        return _DAEMON_WRITER


def labels_in_launchctl_list(output: str, labels: tuple) -> Dict[str, bool]:
    """Which of *labels* have a row in ``launchctl list`` output. Pure (HC-6).

    Exact field equality — substring matching would confuse
    ``com.alex.transcriber`` with ``com.alex.transcriber.watchdog``.
    """
    present = {label: False for label in labels}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        row_label = " ".join(fields[2:])
        if row_label in present:
            present[row_label] = True
    return present


def launchctl_presence(labels: tuple) -> Optional[Dict[str, bool]]:
    """One ``launchctl list`` → presence map, or None when launchctl failed."""
    try:
        result = subprocess.run(
            [_LAUNCHCTL_BIN, "list"],
            capture_output=True,
            text=True,
            check=False,
            timeout=_SUBPROCESS_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return labels_in_launchctl_list(result.stdout, labels)


def pause_sentinel_state(path: str, *, now: Optional[float] = None) -> str:
    """Classify the pause sentinel: "absent", "paused" or "expired" (WD-10)."""
    try:
        mtime = os.stat(os.path.expanduser(path)).st_mtime
    except OSError:
        return "absent"
    current = time.time() if now is None else now
    return "expired" if current - mtime > PAUSE_TTL_S else "paused"


def assess(
    *,
    heartbeat_path: str,
    label: str,
    watchdog_label: str,
    now: Optional[float] = None,
    launchctl: Optional[Dict[str, bool]] = None,
) -> Dict[str, Any]:
    """Liveness assessment of the transcriber stack. Pure apart from launchctl.

    Returns a dict with ``healthy`` (fresh daemon heartbeat AND watchdog
    process present), the individual signals, and diagnostic ``reasons``.
    ``launchctl`` injects a presence map for tests; None means launchctl could
    not be read, which is fail-closed: the watchdog cannot be proven alive.
    """
    age = heartbeat_age_s(heartbeat_path, now=now)
    stale_s = _stale_threshold()
    heartbeat_fresh = age is not None and age <= stale_s
    writer = read_heartbeat_writer(heartbeat_path) if age is not None else None
    writer_is_daemon = writer == _DAEMON_WRITER

    presence: Optional[Dict[str, bool]] = (
        launchctl_presence((label, watchdog_label)) if launchctl is None else launchctl
    )
    watchdog_alive = bool(presence and presence.get(watchdog_label))
    daemon_alive = bool(presence and presence.get(label))

    reasons: list = []
    if age is None:
        reasons.append("heartbeat_missing_or_unreadable")
    elif age > stale_s:
        reasons.append(f"heartbeat_stale ({age:.0f}s > {stale_s:.0f}s)")
    elif not writer_is_daemon:
        reasons.append(f"heartbeat_writer_is_{writer}")
    if presence is None:
        reasons.append("watchdog_unverifiable (launchctl failed)")
    elif not watchdog_alive:
        reasons.append("watchdog_process_absent")

    return {
        "healthy": heartbeat_fresh and writer_is_daemon and watchdog_alive,
        "heartbeat_age_s": age,
        "heartbeat_writer": writer,
        "watchdog_process_present": watchdog_alive,
        "daemon_process_present": daemon_alive,
        "stale_threshold_s": stale_s,
        "reasons": reasons,
    }


def notify(message: str, title: str = "Transcriber deadman") -> bool:
    """One macOS notification via osascript, text passed as argv — never
    interpolated into AppleScript source (ES-1/ES-2). False on any failure."""
    try:
        result = subprocess.run(
            [
                "/usr/bin/osascript",
                "-e",
                "on run argv",
                "-e",
                "display notification (item 1 of argv) with title (item 2 of argv)",
                "-e",
                "end run",
                message,
                title,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=_SUBPROCESS_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def run_gateway_deadmans_switch(
    *,
    now: Optional[float] = None,
    launchctl: Optional[Dict[str, bool]] = None,
    do_notify: bool = True,
    pause_path: str = DEFAULT_PAUSE_SENTINEL_PATH,
) -> Dict[str, Any]:
    """Assess the transcriber stack once and notify when it is not healthy.

    Called once per gateway start from the startup-liveness block. Never
    raises: every failure mode collapses into the returned state dict.
    """
    try:
        if os.environ.get(ENV_DEADMAN, "").strip().lower() in _FALSEY:
            return {"fired": False, "skipped": True, "reason": "disabled_by_env"}

        heartbeat_path = _env_path(ENV_DEADMAN_HEARTBEAT, DEFAULT_HEARTBEAT_PATH)
        if not os.path.exists(os.path.expanduser(heartbeat_path)):
            # Inert on machines without the transcriber stack (LAG-677 gate).
            return {"fired": False, "skipped": True, "reason": "heartbeat_absent"}

        sentinel = pause_sentinel_state(pause_path, now=now)
        if sentinel == "paused":
            return {
                "fired": False,
                "skipped": True,
                "reason": "maintenance_pause_sentinel",
            }

        state = assess(
            heartbeat_path=heartbeat_path,
            label=_env_path(ENV_DEADMAN_LABEL, DEFAULT_LABEL),
            watchdog_label=DEFAULT_WATCHDOG_LABEL,
            now=now,
            launchctl=launchctl,
        )
        if state["healthy"]:
            state["fired"] = False
            state["skipped"] = False
            return state

        state["fired"] = do_notify and notify(
            "Transcriber stack not healthy at gateway start: "
            + "; ".join(state["reasons"])
        )
        state["notified"] = bool(state["fired"])
        return state
    except Exception as error:  # noqa: BLE001 — a deadman failure must never affect startup
        print(f"deadman: assessment failed: {error}", file=sys.stderr)
        return {"fired": False, "skipped": True, "reason": f"internal_error: {error}"}
