"""Dead-man's switch + cloudflared supervisor for ``hermes tunnel``.

The idle-reset policy is the core safety protocol: a tunnel with no
incoming traffic for ``idle_timeout_seconds`` closes (graceful drain +
kill cloudflared), so a forgotten test build cannot leak to the internet.
An admin-approved hold disables the idle timer until ``hold_until``; after
that the idle timer resumes (no hard kill on approval expiry).
"""

from __future__ import annotations


def reset_idle_on(prev_counter: int, cur_counter: int) -> bool:
    """Return True when there has been incoming activity since the last poll.

    Activity = the cloudflared request counter strictly increased.
    A counter that stayed flat or dropped (poll hiccup / restart) is NOT
    activity.
    """
    # Activity = the request counter strictly increased since the last
    # poll. A flat or dropped counter (poll hiccup / cloudflared restart)
    # is NOT activity, so a dead tunnel can't keep itself alive.
    return cur_counter > prev_counter


def should_close_now(state: dict) -> bool:
    """Return True when the tunnel should close now.

    state keys: now, last_activity, idle_timeout_seconds, hold_until (|None).

    Rules (see TestPolicy for the exact contract):
      * If an admin-approved hold is active (hold_until is in the future),
        never close.
      * Otherwise close when (now - last_activity) >= idle_timeout_seconds.
      * A hold whose hold_until is in the past is treated as "no hold"
        (fall back to the idle rule) — do NOT hard-kill just because the
        approval expired.
    """
    # An admin-approved hold that is still in the future disables the idle
    # timer. A hold whose deadline is in the past is treated as "no hold"
    # (fall back to the idle rule) — we must NOT hard-kill just because the
    # approval expired; the idle timer resumes and decides.
    now = state["now"]
    last_activity = state["last_activity"]
    idle_timeout_seconds = state["idle_timeout_seconds"]
    hold_until = state.get("hold_until")
    if hold_until is not None and now < hold_until:
        return False
    return (now - last_activity) >= idle_timeout_seconds


import os
import subprocess
import time
import urllib.request
import json


def _default_counter(metrics_port: int) -> int:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{metrics_port}/metrics", timeout=2) as r:
            for line in r.read().decode("utf-8", "replace").splitlines():
                # cloudflared exposes counters; tolerate either name.
                if line.startswith("cloudflared_request_count") or line.startswith("cloudflared_connection_count"):
                    return int(float(line.split()[-1]))
    except Exception:
        return 0
    return 0


class TunnelSupervisor:
    def __init__(self, config, approvals_path, *, hold_request_id=None,
                 time_source=time.monotonic, metrics_counter=None,
                 spawn_cloudflared=None, sleep=time.sleep):
        self._cfg = config
        self._approvals_path = approvals_path
        self._hold_request_id = hold_request_id
        self._time = time_source
        self._sleep = sleep
        self._idle = float(config.get("idle_timeout_seconds", 1800))
        self._drain = float(config.get("drain_seconds", 15))
        self._poll = float(config.get("poll_interval_seconds", 5))
        self._metrics_port = int(config.get("metrics_port", 0))

        self._last_counter = 0
        self._last_activity = None
        self._hold_until = None
        self._closed = False
        self._proc = None
        self._config_path = None

        if metrics_counter is None:
            metrics_counter = lambda: _default_counter(self._metrics_port)
        self._counter = metrics_counter
        self._spawn = spawn_cloudflared or self._default_spawn

    def _default_spawn(self, config_path, tunnel_name, metrics_port):
        cmd = ["cloudflared", "tunnel", "--config", config_path,
               "--metrics", f"127.0.0.1:{metrics_port}", "run", tunnel_name]
        return subprocess.Popen(cmd)

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def last_activity(self) -> float:
        return self._last_activity

    @property
    def hold_until(self):
        return self._hold_until

    def _check_hold(self):
        if not self._hold_request_id:
            return
        from hermes_cli import tunnel_approvals as ta
        if ta.is_approved(self._approvals_path, self._hold_request_id):
            self._hold_until = ta.approved_until(self._approvals_path, self._hold_request_id)

    def _drain_and_kill(self):
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=max(1.0, self._drain))
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        self._closed = True

    def tick(self) -> bool:
        """One poll iteration. Returns True while running, False once closed."""
        if self._closed:
            return False
        if self._proc is None:
            self._proc = self._spawn(self._config_path, self._cfg.get("tunnel_name", ""),
                                     self._metrics_port)
        now = self._time()
        if self._last_activity is None:
            self._last_activity = now
        cur = int(self._counter())
        if reset_idle_on(self._last_counter, cur):
            self._last_activity = now
        self._last_counter = cur
        self._check_hold()
        state = {"now": now, "last_activity": self._last_activity,
                 "idle_timeout_seconds": self._idle, "hold_until": self._hold_until}
        if should_close_now(state):
            self._drain_and_kill()
            return False
        return True

    def run(self, config_path: str):
        """Blocking loop. ``config_path`` is the generated cloudflared config file."""
        self._config_path = config_path
        while self.tick():
            self._sleep(self._poll)
        return not self._closed