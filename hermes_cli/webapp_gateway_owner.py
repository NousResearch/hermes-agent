"""The gateway a Webapp browser socket attaches to: find or start the profile's owner, as Desktop does.

The browser-hosted Desktop is an attach client of the profile's canonical gateway (#106742), exactly
like local Electron Desktop (``apps/desktop/electron/local-gateway.ts``): ``ensure`` the owner (start
an installed service, else a daemon), require the session-authority and delegated-attach
capabilities, and restart an owner still running other code once. A browser can reach none of it;
the Webapp's relay (``webapp_gateway_relay``) dials on its behalf. There is never a fallback to
hosting turns in the Webapp itself (``apps/desktop/docs/canonical-gateway-protocol.md``).
"""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path

_log = logging.getLogger("hermes_cli.web_server")

REQUIRED_CAPABILITIES = ("session-authority-v1", "delegated-attach-v1")
_RESTART_TIMEOUT_S = 15 * 60
_restarts: dict[tuple, threading.Event] = {}
_restarts_lock = threading.Lock()
_warned_unsupervised: set[str] = set()
_ensuring: dict[str, asyncio.Future] = {}


class GatewayAttachError(RuntimeError):
    """The profile's gateway cannot take the browser's socket right now: the verdict a browser sees."""

    def __init__(self, state: str, reason_code: str | None, detail: str | None = None):
        super().__init__(f"Gateway {state} ({reason_code or 'not_ready'})" + (f". {detail}" if detail else ""))
        self.state, self.reason_code, self.detail = state, reason_code, detail

    def verdict(self) -> dict:
        return {"state": self.state, "reason_code": self.reason_code, "detail": self.detail}


async def owner_endpoint(home: Path):
    """The ready owner serving *home*, started if absent; raises :class:`GatewayAttachError`.

    A named profile the default gateway serves is routed the way Desktop routes it: when nothing
    answers for it, the default gateway is started, never a gateway of the profile's own. The
    choice is made from the profile's ``profiles/<name>`` path, because a profile folder that links
    to storage elsewhere resolves outside the root, and a second owner started for it would stop
    the default gateway from reserving its profiles."""
    from hermes_cli.gateway_runtime import attachable, discover_gateway_endpoint, gateway_profile_home
    from hermes_cli.gateway_runtime_multiplex import implied_host_root
    host = implied_host_root(Path(home))
    home = gateway_profile_home(home)
    result = await asyncio.to_thread(discover_gateway_endpoint, home)
    started = result.state == "absent"
    if not attachable(result):
        result = await _ensure_once(host or home)
        if host is not None and attachable(result):
            result = await asyncio.to_thread(discover_gateway_endpoint, home)
    if not attachable(result):
        raise GatewayAttachError(result.state, result.reason_code, result.detail)
    endpoint = result.endpoint
    if started:
        _warn_if_started_inside_this_unit(home, endpoint)
    _restart_once_if_stale(endpoint)
    missing = [c for c in REQUIRED_CAPABILITIES if c not in endpoint.capabilities]
    if missing:
        # An owner older than browser attach: the stale restart above (when it runs other code) is the
        # way out; until it lands, say so instead of hosting the turn here.
        raise GatewayAttachError("incompatible", f"{missing[0].removesuffix('-v1').replace('-', '_')}_unavailable",
                                 "The gateway serving this profile predates browser attach; restart it on this "
                                 "Hermes version (`hermes gateway restart`).")
    return endpoint


async def _ensure_once(home: Path):
    """``ensure_gateway_runtime`` for *home*, shared by every request that needs it meanwhile.

    A browser's boot asks for the primary and every profile at once; each ensure that finds no
    owner starts one, the extras exit with errors, and their requests answer 503."""
    from hermes_cli.gateway_runtime import ensure_gateway_runtime
    key = os.path.normcase(str(home))
    pending = _ensuring.get(key)
    if pending is None:
        pending = _ensuring[key] = asyncio.ensure_future(
            asyncio.to_thread(ensure_gateway_runtime, home, idle_exit=False))
        pending.add_done_callback(lambda _done: _ensuring.pop(key, None))
    # Shielded: a browser that leaves mid-start does not cancel the start other requests wait on.
    return await asyncio.shield(pending)


def _owner_profile(endpoint) -> str | None:
    """The profile whose ``hermes -p <name> gateway restart`` restarts this owner (the multiplexer's own
    for a served secondary), else None when no profile name maps to its home."""
    from gateway.status import _profile_label_for_home
    return _profile_label_for_home(Path(endpoint.control_home or endpoint.profile_id))


def _restart_once_if_stale(endpoint) -> None:
    """Restart an owner that booted other code than this Webapp runs, once per (owner, its code, ours).

    Electron's ``createStaleGatewayRestarter``: after one restart an owner a restart cannot move (a
    service unit pinned to another checkout) is attached as-is. Never awaited: the restart drains
    the owner, which closes this browser's socket with 1013, and the browser's re-dial lands on the
    replacement."""
    from hermes_cli.version_info import get_code_identity
    ours = get_code_identity().get("sha")
    if not ours or not endpoint.code_sha or endpoint.code_sha == ours:
        return
    profile = _owner_profile(endpoint)
    if profile is None:
        return
    key = (endpoint.control_home or endpoint.profile_id, endpoint.code_sha, ours)
    with _restarts_lock:
        if key in _restarts:
            if _restarts[key].is_set():
                _log.info("gateway for %s still runs %s (this Webapp runs %s) after a restart; attaching as-is",
                          profile, endpoint.code_sha[:10], ours[:10])
            return
        done = _restarts[key] = threading.Event()
    _log.warning("gateway for %s runs %s but this Webapp runs %s; restarting it",
                 profile, endpoint.code_sha[:10], ours[:10])
    from agent.memory_provider import spawn_context_thread
    spawn_context_thread(_run_restart, name="webapp-gateway-restart", args=(profile, done)).start()


def _run_restart(profile: str, done: threading.Event) -> None:
    try:
        completed = subprocess.run([sys.executable, "-m", "hermes_cli.main", "-p", profile, "gateway", "restart"],
                                   stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8",
                                   errors="replace", timeout=_RESTART_TIMEOUT_S, check=False)
        if completed.returncode:
            _log.warning("gateway restart for %s exited %s: %s", profile, completed.returncode,
                         (completed.stderr or completed.stdout).strip()[-500:])
    except (OSError, subprocess.SubprocessError) as exc:
        _log.warning("gateway restart for %s failed: %s", profile, exc)
    finally:
        done.set()


def _warn_if_started_inside_this_unit(home: Path, endpoint) -> None:
    """A Webapp running as a service unit that had to start an unsupervised gateway owns it by cgroup:
    stopping the unit stops the gateway (and its messaging). Say how to give it its own unit, once."""
    from gateway.control_socket import _detect_supervisor
    if endpoint.supervisor != "none" or _detect_supervisor() not in {"systemd", "launchd"}:
        return
    key = str(home)
    if key in _warned_unsupervised:
        return
    _warned_unsupervised.add(key)
    _log.warning("the gateway for %s has no service of its own; started from this Webapp's service it stops "
                 "with it. Install it as a service: `hermes gateway install`.", home)
