"""Which profiles does the LIVE default multiplexer serve? One answer for every CLI/dashboard surface.

``gateway/run_adapters.py::_record_served_profiles`` writes ``served_profiles`` into the default
home's ``gateway_state.json`` at startup. That record is the truth about the running process; the
default ``config.yaml`` plus ``GATEWAY_MULTIPLEX_PROFILES`` as seen by the *CLI* process is only a
guess (``hermes -p coder ...`` loads coder's ``.env``, so an env-only opt-in on the default profile
is invisible to it, and an allowlist edited after start flips the guess before the restart).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def live_default_gateway_pid() -> Optional[int]:
    """PID of the default profile's gateway when a VERIFIED live process owns it, else None.

    ``gateway.status.live_gateway_pid_for_home``: pid file + lock, then the runtime record the gateway
    itself writes, each proven against the live process (start time, gateway command line, home). A
    launch-service gateway can be live with no ``gateway.pid`` at all, and a stale record whose PID was
    recycled by an unrelated process must not make its ``served_profiles`` authoritative. Never key this
    off the record's ``updated_at``: an idle gateway never advances it.
    """
    from hermes_constants import get_default_hermes_root
    from gateway.status import live_gateway_pid_for_home
    return live_gateway_pid_for_home(get_default_hermes_root())


def recorded_served_profiles(default_root: Optional[Path] = None) -> Optional[list[str]]:
    """``served_profiles`` the live default gateway recorded, or None when the key is absent (a record
    from before the multiplexer recorded it, or a stopped/absent gateway). Callers fall back to config
    derivation only on None: an empty list is an authoritative "serves nobody else"."""
    from hermes_constants import get_default_hermes_root
    from gateway.status import read_runtime_status
    if live_default_gateway_pid() is None:
        return None
    runtime = read_runtime_status((default_root or get_default_hermes_root()) / "gateway_state.json")
    served = (runtime or {}).get("served_profiles")
    return [str(p) for p in served] if isinstance(served, list) else None


def multiplexer_served_secondaries() -> list[str]:
    """Named profiles the live default multiplexer serves (excludes ``default``); empty when none."""
    return [p for p in (recorded_served_profiles() or []) if p and p != "default"]


def served_profile_unserved_platforms(profile: str) -> dict[str, str]:
    """``{platform: reason}`` for a served profile's platforms the multiplexer deliberately does not run
    (unpaired WhatsApp or Relay shared ingress; ``gateway.run_adapters`` stamps
    ``<profile>:<platform>`` as ``disabled`` with a reason and remedy)."""
    from hermes_constants import get_default_hermes_root
    from gateway.status import read_runtime_status
    if not profile or live_default_gateway_pid() is None:
        return {}
    runtime = read_runtime_status(get_default_hermes_root() / "gateway_state.json") or {}
    platforms = runtime.get("platforms")
    if not isinstance(platforms, dict):
        return {}
    prefix = f"{profile}:"
    return {
        key[len(prefix):]: str(entry.get("error_message") or "not served under multiplex")
        for key, entry in platforms.items()
        if isinstance(key, str) and key.startswith(prefix) and isinstance(entry, dict)
        and entry.get("error_code") in {"multiplex_shared_ingress", "whatsapp_unpaired"}
    }


def served_profile_ingress_urls(profile: Optional[str] = None) -> dict[str, dict[str, str]]:
    """``{profile: {platform: url}}`` for every secondary inbound-port platform the live multiplexer
    serves on its shared listener (``<profile>:<platform>`` entries carrying ``ingress_url``). This is
    what the user pastes into the vendor console (Twilio, LINE, Teams, ...). ``profile`` narrows the map."""
    from hermes_constants import get_default_hermes_root
    from gateway.status import read_runtime_status, shared_listener_mirror_platforms
    if live_default_gateway_pid() is None:
        return {}
    runtime = read_runtime_status(get_default_hermes_root() / "gateway_state.json") or {}
    platforms = runtime.get("platforms")
    if not isinstance(platforms, dict):
        return {}
    urls: dict[str, dict[str, str]] = {}
    for key, entry in platforms.items():
        if not (isinstance(key, str) and ":" in key and isinstance(entry, dict)):
            continue
        url = entry.get("ingress_url")
        if not url or entry.get("state") in ("fatal", "disconnected", "stopped"):
            continue
        name, platform = key.split(":", 1)
        if profile and name != profile:
            continue
        urls.setdefault(name, {})[platform] = str(url)
    # api_server/webhook are the default's adapters mirrored at /p/<profile>/ (no entry of their own).
    served = [str(p) for p in (runtime.get("served_profiles") or []) if p and p != "default"]
    for name in served if not profile else [p for p in served if p == profile]:
        for platform, entry in shared_listener_mirror_platforms(runtime, name).items():
            if entry.get("ingress_url"):
                urls.setdefault(name, {})[platform] = str(entry["ingress_url"])
    return urls


def format_ingress_url_lines(urls: dict[str, str], indent: str = "  ") -> list[str]:
    """One ``<indent><platform>: <url>`` line per platform, sorted."""
    return [f"{indent}{platform}: {url}" for platform, url in sorted(urls.items())]


def notify_multiplexer_profiles_changed(profile_name: str, *, timeout: float = 8.0) -> Optional[list[str]]:
    """Tell the live default multiplexer that ``profiles/`` changed (``profile_name`` was created or
    deleted) so it hot-serves / unroutes it now instead of at its next periodic rescan. Returns the
    served-profile list the gateway answered with, or None when no multiplexer answered (no live default
    gateway, single-profile gateway, or a gateway predating the verb). Never raises."""
    try:
        from hermes_constants import get_default_hermes_root
        from gateway.control_socket import rescan_gateway_profiles
        if live_default_gateway_pid() is None:
            return None
        answer = rescan_gateway_profiles(get_default_hermes_root(), timeout=timeout)
    except Exception:
        logger.debug("multiplexer rescan notification failed for %r", profile_name, exc_info=True)
        return None
    if not isinstance(answer, dict) or answer.get("multiplex") is False:
        return None
    served = answer.get("served_profiles")
    return [str(p) for p in served] if isinstance(served, list) else None
