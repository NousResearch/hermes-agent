"""Validated, credential-free gateway endpoint discovery for local clients."""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import math
import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from gateway.runtime_contract import RuntimeState


@dataclass(frozen=True)
class GatewayEndpoint:
    profile_id: str
    instance_id: str
    authority_epoch: int
    runtime_protocol: int
    api_origin: str
    supervisor: Literal["none", "systemd", "launchd", "windows", "external"]
    capabilities: frozenset[str]
    # Home whose control socket answers for this endpoint: the profile's own home, or the
    # default multiplexer's root when the profile is served by it.
    control_home: str | None = None


@dataclass(frozen=True)
class GatewayDiscovery:
    state: RuntimeState
    endpoint: GatewayEndpoint | None = None
    reason_code: str | None = None


def _canonical_home(home: str | Path) -> str:
    return os.path.normcase(str(Path(home).expanduser().resolve()))


def _endpoint(payload: dict, home: Path, control_home: Path | None = None) -> GatewayDiscovery:
    if type(payload.get("runtime_protocol")) is not int or payload["runtime_protocol"] != 1:
        return GatewayDiscovery("incompatible", reason_code="runtime_protocol")
    profiles = payload.get("served_profiles")
    if not isinstance(profiles, list):
        return GatewayDiscovery("inaccessible", reason_code="profile_mismatch")
    # Canonicalize BOTH sides: on Windows normcase lower-cases the served spelling, so a
    # caller's mixed-case Path (WorkerRPC) never matched and every worker saw owner_unavailable.
    wanted = _canonical_home(home)
    matches = [p for p in profiles if isinstance(p, dict)
               and isinstance(p.get("home"), str) and _canonical_home(p["home"]) == wanted]
    if len(matches) != 1 or not isinstance(matches[0].get("profile_id"), str) or not matches[0]["profile_id"]:
        return GatewayDiscovery("inaccessible", reason_code="profile_mismatch")
    state = payload.get("state")
    if state in {"starting", "draining", "conflict"}:
        return GatewayDiscovery(state)
    if state != "ready":
        return GatewayDiscovery("inaccessible", reason_code="invalid_runtime_state")
    capabilities = payload.get("capabilities")
    if (not isinstance(capabilities, list) or not all(isinstance(c, str) for c in capabilities)
            or "session-authority-v1" not in capabilities):
        return GatewayDiscovery("incompatible", reason_code="session_authority_unavailable")
    epoch, instance = payload.get("authority_epoch"), payload.get("instance_id")
    if type(epoch) is not int or epoch <= 0 or not isinstance(instance, str) or not instance:
        return GatewayDiscovery("inaccessible", reason_code="invalid_runtime_identity")
    origin = payload.get("api_origin")
    if not isinstance(origin, str):
        return GatewayDiscovery("inaccessible", reason_code="invalid_api_origin")
    try:
        parsed = urlsplit(origin)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.path or parsed.query or parsed.fragment or not parsed.port
                or not ipaddress.ip_address(parsed.hostname).is_loopback):
            return GatewayDiscovery("inaccessible", reason_code="invalid_api_origin")
    except ValueError:
        return GatewayDiscovery("inaccessible", reason_code="invalid_api_origin")
    supervisor = payload.get("supervisor")
    if supervisor not in {"none", "systemd", "launchd", "windows", "external"}:
        return GatewayDiscovery("inaccessible", reason_code="invalid_supervisor")
    return GatewayDiscovery("ready", GatewayEndpoint(
        profile_id=matches[0]["profile_id"], instance_id=instance,
        authority_epoch=epoch, runtime_protocol=1, api_origin=origin,
        supervisor=supervisor, capabilities=frozenset(capabilities),
        control_home=str(control_home) if control_home is not None else None,
    ))


def control_home_for(home: Path, endpoint: GatewayEndpoint | None) -> Path:
    """Home whose control socket mints tickets for *endpoint* (the multiplexer's for a served profile)."""
    if endpoint is not None and endpoint.control_home:
        return Path(endpoint.control_home)
    return home


def _multiplexer_home_for(home: Path) -> Path | None:
    """Default root that may multiplex *home*, when *home* is a named profile under it."""
    from hermes_constants import named_profile_home
    if named_profile_home(home) is None:
        return None
    root = home.parent.parent
    return root if root != home else None


def _multiplexer_starting(home: Path) -> bool:
    from hermes_cli.gateway_runtime_discovery import missing_owner_state
    root = _multiplexer_home_for(home)
    return root is not None and missing_owner_state(root) == "starting"


def _served_by_multiplexer(home: Path, *, timeout: float) -> GatewayDiscovery | None:
    """A served secondary has no socket of its own: the default multiplexer's control socket
    answers for it, and its ``identify`` lists the home under ``served_profiles``. Only the
    multiplexer's live descriptor proves service; a held ``gateway.lock`` alone does not."""
    from hermes_cli.gateway_runtime_discovery import DiscoveryError, query_identify
    root = _multiplexer_home_for(home)
    if root is None:
        return None
    try:
        payload = query_identify(root, timeout=timeout)
    except (FileNotFoundError, ConnectionRefusedError, TimeoutError, DiscoveryError, PermissionError,
            OSError, ValueError, TypeError):
        return None
    result = _endpoint(payload, home, control_home=root)
    if result.state == "inaccessible" and result.reason_code == "profile_mismatch":
        return None  # the multiplexer does not serve this profile
    return result


def discover_gateway_endpoint(profile_home: str | Path, *, timeout: float = 2.0) -> GatewayDiscovery:
    """Read live control identity, preserving uncertainty instead of spawning.

    A named profile served by the default multiplexer resolves to the multiplexer's endpoint
    with ``profile_id`` = the profile's own home, so clients attach to it with that identity.
    """
    from hermes_cli.gateway_runtime_discovery import DiscoveryError, missing_owner_state, query_identify

    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    home = Path(_canonical_home(profile_home))
    try:
        return _endpoint(query_identify(home, timeout=timeout), home)
    except (FileNotFoundError, ConnectionRefusedError):
        served = _served_by_multiplexer(home, timeout=timeout)
        if served is not None:
            return served
        return GatewayDiscovery(missing_owner_state(home))
    except TimeoutError:
        return GatewayDiscovery("inaccessible", reason_code="control_timeout")
    except DiscoveryError as exc:
        return GatewayDiscovery("inaccessible", reason_code=exc.reason)
    except PermissionError:
        return GatewayDiscovery("inaccessible", reason_code="authorization")
    except (OSError, ValueError, TypeError):
        # Raw peer data, paths and URLs may contain secrets; report bounded codes.
        return GatewayDiscovery("inaccessible", reason_code="invalid_control_peer")


def ensure_gateway_runtime(profile_home: str | Path, *, timeout: float = 30.0) -> GatewayDiscovery:
    """Ensure once, never install/replace; pending remains pending at deadline.

    A successful service command or Popen is not session readiness. After an
    owner/start request is observed this invocation never launches another.
    """
    import time
    from hermes_cli.gateway_runtime_service import (
        RuntimeStartError, discover_existing_gateway_service, remaining,
        start_existing_gateway_service,
    )
    from hermes_cli.update_lock import MARKER_NAME
    from hermes_constants import get_default_hermes_root, get_process_hermes_home

    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    deadline = time.monotonic() + timeout
    home = Path(_canonical_home(profile_home))
    requested = False
    delay = 0.025
    try:
        while True:
            # Never clear the updater's fence, even if malformed. Repair is not
            # a client operation. The install-root marker also covers profiles.
            for fence_home in {home, get_process_hermes_home(), get_default_hermes_root()}:
                try:
                    (fence_home / MARKER_NAME).lstat()
                except FileNotFoundError:
                    continue
                return GatewayDiscovery("draining", reason_code="update_paused")
            observed = discover_gateway_endpoint(home, timeout=remaining(deadline))
            if observed.reason_code == "control_timeout":
                return GatewayDiscovery("starting", reason_code="deadline")
            if observed.state not in {"absent", "starting"}:
                return observed
            if observed.state == "starting":
                requested = True
            # A default multiplexer that is still starting will serve this named profile; never
            # spawn a competing per-profile daemon while its reservation is pending.
            if observed.state == "absent" and not requested and _multiplexer_starting(home):
                requested = True
            if observed.state == "absent" and not requested:
                service = discover_existing_gateway_service(home, deadline=deadline)
                # Runtime locks settle races remaining after this second probe.
                observed = discover_gateway_endpoint(home, timeout=remaining(deadline))
                if observed.state != "absent":
                    continue
                if service is not None:
                    start_existing_gateway_service(service, deadline=deadline)
                else:
                    from hermes_cli.gateway_runtime_start import spawn_unmanaged_gateway
                    spawn_unmanaged_gateway(home, deadline=deadline)
                requested = True
            time.sleep(min(delay, remaining(deadline)))
            delay = min(delay * 1.5, 0.25)
    except TimeoutError:
        return GatewayDiscovery("starting", reason_code="deadline")
    except RuntimeStartError as exc:
        return GatewayDiscovery(exc.state, reason_code=exc.reason)
    except PermissionError:
        return GatewayDiscovery("inaccessible", reason_code="authorization")
    except OSError:
        return GatewayDiscovery("inaccessible", reason_code="runtime_start_failed")
