"""``hermes profile create`` through the running gateway: one ``profiles.create`` RPC, one receipt.

The CLI asks the gateway that serves its profile to make the profile (``HERMES_TUI_GATEWAY_URL``, the dial
``connect_gateway`` honours for every client, names a remote one), so a terminal attached to a gateway
creates profiles on the gateway's machine. The gateway answers with the same
:class:`~hermes_cli.profile_provision.ProfileCreateReceipt` a local create builds, so the output does not depend
on where the profile was made.
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

from hermes_cli.profile_provision import (
    ProfileCreateReceipt, ProfileCreateRequest, message_from, redact_create_text, refusal_from,
)

logger = logging.getLogger(__name__)

#: Skill seeding runs a subprocess (60 s cap) and the multiplexer rescan answer waits up to 8 s.
CREATE_RPC_TIMEOUT = 300
#: Answered before the handler runs, so nothing was made there. 4000: a gateway that predates these
#: fields (unknown param); 5035: one retiring between discovery and the request.
_OLDER_OR_RETIRING = (4000, 5035)
#: Refused before any handler ran, by code AND ``data.reason`` (a handler failure is 5062 whatever its text):
#: the authority socket's routing (4001 ``profile_mismatch``: a profile this gateway doesn't serve, or a
#: profile-bound grant naming another) and ``ws_legacy_fallback`` (4064 ``profile_unavailable``; a bare 4064
#: is not this: the dispatcher also answers it for a config outage raised after the profile was made).
_PROFILE_REFUSED = ((4001, "profile_mismatch"), (4064, "profile_unavailable"))
#: The authority socket's refusal of a draining owner, before routing: like a retiring gateway.
_DRAINING = (4001, "runtime_draining")


def _refusal_key(exc) -> tuple:
    return exc.code, exc.data.get("reason")


class GatewayCreateError(RuntimeError):
    """The named gateway cannot take the create, or took it and its outcome is unknown."""


def _wire(request: ProfileCreateRequest, profile: Optional[str]) -> dict[str, Any]:
    # ``no_alias`` always: the wrapper belongs to the terminal's machine, PATH and ``hermes``
    # (written here when the profile landed in this machine's profiles directory). ``profile``: a URL's
    # connection carries no profile-bound ticket, so the terminal's profile is named (Desktop's routing).
    wire = {"name": request.name, "description": request.description, "clone_from": request.clone_from,
            "clone_all": request.clone_all, "clone_config": request.clone_config,
            "clone_channels": request.clone_channels, "no_skills": request.no_skills,
            "no_alias": True, "sync_imports": request.sync_imports,
            "mirror_credentials": request.mirror_credentials, "share_auth": request.share_auth, "report": True}
    return {**wire, "profile": profile} if profile else wire


def _where(remote: str) -> str:
    """How messages name the gateway. A URL ``connect_gateway`` would refuse (not ws/wss with a host) or
    whose port is malformed is a :class:`GatewayCreateError` here; any other URL is dialed. It is named as
    ``scheme://host[:port]`` (``safe_proxy_url_for_log``) or, when that can't be shown safely (an ``@`` past
    the authority), by the variable: never with its token, password or path."""
    if not remote:
        return "the running gateway"
    try:
        parts = urlsplit(remote)
        ok = parts.scheme in ("ws", "wss") and bool(parts.hostname) and (parts.port is None or parts.port > 0)
    except ValueError:  # an out-of-range port or a malformed authority; its message quotes the authority
        ok = False
    if not ok:
        raise GatewayCreateError("HERMES_TUI_GATEWAY_URL is not a ws:// or wss:// URL with a host; nothing was created")
    from gateway.platforms.base_log_safety import safe_proxy_url_for_log  # a URL only: it loads gateway.platforms
    named = safe_proxy_url_for_log(remote)
    return f"the gateway at {named}" if named.startswith(("ws://", "wss://")) else "the HERMES_TUI_GATEWAY_URL gateway"


def _local_profile_dir(name: str) -> Optional[Path]:
    from hermes_cli.profiles import get_profile_dir
    try:
        return Path(get_profile_dir(name))
    except ValueError:
        return None


def _is_profile(local: Optional[Path]) -> bool:
    """Whether ``-p <name>``'s directory here is a profile: it carries an identity marker. A shell a deleted
    profile left behind carries none (``create_profile`` replaces it), and nothing else creates one."""
    from hermes_constants import named_profile_has_identity
    return local is not None and local.is_dir() and named_profile_has_identity(local)


def _made_here(made: ProfileCreateReceipt, local: Optional[Path], was_profile: bool) -> bool:
    """Whether this create made the profile ``-p <name>`` selects on this machine, given the gateway is this
    home's owner: not a profile before the request, one now, at the path the gateway reports."""
    from hermes_cli.profiles import profile_matches_home
    return (not was_profile and _is_profile(local)
            and profile_matches_home(local.name, Path(made.path)))


async def _described_instance(client) -> Optional[str]:
    """The ``instance_id`` the connected gateway reports (``runtime.describe``), or None when it doesn't."""
    try:
        described = await client.rpc("runtime.describe")
    except Exception:  # an older or refusing gateway: its identity is unproven, so no wrapper
        logger.debug("profile create: runtime.describe failed", exc_info=True)
        return None
    instance = described.get("instance_id") if isinstance(described, dict) else None
    return instance if isinstance(instance, str) and instance else None


def create_through_gateway(request: ProfileCreateRequest) -> Optional[ProfileCreateReceipt]:
    """The receipt of the gateway's create for *request*, or None to create here.

    ``HERMES_TUI_GATEWAY_URL`` set: that gateway makes the profile, in this terminal's profile, and anything
    short of its answer is a :class:`GatewayCreateError`, never a local create. Otherwise a live gateway for
    this profile (discovery only; one is never started for this) makes it; with none, or one that predates
    the fields this sends, the caller creates locally. A refused create raises the gateway's own
    ``ValueError`` / ``FileExistsError`` / ``FileNotFoundError``. When this create made the profile in this
    machine's profiles directory, the terminal's ``~/.local/bin`` wrapper is written here.
    """
    from hermes_cli.sessions_owner import live_owner
    remote = os.environ.get("HERMES_TUI_GATEWAY_URL", "").strip()
    where = _where(remote)
    endpoint, profile = None, None
    if remote:
        # The profile this terminal runs in, resolved like the local dial (an overlay task home is served by
        # the profile its lock names); a home outside the profiles tree has no name a gateway can resolve.
        from hermes_cli.gateway_runtime import gateway_profile_home
        from hermes_constants import get_hermes_home, profile_name_for_home
        profile = profile_name_for_home(gateway_profile_home(get_hermes_home()))
        if not profile:
            raise GatewayCreateError(f"this terminal's HERMES_HOME is not a named profile, so {where} cannot "
                                     "create from it; nothing was created")
    if not remote or request.alias:
        try:
            endpoint = live_owner()
        except Exception:
            logger.debug("profile create: gateway discovery failed", exc_info=True)
    if not remote and endpoint is None:
        return None
    from hermes_cli.gateway_client import GatewayRPCError, NOT_SENT, connect_gateway
    machine = " on the gateway's machine" if remote else ""
    unknown = f"{where} may have created '{request.name}'; check `hermes profile list`{machine} before retrying"
    local = _local_profile_dir(request.name)
    was_profile = _is_profile(local)
    # A discovered owner serves this home. A URL's gateway is this home's owner only if it reports the
    # instance this home's discovery names: path equality is no proof (another machine with the same home
    # layout reports the same path while a local create of the name runs), nor is a loopback URL.
    ours = not remote
    owner_here = endpoint.instance_id if remote and endpoint is not None else None
    attached = False

    async def _run():
        nonlocal attached, ours
        async with connect_gateway(None if remote else endpoint) as client:
            attached = True
            if owner_here is not None:
                ours = await _described_instance(client) == owner_here
            return await client.rpc("profiles.create", _timeout=CREATE_RPC_TIMEOUT, **_wire(request, profile))

    try:
        answer = asyncio.run(_run())
    except KeyboardInterrupt:
        if not attached:
            raise
        raise GatewayCreateError(f"interrupted; {unknown}") from None
    except GatewayRPCError as exc:
        draining = _refusal_key(exc) == _DRAINING
        unknown_method = exc.unknown_method and exc.code in (None, -32601)
        if exc.code in _OLDER_OR_RETIRING or unknown_method or draining:
            if remote:
                why = "is restarting; retry" if exc.code == 5035 or draining else (
                    "predates this version; update it, or unset HERMES_TUI_GATEWAY_URL to create locally")
                raise GatewayCreateError(f"{where} cannot create profiles for this terminal (it {why})") from exc
            return None
        refused = refusal_from(exc.data)
        if refused is not None:
            raise refused from exc
        if _refusal_key(exc) in _PROFILE_REFUSED:
            raise GatewayCreateError(f"{where} cannot serve this terminal's profile "
                                     f"({profile or 'its profile'}); nothing was created") from exc
        # Not a refusal of the request: the handler failed, possibly after the profile was made.
        reason = message_from(exc.data) or redact_create_text(exc)
        raise GatewayCreateError(f"{where} did not finish the create ({reason}); {unknown}") from exc
    except Exception as exc:
        if not attached or str(exc) == NOT_SENT:  # the dial, ticket or handshake failed: nothing was sent
            if remote:
                raise GatewayCreateError(f"could not reach {where} ({type(exc).__name__})") from exc
            logger.info("profile create: the running gateway is unreachable (%s); creating locally", exc)
            return None
        raise GatewayCreateError(f"lost {where} during the create ({type(exc).__name__}); {unknown}") from exc
    made = ProfileCreateReceipt.from_wire(answer.get("receipt") if isinstance(answer, dict) else None)
    if made is None:
        # It made a profile without a full, well-typed receipt (``--clone`` / ``--sync-imports`` may be ignored).
        raise GatewayCreateError(f"{where} answered without the details this version needs; {unknown}")
    made = dataclasses.replace(made, warnings=[redact_create_text(w) for w in made.warnings])
    here = request.alias and ours and _made_here(made, local, was_profile)
    return _with_local_alias(made, request) if here else made


def _with_local_alias(made: ProfileCreateReceipt, request: ProfileCreateRequest) -> ProfileCreateReceipt:
    from hermes_cli.profile_provision import write_alias
    try:
        path, collision = write_alias(request.name)
    except Exception as exc:  # the profile exists: a wrapper failure is a warning, as for a local create
        logger.warning("profile %s: alias wrapper failed", request.name, exc_info=True)
        return dataclasses.replace(made, warnings=[*made.warnings, f"alias wrapper: {exc}"])
    return dataclasses.replace(made, alias_path=str(path) if path else None, alias_collision=collision)
