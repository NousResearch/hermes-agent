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
    ProfileCreateReceipt, ProfileCreateRequest, redact_create_text, refusal_from,
)

logger = logging.getLogger(__name__)

#: Skill seeding runs a subprocess (60 s cap) and the multiplexer rescan answer waits up to 8 s.
CREATE_RPC_TIMEOUT = 300
#: Answered before the handler runs, so nothing was made there. 4000: a gateway that predates these
#: fields (unknown param); 5035: one retiring between discovery and the request.
_OLDER_OR_RETIRING = (4000, 5035)
#: Refused before the create (no name). Not 4064: the dispatcher also answers it for a config outage
#: raised mid-handler, after the profile may have been made.
_REFUSED_BEFORE_CREATE = (4061,)


class GatewayCreateError(RuntimeError):
    """The named gateway cannot take the create, or took it and its outcome is unknown."""


def _wire(request: ProfileCreateRequest) -> dict[str, Any]:
    # ``no_alias`` always: the wrapper belongs to the terminal's machine, PATH and ``hermes``
    # (written here when the profile landed in this machine's profiles directory).
    return {"name": request.name, "description": request.description, "clone_from": request.clone_from,
            "clone_all": request.clone_all, "clone_config": request.clone_config,
            "clone_channels": request.clone_channels, "no_skills": request.no_skills,
            "no_alias": True, "sync_imports": request.sync_imports,
            "mirror_credentials": request.mirror_credentials, "share_auth": request.share_auth, "report": True}


def _where(remote: str) -> str:
    """How messages name the gateway: ``scheme://host[:port]`` only (``safe_proxy_url_for_log``: never the
    query's token, the userinfo's password, or a password an unencoded ``/`` pushed into the path). A URL
    ``connect_gateway`` would refuse (not ws/wss with a host) is a :class:`GatewayCreateError` here."""
    if not remote:
        return "the running gateway"
    from gateway.platforms.base_log_safety import safe_proxy_url_for_log  # a URL only: it loads gateway.platforms
    try:
        parts = urlsplit(remote)
        ok = bool(parts.hostname) and (parts.port is None or parts.port > 0)
    except ValueError:  # an out-of-range port or a malformed authority; its message quotes the authority
        ok = False
    named = safe_proxy_url_for_log(remote) if ok else ""
    if not named.startswith(("ws://", "wss://")):
        raise GatewayCreateError("HERMES_TUI_GATEWAY_URL is not a ws:// or wss:// URL with a host; nothing was created")
    return f"the gateway at {named}"


def _is_this_machines_profile(made: ProfileCreateReceipt, request: ProfileCreateRequest) -> bool:
    """Whether the profile the gateway made is the one ``-p <name>`` selects here: same directory, and it
    exists. A loopback URL is not enough (a tunnel or a container can serve another home)."""
    from hermes_cli.profiles import profile_matches_home
    return profile_matches_home(request.name, Path(made.path)) and Path(made.path).is_dir()


def create_through_gateway(request: ProfileCreateRequest) -> Optional[ProfileCreateReceipt]:
    """The receipt of the gateway's create for *request*, or None to create here.

    ``HERMES_TUI_GATEWAY_URL`` set: that gateway makes the profile, and anything short of its answer is
    a :class:`GatewayCreateError`, never a local create. Otherwise a live gateway for this profile
    (discovery only; one is never started for this) makes it; with none, or one that predates the
    fields this sends, the caller creates locally. A refused create raises the gateway's own
    ``ValueError`` / ``FileExistsError`` / ``FileNotFoundError``. When the profile landed in this
    machine's profiles directory, the terminal's ``~/.local/bin`` wrapper is written here.
    """
    from hermes_cli.gateway_client import GatewayRPCError, NOT_SENT, connect_gateway
    from hermes_cli.sessions_owner import live_owner
    remote = os.environ.get("HERMES_TUI_GATEWAY_URL", "").strip()
    where = _where(remote)
    endpoint = None
    if not remote:
        try:
            endpoint = live_owner()
        except Exception:
            logger.debug("profile create: gateway discovery failed; creating locally", exc_info=True)
        if endpoint is None:
            return None
    machine = " on the gateway's machine" if remote else ""
    unknown = f"{where} may have created '{request.name}'; check `hermes profile list`{machine} before retrying"
    attached = False

    async def _run():
        nonlocal attached
        async with connect_gateway(endpoint) as client:
            attached = True
            return await client.rpc("profiles.create", _timeout=CREATE_RPC_TIMEOUT, **_wire(request))

    try:
        answer = asyncio.run(_run())
    except KeyboardInterrupt:
        if not attached:
            raise
        raise GatewayCreateError(f"interrupted; {unknown}") from None
    except GatewayRPCError as exc:
        if exc.code in _OLDER_OR_RETIRING or exc.unknown_method:
            if remote:
                why = "is restarting; retry" if exc.code == 5035 else (
                    "predates this version; update it, or unset HERMES_TUI_GATEWAY_URL to create locally")
                raise GatewayCreateError(f"{where} cannot create profiles for this terminal (it {why})") from exc
            return None
        refused = refusal_from(exc.data)
        if refused is not None:
            raise refused from exc
        reason = redact_create_text(exc)
        if exc.code in _REFUSED_BEFORE_CREATE:
            raise GatewayCreateError(f"{where} refused the create ({reason}); nothing was created") from exc
        # Not a refusal of the request: the handler failed, possibly after the profile was made.
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
    return _with_local_alias(made, request) if request.alias and _is_this_machines_profile(made, request) else made


def _with_local_alias(made: ProfileCreateReceipt, request: ProfileCreateRequest) -> ProfileCreateReceipt:
    from hermes_cli.profile_provision import write_alias
    try:
        path, collision = write_alias(request.name)
    except Exception as exc:  # the profile exists: a wrapper failure is a warning, as for a local create
        logger.warning("profile %s: alias wrapper failed", request.name, exc_info=True)
        return dataclasses.replace(made, warnings=[*made.warnings, f"alias wrapper: {exc}"])
    return dataclasses.replace(made, alias_path=str(path) if path else None, alias_collision=collision)
