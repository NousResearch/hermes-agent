"""``hermes profile create`` through the running gateway: one ``profiles.create`` RPC, one receipt.

The CLI asks the gateway that serves its profile to make the profile (``HERMES_TUI_GATEWAY_URL``, the dial
``connect_gateway`` honours for every client, names a remote one), so a terminal attached to a gateway
creates profiles on the gateway's machine. :func:`receipt` is what every caller prints, read where the
profile was made; the gateway returns it from ``profiles.create`` when asked (``report``), and a local
create builds the same one, so the output does not depend on where the profile was made.
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

from hermes_cli.profile_provision import ProfileCreateRequest, ProvisionResult

logger = logging.getLogger(__name__)

#: Skill seeding runs a subprocess (60 s cap) and the multiplexer rescan answer waits up to 8 s.
CREATE_RPC_TIMEOUT = 300
_ERROR_TYPES = {"FileExistsError": FileExistsError, "FileNotFoundError": FileNotFoundError, "ValueError": ValueError}


class GatewayCreateError(RuntimeError):
    """The named gateway cannot take the create, or took it and its outcome is unknown."""


def error_data(exc: BaseException) -> dict:
    """``error.data`` for a refused create: the client re-raises the same type with the same text,
    redacted (a parse error can quote a line of the source's config/.env) and capped
    (``gateway_client`` keeps only short scalar strings)."""
    from agent.redact import redact_sensitive_text
    return {"type": type(exc).__name__,
            "message": redact_sensitive_text(str(exc), force=True, redact_url_credentials=True)[:200]}


def receipt(request: ProfileCreateRequest, result: ProvisionResult) -> dict[str, Any]:
    """What a surface prints about the new profile, read on the machine (and in the profile scope)
    that made it. Each read is best-effort: the profile exists whatever a read says."""
    from hermes_cli import profiles
    path = result.path
    try:
        display = "~/" + path.relative_to(Path.home()).as_posix()
    except ValueError:
        display = str(path)
    launch = profiles.get_active_profile_name()
    out: dict[str, Any] = {
        "path": str(path), "display_path": display, "launch_label": launch,
        "source_label": request.clone_from or launch, "skills_seeded": result.skills_seeded,
        "alias_path": str(result.alias_path) if result.alias_path else None,
        "alias_collision": result.alias_collision, "mirrored": result.mirrored, "warnings": result.warnings,
        "cloned_plugins": [], "memory_provider": None, "channels_left_behind": [], "shared_channels": [],
        "served_now": None,
    }
    if not request.fresh:
        if not request.clone_all:
            from hermes_cli.profile_memory_config import cloned_memory_provider
            out["cloned_plugins"] = _try(lambda: profiles.cloned_plugin_names(path), [])
            out["memory_provider"] = _try(lambda: cloned_memory_provider(path), None)
        source_dir = profiles.get_profile_dir(out["source_label"])
        if source_dir.is_dir():
            from hermes_cli.profile_channels import channel_platforms_configured, shared_channel_credentials
            if request.clone_channels:
                out["shared_channels"] = _try(lambda: shared_channel_credentials(path, source_dir), [])
            else:
                out["channels_left_behind"] = _try(lambda: channel_platforms_configured(source_dir), [])
    from hermes_cli.gateway_multiplex_served import live_default_gateway_pid, recorded_served_profiles
    served = _try(lambda: recorded_served_profiles() if live_default_gateway_pid() is not None else None, None)
    if served is not None:
        canon = profiles.normalize_profile_name(request.name)
        out["served_now"] = canon in {profiles.normalize_profile_name(p) for p in served}
    return out


#: Codes answered before the handler runs: nothing was made there. 4000 / -32601: a gateway that
#: predates these fields; 5035: one retiring between discovery and the request.
_NOT_RUN = (4000, -32601, 5035)


def _wire(request: ProfileCreateRequest) -> dict[str, Any]:
    # ``no_alias`` always: the wrapper belongs to the terminal's machine, PATH and ``hermes``
    # (``write_alias`` runs here after a local gateway's create; a URL gateway's profile gets none).
    return {"name": request.name, "description": request.description, "clone_from": request.clone_from,
            "clone_all": request.clone_all, "clone_config": request.clone_config,
            "clone_channels": request.clone_channels, "no_skills": request.no_skills,
            "no_alias": True, "sync_imports": request.sync_imports,
            "mirror_credentials": request.mirror_credentials, "share_auth": request.share_auth, "report": True}


def _where(remote: str) -> str:
    """The gateway as a message names it: ``scheme://host[:port]`` only. The query holds its token, the
    userinfo a password, and a URL without a host (``wss:user:pw@h``) puts its authority in the path;
    one that does not parse is not named at all (``urlsplit``'s error quotes the authority)."""
    if not remote:
        return "the running gateway"
    try:
        parts = urlsplit(remote)
        port = parts.port
    except ValueError:
        parts, port = None, None
    if parts is None or parts.scheme not in ("ws", "wss", "http", "https") or not parts.hostname:
        return "the gateway HERMES_TUI_GATEWAY_URL names"
    return f"the gateway at {parts.scheme}://{parts.hostname}{f':{port}' if port else ''}"


def _check(remote: str, command: str) -> str:
    """Where to look before retrying: ``profile list`` / ``show`` read this machine's profiles."""
    return f"`{command}` on the gateway's machine" if remote else f"`{command}`"


def create_through_gateway(request: ProfileCreateRequest) -> Optional[dict[str, Any]]:
    """The receipt of the gateway's create for *request*, or None to create here.

    ``HERMES_TUI_GATEWAY_URL`` set: that gateway makes the profile, and anything short of its answer is
    a :class:`GatewayCreateError`, never a local create. Otherwise a live gateway for this profile
    (discovery only; one is never started for this) makes it; with none, or one that predates the
    fields this sends, the caller creates locally. A refused create raises the gateway's own
    ``ValueError`` / ``FileExistsError`` / ``FileNotFoundError``.
    """
    from hermes_cli.gateway_client import GatewayClientError, GatewayRPCError, NOT_SENT, connect_gateway
    from hermes_cli.sessions_owner import live_owner
    remote = os.environ.get("HERMES_TUI_GATEWAY_URL", "").strip()
    endpoint = None
    if not remote:
        endpoint = _try(live_owner, None)
        if endpoint is None:
            return None
    where = _where(remote)
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
        raise GatewayCreateError(f"interrupted; {where} may still finish the create. Check "
                                 f"{_check(remote, 'hermes profile list')} before retrying") from None
    except GatewayRPCError as exc:
        if exc.code in _NOT_RUN or str(exc) == "unknown_method":
            if remote:
                why = "is restarting; retry" if exc.code == 5035 else (
                    "predates this version; update it, or unset HERMES_TUI_GATEWAY_URL to create locally")
                raise GatewayCreateError(f"{where} cannot create profiles for this terminal (it {why})") from exc
            return None
        kind = _ERROR_TYPES.get(str(exc.data.get("type")))
        if kind is not None:
            raise kind(exc.data.get("message") or str(exc)) from exc
        raise GatewayCreateError(f"{where} could not create the profile: {exc.data.get('message') or exc}") from exc
    except Exception as exc:
        if not attached or str(exc) == NOT_SENT:  # the dial, ticket or handshake failed: nothing was sent
            if remote:
                raise GatewayCreateError(f"could not reach {where} ({type(exc).__name__})") from exc
            logger.info("profile create: the running gateway is unreachable (%s); creating locally", exc)
            return None
        if isinstance(exc, (GatewayClientError, TimeoutError, OSError)):
            raise GatewayCreateError(f"lost {where} during the create; check "
                                     f"{_check(remote, 'hermes profile list')} before retrying") from exc
        raise
    made = (answer or {}).get("receipt")
    if not isinstance(made, dict) or not made.get("path"):
        # It made a profile, without these fields (``--clone`` / ``--sync-imports`` may be ignored).
        raise GatewayCreateError(f"{where} created '{request.name}' but predates this version; check it "
                                 f"with {_check(remote, f'hermes profile show {request.name}')}")
    return {**receipt_defaults(), **made}


def receipt_defaults() -> dict[str, Any]:
    return {"display_path": "", "launch_label": "", "source_label": "", "skills_seeded": None,
            "alias_path": None, "alias_collision": None, "mirrored": {}, "warnings": [], "cloned_plugins": [],
            "memory_provider": None, "channels_left_behind": [], "shared_channels": [], "served_now": None}


def _try(fn, default):
    try:
        return fn()
    except Exception:
        logger.debug("profile create receipt read failed", exc_info=True)
        return default
