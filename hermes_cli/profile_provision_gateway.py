"""``hermes profile create`` through the running gateway: one ``profiles.create`` RPC, one receipt.

The CLI asks the gateway that serves its profile to make the profile (``HERMES_TUI_GATEWAY_URL``, the dial
``connect_gateway`` honours for every client, names a remote one), so a terminal attached to a gateway
creates profiles on the gateway's machine. The gateway answers with the same
:class:`~hermes_cli.profile_provision.CreateReceipt` a local create builds, so the output does not depend
on where the profile was made.
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
from typing import Any, Optional
from urllib.parse import urlsplit

from hermes_cli.profile_provision import CreateReceipt, ProfileCreateRequest

logger = logging.getLogger(__name__)

#: Skill seeding runs a subprocess (60 s cap) and the multiplexer rescan answer waits up to 8 s.
CREATE_RPC_TIMEOUT = 300
_ERROR_TYPES = {"FileExistsError": FileExistsError, "FileNotFoundError": FileNotFoundError, "ValueError": ValueError}
#: Codes answered before the handler runs: nothing was made there. 4000: a gateway that predates
#: these fields (unknown param); 5035: one retiring between discovery and the request.
_NOT_RUN = (4000, 5035)
#: ``gateway_client`` keeps only short strings and short lists of them in ``error.data``.
_PART, _PARTS = 200, 16


class GatewayCreateError(RuntimeError):
    """The named gateway cannot take the create, or took it and its outcome is unknown."""


def _redact(text: str) -> str:
    from agent.redact import redact_sensitive_text
    return redact_sensitive_text(str(text), force=True, redact_url_credentials=True)


def refusal(exc: BaseException) -> tuple[str, dict]:
    """``(error.message, error.data)`` for a refused create. The message is the whole text, redacted (a
    parse error can quote a line of the source's config/.env); ``data`` names the type and repeats the
    text in the short parts ``gateway_client`` keeps (it drops longer strings whole), so the client
    re-raises the same type with the same text."""
    text = _redact(str(exc))
    return text, {"type": type(exc).__name__,
                  "message_parts": [text[i:i + _PART] for i in range(0, len(text), _PART)][:_PARTS]}


def _refusal_text(data: dict) -> str:
    parts = data.get("message_parts")
    return _redact("".join(p for p in parts if isinstance(p, str))) if isinstance(parts, list) else ""


def _wire(request: ProfileCreateRequest) -> dict[str, Any]:
    # ``no_alias`` always: the wrapper belongs to the terminal's machine, PATH and ``hermes``
    # (written here after a gateway on this machine made the profile).
    return {"name": request.name, "description": request.description, "clone_from": request.clone_from,
            "clone_all": request.clone_all, "clone_config": request.clone_config,
            "clone_channels": request.clone_channels, "no_skills": request.no_skills,
            "no_alias": True, "sync_imports": request.sync_imports,
            "mirror_credentials": request.mirror_credentials, "share_auth": request.share_auth, "report": True}


def _target(remote: str) -> tuple[str, bool]:
    """``(how messages name the gateway, whether it runs on this machine)``. Only ``scheme://host[:port]``
    is ever named: the query holds its token, the userinfo a password, and a URL without a host
    (``wss:user:pw@h``) puts its authority in the path. One that is not a ws/wss URL with a host is a
    :class:`GatewayCreateError` (``connect_gateway`` refuses it too; ``urlsplit``'s error quotes the authority)."""
    from agent.proxy_bypass import is_loopback_host
    if not remote:
        return "the running gateway", True
    try:
        parts = urlsplit(remote)
        port = parts.port
    except ValueError:
        parts, port = None, None
    if parts is None or parts.scheme not in ("ws", "wss") or not parts.hostname:
        raise GatewayCreateError("HERMES_TUI_GATEWAY_URL is not a ws:// or wss:// URL with a host; nothing was created")
    host = f"[{parts.hostname}]" if ":" in parts.hostname else parts.hostname
    return f"the gateway at {parts.scheme}://{host}{f':{port}' if port else ''}", is_loopback_host(parts.hostname)


def create_through_gateway(request: ProfileCreateRequest) -> Optional[CreateReceipt]:
    """The receipt of the gateway's create for *request*, or None to create here.

    ``HERMES_TUI_GATEWAY_URL`` set: that gateway makes the profile, and anything short of its answer is
    a :class:`GatewayCreateError`, never a local create. Otherwise a live gateway for this profile
    (discovery only; one is never started for this) makes it; with none, or one that predates the
    fields this sends, the caller creates locally. A refused create raises the gateway's own
    ``ValueError`` / ``FileExistsError`` / ``FileNotFoundError``. A gateway on this machine gets the
    terminal's ``~/.local/bin`` wrapper written here, with this terminal's PATH and ``hermes``.
    """
    from hermes_cli.gateway_client import GatewayRPCError, NOT_SENT, connect_gateway
    from hermes_cli.profile_provision import _try
    from hermes_cli.sessions_owner import live_owner
    remote = os.environ.get("HERMES_TUI_GATEWAY_URL", "").strip()
    where, on_this_machine = _target(remote)
    endpoint = None
    if not remote:
        endpoint = _try("gateway discovery", live_owner, None)
        if endpoint is None:
            return None
    unknown = (f"{where} may have created '{request.name}'; check "
               f"`hermes profile list`{' on the gateway' + chr(39) + 's machine' if remote else ''} before retrying")
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
        if exc.code in _NOT_RUN or exc.unknown_method:
            if remote:
                why = "is restarting; retry" if exc.code == 5035 else (
                    "predates this version; update it, or unset HERMES_TUI_GATEWAY_URL to create locally")
                raise GatewayCreateError(f"{where} cannot create profiles for this terminal (it {why})") from exc
            return None
        text = _refusal_text(exc.data)
        kind = _ERROR_TYPES.get(str(exc.data.get("type")))
        if kind is not None and text:
            raise kind(text) from exc
        # Not a refusal of the request: the handler failed, possibly after the profile was made.
        raise GatewayCreateError(f"{where} did not finish the create ({text or exc}); {unknown}") from exc
    except Exception as exc:
        if not attached or str(exc) == NOT_SENT:  # the dial, ticket or handshake failed: nothing was sent
            if remote:
                raise GatewayCreateError(f"could not reach {where} ({type(exc).__name__})") from exc
            logger.info("profile create: the running gateway is unreachable (%s); creating locally", exc)
            return None
        raise GatewayCreateError(f"lost {where} during the create ({type(exc).__name__}); {unknown}") from exc
    made = CreateReceipt.from_wire(answer.get("receipt") if isinstance(answer, dict) else None)
    if made is None:
        # It made a profile without a full receipt (``--clone`` / ``--sync-imports`` may be ignored).
        raise GatewayCreateError(f"{where} answered without the details this version needs; {unknown}")
    made = dataclasses.replace(made, warnings=[_redact(w) for w in made.warnings if isinstance(w, str)])
    return _with_local_alias(made, request) if on_this_machine and request.alias else made


def _with_local_alias(made: CreateReceipt, request: ProfileCreateRequest) -> CreateReceipt:
    from hermes_cli.profile_provision import write_alias
    try:
        path, collision = write_alias(request.name)
    except Exception as exc:  # the profile exists: a wrapper failure is a warning, as for a local create
        logger.warning("profile %s: alias wrapper failed", request.name, exc_info=True)
        return dataclasses.replace(made, warnings=[*made.warnings, f"alias wrapper: {exc}"])
    return dataclasses.replace(made, alias_path=str(path) if path else None, alias_collision=collision)
