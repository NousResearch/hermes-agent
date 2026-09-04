"""RoomLink room-member grants and capability HTTP handlers."""

import asyncio
import time
import uuid
from typing import Any, Optional

try:
    from aiohttp import web
except ImportError:
    web = None  # type: ignore[assignment]


class RoomGrantReauthorizationRequired(ValueError):
    """A validly signed room grant was revoked or superseded."""


def _json_error(_openai_error, message: str, *, status: int, **error_kwargs) -> "web.Response":
    """JSON error response built with the injected ``_openai_error`` envelope builder."""
    return web.json_response(_openai_error(message, **error_kwargs), status=status)


def _require_unchanged_execution_policy(claims: dict[str, Any], execution_policy: dict[str, Any]) -> None:
    """Keep renewal from silently granting a changed execution policy."""
    if str(execution_policy.get("policy_digest") or "") != str(claims.get("execution_policy_digest") or ""):
        raise RoomGrantReauthorizationRequired("room execution policy changed")


def _room_grant_error_response(exc: Optional[Exception] = None, *, _openai_error) -> "web.Response":
    """401 invalid grant, or 403 reauthorization-required for a revoked/superseded grant."""
    if isinstance(exc, RoomGrantReauthorizationRequired):
        message, code, status = "Room authorization needs to be renewed.", "room_reauthorization_required", 403
    else:
        message, code, status = "Room authorization is invalid or expired.", "invalid_room_grant", 401
    return _json_error(_openai_error, message, err_type="gateway_auth_error", code=code, status=status)


def _hard_expiry(claims: dict[str, Any]) -> float:
    return float(claims.get("status_expires_at", claims["expires_at"]))


_ROOM_IDENTITY_FIELDS = ("room_id", "home_install_id", "authority_gateway_id", "authority_epoch", "member_id")


def _room_identity(source: dict[str, Any], *, coerce: bool = False) -> dict[str, Any]:
    """Room-authority kwargs for ``issue_room_grant``; *coerce* applies ``str``/``int`` to raw body values."""
    text = str if coerce else (lambda v: v)
    return {k: int(source[k]) if k == "authority_epoch" else text(source[k]) for k in _ROOM_IDENTITY_FIELDS}


def _effective_room_profile(_api_request_profile) -> str:
    """Resolve the middleware selection or the daemon's active profile scope."""
    from hermes_cli.profiles import get_active_profile_name, profile_matches_home

    selected = _api_request_profile.get()
    if selected:
        return selected
    active = get_active_profile_name()
    return "default" if active == "custom" and not profile_matches_home(active) else active


def _local_target(claims: dict[str, Any] | None, _api_request_profile) -> tuple[str, str]:
    """Return ``(profile, installation_id)`` for this gateway; *claims* must target it."""
    from gateway import hosted_rooms
    profile = _effective_room_profile(_api_request_profile)
    installation_id = hosted_rooms.local_authority_gateway_id()
    if claims is not None and (claims["target_profile"], claims["target_install_id"]) != (profile, installation_id):
        raise ValueError("room grant target does not match this profile")
    return profile, installation_id


def _local_room_catalog(self, profile: str, installation_id: str) -> tuple[dict, dict]:
    """Return ``(execution_policy, catalog)`` for this gateway's *profile*."""
    from gateway.hosted_room_peer import PROTOCOL_VERSION, catalog_mapping
    from gateway.hosted_room_execution_policy import execution_policy_mapping
    from gateway.platforms.api_server_room_attachments import roomlink_attachments_available
    with self._profile_scope(profile):
        execution_policy = execution_policy_mapping(target_profile=profile)
    catalog = catalog_mapping(
        installation_id=installation_id, protocol_versions=(PROTOCOL_VERSION,), link_modes=("direct",),
        persistent_process=True, text=True, attachments=roomlink_attachments_available(), target_profile=profile,
        execution_policy=execution_policy)
    return execution_policy, catalog


def _http_routes(self) -> list[tuple[str, str, Any]]:
    from gateway.platforms import api_server_room_attachments, api_server_room_artifacts, api_server_room_controls

    async def revoke_exact(request):
        from gateway.platforms import api_server

        return await _handle_room_member_grant_revoke_exact(
            self,
            request,
            _openai_error=api_server._openai_error,
            _api_request_profile=api_server._api_request_profile,
        )

    return [
        (
            "POST",
            "/v1/room-members/invitations",
            self._handle_room_member_invitation,
        ),
        (
            "GET",
            "/v1/room-members/capabilities",
            self._handle_room_member_capabilities,
        ),
        (
            "POST",
            "/v1/room-members/grants/refresh",
            self._handle_room_member_grant_refresh,
        ),
        (
            "POST",
            "/v1/room-members/grants/revoke",
            self._handle_room_member_grant_revoke,
        ),
        ("POST", "/v1/room-members/grants/revoke-exact", revoke_exact),
    ] + api_server_room_controls._http_routes(self) + api_server_room_attachments._http_routes(self) + api_server_room_artifacts._http_routes(self)


def _room_grant_token(request: "web.Request") -> str:
    scheme, separator, token = str(request.headers.get("Authorization") or "").partition(" ")
    return token.strip() if separator and scheme.lower() == "hermesroom" else ""


def _room_grant_secret(self) -> bytes:
    from gateway.hosted_room_peer import gateway_room_grant_secret
    return gateway_room_grant_secret()


def _decode_request_grant(
    self, request: "web.Request", *, permission: str, allow_expired_for_revocation: bool = False,
) -> dict[str, Any]:
    """Signature/scope/horizon check only (no revocation lookup)."""
    from gateway.hosted_room_peer import decode_room_grant
    token = self._room_grant_token(request)
    if not token:
        raise ValueError("room grant is missing")
    return decode_room_grant(
        self._room_grant_secret(), token, permission=permission,
        allow_expired_for_revocation=allow_expired_for_revocation,
    )


def _room_grant_claims(self, request: "web.Request", *, permission: str) -> dict[str, Any]:
    claims = _decode_request_grant(self, request, permission=permission)
    from gateway import hosted_rooms
    db_path = hosted_rooms.default_db_path()
    if hosted_rooms.room_grant_is_revoked(db_path, claims=claims):
        raise RoomGrantReauthorizationRequired("room grant is revoked")
    if not hosted_rooms.peer_room_grant_is_current(db_path, claims=claims):
        raise RoomGrantReauthorizationRequired("room grant is no longer current")
    return claims


async def _handle_room_member_invitation(
    self,
    request: "web.Request",
    *,
    _openai_error,
    _api_request_profile,
) -> "web.Response":
    """Mint a short-lived room/profile grant for a trusted home gateway."""
    auth_err = self._check_auth(request)
    if auth_err:
        return auth_err
    body, error = await self._read_json_body(request)
    if error:
        return error
    required = {
        "room_id",
        "home_install_id",
        "authority_gateway_id",
        "authority_epoch",
        "member_id",
    }
    allowed = required | {"grant_id", "ttl_seconds", "status_ttl_seconds"}
    if set(body) - allowed or not required <= set(body):
        return web.json_response(
            _openai_error(
                "Invitation is missing required room authority fields.",
                code="invalid_room_invitation",
            ),
            status=400,
        )
    try:
        from gateway import hosted_rooms
        from gateway.hosted_room_peer import (
            decode_room_grant,
            issue_room_grant,
        )

        profile, target_install_id = _local_target(None, _api_request_profile)
        ttl = float(body.get("ttl_seconds", 3600))
        if not 60 <= ttl <= 24 * 60 * 60:
            raise ValueError("ttl_seconds must be between 60 and 86400")
        status_ttl = float(body.get("status_ttl_seconds", ttl))
        if not ttl <= status_ttl <= 30 * 24 * 60 * 60:
            raise ValueError(
                "status_ttl_seconds must be at least ttl_seconds and no more than 2592000"
            )
        execution_policy, catalog = _local_room_catalog(
            self, profile, target_install_id
        )
        token = issue_room_grant(
            self._room_grant_secret(),
            grant_id=str(body.get("grant_id") or f"grant-{uuid.uuid4().hex}"),
            room_id=str(body["room_id"]),
            home_install_id=str(body["home_install_id"]),
            authority_gateway_id=str(body["authority_gateway_id"]),
            authority_epoch=int(body["authority_epoch"]),
            member_id=str(body["member_id"]),
            target_install_id=target_install_id,
            target_profile=profile,
            execution_policy_digest=execution_policy["policy_digest"],
            issued_at=time.time(),
            ttl_seconds=ttl,
            status_ttl_seconds=status_ttl,
        )
        claims = decode_room_grant(
            self._room_grant_secret(), token, permission="status"
        )
        from gateway.hosted_room_grant_state import (
            grant_state_db_paths,
            reserve_grant_state,
        )

        reserve_grant_state(
            grant_state_db_paths(),
            claims=claims,
            expires_at=float(claims.get("status_expires_at", claims["expires_at"])),
        )
    except Exception as exc:
        return web.json_response(
            _openai_error(str(exc), code="invalid_room_invitation"),
            status=400,
        )
    return web.json_response(
        {
            "object": "hermes.room_member.invitation",
            "grant": token,
            "target_profile": profile,
            "catalog": catalog,
            "expires_at": float(claims["expires_at"]),
            "status_expires_at": float(claims["status_expires_at"]),
        },
        status=201,
    )


async def _handle_room_member_capabilities(
    self, request: "web.Request", *, _openai_error, _api_request_profile) -> "web.Response":
    """Verify a scoped grant and return this target's live room catalog."""
    try:
        claims = self._room_grant_claims(request, permission="status")
        profile, installation_id = _local_target(claims, _api_request_profile)
        _, catalog = _local_room_catalog(self, profile, installation_id)
    except Exception as exc:
        return _room_grant_error_response(exc, _openai_error=_openai_error)
    return web.json_response({
        "object": "hermes.room_member.capabilities", **{k: claims[k] for k in _ROOM_IDENTITY_FIELDS},
        "target_profile": profile, "catalog": catalog})


async def _handle_room_member_grant_refresh(
    self,
    request: "web.Request",
    *,
    _openai_error,
    _api_request_profile,
) -> "web.Response":
    """Refresh dispatch access without a Desktop or broad gateway key."""
    body, error = await self._read_json_body(request)
    if error:
        return error
    if set(body) - {"ttl_seconds"}:
        return web.json_response(
            _openai_error(
                "Grant refresh accepts only ttl_seconds.",
                code="invalid_room_grant_refresh",
            ),
            status=400,
        )
    try:
        from gateway import hosted_rooms
        from gateway.hosted_room_peer import (
            MAX_DISPATCH_GRANT_TTL_SECONDS,
            issue_room_grant,
        )
        from gateway.hosted_room_execution_policy import execution_policy_mapping

        # A status-only bearer may observe a run but must never mint new
        # dispatch authority. Renewal is possible only while the existing
        # dispatch permission is still live.
        claims = self._room_grant_claims(request, permission="dispatch")
        profile = _effective_room_profile(_api_request_profile)
        installation_id = hosted_rooms.local_authority_gateway_id()
        if (
            claims["target_profile"] != profile
            or claims["target_install_id"] != installation_id
        ):
            raise ValueError("room grant target does not match this profile")
        now = time.time()
        hard_expiry = float(
            claims.get("status_expires_at", claims["expires_at"])
        )
        remaining = hard_expiry - now
        requested = float(
            body.get("ttl_seconds", MAX_DISPATCH_GRANT_TTL_SECONDS)
        )
        if remaining <= 0 or requested <= 0:
            raise ValueError("room grant renewal horizon expired")
        dispatch_ttl = min(
            requested,
            MAX_DISPATCH_GRANT_TTL_SECONDS,
            remaining,
        )
        with self._profile_scope(profile):
            execution_policy = execution_policy_mapping(target_profile=profile)
        _require_unchanged_execution_policy(claims, execution_policy)
        token = issue_room_grant(
            self._room_grant_secret(),
            grant_id=f"grant-refresh-{uuid.uuid4().hex}",
            room_id=claims["room_id"],
            home_install_id=claims["home_install_id"],
            authority_gateway_id=claims["authority_gateway_id"],
            authority_epoch=int(claims["authority_epoch"]),
            member_id=claims["member_id"],
            target_install_id=installation_id,
            target_profile=profile,
            execution_policy_digest=execution_policy["policy_digest"],
            permissions=claims["permissions"],
            issued_at=now,
            ttl_seconds=dispatch_ttl,
            status_expires_at=hard_expiry,
        )
        # Close the refresh-versus-retirement race. If revocation landed after
        # the first authorization check, never return the replacement. If it
        # lands after this check, its scope timestamp also covers this token.
        self._room_grant_claims(request, permission="dispatch")
    except Exception as exc:
        return _room_grant_error_response(exc, _openai_error=_openai_error)
    return web.json_response(
        {
            "object": "hermes.room_member.grant",
            "grant": token,
            "expires_at": now + dispatch_ttl,
            "status_expires_at": hard_expiry,
            "execution_policy": execution_policy,
        }
    )


async def _handle_room_member_grant_revoke(
    self,
    request: "web.Request",
    *,
    _openai_error,
    _api_request_profile,
) -> "web.Response":
    """Retire the room scope authenticated by this grant."""
    body, error = await self._read_json_body(request)
    if error:
        return error
    if body:
        return web.json_response(
            _openai_error(
                "Grant revoke accepts no fields.",
                code="invalid_room_grant_revoke",
            ),
            status=400,
        )
    try:
        from gateway import hosted_rooms
        from gateway.hosted_room_peer import decode_room_grant

        token = self._room_grant_token(request)
        if not token:
            raise ValueError("room grant is missing")
        # Revoke is idempotent: a response-lost retry may authenticate with
        # the grant that was just added to the denylist. Verify signature,
        # scope, and hard horizon directly, then upsert the scope fence.
        claims = decode_room_grant(
            self._room_grant_secret(),
            token,
            permission="status",
            allow_expired_for_revocation=True,
        )
        profile = _effective_room_profile(_api_request_profile)
        installation_id = hosted_rooms.local_authority_gateway_id()
        if (
            claims["target_profile"] != profile
            or claims["target_install_id"] != installation_id
        ):
            raise ValueError("room grant target does not match this profile")
    except Exception as exc:
        return _room_grant_error_response(exc, _openai_error=_openai_error)
    try:
        from gateway.hosted_room_grant_state import (
            grant_state_db_paths,
            revoke_grant_state,
        )

        revoke_grant_state(
            grant_state_db_paths(),
            claims=claims,
            expires_at=float(
                claims.get("status_expires_at", claims["expires_at"])
            ),
        )
        try:
            from gateway.platforms.api_server_room_attachments import _default_spool
            await asyncio.to_thread(_default_spool().discard_scope, claims)
        except Exception:
            # Authorization is already revoked; bounded expiry backs up failed cleanup.
            pass
        try:
            from gateway.hosted_room_artifacts import RoomArtifactOutbox
            from hermes_constants import get_hermes_home

            await asyncio.to_thread(RoomArtifactOutbox(get_hermes_home() / "state.db").discard_claims, claims)
        except Exception:
            # Revocation closes access even if private-byte cleanup must wait.
            pass
    except Exception:
        return _json_error(
            _openai_error, "Room grant revocation could not be persisted; retry required.",
            code="room_grant_revocation_unavailable", status=503,
        )
    try:
        from gateway.hosted_room_control_client import revoke_stored_peer_control

        await asyncio.to_thread(
            revoke_stored_peer_control, hosted_rooms.default_db_path(),
            room_id=str(claims["room_id"]), member_id=str(claims["member_id"]),
        )
    except Exception:
        return _json_error(
            _openai_error, "Room control cleanup is pending; retry this revocation.",
            code="room_control_cleanup_pending", status=503,
        )
    return web.json_response({"object": "hermes.room_member.grant.revocation", "revoked": True})


async def _handle_room_member_grant_revoke_exact(
    self,
    request: "web.Request",
    *,
    _openai_error,
    _api_request_profile,
) -> "web.Response":
    """Retire only an unpublished bearer without disrupting its replacement."""
    body, error = await self._read_json_body(request)
    if error:
        return error
    if body:
        return web.json_response(
            _openai_error(
                "Grant revoke accepts no fields.", code="invalid_room_grant_revoke"
            ),
            status=400,
        )
    try:
        from gateway import hosted_rooms
        from gateway.hosted_room_peer import decode_room_grant
        from gateway.hosted_room_grant_state import (
            grant_state_db_paths,
            revoke_grant_state,
        )

        token = self._room_grant_token(request)
        if not token:
            raise ValueError("room grant is missing")
        claims = decode_room_grant(
            self._room_grant_secret(),
            token,
            permission="status",
            allow_expired_for_revocation=True,
        )
        if (
            claims["target_profile"] != _effective_room_profile(_api_request_profile)
            or claims["target_install_id"] != hosted_rooms.local_authority_gateway_id()
        ):
            raise ValueError("room grant target does not match this profile")
    except Exception as exc:
        return _room_grant_error_response(exc, _openai_error=_openai_error)
    try:
        revoke_grant_state(
            grant_state_db_paths(),
            claims=claims,
            expires_at=float(claims.get("status_expires_at", claims["expires_at"])),
            exact=True,
        )
    except Exception:
        return web.json_response(
            _openai_error(
                "Room grant revocation could not be persisted; retry required.",
                code="room_grant_revocation_unavailable",
            ),
            status=503,
        )
    return web.json_response({
        "object": "hermes.room_member.grant.revocation",
        "revoked": True,
    })
