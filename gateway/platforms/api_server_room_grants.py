"""RoomLink room-member grants and capability HTTP handlers."""

import asyncio
import time
import uuid
from typing import Any

try:
    from aiohttp import web
except ImportError:
    web = None  # type: ignore[assignment]


class RoomGrantReauthorizationRequired(ValueError):
    """A validly signed room grant was revoked or superseded."""


def _require_unchanged_execution_policy(
    claims: dict[str, Any],
    execution_policy: dict[str, Any],
) -> None:
    """Keep renewal from silently granting a changed execution policy."""
    if (
        str(execution_policy.get("policy_digest") or "")
        != str(claims.get("execution_policy_digest") or "")
    ):
        raise RoomGrantReauthorizationRequired(
            "room execution policy changed"
        )


def _room_grant_error_response(exc: Exception, *, _openai_error) -> "web.Response":
    reauthorization = isinstance(exc, RoomGrantReauthorizationRequired)
    return web.json_response(
        _openai_error(
            (
                "Room authorization needs to be renewed."
                if reauthorization
                else "Room authorization is invalid or expired."
            ),
            err_type="gateway_auth_error",
            code=(
                "room_reauthorization_required"
                if reauthorization
                else "invalid_room_grant"
            ),
        ),
        status=403 if reauthorization else 401,
    )


def _http_routes(self) -> list[tuple[str, str, Any]]:
    routes = [
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
    ]
    from gateway.platforms import api_server_room_controls

    routes.extend(api_server_room_controls._http_routes(self))
    return routes


def _room_grant_token(request: "web.Request") -> str:
    authorization = str(request.headers.get("Authorization") or "")
    scheme, separator, token = authorization.partition(" ")
    if not separator or scheme.lower() != "hermesroom":
        return ""
    return token.strip()


def _room_grant_secret(self) -> bytes:
    from gateway.hosted_room_peer import gateway_room_grant_secret

    return gateway_room_grant_secret()


def _room_grant_claims(
    self,
    request: "web.Request",
    *,
    permission: str,
) -> dict[str, Any]:
    from gateway.hosted_room_peer import decode_room_grant

    token = self._room_grant_token(request)
    if not token:
        raise ValueError("room grant is missing")
    claims = decode_room_grant(
        self._room_grant_secret(),
        token,
        permission=permission,
    )
    from gateway import hosted_rooms

    if hosted_rooms.room_grant_is_revoked(
        hosted_rooms.default_db_path(),
        claims=claims,
    ):
        raise RoomGrantReauthorizationRequired("room grant is revoked")
    if not hosted_rooms.peer_room_grant_is_current(
        hosted_rooms.default_db_path(),
        claims=claims,
    ):
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
            PROTOCOL_VERSION as ROOM_LINK_PROTOCOL_VERSION,
            catalog_mapping,
            decode_room_grant,
            issue_room_grant,
        )
        from gateway.hosted_room_execution_policy import execution_policy_mapping

        profile = _api_request_profile.get() or "default"
        target_install_id = hosted_rooms.local_authority_gateway_id()
        ttl = float(body.get("ttl_seconds", 3600))
        if not 60 <= ttl <= 24 * 60 * 60:
            raise ValueError("ttl_seconds must be between 60 and 86400")
        status_ttl = float(body.get("status_ttl_seconds", ttl))
        if not ttl <= status_ttl <= 30 * 24 * 60 * 60:
            raise ValueError(
                "status_ttl_seconds must be at least ttl_seconds and no more than 2592000"
            )
        with self._profile_scope(profile):
            execution_policy = execution_policy_mapping(target_profile=profile)
        catalog = catalog_mapping(
            installation_id=target_install_id,
            protocol_versions=(ROOM_LINK_PROTOCOL_VERSION,),
            link_modes=("direct",),
            persistent_process=True,
            text=True,
            attachments=False,
            target_profile=profile,
            execution_policy=execution_policy,
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
        hosted_rooms.reserve_peer_room(
            hosted_rooms.default_db_path(),
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
    self,
    request: "web.Request",
    *,
    _openai_error,
    _api_request_profile,
) -> "web.Response":
    """Verify a scoped grant and return this target's live room catalog."""
    try:
        from gateway import hosted_rooms
        from gateway.hosted_room_peer import (
            PROTOCOL_VERSION as ROOM_LINK_PROTOCOL_VERSION,
            catalog_mapping,
        )
        from gateway.hosted_room_execution_policy import execution_policy_mapping

        claims = self._room_grant_claims(request, permission="status")
        profile = _api_request_profile.get() or "default"
        installation_id = hosted_rooms.local_authority_gateway_id()
        if (
            claims["target_profile"] != profile
            or claims["target_install_id"] != installation_id
        ):
            raise ValueError("room grant target does not match this profile")
        with self._profile_scope(profile):
            execution_policy = execution_policy_mapping(target_profile=profile)
        catalog = catalog_mapping(
            installation_id=installation_id,
            protocol_versions=(ROOM_LINK_PROTOCOL_VERSION,),
            link_modes=("direct",),
            persistent_process=True,
            text=True,
            attachments=False,
            target_profile=profile,
            execution_policy=execution_policy,
        )
    except Exception as exc:
        return _room_grant_error_response(exc, _openai_error=_openai_error)
    return web.json_response(
        {
            "object": "hermes.room_member.capabilities",
            "room_id": claims["room_id"],
            "home_install_id": claims["home_install_id"],
            "authority_gateway_id": claims["authority_gateway_id"],
            "authority_epoch": claims["authority_epoch"],
            "member_id": claims["member_id"],
            "target_profile": profile,
            "catalog": catalog,
        }
    )


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
        profile = _api_request_profile.get() or "default"
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
    """Revoke exactly the scoped grant authenticating this request."""
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
        # scope, and hard horizon directly, then upsert the same grant id.
        claims = decode_room_grant(
            self._room_grant_secret(),
            token,
            permission="status",
            allow_expired_for_revocation=True,
        )
        profile = _api_request_profile.get() or "default"
        installation_id = hosted_rooms.local_authority_gateway_id()
        if (
            claims["target_profile"] != profile
            or claims["target_install_id"] != installation_id
        ):
            raise ValueError("room grant target does not match this profile")
        hosted_rooms.revoke_room_grant_scope(
            hosted_rooms.default_db_path(),
            claims=claims,
            expires_at=float(
                claims.get("status_expires_at", claims["expires_at"])
            ),
        )
    except Exception:
        return web.json_response(
            _openai_error(
                "Room authorization is invalid or expired.",
                err_type="gateway_auth_error",
                code="invalid_room_grant",
            ),
            status=401,
        )
    try:
        from gateway.hosted_room_control_client import revoke_stored_peer_control

        await asyncio.to_thread(
            revoke_stored_peer_control,
            hosted_rooms.default_db_path(),
            room_id=str(claims["room_id"]),
            member_id=str(claims["member_id"]),
        )
    except Exception:
        return web.json_response(
            _openai_error(
                "Room control cleanup is pending; retry this revocation.",
                code="room_control_cleanup_pending",
            ),
            status=503,
        )
    return web.json_response(
        {
            "object": "hermes.room_member.grant.revocation",
            "revoked": True,
        }
    )
