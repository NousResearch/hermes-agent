"""RoomLink room-member grants and capability HTTP handlers."""

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


def _canonical_room_peer(self, profile: str) -> bool:
    """A missing canonical owner must not fall back to legacy Serve."""
    # Only standalone Serve has no gateway runner. A bound gateway missing its
    # owner is unavailable, never permission to select Serve's executor.
    return self.gateway_runner is not None


def _room_peer_unavailable(self, profile: str, *, _openai_error):
    if _canonical_room_peer(self, profile):
        from gateway.session_peer_target import target_policy
        from hermes_state_runtime import RuntimeStoreError
        try:
            target_policy(self, profile)
        except RuntimeStoreError as exc:
            return _json_error(_openai_error, exc.reason,
                               code='canonical_room_peer_unsupported', status=409)
    return None


def _local_room_catalog(self, profile: str, installation_id: str, *, _connection=None) -> tuple[dict, dict]:
    """One bound target policy for invitation, normalization and execution."""
    from gateway.hosted_room_peer import PROTOCOL_VERSION, catalog_mapping
    from gateway.hosted_room_execution_policy import execution_policy_mapping
    from hermes_state_runtime import RuntimeStoreError
    with self._profile_scope(profile):
        execution_policy = execution_policy_mapping(target_profile=profile)
        text = True
        if _canonical_room_peer(self, profile):
            from gateway.session_peer_target import target_policy
            try:
                _, _, execution_policy = target_policy(self, profile, connection=_connection)
            except RuntimeStoreError:
                text = False
    catalog = catalog_mapping(
        installation_id=installation_id, protocol_versions=(PROTOCOL_VERSION,), link_modes=("direct",),
        persistent_process=True, text=text, attachments=False, target_profile=profile,
        execution_policy=execution_policy)
    return execution_policy, catalog

def _invitation_permissions(self, profile: str, catalog: dict, *, _connection=None) -> tuple[str, ...]:
    """Join input metadata with the Output owner's private readiness selection.

    Output installs one callable on the actual adapter instance as
    ``_room_output_invitation_permissions(*, profile, catalog, connection)``.
    It returns a tuple of unique existing artifact.read/artifact.ack rights (or
    () when unavailable), never a boolean. No provider means no export rights.
    This is not a client field, catalog field, or proof of Output readiness.

    The provider owns initialized outbox, root/process owner, policy and actual
    read/ACK route checks. It must use the supplied owner SQL connection when
    present, without acquiring another writer or either grant store. None is
    the initial read-only selection; issuance rechecks on the held owner writer.
    The provider must be synchronous, read-only and safe to call repeatedly.
    """
    from gateway.hosted_room_peer import HostedRoomGrantError, invitation_permissions
    permissions = invitation_permissions(catalog)
    provider = getattr(self, '_room_output_invitation_permissions', None)
    if provider is None:
        return permissions
    if not callable(provider):
        raise HostedRoomGrantError('room output permission provider is invalid')
    # No legacy/standalone output issuance can bypass the canonical confirmation.
    from gateway.session_peer_target import root_target
    root_target(self, profile, connection=_connection)
    extra = provider(profile=profile, catalog=catalog, connection=_connection)
    if (type(extra) is not tuple or len(extra) > 2
            or any(type(right) is not str or right not in {'artifact.read', 'artifact.ack'} for right in extra)
            or len(set(extra)) != len(extra)):
        raise HostedRoomGrantError('room output permissions are invalid')
    return tuple(sorted((*permissions, *extra)))


def _http_routes(self) -> list[tuple[str, str, Any]]:
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
    ]


def _room_grant_token(request: "web.Request") -> str:
    scheme, separator, token = str(request.headers.get("Authorization") or "").partition(" ")
    return token.strip() if separator and scheme.lower() == "hermesroom" else ""


def _room_grant_secret(self) -> bytes:
    from gateway.hosted_room_peer import gateway_room_grant_secret
    return gateway_room_grant_secret()


def _decode_request_grant(self, request: "web.Request", *, permission: str) -> dict[str, Any]:
    """Signature/scope/horizon check only (no revocation lookup)."""
    from gateway.hosted_room_peer import decode_room_grant
    token = self._room_grant_token(request)
    if not token:
        raise ValueError("room grant is missing")
    return decode_room_grant(self._room_grant_secret(), token, permission=permission)


def _room_grant_claims(self, request: "web.Request", *, permission: str) -> dict[str, Any]:
    claims = _decode_request_grant(self, request, permission=permission)
    from gateway import hosted_rooms
    paths = (hosted_rooms.default_db_path(),)
    if _canonical_room_peer(self, claims['target_profile']):
        from gateway.hosted_room_grant_state import grant_state_db_paths
        paths = grant_state_db_paths()
    for db_path in paths:
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
        from gateway.hosted_room_peer import decode_room_grant, issue_room_grant
        from gateway.session_group_peers import invitation_preflight

        profile = _effective_room_profile(_api_request_profile)
        unavailable = _room_peer_unavailable(self, profile, _openai_error=_openai_error)
        if unavailable is not None:
            return unavailable
        target_install_id = hosted_rooms.local_authority_gateway_id()
        identity, (ttl, status_ttl) = invitation_preflight(body)
        execution_policy, catalog = _local_room_catalog(self, profile, target_install_id)
        if not catalog['text'] or execution_policy['approval_mode'] == 'off':
            raise ValueError('remote room execution requires an enabled approval policy')
        # Freeze the canonical binding before signing/reservation can block. A
        # detached adapter must not switch to the unconfirmed standalone path.
        binding = None
        if _canonical_room_peer(self, profile):
            from gateway.session_peer_target import root_target
            owner, paths = root_target(self, profile)
            binding = (owner, owner.epoch, owner.instance_id, owner.db, self.gateway_runner,
                       self.gateway_runner.session_authorities, self._run_idempotency_store, paths)
        permissions = _invitation_permissions(self, profile, catalog)
        token = issue_room_grant(
            self._room_grant_secret(),
            grant_id=str(body.get("grant_id") or f"grant-{uuid.uuid4().hex}"),
            **identity,
            permissions=permissions,
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
        if binding is not None:
            from gateway.session_peer_target import grant_fence, target_policy, require_current_grant
            with grant_fence(self, profile) as (authority, shared):
                def confirm(conn):
                    owner, paths, current_policy = target_policy(self, profile, connection=conn)
                    current_binding = (owner, owner.epoch, owner.instance_id, owner.db, self.gateway_runner,
                                       self.gateway_runner.session_authorities, self._run_idempotency_store, paths)
                    if owner is not authority or current_binding != binding:
                        raise ValueError('room target binding changed')
                    if current_policy != execution_policy:
                        raise ValueError('room execution policy changed')
                    _, current_catalog = _local_room_catalog(
                        self, profile, target_install_id, _connection=conn)
                    if (current_catalog != catalog
                            or _invitation_permissions(self, profile, current_catalog, _connection=conn) != permissions):
                        raise ValueError('room capability catalog changed')
                    require_current_grant(shared, claims)
                    require_current_grant(conn, claims)
                authority.db._execute_write(confirm)
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
    except Exception:
        return web.json_response(
            _openai_error(
                "Room grant revocation could not be persisted; retry required.",
                code="room_grant_revocation_unavailable",
            ),
            status=503,
        )
    return web.json_response(
        {
            "object": "hermes.room_member.grant.revocation",
            "revoked": True,
        }
    )


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
