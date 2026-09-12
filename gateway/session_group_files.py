"""Canonical, owner-authorized Group Chat Files handlers.

The parent dispatcher registers these methods and runs this synchronous handler
off-loop under the authority profile scope. No worker startup or execution is
required. Upload uses the existing canonical byte-store path; list/download
retain exact room, authority, event and attachment version identity.
"""

from __future__ import annotations

import base64
from pathlib import Path

from gateway.hosted_room_attachments import (
    AttachmentConflictError,
    AttachmentCursorError,
    AttachmentError,
    AttachmentIntegrityError,
    AttachmentNotFoundError,
    AttachmentQuotaError,
    HostedRoomAttachmentStore,
)
from gateway.hosted_rooms import AuthorityConflictError, HostedRoomError, RoomNotFoundError
from hermes_state_runtime import RuntimeStoreError


GROUP_FILE_METHODS = {
    "groups.attachment.upload": "session:submit",
    "groups.attachment.list": "session:read",
    "groups.attachment.download": "session:read",
}
_ORIGIN = {"authority_gateway_id", "authority_epoch"}
GROUP_FILE_FIELDS = {
    "groups.attachment.upload": {"room_id", "upload_id", "kind", "name", "mime", "data_base64"} | _ORIGIN,
    "groups.attachment.list": {"room_id", "cursor", "limit", "query", "producer_member_id"} | _ORIGIN,
    "groups.attachment.download": {"room_id", "event_id", "attachment_id"} | _ORIGIN,
}
GROUP_FILE_REQUIRED_FIELDS = {
    "groups.attachment.upload": {"room_id", "upload_id", "kind", "name", "mime", "data_base64"},
    "groups.attachment.list": {"room_id"},
    "groups.attachment.download": {"room_id", "event_id", "attachment_id"},
}


def _check_principal(service, actor, capability):
    if service is None:
        raise RuntimeStoreError("runtime_coordination_required")
    if capability not in actor.capabilities:
        raise RuntimeStoreError("permission_denied")
    authority = service.authority
    if actor.profile_id != authority.profile_id:
        raise RuntimeStoreError("profile_mismatch")
    home = Path(authority.profile_id).resolve()
    db_path = Path(authority.db.db_path).resolve()
    if db_path.parent != home or Path(service.db_path).resolve() != db_path:
        raise RuntimeStoreError("profile_mismatch")
    return home


def _validate_params(method, params, home):
    if (not isinstance(params, dict) or set(params) - (GROUP_FILE_FIELDS[method] | {"profile"})
            or not GROUP_FILE_REQUIRED_FIELDS[method].issubset(params)):
        raise RuntimeStoreError("invalid_params")
    if any(not isinstance(params[key], str) or not params[key] for key in GROUP_FILE_REQUIRED_FIELDS[method]):
        raise RuntimeStoreError("invalid_params")
    profile = params.get("profile")
    if profile is not None:
        if not isinstance(profile, str):
            raise RuntimeStoreError("invalid_params")
        if profile:
            from hermes_cli.profiles import profile_matches_home
            if not profile_matches_home(profile, home):
                raise RuntimeStoreError("profile_mismatch")
    supplied_origin = _ORIGIN.intersection(params)
    if supplied_origin and (supplied_origin != _ORIGIN
            or not isinstance(params["authority_gateway_id"], str) or not params["authority_gateway_id"]
            or type(params["authority_epoch"]) is not int or params["authority_epoch"] < 1):
        raise RuntimeStoreError("invalid_params")
    return {key: value for key, value in params.items() if key != "profile"}


def _authorize(service, actor, params, capability):
    _check_principal(service, actor, capability)
    room_id = params["room_id"]
    service.authorize_room(actor.subject, room_id)
    gateway_id, epoch = service._owned_authority(room_id)
    if _ORIGIN.issubset(params) and (
            params["authority_gateway_id"] != gateway_id or params["authority_epoch"] != epoch):
        raise RuntimeStoreError("attachment_scope_changed")
    return gateway_id, epoch


def _upload(service, actor, params, gateway_id, epoch):
    from gateway.session_hosted_attachments import upload
    return upload(service, actor, {key: value for key, value in params.items() if key not in _ORIGIN})


def _list(service, actor, params, gateway_id, epoch):
    return HostedRoomAttachmentStore(service.db_path).list_published(
        room_id=params["room_id"], authority_gateway_id=gateway_id, authority_epoch=epoch,
        **{key: params[key] for key in ("cursor", "limit", "query", "producer_member_id") if key in params},
    )


def _download(service, actor, params, gateway_id, epoch):
    saved = HostedRoomAttachmentStore(service.db_path).read_viewer(
        room_id=params["room_id"], event_id=params["event_id"], attachment_id=params["attachment_id"],
        authority_gateway_id=gateway_id, authority_epoch=epoch,
    )
    return {**saved.attachment, "data_base64": base64.b64encode(saved.data).decode("ascii")}


_HANDLERS = {
    "groups.attachment.upload": _upload,
    "groups.attachment.list": _list,
    "groups.attachment.download": _download,
}
_ERRORS = {
    AttachmentCursorError: "attachment_cursor_invalid",
    AttachmentNotFoundError: "attachment_not_found",
    AttachmentIntegrityError: "attachment_integrity_error",
    AttachmentQuotaError: "attachment_quota_exceeded",
    AttachmentConflictError: "attachment_conflict",
    AuthorityConflictError: "attachment_scope_changed",
    RoomNotFoundError: "attachment_not_found",
}


def dispatch_group_files(service, actor, method: str, params: dict) -> dict:
    """Run one Files operation with independent authorization and schema checks.

    Optional authority_gateway_id/authority_epoch are an inseparable selection
    pin, never authority supplied by the client. Older upload/download callers
    may omit both; event_id + attachment_id remain mandatory for downloads.
    """
    capability = GROUP_FILE_METHODS.get(method)
    if capability is None:
        raise RuntimeStoreError("invalid_params")
    home = _check_principal(service, actor, capability)
    supplied = _validate_params(method, params, home)
    try:
        origin = _authorize(service, actor, supplied, capability)
        result = _HANDLERS[method](service, actor, supplied, *origin)
        # The catalog/store fences room state. Recheck the canonical principal's
        # room-owner binding too, after snapshot/hash/file I/O and before return.
        if _authorize(service, actor, supplied, capability) != origin:
            raise RuntimeStoreError("attachment_scope_changed")
    except RuntimeStoreError:
        raise
    except (AttachmentError, HostedRoomError) as exc:
        reason = next((reason for kind, reason in _ERRORS.items() if isinstance(exc, kind)), "invalid_params")
        raise RuntimeStoreError(reason) from exc
    except (TypeError, ValueError, KeyError) as exc:
        raise RuntimeStoreError("invalid_params") from exc
    if method == "groups.attachment.upload":
        return {key: result[key] for key in ("attachment_id", "kind", "name", "size", "mime")}
    return {**result, "room_id": supplied["room_id"], "authority": {"gateway_id": origin[0], "epoch": origin[1]}}
