"""Scoped RoomLink output-artifact download and acknowledgement handlers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from aiohttp import web
except ImportError:
    web = None  # type: ignore[assignment]


def _http_routes(self) -> list[tuple[str, str, Any]]:
    async def read_artifact(request):
        from gateway.platforms import api_server

        return await _handle_room_run_artifact(
            self,
            request,
            _openai_error=api_server._openai_error,
        )

    async def acknowledge(request):
        from gateway.platforms import api_server

        return await _handle_room_run_artifact_ack(
            self,
            request,
            _openai_error=api_server._openai_error,
        )

    async def discard(request):
        from gateway.platforms import api_server

        return await _handle_room_run_artifact_discard(
            self,
            request,
            _openai_error=api_server._openai_error,
        )

    return [
        (
            "GET",
            "/v1/runs/{run_id}/artifacts/{artifact_id}",
            read_artifact,
        ),
        (
            "POST",
            "/v1/runs/{run_id}/artifacts/ack",
            acknowledge,
        ),
        (
            "POST",
            "/v1/runs/{run_id}/artifacts/discard",
            discard,
        ),
    ]


def _load_scope_and_status(
    self,
    request,
    *,
    permission: str,
    validate_manifest: bool = True,
):
    from gateway.hosted_room_artifacts import (
        RoomArtifactScope,
        validate_terminal_artifact_manifest,
    )

    claims = self._room_grant_claims(request, permission=permission)
    run_id = str(request.match_info["run_id"])
    if not self._request_owns_run(request, run_id):
        raise ValueError("run not found")
    status = self._durable_run_status(request, run_id)
    if not isinstance(status, dict):
        raise ValueError("run not found")
    scope = RoomArtifactScope.from_mapping(status.get("room_artifact_scope") or {})
    if any(
        claims.get(key) != value
        for key, value in (
            ("room_id", scope.room_id),
            ("home_install_id", scope.home_install_id),
            ("authority_gateway_id", scope.authority_gateway_id),
            ("authority_epoch", scope.authority_epoch),
            ("member_id", scope.member_id),
            ("target_install_id", scope.target_install_id),
            ("target_profile", scope.target_profile),
        )
    ):
        raise ValueError("run not found")
    manifest = (
        validate_terminal_artifact_manifest(status.get("artifacts"))
        if validate_manifest
        else []
    )
    return run_id, scope, status, manifest


async def _handle_room_run_artifact(
    self,
    request: "web.Request",
    *,
    _openai_error,
) -> "web.Response":
    try:
        _run_id, scope, _status, manifest = _load_scope_and_status(
            self,
            request,
            permission="artifact.read",
        )
        artifact_id = str(request.match_info["artifact_id"])
        expected = next(
            (item for item in manifest if item["artifact_id"] == artifact_id),
            None,
        )
        if expected is None:
            raise ValueError("artifact not found")
        from gateway.hosted_room_artifacts import RoomArtifactOutbox
        from hermes_constants import get_hermes_home

        metadata, data = RoomArtifactOutbox(
            Path(get_hermes_home()) / "state.db"
        ).read(scope, artifact_id)
        if metadata != expected:
            raise ValueError("artifact not found")
        return web.Response(
            body=data,
            content_type=metadata["mime"],
            headers={
                "X-Hermes-Artifact-SHA256": metadata["sha256"],
                "Cache-Control": "no-store",
            },
        )
    except Exception:
        return web.json_response(
            _openai_error("Artifact not found.", code="artifact_not_found"),
            status=404,
        )


async def _handle_room_run_artifact_ack(
    self,
    request: "web.Request",
    *,
    _openai_error,
) -> "web.Response":
    try:
        _run_id, scope, status, manifest = _load_scope_and_status(
            self,
            request,
            permission="artifact.ack",
        )
        body = await request.json()
        if not isinstance(body, dict) or set(body) != {
            "artifact_ids",
            "manifest_digest",
            "message_event_id",
        }:
            raise ValueError("invalid acknowledgement")
        artifact_ids = body["artifact_ids"]
        expected_message_event_id = (
            f"dmessage:{scope.task_id.removeprefix('dtask:')}"
        )
        if (
            not isinstance(artifact_ids, list)
            or tuple(artifact_ids) != tuple(item["artifact_id"] for item in manifest)
            or body["manifest_digest"] != status["artifacts"]["manifest_digest"]
            or body["message_event_id"] != expected_message_event_id
        ):
            raise ValueError("invalid acknowledgement")
        from gateway.hosted_room_artifacts import RoomArtifactOutbox
        from hermes_constants import get_hermes_home

        changed = RoomArtifactOutbox(
            Path(get_hermes_home()) / "state.db"
        ).acknowledge(
            scope,
            artifact_ids,
            message_event_id=expected_message_event_id,
        )
        return web.json_response({"acknowledged": True, "changed": changed})
    except Exception:
        return web.json_response(
            _openai_error(
                "Artifact acknowledgement was rejected.",
                code="invalid_artifact_ack",
            ),
            status=409,
        )


async def _handle_room_run_artifact_discard(
    self,
    request: "web.Request",
    *,
    _openai_error,
) -> "web.Response":
    """Retire one exact run outbox after permanent home-side rejection."""

    try:
        _run_id, scope, _status, _manifest = _load_scope_and_status(
            self,
            request,
            permission="artifact.ack",
            validate_manifest=False,
        )
        body = await request.json()
        if body != {"reason": "verification_failed"}:
            raise ValueError("invalid artifact retirement")
    except Exception:
        return web.json_response(
            _openai_error(
                "Artifact retirement was rejected.",
                code="invalid_artifact_retirement",
            ),
            status=409,
        )
    try:
        from gateway.hosted_room_artifacts import RoomArtifactOutbox
        from hermes_constants import get_hermes_home

        removed = RoomArtifactOutbox(
            Path(get_hermes_home()) / "state.db"
        ).discard_durably(scope)
    except Exception:
        return web.json_response(
            _openai_error(
                "Artifact retirement is temporarily unavailable.",
                code="artifact_retirement_unavailable",
            ),
            status=503,
            headers={"Retry-After": "1"},
        )
    return web.json_response({"discarded": True, "removed": removed})


__all__ = [
    "_handle_room_run_artifact",
    "_handle_room_run_artifact_ack",
    "_handle_room_run_artifact_discard",
    "_http_routes",
]
