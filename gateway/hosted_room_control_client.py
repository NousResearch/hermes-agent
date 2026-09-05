"""Credential-safe client for a room authority's reciprocal control API."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from gateway.hosted_room_controls import StoredPeerRoomControl
from hermes_cli.urllib_security import open_credentialed_url


MAX_CONTROL_RESPONSE_BYTES = 1024 * 1024


class RoomControlClientError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
        user_message: str = "",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable
        self.user_message = user_message


class RoomControlHTTPClient:
    def __init__(
        self,
        link: StoredPeerRoomControl,
        *,
        timeout_seconds: float = 15.0,
    ) -> None:
        self.link = link
        self.timeout_seconds = float(timeout_seconds)

    def _request(
        self,
        *,
        method: str,
        body: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        room_id = urllib.parse.quote(self.link.room_id, safe="")
        request = urllib.request.Request(
            f"{self.link.home_url.rstrip('/')}/v1/room-controls/{room_id}",
            data=(
                json.dumps(body, ensure_ascii=True, separators=(",", ":")).encode(
                    "utf-8"
                )
                if body is not None
                else None
            ),
            method=method,
            headers={
                "Authorization": f"HermesRoomControl {self.link.control_token}",
                "X-Hermes-Room-Member": self.link.member_id,
                "Content-Type": "application/json",
                "User-Agent": "Hermes-RoomControl/1.0",
            },
        )
        try:
            with open_credentialed_url(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                raw = response.read(MAX_CONTROL_RESPONSE_BYTES + 1)
                if len(raw) > MAX_CONTROL_RESPONSE_BYTES:
                    raise RoomControlClientError(
                        "Group Chat control response exceeded the size limit"
                    )
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read(500).decode("utf-8", "replace")
            except Exception:
                detail = ""
            user_message = ""
            try:
                payload = json.loads(detail)
                raw_error = payload.get("error") if isinstance(payload, dict) else None
                if isinstance(raw_error, dict):
                    candidate = str(raw_error.get("message") or "").strip()
                    if candidate and len(candidate) <= 300:
                        user_message = candidate
            except (TypeError, ValueError):
                pass
            raise RoomControlClientError(
                f"Group Chat host rejected control with HTTP {exc.code}",
                status_code=exc.code,
                retryable=exc.code in {408, 425, 429} or exc.code >= 500,
                user_message=user_message,
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RoomControlClientError(
                "Group Chat host is unreachable",
                retryable=True,
            ) from exc
        try:
            payload = json.loads(raw.decode("utf-8", "replace"))
        except (UnicodeError, ValueError) as exc:
            raise RoomControlClientError(
                "Group Chat host returned invalid control data"
            ) from exc
        if not isinstance(payload, dict):
            raise RoomControlClientError(
                "Group Chat host returned a non-object control response"
            )
        return payload

    def summary(self) -> dict[str, Any]:
        return self._request(method="GET")

    def list_files(self, *, target_profile: str, **options):
        from gateway.hosted_room_control_files_client import list_files

        return list_files(self, target_profile=target_profile, **options)

    def read_file(self, *, target_profile: str, **selection):
        from gateway.hosted_room_control_files_client import read_file

        return read_file(self, target_profile=target_profile, **selection)

    def read_shared_message(self, *, target_profile: str, event_id: str):
        from gateway.hosted_room_control_files_client import read_shared_message

        return read_shared_message(self, target_profile=target_profile, event_id=event_id)

    def resolve_file(self, *, target_profile: str, code: str):
        from gateway.hosted_room_control_files_client import resolve_file

        return resolve_file(self, target_profile=target_profile, code=code)

    def latest_shared_message(self, *, target_profile: str):
        from gateway.hosted_room_control_files_client import latest_shared_message

        return latest_shared_message(self, target_profile=target_profile)

    def revoke(self) -> int:
        result = self._request(method="DELETE")
        revoked = result.get("revoked")
        if type(revoked) is not int or revoked not in {0, 1}:
            raise RoomControlClientError(
                "Group Chat host returned an invalid revocation acknowledgement"
            )
        return revoked

    def mutate(
        self,
        *,
        action: str,
        command_id: str,
        text: str = "",
        actor_display_name: str = "Messaging",
    ) -> dict[str, Any]:
        return self._request(
            method="POST",
            body={
                "action": action,
                "command_id": command_id,
                **({"text": text} if text else {}),
                **(
                    {"actor_display_name": actor_display_name}
                    if actor_display_name
                    else {}
                ),
            },
        )


def revoke_stored_peer_control(
    db_path: Path | str,
    *,
    room_id: str,
    member_id: str,
) -> int:
    """Revoke the authority credential, then erase peer bearer material."""

    from gateway import hosted_room_controls

    link = next(
        (
            candidate
            for candidate in hosted_room_controls.load_peer_control_links(
                db_path, include_inactive=True
            ).links
            if candidate.room_id == room_id and candidate.member_id == member_id
        ),
        None,
    )
    if link is None:
        return 0
    if link.status == "expired":
        return hosted_room_controls.delete_peer_control_link_value(
            db_path, expected=link, required_status="expired"
        )
    revoked = hosted_room_controls.revoke_peer_control_link_value(
        db_path, expected=link
    )
    if revoked is None:
        return 0
    RoomControlHTTPClient(revoked).revoke()
    return hosted_room_controls.delete_peer_control_link_value(
        db_path, expected=revoked, required_status="revoked"
    )
