"""Bounded reciprocal catalogue and verified binary reads; no redirects."""

from __future__ import annotations

import base64
import hashlib
import http.client
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request

from gateway.hosted_room_file_contract import (
    FILE_TIMEOUT_SECONDS,
    FileAccessError,
    MAX_CATALOG_BYTES,
    MAX_ERROR_BYTES,
    MAX_FILE_BYTES,
    MAX_QUERY_BYTES,
    MAX_RECEIPT_CHARS,
    attachment_id,
    byte_limit,
    catalog_options,
    catalog_page,
    file_metadata,
    identifier,
    scope,
    validate_scope,
)


def _request(
    client,
    *,
    target_profile,
    options,
    selected_id=None,
    max_bytes=MAX_FILE_BYTES,
    shared_event_id=None,
    lookup_code=None,
    latest_reply=False,
):
    from gateway.hosted_room_peer import validate_room_link_url
    from tui_gateway.hosted_room_peer_http import (
        _PeerResponseDeadlineExceeded,
        _PeerResponseTooLarge,
        _open_roomlink_url,
        _read_bounded_response,
    )

    link = client.link
    expected = scope(link.as_status(), link.member_id, target_profile)
    if (
        link.status != "active"
        or not math.isfinite(link.expires_at)
        or link.expires_at <= time.time()
    ):
        raise FileAccessError("file_access_denied")
    timeout = min(float(client.timeout_seconds), FILE_TIMEOUT_SECONDS)
    if not math.isfinite(timeout) or timeout <= 0:
        raise FileAccessError("file_invalid_request")
    try:
        base_url, _security = validate_room_link_url(link.home_url)
        suffix = "/files" + (
            f"/{attachment_id(selected_id)}" if selected_id is not None else ""
        )
        if shared_event_id is not None:
            suffix = "/messages/" + urllib.parse.quote(
                identifier(shared_event_id), safe=""
            )
        if lookup_code is not None:
            suffix = "/files/resolve"
        if latest_reply:
            suffix = "/latest-reply"
        url = f"{base_url.rstrip('/')}/v1/room-controls/{urllib.parse.quote(link.room_id, safe='')}{suffix}"
        url += "?" + urllib.parse.urlencode(options)
        if len(url.encode("utf-8")) > MAX_QUERY_BYTES:
            raise FileAccessError("file_invalid_request")
        request = urllib.request.Request(
            url,
            headers={
                "Authorization": f"HermesRoomControl {link.control_token}",
                "X-Hermes-Room-Member": link.member_id,
                "X-Hermes-Room-Profile": target_profile,
                "X-Hermes-Room-Authority": link.authority_gateway_id,
                "X-Hermes-Room-Epoch": str(link.authority_epoch),
                "Accept-Encoding": "identity",
                "User-Agent": "Hermes-RoomControl-Files/1.0",
            },
        )
    except (ValueError, UnicodeError):
        raise FileAccessError("file_invalid_request") from None
    deadline = time.monotonic() + timeout
    try:
        with _open_roomlink_url(
            request, timeout=timeout, reject_redirects=True
        ) as response:
            if (
                response.status != 200
                or response.headers.get("Content-Encoding", "identity").lower()
                != "identity"
            ):
                raise FileAccessError("file_invalid_response")
            metadata = None
            maximum = MAX_CATALOG_BYTES
            if shared_event_id is not None or latest_reply:
                from gateway.hosted_room_shared_message_access import (
                    MAX_REPLY_RESPONSE_BYTES,
                )

                maximum = MAX_REPLY_RESPONSE_BYTES
            if selected_id is not None:
                receipt = response.headers.get("X-Hermes-Room-File", "")
                if not receipt or len(receipt) > MAX_RECEIPT_CHARS:
                    raise FileAccessError("file_invalid_response")
                try:
                    decoded = json.loads(
                        base64.b64decode(receipt, altchars=b"-_", validate=True).decode(
                            "utf-8"
                        )
                    )
                    if (
                        not isinstance(decoded, dict)
                        or set(decoded) != {"scope", "attachment"}
                        or decoded["scope"] != expected
                    ):
                        raise ValueError
                    metadata = file_metadata(
                        decoded["attachment"],
                        event_id=options["event_id"],
                        selected_id=selected_id,
                    )
                    validate_scope(decoded["scope"], expected)
                except (ValueError, UnicodeError):
                    raise FileAccessError("file_invalid_response") from None
                if metadata["size"] > max_bytes:
                    raise FileAccessError("file_too_large")
                maximum = metadata["size"]
                if response.headers.get_content_type() != metadata["mime"]:
                    raise FileAccessError("file_invalid_response")
            elif response.headers.get_content_type() != "application/json":
                raise FileAccessError("file_invalid_response")
            declared = response.headers.get("Content-Length")
            if declared is not None:
                if not declared.isdecimal() or int(declared) > maximum:
                    raise FileAccessError("file_too_large")
                if metadata is not None and int(declared) != metadata["size"]:
                    raise FileAccessError("file_invalid_response")
            raw = _read_bounded_response(response, max_bytes=maximum, deadline=deadline)
            if declared is not None and len(raw) != int(declared):
                raise FileAccessError("file_invalid_response")
    except urllib.error.HTTPError as exc:
        code = ""
        try:
            with exc:
                raw_error = _read_bounded_response(
                    exc, max_bytes=MAX_ERROR_BYTES, deadline=deadline
                )
            error = json.loads(raw_error.decode("utf-8"))
            code = error.get("error", {}).get("code", "")
        except _PeerResponseDeadlineExceeded:
            raise FileAccessError("file_timeout", retryable=True) from None
        except _PeerResponseTooLarge:
            raise FileAccessError("file_invalid_response") from None
        except Exception:
            pass
        if exc.code in {404, 405, 501} and code != "file_unavailable":
            raise FileAccessError("file_access_unsupported") from None
        if exc.code in {401, 403}:
            raise FileAccessError("file_access_denied") from None
        if code in {
            "attachment_cursor_reset_required",
            "file_unavailable",
            "file_too_large",
            "file_integrity_failed",
            "file_invalid_request",
            "file_access_unsupported",
            "file_code_ambiguous",
            "file_lookup_limit",
        }:
            raise FileAccessError(code) from None
        raise FileAccessError(
            "file_host_unavailable", retryable=exc.code >= 500 or exc.code == 429
        ) from None
    except _PeerResponseTooLarge:
        raise FileAccessError("file_too_large") from None
    except (_PeerResponseDeadlineExceeded, TimeoutError):
        raise FileAccessError("file_timeout", retryable=True) from None
    except (urllib.error.URLError, OSError):
        raise FileAccessError("file_host_unavailable", retryable=True) from None
    except (http.client.HTTPException, ValueError):
        raise FileAccessError("file_invalid_response") from None
    if metadata is not None:
        if (
            len(raw) != metadata["size"]
            or hashlib.sha256(raw).hexdigest() != metadata["sha256"]
        ):
            raise FileAccessError("file_integrity_failed")
        from gateway.hosted_room_attachments import AttachmentData

        return AttachmentData(metadata, raw)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeError):
        raise FileAccessError("file_invalid_response") from None
    if lookup_code is not None:
        from gateway.hosted_room_file_contract import FileCodeAmbiguous
        from gateway.hosted_room_file_lookup import selection_digest

        if not isinstance(value, dict) or set(value) not in (
            {"scope", "selection"},
            {"scope", "matches"},
        ):
            raise FileAccessError("file_invalid_response")
        validate_scope(value["scope"], expected)
        matches = value.get("matches") if "matches" in value else [value["selection"]]
        if (
            not isinstance(matches, list)
            or not (2 <= len(matches) <= 8 if "matches" in value else len(matches) == 1)
            or not all(
                isinstance(item, dict) and type(item.get("seq")) is int
                for item in matches
            )
        ):
            raise FileAccessError("file_invalid_response")
        catalog_page(
            {
                "scope": expected,
                "items": matches,
                "has_more": False,
                "next_cursor": None,
                "snapshot_seq": max(item["seq"] for item in matches),
                "authority": {
                    "gateway_id": link.authority_gateway_id,
                    "epoch": link.authority_epoch,
                },
            },
            expected_scope=expected,
            limit=8,
        )
        if any(
            not selection_digest(expected, item).startswith(lookup_code)
            for item in matches
        ):
            raise FileAccessError("file_invalid_response")
        if "matches" in value:
            raise FileCodeAmbiguous(matches)
        return matches[0]
    if shared_event_id is not None or latest_reply:
        from gateway.hosted_room_shared_message_access import validate_reply

        if (
            not isinstance(value, dict)
            or set(value) != {"scope", "reply"}
            or value["scope"] != expected
        ):
            raise FileAccessError("file_invalid_response")
        validate_scope(value["scope"], expected)
        reply = value["reply"]
        if latest_reply:
            if not isinstance(reply, dict):
                raise FileAccessError("file_invalid_response")
            shared_event_id = identifier(reply.get("event_id"))
        return validate_reply(reply, event_id=shared_event_id)
    return catalog_page(value, expected_scope=expected, limit=options["limit"])


def list_files(client, *, target_profile: str, **options):
    return _request(
        client,
        target_profile=identifier(target_profile),
        options=catalog_options(options),
    )


def read_file(
    client,
    *,
    target_profile: str,
    event_id: str,
    attachment_id: str,
    max_bytes=MAX_FILE_BYTES,
):
    return _request(
        client,
        target_profile=identifier(target_profile),
        selected_id=attachment_id,
        options={"event_id": identifier(event_id), "max_bytes": byte_limit(max_bytes)},
        max_bytes=max_bytes,
    )


def read_shared_message(client, *, target_profile, event_id):
    return _request(
        client,
        target_profile=identifier(target_profile),
        options={},
        shared_event_id=identifier(event_id),
    )


def resolve_file(client, *, target_profile, code):
    from gateway.hosted_room_file_lookup import validate_code

    code = validate_code(code)
    return _request(
        client,
        target_profile=identifier(target_profile),
        options={"code": code},
        lookup_code=code,
    )


def latest_shared_message(client, *, target_profile):
    return _request(
        client, target_profile=identifier(target_profile), options={}, latest_reply=True
    )
