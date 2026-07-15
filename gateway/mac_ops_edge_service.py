#!/usr/bin/env python3
"""Token-owning GitLab transport for the mechanical Mac operations edge.

The unprivileged Hermes gateway never reads the GitLab token.  It submits one
strict read-only contract over a protected Unix socket.  This service creates
or reads the corresponding confidential issue and returns bounded external
evidence plus a digest-bound receipt.

No task prose is classified here.  The only policy decision is membership in
the explicit :class:`MacOpsReadOnlyClass` enum validated by the protocol.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sqlite3
import stat
import struct
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from gateway.mac_ops_edge_protocol import (
    MAX_TITLE_CHARS,
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    RESPONSE_VERSION,
    MacOpsEdgeOperation,
    MacOpsEdgeProtocolError,
    MacOpsEdgeReceipt,
    MacOpsEdgeRequest,
    MacOpsEdgeState,
    MacOpsPing,
    MacOpsReadOnlySubmit,
    canonical_json_bytes,
    decode_json_object,
)
from gateway.systemd_credentials import (
    MAC_OPS_GITLAB_CREDENTIAL,
    MAC_OPS_UNIT,
    SystemdCredentialError,
    is_expected_systemd_credential,
    read_systemd_credential,
)


CONFIG_SCHEMA = "muncho-mac-ops-edge-config.v1"
DEFAULT_SOCKET_PATH = Path("/run/muncho-mac-ops/edge.sock")
DEFAULT_CONFIG_PATH = Path("/etc/muncho/mac-ops-edge/config.json")
DEFAULT_PROJECT_ID = "69"
ISSUE_MARKER = "MAC_OPS_BRIDGE_TASK"
ISSUE_SCHEMA = "mac_ops_bridge_task.v1"

_FRAME_HEADER = struct.Struct("!I")
_PEER_CREDENTIALS = struct.Struct("3i")
_MAX_CONFIG_BYTES = 64 * 1024
_MAX_SECRET_FILE_BYTES = 32 * 1024
_MAX_HTTP_BODY_BYTES = 512 * 1024
_MAX_NOTE_BODY_CHARS = 12_000
_MAX_NOTES = 40


class MacOpsEdgeServiceError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class MacOpsGitLabTransportError(MacOpsEdgeServiceError):
    def __init__(self, code: str, *, ambiguous: bool) -> None:
        self.ambiguous = ambiguous
        super().__init__(code)


@dataclass(frozen=True)
class MacOpsEdgePeer:
    pid: int
    uid: int
    gid: int


@dataclass(frozen=True)
class MacOpsEdgeConfig:
    socket_path: Path
    gateway_uid: int
    socket_gid: int
    service_identity_sha256: str
    max_connections: int
    gitlab_env_file: Path
    gitlab_project_id: str
    gitlab_timeout_seconds: float
    journal_path: Path
    journal_busy_timeout_ms: int


class GitLabApi(Protocol):
    def create_issue(self, payload: Mapping[str, str]) -> Mapping[str, Any]: ...

    def search_issues(self, search: str) -> list[Mapping[str, Any]]: ...

    def read_issue(self, issue_iid: int) -> Mapping[str, Any]: ...

    def read_notes(self, issue_iid: int) -> list[Mapping[str, Any]]: ...


def linux_peer_credentials(sock: socket.socket) -> MacOpsEdgePeer:
    option = getattr(socket, "SO_PEERCRED", None)
    if option is None:
        raise OSError("peer_credentials_unavailable")
    raw = sock.getsockopt(socket.SOL_SOCKET, option, _PEER_CREDENTIALS.size)
    if len(raw) != _PEER_CREDENTIALS.size:
        raise OSError("peer_credentials_invalid")
    return MacOpsEdgePeer(*_PEER_CREDENTIALS.unpack(raw))


def _canonical_path(value: Any, code: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(code)
    path = Path(value)
    if not path.is_absolute() or path != Path(os.path.normpath(path)):
        raise ValueError(code)
    return path


def _exact_mapping(
    value: Any,
    *,
    fields: frozenset[str],
    code: str,
) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != fields
    ):
        raise ValueError(code)
    return value


def _bounded_integer(value: Any, *, minimum: int, maximum: int, code: str) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(code)
    return value


def _read_stable_regular_file(
    path: Path,
    *,
    maximum: int,
    allowed_modes: frozenset[int],
    expected_uid: int | None,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError("protected_file_unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) not in allowed_modes
            or (expected_uid is not None and before.st_uid != expected_uid)
            or before.st_size <= 0
            or before.st_size > maximum
        ):
            raise ValueError("protected_file_identity_invalid")
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            len(raw) > maximum
            or before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or len(raw) != after.st_size
        ):
            raise ValueError("protected_file_changed")
        return raw
    finally:
        os.close(descriptor)


def load_config(
    path: Path,
    *,
    expected_owner_uid: int = 0,
) -> MacOpsEdgeConfig:
    raw = _read_stable_regular_file(
        path,
        maximum=_MAX_CONFIG_BYTES,
        allowed_modes=frozenset({0o400, 0o440}),
        expected_uid=expected_owner_uid,
    )
    value = decode_json_object(raw, maximum=_MAX_CONFIG_BYTES)
    root = _exact_mapping(
        value,
        fields=frozenset({"schema", "service", "gitlab", "journal"}),
        code="config_shape_invalid",
    )
    if root["schema"] != CONFIG_SCHEMA:
        raise ValueError("config_schema_invalid")
    service = _exact_mapping(
        root["service"],
        fields=frozenset(
            {
                "socket_path",
                "gateway_uid",
                "socket_gid",
                "service_identity_sha256",
                "max_connections",
            }
        ),
        code="config_service_invalid",
    )
    gitlab = _exact_mapping(
        root["gitlab"],
        fields=frozenset({"env_file", "project_id", "timeout_seconds"}),
        code="config_gitlab_invalid",
    )
    journal = _exact_mapping(
        root["journal"],
        fields=frozenset({"path", "busy_timeout_ms"}),
        code="config_journal_invalid",
    )
    identity = service["service_identity_sha256"]
    if (
        not isinstance(identity, str)
        or len(identity) != 64
        or any(char not in "0123456789abcdef" for char in identity)
    ):
        raise ValueError("config_service_identity_invalid")
    project_id = gitlab["project_id"]
    if project_id != DEFAULT_PROJECT_ID:
        raise ValueError("config_project_not_pinned")
    timeout = gitlab["timeout_seconds"]
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError("config_timeout_invalid")
    timeout_float = float(timeout)
    if not 1 <= timeout_float <= 30:
        raise ValueError("config_timeout_invalid")
    return MacOpsEdgeConfig(
        socket_path=_canonical_path(service["socket_path"], "config_socket_invalid"),
        gateway_uid=_bounded_integer(
            service["gateway_uid"], minimum=1, maximum=(1 << 31) - 1,
            code="config_gateway_uid_invalid",
        ),
        socket_gid=_bounded_integer(
            service["socket_gid"], minimum=1, maximum=(1 << 31) - 1,
            code="config_socket_gid_invalid",
        ),
        service_identity_sha256=identity,
        max_connections=_bounded_integer(
            service["max_connections"], minimum=1, maximum=32,
            code="config_connections_invalid",
        ),
        gitlab_env_file=_canonical_path(
            gitlab["env_file"], "config_gitlab_env_invalid"
        ),
        gitlab_project_id=project_id,
        gitlab_timeout_seconds=timeout_float,
        journal_path=_canonical_path(journal["path"], "config_journal_invalid"),
        journal_busy_timeout_ms=_bounded_integer(
            journal["busy_timeout_ms"], minimum=100, maximum=30_000,
            code="config_journal_invalid",
        ),
    )


def _parse_secret_env(raw: bytes) -> tuple[str, str]:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("gitlab_credential_invalid") from exc
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError("gitlab_credential_invalid")
        key, value = line.split("=", 1)
        key = key.strip()
        if key in values:
            raise ValueError("gitlab_credential_invalid")
        values[key] = value.strip().strip('"').strip("'")
    base = values.get("GITLAB_BASE_URL", "").rstrip("/")
    token = values.get("GITLAB_TOKEN", "")
    if (
        not base.startswith("https://")
        or not token
        or any(ord(char) < 33 for char in token)
        or len(token) > 4096
    ):
        raise ValueError("gitlab_credential_invalid")
    return base, token


class _SecretToken:
    __slots__ = ("value",)

    def __init__(self, value: str) -> None:
        self.value = value

    def __repr__(self) -> str:
        return "<GitLab credential: redacted>"


class UrllibGitLabApi:
    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        project_id: str,
        timeout_seconds: float,
    ) -> None:
        self._base_url = base_url
        self._token = _SecretToken(token)
        self._project = urllib.parse.quote(project_id, safe="")
        self._timeout_seconds = timeout_seconds

    @classmethod
    def from_config(
        cls,
        config: MacOpsEdgeConfig,
        *,
        service_uid: int | None = None,
    ) -> "UrllibGitLabApi":
        uid = os.geteuid() if service_uid is None else service_uid  # windows-footgun: ok — macOS/Linux AF_UNIX service boundary
        try:
            systemd_bound = is_expected_systemd_credential(
                config.gitlab_env_file,
                unit=MAC_OPS_UNIT,
                name=MAC_OPS_GITLAB_CREDENTIAL,
            )
            if systemd_bound:
                raw = read_systemd_credential(
                    config.gitlab_env_file,
                    unit=MAC_OPS_UNIT,
                    name=MAC_OPS_GITLAB_CREDENTIAL,
                    service_uid=uid,
                    maximum=_MAX_SECRET_FILE_BYTES,
                    credentials_directory=config.gitlab_env_file.parent,
                )
            else:
                raw = _read_stable_regular_file(
                    config.gitlab_env_file,
                    maximum=_MAX_SECRET_FILE_BYTES,
                    allowed_modes=frozenset({0o400, 0o600}),
                    expected_uid=uid,
                )
        except SystemdCredentialError as exc:
            raise ValueError("gitlab_credential_invalid") from exc
        base, token = _parse_secret_env(raw)
        return cls(
            base_url=base,
            token=token,
            project_id=config.gitlab_project_id,
            timeout_seconds=config.gitlab_timeout_seconds,
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, Any] | None = None,
        form: Mapping[str, Any] | None = None,
    ) -> Any:
        url = self._base_url + "/api/v4" + path
        if query:
            url += "?" + urllib.parse.urlencode(dict(query), doseq=True)
        body = None
        headers = {
            "PRIVATE-TOKEN": self._token.value,
            "Accept": "application/json",
        }
        if form is not None:
            body = urllib.parse.urlencode(dict(form)).encode("utf-8")
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        request = urllib.request.Request(url, data=body, method=method, headers=headers)
        try:
            with urllib.request.urlopen(
                request, timeout=self._timeout_seconds
            ) as response:
                declared = response.headers.get("Content-Length")
                if declared is not None:
                    try:
                        if int(declared) > _MAX_HTTP_BODY_BYTES:
                            raise MacOpsGitLabTransportError(
                                "gitlab_response_too_large", ambiguous=method != "GET"
                            )
                    except ValueError as exc:
                        raise MacOpsGitLabTransportError(
                            "gitlab_response_invalid", ambiguous=method != "GET"
                        ) from exc
                raw = response.read(_MAX_HTTP_BODY_BYTES + 1)
                if len(raw) > _MAX_HTTP_BODY_BYTES:
                    raise MacOpsGitLabTransportError(
                        "gitlab_response_too_large", ambiguous=method != "GET"
                    )
        except urllib.error.HTTPError as exc:
            raise MacOpsGitLabTransportError(
                "gitlab_api_rejected", ambiguous=False
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise MacOpsGitLabTransportError(
                "gitlab_transport_unavailable", ambiguous=method != "GET"
            ) from exc
        try:
            return json.loads(raw.decode("utf-8", errors="strict")) if raw else {}
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MacOpsGitLabTransportError(
                "gitlab_response_invalid", ambiguous=method != "GET"
            ) from exc

    def create_issue(self, payload: Mapping[str, str]) -> Mapping[str, Any]:
        value = self._request(
            "POST", f"/projects/{self._project}/issues", form=payload
        )
        if not isinstance(value, Mapping):
            raise MacOpsGitLabTransportError(
                "gitlab_response_invalid", ambiguous=True
            )
        return value

    def search_issues(self, search: str) -> list[Mapping[str, Any]]:
        value = self._request(
            "GET",
            f"/projects/{self._project}/issues",
            query={"scope": "all", "state": "all", "search": search, "per_page": 20},
        )
        if not isinstance(value, list) or any(
            not isinstance(item, Mapping) for item in value
        ):
            raise MacOpsGitLabTransportError(
                "gitlab_response_invalid", ambiguous=False
            )
        return list(value)

    def read_issue(self, issue_iid: int) -> Mapping[str, Any]:
        value = self._request(
            "GET", f"/projects/{self._project}/issues/{issue_iid}"
        )
        if not isinstance(value, Mapping):
            raise MacOpsGitLabTransportError(
                "gitlab_response_invalid", ambiguous=False
            )
        return value

    def read_notes(self, issue_iid: int) -> list[Mapping[str, Any]]:
        value = self._request(
            "GET",
            f"/projects/{self._project}/issues/{issue_iid}/notes",
            query={"per_page": _MAX_NOTES, "sort": "desc"},
        )
        if not isinstance(value, list) or any(
            not isinstance(item, Mapping) for item in value
        ):
            raise MacOpsGitLabTransportError(
                "gitlab_response_invalid", ambiguous=False
            )
        return list(value)


class MacOpsJournal:
    def __init__(self, path: Path, *, busy_timeout_ms: int) -> None:
        self.path = path
        self.busy_timeout_ms = busy_timeout_ms
        self._lock = threading.Lock()
        parent = path.parent
        parent.mkdir(parents=True, exist_ok=True)
        if stat.S_IMODE(parent.stat().st_mode) & 0o022:
            raise PermissionError("journal_parent_writable_by_untrusted")
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS mac_ops_edge_tasks_v1 (
                    idempotency_key TEXT PRIMARY KEY,
                    request_sha256 TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    issue_iid INTEGER,
                    result_json TEXT,
                    receipt_json TEXT,
                    created_at_unix_ms INTEGER NOT NULL,
                    updated_at_unix_ms INTEGER NOT NULL
                );
                """
            )
        os.chmod(path, 0o600)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=self.busy_timeout_ms / 1000)
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def reserve(self, request: MacOpsEdgeRequest) -> Mapping[str, Any] | None:
        now = int(time.time() * 1000)
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM mac_ops_edge_tasks_v1 WHERE idempotency_key = ?",
                (request.idempotency_key,),
            ).fetchone()
            if row is not None:
                if row["request_sha256"] != request.intent_sha256:
                    raise MacOpsEdgeServiceError("idempotency_binding_mismatch")
                conn.commit()
                return dict(row)
            conn.execute(
                """
                INSERT INTO mac_ops_edge_tasks_v1 (
                    idempotency_key, request_sha256, request_json, state,
                    created_at_unix_ms, updated_at_unix_ms
                ) VALUES (?, ?, ?, 'pending', ?, ?)
                """,
                (
                    request.idempotency_key,
                    request.intent_sha256,
                    canonical_json_bytes(request.to_mapping()).decode("utf-8"),
                    now,
                    now,
                ),
            )
            conn.commit()
            return None

    def complete(
        self,
        request: MacOpsEdgeRequest,
        *,
        state: MacOpsEdgeState,
        issue_iid: int | None,
        result: Mapping[str, Any],
        receipt: MacOpsEdgeReceipt,
    ) -> None:
        now = int(time.time() * 1000)
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                """
                UPDATE mac_ops_edge_tasks_v1
                   SET state = ?, issue_iid = ?, result_json = ?, receipt_json = ?,
                       updated_at_unix_ms = ?
                 WHERE idempotency_key = ? AND request_sha256 = ?
                """,
                (
                    state.value,
                    issue_iid,
                    canonical_json_bytes(result).decode("utf-8"),
                    canonical_json_bytes(receipt.to_mapping()).decode("utf-8"),
                    now,
                    request.idempotency_key,
                    request.intent_sha256,
                ),
            )
            if cursor.rowcount != 1:
                raise MacOpsEdgeServiceError("journal_binding_lost")
            conn.commit()


def _issue_description(request: MacOpsEdgeRequest) -> str:
    payload = request.payload
    if not isinstance(payload, MacOpsReadOnlySubmit):
        raise TypeError("submit payload required")
    record = {
        "schema": ISSUE_SCHEMA,
        "title": payload.title,
        "task_class": payload.task_class.value,
        "approval": {"approved": False},
        "contract": payload.contract,
        "contract_sha256": payload.contract_sha256,
        "idempotency_key": request.idempotency_key,
        "edge_intent_sha256": request.intent_sha256,
    }
    return (
        ISSUE_MARKER
        + "\n\nThis is a read-only mac-ops-bridge task record. "
        "Do not paste secrets here.\n\n```json\n"
        + json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n```\n"
    )


def _validated_issue(value: Mapping[str, Any]) -> dict[str, Any]:
    issue_iid = value.get("iid")
    if type(issue_iid) is not int or issue_iid < 1:
        raise MacOpsEdgeServiceError("gitlab_issue_invalid")
    if value.get("confidential") is not True:
        raise MacOpsEdgeServiceError("gitlab_issue_not_confidential")
    state = value.get("state")
    if state not in {"opened", "closed"}:
        raise MacOpsEdgeServiceError("gitlab_issue_invalid")
    return {
        "issue_iid": issue_iid,
        "state": state,
        "title": str(value.get("title") or "")[:MAX_TITLE_CHARS],
        "web_url": str(value.get("web_url") or "")[:2048],
        "confidential": True,
        "updated_at": str(value.get("updated_at") or "")[:128] or None,
    }


def _project_note(value: Mapping[str, Any]) -> dict[str, Any] | None:
    if value.get("system") is True:
        return None
    note_id = value.get("id")
    if type(note_id) is not int or note_id < 1:
        raise MacOpsEdgeServiceError("gitlab_note_invalid")
    body = str(value.get("body") or "")
    truncated = len(body) > _MAX_NOTE_BODY_CHARS
    return {
        "id": note_id,
        "created_at": str(value.get("created_at") or "")[:128] or None,
        "body": body[:_MAX_NOTE_BODY_CHARS],
        "truncated": truncated,
    }


def _matches_reserved_issue(
    issue: Mapping[str, Any],
    *,
    request: MacOpsEdgeRequest,
) -> bool:
    description = issue.get("description")
    return bool(
        isinstance(description, str)
        and ISSUE_MARKER in description
        and request.idempotency_key in description
        and request.intent_sha256 in description
    )


class MacOpsEdgeRuntime:
    def __init__(
        self,
        *,
        config: MacOpsEdgeConfig,
        api: GitLabApi,
        journal: MacOpsJournal,
        now_ms: Callable[[], int] | None = None,
    ) -> None:
        self.config = config
        self.api = api
        self.journal = journal
        self._now_ms = now_ms or (lambda: int(time.time() * 1000))

    def execute(self, request: MacOpsEdgeRequest) -> dict[str, Any]:
        if request.deadline_unix_ms <= self._now_ms():
            raise MacOpsEdgeServiceError("request_deadline_expired")
        if request.operation is MacOpsEdgeOperation.PING:
            return self._ping(request)
        if request.operation is MacOpsEdgeOperation.READONLY_SUBMIT:
            return self._submit(request)
        return self._read(request)

    def _ping(self, request: MacOpsEdgeRequest) -> dict[str, Any]:
        payload = request.payload
        if not isinstance(payload, MacOpsPing):
            raise MacOpsEdgeServiceError("ping_payload_invalid")
        return self._response(
            request,
            state=MacOpsEdgeState.COMPLETED,
            replayed=False,
            blocker=None,
            result={"nonce": payload.nonce, "external_io": False},
            issue_iid=None,
            external_updated_at=None,
        )

    def _response(
        self,
        request: MacOpsEdgeRequest,
        *,
        state: MacOpsEdgeState,
        replayed: bool,
        blocker: str | None,
        result: Mapping[str, Any],
        issue_iid: int | None,
        external_updated_at: str | None,
    ) -> dict[str, Any]:
        receipt = MacOpsEdgeReceipt.build(
            request=request,
            state=state,
            issue_iid=issue_iid,
            external_updated_at=external_updated_at,
            service_identity_sha256=self.config.service_identity_sha256,
            recorded_at_unix_ms=self._now_ms(),
        )
        return {
            "protocol": RESPONSE_VERSION,
            "request_id": request.request_id,
            "sequence": request.sequence,
            "state": state.value,
            "replayed": replayed,
            "blocker": blocker,
            "result": dict(result),
            "receipt": receipt.to_mapping(),
        }

    def _reconcile_pending(
        self,
        request: MacOpsEdgeRequest,
        *,
        replayed: bool,
    ) -> dict[str, Any]:
        try:
            matches = [
                item
                for item in self.api.search_issues(request.idempotency_key)
                if _matches_reserved_issue(item, request=request)
            ]
        except MacOpsGitLabTransportError:
            matches = []
        if len(matches) == 1:
            result = _validated_issue(matches[0])
            response = self._response(
                request,
                state=MacOpsEdgeState.QUEUED,
                replayed=replayed,
                blocker=None,
                result=result,
                issue_iid=result["issue_iid"],
                external_updated_at=result["updated_at"],
            )
            receipt = MacOpsEdgeReceipt.from_mapping(
                response["receipt"], request=request
            )
            self.journal.complete(
                request,
                state=MacOpsEdgeState.QUEUED,
                issue_iid=result["issue_iid"],
                result=result,
                receipt=receipt,
            )
            return response
        return self._response(
            request,
            state=MacOpsEdgeState.DISPATCH_UNCERTAIN,
            replayed=replayed,
            blocker="gitlab_dispatch_uncertain",
            result={"issue_iid": None},
            issue_iid=None,
            external_updated_at=None,
        )

    def _submit(self, request: MacOpsEdgeRequest) -> dict[str, Any]:
        existing = self.journal.reserve(request)
        if existing is not None:
            if existing.get("state") != "pending":
                result = json.loads(existing.get("result_json") or "{}")
                state = MacOpsEdgeState(existing["state"])
                return self._response(
                    request,
                    state=state,
                    replayed=True,
                    blocker=None,
                    result=result,
                    issue_iid=existing.get("issue_iid"),
                    external_updated_at=result.get("updated_at"),
                )
            return self._reconcile_pending(request, replayed=True)
        payload = request.payload
        if not isinstance(payload, MacOpsReadOnlySubmit):
            raise MacOpsEdgeServiceError("submit_payload_invalid")
        try:
            issue = self.api.create_issue(
                {
                    "title": f"[mac-ops-bridge:queued] {payload.title}",
                    "description": _issue_description(request),
                    "confidential": "true",
                }
            )
        except MacOpsGitLabTransportError:
            return self._reconcile_pending(request, replayed=False)
        result = _validated_issue(issue)
        response = self._response(
            request,
            state=MacOpsEdgeState.QUEUED,
            replayed=False,
            blocker=None,
            result=result,
            issue_iid=result["issue_iid"],
            external_updated_at=result["updated_at"],
        )
        receipt = MacOpsEdgeReceipt.from_mapping(response["receipt"], request=request)
        self.journal.complete(
            request,
            state=MacOpsEdgeState.QUEUED,
            issue_iid=result["issue_iid"],
            result=result,
            receipt=receipt,
        )
        return response

    def _read(self, request: MacOpsEdgeRequest) -> dict[str, Any]:
        payload = request.payload
        issue_iid = getattr(payload, "issue_iid", None)
        if type(issue_iid) is not int:
            raise MacOpsEdgeServiceError("read_payload_invalid")
        try:
            issue = _validated_issue(self.api.read_issue(issue_iid))
            notes = [
                projected
                for item in self.api.read_notes(issue_iid)[:_MAX_NOTES]
                if (projected := _project_note(item)) is not None
            ]
        except MacOpsGitLabTransportError as exc:
            return self._response(
                request,
                state=MacOpsEdgeState.BLOCKED,
                replayed=False,
                blocker=exc.code,
                result={"issue_iid": issue_iid, "notes": []},
                issue_iid=issue_iid,
                external_updated_at=None,
            )
        state = (
            MacOpsEdgeState.COMPLETED
            if issue["state"] == "closed"
            else MacOpsEdgeState.OBSERVED
        )
        result = {**issue, "notes": notes}
        return self._response(
            request,
            state=state,
            replayed=False,
            blocker=None,
            result=result,
            issue_iid=issue_iid,
            external_updated_at=issue["updated_at"],
        )


PeerGetter = Callable[[socket.socket], MacOpsEdgePeer]


def _receive_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise OSError("connection_closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


@dataclass
class MacOpsEdgeServer:
    config: MacOpsEdgeConfig
    runtime: MacOpsEdgeRuntime
    peer_getter: PeerGetter = linux_peer_credentials
    _listener: socket.socket | None = field(default=None, init=False, repr=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _semaphore: threading.BoundedSemaphore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._semaphore = threading.BoundedSemaphore(self.config.max_connections)

    def bind(self) -> None:
        path = self.config.socket_path
        parent = path.parent
        parent_state = os.lstat(parent)
        if (
            stat.S_ISLNK(parent_state.st_mode)
            or not stat.S_ISDIR(parent_state.st_mode)
            or stat.S_IMODE(parent_state.st_mode) & 0o002
        ):
            raise PermissionError("socket_parent_invalid")
        try:
            prior = os.lstat(path)
        except FileNotFoundError:
            prior = None
        if prior is not None:
            if not stat.S_ISSOCK(prior.st_mode) or prior.st_uid != os.geteuid():  # windows-footgun: ok — macOS/Linux AF_UNIX service boundary
                raise FileExistsError("socket_path_not_replaceable")
            os.unlink(path)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(str(path))
            os.chown(path, os.geteuid(), self.config.socket_gid)  # windows-footgun: ok — macOS/Linux AF_UNIX service boundary
            os.chmod(path, 0o660)
            listener.listen(self.config.max_connections)
            listener.settimeout(0.5)
        except Exception:
            listener.close()
            raise
        self._listener = listener

    def serve_forever(self) -> None:
        if self._listener is None:
            self.bind()
        assert self._listener is not None
        while not self._stop.is_set():
            try:
                connection, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                if self._stop.is_set():
                    break
                raise
            if not self._semaphore.acquire(blocking=False):
                connection.close()
                continue
            thread = threading.Thread(
                target=self._handle_connection,
                args=(connection,),
                daemon=True,
            )
            thread.start()

    def shutdown(self) -> None:
        self._stop.set()
        if self._listener is not None:
            self._listener.close()
            self._listener = None
        try:
            state = os.lstat(self.config.socket_path)
            if stat.S_ISSOCK(state.st_mode) and state.st_uid == os.geteuid():  # windows-footgun: ok — macOS/Linux AF_UNIX service boundary
                os.unlink(self.config.socket_path)
        except FileNotFoundError:
            pass

    def _handle_connection(self, connection: socket.socket) -> None:
        try:
            peer = self.peer_getter(connection)
            if peer.uid != self.config.gateway_uid:
                return
            connection.settimeout(MAX_REQUEST_BYTES / 4096)
            header = _receive_exact(connection, _FRAME_HEADER.size)
            (size,) = _FRAME_HEADER.unpack(header)
            if size == 0 or size > MAX_REQUEST_BYTES:
                return
            value = decode_json_object(
                _receive_exact(connection, size), maximum=MAX_REQUEST_BYTES
            )
            try:
                request = MacOpsEdgeRequest.from_mapping(value)
                response = self.runtime.execute(request)
            except (MacOpsEdgeProtocolError, MacOpsEdgeServiceError) as exc:
                response = {
                    "protocol": RESPONSE_VERSION,
                    "error": exc.code,
                }
            raw = canonical_json_bytes(response)
            if len(raw) > MAX_RESPONSE_BYTES:
                raw = canonical_json_bytes(
                    {"protocol": RESPONSE_VERSION, "error": "response_too_large"}
                )
            connection.sendall(_FRAME_HEADER.pack(len(raw)) + raw)
        except (OSError, ValueError):
            return
        finally:
            connection.close()
            self._semaphore.release()


def build_service(config: MacOpsEdgeConfig) -> tuple[MacOpsEdgeRuntime, MacOpsEdgeServer]:
    api = UrllibGitLabApi.from_config(config)
    journal = MacOpsJournal(
        config.journal_path, busy_timeout_ms=config.journal_busy_timeout_ms
    )
    runtime = MacOpsEdgeRuntime(config=config, api=api, journal=journal)
    return runtime, MacOpsEdgeServer(config=config, runtime=runtime)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Muncho privileged Mac operations edge")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(Path(args.config))
    _runtime, server = build_service(config)
    stopping = threading.Event()

    def stop(_signum: int, _frame: Any) -> None:
        stopping.set()
        server.shutdown()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        server.serve_forever()
    finally:
        if not stopping.is_set():
            server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONFIG_SCHEMA",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_PROJECT_ID",
    "DEFAULT_SOCKET_PATH",
    "GitLabApi",
    "MacOpsEdgeConfig",
    "MacOpsEdgePeer",
    "MacOpsEdgeRuntime",
    "MacOpsEdgeServer",
    "MacOpsEdgeServiceError",
    "MacOpsGitLabTransportError",
    "MacOpsJournal",
    "UrllibGitLabApi",
    "build_service",
    "linux_peer_credentials",
    "load_config",
]
