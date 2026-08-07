#!/usr/bin/env python3
"""Archive completed Jack-Hermes sessions before deleting local history."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import stat
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

HERMES_HOME = Path(
    os.environ.get(
        "HERMES_ARCHIVE_HOME",
        "/opt/rent-oleg-runtime/data/hermes_jack_v2",
    )
)
HERMES_SOURCE = Path(
    os.environ.get("HERMES_ARCHIVE_SOURCE", str(HERMES_HOME / "hermes-agent"))
)
WEBUI_SOURCE = Path(
    os.environ.get("HERMES_ARCHIVE_WEBUI_SOURCE", "/opt/hermes-jack-webui")
)
WEBUI_STATE = HERMES_HOME / "webui"
WEBUI_SESSIONS = WEBUI_STATE / "sessions"
WEBUI_URL = "http://127.0.0.1:8787"
WEBUI_ARCHIVE_COOKIE = HERMES_HOME / ".webui-archive-session"
STATE_DB = HERMES_HOME / "state.db"
SESSIONS_DIR = HERMES_HOME / "sessions"
VAULT = Path(
    os.environ.get(
        "HERMES_ARCHIVE_VAULT",
        "/opt/rent-oleg-runtime/data/obsidian/jack_hermes_v2",
    )
)
SESSION_ARCHIVE = VAULT / "10 Sessions"
EVENT_LOG = VAULT / "90 System/Archive Logs/archive-events.jsonl"
SYNCTHING_CONFIG = Path("/var/lib/hermes-jack/.config/syncthing/config.xml")
SYNCTHING_API = "http://127.0.0.1:8384/rest"
SYNCTHING_FOLDER = "jack-hermes-v2"
MAC_DEVICE_ID = os.environ.get("HERMES_ARCHIVE_SYNC_DEVICE_ID", "").strip()

sys.path.insert(0, str(HERMES_SOURCE))
sys.path.insert(0, str(WEBUI_SOURCE))

from hermes_cli.session_export_md import (  # noqa: E402
    append_manifest_entry,
    redact_session_data,
    render_session_markdown,
    safe_session_filename,
    verify_export_file,
    write_session_markdown,
)
from hermes_state import SessionDB  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def syncthing_remote_device_id() -> str:
    if not MAC_DEVICE_ID:
        raise RuntimeError("HERMES_ARCHIVE_SYNC_DEVICE_ID is not configured")
    return MAC_DEVICE_ID


def append_event(event: dict) -> None:
    EVENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    payload = {"at": utc_now(), **event}
    with EVENT_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def syncthing_api_key() -> str:
    root = ET.parse(SYNCTHING_CONFIG).getroot()
    key = root.findtext("./gui/apikey")
    if not key:
        raise RuntimeError("Syncthing API key is missing")
    return key


def syncthing_json(
    endpoint: str,
    *,
    params: dict[str, str] | None = None,
    method: str = "GET",
) -> dict:
    query = "?" + urllib.parse.urlencode(params or {}) if params else ""
    request = urllib.request.Request(
        SYNCTHING_API + endpoint + query,
        data=b"" if method == "POST" else None,
        method=method,
        headers={"X-API-Key": syncthing_api_key()},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response) if response.length != 0 else {}


def remote_connected() -> bool:
    device_id = syncthing_remote_device_id()
    payload = syncthing_json("/system/connections")
    return bool(
        payload.get("connections", {})
        .get(device_id, {})
        .get("connected")
    )


def trigger_scan(relative_path: str) -> None:
    syncthing_json(
        "/db/scan",
        params={"folder": SYNCTHING_FOLDER, "sub": relative_path},
        method="POST",
    )


def local_index_has_file(relative_path: str, expected_size: int) -> bool:
    try:
        payload = syncthing_json(
            "/db/file",
            params={"folder": SYNCTHING_FOLDER, "file": relative_path},
        )
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        raise
    local = payload.get("local") or {}
    return (
        local.get("name") == relative_path
        and int(local.get("size") or -1) == expected_size
        and not local.get("deleted", False)
        and not local.get("invalid", False)
    )


def remote_complete() -> bool:
    device_id = syncthing_remote_device_id()
    if not remote_connected():
        return False
    payload = syncthing_json(
        "/db/completion",
        params={"device": device_id, "folder": SYNCTHING_FOLDER},
    )
    return (
        payload.get("remoteState") == "valid"
        and float(payload.get("completion") or 0) == 100.0
        and int(payload.get("needItems") or 0) == 0
        and int(payload.get("needBytes") or 0) == 0
        and int(payload.get("needDeletes") or 0) == 0
    )


def wait_for_remote(relative_path: str, expected_size: int, timeout: int) -> bool:
    """Wait for Syncthing to index the file and (if online) confirm remote delivery.

    Returns True if remote confirmed, False if remote was offline and only
    local indexing completed.  When the remote device is offline we do not
    block the archive pipeline — the file is already in the local vault and
    Syncthing (sendonly) will sync it when the remote reappears.
    """
    deadline = time.monotonic() + timeout
    indexed = False
    remote_was_online = remote_connected()
    while time.monotonic() < deadline:
        if not indexed:
            indexed = local_index_has_file(relative_path, expected_size)
        if indexed and remote_complete():
            return True
        if indexed and not remote_was_online:
            # Remote device is offline — local index is sufficient.
            # Syncthing will deliver when the remote reconnects.
            return False
        time.sleep(2)
    raise TimeoutError(
        f"Syncthing indexing timed out for {relative_path}"
    )


def webui_sidecar(session_id: str) -> Path:
    candidate = (WEBUI_SESSIONS / f"{session_id}.json").resolve()
    candidate.relative_to(WEBUI_SESSIONS.resolve())
    return candidate


def durable_message_count(data: dict) -> int:
    segments = data.get("segments")
    if isinstance(segments, list) and segments:
        return sum(
            len(segment.get("messages") or [])
            for segment in segments
            if isinstance(segment, dict)
        )
    return len(data.get("messages") or [])


def export_snapshot_fingerprint(data: dict) -> str:
    """Hash only durable transcript identity/content, excluding live counters."""
    segments = data.get("segments")
    if not isinstance(segments, list) or not segments:
        segments = [data]
    payload = {
        "session_id": str(data.get("id") or data.get("session_id") or ""),
        "lineage_session_ids": list(data.get("lineage_session_ids") or []),
        "segments": [
            {
                "id": str(segment.get("id") or segment.get("session_id") or ""),
                "messages": list(segment.get("messages") or []),
            }
            for segment in segments
            if isinstance(segment, dict)
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def manifest_source_fingerprint(output_dir: Path, export_path: Path) -> str | None:
    manifest = output_dir / "manifest.jsonl"
    if not manifest.exists():
        return None
    expected_path = str(export_path)
    found = None
    for raw_line in manifest.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        try:
            entry = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if str(entry.get("path") or "") == expected_path:
            found = entry.get("source_fingerprint")
    return str(found) if found else None


def revision_export_path(base_path: Path, source_fingerprint: str) -> Path:
    return base_path.with_name(
        f"{base_path.stem}-rev-{source_fingerprint[:12]}{base_path.suffix}"
    )


def write_export_revision(
    data: dict,
    output_dir: Path,
    export_path: Path,
    source_fingerprint: str,
) -> Path:
    """Write one immutable export revision and record its source snapshot."""
    if export_path.exists():
        ok, reason = verify_export_file(export_path, data)
        if not ok:
            raise RuntimeError(f"existing export revision failed verification: {reason}")
        recorded = manifest_source_fingerprint(output_dir, export_path)
        if recorded and recorded != source_fingerprint:
            raise RuntimeError("existing export revision fingerprint mismatch")
        if not recorded:
            append_manifest_entry(
                output_dir,
                data,
                export_path,
                fmt="md",
                source_fingerprint=source_fingerprint,
            )
        return export_path

    output_dir.mkdir(parents=True, exist_ok=True)
    with export_path.open("x", encoding="utf-8") as handle:
        handle.write(render_session_markdown(data, fmt="md"))
    append_manifest_entry(
        output_dir,
        data,
        export_path,
        fmt="md",
        source_fingerprint=source_fingerprint,
    )
    return export_path


def webui_session_is_active(session_id: str) -> bool:
    sidecar = webui_sidecar(session_id)
    if sidecar.exists():
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        if (
            data.get("active_stream_id")
            or data.get("pending_started_at")
            or data.get("pending_user_message")
        ):
            return True

    request = urllib.request.Request(
        WEBUI_URL + "/health",
        headers={"Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        health = json.load(response)
    for run in health.get("runs") or []:
        if str((run or {}).get("session_id") or "") == session_id:
            return True
    return False


def webui_archive_cookie() -> str:
    info = WEBUI_ARCHIVE_COOKIE.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
        raise RuntimeError(
            f"unsafe WebUI archive credential permissions: {WEBUI_ARCHIVE_COOKIE}"
        )
    value = WEBUI_ARCHIVE_COOKIE.read_text(encoding="utf-8").strip()
    if not value:
        raise RuntimeError("WebUI archive credential is empty")
    return value


def delete_webui_session(session_id: str) -> None:
    if webui_session_is_active(session_id):
        raise RuntimeError(f"refusing to delete active WebUI session: {session_id}")

    cookie_value = webui_archive_cookie()
    from api.auth import (  # noqa: PLC0415
        _resolve_cookie_name,
        csrf_token_for_session,
        verify_session,
    )

    if not verify_session(cookie_value):
        raise RuntimeError("WebUI archive credential is invalid or expired")
    csrf_token = csrf_token_for_session(cookie_value)
    if not csrf_token:
        raise RuntimeError("could not derive WebUI archive CSRF token")

    payload = json.dumps({"session_id": session_id}).encode("utf-8")
    request = urllib.request.Request(
        WEBUI_URL + "/api/session/delete",
        data=payload,
        method="POST",
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Cookie": f"{_resolve_cookie_name()}={cookie_value}",
            "X-Hermes-CSRF-Token": csrf_token,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read(512).decode("utf-8", errors="replace")
        raise RuntimeError(
            f"WebUI session delete returned HTTP {exc.code}: {body}"
        ) from exc
    if result.get("ok") is not True:
        raise RuntimeError(f"WebUI session delete failed: {result}")
    if webui_sidecar(session_id).exists():
        raise RuntimeError(f"WebUI sidecar survived deletion: {session_id}")


def ended_candidates(db: SessionDB, session_id: str | None, min_age: int) -> list[str]:
    cutoff = time.time() - min_age
    all_rows = db.list_sessions_rich(
        limit=1000,
        include_children=True,
        include_archived=False,
        project_compression_tips=False,
        order_by_last_active=True,
        min_message_count=0,
        compact_rows=True,
    )
    if session_id:
        resolved = db.resolve_session_id(session_id)
        rows = (
            [row for row in all_rows if row.get("id") == resolved]
            if resolved
            else []
        )
        if resolved and not rows:
            rows = [{"id": resolved}]
    else:
        rows = all_rows

    rows_by_id = {str(row.get("id") or ""): row for row in rows}
    candidates: list[str] = []
    seen: set[str] = set()
    for row in rows:
        candidate_id = row.get("id")
        if not candidate_id:
            continue
        lineage = db.get_compression_lineage(candidate_id)
        if not lineage:
            continue
        tip = db.get_session(lineage[-1])
        if not tip or tip.get("ended_at") is None:
            continue
        if float(tip["ended_at"]) > cutoff:
            continue
        tip_row = rows_by_id.get(str(tip.get("id") or ""), tip)
        last_active = float(
            tip_row.get("last_active")
            or tip_row.get("started_at")
            or tip.get("ended_at")
            or 0
        )
        if last_active > cutoff:
            # A WebUI session may resume under the same ID after an idle seal.
            # Wait a full grace period from actual activity, not stale ended_at.
            continue
        logical_id = lineage[0]
        if logical_id not in seen:
            seen.add(logical_id)
            candidates.append(logical_id)
    return candidates


def idle_webui_candidates(db: SessionDB, min_idle: int) -> list[str]:
    """Return inactive WebUI sessions that can be safely sealed.

    A row is eligible only when both persistence layers agree that it is idle:
    the Hermes DB has seen no activity for ``min_idle`` seconds and WebUI has
    neither a live run nor pending input for the session.
    """
    cutoff = time.time() - max(min_idle, 0)
    rows = db.list_sessions_rich(
        limit=1000,
        include_children=False,
        include_archived=True,
        project_compression_tips=False,
        order_by_last_active=True,
        min_message_count=1,
        compact_rows=True,
    )
    candidates: list[str] = []
    for row in rows:
        session_id = str(row.get("id") or "")
        if not session_id or row.get("source") != "webui":
            continue
        if row.get("ended_at") is not None:
            continue
        last_active = float(row.get("last_active") or row.get("started_at") or 0)
        if last_active <= 0 or last_active > cutoff:
            continue
        if webui_session_is_active(session_id):
            continue
        candidates.append(session_id)
    return candidates


def seal_idle_webui_sessions(db: SessionDB, min_idle: int) -> list[str]:
    """Mark safely idle WebUI sessions complete so the archive gate can export them."""
    sealed: list[str] = []
    for session_id in idle_webui_candidates(db, min_idle):
        # Re-check immediately before the mutation to close the check/use gap.
        if webui_session_is_active(session_id):
            continue
        db.end_session(session_id, "webui_idle_archive")
        row = db.get_session(session_id)
        if not row or row.get("ended_at") is None:
            raise RuntimeError(f"failed to seal idle WebUI session: {session_id}")
        sealed.append(session_id)
        append_event(
            {
                "event": "session_sealed",
                "session_id": session_id,
                "source": "webui",
                "end_reason": "webui_idle_archive",
            }
        )
    return sealed


def seal_idle_non_webui_sessions(db: SessionDB, min_idle: int) -> list[str]:
    """Mark idle CLI, Telegram, and ACP sessions complete so the archive gate can export them.

    Unlike WebUI sessions, these sources have no live-stream probe — once they
    have been inactive for ``min_idle`` seconds they are considered safely
    sealable.  Subagent sessions are NOT handled here: they already receive
    ``ended_at`` from the agent loop and are picked up by ``ended_candidates``
    directly.
    """
    cutoff = time.time() - max(min_idle, 0)
    rows = db.list_sessions_rich(
        limit=1000,
        include_children=False,
        include_archived=False,
        project_compression_tips=False,
        order_by_last_active=True,
        min_message_count=0,
        compact_rows=True,
    )
    sealable_sources = {"cli", "telegram", "acp"}
    sealed: list[str] = []
    for row in rows:
        session_id = str(row.get("id") or "")
        source = row.get("source")
        if not session_id or source not in sealable_sources:
            continue
        if row.get("ended_at") is not None:
            continue
        last_active = float(row.get("last_active") or row.get("started_at") or 0)
        if last_active <= 0 or last_active > cutoff:
            continue
        db.end_session(session_id, "idle_archive_seal")
        row2 = db.get_session(session_id)
        if not row2 or row2.get("ended_at") is None:
            raise RuntimeError(f"failed to seal idle {source} session: {session_id}")
        sealed.append(session_id)
        append_event(
            {
                "event": "session_sealed",
                "session_id": session_id,
                "source": source,
                "end_reason": "idle_archive_seal",
                "last_active": last_active,
            }
        )
    return sealed


def archive_one(
    db: SessionDB,
    session_id: str,
    *,
    delete_after_sync: bool,
    timeout: int,
) -> dict:
    data = db.export_session_lineage(session_id, include_compacted=True)
    if not data:
        raise RuntimeError(f"session disappeared before export: {session_id}")
    if data.get("source") == "webui" and webui_session_is_active(session_id):
        raise RuntimeError(f"refusing to export active WebUI session: {session_id}")
    data = redact_session_data(data)
    source_fingerprint = export_snapshot_fingerprint(data)

    ended_at = data.get("ended_at") or time.time()
    month = datetime.fromtimestamp(float(ended_at), tz=timezone.utc)
    output_dir = SESSION_ARCHIVE / f"{month:%Y}" / f"{month:%m}"
    output_dir.mkdir(parents=True, exist_ok=True)
    base_export_path = output_dir / safe_session_filename(data, fmt="md")
    export_path = base_export_path

    if export_path.exists():
        ok, reason = verify_export_file(export_path, data)
        recorded_fingerprint = manifest_source_fingerprint(output_dir, export_path)
        if ok and recorded_fingerprint == source_fingerprint:
            pass
        elif not ok and reason != "message count mismatch":
            raise RuntimeError(f"existing export failed verification: {reason}")
        else:
            # Preserve the earlier internally-valid snapshot.  A resumed
            # session or later compaction legitimately changes the source;
            # publishing a content-addressed revision avoids overwriting it.
            export_path = revision_export_path(
                base_export_path,
                source_fingerprint,
            )
            write_export_revision(
                data,
                output_dir,
                export_path,
                source_fingerprint,
            )
    else:
        export_path = write_session_markdown(
            data,
            output_dir,
            fmt="md",
            force=False,
        )
        append_manifest_entry(
            output_dir,
            data,
            export_path,
            fmt="md",
            source_fingerprint=source_fingerprint,
        )

    ok, reason = verify_export_file(export_path, data)
    if not ok:
        raise RuntimeError(f"export verification failed: {reason}")

    relative_path = export_path.relative_to(VAULT).as_posix()
    trigger_scan(relative_path)
    remote_confirmed = wait_for_remote(relative_path, export_path.stat().st_size, timeout)

    fresh_data = db.export_session_lineage(session_id, include_compacted=True)
    if not fresh_data:
        raise RuntimeError(f"session disappeared during archive: {session_id}")
    fresh_data = redact_session_data(fresh_data)
    if export_snapshot_fingerprint(fresh_data) != source_fingerprint:
        raise RuntimeError(f"session changed during archive: {session_id}")
    if data.get("source") == "webui" and webui_session_is_active(session_id):
        raise RuntimeError(f"refusing to delete active WebUI session: {session_id}")

    lineage = list(data.get("lineage_session_ids") or [data["id"]])
    deleted: list[str] = []
    webui_deleted: list[str] = []
    if delete_after_sync:
        for lineage_id in reversed(lineage):
            if webui_sidecar(lineage_id).exists():
                delete_webui_session(lineage_id)
                webui_deleted.append(lineage_id)
        for lineage_id in reversed(lineage):
            if db.delete_session(lineage_id, sessions_dir=SESSIONS_DIR):
                deleted.append(lineage_id)
            elif not db.get_session(lineage_id):
                deleted.append(lineage_id)

    result = {
        "event": "session_archived",
        "session_id": session_id,
        "lineage_session_ids": lineage,
        "archive_path": relative_path,
        "archive_sha256": hashlib.sha256(export_path.read_bytes()).hexdigest(),
        "archive_message_count": durable_message_count(data),
        "source_fingerprint": source_fingerprint,
        "remote_device": syncthing_remote_device_id(),
        "remote_confirmed": remote_confirmed,
        "deleted_session_ids": deleted,
        "deleted_webui_session_ids": webui_deleted,
        "delete_after_sync": delete_after_sync,
    }
    append_event(result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id")
    parser.add_argument("--min-age-seconds", type=int, default=900)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-sessions", type=int, default=20)
    parser.add_argument("--delete-after-sync", action="store_true")
    parser.add_argument(
        "--seal-idle-webui-seconds",
        type=int,
        default=0,
        help="mark inactive WebUI sessions complete before selecting archive candidates",
    )
    parser.add_argument(
        "--list-idle-webui",
        action="store_true",
        help="print inactive WebUI candidates without changing state",
    )
    parser.add_argument(
        "--seal-idle-non-webui-seconds",
        type=int,
        default=0,
        help="mark inactive CLI/Telegram/ACP sessions complete before selecting archive candidates",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    HERMES_HOME.mkdir(parents=True, exist_ok=True)
    lock_path = HERMES_HOME / "session-archive.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("archive run already in progress")
            return 0

        if not STATE_DB.exists():
            print("no state.db; nothing to archive")
            return 0

        db = SessionDB(db_path=STATE_DB)
        try:
            if args.list_idle_webui:
                candidates = idle_webui_candidates(
                    db,
                    max(args.seal_idle_webui_seconds, 0),
                )
                print(json.dumps({"idle_webui_sessions": candidates}, sort_keys=True))
                return 0

            if args.seal_idle_webui_seconds > 0:
                sealed = seal_idle_webui_sessions(
                    db,
                    args.seal_idle_webui_seconds,
                )
                if sealed:
                    print(
                        json.dumps(
                            {"sealed_idle_webui_sessions": sealed},
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    )

            if args.seal_idle_non_webui_seconds > 0:
                sealed_non_webui = seal_idle_non_webui_sessions(
                    db,
                    args.seal_idle_non_webui_seconds,
                )
                if sealed_non_webui:
                    print(
                        json.dumps(
                            {"sealed_idle_non_webui_sessions": sealed_non_webui},
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    )

            candidates = ended_candidates(
                db,
                args.session_id,
                max(args.min_age_seconds, 0),
            )[: max(args.max_sessions, 0)]
            if not candidates:
                print("no completed sessions eligible for archive")
                return 0

            failures = 0
            for session_id in candidates:
                try:
                    result = archive_one(
                        db,
                        session_id,
                        delete_after_sync=args.delete_after_sync,
                        timeout=max(args.timeout, 1),
                    )
                    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
                except Exception as exc:
                    failures += 1
                    append_event(
                        {
                            "event": "session_archive_failed",
                            "session_id": session_id,
                            "error": str(exc),
                            "delete_after_sync": args.delete_after_sync,
                        }
                    )
                    print(f"archive failed for {session_id}: {exc}", file=sys.stderr)
            return 1 if failures else 0
        finally:
            db.close()


if __name__ == "__main__":
    raise SystemExit(main())
