"""Archive stale Done Kanban tasks and their Discord forum projections.

This module is intentionally script-friendly: Hermes cron can run it as a
``no_agent`` job, while tests can import :func:`run_archival` directly.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from hermes_cli import kanban_db as kb

DEFAULT_DONE_AGE_DAYS = 7
DEFAULT_BATCH_SIZE = 25
DEFAULT_DISCORD_API_BASE = "https://discord.com/api/v10"


@dataclass
class DiscordResult:
    ok: bool
    status: str
    detail: str = ""


@dataclass
class ArchiveStats:
    board: str
    dry_run: bool
    cutoff_ts: int
    candidate_count: int = 0
    reconcile_candidate_count: int = 0
    archived_count: int = 0
    already_archived_count: int = 0
    skipped_count: int = 0
    discord_archived_locked_count: int = 0
    failures: list[dict[str, Any]] = field(default_factory=list)
    tasks: list[dict[str, Any]] = field(default_factory=list)
    skipped_by_reason: dict[str, int] = field(default_factory=dict)

    def skip(self, reason: str) -> None:
        self.skipped_count += 1
        self.skipped_by_reason[reason] = self.skipped_by_reason.get(reason, 0) + 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "board": self.board,
            "dry_run": self.dry_run,
            "cutoff_ts": self.cutoff_ts,
            "candidate_count": self.candidate_count,
            "reconcile_candidate_count": self.reconcile_candidate_count,
            "archived_count": self.archived_count,
            "already_archived_count": self.already_archived_count,
            "skipped_count": self.skipped_count,
            "skipped_by_reason": dict(sorted(self.skipped_by_reason.items())),
            "discord_archived_locked_count": self.discord_archived_locked_count,
            "failure_count": len(self.failures),
            "failures": self.failures,
            "tasks": self.tasks,
        }


class DiscordThreadClient:
    """Small stdlib Discord REST client for archive/lock reconciliation."""

    def __init__(
        self,
        token: str,
        *,
        api_base: str = DEFAULT_DISCORD_API_BASE,
        sleep: Callable[[float], None] = time.sleep,
        max_retries: int = 3,
    ) -> None:
        self.token = token.strip()
        self.api_base = api_base.rstrip("/")
        self.sleep = sleep
        self.max_retries = max(1, int(max_retries))

    def _patch_channel(self, thread_id: str, payload_data: dict[str, Any]) -> DiscordResult:
        payload = json.dumps(payload_data).encode("utf-8")
        try:
            channel_id = int(thread_id)
        except (TypeError, ValueError):
            return DiscordResult(False, "malformed_thread_id", "thread_id is not an integer")
        url = f"{self.api_base}/channels/{channel_id}"
        for attempt in range(1, self.max_retries + 1):
            req = urllib.request.Request(
                url,
                data=payload,
                method="PATCH",
                headers={
                    "Authorization": f"Bot {self.token}",
                    "Content-Type": "application/json",
                    "User-Agent": "Hermes-Kanban-Archiver/1.0",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:  # nosec: URL is fixed Discord API base by default
                    if 200 <= int(resp.status) < 300:
                        return DiscordResult(True, "patched")
                    return DiscordResult(False, f"http_{resp.status}")
            except urllib.error.HTTPError as exc:
                if exc.code == 429 and attempt < self.max_retries:
                    retry_after = _retry_after_seconds(exc)
                    self.sleep(retry_after)
                    continue
                if exc.code == 404:
                    return DiscordResult(True, "missing_thread", "Discord returned 404")
                return DiscordResult(False, f"http_{exc.code}", _safe_http_error_detail(exc))
            except Exception as exc:  # pragma: no cover - defensive network guard
                if attempt < self.max_retries:
                    self.sleep(min(2.0 * attempt, 10.0))
                    continue
                return DiscordResult(False, "request_error", str(exc)[:200])
        return DiscordResult(False, "retry_exhausted")

    def fetch_thread(self, thread_id: str) -> dict[str, Any] | None:
        """Return a small Discord thread snapshot, ``None`` for 404, or an error dict."""
        try:
            channel_id = int(thread_id)
        except (TypeError, ValueError):
            return {"error": "malformed_thread_id", "detail": "thread_id is not an integer"}
        req = urllib.request.Request(
            f"{self.api_base}/channels/{channel_id}",
            headers={
                "Authorization": f"Bot {self.token}",
                "User-Agent": "Hermes-Kanban-Archiver/1.0",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:  # nosec: URL is fixed Discord API base by default
                data = json.load(resp)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            return {"error": f"http_{exc.code}", "detail": _safe_http_error_detail(exc)}
        except Exception as exc:  # pragma: no cover - defensive network guard
            return {"error": "request_error", "detail": str(exc)[:200]}
        thread_metadata = data.get("thread_metadata") if isinstance(data.get("thread_metadata"), dict) else {}
        return {
            "id": str(data.get("id") or thread_id),
            "name": str(data.get("name") or ""),
            "applied_tags": [str(tag) for tag in data.get("applied_tags", [])],
            "archived": bool(thread_metadata.get("archived")),
            "locked": bool(thread_metadata.get("locked")),
        }

    def archive_and_lock(
        self,
        thread_id: str,
        *,
        projection_metadata: Optional[dict[str, Any]] = None,
    ) -> DiscordResult:
        """Set a Discord thread to archived+locked and reconcile forum projection metadata.

        Discord rejects some metadata edits on archived threads. To make reruns
        idempotent, first inspect the thread, no-op if already reconciled, and
        briefly unarchive before applying name/tag/lock fixes when needed.
        """
        payload_data: dict[str, Any] = {"archived": True, "locked": True}
        if projection_metadata:
            if projection_metadata.get("name"):
                payload_data["name"] = str(projection_metadata["name"])
            if projection_metadata.get("applied_tags") is not None:
                payload_data["applied_tags"] = [str(t) for t in projection_metadata.get("applied_tags") or []]

        snapshot = self.fetch_thread(thread_id)
        if snapshot is None:
            return DiscordResult(True, "missing_thread", "Discord returned 404")
        if isinstance(snapshot, dict) and snapshot.get("error"):
            if str(snapshot["error"]) == "http_403":
                return DiscordResult(True, "missing_access", str(snapshot.get("detail") or "")[:200])
            return DiscordResult(False, str(snapshot["error"]), str(snapshot.get("detail") or "")[:200])

        desired_name = payload_data.get("name")
        desired_tags = payload_data.get("applied_tags")
        name_ok = desired_name is None or str(snapshot.get("name") or "") == str(desired_name)
        tags_ok = desired_tags is None or set(str(t) for t in snapshot.get("applied_tags") or []) == set(str(t) for t in desired_tags)
        archived_ok = bool(snapshot.get("archived"))
        locked_ok = bool(snapshot.get("locked"))
        if name_ok and tags_ok and archived_ok and locked_ok:
            return DiscordResult(True, "already_archived_locked")

        if bool(snapshot.get("archived")) and (not name_ok or not tags_ok or not locked_ok):
            unarchive = self._patch_channel(thread_id, {"archived": False})
            if not unarchive.ok:
                if name_ok and tags_ok and archived_ok:
                    return DiscordResult(
                        True,
                        "archived_projection_ok_lock_unreconciled",
                        f"Could not unarchive to apply lock: {unarchive.status}",
                    )
                return unarchive

        patch = self._patch_channel(thread_id, payload_data)
        if not patch.ok:
            if patch.status == "http_403" and name_ok and tags_ok:
                return DiscordResult(
                    True,
                    "projection_ok_patch_denied",
                    "Could not archive/lock thread because Discord returned Missing Access",
                )
            return patch
        return DiscordResult(True, "archived_locked")


def _retry_after_seconds(exc: urllib.error.HTTPError) -> float:
    header = exc.headers.get("Retry-After") if exc.headers else None
    if header:
        try:
            return max(0.0, float(header))
        except ValueError:
            pass
    try:
        body = exc.read().decode("utf-8", errors="replace")
        data = json.loads(body)
        return max(0.0, float(data.get("retry_after", 1.0)))
    except Exception:
        return 1.0


def _safe_http_error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="replace")
    except Exception:
        body = ""
    return body[:300]


def _get_discord_token(profile: Optional[str] = None) -> str:
    if profile:
        candidate_paths = []
        if profile == "default":
            candidate_paths.append(os.path.expanduser("~/.hermes/.env"))
        else:
            candidate_paths.append(os.path.expanduser(f"~/.hermes/profiles/{profile}/.env"))
        for env_path in candidate_paths:
            try:
                with open(env_path, encoding="utf-8") as handle:
                    for line in handle:
                        stripped = line.strip()
                        if not stripped or stripped.startswith("#") or "=" not in stripped:
                            continue
                        key, value = stripped.split("=", 1)
                        if key == "DISCORD_BOT_TOKEN":
                            return value.strip().strip('"').strip("'")
            except OSError:
                pass
    token = os.environ.get("DISCORD_BOT_TOKEN", "").strip()
    if token:
        return token
    try:
        from gateway.config import Platform, load_gateway_config

        cfg = load_gateway_config()
        pconfig = cfg.platforms.get(Platform.DISCORD)
        return str(getattr(pconfig, "token", "") or "").strip()
    except Exception:
        return ""


def _candidate_tasks(conn, *, cutoff_ts: int, limit: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT id, title, status, assignee, completed_at
          FROM tasks
         WHERE status = 'done'
           AND completed_at IS NOT NULL
           AND completed_at <= ?
         ORDER BY completed_at ASC, id ASC
         LIMIT ?
        """,
        (int(cutoff_ts), int(limit)),
    ).fetchall()
    return [dict(r) for r in rows]


def _pending_archived_reconcile_tasks(conn, *, limit: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT DISTINCT t.id, t.title, t.status, t.assignee, t.completed_at
          FROM tasks AS t
          JOIN kanban_notify_subs AS s ON s.task_id = t.id
         WHERE t.status = 'archived'
           AND s.platform = 'discord'
           AND COALESCE(s.thread_id, '') != ''
         ORDER BY COALESCE(t.completed_at, t.created_at) ASC, t.id ASC
         LIMIT ?
        """,
        (int(limit),),
    ).fetchall()
    return [dict(r) for r in rows]


def _discord_subs(conn, task_id: str) -> list[dict[str, Any]]:
    return [s for s in kb.list_notify_subs(conn, task_id) if s.get("platform") == "discord"]


def _remove_sub(conn, sub: dict[str, Any]) -> None:
    kb.remove_notify_sub(
        conn,
        task_id=str(sub.get("task_id") or ""),
        platform=str(sub.get("platform") or ""),
        chat_id=str(sub.get("chat_id") or ""),
        thread_id=str(sub.get("thread_id") or ""),
    )


def _discord_projection_config(board: str) -> dict[str, Any]:
    try:
        path = kb.board_dir(board) / "discord-forum-tags.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def _discord_archived_projection_metadata(task: dict[str, Any], *, board: str) -> dict[str, Any]:
    """Return forum projection metadata for an archived task if configured.

    The Discord projection skill treats title/status tags as part of the forum
    projection, not just the thread's archived bit. Include the same metadata
    in the archive PATCH so the cron job does not depend on the async gateway
    notifier seeing an archived event before the subscription is removed.
    """
    config = _discord_projection_config(board)
    if not config:
        return {}
    status = "archived"
    title = str(task.get("title") or "").strip()
    assignee = str(task.get("assignee") or "").strip()
    icons = {
        "triage": "◇",
        "todo": "◻",
        "ready": "▶",
        "running": "●",
        "scheduled": "⏱",
        "blocked": "🛑",
        "review": "◆",
        "done": "✅",
        "archived": "—",
    }
    raw_tag_defs = config.get("tags")
    tag_defs: dict[str, Any] = raw_tag_defs if isinstance(raw_tag_defs, dict) else {}
    raw_status_tag_def = tag_defs.get(status)
    status_tag_def: dict[str, Any] = raw_status_tag_def if isinstance(raw_status_tag_def, dict) else {}
    icon = str(status_tag_def.get("emoji") or "").strip() or icons[status]
    raw_status_to_tag = config.get("status_to_tag")
    status_to_tag: dict[str, Any] = raw_status_to_tag if isinstance(raw_status_to_tag, dict) else {}
    raw_assignee_to_tag = config.get("assignee_to_tag")
    assignee_to_tag: dict[str, Any] = raw_assignee_to_tag if isinstance(raw_assignee_to_tag, dict) else {}
    tags: list[str] = []
    if status_to_tag.get(status):
        tags.append(str(status_to_tag[status]))
    assignee_tag = assignee_to_tag.get(assignee) or assignee_to_tag.get("default")
    if assignee_tag:
        tags.append(str(assignee_tag))
    name = f"{icon} [{status}] {title}"
    if len(name) > 100:
        name = name[:99] + "…"
    return {"name": name, "applied_tags": tags}


def run_archival(
    *,
    board: Optional[str] = None,
    done_age_days: int = DEFAULT_DONE_AGE_DAYS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    dry_run: bool = False,
    discord_token: Optional[str] = None,
    discord_client: Optional[Any] = None,
    now: Optional[int] = None,
) -> dict[str, Any]:
    """Archive stale Done tasks and reconcile Discord projections.

    Archived tasks with remaining Discord subscription rows are also retried for
    Discord reconciliation. They are counted separately as ``already_archived``
    and are not considered new Kanban archive targets.
    """
    board_name = board or kb.get_current_board()
    now_ts = int(now if now is not None else time.time())
    cutoff_ts = now_ts - int(done_age_days) * 86400
    limit = max(1, int(batch_size))
    stats = ArchiveStats(board=board_name, dry_run=bool(dry_run), cutoff_ts=cutoff_ts)

    old_board = os.environ.get("HERMES_KANBAN_BOARD")
    if board:
        os.environ["HERMES_KANBAN_BOARD"] = board
    try:
        with kb.connect_closing() as conn:
            done_candidates = _candidate_tasks(conn, cutoff_ts=cutoff_ts, limit=limit)
            remaining = max(0, limit - len(done_candidates))
            reconcile_candidates = _pending_archived_reconcile_tasks(conn, limit=remaining) if remaining else []
            candidates = done_candidates + reconcile_candidates
            stats.candidate_count = len(done_candidates)
            stats.reconcile_candidate_count = len(reconcile_candidates)
            if dry_run:
                for task in candidates:
                    subs = _discord_subs(conn, task["id"])
                    stats.tasks.append({
                        "task_id": task["id"],
                        "status": task["status"],
                        "action": "would_archive" if task["status"] == "done" else "would_reconcile_discord",
                        "discord_thread_ids": [str(s.get("thread_id") or "") for s in subs if s.get("thread_id")],
                    })
                return stats.as_dict()

            token = discord_token if discord_token is not None else _get_discord_token()
            default_client = discord_client or (DiscordThreadClient(token) if token else None)
            clients_by_profile: dict[str, DiscordThreadClient] = {}

            def client_for_sub(sub: dict[str, Any]) -> Any:
                if discord_client is not None:
                    return discord_client
                profile = str(sub.get("notifier_profile") or "").strip()
                if profile:
                    if profile not in clients_by_profile:
                        profile_token = discord_token if discord_token is not None else _get_discord_token(profile)
                        clients_by_profile[profile] = DiscordThreadClient(profile_token) if profile_token else None  # type: ignore[assignment]
                    return clients_by_profile.get(profile) or default_client
                return default_client

            for task in candidates:
                task_id = str(task["id"])
                task_entry: dict[str, Any] = {"task_id": task_id, "status": task["status"], "discord": []}
                subs = _discord_subs(conn, task_id)

                if task["status"] == "done":
                    archived = kb.archive_task(conn, task_id)
                    if archived:
                        stats.archived_count += 1
                        task_entry["action"] = "archived"
                    else:
                        stats.skip("archive_race_or_already_archived")
                        task_entry["action"] = "skipped_archive_race_or_already_archived"
                else:
                    stats.already_archived_count += 1
                    task_entry["action"] = "reconcile_discord"

                for sub in subs:
                    thread_id = str(sub.get("thread_id") or "")
                    if not thread_id:
                        stats.skip("missing_discord_thread_ref")
                        task_entry["discord"].append({"thread_id": "", "status": "missing_thread_ref"})
                        _remove_sub(conn, sub)
                        continue
                    if not thread_id.isdigit():
                        stats.skip("malformed_discord_thread_ref")
                        task_entry["discord"].append({"thread_id": thread_id, "status": "malformed_thread_ref"})
                        _remove_sub(conn, sub)
                        continue
                    sub_client = client_for_sub(sub)
                    if sub_client is None:
                        failure = {"task_id": task_id, "thread_id": thread_id, "error": "discord_token_missing"}
                        stats.failures.append(failure)
                        task_entry["discord"].append({"thread_id": thread_id, "status": "discord_token_missing"})
                        continue
                    result = sub_client.archive_and_lock(
                        thread_id,
                        projection_metadata=_discord_archived_projection_metadata(task, board=board_name),
                    )
                    task_entry["discord"].append({"thread_id": thread_id, "status": result.status})
                    if result.ok:
                        if result.status in {"archived_locked", "already_archived_locked"}:
                            stats.discord_archived_locked_count += 1
                        elif result.status == "missing_thread":
                            stats.skip("missing_or_deleted_discord_thread")
                        elif result.status == "missing_access":
                            stats.skip("discord_thread_missing_access")
                        elif result.status in {"archived_projection_ok_lock_unreconciled", "projection_ok_patch_denied"}:
                            stats.skip("discord_projection_ok_but_lock_unreconciled")
                        _remove_sub(conn, sub)
                    else:
                        stats.failures.append({
                            "task_id": task_id,
                            "thread_id": thread_id,
                            "error": result.status,
                            "detail": result.detail,
                        })
                if not subs:
                    stats.skip("no_discord_projection")
                stats.tasks.append(task_entry)
    finally:
        if board:
            if old_board is None:
                os.environ.pop("HERMES_KANBAN_BOARD", None)
            else:
                os.environ["HERMES_KANBAN_BOARD"] = old_board

    return stats.as_dict()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", help="Kanban board slug (defaults to current board)")
    parser.add_argument("--done-age-days", type=int, default=DEFAULT_DONE_AGE_DAYS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    result = run_archival(
        board=args.board,
        done_age_days=args.done_age_days,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )
    has_activity = bool(
        result.get("dry_run")
        or result.get("failure_count")
        or result.get("candidate_count")
        or result.get("reconcile_candidate_count")
        or result.get("archived_count")
        or result.get("already_archived_count")
        or result.get("skipped_count")
    )
    if has_activity:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if result.get("failure_count") else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
