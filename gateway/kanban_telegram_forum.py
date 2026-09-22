"""Telegram forum projection for Kanban tasks.

The Kanban database stays authoritative. This module creates one forum topic per
active card, subscribes that topic to lifecycle notifications, and pins a compact
status card. It is inert until ``kanban.telegram_forum.chat_id`` is configured.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)

_ACTIVE_STATUSES = frozenset({"todo", "ready", "running", "review", "blocked", "triage", "scheduled"})


def _topic_title(task: Any) -> str:
    title = re.sub(r"\s+", " ", str(getattr(task, "title", "") or "Kanban task")).strip()
    task_id = str(getattr(task, "id", "") or "").strip()
    suffix = f" · {task_id}" if task_id else ""
    return (title[: max(1, 128 - len(suffix))] + suffix) or "Kanban task"


def _next_step(status: str) -> str:
    return {
        "todo": "Attendre ses dépendances.",
        "ready": "Démarrer l’exécution.",
        "running": "Poursuivre le travail et publier les preuves.",
        "review": "Contrôler le résultat et décider.",
        "blocked": "Lever le blocage indiqué.",
        "triage": "Arbitrage Hermes requis.",
        "scheduled": "Attendre l’heure planifiée.",
    }.get(status, "Suivre l’état Kanban.")


def render_task_card(task: Any, profile_mentions: Mapping[str, Any] | None = None) -> str:
    status = str(getattr(task, "status", "") or "unknown")
    assignee = str(getattr(task, "assignee", "") or "non attribué")
    mention = str((profile_mentions or {}).get(assignee) or "").strip()
    owner = f"{mention} ({assignee})" if mention else assignee
    blocker = str(getattr(task, "last_failure_error", "") or "Aucun").strip()
    return "\n".join((
        f"📌 {getattr(task, 'title', None) or 'Tâche sans titre'}",
        f"Carte : {getattr(task, 'id', '')}",
        f"État : {status}",
        f"Responsable : {owner}",
        f"Prochaine étape : {_next_step(status)}",
        f"Blocage : {blocker[:500]}",
    ))


def _forum_config(config: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    kanban = config.get("kanban") if isinstance(config, Mapping) else None
    forum = kanban.get("telegram_forum") if isinstance(kanban, Mapping) else None
    if not isinstance(forum, Mapping) or not forum.get("enabled", True):
        return None
    chat_id = str(forum.get("chat_id") or "").strip()
    if not chat_id:
        return None
    return dict(forum, chat_id=chat_id)


async def sync_telegram_forum_tasks(
    runner: Any,
    kb: Any,
    config: Mapping[str, Any],
    *,
    notifier_profile: Optional[str],
) -> None:
    """Create at most one missing task topic per tick and pin its status card."""
    forum = _forum_config(config)
    if forum is None:
        return

    from gateway.config import Platform
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_notify as kbn

    profile = str(forum.get("profile") or notifier_profile or "default")
    adapter = runner._authorization_adapter(Platform.TELEGRAM, profile)
    if adapter is None or not hasattr(adapter, "create_handoff_thread"):
        return
    board = str(forum.get("board") or "default")
    conn = kbc.connect(board=board)
    try:
        tasks = kb.list_tasks(conn, include_archived=False, order_by="priority")
        all_subs = kbn.list_notify_subs(conn)
        group_subs = {
            sub["task_id"]: sub for sub in all_subs
            if sub.get("platform") == "telegram" and str(sub.get("chat_id")) == forum["chat_id"]
        }
        for task in tasks:
            if task.status not in _ACTIVE_STATUSES or task.id in group_subs:
                continue
            thread_id = await adapter.create_handoff_thread(forum["chat_id"], _topic_title(task))
            if not thread_id:
                return
            metadata = {"chat_type": "supergroup", "thread_id": str(thread_id)}
            text = render_task_card(task, forum.get("profile_mentions"))
            result = await adapter.send(forum["chat_id"], text, metadata=metadata)
            if getattr(result, "success", True) is False:
                raise RuntimeError(getattr(result, "error", None) or "task topic intro send failed")
            message_id = getattr(result, "message_id", None)
            if message_id and hasattr(adapter, "pin_message"):
                await adapter.pin_message(forum["chat_id"], str(message_id), thread_id=str(thread_id))
                metadata["kanban_task_message_id"] = str(message_id)
            kbn.add_notify_sub(
                conn,
                task_id=task.id,
                platform="telegram",
                chat_id=forum["chat_id"],
                thread_id=str(thread_id),
                chat_type="supergroup",
                notifier_profile=profile,
                delivery_mode="notify",
                delivery_metadata=metadata,
            )
            assignee = str(getattr(task, "assignee", "") or "").strip()
            if assignee and assignee != profile:
                try:
                    specialist = runner._authorization_adapter(Platform.TELEGRAM, assignee)
                    if specialist is not None and specialist is not adapter:
                        await specialist.send(
                            forum["chat_id"],
                            f"📌 {task.title}\n▶ {assignee} rejoint ce sujet comme responsable.",
                            metadata=metadata,
                        )
                except Exception as exc:
                    logger.warning(
                        "kanban Telegram forum: specialist %s could not join %s: %s",
                        assignee, task.id, exc,
                    )
            logger.info("kanban Telegram forum: created topic %s for %s", thread_id, task.id)
            return
    finally:
        conn.close()


async def refresh_pinned_task_card(notification: Any) -> None:
    """Update the pinned card after a lifecycle event when the adapter supports edits."""
    metadata = notification.sub.get("delivery_metadata") or {}
    message_id = metadata.get("kanban_task_message_id") if isinstance(metadata, dict) else None
    if not message_id or notification.task is None:
        return
    editor = getattr(notification.adapter, "edit_text_message", None)
    if not callable(editor):
        return
    await editor(
        notification.sub["chat_id"],
        str(message_id),
        render_task_card(notification.task),
        thread_id=notification.sub.get("thread_id"),
    )
