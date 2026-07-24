"""Action Item lifecycle and semantic duplicate guard."""

from __future__ import annotations

import hashlib
import json
import re
import time

from .models import ActionItem
from .state import StateStore


def action_fingerprint(item: ActionItem) -> str:
    normalized = re.sub(r"[^0-9A-Za-z가-힣]+", "", f"{item.title}{item.description}").lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def persist_new_actions(
    state: StateStore, meeting_id: str, actions: list[ActionItem]
) -> list[ActionItem]:
    unique: list[ActionItem] = []
    with state.connect() as connection:
        open_rows = connection.execute(
            """
            SELECT fingerprint FROM action_items
            WHERE status IN ('open', 'in_progress', 'blocked')
            """
        ).fetchall()
        fingerprints = {str(row["fingerprint"]) for row in open_rows}
        for item in actions:
            fingerprint = action_fingerprint(item)
            if fingerprint in fingerprints:
                continue
            connection.execute(
                """
                INSERT OR IGNORE INTO action_items(
                    action_id, meeting_id, fingerprint, item_json, status, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    item.action_id,
                    meeting_id,
                    fingerprint,
                    json.dumps(
                        {
                            "action_id": item.action_id,
                            "title": item.title,
                            "description": item.description,
                            "source_message_ids": item.source_message_ids,
                            "owner": item.owner,
                            "priority": item.priority,
                            "status": item.status,
                            "deadline": item.deadline,
                            "project_id": item.project_id,
                            "rationale": item.rationale,
                        },
                        ensure_ascii=False,
                    ),
                    item.status,
                    time.time(),
                ),
            )
            fingerprints.add(fingerprint)
            unique.append(item)
    return unique
