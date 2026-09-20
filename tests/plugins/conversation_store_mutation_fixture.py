from copy import deepcopy
import time

from conversation_store import (
    ConversationConflictError, ConversationMutationResult, ConversationRevision, ConversationStore,
)

class MutationStore(ConversationStore):
    def __init__(self):
        self.rev = {}
        self.conversations = {}
        self.messages = {}
        self.next_id = 10

    @property
    def name(self):
        return "mutation-store"

    def is_available(self):
        return True

    def ensure_conversation(self, conversation):
        sid = conversation["id"]
        self.rev.setdefault(sid, 1)
        self.conversations.setdefault(sid, deepcopy(conversation))
        self.messages.setdefault(sid, [])
        return self.get_revision(sid)

    def get_conversation(self, conversation_id):
        row = self.conversations.get(conversation_id)
        return deepcopy(row) if row is not None else None

    def get_revision(self, conversation_id):
        return ConversationRevision(self.rev[conversation_id])

    def active_message_ids(self, conversation_id):
        return [m["_row_id"] for m in self.messages[conversation_id] if m.get("active", True)]

    def conversation_history(self, conversation_id, *, include_inactive=False, **_kwargs):
        rows = self.messages.get(conversation_id, [])
        return [
            deepcopy(row) for row in rows
            if include_inactive or row.get("active", True)
        ]

    def _check(self, sid, expected_revision, expected_active_ids=None):
        if expected_revision != self.get_revision(sid):
            raise ConversationConflictError("stale revision")
        if expected_active_ids is not None:
            if list(expected_active_ids) != self.active_message_ids(sid):
                raise ConversationConflictError("stale active message ids")

    def _new_row(self, message):
        self.next_id += 1
        return {"active": True, **deepcopy(message), "_row_id": self.next_id}

    def append_messages(self, conversation_id, messages, *, expected_revision, **_kwargs):
        self._check(conversation_id, expected_revision)
        rows = [self._new_row(message) for message in messages]
        self.messages[conversation_id].extend(rows)
        self.rev[conversation_id] += 1
        return ConversationMutationResult(
            revision=self.get_revision(conversation_id),
            affected_count=len(rows),
            message_ids=tuple(row["_row_id"] for row in rows),
            canonical_messages=tuple(deepcopy(rows)),
        )

    def replace_messages(
        self, conversation_id, messages, *, expected_revision, expected_active_ids,
        active_only=False, archive_dropped=False,
    ):
        self._check(conversation_id, expected_revision, expected_active_ids)
        if archive_dropped:
            for row in self.messages[conversation_id]:
                if row.get("active", True):
                    row["active"] = False
        else:
            self.messages[conversation_id] = [
                row for row in self.messages[conversation_id]
                if active_only and not row.get("active", True)
            ]
        rows = [self._new_row(message) for message in messages]
        self.messages[conversation_id].extend(rows)
        self.rev[conversation_id] += 1
        return ConversationMutationResult(
            revision=self.get_revision(conversation_id),
            affected_count=len(rows),
            message_ids=tuple(row["_row_id"] for row in rows),
            canonical_messages=tuple(deepcopy(rows)),
        )

    def rewind_to_message(
        self, conversation_id, message_id, *, expected_revision, expected_active_ids,
        expected_target_content=None, preserve_compaction_handoff=False,
    ):
        self._check(conversation_id, expected_revision, expected_active_ids)
        active = [row for row in self.messages[conversation_id] if row.get("active", True)]
        target = next((row for row in active if row["_row_id"] == message_id), None)
        if target is None or target.get("role") != "user":
            raise ValueError("rewind target unavailable")
        if expected_target_content is not None and target.get("content") != expected_target_content:
            raise ConversationConflictError("rewind target changed")
        index = active.index(target)
        rewound = active[index:]
        for row in rewound:
            row["active"] = False
        replacement_id = None
        if preserve_compaction_handoff:
            replacement = self._new_row({
                "role": "user", "content": "[handoff]", "display_kind": "hidden",
            })
            self.messages[conversation_id].append(replacement)
            replacement_id = replacement["_row_id"]
        self.rev[conversation_id] += 1
        remaining = [row for row in self.messages[conversation_id] if row.get("active", True)]
        details = {
            "target_message": deepcopy(target),
            "new_head_id": remaining[-1]["_row_id"] if remaining else None,
            "message_count": len(remaining),
            "tool_call_count": 0,
        }
        if preserve_compaction_handoff:
            details["replacement_message_id"] = replacement_id
        return ConversationMutationResult(
            revision=self.get_revision(conversation_id),
            affected_count=len(rewound),
            details=details,
        )

    def set_latest_matching_message_display(
        self, conversation_id, *, role, content, display_kind, display_metadata,
        expected_revision,
    ):
        self._check(conversation_id, expected_revision)
        match = next((
            row for row in reversed(self.messages[conversation_id])
            if row.get("active", True) and row.get("role") == role and row.get("content") == content
        ), None)
        if match is None:
            return ConversationMutationResult(revision=self.get_revision(conversation_id))
        match["display_kind"] = display_kind
        match["display_metadata"] = deepcopy(display_metadata)
        self.rev[conversation_id] += 1
        return ConversationMutationResult(
            revision=self.get_revision(conversation_id), affected_count=1)

    def get_message_reactions(self, conversation_id, message_id):
        row = next((m for m in self.messages[conversation_id] if m["_row_id"] == message_id), None)
        return deepcopy((row or {}).get("reactions", []))

    def set_message_reaction(
        self, conversation_id, message_id, emoji, *, author, expected_revision,
    ):
        self._check(conversation_id, expected_revision)
        row = next((
            m for m in self.messages[conversation_id]
            if m["_row_id"] == message_id and m.get("active", True)
        ), None)
        if row is None:
            return ConversationMutationResult(
                revision=self.get_revision(conversation_id), details=None)
        reactions = [r for r in row.get("reactions", []) if r.get("author") != author]
        previous = next((r for r in row.get("reactions", []) if r.get("author") == author), None)
        if emoji and (previous is None or previous.get("emoji") != emoji):
            reactions.append({"emoji": emoji, "author": author, "at": time.time()})
        row["reactions"] = reactions
        self.rev[conversation_id] += 1
        return ConversationMutationResult(
            revision=self.get_revision(conversation_id), affected_count=1,
            details=deepcopy(reactions))

    def set_conversation_title(self, conversation_id, title, *, source, expected_revision):
        self._check(conversation_id, expected_revision)
        row = self.conversations[conversation_id]
        row["title"] = title
        row["title_source"] = source if title else None
        self.rev[conversation_id] += 1
        return ConversationMutationResult(
            revision=self.get_revision(conversation_id), affected_count=1)

    def set_conversation_title_source(self, conversation_id, source, *, expected_revision):
        self._check(conversation_id, expected_revision)
        if not self.conversations[conversation_id].get("title"):
            return ConversationMutationResult(revision=self.get_revision(conversation_id))
        self.conversations[conversation_id]["title_source"] = source
        self.rev[conversation_id] += 1
        return ConversationMutationResult(
            revision=self.get_revision(conversation_id), affected_count=1)

    def update_conversation(
        self, conversation_id, changes, *, expected_revision, include_lineage=False,
    ):
        self._check(conversation_id, expected_revision)
        self.conversations[conversation_id].update(deepcopy(changes))
        self.rev[conversation_id] += 1
        return ConversationMutationResult(
            revision=self.get_revision(conversation_id), affected_count=1)
