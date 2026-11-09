"""Titles, model config, git metadata and lineage flags of the in-memory session reference, mixed into ``InMemorySessionRepository`` (``sessions.py``), which supplies the row storage and the record builder."""

from __future__ import annotations

import copy
import json
import re
import time
from typing import Any, Callable, Mapping, Optional

from persistence import JsonValue, ProfileScope, SessionId
from persistence.conversation.sessions import (
    CANONICAL_BOT_CHAT_TITLE,
    END_REASON_COMPRESSION,
    RECOVERABLE_END_REASONS,
    RUNTIME_LOCK_KEY,
    YOLO_KEY,
    ArchiveChange,
    CanonicalTitleLocked,
    ONBOARDING_RUN_KEY,
    OnboardingAdvance,
    ReplaceSessionConfig,
    RuntimeLock,
    SessionLineageKind,
    SessionRecord,
    TitleConflict,
    TitleSource,
    sanitize_title,
)

MARKER_BY_KIND = {
    SessionLineageKind.DELEGATE: "_delegate_from",
    SessionLineageKind.BRANCH: "_branched_from",
    SessionLineageKind.RESET: "_reset_from",
}

# A title without provenance ranks as the user's.
_TITLE_RANK = {None: 2, TitleSource.DERIVED: 0, TitleSource.LLM: 1, TitleSource.USER: 2}
_NUMBERED_TITLE_RE = re.compile(r"^(.*?) #(\d+)$")
_ASCII_LOWER = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")


def ascii_lower(text: str) -> str:
    """ASCII-only case fold, the store's ``LIKE`` comparison."""
    return text.translate(_ASCII_LOWER)


def sync_lineage_markers(row: dict[str, Any]) -> None:
    """Project the first nonblank string marker without changing its stored value."""
    row["lineage_kind"] = SessionLineageKind.NONE
    row["lineage_origin_id"] = None
    for kind, marker in MARKER_BY_KIND.items():
        origin = row["model_config"].get(marker)
        if isinstance(origin, str) and origin.strip():
            row["lineage_kind"] = kind
            row["lineage_origin_id"] = origin
            break


def tool_pin_text(tool_pin: Optional[JsonValue]) -> Optional[str]:
    """The pin as the reference keeps it: one encoded document (the store content-addresses the same text), so no caller-held object aliases the stored pin."""
    return None if tool_pin is None else json.dumps(tool_pin)


class SessionTitlesConfigMixin:
    """Title, config and flag verbs; ``InMemorySessionRepository`` supplies ``_backend``, ``_table``, ``_record``, ``_lineage_ids``, ``_is_fork_child_row`` and ``_is_empty_ended``."""

    _backend: Any
    _table: Callable[[ProfileScope], dict[str, dict[str, Any]]]
    _record: Callable[[ProfileScope, Mapping[str, Any]], SessionRecord]
    _lineage_ids: Callable[[dict[str, dict[str, Any]], str], list[str]]
    _is_fork_child_row: Callable[..., bool]
    _is_empty_ended: Callable[[ProfileScope, Mapping[str, Any]], bool]

    # ── titles ────────────────────────────────────────────────────────────

    def _is_compression_ancestor(self, table: dict[str, dict[str, Any]], ancestor_id: str, descendant_id: str) -> bool:
        """Whether ``ancestor_id`` is a compression predecessor of ``descendant_id`` over continuation edges."""
        current = table.get(descendant_id)
        seen = {descendant_id}
        while current is not None:
            parent = table.get(current["parent_session_id"]) if current["parent_session_id"] else None
            if parent is None or parent["end_reason"] != END_REASON_COMPRESSION or parent["session_id"] in seen:
                return False
            if self._is_fork_child_row(current, include_reset=True):
                return False
            if parent["session_id"] == ancestor_id:
                return True
            seen.add(parent["session_id"])
            current = parent
        return False

    def _bot_chat_title_released(self, scope: ProfileScope, row: Mapping[str, Any]) -> bool:
        """Whether an archived row no longer holds the canonical name: archived deliberately (not ``auto_archived``) and its compression tip's ``end_reason`` not recoverable."""
        from .lineage import compression_tip

        if not row["archived"] or row.get("auto_archived"):
            return False
        tip = self._table(scope).get(compression_tip(self._backend, scope, str(row["session_id"]))) or row
        return (tip["end_reason"] or "") not in RECOVERABLE_END_REASONS

    def _retire_canonical_bot_chat(self, scope: ProfileScope, session_id: str) -> list[str]:
        """The deliberate archive's retirement of the canonical Bot Chat: every lineage row titled it loses title and source; when any does, the lineage is un-hidden and its recoverable end reasons cleared (``ended_at`` kept). Returns the ids whose title was released."""
        table = self._table(scope)
        ids = self._lineage_ids(table, session_id) if session_id in table else []
        held = [sid for sid in ids if table[sid]["title"] == CANONICAL_BOT_CHAT_TITLE]
        if not held:
            return []
        for sid in ids:
            row = table[sid]
            row["hidden"] = False
            if row["end_reason"] in RECOVERABLE_END_REASONS:
                row["end_reason"] = None
        for sid in held:
            table[sid].update(title=None, title_source=None)
        return held

    def _resolve_title_conflict(self, scope: ProfileScope, table: dict[str, dict[str, Any]], session_id: str, title: str) -> Optional[str]:
        """Free ``title`` for ``session_id`` or raise :class:`TitleConflict`: a compression ancestor hands it to its continuation, a retired canonical Bot Chat and an ended empty visible ghost yield it. Returns the id whose title was cleared."""
        holder = next((row for row in table.values() if row["title"] == title and row["session_id"] != session_id), None)
        if holder is None:
            return None
        holder_id = str(holder["session_id"])
        if self._is_compression_ancestor(table, holder_id, session_id):
            holder["title"] = None
            return holder_id
        retired_bot_chat = title == CANONICAL_BOT_CHAT_TITLE and self._bot_chat_title_released(scope, holder)
        if retired_bot_chat or (self._is_empty_ended(scope, holder) and not holder["hidden"]):
            holder["title"] = None
            holder["title_source"] = None
            return holder_id
        raise TitleConflict(title, SessionId(holder_id))

    def _write_title(self, scope: ProfileScope, session_id: SessionId, title: Optional[str], source: TitleSource) -> bool:
        """Write a title under provenance precedence: a user write always lands, an automatic one only over an untitled row or strictly lower authority; nothing moves the hidden canonical Bot Chat off its title."""
        clean = sanitize_title(title)
        table = self._table(scope)
        row = table.get(session_id)
        if row is None:
            return False
        is_user = source is TitleSource.USER
        if row["title"] == CANONICAL_BOT_CHAT_TITLE and row["hidden"] and clean != CANONICAL_BOT_CHAT_TITLE:
            if is_user:
                raise CanonicalTitleLocked(session_id)
            return False
        if not is_user and row["title"] is not None and _TITLE_RANK[row["title_source"]] >= _TITLE_RANK[source]:
            return False
        if clean:
            self._resolve_title_conflict(scope, table, str(session_id), clean)
        row["title"] = clean
        row["title_source"] = source if clean else None
        return True

    def set_title(self, session_id: SessionId, title: Optional[str]) -> bool:
        scope = self._backend.scope
        return self._write_title(scope, session_id, title, TitleSource.USER)

    def set_auto_title(self, session_id: SessionId, title: str, *, source: TitleSource) -> bool:
        scope = self._backend.scope
        if source is TitleSource.USER:
            raise ValueError(f"invalid automatic title source: {source.value!r}")
        return self._write_title(scope, session_id, title, source)

    def set_title_source(self, session_id: SessionId, source: TitleSource) -> bool:
        scope = self._backend.scope
        row = self._table(scope).get(session_id)
        if row is None or row["title"] is None:
            return False
        row["title_source"] = source
        return True

    def retitle_canonical_bot_chat(self, session_id: SessionId, title: str) -> bool:
        scope = self._backend.scope
        row = self._table(scope).get(session_id)
        if row is None or row["title"] != CANONICAL_BOT_CHAT_TITLE:
            return False
        row.update(title=title, title_source=TitleSource.USER, hidden=False)
        return True

    def get_title(self, session_id: SessionId) -> Optional[str]:
        scope = self._backend.scope
        row = self._table(scope).get(session_id)
        return None if row is None else row["title"]

    def get_title_source(self, session_id: SessionId) -> Optional[TitleSource]:
        scope = self._backend.scope
        row = self._table(scope).get(session_id)
        return None if row is None or row["title"] is None else row["title_source"]

    def _titled(self, scope: ProfileScope, title: str) -> Optional[dict[str, Any]]:
        """The row holding ``title``; a retired canonical Bot Chat reads as absent."""
        row = next((row for row in self._table(scope).values() if row["title"] == title), None)
        if row is not None and title == CANONICAL_BOT_CHAT_TITLE and self._bot_chat_title_released(scope, row):
            return None
        return row

    def get_by_title(self, title: str) -> Optional[SessionRecord]:
        scope = self._backend.scope
        row = self._titled(scope, title)
        # The title lookup joins only the prompt store, so the pin stays its unresolved hash.
        return None if row is None else self._record(scope, row, resolve_tool_pin=False)

    def resolve_by_title(self, title: str) -> Optional[SessionId]:
        scope = self._backend.scope
        exact = self._titled(scope, title)
        if title == CANONICAL_BOT_CHAT_TITLE:
            return None if exact is None else SessionId(exact["session_id"])
        prefix = ascii_lower(title + " #")
        numbered = [row for row in self._table(scope).values() if row["title"] is not None and ascii_lower(row["title"]).startswith(prefix)]
        if numbered:
            return SessionId(max(numbered, key=lambda row: (row["started_at"], row["session_id"]))["session_id"])
        return None if exact is None else SessionId(exact["session_id"])

    def next_title_in_lineage(self, base_title: str) -> str:
        scope = self._backend.scope
        match = _NUMBERED_TITLE_RE.match(base_title)
        base = match.group(1) if match else base_title
        prefix = ascii_lower(base + " #")
        titles = [
            row["title"]
            for row in self._table(scope).values()
            if row["title"] is not None and (row["title"] == base or ascii_lower(row["title"]).startswith(prefix))
        ]
        if not titles:
            return base
        # The unnumbered original counts as #1.
        numbers = [int(found.group(2)) for found in map(_NUMBERED_TITLE_RE.match, titles) if found]
        return f"{base} #{max([1, *numbers]) + 1}"

    def title_holder(self, title: str, *, excluding: SessionId) -> Optional[SessionId]:
        scope = self._backend.scope
        holder = next((sid for sid, row in self._table(scope).items() if row["title"] == title and sid != excluding), None)
        return None if holder is None else SessionId(holder)

    # ── model config ──────────────────────────────────────────────────────

    @staticmethod
    def _set_config(row: dict[str, Any], config: dict[str, Any]) -> None:
        row["model_config"] = copy.deepcopy(config)
        row["model_config_stored"] = True
        sync_lineage_markers(row)

    def replace_session_config(self, session_id: SessionId, command: ReplaceSessionConfig) -> None:
        scope = self._backend.scope
        row = self._table(scope).get(session_id)
        if row is None:
            return
        self._set_config(row, dict(command.model_config))
        if command.model is not None:
            row["model"] = command.model

    @staticmethod
    def _merged(config: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
        """``patch`` merged into a copy of ``config``; ``None`` removes a key."""
        merged = dict(config)
        for key, value in patch.items():
            if value is None:
                merged.pop(key, None)
            else:
                merged[key] = value
        return merged

    def patch_model_config(self, session_id: SessionId, patch: Mapping[str, Optional[JsonValue]]) -> None:
        scope = self._backend.scope
        if not session_id or not patch:
            return
        row = self._table(scope).get(session_id)
        if row is not None:
            self._set_config(row, self._merged(row["model_config"], patch))

    def advance_onboarding_run(self, session_id: SessionId, event: Any, *, now: float) -> OnboardingAdvance:
        scope = self._backend.scope
        from session_rules.onboarding_flow import Facts, apply, is_active

        from .lineage import compression_tip

        with self._backend._lock:
            tip = compression_tip(self._backend, scope, str(session_id))
            row = self._table(scope).get(tip)
            if row is None:
                return OnboardingAdvance(session_id=None, was_active=False, step=None)
            current = row["model_config"].get(ONBOARDING_RUN_KEY)
            if not isinstance(current, dict):
                return OnboardingAdvance(session_id=SessionId(tip), was_active=False, step=None)
            step = apply(current, event, Facts.from_stored(current["facts"]), now=now)
            # Stored as a JSON copy, so the step handed back never aliases the row.
            self._set_config(row, self._merged(row["model_config"], {ONBOARDING_RUN_KEY: json.loads(json.dumps(step.record))}))
            return OnboardingAdvance(session_id=SessionId(tip), was_active=is_active(current), step=step)

    def get_model_config_value(self, session_id: SessionId, key: str, default: JsonValue = None) -> JsonValue:
        scope = self._backend.scope
        row = self._table(scope).get(session_id)
        if row is None:
            return default
        value: JsonValue = copy.deepcopy(row["model_config"].get(key, default))
        return value

    def session_ids_with_model_config_key(self, key: str) -> tuple[SessionId, ...]:
        scope = self._backend.scope
        rows = [row for row in self._table(scope).values() if row["model_config"].get(key) is not None]
        rows.sort(key=lambda row: (row["started_at"], row["session_id"]), reverse=True)
        return tuple(SessionId(row["session_id"]) for row in rows)

    def update_model(
        self,
        session_id: SessionId,
        model: str,
        provider: Optional[str] = None,
        *,
        base_url: Optional[str] = None,
        api_mode: Optional[str] = None,
    ) -> None:
        scope = self._backend.scope
        row = self._table(scope).get(session_id)
        if row is None:
            return
        patch: dict[str, Any] = {RUNTIME_LOCK_KEY: None}
        if model:
            patch["model"] = model
        if provider:
            route = {"provider": provider, "base_url": base_url or None, "api_mode": api_mode or None}
            patch.update(route, gateway_runtime=route)
        row["model"] = model
        self._set_config(row, self._merged(row["model_config"], patch))

    def set_runtime_lock(self, session_id: SessionId, lock: RuntimeLock) -> None:
        scope = self._backend.scope
        row = self._table(scope).get(session_id)
        if row is None:
            return
        payload: dict[str, JsonValue] = {**lock.as_mapping(), "updated_at": time.time()}
        self._set_config(row, {**row["model_config"], RUNTIME_LOCK_KEY: payload})
        if lock.model is not None:
            row["model"] = lock.model

    def set_yolo(self, session_id: SessionId, enabled: bool) -> None:
        scope = self._backend.scope
        row = self._table(scope).get(session_id) if session_id else None
        if row is not None:
            self._set_config(row, {**row["model_config"], YOLO_KEY: bool(enabled)})

    def update_system_prompt(self, session_id: SessionId, system_prompt: Optional[str]) -> None:
        scope = self._backend.scope
        row = self._table(scope).get(session_id)
        if row is not None:
            row["system_prompt"] = system_prompt

    # ── prompt refresh epochs ─────────────────────────────────────────────
    # ``None`` keys the profile-wide epoch, a session id that session's own.

    def request_prompt_refresh(self, session_id: Optional[SessionId] = None) -> int:
        epochs = self._backend.scope_store(self._backend.scope).setdefault("prompt_refresh_epochs", {})
        with self._flags_lock:
            epoch = max(epochs.values(), default=0) + 1
            epochs[None if session_id is None else str(session_id)] = epoch
        return epoch

    def prompt_refresh_epoch(self, session_id: SessionId) -> int:
        epochs = self._backend.scope_store(self._backend.scope).get("prompt_refresh_epochs", {})
        return max(epochs.get(None, 0), epochs.get(str(session_id), 0))

    def count_with_stored_prompt(self) -> int:
        return sum(1 for row in self._table(self._backend.scope).values() if row["system_prompt"] is not None)

    def update_tool_pin(self, session_id: SessionId, tool_pin: Optional[JsonValue]) -> None:
        scope = self._backend.scope
        row = self._table(scope).get(session_id)
        if row is not None:
            row["tool_pin"] = tool_pin_text(tool_pin)

    # ── working directory and git metadata ────────────────────────────────

    def update_cwd(
        self,
        session_id: SessionId,
        cwd: str,
        git_branch: Optional[str] = None,
        git_repo_root: Optional[str] = None,
        *,
        replace_git_meta: bool = False,
    ) -> Optional[int]:
        scope = self._backend.scope
        if not session_id or not cwd:
            return None
        row = self._table(scope).get(session_id)
        if row is None:
            return None
        captured = {"git_branch": (git_branch or "").strip(), "git_repo_root": (git_repo_root or "").strip()}
        if row["cwd"] != cwd or replace_git_meta:
            row.update({field: value or None for field, value in captured.items()})
        else:
            # Same cwd: only a captured value overwrites, so a failed probe never clobbers one.
            row.update({field: value for field, value in captured.items() if value})
        row["cwd"] = cwd
        row["cwd_generation"] = int(row.get("cwd_generation", 0)) + 1
        generation: int = row["cwd_generation"]
        return generation

    def publish_git_metadata(
        self,
        session_id: SessionId,
        cwd: str,
        generation: int,
        git_branch: Optional[str] = None,
        git_repo_root: Optional[str] = None,
    ) -> bool:
        scope = self._backend.scope
        valid_generation = isinstance(generation, int) and not isinstance(generation, bool) and generation >= 1
        if not session_id or not cwd or not valid_generation:
            return False
        fields = {field: value for field, value in (("git_branch", (git_branch or "").strip()), ("git_repo_root", (git_repo_root or "").strip())) if value}
        row = self._table(scope).get(session_id)
        if not fields or row is None or row["cwd"] != cwd or row.get("cwd_generation") != generation:
            return False
        row.update(fields)
        return True

    # ── lineage flags ─────────────────────────────────────────────────────

    def _set_lineage_flag(self, scope: ProfileScope, session_id: SessionId, values: Mapping[str, Any]) -> list[str]:
        """Set ``values`` on every row of the session's compression lineage; returns the ids written (empty without the row)."""
        table = self._table(scope)
        if session_id not in table:
            return []
        ids = self._lineage_ids(table, str(session_id))
        for sid in ids:
            table[sid].update(values)
        return ids

    def set_archived(self, session_id: SessionId, archived: bool, *, notify: bool = True) -> ArchiveChange:
        scope = self._backend.scope
        row = self._table(scope).get(session_id)
        was_archived = row is not None and row["archived"]
        # The deliberate archive clears the idle sweep's provenance either way.
        ids = self._set_lineage_flag(scope, session_id, {"archived": bool(archived), "auto_archived": False})
        if archived and ids:
            self._retire_canonical_bot_chat(scope, str(session_id))
        newly_archived = bool(archived and ids and not was_archived)
        return ArchiveChange(changed=bool(ids), archived_ids=tuple(SessionId(sid) for sid in ids) if newly_archived and notify else ())

    def set_pinned(self, session_id: SessionId, pinned: bool) -> bool:
        scope = self._backend.scope
        ids = self._set_lineage_flag(scope, session_id, {"pinned": bool(pinned)})
        row = self._table(scope).get(session_id)
        canonical_bot_chat = row is not None and row["hidden"] and row["title"] == CANONICAL_BOT_CHAT_TITLE
        if pinned and not canonical_bot_chat:
            self._set_lineage_flag(scope, session_id, {"hidden": False})
        return bool(ids)

    def set_hidden(self, session_id: SessionId, hidden: bool) -> bool:
        scope = self._backend.scope
        return bool(self._set_lineage_flag(scope, session_id, {"hidden": bool(hidden)}))

    def set_read(self, session_id: SessionId, read: bool = True) -> bool:
        scope = self._backend.scope
        return bool(self._set_lineage_flag(scope, session_id, {"last_read_at": time.time() if read else 0.0}))

    def _unarchive_auto_archived_lineage(self, table: dict[str, dict[str, Any]], session_id: str) -> bool:
        """Un-hide a lineage only the idle sweep archived; any deliberately archived row keeps the whole lineage archived."""
        rows = [table[sid] for sid in self._lineage_ids(table, session_id)]
        if any(row["archived"] and not row.get("auto_archived") for row in rows):
            return False
        archived = [row for row in rows if row["archived"]]
        for row in archived:
            row.update(archived=False, auto_archived=False)
        return bool(archived)

    def unarchive_recoverable(self, session_id: SessionId) -> bool:
        scope = self._backend.scope
        from .lineage import compression_tip

        if not session_id:
            return False
        try:
            table = self._table(scope)
            row = table.get(session_id)
            if row is None or not row["archived"]:
                return False
            # The accidental stamp lives on the live tip.
            tip = table.get(compression_tip(self._backend, scope, str(session_id))) or row
            if (tip["end_reason"] or "") not in RECOVERABLE_END_REASONS:
                return False
            if not self.set_archived(session_id, False).changed:
                return False
            tip["ended_at"] = None
            tip["end_reason"] = None
        except Exception:
            return False
        return True
