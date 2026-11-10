"""``admit`` and ``retired_replay`` of the admission reference, with the typed extras (``gate``, ``replaces``, ``supersedes``, ``api_target``, ``rewrite``, ``hosted``) checked inside the admission write.

The admission write runs under ``AuthorityState.guarded``: the ``/retry`` rewrite composes the message reference's own rewind and replace steps, and a refusal anywhere restores every table.
"""

from __future__ import annotations

import copy
import json
import uuid
from dataclasses import fields
from typing import Any, Mapping

from persistence.authority.admissions import (
    ActiveReplace,
    AdmissionRecord,
    ApiTargetClaim,
    CarrierRewind,
    GateRefusal,
    QueuedFold,
    QueuedReplacement,
    RetryRewriteRefused,
    TranscriptRewrite,
)
from persistence.authority.vocabulary import ADMISSION_QUEUE_MAX, ADMITTED_REQUEST_CONFLICT, admission_fingerprint
from persistence.conversation.messages import NewMessage, ReplaceMode, RewindPlan
from persistence.core.canonical import canonical_json
from persistence.core.errors import STORE_FAILURES, AuthorityRefusal, CapabilityUnsupported
from persistence.core.identity import MessageId, SessionId
from session_rules.automation import folded_payload

from . import admissions_rows as rows
from .admissions_rows import _identity_key, _terminal_key


def _replacement_binding(principal_id: str, session_id: str, request_id: str) -> str:
    """Meta key binding a queued replacement to its principal, session and request."""
    return rows.REPLACEMENT_PREFIX + admission_fingerprint(canonical_target=session_id, payload={"principal": principal_id, "request": request_id})


def _new_message(message: Mapping[str, Any]) -> NewMessage:
    known = {f.name for f in fields(NewMessage)}
    values = {key: value for key, value in message.items() if key in known}
    if isinstance(values.get("tool_calls"), list):
        values["tool_calls"] = tuple(values["tool_calls"])
    elif values.get("tool_calls") is None:
        values.pop("tool_calls", None)
    return NewMessage(**values)


class AdmitMixin:
    """``admit`` and ``retired_replay``; mixed into ``InMemoryAdmissionRepository``."""

    def _tombstone_replay(self, committed: Mapping[str, Any], epoch: Any, principal_id: str, session_id: str, request_id: str, payload: Any, intent: str) -> AdmissionRecord | None:
        """The stored tombstone replay of a retired target past the input checks: ``None`` for a live target."""
        self._state().require_epoch(committed, epoch)
        if session_id not in committed["retired"]:
            return None
        saved = committed["meta"].get(_identity_key(principal_id, session_id, request_id))
        if saved is None:
            raise AuthorityRefusal("not_found")
        row = committed["meta"][_terminal_key(saved)]
        if row["digest"] != rows.digest(session_id, payload, intent):
            raise AuthorityRefusal("admission_conflict", **ADMITTED_REQUEST_CONFLICT)
        return rows.record(row)

    def retired_replay(self, *, epoch: int, principal_id: str, session_id: str, request_id: str, payload: Mapping[str, Any], intent: str = "queue") -> AdmissionRecord | None:
        for value in (principal_id, session_id, request_id):
            rows.text(value)
        canonical = rows.encode(payload)
        return self._tombstone_replay(self._state().snapshot, epoch, principal_id, session_id, request_id, canonical, intent)

    def admit(self, *, epoch: int, principal_id: str, session_id: str, request_id: str, payload: Mapping[str, Any], intent: str = "queue", gate: GateRefusal | None = None,
              replaces: QueuedReplacement | None = None, supersedes: str | None = None, api_target: ApiTargetClaim | None = None,
              rewrite: TranscriptRewrite | None = None, hosted: Any = None, fold: QueuedFold | None = None) -> tuple[AdmissionRecord, bool]:
        for value in (principal_id, session_id, request_id):
            rows.text(value)
        if intent not in ("queue", "steer", "redirect"):
            raise AuthorityRefusal("invalid_params")
        canonical = rows.encode(payload)
        state = self._state()
        retired = self._tombstone_replay(state.snapshot, epoch, principal_id, session_id, request_id, canonical, intent)
        if retired is not None:
            return retired, False
        digest = rows.digest(session_id, canonical, intent)
        with state.guarded():
            authority = state.snapshot
            state.require_epoch(authority, epoch)
            rows.require_session(self._tables(), session_id)
            old = next((row for row in rows.rows_by_seq(authority) if (row["principal_id"], row["target_session_id"], row["request_id"]) == (principal_id, session_id, request_id)), None)
            if old is not None:
                if old["digest"] != digest:
                    raise AuthorityRefusal("admission_conflict", **ADMITTED_REQUEST_CONFLICT)
                if replaces is not None:
                    saved = authority["meta"].get(_replacement_binding(principal_id, session_id, request_id))
                    if saved is None or json.loads(saved) != replaces.admission_id:
                        raise AuthorityRefusal("admission_conflict")
                return rows.record(old), False
            if sum(1 for row in rows.session_rows(authority, session_id) if row["status"] == "queued") >= ADMISSION_QUEUE_MAX:
                raise AuthorityRefusal("queue_full")
            if gate is not None:
                raise gate.refusal()
            self._run_extras(authority, principal_id, session_id, request_id, replaces, supersedes, api_target, rewrite, hosted, fold)
            authority = state.snapshot
            meta = authority["meta"]
            meta["admissions_seq"] = meta.get("admissions_seq", 0) + 1
            admission_id = uuid.uuid4().hex
            row = authority["admissions"][admission_id] = {
                "admission_id": admission_id, "request_id": request_id, "principal_id": principal_id, "target_session_id": session_id,
                "lineage": [session_id], "payload": canonical, "digest": digest, "intent": intent, "status": "queued", "outcome": None,
                "owner_epoch": epoch, "generation": None, "seq": meta["admissions_seq"],
            }
            return rows.record(row), True

    def _run_extras(self, authority: dict[str, Any], principal_id: str, session_id: str, request_id: str, replaces: QueuedReplacement | None, supersedes: str | None,
                    api_target: ApiTargetClaim | None, rewrite: TranscriptRewrite | None, hosted: Any, fold: QueuedFold | None = None) -> None:
        if replaces is not None:
            replaced = rows.admission(authority, replaces.admission_id)
            if replaced["status"] != "queued":
                raise AuthorityRefusal("admission_started")
            replaced.update(status="terminal", outcome="cancelled")
            authority["meta"][_replacement_binding(principal_id, session_id, request_id)] = json.dumps(replaces.admission_id)
        if supersedes is not None:
            # The replaced tick must still be queued on this target.
            row = authority["admissions"].get(supersedes)
            if row is None or row["target_session_id"] != session_id or row["status"] != "queued":
                raise rows.stale("admission", supersedes)
            row.update(status="terminal", outcome="cancelled")
        if api_target is not None:
            self._claim_api_target(authority, session_id, request_id, api_target)
        if rewrite is not None:
            self._rewrite(rewrite)
        if hosted is not None:
            self._hosted_check(hosted)
        if fold is not None:
            # The folded completion must still be the queued tail on this target, but for the skipped ticks.
            row = authority["admissions"].get(fold.admission_id)
            if (row is None or row["target_session_id"] != session_id or row["status"] != "queued"
                    or any(other["admission_id"] not in fold.skip_ids for other in rows.session_rows(authority, session_id)
                           if other["status"] == "queued" and other["seq"] > row["seq"])):
                raise rows.stale("admission", fold.admission_id)
            row.update(status="terminal", outcome="cancelled", payload=folded_payload(copy.deepcopy(row["payload"])))

    def _claim_api_target(self, authority: dict[str, Any], session_id: str, request_id: str, claim: ApiTargetClaim) -> None:
        """Claim an API request for one target: an API admission of the request on another target, or with a client key its tombstone there, refuses ``admission_conflict``; the settings are stored."""
        if any(row["principal_id"] == "api" and row["request_id"] == request_id and row["target_session_id"] != session_id for row in authority["admissions"].values()):
            raise AuthorityRefusal("admission_conflict")
        if claim.client_key and any(row["target_session_id"] != session_id for row in self._api_tombstones(authority, request_id)):
            raise AuthorityRefusal("admission_conflict")
        authority["meta"][rows.API_SETTINGS_PREFIX + session_id] = canonical_json(dict(claim.settings))

    @staticmethod
    def _api_tombstones(authority: Mapping[str, Any], request_id: Any) -> list[dict[str, Any]]:
        """The terminal tombstones of API admissions with ``request_id``."""
        return [value for key, value in authority["meta"].items() if isinstance(key, tuple) and key[0] == "terminal_admission"
                and value.get("principal_id") == "api" and value.get("request_id") == request_id]

    def _rewrite(self, rewrite: TranscriptRewrite) -> None:
        """Run the ``/retry`` rewrite: refusals and store failures keep their class; any other failure is ``RetryRewriteRefused``."""
        try:
            if isinstance(rewrite, CarrierRewind):
                plan = RewindPlan(target_row_id=MessageId(rewrite.target_row_id), expected_active_ids=tuple(MessageId(i) for i in rewrite.expected_active_ids),
                                  expected_target_content=rewrite.expected_target_content, scaffold={}, prefix=(), durable_prefix=(), live_view={},
                                  live_text=None, durable_user=(), user_ordinal=-1)
                self._backend.messages._rewind_to(self._backend.scope, SessionId(rewrite.session_id), plan)
            elif isinstance(rewrite, ActiveReplace):
                messages = self._backend.messages
                transcript = messages._transcript(self._backend.scope, SessionId(rewrite.session_id))
                active = tuple(row.id for row in transcript.active()) if transcript is not None else ()
                if active != tuple(rewrite.expected_active_ids):
                    raise RuntimeError("active transcript changed before the replacement could be persisted")
                messages.replace_transcript(SessionId(rewrite.session_id), [_new_message(m) for m in rewrite.messages], ReplaceMode.PRESERVE_EXISTING_ARCHIVE, reject_active_turn_lease=True)
        except (AuthorityRefusal, *STORE_FAILURES):
            raise
        except Exception as exc:
            raise RetryRewriteRefused(rewrite.session_id, str(exc)) from exc

    def _hosted_check(self, guard: Any) -> None:
        """The hosted member check, refusing ``permission_denied``."""
        try:
            from ..gateway.hosted_rooms import hosted_member_check
        except ImportError as exc:
            raise CapabilityUnsupported("hosted_member_check") from exc
        if not hosted_member_check(self._backend, guard, operation="submit"):
            raise AuthorityRefusal("permission_denied")
