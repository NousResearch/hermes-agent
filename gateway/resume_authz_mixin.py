"""Session-resume authorization for the gateway slash-command surface.

Extracted from ``gateway/slash_commands.py`` as part of the god-file
decomposition campaign, alongside ``gateway/authz_mixin.py``.

``GatewaySlashCommandsMixin`` was itself lifted out of ``gateway/run.py`` by
that campaign, and has since grown to 5,383 lines and 65 methods, so it is now
a decomposition target in its own right. This takes the cluster with the
clearest boundary in it: deciding whether a caller may resume, or even see,
somebody else's session.

That is a trust boundary rather than command plumbing. ``_resume_target_allowed``
is what stops a user in one chat from resuming a session belonging to a user in
another, and ``tests/gateway/test_resume_command.py`` exercises it with explicit
cross-user and IDOR cases. Reading it next to sixty-odd ``_handle_*_command``
methods makes that harder than it needs to be.

Mixin contract: a plain mixin, mixed into ``GatewaySlashCommandsMixin`` and so
reaching ``GatewayRunner`` through the existing MRO. It defines no ``__init__``
and no state of its own; the host's attributes (``self.session_store``,
``self.config``) resolve through the MRO. It calls no method that stays
behind, and it never imports ``gateway.slash_commands`` or ``gateway.run``, so
there is no cycle.

Behavior-neutral: every method is lifted verbatim.
"""

from __future__ import annotations

from typing import Optional

from gateway.config import Platform
from gateway.session import SessionSource, is_shared_multi_user_session


class GatewayResumeAuthorizationMixin:
    """See module docstring - resume-authorization cluster lifted verbatim."""

    def _gateway_session_origin_for_id(self, session_id: str) -> Optional[SessionSource]:
        """Best-effort origin lookup for gateway session IDs."""
        lookup = getattr(type(self.session_store), "lookup_by_session_id", None)
        if callable(lookup):
            entry = lookup(self.session_store, session_id)
            return getattr(entry, "origin", None) if entry is not None else None

        # Test doubles and older stores may not expose the public lookup helper.
        # Keep the Matrix resume guard fail-closed if no origin can be resolved.
        entries = getattr(self.session_store, "_entries", {}) or {}
        for entry in entries.values():
            if getattr(entry, "session_id", None) == session_id:
                return getattr(entry, "origin", None)
        return None

    @staticmethod
    def _same_matrix_room(current: SessionSource, origin: Optional[SessionSource]) -> bool:
        return (
            origin is not None
            and origin.platform == Platform.MATRIX
            and current.platform == Platform.MATRIX
            and origin.chat_id == current.chat_id
            # thread_id is part of the session key (build_session_key appends it
            # for every chat type when present), and Matrix scopes the model's
            # turn to the current room/thread. A live session in another thread
            # of the SAME room is a DIFFERENT session, so a caller in thread A
            # must not resume/enumerate a target whose origin is in thread B.
            # Non-threaded rooms have empty thread_id on both sides ("" == ""),
            # so room-level sharing is preserved unchanged.
            and str(getattr(current, "thread_id", "") or "")
            == str(getattr(origin, "thread_id", "") or "")
        )

    def _same_origin_chat(self, current: SessionSource, origin: Optional[SessionSource]) -> bool:
        """Platform-agnostic counterpart to ``_same_matrix_room``.

        True when *origin* shares *current*'s platform and chat, and the same
        participant whenever the session key for this source is per-user. Group
        and thread sessions that ``build_session_key`` isolates per participant
        (the default ``group_sessions_per_user=True``) must also be scoped by
        participant here — otherwise a co-member could resume another member's
        live per-user group session (IDOR). Only an explicitly shared
        group/thread (``group_sessions_per_user=False`` /
        ``thread_sessions_per_user``) lets co-members share, mirroring the key
        contract via ``is_shared_multi_user_session``.
        """
        if origin is None or current is None:
            return False
        if origin.platform != current.platform:
            return False
        if origin.chat_id != current.chat_id:
            return False
        # thread_id is part of the session key for every chat type when present
        # (build_session_key appends it unconditionally), so a session in one
        # thread is a DIFFERENT session from another thread of the same parent
        # chat. is_shared_multi_user_session only decides participant sharing
        # WITHIN a thread, never across threads — require thread equality before
        # any sharing logic so a live origin in thread A cannot match a caller in
        # thread B of the same parent chat.
        if str(getattr(current, "thread_id", "") or "") != str(
            getattr(origin, "thread_id", "") or ""
        ):
            return False
        chat_type = (getattr(current, "chat_type", "") or "").lower()
        # DM-like chats are always per-user.
        if chat_type in {"dm", "direct", "private", ""}:
            # chat_id was already required equal above and, when present, IS the
            # DM session key — so an equal non-empty chat_id is sufficient.
            # build_session_key only falls back to the participant id
            # (``user_id_alt or user_id`` — Signal/Feishu key on user_id_alt)
            # when there is NO chat_id; mirror that and fail closed on a
            # missing/different participant so two no-chat_id DM origins are
            # never conflated (was: compared user_id only and allowed when
            # either side was missing).
            if str(getattr(current, "chat_id", "") or ""):
                return True
            cur_pid = str(current.user_id_alt or current.user_id or "")
            org_pid = str(origin.user_id_alt or origin.user_id or "")
            return bool(cur_pid) and cur_pid == org_pid
        # Non-DM: scope by participant whenever the session key for this source
        # is per-user. is_shared_multi_user_session mirrors build_session_key's
        # isolation rules exactly, so the guard stays in lock-step with the key.
        shared = is_shared_multi_user_session(
            current,
            group_sessions_per_user=getattr(self.config, "group_sessions_per_user", True),
            thread_sessions_per_user=getattr(self.config, "thread_sessions_per_user", False),
        )
        if shared:
            return True
        # Per-user key: compare the participant id the key is actually built
        # from (user_id_alt or user_id — Signal/Feishu key on user_id_alt).
        cur_pid = current.user_id_alt or current.user_id
        org_pid = origin.user_id_alt or origin.user_id
        if cur_pid and org_pid:
            return cur_pid == org_pid
        # Per-user key but a participant id is missing on one side: cannot prove
        # the same owner — fail closed.
        return False

    def _resume_caller_is_admin(self, source: SessionSource) -> bool:
        """Whether *source* is an EXPLICITLY-configured admin allowed to make a
        cross-origin /resume or /sessions listing.

        Deliberately stricter than ``SlashAccessPolicy.is_admin()``: that returns
        True for every allowed caller when slash gating is DISABLED (so commands
        stay runnable by default), but cross-ORIGIN DATA ACCESS must require a
        real, configured admin. Otherwise the default (no admin list) config
        would treat every gateway caller as cross-origin-capable and re-open the
        enumeration IDOR.
        """
        try:
            from gateway.slash_access import policy_for_source
            policy = policy_for_source(self.config, source)
            uid = getattr(source, "user_id", None)
            return bool(policy.enabled and uid and policy.is_admin(uid))
        except Exception:
            return False

    async def _resume_target_allowed(
        self, source: SessionSource, target_id: str, allow_override: bool = False
    ) -> bool:
        """Whether *source* may resume the persisted session *target_id*.

        Generalizes the Matrix-only room guard to every adapter so a caller
        cannot bind their gateway session to another user's/room's persisted
        session id (IDOR). Uses the live origin when the target is active;
        otherwise falls back to the DB row's source + user_id (the sessions
        table has no chat_id). An identity-bearing caller is allowed only when
        the row PROVES the same owner; a row that lacks enough ownership data
        fails closed. An explicit admin ``--all`` override bypasses scoping.
        """
        if allow_override and self._resume_caller_is_admin(source):
            return True
        # Use the live origin only when it resolves to a real SessionSource; a
        # store that can't resolve it (or an unexpected lookup error) must not
        # silently allow/deny — fall through to the deterministic DB scoping.
        try:
            origin = self._gateway_session_origin_for_id(target_id)
        except Exception:
            origin = None
        if isinstance(origin, SessionSource):
            return self._same_origin_chat(source, origin)
        # Inactive/persisted-only: best-effort scope by DB row source + user.
        try:
            row = await self._session_db.get_session(target_id) or {}
        except Exception:
            return False
        caller_src = source.platform.value if source.platform else None
        row_src = row.get("source")
        if row_src and caller_src and str(row_src) != str(caller_src):
            return False  # different platform / source
        caller_uid = str(getattr(source, "user_id", "") or "")
        row_uid = str(row.get("user_id") or "")
        # Chat/thread origin recorded at session creation (see
        # SessionDB._insert_session_row). The sessions table historically stored
        # only source + user_id, so a same-user row could belong to a DIFFERENT
        # chat; comparing the persisted origin closes that gap. Legacy rows
        # created before origin capture have NULL here and therefore fail closed
        # (they cannot prove the caller's chat) — resume them via a live session
        # or an admin override.
        caller_chat = str(getattr(source, "chat_id", "") or "")
        row_chat = str(row.get("chat_id") or "")
        caller_thread = str(getattr(source, "thread_id", "") or "")
        row_thread = str(row.get("thread_id") or "")
        chat_type = (getattr(source, "chat_type", "") or "").lower()
        caller_is_dm = chat_type in {"dm", "direct", "private", ""}
        # build_session_key keys the participant on ``user_id_alt or user_id``
        # (Signal/Feishu carry the canonical participant in user_id_alt), but the
        # sessions table only ever stored user_id — it has no user_id_alt column.
        # So when the caller carries a user_id_alt, the row CANNOT prove the
        # canonical participant that the live session key is built from: two
        # members sharing one user_id but different user_id_alt map to DIFFERENT
        # session keys, yet the persisted row's user_id would match both. The
        # live-origin guard (_same_origin_chat) compares user_id_alt correctly;
        # the persisted fallback cannot, so any per-user comparison that would
        # otherwise rely on row_uid == caller_uid must fail closed here to stay
        # in lock-step with the key boundary (CWE-639). Shared group/thread
        # sessions are unaffected (they don't scope by participant at all), and
        # an admin --all override still bypasses this above.
        caller_keys_on_alt = bool(str(getattr(source, "user_id_alt", "") or ""))
        if caller_uid:
            # Identity-bearing caller: allow only when the row PROVES the same
            # owner AND the same platform/origin AND the same chat/thread. A row
            # with no/blank user_id cannot be proven to belong to this caller; a
            # row with no/blank source cannot be proven to share the caller's
            # platform (the row_src check above only rejects a *mismatching*
            # non-blank source, so a blank/legacy source would otherwise slip
            # through on user_id equality alone); and a row whose origin chat
            # (or thread) differs from the caller's belongs to a different
            # conversation. Any gap fails closed — an identified user must not
            # bind to an unowned, other-owned, other-chat, or unproven-origin
            # persisted session by id/title. (Legacy NULL-owner/blank-source/
            # NULL-chat rows are intentionally not resumable this way; use a
            # live session or an explicit admin override.)
            # Common origin proof for any identity-bearing caller: a non-blank
            # source that matches the caller's platform, and the same thread. A
            # blank/legacy source can't prove the platform; a different thread is
            # a different session (build_session_key appends thread_id).
            origin_ok = (
                bool(row_src) and bool(caller_src)
                and str(row_src) == str(caller_src)
                and row_thread == caller_thread
            )
            if not origin_ok:
                return False
            if caller_is_dm:
                # DMs are keyed on user_id; require the same owner. chat_id is
                # legitimately absent on both sides for a no-chat_id DM (scoped
                # by user_id), but a mismatching chat_id (when present) is still
                # rejected.
                #
                # A no-chat_id DM is keyed PURELY on the participant
                # (``user_id_alt or user_id``). If the caller keys on user_id_alt
                # the persisted row (user_id only) cannot prove that participant,
                # so fail closed. When chat_id is present on both sides it is the
                # DM key and equal chat_id is sufficient, so the alt gap doesn't
                # apply there.
                if caller_keys_on_alt and not (bool(row_chat) and bool(caller_chat)):
                    return False
                return (
                    bool(row_uid) and row_uid == caller_uid
                    and row_chat == caller_chat
                )
            # Non-DM (group/channel/forum/thread): build_session_key includes
            # chat_id, so a row (or caller) with NO chat provenance cannot prove
            # same-chat. Require both sides non-blank and equal — a legacy
            # NULL-chat row (or a caller missing its chat_id) fails closed even
            # when both normalize to "". (CWE-639)
            if not (bool(row_chat) and bool(caller_chat) and row_chat == caller_chat):
                return False
            # Within the same non-DM chat/thread, mirror build_session_key's
            # participant scoping: a SHARED group/thread session
            # (group_sessions_per_user=False, or a shared thread) is one session
            # for every participant, so the same-chat proof above is sufficient —
            # do NOT also require user-id equality (otherwise a co-member is
            # wrongly blocked from their own shared session). A per-user session
            # still requires the same owner.
            shared = is_shared_multi_user_session(
                source,
                group_sessions_per_user=getattr(self.config, "group_sessions_per_user", True),
                thread_sessions_per_user=getattr(self.config, "thread_sessions_per_user", False),
            )
            if shared:
                return True
            # Per-user non-DM: the session key includes the participant
            # (``user_id_alt or user_id``). If the caller keys on user_id_alt,
            # the persisted row (user_id only) cannot prove the canonical
            # participant, so fail closed rather than matching on user_id alone.
            if caller_keys_on_alt:
                return False
            return bool(row_uid) and row_uid == caller_uid
        # No caller identity: the persisted row carries only source + user_id
        # (the sessions table has no chat_id), so a same-platform row can belong
        # to a DIFFERENT chat or user. Same-platform alone is therefore NOT
        # ownership proof — an identity-less caller must not bind to, or
        # enumerate, a persisted session by id/title. Fail closed. A legitimate
        # same-chat resume of an ACTIVE session still works through the
        # live-origin branch above (which compares chat_id), and an operator can
        # use the admin --all override. (CWE-639: IDOR on session routing.)
        return False

    async def _resume_row_visible(
        self, source: SessionSource, row: dict, allow_all: bool
    ) -> bool:
        """Whether a titled-session listing *row* belongs to the caller's origin.

        Prevents cross-origin enumeration of session ids/previews via the
        numbered /resume list. Preserves the existing Matrix room-scoping
        semantics; scopes every other platform to the caller's own sessions
        unless an admin passes ``--all``.
        """
        sid = str(row.get("id") or "")
        if source.platform == Platform.MATRIX:
            # Cross-room enumeration is cross-ORIGIN data access: gate the
            # ``--all`` short-circuit behind a real configured admin, exactly
            # like the non-Matrix branch below. A non-admin Matrix ``--all``
            # falls back to same-room scoping rather than exposing every Matrix
            # titled session.
            if allow_all and self._resume_caller_is_admin(source):
                return True
            return self._same_matrix_room(source, self._gateway_session_origin_for_id(sid))
        if allow_all and self._resume_caller_is_admin(source):
            return True
        return await self._resume_target_allowed(source, sid, allow_override=False)
