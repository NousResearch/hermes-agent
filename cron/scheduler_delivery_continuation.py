"""Cron continuation eligibility, transcript mirroring and reply-session seeding."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from cron.scheduler_delivery import _TargetDelivery
    from gateway.session import SessionSource

logger = logging.getLogger("cron.scheduler")


# chat_type slot a platform's adapter puts on a NON-DM in-thread reply. Discord (and the default)
# key the shared "thread" lane; Slack, Matrix and Telegram (forum topics: ``_build_message_event``
# types every supergroup "group") keep the parent channel/room's "group" — a seed on the wrong slot
# is a row no reply ever resolves to (#111896, #112918).
_THREAD_REPLY_CHAT_TYPE = {"slack": "group", "matrix": "group", "telegram": "group"}


def _cron_mirror_delivery_enabled(job: dict, cfg: Optional[dict] = None) -> bool:
    """Whether a cron delivery is also mirrored into the target chat's session transcript.

    Default OFF. Precedence: per-job ``attach_to_session`` (bool) → global
    ``cron.mirror_delivery`` → False. CARVE-OUT: the ``in_channel`` surface seeds its session
    independently of this knob (the seed IS that feature) — this governs only the thread-surface
    mirror. ``mirror_to_session`` runs at a turn boundary, so it is alternation- and cache-safe.
    """
    from cron import scheduler as _sched

    per_job = job.get("attach_to_session")
    if isinstance(per_job, bool):
        return per_job
    try:
        if cfg is None:
            cfg = _sched.load_config() or {}
        return bool((cfg.get("cron", {}) or {}).get("mirror_delivery", False))
    except Exception:
        return False


def _target_matches_origin(
    origin: dict, platform_name: str, chat_id: str, thread_id: Optional[str]
) -> bool:
    """True when a delivery target is the job's own origin conversation. A pinned origin
    thread_id must match — a target without it is a different lane. Mirror eligibility for
    non-origin targets is decided by ``_target_mirror_eligible``."""
    if (
        not origin
        or str(origin.get("platform", "")).lower() != str(platform_name).lower()
        or str(origin.get("chat_id", "")) != str(chat_id)
    ):
        return False
    origin_thread = origin.get("thread_id")
    return origin_thread is None or str(origin_thread) == str(thread_id or "")


def _target_mirror_eligible(
    job: dict, target: dict, *, global_mirror: bool, origin_match: Optional[bool] = None
) -> bool:
    """Whether a resolved delivery target may receive the transcript mirror. Origin targets:
    always. ``origin_fallback`` (deliver=origin with no captured origin → home channel, standing
    in for the primary conversation) and ``home`` (user-written bare-platform token, e.g.
    ``deliver: slack`` — deliberately addresses that platform's home channel): same flags as a
    true origin. ``explicit`` ``platform:chat_id``: ONLY with per-job ``attach_to_session: true``
    — the global flag must never write transcripts into arbitrary explicitly-addressed chats.
    Untagged broadcast expansions (``all``) are never eligible. ``origin_match`` may be
    precomputed."""
    from cron import scheduler_delivery as _delivery

    if origin_match is None:
        origin = _delivery._resolve_origin(job) or {}
        origin_match = _target_matches_origin(
            origin,
            target.get("platform", ""),
            target.get("chat_id", ""),
            target.get("thread_id"),
        )
    if origin_match:
        return True
    resolved_from = target.get("_resolved_from")
    if resolved_from in ("origin_fallback", "home"):
        # Same precedence as _cron_mirror_delivery_enabled (keep in sync): a per-job False must
        # beat a global True even for callers that don't pre-merge `global_mirror`.
        per_job = job.get("attach_to_session")
        return per_job if isinstance(per_job, bool) else bool(global_mirror)
    if resolved_from == "explicit":
        return job.get("attach_to_session") is True
    return False


def _inchannel_seed_allowed(*, is_dm: bool, user_id: Optional[str]) -> bool:
    """Whether the flat in_channel seed may run. Group keys are user-isolated
    (``…:group:<chat_id>:<user_id>``): seeding without a real user_id creates an orphan session no
    reply resolves to — worse than no seed. DM keys omit user_id, so DMs are always seedable."""
    return bool(is_dm or user_id)


def _redact_cron_payload(text: str, what: str) -> str:
    """Fail-closed secret redaction for anything a cron job emits outward.

    Every outward lane — chat message, session mirror, bot-chat turn — must apply the same policy,
    so the policy lives in one place. ``force=True`` because this is a safety boundary, not
    logging: the ``security.redact_secrets`` preference governs how much is scrubbed from the
    user's own logs and must not be able to turn scrubbing off on the way out to a chat (same
    reasoning as ``tools/delegation_live_log.py``). Empty input is returned as-is; any failure
    inside the redactor replaces the payload entirely rather than letting an unscanned value out.
    """
    if not text:
        return text
    try:
        from agent.redact import redact_sensitive_text

        return redact_sensitive_text(text, force=True)
    except Exception as e:
        logger.warning("Failed to redact secrets from cron %s: %s", what, e)
        return "[REDACTED - redaction failed]"


def _cron_display_name(job: dict) -> str:
    """Job name/id as it appears in outward-facing text. The mirror sinks and the thread title
    splice the job *name* around the redacted payload, and the name is user-controlled config — a
    name embedding a credential would re-leak it next to the scrubbed body."""
    return _redact_cron_payload(job.get("name") or job.get("id", "cron"), "job name")


def _cron_mirror_message(job: dict, text: str) -> str:
    return f"[Cron delivery: {_cron_display_name(job)}]\n{text}"


def _maybe_mirror_cron_delivery(
    job: dict,
    platform_name: str,
    chat_id: str,
    mirror_text: str,
    thread_id: Optional[str] = None,
    user_id: Optional[str] = None,
    *,
    enabled: bool = False,
    chat_type: Optional[str] = None,
) -> None:
    """Best-effort mirror of a cron delivery into the origin chat's session. No-op unless
    ``enabled`` (caller resolves it, scoped to the origin target). Rides the same
    ``mirror_to_session`` path as ``send_message``, passing ``user_id`` so user-isolated group
    chats resolve to the scheduling member. All failures swallowed — a successful delivery must
    never be reported failed because the mirror broke."""
    if not enabled:
        return
    text = (mirror_text or "").strip()
    if not text:
        return
    try:
        from gateway.mirror import mirror_to_session

        # USER role + labelled prefix, NOT assistant: an assistant-role mirror lands
        # assistant→assistant and breaks strict alternation; consecutive user turns merge safely.
        # The brief is not the agent speaking; an assistant-role mirror lands as assistant→assistant after
        # the agent's last turn and breaks strict alternation (issue #2221, the exact failure #2313
        # removed). A user-role turn collapses safely via repair_message_sequence's consecutive-user merge
        # on every provider, and the prefix preserves the "this came from cron" context that the dropped
        # SQLite mirror metadata would otherwise lose on replay.
        ok = mirror_to_session(
            platform_name,
            str(chat_id),
            _cron_mirror_message(job, text),
            source_label="cron",
            thread_id=thread_id,
            user_id=user_id,
            role="user",
            **({"chat_type": chat_type} if platform_name == "matrix" else {}),
        )
        if ok:
            logger.info(
                "Job '%s': mirrored delivery into %s:%s session transcript",
                job.get("id", "?"),
                platform_name,
                chat_id,
            )
        else:
            logger.debug(
                "Job '%s': delivery mirror skipped for %s:%s "
                "(no matching gateway session — cold start)",
                job.get("id", "?"),
                platform_name,
                chat_id,
            )
    except Exception as e:
        logger.debug(
            "Job '%s': delivery mirror failed for %s:%s: %s",
            job.get("id", "?"),
            platform_name,
            chat_id,
            e,
        )


def _open_continuable_cron_thread(
    job: dict, adapter, chat_id: str, loop
) -> Optional[str]:
    """Open a thread for a continuable cron job via ``adapter.create_handoff_thread``. Returns the
    thread_id, or ``None`` (no thread primitive / failed) = caller falls back to the DM mirror."""
    create_thread = getattr(adapter, "create_handoff_thread", None)
    if not callable(create_thread) or loop is None:
        return None
    thread_name = f"Hermes — {_cron_display_name(job)}"
    try:
        from agent.async_utils import safe_schedule_threadsafe

        coro = create_thread(str(chat_id), thread_name)
        future = safe_schedule_threadsafe(coro, loop)  # type: ignore[arg-type]
        if future is None:
            return None
        new_thread_id = future.result(timeout=30)
        return str(new_thread_id) if new_thread_id else None
    except Exception as e:
        logger.debug(
            "Job '%s': create_handoff_thread failed on %s — falling back to "
            "DM-session mirror: %s",
            job.get("id", "?"),
            getattr(adapter, "name", "?"),
            e,
        )
        return None


def _seed_cron_session(
    job: dict,
    adapter,
    platform_name: str,
    chat_id: str,
    text: str,
    *,
    thread_id: Optional[str],
    chat_type: str,
    user_id: Optional[str],
    user_name: Optional[str] = None,
    chat_name: Optional[str],
    scope_id: Optional[str],
    discord_keys_on_thread: bool = False,
    destination_source: Optional[SessionSource] = None,
) -> bool:
    """Create the session row (so the mirror has a target) and mirror the brief as a USER turn.
    The seeded key must equal the reply's ``build_session_key``: chat_type, user_id, thread_id and
    scope_id (Slack team id) are all part of it, so callers pass exactly what the reply carries."""
    from gateway.config import Platform
    from gateway.session import SessionSource
    from gateway.mirror import mirror_to_session

    seeded_session_id: Optional[str] = None
    session_store = getattr(adapter, "_session_store", None)
    if session_store is not None:
        try:
            platform_enum = Platform(platform_name.lower())
        except (ValueError, KeyError):
            platform_enum = None
        if platform_enum is not None:
            # Discord keys in-thread messages with chat_id == thread_id; Slack/Telegram use the
            # parent channel.
            seed_chat_id = (
                str(thread_id)
                if discord_keys_on_thread and platform_enum == Platform.DISCORD
                else str(chat_id)
            )
            from gateway.session_identity import replace_source

            base_source = destination_source or SessionSource(
                platform=platform_enum, chat_id=seed_chat_id
            )
            dest_source = replace_source(
                base_source,
                chat_id=seed_chat_id,
                chat_name=chat_name,
                chat_type=chat_type,
                user_id=user_id,
                user_name=user_name,
                thread_id=thread_id,
                scope_id=str(scope_id) if scope_id else None,
            )
            _entry = None
            if platform_enum == Platform.MATRIX:
                participant_isolated = (
                    dest_source.chat_type != "dm"
                    and session_store.config.group_sessions_per_user
                    and (
                        not dest_source.thread_id
                        or session_store.config.thread_sessions_per_user
                    )
                )
                if participant_isolated:
                    _entry = session_store.lookup_by_session_key(
                        session_store._generate_session_key(dest_source)
                    )
                    if _entry is None:
                        return False
            if _entry is None:
                _entry = session_store.get_or_create_session(dest_source)
            seeded_session_id = getattr(_entry, "session_id", None)
    return mirror_to_session(
        platform_name,
        str(chat_id),
        _cron_mirror_message(job, text),
        source_label="cron",
        thread_id=thread_id,
        user_id=user_id,
        role="user",
        session_id=seeded_session_id,
    )


def _seed_cron_thread_session(
    job: dict,
    adapter,
    platform_name: str,
    chat_id: str,
    thread_id: str,
    mirror_text: str,
    chat_name: Optional[str] = None,
    is_dm: bool = False,
    scope_id: Optional[str] = None,
    *,
    destination_source: Optional[SessionSource] = None,
    user_id: Optional[str] = None,
) -> None:
    """Seed the cron delivery's reply session with the brief (never raises).
    Participant-isolated threads require the originating user. A DM thread must seed
    ``chat_type="dm"`` because DM-thread replies use the DM session key.
    Non-DM threads seed the slot the platform's adapter puts on an in-thread reply
    (``_THREAD_REPLY_CHAT_TYPE``)."""
    text = (mirror_text or "").strip()
    if not text:
        return
    session_config = getattr(getattr(adapter, "_session_store", None), "config", None)
    if (
        not is_dm
        and getattr(session_config, "group_sessions_per_user", True) is not False
        and getattr(session_config, "thread_sessions_per_user", False) is True
        and not user_id
    ):
        logger.warning(
            "Job '%s': thread seed skipped for %s:%s thread=%s without an originating participant",
            job.get("id", "?"),
            platform_name,
            chat_id,
            thread_id,
        )
        return
    try:
        ok = _seed_cron_session(
            job,
            adapter,
            platform_name,
            chat_id,
            text,
            thread_id=str(thread_id),
            chat_type="dm"
            if is_dm
            else _THREAD_REPLY_CHAT_TYPE.get(platform_name.lower(), "thread"),
            user_id=user_id or "system:cron",
            user_name=None if user_id else "Cron",
            chat_name=chat_name,
            scope_id=scope_id,
            discord_keys_on_thread=True,
            destination_source=destination_source,
        )
        if ok:
            logger.info(
                "Job '%s': seeded the brief in continuable thread %s on %s:%s",
                job.get("id", "?"),
                thread_id,
                platform_name,
                chat_id,
            )
        else:
            logger.warning(
                "Job '%s': thread seed did NOT land on %s:%s thread=%s — an "
                "in-thread reply will not see this brief",
                job.get("id", "?"),
                platform_name,
                chat_id,
                thread_id,
            )
    except Exception as e:
        # WARNING, not debug: a silent seed failure IS the continuation-amnesia bug.
        logger.warning(
            "Job '%s': seeding cron thread session failed for %s:%s:%s: %s",
            job.get("id", "?"),
            platform_name,
            chat_id,
            thread_id,
            e,
        )


def _seed_cron_channel_session(
    job: dict,
    adapter,
    platform_name: str,
    chat_id: str,
    mirror_text: str,
    *,
    is_dm: bool,
    user_id: Optional[str],
    chat_name: Optional[str] = None,
    scope_id: Optional[str] = None,
    destination_source: Optional[SessionSource] = None,
) -> bool:
    """Seed the FLAT (thread_id=None) session for an ``in_channel`` delivery; True on success.
    ``mirror_to_session`` only APPENDS to an existing session and the flat row is only created by
    an inbound human message, so create the row first or the brief is silently dropped. Group keys
    are user-isolated (``…:group:<chat_id>:<user_id>``): the seed MUST carry the origin's real
    user_id, not ``system:cron``; DM keys omit user_id. chat_type mirrors the inbound handler."""
    text = (mirror_text or "").strip()
    if not text:
        return False
    try:
        chat_type = "dm" if is_dm else "group"
        ok = _seed_cron_session(
            job,
            adapter,
            platform_name,
            chat_id,
            text,
            thread_id=None,  # flat — the whole-channel/DM session
            chat_type=chat_type,
            user_id=str(user_id) if user_id else None,
            chat_name=chat_name,
            scope_id=scope_id,
            destination_source=destination_source,
        )
        if ok:
            logger.info(
                "Job '%s': seeded flat in_channel session on %s:%s (chat_type=%s)",
                job.get("id", "?"),
                platform_name,
                chat_id,
                chat_type,
            )
        return bool(ok)
    except Exception as e:
        # WARNING, not debug: a silent seed failure IS the continuation-amnesia bug.
        logger.warning(
            "Job '%s': seeding in_channel session failed for %s:%s: %s",
            job.get("id", "?"),
            platform_name,
            chat_id,
            e,
        )
        return False


def _seed_live_delivery_sessions(t: _TargetDelivery, delivered_message_id) -> None:
    """After a confirmed live send, seed continuation session(s) and run the generic mirror.
    Thread seeding is deferred here so open-succeeds/deliver-fails never seeds an unseen brief."""
    job = t.job
    origin = t.origin
    if t.platform_name == "matrix":
        from cron.scheduler_delivery_destination import resolve_live_destination

        source = t.resolved_source
        if source is None or source.chat_type not in {"dm", "group"}:
            return
        current = resolve_live_destination(
            t.transport,
            t.platform,
            t.chat_id,
            t.thread_id,
            source.to_dict(),
            t.loop,
        )
        if current is None or current.source.chat_type != source.chat_type:
            return
    seed_kwargs = dict(
        chat_name=origin.get("chat_name"),
        is_dm=t.is_dm_target,
        scope_id=origin.get("scope_id"),
    )
    thread_seeded = False
    inchannel_seeded = False
    seed_thread_id = t.opened_thread_id or (
        t.thread_id if t.resolved_source is not None and t.mirror_this_target else None
    )
    if seed_thread_id:
        _seed_cron_thread_session(
            job,
            t.runtime_adapter,
            t.platform_name,
            t.chat_id,
            seed_thread_id,
            t.mirror_text,
            destination_source=t.resolved_source,
            user_id=t.origin_user_id,
            **seed_kwargs,
        )
        thread_seeded = True
    # in_channel: CREATE + seed the flat session (the mirror only APPENDS to an existing one). Same
    # `inchannel_continuable` gate as the flatten in _deliver_result (must not drift). Origin
    # seed without mirror opt-in; others only via _inchannel_seed_allowed (user-less seed = orphan).
    if t.in_channel_surface and t.inchannel_continuable and not thread_seeded:
        inchannel_seeded = _seed_cron_channel_session(
            job,
            t.runtime_adapter,
            t.platform_name,
            t.chat_id,
            t.mirror_text,
            user_id=t.origin_user_id,
            destination_source=t.resolved_source,
            **seed_kwargs,
        )
        if not inchannel_seeded:
            logger.warning(
                "Job '%s': in_channel seed did NOT land on %s:%s "
                "— a plain reply will not see this brief",
                job["id"],
                t.platform_name,
                t.chat_id,
            )
        # Companion THREAD seed: a reply in the brief's own thread keys to (chat, thread=<ts>),
        # which the flat seed never touches. Seed it too so BOTH reply surfaces continue the job.
        if delivered_message_id:
            _seed_cron_thread_session(
                job,
                t.runtime_adapter,
                t.platform_name,
                t.chat_id,
                str(delivered_message_id),
                t.mirror_text,
                destination_source=t.resolved_source,
                user_id=t.origin_user_id,
                **seed_kwargs,
            )
    elif t.in_channel_surface and not t.inchannel_continuable:
        logger.warning(
            "Job '%s': in_channel delivery to %s:%s is not a "
            "continuable target (origin=%s:%s thread=%s; not the "
            "origin conversation, and not a mirror-eligible "
            "fallback/opted-in target the seed can key) — seed "
            "skipped; the plain mirror below may still apply",
            job["id"],
            t.platform_name,
            t.chat_id,
            origin.get("platform"),
            origin.get("chat_id"),
            origin.get("thread_id"),
        )
    _maybe_mirror_cron_delivery(
        job,
        t.platform_name,
        t.chat_id,
        t.mirror_text,
        thread_id=t.thread_id,
        user_id=t.origin_user_id,
        enabled=t.mirror_this_target and not thread_seeded and not inchannel_seeded,
        chat_type=t.resolved_source.chat_type
        if t.resolved_source is not None
        else None,
    )
