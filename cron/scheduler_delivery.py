"""Cron delivery targets, transports, and receipt-backed delivery outcomes.

This is the defining delivery module for scheduler receipt behavior.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import contextvars
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, List, Optional

from hermes_cli._subprocess_compat import windows_hide_flags
from hermes_cli.config import load_config
from cron.executions import (
    observe_transport_unknown,
    preregister_receipt_plan,
    receipt_summary,
    record_transport_receipt,
)

logger = logging.getLogger("cron.scheduler")

_KNOWN_DELIVERY_PLATFORMS = frozenset({
    "telegram", "discord", "slack", "whatsapp", "signal", "matrix", "mattermost",
    "homeassistant", "dingtalk", "feishu", "wecom", "wecom_callback", "weixin", "sms",
    "email", "webhook", "bluebubbles", "qqbot", "yuanbao",
})
_HOME_TARGET_ENV_VARS = {
    "matrix": "MATRIX_HOME_ROOM", "telegram": "TELEGRAM_HOME_CHANNEL",
    "discord": "DISCORD_HOME_CHANNEL", "slack": "SLACK_HOME_CHANNEL",
    "signal": "SIGNAL_HOME_CHANNEL", "mattermost": "MATTERMOST_HOME_CHANNEL",
    "sms": "SMS_HOME_CHANNEL", "email": "EMAIL_HOME_ADDRESS",
    "dingtalk": "DINGTALK_HOME_CHANNEL", "feishu": "FEISHU_HOME_CHANNEL",
    "wecom": "WECOM_HOME_CHANNEL", "weixin": "WEIXIN_HOME_CHANNEL",
    "bluebubbles": "BLUEBUBBLES_HOME_CHANNEL", "qqbot": "QQBOT_HOME_CHANNEL",
    "whatsapp": "WHATSAPP_HOME_CHANNEL", "whatsapp_cloud": "WHATSAPP_CLOUD_HOME_CHANNEL",
}
_LEGACY_HOME_TARGET_ENV_VARS = {"QQBOT_HOME_CHANNEL": "QQ_HOME_CHANNEL"}

def _resolve_cron_surface_mode(pconfig, logical_platform_name: str) -> str:
    """Resolve the continuable-cron delivery surface for a platform config.

    Returns ``"in_channel"`` or ``"thread"`` (default). Two config shapes:

    - Native adapter: the flat key ``platforms.<p>.extra.cron_continuable_surface``
      (shipped shape, unchanged).
    - Relay-fronted: ``platforms.relay.extra.<logical>.cron_continuable_surface``
      — the same per-logical-platform sub-block the relay's documented Slack
      knobs use (``reply_in_thread``, ``dm_top_level_threads_as_sessions``;
      see RelayAdapter._relay_slack_extra). The sub-block wins over a flat
      key when both exist, matching _relay_slack_extra precedence, and is
      scoped to its logical platform so a ``slack:`` block cannot leak onto
      another fronted platform.

    Precedence nuance vs _relay_slack_extra: that helper is all-or-nothing
    (a sub-dict REPLACES the flat extra entirely), while this one falls back
    to the flat key when the sub-block exists but omits the knob. The
    difference is deliberate — the flat key is the legacy staging shape and
    must keep working — but note a flat ``cron_continuable_surface`` then
    applies to EVERY platform this relay fronts; only the per-platform D6
    capability gate contains it. Scope the knob under the sub-block on
    multi-platform relays.

    Field gap (2026-08-18): the scheduler read only the flat key, so on the
    relay lane — where pconfig is platforms.relay — operators had NO working
    location for the knob and briefs always threaded.
    """
    try:
        extra = getattr(pconfig, "extra", None) or {}
        sub = extra.get(str(logical_platform_name or "").lower())
        if isinstance(sub, dict) and sub.get("cron_continuable_surface") is not None:
            raw = sub.get("cron_continuable_surface")
        else:
            raw = extra.get("cron_continuable_surface")
        if raw is not None and str(raw).strip().lower() == "in_channel":
            return "in_channel"
    except Exception:
        pass
    return "thread"


def _resolve_origin(job: dict) -> Optional[dict]:
    """Extract origin info from a job, preserving any extra routing metadata.

    Treats non-dict origins (free-form provenance strings, ints, lists from
    migration scripts or hand-edited jobs.json) as missing instead of
    crashing with ``AttributeError`` on ``origin.get(...)``. Without this
    guard, a job tagged with e.g. ``"combined-digest-replaces-x-and-y"``
    crashed every fire attempt with
    ``'str' object has no attribute 'get'`` — ``mark_job_run`` recorded the
    failure, but the next tick re-loaded the same poisoned origin and
    crashed identically until the field was patched manually (#18722).
    """
    origin = job.get("origin")
    if type(origin) is not dict:
        return None
    platform = origin.get("platform")
    chat_id = origin.get("chat_id")
    if platform is None or chat_id is None:
        return None
    if type(platform) is not str or type(chat_id) not in {str, int}:
        raise ValueError("origin target identity is invalid")
    thread_id = origin.get("thread_id")
    if thread_id is not None and type(thread_id) not in {str, int}:
        raise ValueError("origin target identity is invalid")
    if platform and chat_id not in {"", 0}:
        return origin
    return None


def _cron_mirror_delivery_enabled(job: dict, cfg: Optional[dict] = None) -> bool:
    """Whether a cron delivery should also be mirrored into the target chat's
    gateway session transcript.

    Default OFF — preserves the historical isolation guarantee (cron deliveries
    live only in the cron job's own session, never the target chat's history)
    byte-for-byte for everyone who does not opt in.

    CARVE-OUT: the ``in_channel`` continuable surface seeds its target
    session independently of this knob (see ``_deliver_result`` /
    ``_seed_cron_channel_session``). in_channel is itself opt-in
    (``cron_continuable_surface: in_channel`` + the adapter capability bit),
    and the seed IS the feature — a continuable flat brief without its seed
    is a brief the next reply can't see. This knob keeps governing the
    SEPARATE default/thread-surface transcript mirror only.

    Precedence (first decisive value wins):
      1. Per-job ``attach_to_session`` (bool) — set via the ``cronjob`` tool,
         lets one briefing job opt in without flipping global behaviour.
      2. Global ``cron.mirror_delivery`` (bool) in config.yaml.
      3. False.

    When enabled, the cron's final output is appended to the target session as
    an assistant turn via the existing ``gateway.mirror.mirror_to_session`` —
    the same primitive ``send_message`` uses — so the next user reply in that
    chat sees the brief in context (no "what is Task #2?" amnesia). This is
    alternation- and cache-safe: the append lands at a turn boundary between
    user turns, never mid-loop, and never mutates the cached system prompt.
    """
    per_job = job.get("attach_to_session")
    if isinstance(per_job, bool):
        return per_job
    try:
        if cfg is None:
            cfg = load_config() or {}
        return bool((cfg.get("cron", {}) or {}).get("mirror_delivery", False))
    except Exception:
        return False


def _target_matches_origin(origin: dict, platform_name: str, chat_id: str,
                           thread_id: Optional[str]) -> bool:
    """True when a delivery target is the job's own origin conversation. A pinned origin
    thread_id must match — a target without it is a different lane. Mirror eligibility for
    non-origin targets is decided by ``_target_mirror_eligible``."""
    if (
        not origin
        or str(origin.get("platform", "")).lower() != str(platform_name).lower()
        or str(origin.get("chat_id", "")) != str(chat_id)
    ):
        return False
    # thread_id must match when the origin pins one (topic-scoped chats); a
    # target that lost the thread_id is not the same conversation lane.
    origin_thread = origin.get("thread_id")
    if origin_thread is not None and str(origin_thread) != str(thread_id or ""):
        return False
    return True


# Provenance rank for the dedup OR-merge in _resolve_delivery_targets (higher = stronger mirror
# claim). Broadcasts rank 0 so "origin,all"/"all,origin" keep the origin tag regardless of order.
_MIRROR_PROVENANCE_RANK = {"origin": 3, "origin_fallback": 2, "home": 2, "explicit": 1}


def _target_mirror_eligible(
    job: dict, target: dict, *, global_mirror: bool, origin_match: Optional[bool] = None) -> bool:
    """Whether a resolved delivery target may receive the transcript mirror. Origin targets:
    always. ``origin_fallback`` (deliver=origin with no captured origin → home channel, standing
    in for the primary conversation) and ``home`` (user-written bare-platform token, e.g.
    ``deliver: slack`` — deliberately addresses that platform's home channel): same flags as a
    true origin. ``explicit`` ``platform:chat_id``: ONLY with per-job ``attach_to_session: true``
    — the global flag must never write transcripts into arbitrary explicitly-addressed chats.
    Untagged broadcast expansions (``all``) are never eligible. ``origin_match`` may be
    precomputed."""
    if origin_match is None:
        origin = _resolve_origin(job) or {}
        origin_match = _target_matches_origin(
            origin, target.get("platform", ""), target.get("chat_id", ""),
            target.get("thread_id"),
        )
    if origin_match:
        return True
    resolved_from = target.get("_resolved_from")
    if resolved_from in ("origin_fallback", "home"):
        # Same precedence as _cron_mirror_delivery_enabled (keep in sync): a per-job False must
        # beat a global True even for callers that don't pre-merge `global_mirror`.
        per_job = job.get("attach_to_session")
        if isinstance(per_job, bool):
            return per_job
        return bool(global_mirror)
    if resolved_from == "explicit":
        return job.get("attach_to_session") is True
    return False


def _inchannel_seed_allowed(*, is_dm: bool, user_id: Optional[str]) -> bool:
    """Whether the flat in_channel session seed may run for a target.

    Group-channel session keys are user-isolated
    (``…:group:<chat_id>:<user_id>`` — see _seed_cron_channel_session); a
    seed without a real user_id would create an orphan session that no
    inbound reply ever resolves to, which is worse than no seed (the plain
    mirror can still land if a session exists). DM keys don't embed
    user_id, so DM targets are always seedable. Origin-captured jobs carry
    the scheduler's user_id; origin-less managed jobs typically don't, and
    their group-channel targets must fall back to the plain mirror.
    """
    return bool(is_dm or user_id)


def _maybe_mirror_cron_delivery(
    job: dict,
    platform_name: str,
    chat_id: str,
    mirror_text: str,
    thread_id: Optional[str] = None,
    user_id: Optional[str] = None,
    *,
    enabled: bool = False,
) -> None:
    """Best-effort mirror of a cron delivery into the origin chat's session.

    No-op unless ``enabled`` (resolved once by the caller, and already scoped to
    the origin target — see ``_target_matches_origin``). Reuses the shipped
    ``mirror_to_session`` so cron rides exactly the same path that interactive
    ``send_message`` mirroring already uses, including passing ``user_id`` so a
    per-user-isolated group chat resolves to the exact member who scheduled the
    job (parity with ``send_message``). All failures are swallowed — a delivery
    that succeeded must never be reported as failed because the transcript
    mirror hit a problem.

    Because the caller only enables this for the target that equals the job's
    origin conversation, the session is expected to exist (the job was born in
    that session). A missing session therefore indicates an origin-less /
    fan-out delivery that should not have been mirrored anyway, and is treated
    as a silent no-op — never a synthetic session is created.
    """
    if not enabled:
        return
    text = (mirror_text or "").strip()
    if not text:
        return
    try:
        from gateway.mirror import mirror_to_session

        # Mirror as a USER turn with a labelled prefix, NOT an assistant turn.
        # The brief is not the agent speaking; an assistant-role mirror lands as
        # assistant→assistant after the agent's last turn and breaks strict
        # alternation (issue #2221, the exact failure #2313 removed). A
        # user-role turn collapses safely via repair_message_sequence's
        # consecutive-user merge on every provider, and the prefix preserves the
        # "this came from cron" context that the dropped SQLite mirror metadata
        # would otherwise lose on replay.
        ok = mirror_to_session(
            platform_name,
            str(chat_id),
            f"[Cron delivery: {job.get('name') or job.get('id', 'cron')}]\n{text}",
            source_label="cron",
            thread_id=thread_id,
            user_id=user_id,
            role="user",
        )
        if ok:
            logger.info(
                "Job '%s': mirrored delivery into %s:%s session transcript",
                job.get("id", "?"), platform_name, chat_id,
            )
        else:
            logger.debug(
                "Job '%s': delivery mirror skipped for %s:%s "
                "(no matching gateway session — cold start)",
                job.get("id", "?"), platform_name, chat_id,
            )
    except Exception as e:
        logger.debug(
            "Job '%s': delivery mirror failed for %s:%s: %s",
            job.get("id", "?"), platform_name, chat_id, e,
        )


def _open_continuable_cron_thread(
    job: dict,
    adapter,
    chat_id: str,
    loop,
) -> Optional[str]:
    """Open a dedicated thread for a continuable cron job (thread-preferred).

    Returns the new ``thread_id`` on success, or ``None`` when the platform has
    no thread primitive (WhatsApp/Signal/SMS) or creation failed — the ``None``
    return is the caller's signal to fall back to the origin-DM mirror, the same
    open-thread-or-fallback shape as ``GatewayRunner._process_handoff``. Reuses
    the shipped ``adapter.create_handoff_thread``; no new adapter surface.
    """
    create_thread = getattr(adapter, "create_handoff_thread", None)
    if not callable(create_thread) or loop is None:
        return None
    task_name = job.get("name") or job.get("id", "cron")
    thread_name = f"Hermes — {task_name}"
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
            job.get("id", "?"), getattr(adapter, "name", "?"), e,
        )
        return None


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
) -> None:
    """Seed the freshly-opened cron thread's session with the brief.

    Without this the brief is *visible* in the new thread but absent from any
    transcript, so the user's first reply in-thread would hit a session with no
    record of it ("what is Task #2?"). We create the thread-keyed session (the
    same key the user's reply will resolve to — ``build_session_key`` keys
    threads as participant-shared, so no ``user_id`` is needed) and append the
    brief as an assistant turn via the shipped ``mirror_to_session``.

    ``scope_id`` is the workspace/server scope (Slack team id).
    ``build_session_key`` embeds it in every Slack key, so a scoped reply's
    key carries it — the seed must reproduce it or the seeded row is
    unreachable (the scope-less flat-seed sibling of the is_dm keying bug).
    Best-effort None for platforms without scope.

    ``is_dm`` selects the seeded ``chat_type``: a thread under a DM must seed
    ``chat_type="dm"`` because the user's in-thread DM reply arrives with
    chat_type="dm" and ``build_session_key`` routes DM threads through the DM
    arm (``...:dm:<chat>:<thread>``) — a "thread"-typed seed lands in
    ``...:thread:<chat>:<thread>``, a row no DM reply ever resolves to
    (continuation amnesia, Alice live 2026-08-20, job 8e21a957b77b). Channel
    threads keep ``chat_type="thread"`` (their replies really do arrive as
    threads). Same sibling-lane class as the flat seed's ``is_dm``
    (dcca9d8cfe).

    Mirrors ``GatewayRunner._process_handoff``'s seed step, but standalone:
    cron reaches the live ``SessionStore`` through the adapter's
    ``_session_store`` handle rather than the gateway object. Best-effort — a
    delivery that already succeeded is never failed by a seeding problem.
    """
    text = (mirror_text or "").strip()
    if not text:
        return
    try:
        from gateway.config import Platform
        from gateway.session import SessionSource

        seeded_session_id: Optional[str] = None
        session_store = getattr(adapter, "_session_store", None)
        if session_store is not None:
            try:
                platform_enum = Platform(platform_name.lower())
            except (ValueError, KeyError):
                platform_enum = None
            if platform_enum is not None:
                # Discord thread destinations must key on the thread's OWN id
                # to match how the Discord adapter keys organic in-thread
                # messages (chat_id == thread_id). Other platforms (Slack,
                # Telegram) use chat_id == parent_channel for thread messages,
                # so the parent chat_id is correct for them. See the matching
                # guard in GatewayRunner._process_handoff.
                if platform_enum == Platform.DISCORD:
                    seed_chat_id = str(thread_id)
                else:
                    seed_chat_id = str(chat_id)
                dest_source = SessionSource(
                    platform=platform_enum,
                    chat_id=seed_chat_id,
                    chat_name=chat_name,
                    # DM threads key through the DM arm (see docstring); the
                    # reply's chat_type is what the seed must reproduce.
                    chat_type="dm" if is_dm else "thread",
                    user_id="system:cron",
                    user_name="Cron",
                    thread_id=str(thread_id),
                    scope_id=str(scope_id) if scope_id else None,
                )
                # Ensure the thread-keyed session row exists so the mirror has
                # a target and the user's later reply joins the same session.
                # Capture the exact id — the mirror writes into THIS row, not
                # an origin-heuristic rediscovery (which bails on populated
                # chats; same class as the flat-seed live failure 2026-08-19).
                _entry = session_store.get_or_create_session(dest_source)
                seeded_session_id = getattr(_entry, "session_id", None)

        from gateway.mirror import mirror_to_session

        # User-role + labelled prefix (see _maybe_mirror_cron_delivery): the
        # seeded brief must not read as an assistant turn, or the user's first
        # in-thread reply produces assistant→user→... off a phantom assistant
        # message. Pass the seed user_id so the mirror resolves the exact
        # thread-keyed session row we just created.
        ok = mirror_to_session(
            platform_name,
            str(chat_id),
            f"[Cron delivery: {job.get('name') or job.get('id', 'cron')}]\n{text}",
            source_label="cron",
            thread_id=str(thread_id),
            user_id="system:cron",
            role="user",
            session_id=seeded_session_id,
        )
        if ok:
            logger.info(
                "Job '%s': opened continuable thread %s on %s:%s and seeded the brief",
                job.get("id", "?"), thread_id, platform_name, chat_id,
            )
        else:
            logger.warning(
                "Job '%s': thread seed did NOT land on %s:%s thread=%s — an "
                "in-thread reply will not see this brief",
                job.get("id", "?"), platform_name, chat_id, thread_id,
            )
    except Exception as e:
        # WARNING, not debug: a silent seed failure IS the continuation-
        # amnesia bug (Alice 2026-08-19) — it must be visible in production.
        logger.warning(
            "Job '%s': seeding cron thread session failed for %s:%s:%s: %s",
            job.get("id", "?"), platform_name, chat_id, thread_id, e,
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
) -> bool:
    """Seed the FLAT (thread_id=None) session for an ``in_channel`` cron delivery.

    The ``in_channel`` surface (D1/D2) delivers the brief flat into the channel
    with no thread, so the continuation surface is the whole-channel /
    whole-DM session keyed ``thread_id=None`` — the same bucket
    ``reply_in_thread: false`` routes an inbound plain reply to.

    Unlike the thread path, the shipped delivery-mirror alone is NOT sufficient
    here: ``mirror_to_session`` only APPENDS to a session that already EXISTS
    (``_find_session_id`` → no-op when none matches), and a flat channel
    ``(…, None)`` row is only created when a human posts a top-level message the
    bot processes — a ``chat_postMessage`` cron delivery never goes through the
    inbound handler, so the row is usually absent and the mirror silently drops
    the brief (verified live: the brief never landed, the reply had no context).
    So we CREATE the flat session row first, exactly like
    ``_seed_cron_thread_session`` does for threads, then mirror into it.

    The session KEY must match what the user's later inbound reply resolves to
    (``build_session_key``):
    - **Channel** (``chat_type="group"``): key is
      ``…:group:<chat_id>:<user_id>`` — user-isolated — so the seed MUST carry
      the **origin's real ``user_id``** (the member who scheduled the job), NOT
      a synthetic ``system:cron`` id, or the reply keys to a different session.
    - **1:1 DM** (``chat_type="dm"``): the key is ``…:dm:<chat_id>`` and does
      NOT embed ``user_id``, so any ``user_id`` resolves to the same session.
    ``chat_type`` mirrors the inbound handler's own choice
    (``"dm" if is_dm else "group"``, ``adapter.py``), so the seeded key is
    byte-identical to the reply's key.

    Returns True if a seed row was created and the brief mirrored, else False
    (caller falls back to the plain mirror). Best-effort — a delivery that
    already succeeded is never failed by a seeding problem.
    """
    text = (mirror_text or "").strip()
    if not text:
        return False
    try:
        from gateway.config import Platform
        from gateway.session import SessionSource

        chat_type = "dm" if is_dm else "group"
        session_store = getattr(adapter, "_session_store", None)
        seeded_session_id: Optional[str] = None
        if session_store is not None:
            try:
                platform_enum = Platform(platform_name.lower())
            except (ValueError, KeyError):
                platform_enum = None
            if platform_enum is not None:
                dest_source = SessionSource(
                    platform=platform_enum,
                    chat_id=str(chat_id),
                    chat_name=chat_name,
                    chat_type=chat_type,
                    user_id=str(user_id) if user_id else None,
                    thread_id=None,  # flat — the whole-channel/DM session
                    # Workspace scope: build_session_key embeds it in every
                    # Slack key, so a scoped reply only resolves to this row
                    # when the seed carries it too (see thread-seed docstring).
                    scope_id=str(scope_id) if scope_id else None,
                )
                # Create the flat session row so the mirror has a target and the
                # user's later plain reply joins the SAME session. Capture the
                # exact session id: the mirror must write into THIS row, not
                # re-discover it via origin heuristics (which bail out on
                # populated chats where the flat session coexists with
                # per-message thread sessions — live failure, Alice 2026-08-19).
                _entry = session_store.get_or_create_session(dest_source)
                seeded_session_id = getattr(_entry, "session_id", None)

        from gateway.mirror import mirror_to_session

        ok = mirror_to_session(
            platform_name,
            str(chat_id),
            f"[Cron delivery: {job.get('name') or job.get('id', 'cron')}]\n{text}",
            source_label="cron",
            thread_id=None,
            user_id=str(user_id) if user_id else None,
            session_id=seeded_session_id,
            role="user",
        )
        if ok:
            logger.info(
                "Job '%s': seeded flat in_channel session on %s:%s (chat_type=%s)",
                job.get("id", "?"), platform_name, chat_id, chat_type,
            )
        return bool(ok)
    except Exception as e:
        # WARNING, not debug: a silent seed failure IS the "agent has no idea
        # about its own brief" bug (Alice 2026-08-19) — it must be visible in
        # production logs.
        logger.warning(
            "Job '%s': seeding in_channel session failed for %s:%s: %s",
            job.get("id", "?"), platform_name, chat_id, e,
        )
        return False


def _cron_job_origin_log_suffix(job: dict) -> str:
    """Return safe provenance details for security warnings about a cron job.

    The scheduler normally has no live HTTP request object when it detects a
    bad stored ``context_from`` reference. Including the job's saved origin
    makes future probe logs actionable without exposing secrets: platform/chat
    metadata for gateway-created jobs, and optional source-IP fields for API
    surfaces that persist them in origin metadata.
    """
    origin = job.get("origin")
    if not isinstance(origin, dict):
        return ""

    fields = []
    for key in ("platform", "chat_id", "thread_id", "source_ip", "remote", "forwarded_for"):
        value = origin.get(key)
        if value is None:
            continue
        text = str(value).replace("\r", " ").replace("\n", " ").strip()
        if text:
            fields.append(f"origin_{key}={text[:200]!r}")
    return " " + " ".join(fields) if fields else ""


def _plugin_cron_env_var(platform_name: str) -> str:
    """Return the cron home-channel env var registered by a plugin platform.

    Falls through the platform registry so plugins that set
    ``cron_deliver_env_var`` on their ``PlatformEntry`` get cron delivery
    support without editing this module.
    """
    try:
        from hermes_cli.plugins import discover_plugins
        discover_plugins()  # idempotent
        from gateway.platform_registry import platform_registry
        entry = platform_registry.get(platform_name.lower())
        if entry and entry.cron_deliver_env_var:
            return entry.cron_deliver_env_var
    except Exception:
        pass
    return ""


def _is_known_delivery_platform(platform_name: str) -> bool:
    """Whether ``platform_name`` is a valid cron delivery target.

    Hardcoded built-ins in ``_KNOWN_DELIVERY_PLATFORMS`` are checked first;
    plugin platforms registered via ``PlatformEntry`` are accepted if they
    provide a ``cron_deliver_env_var``.
    """
    name = platform_name.lower()
    if name in _KNOWN_DELIVERY_PLATFORMS:
        return True
    return bool(_plugin_cron_env_var(name))


def _resolve_home_env_var(platform_name: str) -> str:
    """Return the env var name for a platform's cron home channel.

    Built-in platforms are in ``_HOME_TARGET_ENV_VARS``; plugin platforms are
    resolved from the platform registry.
    """
    name = platform_name.lower()
    env_var = _HOME_TARGET_ENV_VARS.get(name)
    if env_var:
        return env_var
    return _plugin_cron_env_var(name)


def _get_config_home_channel(platform_name: str):
    """Return the persisted ``HomeChannel`` for a platform from gateway config.

    ``/sethome`` declares ``config.yaml`` canonical (it is the only store that
    survives for relay-fronted logical platforms, whose adapters are not
    natively enabled) and mirrors the value into the legacy
    ``<PLATFORM>_HOME_CHANNEL`` env var only as a best-effort compatibility
    shim.  Cron historically read ONLY the env mirror, so a home channel that
    existed solely in config.yaml — e.g. Discord fronted by the relay
    connector, where no ``DISCORD_HOME_CHANNEL`` was ever exported — was
    invisible and jobs silently fell back to local-only.  Reading the
    canonical store here fixes that for every relay-fronted platform at once.
    """
    try:
        from gateway.config import load_gateway_config, Platform

        config = load_gateway_config()
        platform = Platform(platform_name.lower())
        return config.get_home_channel(platform)
    except Exception:
        logger.debug(
            "config home_channel lookup failed for platform %r",
            platform_name, exc_info=True,
        )
        return None


def _env_home_target_chat_id(platform_name: str) -> str:
    """Return the home chat id from the legacy env mirror only (no config).

    Reads through ``get_secret`` (not raw ``os.getenv``) so a profile-scoped
    secret scope wins in a multiplex gateway. ``DISCORD_HOME_CHANNEL`` lives in
    each profile's ``.env``; in a multiplex process the winning cron tick runs
    with the job-owning profile's scope installed (run_one_job sets it), so
    reading via ``get_secret`` resolves the OWNING profile's chat id rather
    than the host process's ``os.environ`` (#83182, chat-id leg — the token
    leg was fixed earlier; chat id / thread id resolve through the same leak).
    """
    env_var = _resolve_home_env_var(platform_name)
    if not env_var:
        return ""
    try:
        from agent.secret_scope import get_secret
    except Exception:
        get_secret = None  # type: ignore
    if get_secret is not None:
        value = get_secret(env_var, "")
        if not value:
            legacy = _LEGACY_HOME_TARGET_ENV_VARS.get(env_var)
            if legacy:
                value = get_secret(legacy, "")
        return value or ""
    value = os.getenv(env_var, "")
    if not value:
        legacy = _LEGACY_HOME_TARGET_ENV_VARS.get(env_var)
        if legacy:
            value = os.getenv(legacy, "")
    return value


def _get_home_target_chat_id(platform_name: str) -> str:
    """Return the configured home target chat/room ID for a delivery platform.

    Resolution order: platform env var (legacy mirror, kept first so an
    operator override keeps winning) → legacy env var name → the canonical
    ``home_channel`` block persisted in config.yaml by ``/sethome``.
    """
    value = _env_home_target_chat_id(platform_name)
    if value:
        return value
    home = _get_config_home_channel(platform_name)
    if home is not None and home.chat_id:
        return str(home.chat_id)
    return ""


def _get_home_target_thread_id(platform_name: str) -> Optional[str]:
    """Return the optional thread/topic ID for a platform home target.

    Telegram-only override: ``TELEGRAM_CRON_THREAD_ID`` takes precedence over
    ``TELEGRAM_HOME_CHANNEL_THREAD_ID`` for cron delivery. When topic mode is
    enabled, deliveries that land in the root DM (thread_id unset) end up in
    the system-only lobby where the user cannot reply — the gateway returns
    the lobby reminder and drops ``reply_to_message_id`` (#24409). Pointing
    cron at a dedicated topic via this env var lets replies work as expected
    without changing the lobby invariant.
    """
    env_var = _resolve_home_env_var(platform_name)
    try:
        from agent.secret_scope import get_secret
    except Exception:
        get_secret = None  # type: ignore

    def _scope_get(name: str) -> str:
        if get_secret is None:
            return ""
        v = get_secret(name, "")
        return v if v is not None else ""

    if platform_name.lower() == "telegram":
        cron_thread = _scope_get("TELEGRAM_CRON_THREAD_ID").strip()
        if cron_thread:
            return cron_thread
    if get_secret is not None:
        value = _scope_get(f"{env_var}_THREAD_ID").strip() if env_var else ""
        if not value and env_var:
            legacy = _LEGACY_HOME_TARGET_ENV_VARS.get(env_var)
            if legacy:
                value = _scope_get(f"{legacy}_THREAD_ID").strip()
    else:
        value = os.getenv(f"{env_var}_THREAD_ID", "").strip() if env_var else ""
        if not value and env_var:
            legacy = _LEGACY_HOME_TARGET_ENV_VARS.get(env_var)
            if legacy:
                value = os.getenv(f"{legacy}_THREAD_ID", "").strip()
    if value:
        return value
    # Canonical config.yaml fallback — same rationale as
    # _get_home_target_chat_id, and thread affinity only applies when the
    # chat itself resolved from the same config block (an env-provided chat
    # id keeps its env-provided thread semantics).
    if not _env_home_target_chat_id(platform_name):
        home = _get_config_home_channel(platform_name)
        if home is not None and home.thread_id:
            return str(home.thread_id)
    return None


def _iter_home_target_platforms():
    """Iterate built-in + plugin platform names that expose a home channel.

    Used by the ``deliver=origin`` fallback when the job has no origin.
    """
    for name in _HOME_TARGET_ENV_VARS:
        yield name
    try:
        from hermes_cli.plugins import discover_plugins
        discover_plugins()  # idempotent
        from gateway.platform_registry import platform_registry
        for entry in platform_registry.plugin_entries():
            if entry.cron_deliver_env_var and entry.name not in _HOME_TARGET_ENV_VARS:
                yield entry.name
    except Exception:
        pass


def _relay_fronted_delivery_platforms(connected: set) -> set:
    """Logical platforms deliverable through a connected relay connector.

    ``get_connected_platforms()`` only sees NATIVELY configured platforms.
    On a relay-fronted deployment (relay in ``config.platforms``, the real
    platform credential living in the connector) the fronted platforms are
    absent from that set although fire-time routing delivers to them via
    ``resolve_delivery_transport`` + ``RelayAdapter.fronts_platform``. This
    keeps validation symmetric with routing by consulting the same
    env-derived deploy stamp (``GATEWAY_RELAY_PLATFORMS``) the live
    adapter's identity set is seeded from. No relay connected -> empty set,
    so native topologies keep the strict credential check unchanged.
    """
    if "relay" not in connected:
        return set()
    try:
        from gateway.relay import relay_fronted_platforms

        return relay_fronted_platforms()
    except Exception:
        logger.debug("relay fronted-platform lookup failed", exc_info=True)
        return set()


def cron_delivery_targets() -> list[dict]:
    """Return the platforms a cron job can auto-deliver to.

    Single source of truth for any UI (dashboard dropdown, etc.) that lets a
    user pick a cron delivery target. A platform is included when it is a valid
    cron delivery platform AND its gateway is configured (enabled + credentials
    present). Each entry reports whether the platform's home target (the
    room/channel cron posts to) is set — a platform can be configured for
    interactive use but still lack the home target an unattended cron job needs.

    Returns a list of dicts: ``{"id", "name", "home_target_set", "home_env_var"}``
    ordered by the gateway's canonical platform order. Callers should always
    prepend the implicit ``local`` option themselves — it needs no config.
    """
    targets: list[dict] = []
    try:
        from gateway.config import load_gateway_config

        gateway_config = load_gateway_config()
        connected = {p.value for p in gateway_config.get_connected_platforms()}
        connected |= _relay_fronted_delivery_platforms(connected)
    except Exception:
        logger.debug("cron_delivery_targets: gateway config unavailable", exc_info=True)
        connected = set()

    for name in _iter_home_target_platforms():
        if name not in connected:
            continue
        if not _is_known_delivery_platform(name):
            continue
        env_var = _resolve_home_env_var(name)
        targets.append(
            {
                "id": name,
                "name": name.replace("_", " ").title(),
                "home_target_set": bool(_get_home_target_chat_id(name)),
                "home_env_var": env_var or None,
            }
        )

    # Bot Chat targets: one per local profile. Machine-local by design (the
    # scheduler delivers via a local chat subprocess), so the names listed
    # here are exactly the names that resolve at fire time — no gateway
    # config, no home channel needed.
    try:
        from hermes_cli.profiles import list_profile_names

        for profile_name in list_profile_names():
            targets.append(
                {
                    "id": f"{BOT_CHAT_PLATFORM}:{profile_name}",
                    "name": f"Bot Chat ({profile_name})",
                    "home_target_set": True,
                    "home_env_var": None,
                }
            )
    except Exception:
        logger.debug("cron_delivery_targets: profile listing unavailable", exc_info=True)
    return targets


def _origin_thread_is_stale(origin: dict) -> bool:
    """True when a Slack origin's thread is a stale creation-turn artifact.

    Relay-fronted Slack in thread-per-message mode stamps each top-level
    message's own id as the session thread (a session KEY, not a durable
    location). Jobs persisted before origin capture learned to drop that
    stamp carry it as ``origin.thread_id`` forever. Heuristic that repairs
    them at fire time without touching genuine threads: when the origin
    chat IS the configured Slack home chat (the ``/sethome`` conversation),
    a pinned origin thread is the creation-message artifact — the user's
    delivery expectation for their home conversation is top-level (or the
    home target's own configured thread). Non-home chats keep their
    threads: a job deliberately created inside a working thread stays there.
    """
    if str(origin.get("platform") or "").lower() != "slack":
        return False
    if not origin.get("thread_id"):
        return False
    home_chat = _get_home_target_chat_id("slack")
    return bool(home_chat) and str(origin.get("chat_id")) == str(home_chat)


def _origin_delivery_thread(origin: dict):
    """The thread a deliver=origin job should use, stale stamps dropped."""
    if _origin_thread_is_stale(origin):
        home_thread = _get_home_target_thread_id("slack")
        return home_thread if home_thread else None
    return origin.get("thread_id")


def _home_target(platform_name: str, chat_id: str, resolved_from: Optional[str] = None) -> dict:
    """Target dict for a platform's configured home channel (+ optional mirror provenance)."""
    target = {
        "platform": platform_name,
        "chat_id": chat_id,
        "thread_id": _get_home_target_thread_id(platform_name)}
    if resolved_from:
        target["_resolved_from"] = resolved_from
    return target


def _resolve_single_delivery_target(
    job: dict, deliver_value: str, *, from_broadcast: bool = False
) -> Optional[dict]:
    """Resolve one concrete auto-delivery target for a cron job.

    ``from_broadcast`` marks a bare-platform token that was produced by expanding a broadcast
    token (``all``) rather than written by the user; broadcast expansions carry no mirror
    provenance (fan-out is never continuable), while a user-written bare platform token is a
    deliberate home-channel address and gets the ``home`` tag."""
    origin = _resolve_origin(job)

    if deliver_value == "local":
        return None

    # bot-chat[:<profile>] — checked before the generic platform:chat_id
    # split below so the profile-name argument is never misparsed as a
    # chat_id on an unknown platform.
    bot_chat_profile = parse_bot_chat_deliver_token(deliver_value)
    if bot_chat_profile is not None:
        return _resolve_bot_chat_target(job, bot_chat_profile)

    if deliver_value == "origin":
        if origin:
            return {
                "platform": origin["platform"],
                "chat_id": str(origin["chat_id"]),
                "thread_id": _origin_delivery_thread(origin),
                # Resolution provenance for mirror eligibility (see
                # _target_mirror_eligible): this IS the origin conversation.
                "_resolved_from": "origin",
            }
        # Origin missing (e.g. job created via API/script) — try each
        # platform's home channel as a fallback instead of silently dropping.
        for platform_name in _iter_home_target_platforms():
            chat_id = _get_home_target_chat_id(platform_name)
            if chat_id:
                logger.info(
                    "Job '%s' has deliver=origin but no origin; falling back to %s home channel",
                    job.get("name", job.get("id", "?")),
                    platform_name,
                )
                return {
                    "platform": platform_name,
                    "chat_id": chat_id,
                    "thread_id": _get_home_target_thread_id(platform_name),
                    # The fallback stands in for the user's primary
                    # conversation (NOT a broadcast) — mirror-eligible so
                    # continuable crons work for script-provisioned jobs
                    # that never captured an origin.
                    "_resolved_from": "origin_fallback",
                }
        return None

    if ":" in deliver_value:
        platform_name, rest = deliver_value.split(":", 1)
        platform_key = platform_name.lower()

        from tools.send_message_tool import (
            prepare_send_message_platforms,
            resolve_send_target,
        )

        prepare_send_message_platforms()
        # pass_unresolved_references: stored jobs have no model in the loop to react
        # to a resolution error, and a target the directory doesn't know
        # (fresh install, platform-native id) used to be handed to the
        # adapter as written. Dropping it here silently loses the job's
        # output.
        chat_id, thread_id, resolution_error = resolve_send_target(
            platform_key, rest, pass_unresolved_references=True
        )
        if resolution_error:
            logger.warning(
                "Invalid cron delivery target '%s': %s",
                deliver_value,
                resolution_error,
            )
            return None

        if (
            thread_id is None
            and platform_key == "slack"
            and origin
            and str(origin.get("platform") or "").lower() == platform_key
            and str(origin.get("chat_id")) == str(chat_id)
            and origin.get("thread_id")
            and not _origin_thread_is_stale(origin)
        ):
            thread_id = origin.get("thread_id")

        return {
            "platform": platform_name,
            "chat_id": chat_id,
            "thread_id": thread_id,
            # Explicit platform:chat target — mirror-eligible only under the
            # job's own attach_to_session opt-in (see _target_mirror_eligible).
            "_resolved_from": "explicit",
        }

    platform_name = deliver_value
    home_provenance = None if from_broadcast else "home"
    if origin and origin.get("platform") == platform_name:
        chat_id = _get_home_target_chat_id(platform_name)
        if chat_id:
            return _home_target(platform_name, chat_id, home_provenance)
        # No home configured: falls back to the origin chat. No tag needed — the
        # origin-match check in _target_mirror_eligible already covers this target.
        return {
            "platform": platform_name,
            "chat_id": str(origin["chat_id"]),
            "thread_id": origin.get("thread_id"),
        }

    if not _is_known_delivery_platform(platform_name):
        return None
    chat_id = _get_home_target_chat_id(platform_name)
    return _home_target(platform_name, chat_id, home_provenance) if chat_id else None


def _get_bot_chat_delivery_timeout() -> int:
    """Timeout for one bot-chat delivery turn (the target bot runs a full
    agent turn on the injected output, so this is minutes, not seconds).

    ``cron.bot_chat_delivery_timeout_seconds`` in config.yaml; default 600.
    """
    try:
        cfg = load_config()
        value = int(cfg.get("cron", {}).get("bot_chat_delivery_timeout_seconds", 600))
        return value if value > 0 else 600
    except Exception:
        return 600


def _bot_chat_query_message(job: dict, content: str) -> str:
    """Compose the exact child query bytes for planning and dispatch."""
    job_name = job.get("name", job.get("id", "?"))
    return (
        f'[Cronjob "{job_name}" output — scheduled job, not the user. '
        f"Review it, act on anything that needs action, and summarize "
        f"for the chat.]\n\n{content}"
    )


def _deliver_to_bot_chat(job: dict, content: str, profile: str) -> Optional[str]:
    """Hand output to its live Bot Chat owner, or use the unowned CLI lane.

    Admission is not delivery: queued/claimed results remain unverified and
    never authorize a CLI replay. ``profile`` is empty for the job's profile.
    """
    import hashlib
    import json
    import shutil as _shutil
    import tempfile
    import uuid
    from hermes_constants import get_hermes_home
    from hermes_cli.profiles import get_profile_dir
    from tools.bot_live_delivery import (
        deliver_to_live_owner, find_canonical_live_owner, read_delivery_result,
    )

    job_id = job.get("id", "?")
    message = _bot_chat_query_message(job, content)
    try:
        source_home = get_hermes_home().resolve()
        home = (get_profile_dir(profile) if profile else source_home).resolve()
        run_id = job.get("execution_id")
        if not run_id:
            run_id = job.setdefault("_bot_chat_run_id", uuid.uuid4().hex)
        key = hashlib.sha256(json.dumps(
            [str(source_home), job_id, str(run_id), str(home)],
            ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        # Read before discovery: an owner can exit after admission. No receipt
        # state, including failed/ambiguous, permits a second-writer fallback.
        receipt = read_delivery_result(home, key)
        if receipt is None:
            owner = find_canonical_live_owner(home)
            if owner is not None:
                receipt = deliver_to_live_owner(home, owner, message, delivery_id=key)
        if receipt is not None:
            if receipt["message"] != message:
                raise ValueError("delivery id already belongs to a different payload")
            status = receipt["status"]
            target = f"bot-chat:{profile or '(own)'}"
            job.setdefault("_bot_chat_delivery_receipts", {})[target] = {
                "status": status, "delivery_id": key,
            }
            if status == "settled":
                return None
            if status in ("queued", "claimed", "ambiguous", "failed"):
                return f"bot-chat {status}; completion unverified; do not resend"
            return "bot-chat delivery confirmation unavailable"
    except Exception:
        return "bot-chat delivery confirmation unavailable"

    hermes_bin = _shutil.which("hermes")
    if hermes_bin:
        argv = [hermes_bin]
    else:
        try:
            import importlib.util as _ilu

            if _ilu.find_spec("hermes_cli") is not None:
                argv = [sys.executable, "-m", "hermes_cli.main"]
            else:
                return "bot-chat delivery failed"
        except Exception:
            return "bot-chat delivery failed"

    from agent.delegation_context import delegated_child_subprocess_env
    env = delegated_child_subprocess_env(os.environ)
    if profile:
        argv += ["-p", profile]
        # -p owns profile resolution in the child; a leftover HERMES_HOME
        # from THIS scheduler's profile must not shadow it.
        env.pop("HERMES_HOME", None)
    else:
        env["HERMES_HOME"] = str(source_home)

    query_file = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", suffix=".txt", prefix="hermes-cron-botchat-",
            delete=False,
        ) as fh:
            fh.write(message)
            query_file = fh.name

        argv += [
            "chat", "--in", "~", "-c", "Bot Chat", "--create-if-missing",
            "-Q", "--query-file", query_file,
        ]

        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=_get_bot_chat_delivery_timeout(),
            env=env,
            creationflags=windows_hide_flags(),
        )
        if result.returncode != 0:
            logger.warning("Job '%s': bot-chat delivery confirmation unavailable", job_id)
            return "bot-chat delivery confirmation unavailable"
        logger.info(
            "Job '%s': bot-chat child completed without provider receipt", job_id
        )
        return None
    except subprocess.TimeoutExpired:
        logger.warning("Job '%s': bot-chat delivery confirmation unavailable", job_id)
        return "bot-chat delivery confirmation unavailable"
    except Exception:
        logger.warning("Job '%s': bot-chat delivery confirmation unavailable", job_id)
        return "bot-chat delivery confirmation unavailable"
    finally:
        if query_file:
            try:
                os.unlink(query_file)
            except OSError:
                pass


def _normalize_deliver_value(deliver) -> str:
    """Normalize a stored/submitted ``deliver`` value to its canonical string form.

    The contract is that ``deliver`` is a string (``"local"``, ``"origin"``,
    ``"telegram"``, ``"telegram:-1001:17"``, or comma-separated combinations).
    Historically some callers — MCP clients passing an array, direct edits of
    ``jobs.json``, or stale code paths — have stored a list/tuple like
    ``["telegram"]``.  ``str(["telegram"])`` would serialize to the literal
    string ``"['telegram']"``, which is not a known platform and fails
    resolution silently.  Flatten lists/tuples into a comma-separated string
    so both forms work.  Returns ``"local"`` for anything falsy.
    """
    if deliver is None:
        return "local"
    if type(deliver) is str:
        return deliver or "local"
    if type(deliver) in {list, tuple}:
        parts = [p.strip() for p in deliver if type(p) is str and p.strip()]
        return ",".join(parts) if parts else "local"
    return "local"


def _normalize_delivery_target_identity(target: Any) -> dict:
    """Return a content-free target using only inert built-in scalar values."""
    if type(target) is not dict:
        raise ValueError("delivery target must be an object")
    platform = target.get("platform")
    chat_id = target.get("chat_id")
    thread_id = target.get("thread_id")
    if type(platform) is not str or not platform:
        raise ValueError("delivery target platform is invalid")
    if type(chat_id) not in {str, int} or chat_id in {"", 0}:
        raise ValueError("delivery target chat_id is invalid")
    if thread_id is not None and (
        type(thread_id) not in {str, int} or thread_id in {"", 0}
    ):
        raise ValueError("delivery target thread_id is invalid")
    normalized = {
        "platform": platform,
        "chat_id": str(chat_id),
        "thread_id": str(thread_id) if thread_id is not None else None,
    }
    resolved_from = target.get("_resolved_from")
    if resolved_from is not None:
        if type(resolved_from) is not str or resolved_from not in {
            "origin", "origin_fallback", "home", "explicit",
        }:
            raise ValueError("delivery target provenance is invalid")
        normalized["_resolved_from"] = resolved_from
    return normalized


# Routing intent tokens — resolved at fire time, not create time, so a
# job created before Telegram was wired up will pick up Telegram once it
# comes online.  ``all`` expands into the set of connected platforms
# (those with a configured home chat_id) in _expand_routing_tokens.
_ROUTING_TOKENS = frozenset({"all"})

# Pseudo-platform for delivering job output INTO a profile's canonical
# "Bot Chat" session as a real inbound turn (the bot sees it, runs a turn,
# and can respond — Bot Mode's agent-to-agent lane, not a transcript
# mirror).  ``bot-chat`` targets the job's own profile; ``bot-chat:<name>``
# targets a named profile on THIS machine.  Deliberately excluded from the
# ``all`` routing token: ``all`` fans out to messaging home channels, and a
# bot-chat delivery costs a full agent turn.
BOT_CHAT_PLATFORM = "bot-chat"
BOT_CHAT_SELF_TARGET = "_self"


def parse_bot_chat_deliver_token(part: str) -> Optional[str]:
    """Return the target profile for a ``bot-chat[:<name>]`` deliver token.

    Returns ``""`` for the bare token (the job's own profile), the profile
    name for the explicit form, or ``None`` when ``part`` is not a bot-chat
    token at all.  Case-insensitive on the token; the profile name is
    normalized by the profile layer at resolve time.
    """
    raw = (part or "").strip()
    lowered = raw.lower()
    if lowered == BOT_CHAT_PLATFORM:
        return ""
    prefix = BOT_CHAT_PLATFORM + ":"
    if lowered.startswith(prefix):
        return raw[len(prefix):].strip()
    return None


def _resolve_bot_chat_target(job: dict, profile_arg: str) -> Optional[dict]:
    """Resolve a bot-chat deliver token to a concrete delivery target.

    ``profile_arg`` is ``""`` for the job's own profile (the HERMES_HOME
    this scheduler runs under — machine-local and self-referential, so no
    ``-p`` flag is needed at send time) or an explicit profile name that
    must exist in THIS machine's profile root.  Cross-machine delivery is
    intentionally unsupported: names resolve only against the local
    ``~/.hermes/profiles/`` tree, so same-named profiles on other gateways
    can never be targeted by accident.
    """
    if not profile_arg:
        # Own profile: the child inherits HERMES_HOME; the ledger still needs a
        # concrete non-empty requested-target identity.
        return {
            "platform": BOT_CHAT_PLATFORM,
            "chat_id": BOT_CHAT_SELF_TARGET,
            "thread_id": None,
        }
    try:
        from hermes_cli.profiles import normalize_profile_name, profile_exists

        canon = normalize_profile_name(profile_arg)
        if not profile_exists(canon):
            logger.warning(
                "Job '%s': bot-chat delivery profile '%s' not found on this "
                "machine — skipping target",
                job.get("id", "?"), profile_arg,
            )
            return None
        return {"platform": BOT_CHAT_PLATFORM, "chat_id": canon, "thread_id": None}
    except Exception:
        logger.warning(
            "Job '%s': failed to resolve bot-chat profile '%s'",
            job.get("id", "?"), profile_arg, exc_info=True,
        )
        return None


def _expand_routing_tokens(part: str) -> List[str]:
    """Expand a routing-intent token to concrete platform names.

    ``all`` expands to every platform in ``_iter_home_target_platforms()``
    that has a configured home chat_id right now.  Unknown / non-token
    values pass through unchanged as a single-element list, so the caller
    can treat every token uniformly.
    """
    token = part.lower()
    if token not in _ROUTING_TOKENS:
        return [part]
    expanded: List[str] = []
    for platform_name in _iter_home_target_platforms():
        if _get_home_target_chat_id(platform_name):
            expanded.append(platform_name)
    return expanded


def _delivery_lane_value(job: dict, *, for_failure: bool = False):
    """Raw deliver-lane value for a run outcome: the failure lane when
    ``for_failure`` and the job overrides it, else ``deliver``. Keeps
    delivery bookkeeping (outcome classification, unresolved-origin,
    incident 'alerted' marking) reading the SAME lane the notice was
    actually routed through (NS-788 review finding B1)."""
    if for_failure:
        failure_deliver = job.get("failure_deliver")
        if failure_deliver is not None and str(failure_deliver).strip():
            return failure_deliver
    return job.get("deliver", "local")


def _resolve_delivery_targets(job: dict, *, for_failure: bool = False) -> List[dict]:
    """Resolve all concrete auto-delivery targets for a cron job.

    Accepts the legacy comma-separated ``deliver`` string plus the
    ``all`` routing-intent token, which expands to every platform with
    a configured home channel.  Tokens may be combined with explicit
    targets: ``origin,all`` and ``all,telegram:-100:17`` both work.
    Duplicate (platform, chat_id, thread_id) tuples are collapsed by the
    existing dedup pass.

    ``for_failure=True`` resolves failure-category engine notices
    (failure summaries, interrupted-run notices, drift/preflight
    alerts): when the job carries a ``failure_deliver`` value, targets
    resolve from it INSTEAD of ``deliver`` — ``failure_deliver: local``
    is the structural opt-out for shared channels (NS-788, Coatue).
    Absent ``failure_deliver``, failure delivery follows ``deliver``
    exactly as before.
    """
    deliver_raw = _delivery_lane_value(job, for_failure=for_failure)
    deliver = _normalize_deliver_value(deliver_raw)
    if deliver == "local":
        return []

    seen = {}
    targets = []
    for raw in deliver.split(","):
        raw = raw.strip()
        if not raw:
            continue
        from_broadcast = raw.lower() in _ROUTING_TOKENS
        for part in _expand_routing_tokens(raw):
            target = _resolve_single_delivery_target(job, part, from_broadcast=from_broadcast)
            if not target:
                continue
            target = _normalize_delivery_target_identity(target)
            key = (target["platform"].lower(), target["chat_id"], target["thread_id"])
            kept = seen.get(key)
            if kept is None:
                seen[key] = target
                targets.append(target)
            elif (
                # Keep origin/origin_fallback/home provenance regardless of broadcast token order.
                _MIRROR_PROVENANCE_RANK.get(str(target.get("_resolved_from") or ""), 0)
                > _MIRROR_PROVENANCE_RANK.get(str(kept.get("_resolved_from") or ""), 0)
            ):
                kept["_resolved_from"] = target.get("_resolved_from")
    return targets


def _resolve_delivery_target(job: dict) -> Optional[dict]:
    """Resolve the concrete auto-delivery target for a cron job, if any."""
    targets = _resolve_delivery_targets(job)
    return targets[0] if targets else None


# Media extension sets — audio routing is centralized in gateway.platforms.base
# via should_send_media_as_audio() so Telegram-specific rules stay in one place.
_VIDEO_EXTS = frozenset({'.mp4', '.mov', '.avi', '.mkv', '.webm', '.3gp'})
_IMAGE_EXTS = frozenset({'.jpg', '.jpeg', '.png', '.webp', '.gif'})


def _send_media_via_adapter(
    adapter,
    chat_id: str,
    media_files: list,
    metadata: dict | None,
    loop,
    job: dict,
    platform=None,
    receipts_out: Optional[list] = None,
) -> list:
    """Send extracted MEDIA files as native platform attachments via a live adapter.

    Routes each file to the appropriate adapter method (send_voice, send_image_file,
    send_video, send_document) based on file extension — mirroring the routing logic
    in ``BasePlatformAdapter._process_message_background``.

    Returns a list of per-file error strings (empty when every attachment
    delivered). Callers surface these into the job's delivery errors so a
    dropped attachment is visible in ``last_error``/run status instead of
    only in the gateway log (the silent-drop half of the manual-run
    attachment bug: text delivered, file vanished, job marked ok).
    """
    from pathlib import Path

    from gateway.platforms.base import (
        BasePlatformAdapter,
        SendResult,
        should_send_media_as_audio,
    )

    errors: list = []
    requested = [(str(p), v) for p, v in (media_files or [])]
    media_files = BasePlatformAdapter.filter_media_delivery_paths(media_files)
    # Report paths the safety filter dropped: the model referenced them in
    # MEDIA: tags but they will never be sent (missing file, denied prefix,
    # or strict-mode policy miss).
    kept = {p for p, _ in media_files}
    for raw_path, _v in requested:
        try:
            from gateway.platforms.base import validate_media_delivery_path

            if validate_media_delivery_path(raw_path) not in kept:
                errors.append(
                    f"attachment dropped by media path policy: {raw_path}"
                )
        except Exception:
            errors.append(f"attachment dropped by media path policy: {raw_path}")

    for media_ordinal, (media_path, _is_voice) in enumerate(media_files):
        try:
            send_metadata = dict(metadata or {})
            send_metadata["_transport_receipt_component"] = "media"
            send_metadata["_transport_receipt_ordinal"] = media_ordinal
            ext = Path(media_path).suffix.lower()
            route_platform = platform if platform is not None else getattr(adapter, "platform", None)
            if should_send_media_as_audio(route_platform, ext, is_voice=_is_voice):
                coro = adapter.send_voice(chat_id=chat_id, audio_path=media_path, metadata=send_metadata)
            elif ext in _VIDEO_EXTS:
                coro = adapter.send_video(chat_id=chat_id, video_path=media_path, metadata=send_metadata)
            elif ext in _IMAGE_EXTS:
                coro = adapter.send_image_file(chat_id=chat_id, image_path=media_path, metadata=send_metadata)
            else:
                coro = adapter.send_document(chat_id=chat_id, file_path=media_path, metadata=send_metadata)

            from agent.async_utils import safe_schedule_threadsafe
            future = safe_schedule_threadsafe(coro, loop)
            if future is None:
                msg = f"cannot send media {media_path}: gateway loop unavailable"
                logger.warning("Job '%s': %s", job.get("id", "?"), msg)
                errors.append(msg)
                return errors
            try:
                # Large attachments (long TTS audio, concatenated recordings,
                # big exports) can legitimately exceed a fixed 30s upload
                # window. Configurable, matching the other cron timeouts
                # (cron.media_send_timeout_seconds in config.yaml, or the
                # HERMES_CRON_MEDIA_SEND_TIMEOUT env override).
                result = future.result(timeout=_get_media_send_timeout())
            except TimeoutError:
                future.cancel()
                raise
            receipt_bound = (
                type(metadata) is dict
                and "_transport_receipt_requested_target" in metadata
            )
            if type(result) is SendResult:
                result_success = result.success is True
                result_error = result.error
                if receipts_out is not None:
                    receipts_out.extend(result.receipts)
            else:
                legacy = _inert_legacy_send_result_fields(result)
                if receipt_bound or legacy is None:
                    errors.append("media adapter returned an invalid result")
                    return errors
                result_success = legacy["success"] is True
                result_error = legacy["error"]
            if not result_success:
                msg = (
                    f"media send failed for {media_path}: "
                    f"{result_error or 'unknown'}"
                )
                logger.warning("Job '%s': %s", job.get("id", "?"), msg)
                errors.append(msg)
        except Exception as e:
            # Argument-less exceptions (notably TimeoutError, the most likely
            # failure on this path) have an empty str(), which would render
            # the reason as nothing at all. Fall back to the class name.
            msg = (
                f"failed to send media {media_path}: {str(e) or type(e).__name__}"
            )
            logger.warning("Job '%s': %s", job.get("id", "?"), msg)
            errors.append(msg)
    return errors


def _confirm_adapter_delivery(send_result, job_id: str = "?", unverified: Optional[list] = None) -> bool:
    """Return True only if ``send_result`` unambiguously confirms delivery.

    A live adapter that returns ``None`` (e.g. a swallowed exception, a busy
    platform, or a code path that returns early without producing a
    ``SendResult``) must NOT be treated as success — doing so causes the
    scheduler to log ``"delivered to <chat> via live adapter"`` while the
    gateway never actually sees the message (#47056).

    Likewise, a result carrying no ``success`` at all (a partial mock, or a
    ``dict`` from a code path that never reached the adapter) is a contract
    violation: it does not actually tell us whether the send succeeded.
    Require an explicit, truthy ``success`` to count as confirmed.

    Both shapes are inspected the same way, because ``_deliver_to_platform``
    returns either a ``SendResult`` object or a plain ``dict``:

    * ``delivered is False`` is a REJECTION even when ``success`` is truthy.
      The silence-narration filter returns
      ``{"success": True, "delivered": False}`` — a successfully *dropped*
      message, not a delivered one.  Reading only ``success`` there is how a
      cron brief was logged as delivered while the user got nothing (#77763).
    * No ``message_id`` and no ``raw_response`` means we have no positive
      evidence of a send.  That is not proof of failure either (some adapters
      legitimately return a bare success), so it is still accepted — but
      logged at WARNING so an UNVERIFIED delivery is visible in the log
      instead of masquerading as a confirmed one.  Telegram ``SendResult``
      objects carry ``message_id``; the dict-filter shape does not.
    """
    from gateway.platforms.base import SendResult

    if type(send_result) is dict:
        if type(send_result.get("success")) is not bool:
            return False
        success = send_result["success"]
        delivered = send_result.get("delivered")
        message_id = send_result.get("message_id")
        raw_response = (
            send_result.get("raw_response")
            if type(send_result.get("raw_response")) is dict else None
        )
    elif type(send_result) is SendResult:
        if type(send_result.success) is not bool:
            return False
        success = send_result.success
        delivered = getattr(send_result, "delivered", None)
        message_id = send_result.message_id
        raw_response = send_result.raw_response
    else:
        legacy = _inert_legacy_send_result_fields(send_result)
        if legacy is None:
            return False
        success = legacy["success"]
        delivered = None
        message_id = legacy["message_id"]
        raw_response = legacy["raw_response"]
    if success is not True or delivered is False:
        return False
    if message_id is None and not raw_response:
        logger.warning(
            "Job '%s': live adapter reported success with no delivery evidence "
            "(no message_id, no raw_response) — treating as delivered but "
            "UNVERIFIED",
            job_id,
        )
        if unverified is not None:
            unverified.append(True)
    return True


def _inert_legacy_send_result_fields(send_result: Any) -> Optional[dict]:
    """Read the one supported inert legacy container without object magic."""
    if type(send_result) is not SimpleNamespace:
        return None
    fields = object.__getattribute__(send_result, "__dict__")
    if type(fields) is not dict or type(fields.get("success")) is not bool:
        return None
    error = fields.get("error")
    message_id = fields.get("message_id")
    if error is not None and type(error) is not str:
        error = None
    if message_id is not None and type(message_id) not in {str, int}:
        message_id = None
    return {
        "success": fields["success"],
        "error": error,
        "message_id": message_id,
        "raw_response": fields.get("raw_response") if type(fields.get("raw_response")) is dict else None,
    }


def _is_channel_dm_topic(
    runtime_adapter: Any,
    chat_id: Any,
    loop: Any,
    job_id: str,
) -> bool:
    """Decide whether an (already-ambiguous) Telegram topic target is a genuine
    Bot API *channel* Direct-Messages topic (route via
    ``direct_messages_topic_id``) rather than a forum-style topic in a private
    chat (route via ``message_thread_id``).

    Callers gate this on the ambiguous shape first
    (``telegram:<positive_chat_id>:<numeric_thread_id>``) — that shape is
    identical for both cases, so shape alone cannot decide (this was the #52060
    regression).  The real signal is the chat *type*: a genuine channel DM topic
    lives on a ``channel`` chat.  Probe the live adapter's ``get_chat_info`` once
    and only return True when the chat is a channel.

    Fails SAFE to ``message_thread_id`` (returns False) for adapters without a
    probe, or any probe error/timeout — that is the pre-#22773 behaviour and the
    correct default for the common forum-topic case.
    """
    # Resolve on the CLASS, not the instance (general pitfall #11): a MagicMock
    # instance auto-creates a truthy ``get_chat_info`` attribute, so an
    # instance-level probe would misclassify test doubles. Real adapters expose
    # the coroutine on the class regardless.
    get_chat_info = getattr(type(runtime_adapter), "get_chat_info", None)
    if not callable(get_chat_info):
        return False
    try:
        from agent.async_utils import safe_schedule_threadsafe

        future = safe_schedule_threadsafe(
            get_chat_info(runtime_adapter, str(chat_id)), loop,  # type: ignore[arg-type]
        )
        if future is None:
            return False
        # Lighter than a send (metadata-only Bot API call), so a shorter bound
        # than the 30s/60s send waits elsewhere in this file is intentional.
        info = future.result(timeout=10)
    except Exception:
        logger.debug(
            "Job '%s': get_chat_info probe failed for chat=%s — "
            "defaulting to message_thread_id routing",
            job_id, chat_id, exc_info=True,
        )
        return False
    is_channel = isinstance(info, dict) and str(info.get("type") or "").lower() == "channel"
    if is_channel:
        logger.info(
            "Job '%s': chat=%s is a channel — routing via direct_messages_topic_id",
            job_id, chat_id,
        )
    return is_channel


def _receipt_text_chunks_for_target(
    adapters: Any, platform_name: str, content: str, media_files=None,
) -> list[str]:
    """Return exact adapter-planned chunks when the adapter can prove them.

    Opaque and standalone transports deliberately get one logical component:
    a later multi-ack cannot be upgraded to delivery unless it binds that plan.
    Matrix and Telegram expose this preflight because their send paths own the
    deterministic formatting/splitting algorithm.
    """
    if not adapters:
        if platform_name.lower() == "telegram":
            from tools.send_message_senders import _plan_standalone_telegram_text

            return _plan_standalone_telegram_text(
                content, media_files=media_files,
            )[1]
        return [content]
    candidate = None
    try:
        from gateway.config import Platform
        candidate = adapters.get(Platform(platform_name.lower()))
    except Exception:
        candidate = None
    if candidate is None:
        try:
            candidate = adapters.get(platform_name) or adapters.get(platform_name.lower())
        except Exception:
            candidate = None
    if candidate is None:
        if platform_name.lower() == "telegram":
            from tools.send_message_senders import _plan_standalone_telegram_text

            return _plan_standalone_telegram_text(
                content, media_files=media_files,
            )[1]
        return [content]
    planner = getattr(candidate, "plan_transport_text", None)
    if not callable(planner):
        return [content]
    try:
        chunks = planner(content)
    except Exception as exc:
        raise ValueError("transport planner failed before dispatch") from exc
    if (
        type(chunks) not in {list, tuple}
        or len(chunks) == 0
        or not all(type(chunk) is str and chunk for chunk in chunks)
    ):
        raise ValueError("transport planner returned invalid chunks")
    return list(chunks)


def _persist_target_text_receipts(
    receipts: Any,
    attempts: dict,
    requested_target: dict[str, str],
    components: Optional[set[str]] = None,
    expected_actual_target: Optional[dict[str, str]] = None,
) -> bool:
    """Persist exact acknowledgements and prove the selected planned set.

    Matching partial acknowledgements are retained even when the final result
    is false. ``components=None`` requires every preregistered component;
    callers passing a set require only those component kinds.
    """
    from gateway.platforms.base import TransportReceipt, TransportTarget

    if type(receipts) is not tuple:
        return False
    if type(attempts) is not dict or type(requested_target) is not dict:
        return False
    if components is not None and type(components) is not set:
        return False
    if expected_actual_target is not None and type(expected_actual_target) is not dict:
        return False
    if not all(type(receipt) is TransportReceipt for receipt in receipts):
        return False
    if not attempts:
        return bool(receipts)
    expected = {
        key for key in attempts
        if key[:3] == (
            requested_target["platform"], requested_target["chat_id"],
            requested_target["thread_id"],
        ) and (components is None or key[3] in components)
    }
    observed = set()
    persisted_all = True
    expected_actual = expected_actual_target or requested_target
    try:
        planned_target = (
            expected_actual["platform"],
            expected_actual["chat_id"],
            expected_actual["thread_id"],
        )
    except (KeyError, TypeError):
        return False
    if not all(type(value) is str for value in planned_target):
        return False
    for receipt in receipts:
        try:
            requested = receipt.requested_target
            key = (
                requested.platform, requested.chat_id,
                requested.thread_id or "", receipt.component, receipt.ordinal,
            )
            attempt_id = attempts.get(key)
            persisted = bool(attempt_id) and record_transport_receipt(attempt_id, receipt)
        except Exception:
            persisted = False
            key = None
        actual = receipt.actual_target
        actual_target = (
            (actual.platform, actual.chat_id, actual.thread_id or "")
            if type(actual) is TransportTarget
            else None
        )
        if (
            persisted
            and key is not None
            and receipt.outcome == "delivered"
            and actual_target == planned_target
        ):
            observed.add(key)
        else:
            persisted_all = False
    return bool(expected) and persisted_all and observed == expected


def _receipt_delivery_outcome(execution_id: str) -> Optional[str]:
    """Project a transport outcome only when this execution has a receipt plan."""
    try:
        counts = receipt_summary(execution_id)
    except Exception:
        return None
    if counts.get("unknown", 0) > 0:
        return "unknown"
    if counts.get("failed", 0) > 0:
        return "failed"
    if counts.get("delivered", 0) > 0 and counts.get("targets_delivered", 0) > 0:
        return "delivered"
    return None


def _cron_delivery_notify_enabled(cfg: Optional[dict]) -> bool:
    """Resolve ``cron.delivery.notify`` (config.yaml). Default True.

    Only an explicit boolean ``False`` (or a YAML ``false``/``off`` that parses
    to it) disables the push notification; a missing/malformed section keeps
    the default so a typo can never silently make cron briefs silent.
    """
    try:
        cron_cfg = (cfg or {}).get("cron")
        if not isinstance(cron_cfg, dict):
            return True
        delivery_cfg = cron_cfg.get("delivery")
        if not isinstance(delivery_cfg, dict):
            return True
        return delivery_cfg.get("notify", True) is not False
    except Exception:
        return True


def _record_delivery_verification(job: dict, unverified_targets: list) -> None:
    """Persist the UNVERIFIED-delivery marker on the job record.

    ``last_delivery_unverified`` is a list of ``platform:chat_id`` targets
    whose live adapter acked the send with no message_id/raw_response, or
    ``None`` once a run delivered with positive evidence (or to no live
    target). Skips the write when nothing changed so the common verified
    path costs no jobs.json save. Never raises — status bookkeeping must not
    fail a delivery.
    """
    new_value = list(unverified_targets) or None
    queued = {target: receipt for target, receipt in
              job.get("_bot_chat_delivery_receipts", {}).items()
              if receipt["status"] in ("queued", "claimed")} or None
    values = {key: value for key, value in {
        "last_delivery_unverified": new_value, "last_delivery_queued": queued,
    }.items() if (job.get(key) or None) != value}
    if not values:
        return
    job.update(values)
    try:
        from cron.jobs import update_job

        update_job(job["id"], values)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug(
            "Job '%s': could not record delivery verification: %s", job.get("id"), exc,
        )


@dataclass
class _TargetDelivery:
    """Per-target delivery state shared by the live-adapter and standalone lanes."""

    job: dict
    platform: Any
    platform_name: str
    chat_id: str
    thread_id: Optional[str]
    transport: Any
    pconfig: Any
    runtime_adapter: Any
    target_adapters: Any
    config: Any
    loop: Any
    notify_delivery: bool
    origin: dict
    origin_target: bool
    origin_user_id: Optional[str]
    is_dm_target: bool
    mirror_text: str
    mirror_this_target: bool
    in_channel_surface: bool
    inchannel_continuable: bool
    opened_thread_id: Optional[str]
    live_adapter_ready: bool = False
    receipt_attempts: dict = None
    receipt_requested_target: dict = None

    @property
    def is_relay(self) -> bool:
        return self.transport is not None and self.transport.is_relay

    @property
    def where(self) -> str:
        return f"{self.platform_name}:{self.chat_id}"


def _note_target_error(job: dict, msg: str, errors: list) -> None:
    """Log a per-target delivery failure as a WARNING and record it in ``errors``."""
    logger.warning("Job '%s': %s", job["id"], msg)
    errors.append(msg)


def _warn_live_lane_failure(job: dict, msg: str, is_relay: bool) -> None:
    """Relay targets have no standalone fallback, so the log line must not promise one."""
    if is_relay:
        logger.warning("Job '%s': %s", job["id"], msg)
    else:
        logger.warning("Job '%s': %s, falling back to standalone", job["id"], msg)


def _resolve_target_transport(
    job: dict, platform, platform_name: str, target: dict, adapters, config):
    """Resolve ``(transport, pconfig, runtime_adapter, target_adapters)`` for one target, or
    ``(None, error)`` when it cannot be served (relay-fronted with no live transport, or not
    configured/enabled)."""
    from gateway.delivery import resolve_delivery_transport
    target_adapters = adapters
    if isinstance(adapters, _preflight.SharedRouteAdapters):
        # Credentialless satellite: the primary adapter serves THIS target only when an exact
        # primary route maps it to this profile; a miss fails closed below.
        # See #101113.
        shared = adapters.get(platform, target)
        target_adapters = {platform: shared} if shared is not None else {}
    transport = resolve_delivery_transport(platform, config, target_adapters)
    if transport is not None:
        pconfig = transport.config
        runtime_adapter = transport.adapter
    else:
        # Relay-fronted platforms have NO standalone fallback (the connector owns the credential),
        # so surface that instead of the native configured/enabled gate, which misdiagnoses them.
        from gateway.relay import relay_fronted_platforms
        if platform_name in relay_fronted_platforms():
            return None, (
                f"platform '{platform_name}' is relay-fronted and has no "
                "live gateway transport; start the gateway (its ticker "
                "owns relay-fronted delivery and will fire the job on "
                "schedule)"
            )
        pconfig = config.platforms.get(platform)
        runtime_adapter = None

    if transport is not None and transport.is_relay:
        # Relay transport carries the RELAY adapter's config (enablement already checked). The
        # logical platform is deliberately NOT natively enabled, so the native gate must not apply.
        if pconfig is None:
            from gateway.config import PlatformConfig
            pconfig = PlatformConfig(enabled=True)
    elif not pconfig or not pconfig.enabled:
        return None, f"platform '{platform_name}' not configured/enabled"
    return (transport, pconfig, runtime_adapter, target_adapters), None


def _inchannel_surface_supported(runtime_adapter, platform_name: str) -> bool:
    """D6 probe: can this adapter deliver a continuable in_channel brief on ``platform_name``?
    Per-platform check first (one RelayAdapter fronts N platforms; the scalar attr only carries
    the PRIMARY identity's bit); native adapters use the class attribute."""
    per_platform_check = getattr(
        runtime_adapter, "supports_inchannel_continuable_for_platform", None)
    if callable(per_platform_check):
        try:
            return bool(per_platform_check(platform_name))
        except Exception:
            return False
    return bool(getattr(runtime_adapter, "supports_inchannel_continuable", False))


def _live_route_metadata(t: _TargetDelivery) -> tuple[Optional[str], dict, dict]:
    """Compute ``(route_thread_id, route_metadata, media_metadata)`` for a live send, ONCE so text
    and media agree. ``telegram:<positive_chat_id>:<numeric_thread_id>`` is ambiguous (private
    forum topic vs channel DM topic need OPPOSITE routing) — see ``_is_channel_dm_topic``.
    ``thread_id`` rides in ``route_metadata`` to bypass the router's private-chat anchor rule."""
    from gateway.config import Platform
    from gateway.delivery import _looks_like_int, looks_like_telegram_private_chat_id
    job = t.job
    thread_id = t.thread_id
    is_ambiguous_telegram_topic = (
        t.platform == Platform.TELEGRAM
        and thread_id is not None
        and looks_like_telegram_private_chat_id(str(t.chat_id))
        and _looks_like_int(str(thread_id))
    )
    if is_ambiguous_telegram_topic and _is_channel_dm_topic(
        t.runtime_adapter, t.chat_id, t.loop, job["id"]):
        # Channel DM topic: direct_messages_topic_id, no bare thread_id; media mirrors text.
        # See #22773.
        route_thread_id = None
        route_metadata = {
            "direct_messages_topic_id": str(thread_id), "job_id": job["id"],
            "notify": t.notify_delivery,
        }
        media_metadata = {"direct_messages_topic_id": str(thread_id), "notify": t.notify_delivery}
    else:
        # Forum-style topic or non-topic target: message_thread_id.
        # Put thread_id in *route_metadata* (not just the DeliveryTarget) deliberately — the
        # DeliveryRouter's private-chat topic detection (gateway/delivery.py) demands a reply anchor when
        # thread_id is absent from metadata; cron deliveries have no inbound reply anchor, so the metadata
        # key bypasses that check and lets the adapter route via a plain message_thread_id. See #52060.
        route_thread_id = str(thread_id) if thread_id is not None else None
        route_metadata = {"job_id": job["id"], "notify": t.notify_delivery}
        if route_thread_id:
            route_metadata["thread_id"] = route_thread_id
        media_metadata = {"notify": t.notify_delivery}
        if thread_id:
            media_metadata["thread_id"] = thread_id

    # Relay egress needs metadata.scope_id (fail-closed tenant guard; scope cache is COLD after a
    # restart; router stamps HOME only). Origin targets only: a wrong fan-out scope is worse than
    # none.
    if t.origin_target and t.origin.get("scope_id"):
        route_metadata.setdefault("scope_id", str(t.origin["scope_id"]))
        media_metadata.setdefault("scope_id", str(t.origin["scope_id"]))
    if t.receipt_requested_target is not None:
        route_metadata["_transport_receipt_requested_target"] = t.receipt_requested_target
    return route_thread_id, route_metadata, media_metadata


def _receipt_component_planned(t: _TargetDelivery, component: str) -> bool:
    return bool(t.receipt_attempts) and any(key[3] == component for key in t.receipt_attempts)


def _routed_actual_target(t: _TargetDelivery, route_thread_id: Optional[str], metadata: dict) -> dict:
    return {"platform": t.platform_name, "chat_id": t.chat_id, "thread_id": (
        str(metadata["direct_messages_topic_id"])
        if metadata.get("direct_messages_topic_id") is not None else route_thread_id or ""
    )}


def _live_send_text(
    t: _TargetDelivery, text_to_send: str, route_thread_id: Optional[str], route_metadata: dict, *,
    target_errors: list, delivery_errors: list, unverified_targets: list,
) -> tuple[bool, bool, Any]:
    """Send text and persist its typed acknowledgements before any side effect.

    The second result is ``uncertain``: once the live lane may have crossed the
    provider boundary, same-identity fallback, mirroring, and seeding are unsafe.
    """
    from agent.async_utils import safe_schedule_threadsafe
    from gateway.delivery import DeliveryRouter, DeliveryTarget
    from gateway.platforms.base import SendResult

    router = DeliveryRouter(t.config, t.target_adapters)
    route_target = DeliveryTarget(
        platform=t.platform, chat_id=str(t.chat_id), thread_id=route_thread_id, is_explicit=True)
    future = safe_schedule_threadsafe(
        router._deliver_to_platform(route_target, text_to_send, route_metadata), t.loop)
    if future is None:
        target_errors.append("live adapter event loop scheduling failed")
        return False, False, None
    send_result = None
    try:
        send_result = future.result(timeout=60)
    except TimeoutError:
        future.cancel()
        msg = f"live adapter confirmation timed out for {t.where}; delivery is unknown"
        target_errors.append(msg)
        logger.warning("Job '%s': %s", t.job["id"], msg)
        return False, True, None
    except Exception as exc:
        target_errors.append(f"live adapter send failed: {exc}")
        partial = getattr(exc, "send_result", None)
        if type(partial) is SendResult:
            send_result = partial
        else:
            return False, True, None

    receipts = send_result.receipts if type(send_result) is SendResult else ()
    message_id = None
    raw_response = None
    legacy_fields = None
    if type(send_result) is dict:
        raw_response = send_result.get("raw_response") if type(send_result.get("raw_response")) is dict else None
        value = send_result.get("message_id")
        message_id = value if type(value) in {str, int} else None
    elif type(send_result) is SendResult:
        raw_response, message_id = send_result.raw_response, send_result.message_id
    else:
        legacy_fields = _inert_legacy_send_result_fields(send_result)
        if legacy_fields is not None:
            raw_response, message_id = legacy_fields["raw_response"], legacy_fields["message_id"]

    evidence_gap: list = []
    confirmed = _confirm_adapter_delivery(send_result, t.job["id"], evidence_gap)
    if confirmed and evidence_gap:
        unverified_targets.append(t.where)

    # Persist partial receipts even when the adapter reports failure. Their plan
    # entries remain unknown unless every planned component is acknowledged.
    receipt_ok = True
    if _receipt_component_planned(t, "text"):
        receipt_ok = _persist_target_text_receipts(
            receipts, t.receipt_attempts, t.receipt_requested_target, components={"text"},
            expected_actual_target=_routed_actual_target(t, route_thread_id, route_metadata))

    if not confirmed:
        if type(send_result) is dict:
            err = send_result.get("error") or send_result.get("filtered") or "unknown"
            shape = "dict"
        elif type(send_result) is SendResult:
            err, shape = send_result.error, "SendResult"
        elif legacy_fields is not None:
            err, shape = legacy_fields["error"] or "unknown", "legacy"
        elif send_result is None:
            err, shape = "no response from adapter", "None"
        else:
            err, shape = "invalid adapter result", "invalid"
        msg = f"live adapter send to {t.where} returned unconfirmed result ({shape}, error={err})"
        _warn_live_lane_failure(t.job, msg, t.is_relay)
        target_errors.append(msg)
        return False, True, None
    if not receipt_ok:
        target_errors.append(
            f"live adapter acknowledgement for {t.where} could not be persisted; delivery is unknown")
        return False, True, None
    if not receipts and (_receipt_component_planned(t, "text") or type(send_result) is SendResult):
        msg = f"live adapter send to {t.where} returned legacy success without typed receipt; delivery is unknown"
        target_errors.append(msg)
        logger.warning("Job '%s': %s", t.job["id"], msg)
        return False, True, None
    if raw_response and t.thread_id and raw_response.get("thread_fallback"):
        requested_thread_id = raw_response.get("requested_thread_id") or t.thread_id
        _note_target_error(
            t.job, f"configured thread_id {requested_thread_id} for {t.where} was not found; "
            "delivered without thread_id", delivery_errors)
    return True, False, message_id

def _live_send_media(
    t: _TargetDelivery, media_metadata: dict, media_files: list, delivery_errors: list,
    route_thread_id: Optional[str],
) -> bool:
    """Send media and require all planned typed acknowledgements before success."""
    routed_metadata = dict(media_metadata or {})
    if t.receipt_requested_target is not None:
        routed_metadata["_transport_receipt_requested_target"] = t.receipt_requested_target
    if t.is_relay:
        routed_metadata["_relay_logical_platform"] = t.platform.value
        logical_home = t.config.get_home_channel(t.platform)
        if logical_home is not None and logical_home.chat_id == t.chat_id:
            if logical_home.user_id:
                routed_metadata["user_id"] = logical_home.user_id
            if logical_home.scope_id:
                routed_metadata["scope_id"] = logical_home.scope_id
    receipts: list = []
    media_errors = _send_media_via_adapter(
        t.runtime_adapter, t.chat_id, media_files, routed_metadata or None, t.loop, t.job,
        platform=t.platform, receipts_out=receipts)
    for error in media_errors:
        delivery_errors.append(f"{error} (target {t.where})")
    # Even an unbound live media send needs a provider acknowledgement before
    # we claim the attachment arrived. With no durable plan the persistence
    # helper deliberately reduces this to ``bool(receipts)``; with a plan it
    # additionally requires every media component to be committed.
    receipt_ok = _persist_target_text_receipts(
        tuple(receipts), t.receipt_attempts or {}, t.receipt_requested_target,
        components={"media"},
        expected_actual_target=_routed_actual_target(t, route_thread_id, routed_metadata))
    if not receipt_ok:
        delivery_errors.append(f"media acknowledgement for {t.where} is unavailable; delivery is partial")
    return not media_errors and receipt_ok

def _seed_live_delivery_sessions(t: _TargetDelivery, delivered_message_id) -> None:
    """After a confirmed live send, seed continuation session(s) and run the generic mirror.
    Thread seeding is deferred here so open-succeeds/deliver-fails never seeds an unseen brief."""
    job = t.job
    origin = t.origin
    seed_kwargs = dict(
        chat_name=origin.get("chat_name"), is_dm=t.is_dm_target, scope_id=origin.get("scope_id"))
    thread_seeded = False
    inchannel_seeded = False
    if t.opened_thread_id:
        _seed_cron_thread_session(
            job, t.runtime_adapter, t.platform_name, t.chat_id, t.opened_thread_id, t.mirror_text,
            **seed_kwargs,
        )
        thread_seeded = True
    # in_channel: CREATE + seed the flat session (the mirror only APPENDS to an existing one). Same
    # `inchannel_continuable` gate as the flatten in _deliver_result (must not drift). Origin
    # seed without mirror opt-in; others only via _inchannel_seed_allowed (user-less seed = orphan).
    if t.in_channel_surface and t.inchannel_continuable and not thread_seeded:
        inchannel_seeded = _seed_cron_channel_session(
            job, t.runtime_adapter, t.platform_name, t.chat_id, t.mirror_text,
            user_id=t.origin_user_id, **seed_kwargs)
        if not inchannel_seeded:
            logger.warning(
                "Job '%s': in_channel seed did NOT land on %s:%s "
                "— a plain reply will not see this brief",
                job["id"], t.platform_name, t.chat_id)
        # Companion THREAD seed: a reply in the brief's own thread keys to (chat, thread=<ts>),
        # which the flat seed never touches. Seed it too so BOTH reply surfaces continue the job.
        if delivered_message_id:
            _seed_cron_thread_session(
                job, t.runtime_adapter, t.platform_name, t.chat_id, str(delivered_message_id),
                t.mirror_text,
                **seed_kwargs)
    elif t.in_channel_surface and not t.inchannel_continuable:
        logger.warning(
            "Job '%s': in_channel delivery to %s:%s is not a "
            "continuable target (origin=%s:%s thread=%s; not the "
            "origin conversation, and not a mirror-eligible "
            "fallback/opted-in target the seed can key) — seed "
            "skipped; the plain mirror below may still apply",
            job["id"], t.platform_name, t.chat_id,
            origin.get("platform"), origin.get("chat_id"), origin.get("thread_id"))
    _maybe_mirror_cron_delivery(
        job, t.platform_name, t.chat_id, t.mirror_text, thread_id=t.thread_id,
        user_id=t.origin_user_id,
        enabled=t.mirror_this_target and not thread_seeded and not inchannel_seeded)


def _deliver_via_live_adapter(
    t: _TargetDelivery, cleaned_text: str, media_files: list, *, target_errors: list,
    delivery_errors: list, unverified_targets: list,
) -> tuple[bool, bool]:
    """Deliver one live target: ``(delivered, uncertain)``.

    ``uncertain`` prohibits standalone retry and session side effects because a
    provider write may already exist without a complete durable receipt set.
    """
    route_thread_id, route_metadata, media_metadata = _live_route_metadata(t)
    try:
        text = cleaned_text.strip()
        if not text and not media_files:
            _note_target_error(t.job, f"live adapter send skipped (empty text and no media) for {t.where}", target_errors)
            return False, False
        if text:
            delivered, uncertain, message_id = _live_send_text(
                t, text, route_thread_id, route_metadata, target_errors=target_errors,
                delivery_errors=delivery_errors, unverified_targets=unverified_targets)
            if uncertain or not delivered:
                return False, uncertain
        else:
            message_id = None
        if media_files and not _live_send_media(
            t, media_metadata, media_files, delivery_errors, route_thread_id):
            return False, True
        logger.info("Job '%s': delivered to %s:%s via live adapter thread=%s message_id=%s",
                    t.job["id"], t.platform_name, t.chat_id,
                    route_thread_id if route_thread_id is not None else "-",
                    message_id if message_id is not None else "-")
        _seed_live_delivery_sessions(t, message_id)
        return True, False
    except Exception as exc:
        msg = f"live adapter delivery to {t.where} failed: {exc}"
        if not any(msg in item for item in target_errors):
            target_errors.append(msg)
        _warn_live_lane_failure(t.job, msg, t.is_relay)
        return False, True

def _standalone_send(
    t: _TargetDelivery, content: str, media_files: list) -> tuple[Any, Optional[str]]:
    """Run the standalone sender for one target without leaking a running loop."""
    from tools.send_message_tool import _send_to_platform
    shutdown_msg = f"delivery to {t.where} skipped — interpreter is shutting down"
    def send():
        return _send_to_platform(t.platform, t.pconfig, t.chat_id, content, thread_id=t.thread_id,
                                 media_files=media_files, receipt_bound=bool(t.receipt_attempts))
    def warned(message):
        logger.warning("Job '%s': %s", t.job["id"], message)
        return None, message
    def failed(exc):
        message = f"delivery to {t.where} failed: {exc}"
        logger.error("Job '%s': %s", t.job["id"], message, exc_info=True)
        return None, message
    if _sched._interpreter_shutting_down():
        return warned(shutdown_msg)
    if not content.strip() and not media_files:
        return warned(f"standalone send skipped (empty text and no media) for {t.where}")
    coro = send()
    try:
        return asyncio.run(coro), None
    except RuntimeError as exc:
        coro.close()
        if _sched._interpreter_shutting_down(exc):
            return warned(shutdown_msg)
        try:
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            try:
                # Create the coroutine inside the worker. Passing ``send()``
                # into submit leaks an unawaited coroutine when executor
                # scheduling itself fails, and is unsafe after a loop error.
                def run_in_worker():
                    return asyncio.run(send())
                return pool.submit(contextvars.copy_context().run, run_in_worker).result(timeout=30), None
            finally:
                pool.shutdown(wait=False)
        except Exception as thread_exc:
            return warned(shutdown_msg) if _sched._interpreter_shutting_down(thread_exc) else failed(thread_exc)
    except Exception as exc:
        return failed(exc)

def _deliver_standalone(
    t: _TargetDelivery, content: str, media_files: list, target_errors: list, delivery_errors: list,
) -> bool:
    """Standalone fallback only for a target whose live lane was known not dispatched."""
    if t.is_relay:
        if not target_errors:
            target_errors.append(f"relay delivery to {t.where} failed")
        delivery_errors.extend(target_errors)
        return False
    result, error = _standalone_send(t, content, media_files)
    if error is None and result and result.get("error"):
        error = f"delivery error: {result['error']} (target {t.where})"
        logger.error("Job '%s': %s", t.job["id"], error)
    if error is not None:
        target_errors.append(error)
        delivery_errors.extend(target_errors)
        return False
    if t.receipt_attempts and not _persist_target_text_receipts(
        result.get("receipts", ()) if isinstance(result, dict) else (), t.receipt_attempts,
        t.receipt_requested_target):
        msg = (f"media acknowledgement for {t.where} is unavailable; delivery is partial"
               if media_files else f"standalone send to {t.where} returned without a complete typed receipt; delivery is unknown")
        target_errors.append(msg)
        delivery_errors.extend(target_errors)
        return False
    for warning in (result.get("warnings") if isinstance(result, dict) else None) or []:
        msg = f"delivery warning: {warning} (target {t.where})"
        logger.error("Job '%s': %s", t.job["id"], msg)
        delivery_errors.append(msg)
    logger.info("Job '%s': delivered to %s:%s", t.job["id"], t.platform_name, t.chat_id)
    _maybe_mirror_cron_delivery(t.job, t.platform_name, t.chat_id, t.mirror_text, thread_id=t.thread_id,
                                user_id=t.origin_user_id, enabled=t.mirror_this_target)
    return True

def _prepare_target_delivery(
    job: dict, target: dict, *, adapters, loop, config, notify_delivery: bool, mirror_enabled: bool,
    mirror_text: str, delivery_errors: list, receipt_attempts: dict,
) -> Optional[_TargetDelivery]:
    """Per-target prologue of ``_deliver_result``: origin/mirror/in_channel gates, transport
    resolution, continuable-thread open. None (error noted in ``delivery_errors``) if unservable."""
    from gateway.config import Platform
    platform_name = target["platform"]
    chat_id = target["chat_id"]
    thread_id = target.get("thread_id")

    origin = _resolve_origin(job) or {}
    origin_thread = origin.get("thread_id")
    if origin_thread and not thread_id:
        logger.warning(
            "Job '%s': origin has thread_id=%s but delivery target lost it (deliver=%s, target=%s)",
            job["id"], origin_thread, job.get("deliver", "local"), target)
    elif thread_id:
        logger.debug(
            "Job '%s': delivering to %s:%s thread_id=%s",
            job["id"], platform_name, chat_id, thread_id)

    # Mirror: origin, origin-less home fallback, user-written home, or explicit-target opt-in.
    origin_target = _target_matches_origin(origin, platform_name, chat_id, thread_id)
    mirror_this_target = mirror_enabled and _target_mirror_eligible(
        job, target, global_mirror=mirror_enabled, origin_match=origin_target)
    # Resolved for ANY origin match (not just mirror-enabled): the in_channel seed needs it too.
    origin_user_id = origin.get("user_id") if origin_target else None

    # DM shape for BOTH the flatten gate and seed chat_type (Slack DM ids start with "D").
    origin_chat_type = str(origin.get("chat_type") or "").lower()
    is_dm_target = origin_chat_type == "dm" or (
        not origin_chat_type and str(chat_id).startswith("D"))

    # in_channel gate shared by thread-flatten and flat seed — they MUST match or brief and
    # session land in different places. Origin qualifies unconditionally; others only when the
    # seed can create a resolvable session (_inchannel_seed_allowed).
    inchannel_continuable = origin_target or (
        mirror_this_target and _inchannel_seed_allowed(is_dm=is_dm_target, user_id=origin_user_id))

    # Plugin platform names create dynamic members via Platform._missing_().
    try:
        platform = Platform(platform_name.lower())
    except (ValueError, KeyError):
        _note_target_error(job, f"unknown platform '{platform_name}'", delivery_errors)
        return None

    resolved, resolve_err = _resolve_target_transport(
        job, platform, platform_name, target, adapters, config)
    if resolved is None:
        _note_target_error(job, resolve_err, delivery_errors)
        return None
    transport, pconfig, runtime_adapter, target_adapters = resolved

    # Live send needs a RUNNING loop, not just an adapter. Computed ONCE so the in_channel
    # thread_id clear below stays in lockstep with the seed (standalone cannot seed flat).
    live_adapter_ready = (
        runtime_adapter is not None
        and loop is not None
        and getattr(loop, "is_running", lambda: False)()
    )

    # Continuable surface (D1/D2/D6) from platform config ``extra``; default "thread".
    # ``in_channel`` delivers FLAT so a plain channel reply continues via the shared session
    # ``(platform, chat_id, None)``. Unsupported adapters fail SAFE to thread.
    in_channel_surface = _resolve_cron_surface_mode(pconfig, platform_name) == "in_channel"
    if (
        in_channel_surface
        and runtime_adapter is not None
        and not _inchannel_surface_supported(runtime_adapter, platform_name)
    ):
        logger.debug(
            "Job '%s': cron_continuable_surface=in_channel not supported on %s, using thread",
            job.get("id", "?"), platform_name)
        in_channel_surface = False
    if in_channel_surface and inchannel_continuable and live_adapter_ready:
        # Force flat (D2): an inherited thread_id would never match the flat seed (None). Gated
        # on `inchannel_continuable` (SAME gate as the seed) AND `live_adapter_ready` (fallback
        # never seeds). Stay AFTER mirror_this_target/origin_user_id (need ORIGINAL thread_id).
        thread_id = None

    # Thread-preferred continuable cron: open a DEDICATED thread; its session is seeded after a
    # successful send. DM-only platforms return None → mirror the origin DM. in_channel SKIPS
    # this: it posts flat and _seed_cron_channel_session CREATES the session.
    opened_thread_id: Optional[str] = None
    if (
        mirror_this_target
        and not in_channel_surface
        and runtime_adapter is not None
        and loop is not None
        and not thread_id  # never override an explicit origin thread/topic
    ):
        opened_thread_id = _open_continuable_cron_thread(
            job, runtime_adapter, chat_id, loop) or None
        if opened_thread_id:
            thread_id = opened_thread_id
    return _TargetDelivery(
        job=job, platform=platform, platform_name=platform_name, chat_id=chat_id,
        thread_id=thread_id, transport=transport, pconfig=pconfig, runtime_adapter=runtime_adapter,
        target_adapters=target_adapters, config=config, loop=loop, notify_delivery=notify_delivery,
        origin=origin, origin_target=origin_target, origin_user_id=origin_user_id,
        is_dm_target=is_dm_target, mirror_text=mirror_text, mirror_this_target=mirror_this_target,
        in_channel_surface=in_channel_surface, inchannel_continuable=inchannel_continuable,
        opened_thread_id=opened_thread_id, live_adapter_ready=live_adapter_ready,
        receipt_attempts=receipt_attempts, receipt_requested_target={
            "platform": platform_name, "chat_id": chat_id, "thread_id": target.get("thread_id") or ""})


def _unresolved_delivery_outcome(job: dict, for_failure: bool) -> Optional[str]:
    """``_deliver_result`` outcome when no target resolved: None (not a failure) for ``local`` and
    origin-less ``origin`` (CLI jobs never capture an origin — a spurious error every run), else
    an error string."""
    deliver_value = _normalize_deliver_value(_delivery_lane_value(job, for_failure=for_failure))
    if deliver_value == "local":
        return None
    if deliver_value == "origin":
        logger.info(
            # deliver=origin with no resolvable origin and no configured home channels: treat as local
            # rather than reporting an error. CLI-created jobs never capture a {platform, chat_id} origin,
            # so failing here would make every CLI `deliver=origin` (or auto-detect) job emit a spurious "no
            # delivery target resolved" error on every run (#43014). The output is still persisted in
            # last_output for `cron list`/resume.
            "Job '%s': deliver=origin but no origin or home channels — "
            "skipping delivery (output saved in last_output)",
            job.get("name", job.get("id", "?")))
        return None
    msg = f"no delivery target resolved for deliver={deliver_value}"
    logger.warning("Job '%s': %s", job["id"], msg)
    return msg



def _preregister_delivery_receipts(job, targets, content, cleaned_content, media_files, adapters, loop,
                                    execution_id, fire_identity):
    """Durably register all target components before any transport dispatch."""
    if execution_id is None:
        return {}, None
    if type(execution_id) is not str or not execution_id or type(fire_identity) is not str or not fire_identity:
        return {}, "delivery receipt identity is invalid; no delivery was sent"
    planning_adapters = adapters if adapters is not None and loop is not None and getattr(loop, "is_running", lambda: False)() else None
    plan = []
    for target in targets:
        identity = dict(target)
        identity["thread_id"] = identity.get("thread_id") or ""
        if identity["platform"] == BOT_CHAT_PLATFORM:
            plan.append({"target": identity, "component": "text", "ordinal": 0,
                         "content": _bot_chat_query_message(job, content)})
            continue
        if cleaned_content.strip():
            try:
                chunks = _receipt_text_chunks_for_target(planning_adapters, identity["platform"],
                                                         cleaned_content.strip(), media_files=media_files)
            except (TypeError, ValueError):
                return {}, "delivery receipt planner is invalid; no delivery was sent"
            plan.extend({"target": identity, "component": "text", "ordinal": ordinal, "content": chunk}
                        for ordinal, chunk in enumerate(chunks))
        for ordinal, (media_path, _voice) in enumerate(media_files):
            if type(media_path) is not str:
                return {}, "delivery media identity is invalid; no delivery was sent"
            plan.append({"target": identity, "component": "media", "ordinal": ordinal, "content": media_path})
    if not plan:
        return {}, None
    try:
        attempts = preregister_receipt_plan(execution_id, fire_identity=fire_identity, components=plan)
    except Exception:
        logger.warning("Job '%s': receipt-plan preregistration failed", job["id"])
        return {}, "delivery receipt plan could not be persisted; no delivery was sent"
    return {(a["platform"], a["chat_id"], a["thread_id"], a["component"], a["ordinal"]): a["id"] for a in attempts}, None


def _deliver_bot_chat_target(job, target, content, attempts, delivery_errors):
    """Deliver bot-chat and persist its only honest outcome (failure or unknown)."""
    from gateway.platforms.base import TransportReceipt, TransportTarget
    chat_id = target["chat_id"]
    profile = "" if chat_id == BOT_CHAT_SELF_TARGET else chat_id
    error = _deliver_to_bot_chat(job, content, profile)
    bot_receipt = job.get("_bot_chat_delivery_receipts", {}).get(f"bot-chat:{profile or '(own)'}")
    queued = bool(bot_receipt and bot_receipt["status"] in ("queued", "claimed"))
    if not attempts:
        if error and not queued:
            delivery_errors.append(error)
        return
    requested = TransportTarget(BOT_CHAT_PLATFORM, chat_id, target.get("thread_id"))
    receipt = TransportReceipt(outcome="failed" if error == "bot-chat delivery failed" else "unknown",
        requested_target=requested, failure_kind="pre_dispatch" if error == "bot-chat delivery failed" else None,
        component="text", ordinal=0)
    if receipt.outcome == "unknown":
        attempt_id = attempts.get((BOT_CHAT_PLATFORM, chat_id, target.get("thread_id") or "", "text", 0))
        persisted = bool(attempt_id) and observe_transport_unknown(attempt_id, receipt)
    else:
        persisted = _persist_target_text_receipts((receipt,), attempts,
            {"platform": BOT_CHAT_PLATFORM, "chat_id": chat_id, "thread_id": target.get("thread_id") or ""},
            components={"text"})
    if not persisted:
        delivery_errors.append("bot-chat delivery receipt could not be persisted; delivery is unknown")
    elif queued:
        # Durable admission is visible separately; it cannot upgrade a
        # provider receipt or authorize an automatic resend.
        return
    elif receipt.outcome == "unknown":
        delivery_errors.append("bot-chat delivery confirmation unavailable")
    else:
        delivery_errors.append("bot-chat delivery failed")


def _deliver_result(
    job: dict, content: str, adapters=None, loop=None, *, execution_id: Optional[str] = None,
    fire_identity: Optional[str] = None, for_failure: bool = False,
) -> Optional[str]:
    """Orchestrate target planning and the independent live/standalone lanes."""
    if type(job) is not dict or type(content) is not str:
        return "delivery input is invalid; no delivery was sent"
    job.pop("_bot_chat_delivery_receipts", None)
    try:
        targets = _resolve_delivery_targets(job, for_failure=for_failure)
    except (TypeError, ValueError):
        return "delivery target is invalid; no delivery was sent"
    if not targets:
        _record_delivery_verification(job, [])
        return _unresolved_delivery_outcome(job, for_failure)
    external_execution = os.environ.get("_HERMES_CRON_EXTERNAL_WORKER", "")
    if (external_execution and adapters is None and external_execution == str(job.get("execution_id") or "")
            and any(target["platform"] != BOT_CHAT_PLATFORM for target in targets)):
        from cron.delivery_queue import enqueue_and_wait
        _record_delivery_verification(job, [])
        error = enqueue_and_wait(external_execution, job, content, for_failure=for_failure)
        from cron.jobs import get_job
        refreshed = get_job(job["id"]) or {}
        job["last_delivery_queued"] = refreshed.get("last_delivery_queued")
        return error
    from gateway.config import load_gateway_config
    from gateway.platforms.base import BasePlatformAdapter
    wrap_response, user_cfg = True, None
    with contextlib.suppress(Exception):
        user_cfg = load_config()
        wrap_response = user_cfg.get("cron", {}).get("wrap_response", True)
    delivery_content = content
    if wrap_response:
        name = job.get("name", job["id"])
        delivery_content = (f"Cronjob Response: {name}\n(job_id: {job.get('id', '')})\n-------------\n\n"
                            f"{content}\n\nTo stop or manage this job, send me a new message "
                            f"(e.g. \"stop reminder {name}\").")
    from gateway.media_policy import apply_media_policy_env
    apply_media_policy_env(user_cfg)
    media_files, cleaned_content = BasePlatformAdapter.extract_media(delivery_content)
    requested_media = len(media_files)
    media_files = BasePlatformAdapter.filter_media_delivery_paths(media_files)
    policy_errors = ([f"{requested_media - len(media_files)} media attachment(s) dropped by media path policy "
                      "(missing file, denied prefix, or strict-mode miss); see gateway.strict / "
                      "media_delivery_allow_dirs in config.yaml"] if requested_media > len(media_files) else [])
    if execution_id is None:
        execution_id = job.get("execution_id")
    if fire_identity is None:
        fire_identity = job.get("fire_identity") or execution_id
    attempts, plan_error = _preregister_delivery_receipts(job, targets, content, cleaned_content, media_files,
                                                           adapters, loop, execution_id, fire_identity)
    if plan_error:
        return plan_error
    try:
        config = load_gateway_config()
    except Exception as exc:
        msg = f"failed to load gateway config: {exc}"
        logger.error("Job '%s': %s", job["id"], msg)
        return msg
    try:
        mirror_enabled = _cron_mirror_delivery_enabled(job, user_cfg)
    except Exception:
        mirror_enabled = False
    _, mirror_text = BasePlatformAdapter.extract_media(content)
    mirror_text = (mirror_text or "").strip()
    errors, unverified = [], []
    notify = _cron_delivery_notify_enabled(user_cfg)
    for target in targets:
        if target["platform"] == BOT_CHAT_PLATFORM:
            _deliver_bot_chat_target(job, target, content, attempts, errors)
            continue
        t = _prepare_target_delivery(job, target, adapters=adapters, loop=loop, config=config,
            notify_delivery=notify, mirror_enabled=mirror_enabled, mirror_text=mirror_text,
            delivery_errors=errors, receipt_attempts=attempts)
        if t is None:
            continue
        target_errors = []
        delivered, uncertain = (False, False)
        if t.live_adapter_ready:
            delivered, uncertain = _deliver_via_live_adapter(t, cleaned_content, media_files,
                target_errors=target_errors, delivery_errors=errors, unverified_targets=unverified)
        if uncertain:
            errors.extend(target_errors)
        elif not delivered:
            _deliver_standalone(t, cleaned_content, media_files, target_errors, errors)
    errors.extend(policy_errors)
    _record_delivery_verification(job, unverified)
    return "; ".join(errors) if errors else None

_DEFAULT_MEDIA_SEND_TIMEOUT = 300


def _get_media_send_timeout() -> int:
    """Resolve the per-attachment media-send timeout from env/config.

    Mirrors the ``script_timeout_seconds`` resolution pattern: the
    HERMES_CRON_MEDIA_SEND_TIMEOUT env var wins, then
    ``cron.media_send_timeout_seconds`` in config.yaml, then the default
    (300s — large attachments like long TTS audio can legitimately exceed
    the old fixed 30s upload window).
    """
    env_value = os.getenv("HERMES_CRON_MEDIA_SEND_TIMEOUT", "").strip()
    if env_value:
        try:
            timeout = int(float(env_value))
            if timeout > 0:
                return timeout
        except Exception:
            logger.warning(
                "Invalid HERMES_CRON_MEDIA_SEND_TIMEOUT=%r; using config/default",
                env_value,
            )

    try:
        cfg = load_config() or {}
        cron_cfg = cfg.get("cron", {}) if isinstance(cfg, dict) else {}
        configured = cron_cfg.get("media_send_timeout_seconds")
        if configured is not None:
            timeout = int(float(configured))
            if timeout > 0:
                return timeout
    except Exception as exc:
        logger.debug("Failed to load cron media-send timeout from config: %s", exc)

    return _DEFAULT_MEDIA_SEND_TIMEOUT

# Late-bound scheduler siblings avoid import cycles and preserve direct-module test seams.
from cron import scheduler as _sched  # noqa: E402
from cron import scheduler_preflight as _preflight  # noqa: E402
from cron import scheduler_script as _script  # noqa: E402
SharedRouteAdapters = _preflight.SharedRouteAdapters

def _interpreter_shutting_down(exc=None):
    return _sched._interpreter_shutting_down(exc)
