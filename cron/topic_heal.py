"""In-delivery self-heal for cron Telegram forum-topic targets.

THE GAP THIS CLOSES
-------------------
A cron job whose ``deliver`` target is a Telegram forum topic
(``telegram:<chat>:<thread>``) silently misroutes to the group root the moment
the user deletes that topic: the live adapter falls back to a no-thread send
(``thread_fallback=True``) and the standalone path drops the thread too. The old
fix was a separate sweep cron (``agent-topic-heal.py``) that probed + repointed
BEFORE the next fire. This module folds that heal INTO the delivery path so it is
automatic, inline, and one moving part fewer.

HOW
---
``cron/scheduler.py::_deliver_result`` calls :func:`heal_dead_thread` when a send
reports the thread is gone. This module:
  1. Resolves the topic NAME + SEED + icon for the job. A cron deliver target
     carries only a numeric thread id, so the human-readable metadata lives on the
     job record as ``job["telegram_topic"] = {name, seed, icon_emoji_id}`` (read
     first), with the legacy ``state/agent-topic-registry.json`` sidecar as a
     migration-era fallback.
  2. Recreates the topic via the live gateway adapter when one is running, else a
     one-shot standalone Bot API call.
  3. Writes the seed sidecar (same contract as ``scripts/tg_topic_seed.py``:
     ``state/topic-seeds/<thread>.json`` => ``{thread_id, seed_text, created_at}``)
     so the user's first reply in the recreated topic wakes a context-aware
     session. Uses ``_get_hermes_home()`` so it is testable against a temp home.
  4. Repoints THIS job and every co-located enabled job (same chat+old-thread) to
     the new thread via ``cron.jobs.update_job`` (atomic, persists across ticks).

The send result is the probe — no separate ``editForumTopic`` liveness check.
Returns the new thread id on success, ``None`` when it cannot heal (no metadata,
recreate failed) so the caller can preserve the legacy group-root fallback and an
alert is never lost.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

TELEGRAM_TOPIC_FIELD = "telegram_topic"
# Metadata flag that tells the Telegram adapter to REPORT a dead thread instead of
# silently falling back to the group root (so we can recreate the topic first).
NO_THREAD_FALLBACK_METADATA = "telegram_no_thread_fallback"
DEFAULT_TOPIC_ICON_COLOR = 0x6FB9F0


def _hermes_home() -> Path:
    """Resolve Hermes home via the scheduler's hook so a test monkeypatch on
    ``cron.scheduler._hermes_home`` is honored here too."""
    try:
        from cron.scheduler import _get_hermes_home
        return _get_hermes_home()
    except Exception:
        return Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))


# --- metadata resolution -----------------------------------------------------

def topic_metadata_for_job(job: dict) -> Optional[Dict[str, Any]]:
    """Return normalized {name, seed, icon_color?, icon_emoji_id?} for a job.

    Job record (``job["telegram_topic"]``) wins; legacy sidecar registry is the
    fallback. Returns None when neither yields a name+seed (cannot recreate a
    topic without a name, and a recreated topic needs a seed to be context-aware).
    """
    raw = job.get(TELEGRAM_TOPIC_FIELD)
    meta = _normalize_topic_meta(raw)
    if meta:
        return meta
    # Legacy sidecar registry fallback (migration era).
    reg = _hermes_home() / "state" / "agent-topic-registry.json"
    try:
        data = json.loads(reg.read_text())
        entry = (data.get("jobs") or {}).get(job.get("id"))
        return _normalize_topic_meta(entry)
    except Exception:
        return None


def _normalize_topic_meta(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    name = str(raw.get("name") or raw.get("topic_name") or "").strip()
    seed = str(raw.get("seed") or raw.get("seed_text") or "").strip()
    if not name or not seed:
        return None
    meta: Dict[str, Any] = {"name": name[:128], "seed": seed}
    icon_color = raw.get("icon_color", DEFAULT_TOPIC_ICON_COLOR)
    if icon_color is not None:
        try:
            meta["icon_color"] = int(icon_color)
        except (TypeError, ValueError):
            meta["icon_color"] = DEFAULT_TOPIC_ICON_COLOR
    icon_emoji_id = raw.get("icon_emoji_id") or raw.get("icon_custom_emoji_id")
    if icon_emoji_id:
        meta["icon_emoji_id"] = str(icon_emoji_id)
    return meta


def is_telegram_forum_topic_target(platform_name: str, chat_id: Any, thread_id: Any) -> bool:
    """True only for a Telegram forum-supergroup topic target (chat -100..., a
    numeric thread that is not the General topic 1). DM topics and the General
    topic are out of scope here (the gateway DeliveryManager handles DM topics)."""
    if str(platform_name or "").lower() != "telegram":
        return False
    if thread_id is None or str(thread_id) == "1":
        return False
    if not str(chat_id or "").startswith("-100"):
        return False
    try:
        int(thread_id)
        return True
    except (TypeError, ValueError):
        return False


def thread_not_found_in_result(result: Any) -> bool:
    """Detect a deleted-topic signal from a live SendResult OR a standalone dict.

    Live adapter: SendResult.raw_response has thread_not_found/thread_fallback.
    Standalone:   dict has thread_not_found/thread_fallback, or an error string
                  containing "thread not found".
    """
    if result is None:
        return False
    raw = None
    error = None
    if isinstance(result, dict):
        raw = result.get("raw_response") or result
        error = result.get("error")
    else:
        raw = getattr(result, "raw_response", None)
        error = getattr(result, "error", None)
    if isinstance(raw, dict) and (raw.get("thread_not_found") or raw.get("thread_fallback")):
        return True
    return bool(error and "thread not found" in str(error).lower())


def requested_thread_from_result(result: Any, fallback: Any) -> Any:
    """Pull the requested_thread_id out of a send result, else use fallback."""
    raw = None
    if isinstance(result, dict):
        raw = result.get("raw_response") or result
        direct = result.get("requested_thread_id")
        if direct:
            return direct
    else:
        raw = getattr(result, "raw_response", None)
    if isinstance(raw, dict) and raw.get("requested_thread_id"):
        return raw["requested_thread_id"]
    return fallback


# --- seed sidecar (core copy of scripts/tg_topic_seed.write_seed) -------------

def write_topic_seed(thread_id, seed_text: str) -> bool:
    """Atomically write the per-topic seed sidecar the topic-rename-toggle plugin
    consumes on the user's first reply. Shape MUST match scripts/tg_topic_seed.py:
    ``{"thread_id": str, "seed_text": str, "created_at": float}`` under
    ``$HERMES_HOME/state/topic-seeds/<thread_id>.json``.
    """
    if thread_id is None or not str(seed_text or "").strip():
        return False
    try:
        import time
        seed_dir = _hermes_home() / "state" / "topic-seeds"
        seed_dir.mkdir(parents=True, exist_ok=True)
        p = seed_dir / f"{thread_id}.json"
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(
                {"thread_id": str(thread_id), "seed_text": str(seed_text), "created_at": time.time()},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        os.replace(tmp, p)
        return True
    except Exception:
        logger.warning("topic_heal: seed write failed for thread %s", thread_id, exc_info=True)
        return False


# --- topic recreate (live adapter, else standalone) ---------------------------

def _run_on_gateway_loop(coro, loop, timeout: int = 30):
    from agent.async_utils import safe_schedule_threadsafe
    future = safe_schedule_threadsafe(coro, loop)
    if future is None:
        try:
            coro.close()
        except Exception:
            pass
        raise RuntimeError("gateway loop scheduling failed")
    try:
        return future.result(timeout=timeout)
    except TimeoutError:
        future.cancel()
        raise


def _run_standalone(coro_factory, timeout: int = 30):
    """Run an async coro factory in a fresh event loop, safe from any thread.

    Only the 'event loop is already running' RuntimeError is handled by retrying
    in a worker thread; any other RuntimeError (raised by the coroutine itself) is
    re-raised so a real error is not silently retried (which could double a
    create/delete side effect).
    """
    try:
        return asyncio.run(coro_factory())
    except RuntimeError as e:
        if "running event loop" not in str(e).lower() and "cannot be called from a running" not in str(e).lower():
            raise
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro_factory()).result(timeout=timeout)


def _standalone_bot(token):
    """Build a Telegram Bot honoring the configured proxy, mirroring
    tools.send_message_tool._send_telegram so standalone recreate works in
    proxy-required regions (without a proxy the create would fail and every fire
    would fall back to the group root)."""
    from telegram import Bot
    try:
        from gateway.platforms.base import resolve_proxy_url
        proxy = resolve_proxy_url("TELEGRAM_PROXY", target_hosts=["api.telegram.org"])
    except Exception:
        proxy = None
    if proxy:
        try:
            from telegram.request import HTTPXRequest
            return Bot(token=token, request=HTTPXRequest(proxy=proxy),
                       get_updates_request=HTTPXRequest(proxy=proxy))
        except Exception:
            pass
    return Bot(token=token)


def recreate_topic(
    chat_id: str,
    meta: Dict[str, Any],
    *,
    adapter=None,
    loop=None,
    token: Optional[str] = None,
) -> Optional[int]:
    """Recreate a forum topic; prefer the live adapter, else a standalone Bot call.
    Returns the new thread id (int) or None."""
    name = meta["name"]
    icon_color = meta.get("icon_color")
    icon_emoji_id = meta.get("icon_emoji_id")

    # Live adapter path (gateway running).
    if (
        adapter is not None
        and loop is not None
        and getattr(loop, "is_running", lambda: False)()
        and getattr(adapter, "create_forum_topic", None) is not None
    ):
        try:
            tid = _run_on_gateway_loop(
                adapter.create_forum_topic(
                    chat_id, name, icon_color=icon_color, icon_custom_emoji_id=icon_emoji_id,
                ),
                loop,
            )
            if tid:
                return int(tid)
        except Exception as e:
            logger.warning("topic_heal: live recreate failed for '%s' in %s: %s", name, chat_id, e)

    # Standalone path (no gateway, or live path failed).
    if token:
        def _factory():
            return _standalone_create(token, chat_id, name, icon_color, icon_emoji_id)
        try:
            tid = _run_standalone(_factory)
            if tid:
                return int(tid)
        except Exception as e:
            logger.warning("topic_heal: standalone recreate failed for '%s' in %s: %s", name, chat_id, e)
    return None


async def _standalone_create(token, chat_id, name, icon_color, icon_emoji_id) -> Optional[int]:
    bot = _standalone_bot(token)
    kwargs: Dict[str, Any] = {"chat_id": int(chat_id), "name": str(name).strip()[:128]}
    if icon_color is not None:
        kwargs["icon_color"] = int(icon_color)
    if icon_emoji_id:
        kwargs["icon_custom_emoji_id"] = str(icon_emoji_id)
    try:
        topic = await bot.create_forum_topic(**kwargs)
    except Exception as e:
        if icon_emoji_id and "emoji" in str(e).lower():
            kwargs.pop("icon_custom_emoji_id", None)
            topic = await bot.create_forum_topic(**kwargs)
        else:
            raise
    return int(topic.message_thread_id)


# --- co-located repoint -------------------------------------------------------

def _resolved_target(job: dict) -> Tuple[Optional[str], Optional[str]]:
    """(chat_id, thread_id) this job delivers to right now, mirroring
    cron.scheduler._resolve_single_delivery_target for the telegram case."""
    deliver = str(job.get("deliver") or "")
    origin = job.get("origin")
    if not isinstance(origin, dict):
        origin = {}
    ochat = origin.get("chat_id")
    otid = origin.get("thread_id")
    # Only the FIRST telegram part matters for co-location; multi-target deliver
    # is rewritten part-by-part in repoint_colocated_jobs.
    for part in [p.strip() for p in deliver.split(",") if p.strip()]:
        bits = part.split(":")
        if len(bits) == 3 and bits[0] == "telegram":
            return (bits[1] or ochat), (bits[2] or None)
        if len(bits) == 2 and bits[0] == "telegram":
            return (bits[1] or ochat), None
        if part in ("origin", origin.get("platform")):
            return (str(ochat) if ochat is not None else None), (str(otid) if otid is not None else None)
    return (str(ochat) if ochat is not None else None), None


def _rewrite_deliver(job: dict, chat_id: str, old_tid: str, new_tid) -> Tuple[str, dict]:
    """Rewrite only the matching telegram part(s) of a (possibly multi-target)
    deliver string; refresh origin.thread_id when origin matched. Other targets
    (e.g. 'local', a different chat) are preserved."""
    deliver = str(job.get("deliver") or "local")
    parts = [p.strip() for p in deliver.split(",") if p.strip()] or [deliver]
    origin = dict(job.get("origin") or {}) if isinstance(job.get("origin"), dict) else {}
    origin_matches = (
        str(origin.get("platform", "")).lower() == "telegram"
        and str(origin.get("chat_id")) == str(chat_id)
        and str(origin.get("thread_id")) == str(old_tid)
    )
    if origin_matches:
        origin["thread_id"] = str(new_tid)
        origin["chat_id"] = str(chat_id)

    new_parts: List[str] = []
    for part in parts:
        bits = part.split(":")
        explicit = (
            len(bits) == 3 and bits[0] == "telegram"
            and str(bits[1]) == str(chat_id) and str(bits[2]) == str(old_tid)
        )
        inherits = part in ("origin",) or (part.lower() == "telegram" and origin_matches)
        if explicit:
            new_parts.append(f"telegram:{chat_id}:{new_tid}")
        elif inherits:
            new_parts.append(part)  # origin re-inherits the refreshed origin.thread_id
        else:
            new_parts.append(part)
    return ",".join(new_parts), origin


def _job_targets_thread(job: dict, chat_id: str, old_tid: str) -> bool:
    """True if ANY of the job's delivery targets resolves to (chat_id, old_tid).

    Unlike _resolved_target (first telegram part only), this scans EVERY comma
    part plus the origin re-inherit, so a co-located job whose dead thread is its
    SECOND target is still matched and repointed. _rewrite_deliver then rewrites
    only the matching part(s).
    """
    chat_id = str(chat_id)
    old_tid = str(old_tid)
    deliver = str(job.get("deliver") or "")
    origin = job.get("origin")
    if not isinstance(origin, dict):
        origin = {}
    ochat = origin.get("chat_id")
    otid = origin.get("thread_id")
    for part in [p.strip() for p in deliver.split(",") if p.strip()]:
        bits = part.split(":")
        if len(bits) == 3 and bits[0] == "telegram":
            if str(bits[1] or ochat) == chat_id and str(bits[2]) == old_tid:
                return True
        elif part in ("origin", origin.get("platform")):
            if ochat is not None and otid is not None and str(ochat) == chat_id and str(otid) == old_tid:
                return True
    return False


def repoint_colocated_jobs(chat_id: str, old_tid: str, new_tid: int, meta: Dict[str, Any]) -> List[str]:
    """Repoint every enabled job targeting (chat_id, old_tid) to new_tid and copy
    the topic metadata onto them so they can self-heal independently next time.
    Returns the repointed job ids."""
    from cron.jobs import load_jobs, update_job

    chat_id = str(chat_id)
    old_tid = str(old_tid)
    repointed: List[str] = []
    topic_block = {"name": meta["name"], "seed": meta["seed"]}
    if "icon_color" in meta:
        topic_block["icon_color"] = meta["icon_color"]
    if "icon_emoji_id" in meta:
        topic_block["icon_emoji_id"] = meta["icon_emoji_id"]

    for job in load_jobs():
        if not job.get("enabled", True):
            continue
        # Scan ALL targets (not just the first) so a dead thread in a non-first
        # multi-target slot is still repointed.
        if not _job_targets_thread(job, chat_id, old_tid):
            continue
        new_deliver, new_origin = _rewrite_deliver(job, chat_id, old_tid, new_tid)
        updates: Dict[str, Any] = {"deliver": new_deliver, TELEGRAM_TOPIC_FIELD: topic_block}
        if new_origin:
            updates["origin"] = new_origin
        try:
            if update_job(job["id"], updates) is not None:
                repointed.append(job["id"])
        except Exception as e:
            logger.warning("topic_heal: repoint failed for %s: %s", job.get("id"), e)
    return repointed


# --- orchestrator -------------------------------------------------------------

# Per-(chat, old_thread) locks serialize heal attempts ACROSS the parallel cron
# pool (threads in one process — see cron/scheduler._parallel_pool). Without this,
# two co-located jobs (e.g. a monitor + its watchdog on one topic) firing in the
# same tick both recreate the topic -> duplicate orphan topics + split delivery.
_heal_locks: Dict[Tuple[str, str], threading.Lock] = {}
_heal_locks_guard = threading.Lock()


def _heal_lock(chat_id: str, old_thread_id: str) -> threading.Lock:
    key = (str(chat_id), str(old_thread_id))
    with _heal_locks_guard:
        lock = _heal_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _heal_locks[key] = lock
        return lock


def _live_thread_for_job(job_id, chat_id: str) -> Optional[str]:
    """Re-read the job's CURRENTLY persisted Telegram thread for chat_id from disk.

    Used inside the heal lock to detect that a co-located sibling already healed
    this topic (the job's deliver was repointed off the dead thread), so we return
    that fresh thread instead of recreating a duplicate.
    """
    try:
        from cron.jobs import load_jobs
        for j in load_jobs():
            if j.get("id") != job_id:
                continue
            c, t = _resolved_target(j)
            if c is not None and str(c) == str(chat_id):
                return str(t) if t is not None else None
    except Exception:
        return None
    return None


def heal_dead_thread(
    job: dict,
    chat_id: str,
    old_thread_id: str,
    *,
    adapter=None,
    loop=None,
    token: Optional[str] = None,
) -> Optional[int]:
    """Recreate + seed the topic, repoint co-located jobs, return the new thread id.

    Returns None when the job has no resolvable topic metadata or recreate fails;
    the caller then keeps the legacy group-root fallback so an alert is not lost.
    On a seed-write failure the just-created topic is rolled back (deleted) and
    None is returned, so we never deliver into an unseeded (context-blind) topic.

    Concurrency-safe: serialized per (chat, old_thread). Inside the lock it
    re-reads the job's live persisted thread; if a co-located sibling already
    recreated the topic this tick, it returns that fresh thread WITHOUT creating
    a duplicate.
    """
    meta = topic_metadata_for_job(job)
    if not meta:
        logger.info(
            "topic_heal: job '%s' has no topic metadata; cannot self-heal dead thread %s",
            job.get("id"), old_thread_id,
        )
        return None

    with _heal_lock(chat_id, old_thread_id):
        # Another co-located job may have already healed this topic while we
        # waited on the lock. If THIS job's persisted thread moved off the dead
        # one, reuse it instead of recreating a duplicate.
        live = _live_thread_for_job(job.get("id"), chat_id)
        if live and str(live) != str(old_thread_id):
            try:
                logger.info(
                    "topic_heal: job '%s' dead thread %s already healed -> %s (sibling); reusing",
                    job.get("id"), old_thread_id, live,
                )
                return int(live)
            except (TypeError, ValueError):
                return None

        new_tid = recreate_topic(chat_id, meta, adapter=adapter, loop=loop, token=token)
        if not new_tid:
            return None

        # Seed BEFORE repoint so the sidecar exists the instant the topic is live.
        if not write_topic_seed(new_tid, meta["seed"]):
            # Roll back the just-created (unseeded) topic; do not deliver context-blind.
            _rollback_topic(chat_id, new_tid, adapter=adapter, loop=loop, token=token)
            logger.warning(
                "topic_heal: seed write failed for new thread %s; rolled back, no repoint", new_tid,
            )
            return None

        repointed = repoint_colocated_jobs(chat_id, old_thread_id, new_tid, meta)
        logger.warning(
            "topic_heal: recreated topic '%s' (dead %s -> %s); repointed %s",
            meta["name"], old_thread_id, new_tid, ", ".join(repointed) or "(none)",
        )
        return int(new_tid)


def _rollback_topic(chat_id, thread_id, *, adapter=None, loop=None, token=None) -> None:
    """Best-effort delete of a just-created topic (seed-write failed). Live adapter
    first, else a standalone Bot call."""
    if (
        adapter is not None and loop is not None
        and getattr(loop, "is_running", lambda: False)()
        and getattr(adapter, "delete_forum_topic", None) is not None
    ):
        try:
            _run_on_gateway_loop(adapter.delete_forum_topic(str(chat_id), str(thread_id)), loop)
            return
        except Exception:
            logger.debug("topic_heal: live rollback delete failed for %s:%s", chat_id, thread_id, exc_info=True)
    if token:
        async def _del():
            await _standalone_bot(token).delete_forum_topic(chat_id=int(chat_id), message_thread_id=int(thread_id))
        try:
            _run_standalone(_del)
        except Exception:
            logger.debug("topic_heal: standalone rollback delete failed for %s:%s", chat_id, thread_id, exc_info=True)
