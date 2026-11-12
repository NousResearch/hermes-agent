"""``cron/bot_chat_pending``: Bot Chat outputs whose target owner was not running yet.

``defer`` parks one (a cross-profile target whose gateway is down or still starting), and
the retired CLI lane left records of the same shape behind. Each still-``queued`` record is handed,
under its own id, to the target profile's live owner through the canonical
``tools.bot_live_delivery.deliver_to_live_owner`` door (idempotent per id). Records are
never deleted: they keep their final status as evidence, exactly as the old drain did.
A ``claimed`` record was an uncertain CLI turn and is never resent.

SINGLE PAYLOAD CONVENTION (both sides; see ``defer``, ``_dispose`` and the two producers
``cron/scheduler_delivery.py`` + ``agent-hooks/kanban-botchat-deliver.py``):
    ``content`` is the EXACT final inbound Bot Chat message to admit, never a raw job output.
    A producer wraps/formats it exactly once, BEFORE calling ``defer``; ``defer`` stamps the
    record ``content_format="final"`` and stores that payload verbatim. ``_dispose`` never
    re-wraps a stamped record. Before this convention, ``defer`` stored raw cron output and
    ``_dispose`` applied ``bot_chat_message``; a producer that had already formatted its body
    (the kanban handoff lane) was therefore wrapped twice, so the same delivery id carried two
    different payloads and every drain hit ``admission_conflict`` forever (376f9c…, 2026-10-27).
    Unstamped records are pre-convention/retired-CLI-lane only: ``_dispose`` keeps exactly one
    compatibility wrap for them. Any other ``content_format`` value is refused, never guessed.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any
from hermes_dir_acl import make_user_dir

from hermes_constants import get_hermes_home
from utils import atomic_json_write

logger = logging.getLogger(__name__)
_warned: set[Path] = set()

# Marks "content is already the final message; do not wrap it again".
CONTENT_FORMAT_FINAL = "final"
# A receipt in one of these states (or carrying an admission_id) proves the id was admitted.
_ADMITTED = frozenset({"queued", "claimed", "canonical", "settled", "delivered"})


def defer(key: str, job: dict, content: str, home: Path, *, for_failure: bool) -> None:
    """Park one never-admitted output under its delivery id; a later tick hands it to the owner.

    ``content`` MUST be the exact final message to admit (see module docstring): the caller
    wraps/formats once before calling here. The ``content_format="final"`` stamp is the
    contract with ``_dispose`` -- it must never wrap a stamped record a second time.
    """
    import time
    root = get_hermes_home().resolve() / "cron" / "bot_chat_pending"
    make_user_dir(root, parents=True, exist_ok=True, mode=0o700)
    path = root / f"{key}.json"
    if not path.exists():  # an existing record (queued or already transferred) keeps its status
        atomic_json_write(path, dict(
            id=key, status="queued", job={k: job[k] for k in ("id", "name") if k in job}, content=content,
            content_format=CONTENT_FORMAT_FINAL,
            home=str(home), for_failure=for_failure, sequence=time.time_ns()), fsync_dir=True, mode=0o600)


def _queued(root: Path) -> list[tuple[Path, dict]]:
    found = []
    for path in root.glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(record, dict):
                raise ValueError(f"expected a JSON object, got {type(record).__name__}")
        except (OSError, ValueError) as exc:
            if path not in _warned:
                _warned.add(path)
                logger.error("Unreadable legacy Bot Chat pending record %s kept as evidence: %s", path, exc)
            continue
        if record.get("status") == "queued":
            found.append((path, record))
    return sorted(found, key=lambda item: (item[1].get("sequence", 0), item[0].name))


def _malformed(record: dict) -> str:
    """Why *record* can never be handed off (empty when its required fields are usable)."""
    missing = [key for key in ("id", "home", "content")
               if not isinstance(record.get(key), str) or not record.get(key)]
    if missing:
        return f"malformed legacy bot-chat record (missing {', '.join(missing)}); not delivered"
    if not isinstance(record.get("job") or {}, dict):
        return "malformed legacy bot-chat record (job is not an object); not delivered"
    return ""


def _sha256(value: Any) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest() if isinstance(value, str) else ""


def _is_admission_conflict(exc: BaseException) -> bool:
    """A same-id identity refusal is permanent (the door will never accept that payload),
    not a transient dial error. Match the reason across GatewayRPCError / PreDialError /
    RuntimeStoreError without importing any of their modules here."""
    return ("admission_conflict" in str(exc)
            or getattr(exc, "reason", None) == "admission_conflict"
            or any("admission_conflict" in str(a) for a in getattr(exc, "args", ())))


def _existing_receipt(read, home: Path, delivery_id: str) -> dict | None:
    """Read the durable receipt for this id without dialing. Unreadable -> None (let the
    authority decide); never let a corrupt file stop the drain with an uncaught error."""
    try:
        value = read(home, delivery_id)
    except Exception as exc:  # a read failure is never a disposition
        logger.debug("Cannot read delivery receipt %s in %s: %s", delivery_id, home, exc)
        return None
    return value if isinstance(value, dict) else None


def _dispose_from_receipt(record: dict, existing: dict, message: str, *, conflict: str = "") -> None:
    """An id already holding a receipt decides the record WITHOUT resending.

    The door refuses a same-id replay with a different payload forever, so a mismatch is
    ``ambiguous`` evidence (kept, never re-sent) instead of a retry. A receipt holding this
    record's content -- verbatim (the direct lane admitted the formatted body) or in the
    single legacy wrapped shape -- proves it was already admitted, so ``transferred``.
    """
    held = existing.get("message")
    # An unmarked retired-CLI record may have been admitted either verbatim (direct lane)
    # or in its single legacy wrapped shape; record WHICH shape actually matched.
    candidates = {"final": record["content"]}
    if message != record["content"]:
        candidates["legacy_wrapped"] = message
    matched = next((shape for shape, payload in candidates.items()
                    if isinstance(held, str) and held == payload), None)
    evidence = dict(receipt_status=existing.get("status"),
                    receipt_message_sha256=_sha256(held),
                    pending_message_sha256=_sha256(candidates[matched] if matched else message))
    if matched:
        record.update(status="transferred", error=None, payload_shape=matched,
                      disposition="already admitted under this delivery id; no resend", **evidence)
    else:
        record.update(status="ambiguous",
                      error=("admission_conflict: delivery id already holds a different payload; "
                             "kept as evidence, not resent" + (f" ({conflict})" if conflict else "")),
                      **evidence)


def _dispose(path: Path, record: dict) -> bool:
    """Decide one queued record in place; False leaves it queued for a later tick.

    False is reserved for a genuinely transient condition (owner not ready, dial failure).
    ``admission_conflict`` is a permanent identity refusal: it becomes a terminal
    ``ambiguous`` disposition with evidence, never an endless per-tick retry.
    """
    from cron.scheduler_delivery import BOT_CHAT_POLICY_PLATFORM, bot_chat_message
    from gateway.warning_notifications import warning_notifications_enabled
    from hermes_cli.config_effective import load_user_config_effective
    from tools.bot_live_delivery import (deliver_to_live_owner, find_canonical_live_owner,
                                         read_delivery_result)

    if reason := _malformed(record):
        record.update(status="ambiguous", error=reason)
        return True

    # Resolve the exact payload ONCE, per the module-docstring convention. This is the
    # deterministic guard against the double-wrap that caused every drain to conflict.
    fmt = record.get("content_format")
    if fmt == CONTENT_FORMAT_FINAL:
        message = record["content"]                 # producer already wrapped: never wrap again
    elif fmt is None:
        message = bot_chat_message(record.get("job") or {}, record["content"])  # legacy CLI lane
    else:
        record.update(status="ambiguous",
                      error=f"unknown content_format {fmt!r}; kept as evidence, not delivered")
        return True

    home = Path(record["home"]).resolve()
    for_failure = bool(record.get("for_failure"))
    if not (home / "state.db").is_file():
        record.update(status="ambiguous", error=f"bot-chat delivery target no longer exists: {home}")
    elif for_failure and not warning_notifications_enabled(
            BOT_CHAT_POLICY_PLATFORM, load_user_config_effective(home / "config.yaml")):
        record.update(status="suppressed", error=None)
    else:
        # Read BEFORE discovery (same pattern as scheduler_delivery._deliver_to_bot_chat):
        # a receipt under this id is durable evidence, so no dial is needed at all.
        existing = _existing_receipt(read_delivery_result, home, record["id"])
        if existing is not None:
            _dispose_from_receipt(record, existing, message)
            return True
        try:
            owner = find_canonical_live_owner(home)
        except ValueError:
            return False  # Owner not ready: discovery uncertainty is not a disposition.
        if owner is None:
            return False
        try:
            receipt = deliver_to_live_owner(
                home, owner, message, delivery_id=record["id"],
                notification_category="diagnostic" if for_failure else "result")
        except Exception as exc:
            if _is_admission_conflict(exc):
                # A receipt won the race, or the admitted identity (author/category) differs.
                # Either way the id is settled: keep the evidence, never retry forever.
                raced = _existing_receipt(read_delivery_result, home, record["id"])
                if raced is not None:
                    _dispose_from_receipt(record, raced, message, conflict=str(exc))
                else:
                    record.update(status="ambiguous",
                                  error=(f"admission_conflict: {exc}; delivery id is already "
                                         "claimed by a different identity; kept as evidence, "
                                         "not resent"),
                                  pending_message_sha256=_sha256(message))
                return True
            logger.log(logging.DEBUG if path in _warned else logging.WARNING,
                       "Legacy Bot Chat pending record %s not handed off yet: %s", path, exc, exc_info=True)
            _warned.add(path)
            return False
        receipt_status = receipt.get("status") if isinstance(receipt, dict) else None
        record.update(status="transferred", error=None, receipt_status=receipt_status)
    return True


def drain_legacy_pending() -> None:
    """Admit queued legacy records to their live owner; an absent owner leaves them for a later tick.

    Each record is isolated: a malformed one is kept as ``ambiguous`` evidence and an
    unexpected failure leaves only that record queued, so one bad file never stops cron.
    """
    root = get_hermes_home().resolve() / "cron" / "bot_chat_pending"
    if not root.is_dir():
        return
    for path, record in _queued(root):
        try:
            if not _dispose(path, record):
                continue
            atomic_json_write(path, record, fsync_dir=True, mode=0o600)
        except Exception as exc:
            logger.log(logging.DEBUG if path in _warned else logging.WARNING,
                       "Legacy Bot Chat pending record %s left queued: %s", path, exc, exc_info=True)
            _warned.add(path)
            continue
        logger.info("Legacy Bot Chat pending record %s -> %s", path.name, record["status"])
