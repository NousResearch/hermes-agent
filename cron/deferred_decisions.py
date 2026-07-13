"""Strict, durable deferred-decision protocol for autonomous cron delivery.

This module deliberately sits at the delivery boundary.  Cron agents remain
non-interactive and never receive the clarify or messaging toolsets; a final
response may only *describe* bounded choices for a capable live adapter to
render after the run has finished.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import secrets
import tempfile
import threading
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

from hermes_constants import get_hermes_home
from utils import atomic_replace

try:
    import fcntl
except ImportError:  # pragma: no cover - non-Unix
    fcntl = None
try:
    import msvcrt
except ImportError:  # pragma: no cover - non-Windows
    msvcrt = None


PROTOCOL_FENCE = "hermes-deferred-decisions"
MAX_CARDS = 3
MIN_CHOICES = 2
MAX_CHOICES = 4
MAX_QUESTION_CHARS = 500
MAX_CHOICE_CHARS = 100
MAX_CONTROL_BYTES = 8192
MAX_PENDING_RECORDS = 96
DECISION_TTL = timedelta(hours=24)
CLAIM_LEASE = timedelta(minutes=5)

_BLOCK_RE = re.compile(
    rf"(?:^|\n)```{re.escape(PROTOCOL_FENCE)}\n(?P<payload>[^`]{{1,{MAX_CONTROL_BYTES}}})\n```\s*\Z"
)
_JOB_ID_RE = re.compile(r"^[0-9a-f]{12}$")
_DECISION_ID_RE = re.compile(r"^[0-9a-f]{16}$")
_CLAIM_TOKEN_RE = re.compile(r"^[0-9a-f]{32}$")
_process_lock = threading.RLock()


@dataclass(frozen=True)
class DeferredDecisionCard:
    question: str
    choices: tuple[str, ...]


@dataclass(frozen=True)
class ParsedDeferredDecisions:
    visible_text: str
    cards: tuple[DeferredDecisionCard, ...]


@dataclass(frozen=True)
class DeferredDecisionRecord:
    job_id: str
    decision_id: str
    job_name: str
    card_index: int
    platform: str
    chat_id: str
    thread_id: Optional[str]
    user_id: Optional[str]
    question: str
    choices: tuple[str, ...]
    created_at: str
    expires_at: str
    context_ready: bool
    session_source: dict[str, Any]
    session_key: str
    message_id: Optional[str] = None
    claimed: bool = False
    claim_token: Optional[str] = None
    claim_expires_at: Optional[str] = None


@dataclass(frozen=True)
class ClaimedChoice:
    record: DeferredDecisionRecord
    choice: str
    choice_index: int
    claim_token: str


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key}")
        result[key] = value
    return result


def _valid_bounded_line(value: Any, *, maximum: int) -> Optional[str]:
    if not isinstance(value, str) or not value or value != value.strip():
        return None
    if len(value) > maximum or any(ord(char) < 32 for char in value):
        return None
    return value


def parse_deferred_decisions(content: str) -> Optional[ParsedDeferredDecisions]:
    """Parse one strict terminal control block, returning ``None`` on any doubt.

    ``None`` means the caller must deliver the original response unchanged.
    The parser never performs partial recovery and never strips malformed data.
    """
    if not isinstance(content, str) or content.count(f"```{PROTOCOL_FENCE}") != 1:
        return None
    match = _BLOCK_RE.search(content)
    if not match:
        return None
    payload_text = match.group("payload")
    if len(payload_text.encode("utf-8")) > MAX_CONTROL_BYTES:
        return None
    try:
        payload = json.loads(payload_text, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(payload, dict) or set(payload) != {"version", "cards"}:
        return None
    if type(payload["version"]) is not int or payload["version"] != 1:
        return None
    raw_cards = payload["cards"]
    if not isinstance(raw_cards, list) or not 1 <= len(raw_cards) <= MAX_CARDS:
        return None

    cards: list[DeferredDecisionCard] = []
    for raw_card in raw_cards:
        if not isinstance(raw_card, dict) or set(raw_card) != {"question", "choices"}:
            return None
        question = _valid_bounded_line(
            raw_card["question"], maximum=MAX_QUESTION_CHARS
        )
        raw_choices = raw_card["choices"]
        if question is None or not isinstance(raw_choices, list):
            return None
        if not MIN_CHOICES <= len(raw_choices) <= MAX_CHOICES:
            return None
        choices: list[str] = []
        for raw_choice in raw_choices:
            choice = _valid_bounded_line(raw_choice, maximum=MAX_CHOICE_CHARS)
            if choice is None:
                return None
            choices.append(choice)
        if len(set(choices)) != len(choices):
            return None
        cards.append(DeferredDecisionCard(question, tuple(choices)))

    return ParsedDeferredDecisions(
        visible_text=content[: match.start()].rstrip(),
        cards=tuple(cards),
    )


def delivery_is_eligible(job: dict, target: dict, adapter: Any) -> bool:
    """Return whether a target may receive native deferred-decision cards."""
    if job.get("attach_to_session") is not True:
        return False
    if str(job.get("deliver") or "").strip().lower() != "origin":
        return False
    origin = job.get("origin")
    if not isinstance(origin, dict):
        return False
    if str(origin.get("platform") or "").lower() != str(target.get("platform") or "").lower():
        return False
    if str(origin.get("chat_id") or "") != str(target.get("chat_id") or ""):
        return False
    origin_thread = origin.get("thread_id")
    target_thread = target.get("thread_id")
    if (None if origin_thread is None else str(origin_thread)) != (
        None if target_thread is None else str(target_thread)
    ):
        return False
    return callable(getattr(adapter, "send_deferred_decision", None))


def callback_data(
    job_id: str,
    decision_id: str,
    card_index: int,
    choice_index: int,
) -> str:
    """Build the compact Telegram callback token (Bot API limit: 64 bytes)."""
    if not _JOB_ID_RE.fullmatch(str(job_id)):
        raise ValueError("invalid cron job id")
    if not _DECISION_ID_RE.fullmatch(str(decision_id)):
        raise ValueError("invalid decision id")
    if not 0 <= int(card_index) < MAX_CARDS:
        raise ValueError("invalid card index")
    if not 0 <= int(choice_index) < MAX_CHOICES:
        raise ValueError("invalid choice index")
    value = f"cd:{job_id}:{decision_id}:{int(card_index)}:{int(choice_index)}"
    if len(value.encode("utf-8")) > 64:  # defensive invariant
        raise ValueError("callback data exceeds platform limit")
    return value


def _state_paths() -> tuple[Path, Path]:
    cron_dir = get_hermes_home().resolve() / "cron"
    return cron_dir / "deferred_decisions.json", cron_dir / ".deferred_decisions.lock"


@contextlib.contextmanager
def _locked_store() -> Iterator[Path]:
    state_path, lock_path = _state_paths()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(state_path.parent, 0o700)
    except (OSError, NotImplementedError):
        pass
    with _process_lock:
        with open(lock_path, "a+b") as lock_file:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            elif msvcrt is not None:  # pragma: no cover - Windows
                lock_file.seek(0, os.SEEK_END)
                if lock_file.tell() == 0:
                    lock_file.write(b"0")
                    lock_file.flush()
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield state_path
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                elif msvcrt is not None:  # pragma: no cover - Windows
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)


def _aware_datetime(value: Any) -> Optional[datetime]:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _canonical_session_source(
    raw: Any,
    *,
    platform: str,
    chat_id: str,
    thread_id: Optional[str],
) -> Optional[dict[str, Any]]:
    """Validate and canonicalize the exact source whose session was seeded."""
    if not isinstance(raw, dict):
        return None
    try:
        from gateway.session import SessionSource

        source = SessionSource.from_dict(raw)
        canonical = source.to_dict()
    except (KeyError, TypeError, ValueError):
        return None
    if canonical != raw:
        return None
    if source.platform.value.lower() != platform.lower():
        return None
    if str(source.chat_id) != str(chat_id):
        return None
    normalized_thread = str(thread_id) if thread_id is not None else None
    source_thread = str(source.thread_id) if source.thread_id is not None else None
    if source_thread != normalized_thread:
        return None
    if _valid_bounded_line(source.chat_type, maximum=32) is None:
        return None
    for value, maximum in (
        (source.chat_name, 512),
        (source.user_id, 256),
        (source.user_name, 512),
        (source.chat_topic, 1024),
        (source.profile, 100),
    ):
        if value is not None and _valid_bounded_line(value, maximum=maximum) is None:
            return None
    return canonical


def _valid_session_key(value: Any, *, platform: str, chat_type: str) -> bool:
    if _valid_bounded_line(value, maximum=2048) is None:
        return False
    parts = str(value).split(":", 4)
    return (
        len(parts) >= 4
        and parts[0] == "agent"
        and parts[2] == platform
        and parts[3] == chat_type
    )


def _record_from_dict(raw: Any) -> Optional[DeferredDecisionRecord]:
    if not isinstance(raw, dict):
        return None
    expected = {
        "job_id", "decision_id", "job_name", "card_index", "platform", "chat_id",
        "thread_id", "user_id", "question", "choices", "created_at",
        "expires_at", "context_ready", "session_source", "session_key",
        "message_id", "claimed", "claim_token", "claim_expires_at",
    }
    if set(raw) != expected or not isinstance(raw.get("choices"), list):
        return None
    try:
        record = DeferredDecisionRecord(**{**raw, "choices": tuple(raw["choices"])})
    except (TypeError, ValueError):
        return None
    if not isinstance(record.job_id, str) or not _JOB_ID_RE.fullmatch(record.job_id):
        return None
    if (
        not isinstance(record.decision_id, str)
        or not _DECISION_ID_RE.fullmatch(record.decision_id)
    ):
        return None
    if type(record.card_index) is not int or not 0 <= record.card_index < MAX_CARDS:
        return None
    if type(record.context_ready) is not bool or type(record.claimed) is not bool:
        return None
    if not _valid_bounded_line(record.job_name, maximum=100):
        return None
    if not _valid_bounded_line(record.platform, maximum=32):
        return None
    if not _valid_bounded_line(record.chat_id, maximum=256):
        return None
    if record.thread_id is not None and not _valid_bounded_line(record.thread_id, maximum=256):
        return None
    if record.user_id is not None and not _valid_bounded_line(record.user_id, maximum=256):
        return None
    if record.message_id is not None and not _valid_bounded_line(
        record.message_id, maximum=256
    ):
        return None
    session_source = _canonical_session_source(
        record.session_source,
        platform=record.platform,
        chat_id=record.chat_id,
        thread_id=record.thread_id,
    )
    if session_source is None or not _valid_session_key(
        record.session_key,
        platform=record.platform,
        chat_type=str(session_source["chat_type"]),
    ):
        return None
    if record.claimed:
        if (
            not isinstance(record.claim_token, str)
            or not _CLAIM_TOKEN_RE.fullmatch(record.claim_token)
            or _aware_datetime(record.claim_expires_at) is None
        ):
            return None
    elif record.claim_token is not None or record.claim_expires_at is not None:
        return None
    card = parse_deferred_decisions(
        "```hermes-deferred-decisions\n"
        + json.dumps({
            "version": 1,
            "cards": [{"question": record.question, "choices": list(record.choices)}],
        }, ensure_ascii=False)
        + "\n```"
    )
    if card is None:
        return None
    if _aware_datetime(record.created_at) is None or _aware_datetime(record.expires_at) is None:
        return None
    return record


def _read_records(path: Path) -> list[DeferredDecisionRecord]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError, UnicodeError):
        return []
    if not isinstance(raw, dict) or raw.get("version") != 1 or not isinstance(raw.get("records"), list):
        return []
    records = [_record_from_dict(item) for item in raw["records"]]
    return [record for record in records if record is not None]


def _prune_records(
    records: Sequence[DeferredDecisionRecord],
    *,
    now: Optional[datetime] = None,
) -> list[DeferredDecisionRecord]:
    """Drop expired records and release abandoned leased claims."""
    current = now or datetime.now(timezone.utc)
    kept: list[DeferredDecisionRecord] = []
    for record in records:
        expires_at = _aware_datetime(record.expires_at)
        if expires_at is None or expires_at <= current:
            continue
        if record.claimed:
            claim_expires_at = _aware_datetime(record.claim_expires_at)
            if claim_expires_at is None:
                continue
            if claim_expires_at <= current:
                record = replace(
                    record,
                    claimed=False,
                    claim_token=None,
                    claim_expires_at=None,
                )
        kept.append(record)
    return kept


def _write_records(path: Path, records: Sequence[DeferredDecisionRecord]) -> None:
    active_records = _prune_records(records)
    payload = {
        "version": 1,
        "records": [
            {**asdict(record), "choices": list(record.choices)}
            for record in active_records
        ],
    }
    fd, temp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=".deferred_decisions_", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        atomic_replace(temp_name, path)
        try:
            os.chmod(path, 0o600)
        except (OSError, NotImplementedError):
            pass
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def save_records(records: Sequence[DeferredDecisionRecord]) -> None:
    """Replace pending records atomically. Primarily used by delivery/tests."""
    with _locked_store() as path:
        _write_records(path, list(records))


def register_cards(
    *,
    job: dict,
    cards: Sequence[DeferredDecisionCard],
    platform: str,
    chat_id: str,
    thread_id: Optional[str],
    user_id: Optional[str],
    context_ready: bool,
    session_source: dict[str, Any],
    session_key: str,
) -> list[DeferredDecisionRecord]:
    """Persist the cards before exposing callbacks to the user."""
    now = datetime.now(timezone.utc)
    job_id = str(job.get("id") or "")
    if not _JOB_ID_RE.fullmatch(job_id):
        return []
    job_name = str(job.get("name") or job_id).strip()[:100]
    if _valid_bounded_line(job_name, maximum=100) is None:
        job_name = job_id
    normalized_platform = str(platform).lower()
    normalized_chat_id = str(chat_id)
    normalized_thread_id = str(thread_id) if thread_id is not None else None
    canonical_source = _canonical_session_source(
        session_source,
        platform=normalized_platform,
        chat_id=normalized_chat_id,
        thread_id=normalized_thread_id,
    )
    if (
        not context_ready
        or canonical_source is None
        or not _valid_session_key(
            session_key,
            platform=normalized_platform,
            chat_type=str(canonical_source["chat_type"]),
        )
    ):
        return []
    decision_id = secrets.token_hex(8)
    new_records = [
        DeferredDecisionRecord(
            job_id=job_id,
            decision_id=decision_id,
            job_name=job_name,
            card_index=index,
            platform=normalized_platform,
            chat_id=normalized_chat_id,
            thread_id=normalized_thread_id,
            user_id=str(user_id) if user_id is not None else None,
            question=card.question,
            choices=card.choices,
            created_at=now.isoformat(),
            expires_at=(now + DECISION_TTL).isoformat(),
            context_ready=bool(context_ready),
            session_source=dict(canonical_source),
            session_key=str(session_key),
        )
        for index, card in enumerate(cards)
    ]
    with _locked_store() as path:
        existing = _prune_records(_read_records(path), now=now)
        if len(existing) + len(new_records) > MAX_PENDING_RECORDS:
            _write_records(path, existing)
            return []
        _write_records(path, existing + new_records)
    return new_records


def _record_identity(record: DeferredDecisionRecord) -> tuple[str, str, int]:
    return record.job_id, record.decision_id, record.card_index


def _bind_message_identity(
    identity: tuple[str, str, int], message_id: Any
) -> bool:
    normalized_message_id = str(message_id) if message_id is not None else ""
    if _valid_bounded_line(normalized_message_id, maximum=256) is None:
        return False
    with _locked_store() as path:
        records = _prune_records(_read_records(path))
        for index, current in enumerate(records):
            if _record_identity(current) != identity:
                continue
            if current.message_id not in (None, normalized_message_id):
                return False
            records[index] = replace(current, message_id=normalized_message_id)
            _write_records(path, records)
            return True
    return False


def bind_message(record: DeferredDecisionRecord, message_id: Any) -> bool:
    """Bind one exposed card to the exact platform message that owns its buttons."""
    return _bind_message_identity(_record_identity(record), message_id)


def bind_message_by_identity(
    *,
    job_id: str,
    decision_id: str,
    card_index: int,
    message_id: Any,
) -> bool:
    """Adapter-facing message binder used before a send coroutine completes."""
    if (
        not _JOB_ID_RE.fullmatch(str(job_id))
        or not _DECISION_ID_RE.fullmatch(str(decision_id))
        or type(card_index) is not int
        or not 0 <= card_index < MAX_CARDS
    ):
        return False
    return _bind_message_identity(
        (str(job_id), str(decision_id), card_index), message_id
    )


def discard_records(records_to_discard: Sequence[DeferredDecisionRecord]) -> int:
    """Revoke unclaimed callbacks for cards that were not actually exposed."""
    identities = {_record_identity(record) for record in records_to_discard}
    if not identities:
        return 0
    with _locked_store() as path:
        records = _prune_records(_read_records(path))
        remaining = [
            record
            for record in records
            if _record_identity(record) not in identities or record.claimed
        ]
        removed = len(records) - len(remaining)
        if removed:
            _write_records(path, remaining)
        return removed


def claim_choice(
    *,
    job_id: str,
    decision_id: str,
    card_index: int,
    choice_index: int,
    platform: str,
    chat_id: str,
    thread_id: Optional[str],
    user_id: Optional[str],
    message_id: Optional[str] = None,
) -> Optional[ClaimedChoice]:
    """Atomically lease one canonical choice to exactly one dispatcher."""
    if not _JOB_ID_RE.fullmatch(str(job_id)):
        return None
    if not _DECISION_ID_RE.fullmatch(str(decision_id)):
        return None
    if type(card_index) is not int or type(choice_index) is not int:
        return None
    if not 0 <= card_index < MAX_CARDS or not 0 <= choice_index < MAX_CHOICES:
        return None
    normalized_thread = str(thread_id) if thread_id is not None else None
    normalized_user = str(user_id) if user_id is not None else None
    normalized_message = str(message_id) if message_id is not None else None
    now = datetime.now(timezone.utc)
    with _locked_store() as path:
        records = _prune_records(_read_records(path), now=now)
        for index, record in enumerate(records):
            if (
                record.job_id != job_id
                or record.decision_id != decision_id
                or record.card_index != card_index
            ):
                continue
            if (
                record.claimed
                or not record.context_ready
                or record.platform != str(platform).lower()
                or record.chat_id != str(chat_id)
                or record.thread_id != normalized_thread
                or record.user_id != normalized_user
                or record.message_id is None
                or record.message_id != normalized_message
                or choice_index >= len(record.choices)
            ):
                return None
            claim_token = secrets.token_hex(16)
            claimed_record = replace(
                record,
                claimed=True,
                claim_token=claim_token,
                claim_expires_at=(now + CLAIM_LEASE).isoformat(),
            )
            records[index] = claimed_record
            _write_records(path, records)
            return ClaimedChoice(
                claimed_record,
                record.choices[choice_index],
                choice_index,
                claim_token,
            )
    return None


def _finish_claim(claimed: ClaimedChoice, *, consume: bool) -> bool:
    identity = _record_identity(claimed.record)
    with _locked_store() as path:
        records = _prune_records(_read_records(path))
        for index, record in enumerate(records):
            if _record_identity(record) != identity:
                continue
            if not record.claimed or record.claim_token != claimed.claim_token:
                return False
            if consume:
                del records[index]
            else:
                records[index] = replace(
                    record,
                    claimed=False,
                    claim_token=None,
                    claim_expires_at=None,
                )
            _write_records(path, records)
            return True
    return False


def acknowledge_choice(claimed: ClaimedChoice) -> bool:
    """Consume a leased choice only after its event was successfully enqueued."""
    return _finish_claim(claimed, consume=True)


def release_choice(claimed: ClaimedChoice) -> bool:
    """Return a failed or cancelled dispatch lease to pending."""
    return _finish_claim(claimed, consume=False)


def _reset_for_tests() -> None:
    """Compatibility hook: all authoritative state is already disk-backed."""
    return None
