"""Inline-button questions in cron reports (issue #107138).

Cron reports regularly end with a decision the user has to make ("do I merge PR
#1827?"). The agent cannot call ``clarify`` during a cron run (it blocks the run
awaiting input), so today those decisions arrive as dead text. This module lets
a job's final response carry the decision as a documented ``<question>`` block
that the DELIVERY layer turns into inline buttons: the run finishes without
blocking, and the answer arrives whenever the user taps.

Markup contract (one block per question, options are ``-``/``*``/``•``/numbered
list items)::

    <question>
    Je merge la PR #1827 ?
    - ✅ Oui (Recommended)
    - ⏸️ Plus tard
    </question>

The parser is lossless by construction: a block that does not match the contract
(no options, too many options, a prose line among the options, an unterminated
tag) is left VERBATIM in the delivered text. A malformed block therefore degrades
to today's dead text instead of swallowing part of the report.

The pending-question store is profile-local SQLite next to the executions ledger
(same connection/pragma pattern as ``cron/notepad.py``) because the tap can land
days later, from a gateway that restarted in between. Recording an answer is
first-write-wins so a double tap cannot re-inject the same turn twice.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from cron.ledger import ledger_transaction, open_ledger, prepare_ledger
from hermes_constants import get_hermes_home
from hermes_time import now as _hermes_now

# Optional test override; production resolves the path at transaction time so a
# multiplexed profile cannot read another profile's pending questions.
QUESTIONS_FILE: Optional[Path] = None

# Telegram caps callback_data at 64 bytes; "cq:<12-hex token>:<index>" is ~18.
CALLBACK_PREFIX = "cq:"
MAX_CALLBACK_DATA_BYTES = 64
# A question is a decision, not a menu: past this the block stays verbatim text.
MAX_OPTIONS = 8
MAX_QUESTION_CHARS = 500
TOKEN_HEX_CHARS = 12
MAX_PENDING_ROWS = 500
# A button nobody tapped within this window loses its context: the row goes, and a
# later tap is answered with "no longer available" (bounded growth for unattended jobs).
MAX_PENDING_AGE_DAYS = 90

_lock = threading.RLock()

_BLOCK_RE = re.compile(r"<question>(.*?)</question>", re.DOTALL | re.IGNORECASE)
_OPTION_RE = re.compile(r"^(?:[-*\u2022]|\d+[.)])\s+(?P<label>.+)$")
# Models bold the recommended option but not the others; keep labels comparable.
_EMPHASIS_WRAPPERS = (("**", "**"), ("__", "__"), ("`", "`"))
_RECOMMENDED_RE = re.compile(
    r"\s*[\(\[]\s*(?:recommended|recommand\u00e9|empfohlen|recomendado)\s*[\)\]]\s*$",
    re.IGNORECASE,
)

QUESTION_MARKUP = "<question>"


@dataclass
class QuestionOption:
    """One tappable answer. ``label`` is both the button text and the injected answer."""

    label: str
    recommended: bool = False


@dataclass
class Question:
    """One parsed ``<question>`` block."""

    text: str
    options: List[QuestionOption] = field(default_factory=list)


@dataclass
class PendingQuestion:
    """A parsed question with the store token a tap will resolve (the adapter wire shape)."""

    token: str
    question: str
    options: List[str] = field(default_factory=list)
    recommended_index: Optional[int] = None


def to_payload(questions: List[Question], tokens: List[str]) -> List[PendingQuestion]:
    """Pair parsed questions with their tokens for the adapter."""
    return [
        PendingQuestion(
            token=token,
            question=question.text,
            options=[option.label for option in question.options],
            recommended_index=next(
                (i for i, option in enumerate(question.options) if option.recommended), None),
        )
        for question, token in zip(questions, tokens)
    ]


def _strip_emphasis(value: str) -> str:
    label = value.strip()
    for opening, closing in _EMPHASIS_WRAPPERS:
        if label.startswith(opening) and label.endswith(closing) and len(label) > len(opening) + len(closing):
            return label[len(opening):-len(closing)].strip()
    return label


def _parse_block(body: str) -> Optional[Question]:
    """Parse one block body; None when it does not match the contract."""
    lines = [line.strip() for line in body.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return None
    question_text = _strip_emphasis(lines[0])
    if not question_text or len(question_text) > MAX_QUESTION_CHARS:
        return None
    options: List[QuestionOption] = []
    for line in lines[1:]:
        match = _OPTION_RE.match(line)
        if match is None:
            # A prose line among the options is not a question block: keep the text as-is.
            return None
        label = _strip_emphasis(match.group("label"))
        if not label:
            return None
        recommended = bool(_RECOMMENDED_RE.search(label))
        label = _RECOMMENDED_RE.sub("", label).strip()
        if not label:
            return None
        options.append(QuestionOption(label=label, recommended=recommended))
    if not options or len(options) > MAX_OPTIONS:
        return None
    return Question(text=question_text, options=options)


def parse_questions(text: str) -> Tuple[str, List[Question]]:
    """Extract ``<question>`` blocks: ``(text_without_them, questions)``.

    Returns the input unchanged with an empty list when there is nothing to
    parse, so callers can branch on ``questions`` alone.
    """
    raw = "" if text is None else str(text)
    if QUESTION_MARKUP not in raw.lower():
        return raw, []
    questions: List[Question] = []
    spans: List[Tuple[int, int]] = []
    for match in _BLOCK_RE.finditer(raw):
        question = _parse_block(match.group(1))
        if question is None:
            continue
        questions.append(question)
        spans.append(match.span())
    if not spans:
        return raw, []
    body = raw
    for start, end in reversed(spans):
        body = body[:start] + body[end:]
    # The blocks leave their surrounding blank lines behind; collapse them so the
    # delivered report does not gain a gap where the buttons used to be.
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip("\n"), questions


def render_questions_text(questions: List[PendingQuestion], *, hint: bool = True) -> str:
    """Platform-agnostic rendering (button-less adapters and the text fallback)."""
    blocks = []
    for index, question in enumerate(questions, start=1):
        head = f"\u2753 {question.question}" if len(questions) == 1 else f"\u2753 {index}. {question.question}"
        options = "\n".join(
            f"  {position}. {label}"
            for position, label in enumerate(question.options, start=1)
        )
        blocks.append(f"{head}\n{options}")
    body = "\n\n".join(blocks)
    if hint and len(questions) == 1:
        return (
            f"{body}\n\nReply with the number or the option text to answer "
            "\u2014 your answer continues the conversation."
        )
    return body


def build_callback_data(token: str, index: int) -> str:
    """``cq:<token>:<index>``, the tap payload (callback_data size is the constraint)."""
    return f"{CALLBACK_PREFIX}{token}:{int(index)}"


def parse_callback_data(data: str) -> Optional[Tuple[str, int]]:
    """Inverse of :func:`build_callback_data`; None for anything else."""
    if not data or not data.startswith(CALLBACK_PREFIX):
        return None
    remainder = data[len(CALLBACK_PREFIX):]
    token, separator, raw_index = remainder.rpartition(":")
    if not separator or not token or len(data.encode("utf-8")) > MAX_CALLBACK_DATA_BYTES:
        return None
    try:
        return token, int(raw_index)
    except ValueError:
        return None


def button_rows(questions: List[PendingQuestion]) -> List[List[Dict[str, str]]]:
    """Button rows in the shape ``telegram_notification_markup`` consumes.

    One option per row (labels are the answers, and a multi-option row truncates
    on mobile); the question number prefixes the label when a report asks more
    than one thing, so a row is never ambiguous.
    """
    rows: List[List[Dict[str, str]]] = []
    for question_index, question in enumerate(questions):
        for option_index, label in enumerate(question.options):
            text = label if len(questions) == 1 else f"{question_index + 1}. {label}"
            rows.append([{
                "label": text,
                "callback_data": build_callback_data(question.token, option_index),
            }])
    return rows


def answers_prompt(questions: List[PendingQuestion]) -> str:
    """Intro line above the questions in the delivered button message."""
    if len(questions) == 1:
        return "\u2753 This report has a question for you:"
    return f"\u2753 This report has {len(questions)} questions for you:"


# -- pending-question store ---------------------------------------------------


def _current_file() -> Path:
    return QUESTIONS_FILE or (get_hermes_home().resolve() / "cron" / "questions.db")


def _connect() -> sqlite3.Connection:
    return open_ledger(_current_file())


def _initialize_schema(conn: sqlite3.Connection) -> None:
    prepare_ledger(conn, db_label="cron/questions.db", synchronous_full=False)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS cron_questions (
             token TEXT PRIMARY KEY,
             job_id TEXT NOT NULL,
             platform TEXT NOT NULL,
             chat_id TEXT NOT NULL,
             question TEXT NOT NULL,
             options_json TEXT NOT NULL,
             created_at TEXT NOT NULL,
             answered_at TEXT,
             answer_index INTEGER,
             answer_text TEXT
           )"""
    )


@contextmanager
def _transaction() -> Iterator[sqlite3.Connection]:
    with ledger_transaction(_lock, _connect, _initialize_schema) as conn:
        yield conn


def _prune_unlocked(conn: sqlite3.Connection) -> None:
    """Bounded retention: a job whose questions are never answered must not grow forever.

    Answered rows past the cap go first (they carry no more information than the
    injected answer turn), then unanswered rows older than ``MAX_PENDING_AGE_DAYS``
    — a tap that late has no context left to continue from anyway.
    """
    answered = int(
        conn.execute(
            "SELECT COUNT(*) FROM cron_questions WHERE answered_at IS NOT NULL"
        ).fetchone()[0]
    )
    excess = answered - MAX_PENDING_ROWS
    if excess > 0:
        conn.execute(
            """DELETE FROM cron_questions WHERE token IN (
                 SELECT token FROM cron_questions WHERE answered_at IS NOT NULL
                 ORDER BY answered_at, token LIMIT ?)""",
            (excess,),
        )
    cutoff = (_hermes_now() - timedelta(days=MAX_PENDING_AGE_DAYS)).isoformat()
    conn.execute(
        "DELETE FROM cron_questions WHERE answered_at IS NULL AND created_at < ?",
        (cutoff,),
    )


def record_questions(
    job_id: str, platform: str, chat_id: str, questions: List[Question]
) -> List[str]:
    """Persist one pending row per question; returns tokens in question order.

    Called BEFORE the button message is sent, so a tap can never arrive for a
    token the store does not know yet.
    """
    if not questions:
        return []
    tokens = [
        uuid.uuid4().hex[:TOKEN_HEX_CHARS] for _ in questions
    ]
    now = _hermes_now().isoformat()
    with _transaction() as conn:
        for token, question in zip(tokens, questions):
            conn.execute(
                """INSERT INTO cron_questions
                   (token, job_id, platform, chat_id, question, options_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    token,
                    str(job_id),
                    str(platform),
                    str(chat_id),
                    question.text,
                    json.dumps([option.label for option in question.options], ensure_ascii=False),
                    now,
                ),
            )
        _prune_unlocked(conn)
    return tokens


def get_question(token: str) -> Optional[Dict[str, Any]]:
    with _transaction() as conn:
        row = conn.execute(
            "SELECT * FROM cron_questions WHERE token=?", (str(token),)
        ).fetchone()
    return None if row is None else dict(row)


def forget_questions(tokens: List[str]) -> int:
    """Drop pending rows whose buttons provably never reached the platform."""
    if not tokens:
        return 0
    with _transaction() as conn:
        cursor = conn.executemany(
            "DELETE FROM cron_questions WHERE token=? AND answered_at IS NULL",
            [(str(token),) for token in tokens],
        )
        return cursor.rowcount if cursor.rowcount is not None else 0


def claim_answer(token: str, index: int) -> Dict[str, Any]:
    """Claim a tap; first write wins. ``status`` is one of:

    ``answered`` (this tap recorded the answer), ``already_answered`` (a previous
    tap did, or the same tap was replayed), ``unknown`` (expired/pruned token),
    ``invalid_option`` (index outside the recorded options).
    """
    token = str(token)
    now = _hermes_now().isoformat()
    with _transaction() as conn:
        row = conn.execute(
            "SELECT * FROM cron_questions WHERE token=?", (token,)
        ).fetchone()
        if row is None:
            return {"status": "unknown"}
        record = dict(row)
        options = json.loads(record["options_json"])
        if not 0 <= int(index) < len(options):
            return {
                "status": "invalid_option",
                "question": record["question"],
                "options": options,
            }
        if record["answered_at"] is not None:
            return {
                "status": "already_answered",
                "question": record["question"],
                "answer_text": record["answer_text"],
                "job_id": record["job_id"],
            }
        answer_text = options[int(index)]
        cursor = conn.execute(
            """UPDATE cron_questions SET answered_at=?, answer_index=?, answer_text=?
               WHERE token=? AND answered_at IS NULL""",
            (now, int(index), answer_text, token),
        )
        if cursor.rowcount != 1:  # raced by another tap: that one owns the answer
            fresh = conn.execute(
                "SELECT * FROM cron_questions WHERE token=?", (token,)
            ).fetchone()
            return {
                "status": "already_answered",
                "question": record["question"],
                "answer_text": fresh["answer_text"] if fresh else None,
                "job_id": record["job_id"],
            }
        _prune_unlocked(conn)
    return {
        "status": "answered",
        "token": token,
        "question": record["question"],
        "answer_text": answer_text,
        "job_id": record["job_id"],
        "platform": record["platform"],
        "chat_id": record["chat_id"],
    }


def clear_for_job(job_id: str) -> int:
    """Delete a removed job's pending questions; no-op without creating the DB."""
    if not _current_file().exists():
        return 0
    with _transaction() as conn:
        cursor = conn.execute("DELETE FROM cron_questions WHERE job_id=?", (str(job_id),))
    return cursor.rowcount


def answer_reply_text(question: str, answer: str) -> str:
    """The user turn injected back into the conversation when a button is tapped."""
    return (
        "You asked this question in a scheduled report:\n\n"
        f"Q: {question}\n"
        f"A: {answer}"
    )
