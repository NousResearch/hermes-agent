"""Email platform adapter for the Hermes gateway: users talk to Hermes by sending email; IMAP (polled)
receives, SMTP sends. Configured via EMAIL_* env vars or ``platforms.email`` in config.yaml (see website docs).
Cron deliveries can open mail threads (``create_handoff_thread``), and replies are routed per thread (see the user guide)."""

import asyncio
import email as email_lib
from contextlib import contextmanager, suppress
from functools import partial
import hashlib
import imaplib
import json
import logging
import os
import re
import smtplib
import socket
import ssl
import threading
import time
import uuid
from email.header import decode_header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.parser import BytesHeaderParser
from email.utils import formatdate, parseaddr
from email import encoders
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent.async_utils import safe_schedule_threadsafe
from agent.i18n import t
from gateway.platforms.base import (
    BasePlatformAdapter, SendResult,
    cache_document_from_bytes, cache_image_from_bytes,
)
from gateway.platforms.helpers import cancel_task
from gateway.platforms.event import MessageEvent, MessageType
from gateway.config import Platform, PlatformConfig
from plugins.plugin_storage import plugin_data_dir
from utils import is_truthy_value
from gateway.platforms._shared import get_scoped_secret as _get_secret, coerce_port, decode_json_list_literal, send_error
from gateway.platforms.send_targets import StandaloneSendPolicy, regex_stripped_parser

logger = logging.getLogger(__name__)

_SECURITY_ALIASES = {"tls": "tls", "ssl": "tls", "implicit": "tls", "starttls": "starttls", "plain": "plain", "none": "plain"}
# Automated senders (address substrings / bulk-mail headers) are silently ignored.
_NOREPLY_PATTERNS = ("noreply", "no-reply", "no_reply", "donotreply", "do-not-reply", "mailer-daemon", "postmaster",
                     "bounce", "notifications@", "automated@", "auto-confirm", "auto-reply", "automailer")
_AUTOMATED_HEADERS = {"Auto-Submitted": lambda v: v.lower() != "no",
                      "Precedence": lambda v: v.lower() in {"bulk", "list", "junk"},
                      "X-Auto-Response-Suppress": lambda v: bool(v), "List-Unsubscribe": lambda v: bool(v)}
MAX_MESSAGE_LENGTH = 50_000  # Gmail-safe max length per email body
SMTP_CONNECT_TIMEOUT = 30
_TRUTHY = {"true", "1", "yes"}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
# Charset labels seen in the wild that Python's codec registry doesn't know: "unknown-8bit"/"x-unknown" are
# RFC 1428 placeholders (QQ Mail emits them); gb2312/gbk map to the gb18030 superset so GBK extensions decode.
_CHARSET_ALIASES = {"unknown-8bit": "utf-8", "unknown": "utf-8", "x-unknown": "utf-8", "default": "utf-8",
                    "ansi_x3.110-1983": "latin-1", "cp-850": "cp850",
                    "gb2312": "gb18030", "gbk": "gb18030", "ks_c_5601-1987": "cp949"}
# Ordered (pattern, replacement) substitutions for _strip_html.
_HTML_SUBS = ((re.compile(r"<br\s*/?>", re.IGNORECASE), "\n"), (re.compile(r"<p[^>]*>", re.IGNORECASE), "\n"),
              (re.compile(r"</p>", re.IGNORECASE), "\n"), (re.compile(r"<[^>]+>"), ""), (re.compile(r"&nbsp;"), " "),
              (re.compile(r"&amp;"), "&"), (re.compile(r"&lt;"), "<"), (re.compile(r"&gt;"), ">"), (re.compile(r"\n{3,}"), "\n\n"))
# ``display <bracketed>`` split for the _extract_email_address fallback (linear: neither part can match the other's delimiters).
_SINGLE_BRACKET_FROM_RE = re.compile(r'([^"<>]*)<([^<>\s]+)>\s*')
_COMMENT_RE = re.compile(r"\([^()]*\)")
# Longest From: value we parse. parseaddr is pure Python and superlinear on hostile input (~1s at 100KB, GIL held);
# a real mailbox plus display name stays far below this (RFC 5322 caps a line at 998 chars).
_MAX_FROM_LEN = 2048
# Only these headers are needed to reject a sender. Request one byte past the
# limit so a truncated preflight fails closed without downloading the MIME body.
_MAX_PREAUTH_HEADER_BYTES = 64 * 1024
_PREAUTH_FETCH = (
    f"(BODY.PEEK[HEADER.FIELDS (FROM AUTHENTICATION-RESULTS {' '.join(h.upper() for h in _AUTOMATED_HEADERS)})]"
    f"<0.{_MAX_PREAUTH_HEADER_BYTES + 1}>)"
)
# Authentication-Results clause head (``dmarc=pass``), matched only at the start of a clause.
_AUTH_METHOD_RE = re.compile(r"\s*(dmarc|dkim|spf)\s*=\s*([a-z]+)", re.IGNORECASE)
_NO_AUTH_RESULTS_REASON = "no Authentication-Results header"
_UNTRUSTED_AUTHSERV_REASON = "no Authentication-Results from trusted authserv-id"
_MISSING_AUTHSERV_REASON = "authserv-id is not configured; refusing to trust Authentication-Results"
_OPT_OUT_HINT = "set platforms.email.require_authenticated_sender: false (or EMAIL_TRUST_FROM_HEADER=true) to accept the risk."
# Operator-fixable reasons a granted sender's mail fails authentication, and the fix each log line names.
_DROP_HINTS = {
    _NO_AUTH_RESULTS_REASON: " If your mail server does not stamp Authentication-Results, " + _OPT_OUT_HINT,
    _UNTRUSTED_AUTHSERV_REASON: " Check that platforms.email.authserv_id (EMAIL_AUTHSERV_ID) names your mail server.",
}
# A missing pin is account config, not one sender's mail, so connect() names its fix once per account.
_MISSING_AUTHSERV_HINT = (" Set EMAIL_AUTHSERV_ID (or platforms.email.authserv_id) to the receiving MTA's exact authserv-id, "
                          "or " + _OPT_OUT_HINT)
# One token of a clause: a property we read (``header.from=x``; the value may be or contain a quoted-string), or
# any other whitespace-delimited token consumed whole, so text inside quotes or other values is never read as a prop.
_QUOTED = r'"(?:[^"\\]|\\.)*"'
_AUTH_PROP_RE = re.compile(rf'(header\.from|header\.d|smtp\.mailfrom|smtp\.from|envelope-from)\s*=\s*((?:{_QUOTED}|[^\s";])+)'
                           rf'|(?:{_QUOTED}|[^\s"])+', re.IGNORECASE)
# A leading "Subject: ..." line (optionally after a "[CRON_FAILURE]" marker line) names the mail and leaves the body.
_SUBJECT_LINE_RE = re.compile(r"\A(?:\s*\[CRON_FAILURE\][ \t\r]*\n)?\s*Subject:[ \t]*([^\n]*\S)[ \t\r]*(?:\n|\Z)", re.IGNORECASE)
_MAX_SUBJECT_LEN = 250
_THREAD_NAME_PREFIX = "Hermes — "  # cron's handoff thread name: "Hermes — <job name>"
_MD_RENDERER: Any = None  # None = not tried yet, False = markdown-it-py unavailable


def _split_subject(content: str, fallback: str) -> tuple[str, str]:
    """``(subject, body)``: subject from a leading ``Subject:`` line (stripped from the body), else *fallback*."""
    m = _SUBJECT_LINE_RE.match(content or "")
    if not m:
        return fallback, content
    subject = re.sub(r"[\r\x00-\x1f\x7f]", "", m.group(1)).strip()[:_MAX_SUBJECT_LEN]
    return (subject, content[m.end():].lstrip("\n")) if subject else (fallback, content)


def _subject_from_thread_name(name: Optional[str]) -> str:
    """Job name from cron's ``"Hermes — <job name>"`` thread name; ``"Hermes Agent"`` when unknown."""
    name = (name or "").strip()
    if name.startswith(_THREAD_NAME_PREFIX):
        name = name[len(_THREAD_NAME_PREFIX):].strip()
    return name[:_MAX_SUBJECT_LEN] or "Hermes Agent"


def _render_html(text: str) -> Optional[str]:
    """Markdown -> HTML (raw HTML escaped, images off, ``javascript:`` links refused); ``None`` when markdown-it-py is missing."""
    global _MD_RENDERER
    if _MD_RENDERER is None:
        try:
            from markdown_it import MarkdownIt
            _MD_RENDERER = MarkdownIt("js-default", {"html": False}).disable("image")
        except ImportError:
            logger.warning("[Email] html is on but markdown-it-py is not installed; sending plain text only")
            _MD_RENDERER = False
    if not _MD_RENDERER:
        return None
    return f'<html><head><meta charset="utf-8"></head><body>\n{_MD_RENDERER.render(text)}</body></html>'


def _attach_body(msg: MIMEMultipart, body: str, html: bool) -> None:
    """Attach *body* as text/plain, or as multipart/alternative (plain, then rendered HTML) when *html*."""
    rendered = _render_html(body) if html and body else None
    if rendered is None:
        msg.attach(MIMEText(body, "plain", "utf-8"))
        return
    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(body, "plain", "utf-8"))
    alt.attach(MIMEText(rendered, "html", "utf-8"))
    msg.attach(alt)


_MSGID_RE = re.compile(r"<[^<>\s]+>")
_ROOT_KEY_MAX = 120
_TOPIC_ID_MAX = 200  # a thread's root Message-ID as shown in the session topic
_THREAD_TTL_SECONDS = 90 * 86400  # thread roots (and the ids indexed to them) older than this are pruned on load
_INDEX_MAX = 5000  # message-id -> root entries kept; the oldest go first
_REFERENCES_MAX = 20  # ids in an outgoing References header: the root plus the newest 19
_SENT_BACKFILL_MAX = 200  # Sent-folder copies read back on the first poll (the newest ones)
_SENT_FOLDER_NAMES = ("Sent", "Sent Items", "Sent Messages", "Sent Mail", "INBOX.Sent")  # without SPECIAL-USE
_SENT_HEADERS_FETCH = "(UID BODY.PEEK[HEADER.FIELDS (MESSAGE-ID IN-REPLY-TO REFERENCES)])"
_LIST_LINE_RE = re.compile(r'^\((?P<flags>[^)]*)\)\s+(?:"(?:[^"\\]|\\.)*"|NIL)\s+(?P<name>.+)$', re.IGNORECASE)
_FETCH_UID_RE = re.compile(rb"\bUID\s+(\d+)", re.IGNORECASE)


def _message_ids(header: Optional[str]) -> list[str]:
    """``<id>`` tokens of a Message-ID / In-Reply-To / References header, in order (folding and junk ignored)."""
    return _MSGID_RE.findall(header or "")


def _root_key(raw: str) -> str:
    """Session-key-safe thread root from a Message-ID: no ``<>``; anything with whitespace, ``..``, ``/``, ``\\``,
    ``:`` (session keys are split on it), or over 120 chars is replaced by ``h-`` + 24 hex of its SHA-256."""
    key = (raw or "").strip().strip("<>").strip()
    unsafe = (not key or len(key) > _ROOT_KEY_MAX or ".." in key or "/" in key or "\\" in key or ":" in key
              or re.search(r"\s", key))
    return "h-" + hashlib.sha256((raw or "").strip().encode("utf-8", "replace")).hexdigest()[:24] if unsafe else key


def _shorten(text: str, limit: int = _TOPIC_ID_MAX) -> str:
    """*text* cut to *limit* characters with a trailing ``…`` (a Message-ID shown in a prompt has no length limit)."""
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _thread_key(metadata: Optional[dict[str, Any]]) -> Optional[str]:
    """The thread root a send's ``metadata["thread_id"]`` names, or ``None`` (a flat reply)."""
    return str(metadata["thread_id"]) if metadata and metadata.get("thread_id") else None


def _bare_id(message_id: Optional[str]) -> str:
    """A Message-ID without surrounding whitespace and ``<>`` (the store's index key)."""
    return (message_id or "").strip().strip("<>").strip()


def _quote_mailbox(name: str) -> str:
    """A mailbox name as an IMAP quoted string (imaplib sends arguments as they are)."""
    return '"' + name.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _find_sent_folder(imap: Any) -> Optional[str]:
    """The Sent mailbox: the one LIST flags ``\\Sent`` (RFC 6154), else one with a common Sent name, else ``None``."""
    status, lines = imap.list()
    if status != "OK":
        return None
    by_name: dict[str, str] = {}
    for line in lines or ():
        if not isinstance(line, (bytes, bytearray)):
            continue  # a name sent as a literal: not worth the parsing
        match = _LIST_LINE_RE.match(bytes(line).decode("utf-8", "replace").strip())
        if not match:
            continue
        name = match["name"].strip()
        if len(name) >= 2 and name[0] == name[-1] == '"':
            name = re.sub(r"\\(.)", r"\1", name[1:-1])
        if "\\sent" in match["flags"].lower().split():
            return name
        by_name.setdefault(name.lower(), name)
    return next((by_name[n.lower()] for n in _SENT_FOLDER_NAMES if n.lower() in by_name), None)


def _sent_copies(data: Any) -> list[tuple[str, list[str], list[str]]]:
    """``(Message-ID, In-Reply-To ids, References ids)`` of each message in a header FETCH response, by UID."""
    copies = []
    for n, item in enumerate(data or ()):
        if not isinstance(item, tuple) or len(item) < 2 or not isinstance(item[1], (bytes, bytearray)):
            continue
        match = _FETCH_UID_RE.search(item[0]) if isinstance(item[0], (bytes, bytearray)) else None
        headers = email_lib.message_from_bytes(bytes(item[1]))
        own = _message_ids(str(headers.get("Message-ID") or ""))
        if own:
            copies.append((int(match.group(1)) if match else n, own[0], _message_ids(str(headers.get("In-Reply-To") or "")),
                           _message_ids(str(headers.get("References") or ""))))
    return [copy[1:] for copy in sorted(copies, key=lambda c: c[0])]


def _uid_int(uid: Any) -> int:
    """An IMAP UID (bytes or str) as an int; ``-1`` for anything unparseable, so a bad UID counts as old mail."""
    try:
        return int(uid.decode("ascii") if isinstance(uid, (bytes, bytearray)) else uid)
    except (ValueError, TypeError, AttributeError, UnicodeDecodeError):
        return -1


def _uidvalidity(imap: Any) -> Optional[str]:
    """The selected mailbox's UIDVALIDITY as a string, or ``None`` when the server sent none (or anything odd)."""
    try:
        _, items = imap.response("UIDVALIDITY")
        for item in items or ():
            if isinstance(item, (bytes, bytearray)):
                item = bytes(item).decode("ascii")
            if isinstance(item, str) and item.strip():
                return item.strip()
    except (ValueError, TypeError) as exc:  # an odd response shape, or a non-ASCII value
        logger.debug("[Email] Ignoring an unreadable UIDVALIDITY response: %s", exc)
    return None


_REPLY_PREFIX_RE = re.compile(r"^(?:\s*(?:re|fwd|fw)\s*:)+\s*", re.IGNORECASE)


def _strip_reply_prefix(subject: str) -> str:
    """A thread's subject from a reply's: leading ``Re:``/``Fwd:``/``Fw:`` (any case, repeated) removed, control
    characters dropped, capped like other subjects; ``"(no subject)"`` when nothing is left."""
    subject = re.sub(r"[\r\x00-\x1f\x7f]", " ", subject or "")
    return _REPLY_PREFIX_RE.sub("", subject).strip()[:_MAX_SUBJECT_LEN] or "(no subject)"


class _EmailThreadStore:
    """Persisted thread roots for the email adapter (``plugin_data_dir("email")/threads.json``).

    Schema ``{"version": 1, "roots": {key: {header, name, subject, job_id, chat_id, pending, created}},
    "index": {bare message-id: key}, "mailboxes": {address: {uidvalidity, uid_mark}}}``. A missing file is
    empty; a corrupt one is reset to empty with a warning. Roots older than 90 days are pruned on load. Every
    mutation writes through atomically (tmp file + ``os.replace``) under one lock, as the IMAP poll and the SMTP
    sends run in different executor threads. A failed write is a warning; the in-memory state stays."""

    _ROOT_FIELDS = {"header": str, "name": str, "subject": str, "job_id": str, "chat_id": str}

    def __init__(self, path: Path):
        self._path = path
        self._lock = threading.Lock()
        self._roots: dict[str, dict[str, Any]] = {}
        self._index: dict[str, str] = {}
        self._mailboxes: dict[str, dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        """(Re)read the file: missing → empty; corrupt/unreadable → empty + warning; off-schema entries dropped."""
        with self._lock:
            self._roots, self._index, self._mailboxes = {}, {}, {}
            if not self._path.exists():
                return
            try:
                raw = self._path.read_text(encoding="utf-8-sig")
                data = json.loads(raw) if raw.strip() else {}
                if not isinstance(data, dict):
                    raise ValueError(f"expected an object, got {type(data).__name__}")
            except (ValueError, OSError) as exc:  # json.JSONDecodeError is a ValueError
                logger.warning("[Email] thread store at %s is corrupt or unreadable; starting fresh: %s", self._path, exc)
                return
            cutoff = time.time() - _THREAD_TTL_SECONDS
            roots = data.get("roots")
            for key, root in (roots.items() if isinstance(roots, dict) else ()):
                if not isinstance(key, str) or not key or not isinstance(root, dict):
                    continue
                created = root.get("created")
                if not isinstance(created, (int, float)) or isinstance(created, bool) or created < cutoff:
                    continue
                clean = {f: root.get(f) if isinstance(root.get(f), t) else "" for f, t in self._ROOT_FIELDS.items()}
                clean.update(pending=root.get("pending") is True, created=float(created))
                self._roots[key] = clean
            index = data.get("index")
            for mid, key in (index.items() if isinstance(index, dict) else ()):
                if isinstance(mid, str) and mid and isinstance(key, str) and key in self._roots:
                    self._index[mid] = key
            self._cap_index()
            boxes = data.get("mailboxes")
            for address, state in (boxes.items() if isinstance(boxes, dict) else ()):
                if isinstance(address, str) and address and isinstance(state, dict):
                    validity, mark = state.get("uidvalidity"), state.get("uid_mark")
                    self._mailboxes[address.lower()] = {
                        "uidvalidity": validity if isinstance(validity, str) else None,
                        "uid_mark": mark if isinstance(mark, int) and not isinstance(mark, bool) else None}

    def _cap_index(self) -> None:
        """Drop the oldest (insertion-order) index entries over ``_INDEX_MAX``. Caller holds the lock."""
        excess = len(self._index) - _INDEX_MAX
        if excess > 0:
            for mid in list(self._index)[:excess]:
                del self._index[mid]

    def _save(self) -> None:
        """Atomic write-through; failure is a warning (in-memory state is kept). Caller holds the lock."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({"version": 1, "roots": self._roots, "index": self._index,
                                       "mailboxes": self._mailboxes}, separators=(",", ":")), encoding="utf-8")
            os.replace(tmp, self._path)
        except OSError as exc:
            logger.warning("[Email] could not persist thread store to %s: %s", self._path, exc)

    def _set_index(self, message_id: Optional[str], key: str) -> None:
        """Index *message_id* to *key*, newest last. Caller holds the lock."""
        if mid := _bare_id(message_id):
            self._index.pop(mid, None)
            self._index[mid] = key
            self._cap_index()

    def _index_new(self, message_id: Optional[str], key: str) -> bool:
        """Index *message_id* to *key* unless it already maps to another live root: a mail id never moves to a
        different thread once known (a resent copy, a client that reuses ids, a crafted id naming a root's own
        header). Only the Sent-folder learner re-points ids. Caller holds the lock; True when indexed."""
        current = self._index.get(_bare_id(message_id))
        if current and current != key and current in self._roots:
            return False
        self._set_index(message_id, key)
        return True

    def add_pending_root(self, key: str, header: str, name: str, chat_id: str) -> None:
        """Record a minted root that has not been sent yet (its first send goes out as a fresh mail)."""
        with self._lock:
            self._roots[key] = {"header": header, "name": name or "", "subject": "", "job_id": "",
                                "chat_id": chat_id or "", "pending": True, "created": time.time()}
            self._set_index(header, key)
            self._save()

    def get_root(self, key: Optional[str]) -> Optional[dict[str, Any]]:
        """A copy of the root entry, or ``None``."""
        with self._lock:
            root = self._roots.get(key or "")
            return dict(root) if root else None

    def is_pending(self, key: Optional[str]) -> bool:
        with self._lock:
            return bool(self._roots.get(key or "", {}).get("pending"))

    def mark_sent(self, key: str, subject: str, job_id: str) -> None:
        """The root's first mail went out: no longer pending; keep its subject and (non-empty) job id."""
        with self._lock:
            if (root := self._roots.get(key)) is None:
                return
            root.update(pending=False, subject=subject or root.get("subject", ""))
            if job_id:
                root["job_id"] = job_id
            self._save()

    def remember_root(self, key: str, header: str, subject: str) -> None:
        """Record a root Hermes did not mint (sent, not pending); a known key is left as it is."""
        with self._lock:
            if key in self._roots:
                return
            self._roots[key] = {"header": header or "", "name": "", "subject": subject or "", "job_id": "",
                                "chat_id": "", "pending": False, "created": time.time()}
            self._index_new(header, key)
            self._save()

    def index_message(self, message_id: Optional[str], key: str) -> None:
        """Map a message in the thread to its root (empty ids, and ids already in another live thread, are
        ignored)."""
        if not _bare_id(message_id):
            return
        with self._lock:
            if self._index_new(message_id, key):
                self._save()

    def has_roots(self) -> bool:
        with self._lock:
            return bool(self._roots)

    def learn_delivered_ids(self, copies: list[tuple[str, list[str], list[str]]]) -> int:
        """Index the Message-ID a mail was delivered under (``(id, In-Reply-To ids, References ids)`` of its copy in
        the Sent folder, oldest first) to the thread it was sent in: the root of its In-Reply-To, else of the first
        known References id. Some providers replace the Message-ID the adapter wrote, and a reply that names only
        the delivered id would otherwise open a new thread. The learned id wins over a root a reply minted from
        that id while it was unknown (the root and its own mail stay); an id the adapter indexed to another root
        when it sent or received the mail is left alone. Idempotent; returns the number of ids (re)indexed."""
        learned = 0
        with self._lock:
            for message_id, in_reply_to, references in copies:
                mid = _bare_id(message_id)
                if not mid:
                    continue
                key = next((k for parent in [*in_reply_to, *references] if _bare_id(parent) != mid
                            and (k := self._index.get(_bare_id(parent))) in self._roots), None)
                current = self._index.get(mid)
                if key is None or current == key:
                    continue
                if current in self._roots and current != _root_key(f"<{mid}>"):
                    continue
                if current in self._roots:
                    logger.info("[Email] Delivered id %s: replies move from thread %s to %s", mid, current, key)
                self._set_index(mid, key)
                learned += 1
            if learned:
                self._save()
        return learned

    def root_for(self, ids: list[str]) -> Optional[str]:
        """The root of the first id in *ids* (in order) that the index maps to a known root."""
        with self._lock:
            for mid in ids or ():
                if (key := self._index.get(_bare_id(mid))) and key in self._roots:
                    return key
            return None

    def uid_state(self, address: str) -> tuple[Optional[str], Optional[int]]:
        """``(uidvalidity, uid_mark)`` stored for a mailbox address, ``(None, None)`` when unknown."""
        with self._lock:
            state = self._mailboxes.get((address or "").lower(), {})
            return state.get("uidvalidity"), state.get("uid_mark")

    def set_uid_state(self, address: str, uidvalidity: Optional[str], uid_mark: Optional[int]) -> None:
        with self._lock:
            self._mailboxes[(address or "").lower()] = {"uidvalidity": uidvalidity, "uid_mark": uid_mark}
            self._save()

    def advance_uid_mark(self, address: str, uid: int) -> None:
        """Raise the mailbox's UID mark to *uid* (never lowers it). A no-op while no UIDVALIDITY is stored for the
        address: a mark without one is never trusted, so none is written."""
        with self._lock:
            state = self._mailboxes.get((address or "").lower())
            if not state or state.get("uidvalidity") is None or uid < 0:
                return
            if state.get("uid_mark") is not None and state["uid_mark"] >= uid:
                return
            state["uid_mark"] = int(uid)
            self._save()


def _esecret_int(name: str, default: int) -> int:
    """Scope-aware integer read."""
    return coerce_port(str(_get_secret(name, "")).strip() or default, default)


def _esecret_bool(name: str, default: bool = False) -> bool:
    """Scope-aware boolean read."""
    return is_truthy_value(raw, default=default) if (raw := str(_get_secret(name, "")).strip()) else default


def _normalize_security(value: Any, default: str = "tls") -> str:
    """Map to ``tls`` | ``starttls`` | ``plain``; unknown values warn and fall back to *default* (a typo never downgrades to plaintext)."""
    raw = str(value or "").strip().lower().replace("-", "").replace("_", "")
    if raw and raw not in _SECURITY_ALIASES:
        logger.warning("Unknown email security mode %r; using %r", value, default)
    return _SECURITY_ALIASES.get(raw, default)


def _tls_context(verify: bool, host: str) -> ssl.SSLContext:
    """Verified context by default; unverified only when explicitly opted out."""
    if verify:
        return ssl.create_default_context()
    if host not in ("127.0.0.1", "::1", "localhost"):
        logger.warning("TLS verification disabled for non-loopback host %s", host)
    return ssl._create_unverified_context()


def _close_imap(imap: imaplib.IMAP4) -> None:
    """Teardown that guarantees the socket closes: ``logout()`` only guards ``OSError``, so ``IMAP4.abort`` on a
    broken connection skipped ``shutdown()`` and leaked one fd per failed poll (fatal on macOS's 256 soft limit).

    ``IMAP4.logout()`` only guards against ``OSError`` internally: a broken connection makes
    ``_simple_command('LOGOUT')`` raise ``IMAP4.abort`` (which is *not* an ``OSError``), so ``logout()``
    propagates before its own ``shutdown()`` call and the TCP socket stays open. On macOS, where the default
    soft fd limit is 256 and pollers may run through a local proxy, these abandoned sockets accumulate one
    per failed poll until the gateway hits ``[Errno 24] Too many open files`` (#79889).
    """
    try:
        imap.logout()
    except Exception:
        with suppress(Exception):
            imap.shutdown()


def _create_ipv4_connection(host: str, port: int, timeout: float, source_address: Any = None) -> socket.socket:
    """``socket.create_connection`` constrained to ``AF_INET`` (no process-global socket mutation — sends run in executor threads)."""
    last_error: OSError | None = None
    for family, socktype, proto, _canonname, sockaddr in socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM):
        sock = socket.socket(family, socktype, proto)
        sock.settimeout(timeout)
        try:
            if source_address:
                sock.bind(source_address)
            sock.connect(sockaddr)
            return sock
        except OSError as exc:
            last_error = exc
            sock.close()
    raise last_error if last_error is not None else OSError(f"No IPv4 address found for {host}:{port}")


class _IPv4SMTP(smtplib.SMTP):
    def _get_socket(self, host, port, timeout):  # type: ignore[override]
        return _create_ipv4_connection(host, port, timeout, source_address=self.source_address)


class _IPv4SMTP_SSL(smtplib.SMTP_SSL):
    def _get_socket(self, host, port, timeout):  # type: ignore[override]
        return self.context.wrap_socket(_create_ipv4_connection(host, port, timeout, source_address=self.source_address), server_hostname=getattr(self, "_host", host))


def _open_smtp(host: str, port: int, security: str, ctx: ssl.SSLContext, smtp_cls: type, smtp_ssl_cls: type, **kwargs: Any) -> smtplib.SMTP:
    """Open one SMTP connection with TLS established per *security*; *kwargs* go to the constructor."""
    if security == "tls":
        return smtp_ssl_cls(host, port, context=ctx, **kwargs)
    smtp = smtp_cls(host, port, **kwargs)
    if security == "starttls":
        try:
            smtp.starttls(context=ctx)
        except Exception:
            smtp.close()
            raise
    return smtp


def _send_imap_id(imap: imaplib.IMAP4) -> None:
    """Send RFC 2971 IMAP ID: 163/NetEase require it after LOGIN (else every UID command
    returns ``BYE Unsafe Login``); other servers may reject it, so failures are swallowed.

    Sent only when the server advertises ``ID`` (RFC 2971 requires advertising it): a server
    without the extension can answer an untagged ``* BYE Unknown command.`` and close the
    connection, which imaplib cannot surface here — the failure appears one command later
    as a misleading SELECT error and the adapter retries forever (Purelymail, #39856).
    ``imap.capabilities`` is populated by imaplib at connect, so the check is free."""
    if "ID" not in imap.capabilities:
        logger.debug(
            "[Email] Server does not advertise IMAP ID capability; skipping ID"
        )
        return
    try:
        try:
            from hermes_cli.version_info import get_version_info
            version = get_version_info().base_version
        except Exception:
            version = "0"
        imap.xatom("ID", f'("name" "hermes-agent" "version" "{version}" '
                         '"vendor" "NousResearch" "support-email" "noreply@nousresearch.com")')
    except Exception as e:
        logger.debug("[Email] IMAP ID command not accepted: %s", e)


def _is_automated_sender(address: str, headers: dict) -> bool:
    """True if this email is from an automated/noreply source."""
    addr = address.lower()
    return any(pattern in addr for pattern in _NOREPLY_PATTERNS) or any(
        (value := headers.get(header, "")) and check(value) for header, check in _AUTOMATED_HEADERS.items())


def _imap_payload(data: Any) -> Optional[bytes]:
    """The bytes of a one-message IMAP FETCH response, or ``None`` for an unexpected shape (see #80032)."""
    try:
        payload = data[0][1]
    except (IndexError, TypeError):
        return None
    return bytes(payload) if isinstance(payload, (bytes, bytearray)) else None


def check_email_requirements() -> bool:
    """True when all email settings are present and non-blank (blank keys left by an abandoned setup must not enable the platform).

    Treats blank/whitespace-only values as missing so an abandoned setup that left empty ``EMAIL_*`` keys in
    ``.env`` does not enable the platform (#40715).
    """
    return all(_get_secret(name, "").strip() for name in ("EMAIL_ADDRESS", "EMAIL_PASSWORD", "EMAIL_IMAP_HOST", "EMAIL_SMTP_HOST"))


def _safe_decode(payload: bytes, charset: Optional[str]) -> str:
    """Decode without ever raising: ``errors="replace"`` does not guard a missing codec (``LookupError``), so fall back alias → UTF-8 → latin-1.

    Unknown or malformed charset labels (``unknown-8bit``, misspelled names, attacker-controlled garbage)
    previously raised ``LookupError`` from ``bytes.decode`` — ``errors="replace"`` only guards decode
    errors, not a missing codec — which aborted the whole IMAP fetch and dropped every message in the batch
    (#35901, #55381, #55383). Fall back through a small alias table, then UTF-8, then latin-1 (which never
    fails).
    """
    label = (charset or "utf-8").strip().strip("\"'").lower() or "utf-8"
    for candidate in (_CHARSET_ALIASES.get(label, label), "utf-8"):
        try:
            return payload.decode(candidate, errors="replace")
        except (LookupError, ValueError):
            continue
    return payload.decode("latin-1", errors="replace")


def _decode_header_value(raw: str) -> str:
    """Decode an RFC 2047 header into a plain string; never raises.

    Never raises: malformed encoded-words or unknown charsets degrade to replacement characters instead of
    crashing the fetch loop (#55381).
    """
    try:
        parts = decode_header(raw)
    except Exception:  # malformed RFC 2047 structure
        return raw
    return " ".join(_safe_decode(part, charset) if isinstance(part, bytes) else part for part, charset in parts)


def _first_body_part(msg: email_lib.message.Message, content_type: str) -> str:
    """Decoded text of the first non-attachment part of *content_type*, or ''."""
    for part in msg.walk():
        if "attachment" in str(part.get("Content-Disposition", "")) or part.get_content_type() != content_type:
            continue
        if payload := part.get_payload(decode=True):
            return _safe_decode(payload, part.get_content_charset())
    return ""


def _extract_text_body(msg: email_lib.message.Message) -> str:
    """Extract the plain-text body from a potentially multipart email."""
    if msg.is_multipart():
        html = _first_body_part(msg, "text/html")
        return _first_body_part(msg, "text/plain") or (_strip_html(html) if html else "")
    text = _safe_decode(payload, msg.get_content_charset()) if (payload := msg.get_payload(decode=True)) else ""
    return _strip_html(text) if msg.get_content_type() == "text/html" else text


def _strip_html(html: str) -> str:
    """Naive HTML tag stripper for fallback text extraction."""
    for pattern, repl in _HTML_SUBS:
        html = pattern.sub(repl, html)
    return html.strip()


def _extract_email_address(raw: str) -> str:
    """Bare lowercased addr-spec from a From: value. Uses parseaddr, not a first-<...> regex (GHSA-rxqh-5572-8m77);
    RFC 5322 folding is unfolded first because parseaddr misreads a folded quoted display name. Unquoted
    ``Name <addr>`` values parseaddr rejects (``Doe, John <j@x>``) fall back to their single bracketed address,
    but only when the display part (``(comments)`` removed) cannot hold another mailbox or group: no quotes, ``;``,
    ``:`` or stray parens, and no ``@`` unless it is exactly the bracketed address (``a@x <a@x>``). Values over
    ``_MAX_FROM_LEN`` (or with more than 64 ``(``) and results without ``@`` return ``""`` so the caller drops the message."""
    value = re.sub(r"\r?\n[ \t]+", " ", str(raw or ""))
    if len(value) > _MAX_FROM_LEN or value.count("(") > 64:
        return ""  # hostile size/nesting: take the empty-sender drop (parseaddr recurses per nested comment)
    _, addr = parseaddr(value)
    if not addr and (m := _SINGLE_BRACKET_FROM_RE.fullmatch(value)):
        display, bracketed = _strip_comments(m.group(1)).strip(), m.group(2)
        if ("@" in bracketed[1:-1] and not any(c in display for c in ";:()")
                and ("@" not in display or display.lower() == bracketed.lower())):
            addr = bracketed
    addr = addr.strip().lower()
    return addr if "@" in addr else ""  # a bare word (``John``) is not a sender identity


def _strip_comments(text: str) -> str:
    """Remove (possibly nested) ``(comments)``, innermost first, until nothing changes."""
    while (stripped := _COMMENT_RE.sub(" ", text)) != text:
        text = stripped
    return text


def _ar_clauses(text: str) -> Optional[list[str]]:
    """Split an Authentication-Results value on ``;`` outside quoted-strings and (nested) comments; comments are
    dropped, quoted-strings kept (``header.from="x"`` stays readable). ``None`` when a quote or comment is unbalanced."""
    clauses, cur, depth, quoted, i = [], [], 0, False, 0
    while i < len(text):
        c = text[i]
        if c == "\\" and (quoted or depth):
            if quoted:
                cur.append(text[i:i + 2])
            i += 2
            continue
        if quoted:
            quoted = c != '"'
            cur.append(c)
        elif depth:
            depth += {"(": 1, ")": -1}.get(c, 0)
            if not depth:
                cur.append(" ")
        elif c == "(":
            depth = 1
        elif c == '"':
            quoted = True
            cur.append(c)
        elif c == ")":
            return None  # stray close paren: unbalanced
        elif c == ";":
            clauses.append("".join(cur))
            cur = []
        else:
            cur.append(c)
        i += 1
    return None if quoted or depth else clauses + ["".join(cur)]


def _auth_props(text: str) -> list[tuple[str, str]]:
    """``(property, value)`` pairs (``header.from=x``) of one comment-free Authentication-Results clause, property
    lowercased, surrounding quotes stripped. Quoted-string contents are never scanned for properties."""
    return [(p.lower(), v.strip('"')) for p, v in _AUTH_PROP_RE.findall(text) if p]


def _domain_of(address: str) -> str:
    """Lowercased domain part of an email address, or ''."""
    return address.rpartition("@")[2].strip().lower()


def _domains_aligned(a: str, b: str) -> bool:
    """Relaxed DMARC alignment: equal, or one is a dot-suffix of the other."""
    a = (a or "").strip().lower().rstrip(".")
    b = (b or "").strip().lower().rstrip(".")
    return bool(a and b) and (a == b or a.endswith("." + b) or b.endswith("." + a))


def _verify_sender_authentication(msg: email_lib.message.Message, from_addr: str, *, authserv_id: str) -> tuple[bool, str]:
    """Verify the ``From:`` domain is authenticated; returns ``(authenticated, reason)``.
    ``From:`` is attacker-controlled (GHSA-rxqh-5572-8m77); the only trustworthy signal is the
    ``Authentication-Results`` header stamped by the *receiving* server. It prepends, so only the
    FIRST instance is authoritative, and it must match the required, already-normalised *authserv_id* exactly. A matching id is a
    pin, not proof of provenance: the receiving MTA must strip inbound results claiming its id (RFC 8601).
    True on DMARC pass, aligned SPF pass, or aligned DKIM (``header.d``) pass. No header or no pin → fail-closed
    (opt out via ``EmailAdapter._require_authenticated_sender``)."""
    from_domain = _domain_of(from_addr)
    if not from_domain:
        return False, "missing From domain"
    if not (headers := msg.get_all("Authentication-Results")):
        return False, _NO_AUTH_RESULTS_REASON
    trusted = " ".join(str(headers[0]).split())
    # _ar_clauses removes RFC 8601 CFWS comments and ignores semicolons inside them, so supported
    # receiver variants remain valid without allowing a lower field or a related domain to satisfy the pin.
    if (clauses := _ar_clauses(trusted)) is None:
        return False, "unbalanced quote or comment in Authentication-Results"
    if not authserv_id:  # without a pin the topmost header may be one the sender wrote
        return False, _MISSING_AUTHSERV_REASON
    if clauses[0].strip().lower() != authserv_id:
        return False, _UNTRUSTED_AUTHSERV_REASON
    # Each verdict comes from the head of its own clause (split outside quotes/comments) and its domains only from that
    # clause: a quoted local part or comment can otherwise smuggle ``spf=pass``/``header.d=`` (GHSA-rxqh-5572-8m77).
    results: dict[str, list[tuple[str, list[tuple[str, str]]]]] = {"dmarc": [], "spf": [], "dkim": []}
    for clause in clauses:
        if m := _AUTH_METHOD_RE.match(clause):
            results[m.group(1).lower()].append((m.group(2).lower(), _auth_props(clause)))

    def aligned(props: list[tuple[str, str]], names: tuple[str, ...], *, required: bool = True) -> bool:
        domains = [_domain_of(v) for p, v in props if p in names]
        return (bool(domains) or not required) and all(_domains_aligned(d, from_domain) for d in domains)

    if len(results["dmarc"]) > 1:
        return False, "ambiguous dmarc result"
    # every header.from in the dmarc clause must be the From domain we parsed (absent header.from: trust the verdict)
    if any(r == "pass" and aligned(props, ("header.from",), required=False) for r, props in results["dmarc"]):
        return True, "dmarc=pass"
    # one SMTP transaction has one MAIL FROM verdict: a second spf clause means the SPF signal is not trusted
    if len(results["spf"]) == 1 and (spf := results["spf"][0])[0] == "pass" and aligned(
            spf[1], ("smtp.mailfrom", "smtp.from", "envelope-from")):
        return True, "spf=pass aligned"
    # several dkim clauses are normal (one per signature): any single pass whose own header.d aligns is enough
    if any(r == "pass" and aligned(props, ("header.d",) if any(p == "header.d" for p, _ in props) else ("header.from",))
           for r, props in results["dkim"]):
        return True, "dkim=pass aligned"
    return False, f"authentication failed ({trusted[:120]})"


def _extract_attachments(msg: email_lib.message.Message, skip_attachments: bool = False) -> list[dict[str, Any]]:
    """Extract attachment metadata and cache files locally (nothing when *skip_attachments*)."""
    attachments = []
    if not msg.is_multipart():
        return attachments
    for part in msg.walk():
        disposition, content_type = str(part.get("Content-Disposition", "")), part.get_content_type()
        if skip_attachments or ("attachment" not in disposition and (
                "inline" not in disposition or content_type in {"text/plain", "text/html"})):
            continue  # not an attachment, or an inline text/html body part
        filename = _decode_header_value(fn) if (fn := part.get_filename()) else f"attachment.{part.get_content_subtype() or 'bin'}"
        if not (payload := part.get_payload(decode=True)):
            continue
        if (ext := Path(filename).suffix.lower()) in _IMAGE_EXTS:
            try:
                cached_path, kind = cache_image_from_bytes(payload, ext), "image"
            except ValueError:
                logger.debug("Skipping non-image attachment %s (invalid magic bytes)", filename)
                continue
        else:
            cached_path, kind = cache_document_from_bytes(payload, filename), "document"
        attachments.append({"path": cached_path, "filename": filename, "type": kind, "media_type": content_type})
    return attachments


def _attach_file(msg: MIMEMultipart, path: Path, filename: str) -> None:
    """Attach *path* to *msg* as base64 application/octet-stream."""
    with open(path, "rb") as f:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={filename}")
        msg.attach(part)


class EmailAdapter(BasePlatformAdapter):
    """Email gateway adapter using IMAP (receive) and SMTP (send)."""
    serial_receive_loop = True  # the IMAP poll loop awaits each message before dispatching the next
    # One email carries the whole body, so cron delivery hands over the full payload untruncated.
    splits_long_messages = True

    # Per-account seen-UID snapshot surviving adapter recreation: the reconnect watcher builds a FRESH
    # adapter per retry; without this connect(is_reconnect=True) would re-mark the mailbox seen and skip
    # mail that arrived during the outage. Keyed by address (multiplex runs several accounts); same-process only.
    _seen_uids_snapshot: dict[str, set] = {}
    # Accounts already warned about a missing authserv_id pin. Per address, not per first connect: an account whose
    # first connect fails is brought up by the reconnect watcher (is_reconnect=True) and must still warn once.
    _missing_pin_warned: set = set()

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.EMAIL)
        # Env first, then PlatformConfig.extra (config.yaml-only setups). Host/address are stripped: a stray
        # newline made IMAP4_SSL raise ``[Errno 8] nodename nor servname`` instead of "host not set".
        extra = config.extra or {}
        setting = lambda env, key: _get_secret(env, "") or extra.get(key, "")
        tls_verify = lambda env, key: _esecret_bool(env, is_truthy_value(extra.get(key), default=True))
        self._address = setting("EMAIL_ADDRESS", "address").strip()
        self._password = _get_secret("EMAIL_PASSWORD", "")
        self._imap_host = setting("EMAIL_IMAP_HOST", "imap_host").strip()
        self._imap_port = _esecret_int("EMAIL_IMAP_PORT", 993)
        self._imap_security = _normalize_security(setting("EMAIL_IMAP_SECURITY", "imap_security"))
        self._imap_tls_verify = tls_verify("EMAIL_IMAP_TLS_VERIFY", "imap_tls_verify")
        self._smtp_host = setting("EMAIL_SMTP_HOST", "smtp_host").strip()
        self._smtp_port = _esecret_int("EMAIL_SMTP_PORT", 587)
        self._smtp_security = _normalize_security(setting("EMAIL_SMTP_SECURITY", "smtp_security"), default="tls" if self._smtp_port == 465 else "starttls")
        self._smtp_tls_verify = tls_verify("EMAIL_SMTP_TLS_VERIFY", "smtp_tls_verify")
        self._poll_interval = _esecret_int("EMAIL_POLL_INTERVAL", 15)
        self._skip_attachments = extra.get("skip_attachments", False)  # platforms.email.skip_attachments
        self._html = is_truthy_value(extra.get("html"), default=False)  # platforms.email.html: add a text/html part rendered from Markdown
        self._thread_sessions = is_truthy_value(extra.get("thread_sessions"), default=False)  # platforms.email.thread_sessions: own session per thread for mail Hermes didn't thread
        # platforms.email.sent_folder: where the provider files sent mail (read back for the Message-ID it delivered
        # each one under); unset = found by LIST, false/"off" = not read.
        sent = extra.get("sent_folder")
        off = sent is False or (isinstance(sent, str) and sent.strip().lower() in {"off", "false", "no", "none"})
        self._sent_folder: Optional[str] = None if off or not isinstance(sent, str) else sent.strip() or None
        if self._sent_folder and self._sent_folder.upper() == "INBOX":  # inbound mail is not authenticated there
            logger.warning("[Email] platforms.email.sent_folder may not be INBOX; not reading sent mail")
            off, self._sent_folder = True, None
        self._sent_select_failed = False  # warned once that the Sent folder cannot be selected
        self._sent_lookup = not off and self._sent_folder is None  # find the folder by LIST on first use
        self._sent_mark: Optional[tuple[Optional[str], int]] = None  # (UIDVALIDITY, highest UID read) this process
        # Require an authenticated From: domain (SPF/DKIM/DMARC) before trusting it for authorization
        # (GHSA-rxqh-5572-8m77). Default ON; opt out via require_authenticated_sender: false / EMAIL_TRUST_FROM_HEADER=true.
        if "require_authenticated_sender" in extra:
            self._require_authenticated_sender = bool(extra["require_authenticated_sender"])
        else:
            self._require_authenticated_sender = not _esecret_bool("EMAIL_TRUST_FROM_HEADER", False)
        # Pin Authentication-Results to the operator's receiving MTA. With sender auth on, an absent pin fails
        # closed for EVERY sender rather than trusting a sender-supplied result header; connect() warns once per account.
        self._authserv_id = (extra.get("authserv_id", "") or _get_secret("EMAIL_AUTHSERV_ID", "")).strip().lower()
        self._seen_uids: set = set()
        self._seen_uids_max: int = 2000   # cap to prevent unbounded memory growth
        self._poll_task: Optional[asyncio.Task] = None
        self._last_fetch_failed, self._last_fetch_error = False, ""  # "checked, nothing new" vs "the check itself failed"
        self._fetched_max_uid = 0  # highest UID the last _fetch_new_messages fetched OK or rejected (the cold-start mark follows it)
        # chat_id (sender email) -> last subject + message-id for threading
        # Track the last IMAP fetch attempt so the poll loop can distinguish "checked, nothing new" from
        # "the check itself failed" (#80016).
        self._thread_context: dict[str, dict[str, str]] = {}
        # (sender, thread root) -> subject, last message id and References list of that thread
        self._root_context: dict[tuple[str, str], dict[str, Any]] = {}
        # Resolved once and logged: plugin_data_dir follows the active profile's home.
        self._store_path = plugin_data_dir("email") / "threads.json"
        self._store = _EmailThreadStore(self._store_path)
        logger.info("[Email] Thread store: %s", self._store_path)
        logger.info("[Email] Adapter initialized for %s", self._address)

    def _trim_seen_uids(self) -> None:
        """Keep only the highest half of UIDs once over the cap (UIDs are monotonic; UNSEEN prevents re-delivery)."""
        if len(self._seen_uids) <= self._seen_uids_max:
            return
        try:
            sorted_uids = sorted(self._seen_uids, key=lambda u: int(u))  # UIDs are bytes like b'1234'
            self._seen_uids = set(sorted_uids[-(self._seen_uids_max // 2):])
            logger.debug("[Email] Trimmed seen UIDs to %d entries", len(self._seen_uids))
        except (ValueError, TypeError):
            self._seen_uids = set(list(self._seen_uids)[-self._seen_uids_max // 2:])

    def _connect_imap(self) -> imaplib.IMAP4:
        """Create an IMAP connection using implicit TLS, STARTTLS, or plaintext."""
        if self._imap_security == "tls":
            return imaplib.IMAP4_SSL(self._imap_host, self._imap_port, timeout=30, ssl_context=_tls_context(self._imap_tls_verify, self._imap_host))
        imap = imaplib.IMAP4(self._imap_host, self._imap_port, timeout=30)
        if self._imap_security == "starttls":
            try:
                imap.starttls(ssl_context=_tls_context(self._imap_tls_verify, self._imap_host))
            except Exception:
                _close_imap(imap)
                raise
        return imap

    @contextmanager
    def _inbox(self):
        """Logged-in IMAP handle on INBOX; always ``_close_imap``-ed on exit (a login/select failure used to leak one fd per reconnect)."""
        # Test IMAP connection. The handle is closed in ``finally`` — before this, a failure in
        # login/select/search left the TCP socket open with no owner, leaking one fd per connect attempt.
        # Under the gateway's reconnect watcher (fresh adapter instance per retry) against an
        # unreachable/proxied host this grew monotonically until fd exhaustion on macOS's 256 soft limit
        # (#79889).
        imap = self._connect_imap()
        try:
            imap.login(self._address, self._password)
            _send_imap_id(imap)
            imap.select("INBOX")
            yield imap
        finally:
            _close_imap(imap)

    def _connect_smtp(self) -> smtplib.SMTP:
        """SMTP connection with TLS established (callers go straight to ``login()``). An unreachable IPv6 address can
        hang until the socket timeout, so connection-level failures retry through an IPv4-only socket path (no global
        resolver mutation); TLS verification errors are not retried."""
        host, port, security, ctx = self._smtp_host, self._smtp_port, self._smtp_security, _tls_context(self._smtp_tls_verify, self._smtp_host)
        try:
            return _open_smtp(host, port, security, ctx, smtplib.SMTP, smtplib.SMTP_SSL, timeout=SMTP_CONNECT_TIMEOUT)
        except (TimeoutError, ConnectionError, OSError) as exc:
            if isinstance(exc, ssl.SSLError):
                raise
            return _open_smtp(host, port, security, ctx, _IPv4SMTP, _IPv4SMTP_SSL, timeout=SMTP_CONNECT_TIMEOUT)

    def _fail(self, log_fmt: str, err: object, code: str, detail: str, *, retryable: bool) -> bool:
        """Log *err*, record a fatal error for the gateway's reconnect machinery, return False."""
        logger.error(log_fmt, err)
        self._set_fatal_error(code, detail, retryable=retryable)
        return False

    def _probe_imap(self, is_reconnect: bool) -> bool:
        """Connection test + seen-UID baseline. Sets a fatal error and returns False on failure."""
        try:
            with self._inbox() as imap:
                snapshot = self._seen_uids_snapshot.get(self._address)
                if is_reconnect and snapshot is not None:
                    # Same-process reconnect: restore the previous adapter's baseline so mail that
                    # arrived during the outage stays eligible for the next poll.
                    self._seen_uids = set(snapshot)
                    passed = "[Email] IMAP reconnect test passed. Restored %d seen UIDs; messages received during the outage will be processed."
                else:  # first connect (or no snapshot): mark existing messages seen, up to the stored UID mark
                    status, data = imap.uid("search", None, "ALL")
                    all_uids = data[0].split() if status == "OK" and data and data[0] else []
                    # UIDs above a mark stored under the same UIDVALIDITY arrived while the gateway was down and
                    # stay eligible for the UNSEEN poll. Classified here by int(uid): ``UID SEARCH UID n:*`` always
                    # includes the last message. No mark, or a missing or changed UIDVALIDITY: mark everything seen.
                    validity = _uidvalidity(imap)
                    stored_validity, mark = self._store.uid_state(self._address)
                    if validity is None or stored_validity != validity:
                        mark = None
                    self._seen_uids.update(u for u in all_uids if mark is None or _uid_int(u) <= mark)
                    if validity is not None and mark is None:
                        self._store.set_uid_state(self._address, validity, max((_uid_int(u) for u in all_uids), default=0))
                    caught_up = 0 if mark is None else sum(1 for u in all_uids if _uid_int(u) > mark)
                    passed = ("[Email] IMAP connection test passed. %d existing messages skipped; "
                              f"{caught_up} newer than the stored mark stay eligible.")
                self._trim_seen_uids()
                logger.info(passed, len(self._seen_uids))
            self._seen_uids_snapshot[self._address] = set(self._seen_uids)
            return True
        except Exception as e:
            # Always set an explicit fatal code, else the gateway treats every failure as transient with zero
            # owner signal. retryable=True because imaplib raises the same generic IMAP4.error for bad credentials
            # AND transient NOs (Gmail "too many simultaneous connections"); loops surface via NEEDS_ATTENTION.
            return self._fail("[Email] IMAP connection failed: %s", e, "email_imap_connect_error",
                              f"IMAP connection to {self._imap_host}:{self._imap_port} failed: {e}", retryable=True)

    def _probe_smtp(self) -> bool:
        """SMTP connect + login test. Sets a fatal error and returns False on failure."""
        try:
            smtp = self._connect_smtp()
            try:
                smtp.login(self._address, self._password)
            finally:
                smtp.quit()
            logger.info("[Email] SMTP connection test passed.")
            return True
        except smtplib.SMTPAuthenticationError as e:
            # Typed auth failure (535 & friends) can never self-heal, so drop out of the reconnect queue — unambiguous, unlike IMAP4.error.
            return self._fail("[Email] SMTP authentication failed: %s", e, "email_auth_error",
                              f"SMTP authentication failed for {self._address}: {e}. Check EMAIL_PASSWORD (for Gmail/Outlook "
                              "this must be an app password, not the account password).", retryable=False)
        except Exception as e:
            return self._fail("[Email] SMTP connection failed: %s", e, "email_smtp_connect_error",
                              f"SMTP connection to {self._smtp_host} failed: {e}", retryable=True)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Connect to the IMAP server and start polling for new messages."""
        # Validate up front so a missing host is an actionable config error, not IMAP4_SSL("") raising ``[Errno 8]``.
        required = (("EMAIL_ADDRESS", self._address), ("EMAIL_PASSWORD", self._password), ("EMAIL_IMAP_HOST", self._imap_host), ("EMAIL_SMTP_HOST", self._smtp_host))
        if missing := [name for name, value in required if not value]:
            message = f"Not configured — missing {', '.join(missing)}. Set it via `hermes gateway setup` (env) or platforms.email in config.yaml."
            # Non-retryable: a blank-but-present env var used to drive an indefinite retry loop that leaked until OOM.
            return self._fail("[Email] %s", message, "email_missing_configuration", message, retryable=False)
        if not self._probe_imap(is_reconnect) or not self._probe_smtp():
            return False
        if self._require_authenticated_sender and not self._authserv_id and self._address not in EmailAdapter._missing_pin_warned:
            EmailAdapter._missing_pin_warned.add(self._address)
            logger.warning("[Email] %s: %s.%s", self._address, _MISSING_AUTHSERV_REASON, _MISSING_AUTHSERV_HINT)
        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        print(f"[Email] Connected as {self._address}")
        self._wire_plugin_handlers(None)  # plugin-registered native handlers
        return True

    async def disconnect(self) -> None:
        """Stop polling and disconnect."""
        self._running = False
        await cancel_task(self._poll_task)
        self._poll_task = None
        logger.info("[Email] Disconnected.")

    async def _poll_loop(self) -> None:
        """Poll IMAP for new messages at regular intervals."""
        while self._running:
            try:
                await self._check_inbox()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("[Email] Poll error: %s", e)
            await asyncio.sleep(self._poll_interval)

    async def _check_inbox(self) -> None:
        """Check INBOX for unseen messages and dispatch them."""
        loop = asyncio.get_running_loop()

        async def authorize_on_loop(msg_data: dict[str, Any]) -> bool:
            return self._sender_accepted(msg_data["sender_addr"], msg_data)

        def gate(candidate: dict[str, Any]) -> bool:
            # Authorization reads profile-scoped policy and pairing state on the adapter's event loop;
            # the blocking IMAP worker waits for that verdict before asking the server for RFC822.
            # A closed loop (stop race) yields None: fail closed without leaking the coroutine.
            fut = safe_schedule_threadsafe(authorize_on_loop(candidate), loop, logger=logger,
                                           log_message="[Email] Could not schedule sender authorization")
            return fut is not None and fut.result()

        messages = await asyncio.to_thread(self._fetch_new_messages, gate)
        # Dispatch partial results BEFORE escalating a failure — a mid-batch exception returns what was fetched (already marked seen).
        for msg_data in messages:
            try:
                await self._dispatch_message(msg_data)
            except Exception:  # one bad mail must not drop the rest of the batch: all of it is already \Seen
                logger.exception("[Email] Dispatch failed for UID %s, skipping", msg_data.get("uid"))
            self._store.advance_uid_mark(self._address, _uid_int(msg_data.get("uid")))
        # UIDs the filters dropped (automated senders, unparseable mail) move the mark too, or they replay after a restart.
        self._store.advance_uid_mark(self._address, self._fetched_max_uid)
        if self._last_fetch_failed:
            # The IMAP check itself failed (not an empty inbox): route through the fatal-error hook so the gateway's
            # reconnect/backoff re-establishes the mailbox. The handler runs detached (gateway/run.py), so awaiting it is safe.
            # The handler runs in a detached task (gateway/run.py), so awaiting it from our own poll task is
            # safe even though teardown cancels this task. See #80016.
            self._last_fetch_failed = False
            self._set_fatal_error("email_imap_fetch_failed", self._last_fetch_error or "IMAP fetch failed", retryable=True)
            await self._notify_fatal_error()

    def _mark_uid_consumed(self, imap: imaplib.IMAP4, uid: Any) -> None:
        """Remember a rejected UID and mark it seen without fetching its MIME body."""
        self._seen_uids.add(uid)
        self._trim_seen_uids()
        self._fetched_max_uid = max(self._fetched_max_uid, _uid_int(uid))  # rejected mail moves the cold-start mark too
        status, _ = imap.uid("store", uid, "+FLAGS.SILENT", r"(\Seen)")
        if status != "OK":
            logger.warning("[Email] Could not mark rejected UID %s seen", uid)

    def _fetch_new_messages(self, preauthorize: Callable[[dict[str, Any]], bool]) -> list[dict[str, Any]]:
        """Fetch unseen messages; bounded headers pass *preauthorize* before RFC822 is requested."""
        results = []
        self._fetched_max_uid = 0  # highest UID fetched OK or rejected this call; _check_inbox moves the stored mark past it
        try:
            with self._inbox() as imap:
                if self._learn_delivered_ids(imap):
                    imap.select("INBOX")
                status, data = imap.uid("search", None, "UNSEEN")
                for uid in (data[0].split() if status == "OK" and data and data[0] else []):
                    if uid in self._seen_uids:
                        continue
                    header_status, header_data = imap.uid("fetch", uid, _PREAUTH_FETCH)
                    if header_status != "OK":
                        continue
                    raw_headers = _imap_payload(header_data)
                    if raw_headers is None or len(raw_headers) > _MAX_PREAUTH_HEADER_BYTES:
                        logger.warning("[Email] Unusable pre-authorization headers for UID %s, skipping", uid)
                        self._mark_uid_consumed(imap, uid)
                        continue
                    try:
                        candidate = self._parse_fetched_headers(uid, raw_headers)
                        accepted = candidate is not None and preauthorize(candidate)
                    except Exception as auth_exc:
                        logger.error("[Email] Failed to authorize message UID %s, skipping: %s", uid, auth_exc)
                        accepted = False
                    if not accepted:
                        self._mark_uid_consumed(imap, uid)
                        continue
                    status, msg_data = imap.uid("fetch", uid, "(RFC822)")
                    if status != "OK":
                        continue  # transient per-UID refusal: leave unseen so the next poll retries
                    self._fetched_max_uid = max(self._fetched_max_uid, _uid_int(uid))
                    # Mark seen once a response arrived (even malformed) so garbage is skipped once, not retried forever —
                    # but NOT before the fetch: a connection failure must leave the rest of the batch eligible for the next poll.
                    self._seen_uids.add(uid)
                    self._trim_seen_uids()
                    if (raw_email := _imap_payload(msg_data)) is None:
                        logger.warning("[Email] Unexpected IMAP response structure for UID %s, skipping", uid)
                        continue
                    # One poison message (unparseable headers, pathological attachment, DNS hiccup) must not abort the batch or force a reconnect.
                    try:
                        # See #80032.
                        parsed = self._parse_fetched_message(uid, raw_email)
                    except Exception as parse_exc:
                        logger.error("[Email] Failed to process message UID %s, skipping: %s", uid, parse_exc)
                        continue
                    if parsed is not None:
                        results.append(parsed)
        except Exception as e:
            # _close_imap guarantees the socket dies even when logout() raises IMAP4.abort on a broken
            # connection (#79889).
            logger.error("[Email] IMAP fetch error: %s", e)
            self._last_fetch_failed, self._last_fetch_error = True, str(e)
        # Keep the reconnect snapshot current so a mid-outage adapter recreation does not re-dispatch messages already processed.
        self._seen_uids_snapshot[self._address] = set(self._seen_uids)
        return results

    def _learn_delivered_ids(self, imap: "imaplib.IMAP4") -> bool:
        """Read the copies of sent mail new in the Sent folder (the newest ``_SENT_BACKFILL_MAX`` on the first poll)
        and index the Message-ID each was delivered under to its thread (``learn_delivered_ids``). Runs before the
        INBOX search so a reply in the same poll already routes. Only while the store has thread roots; a missing
        folder turns it off for this adapter. Never raises. Returns True when another mailbox was selected."""
        if not (self._sent_folder or self._sent_lookup) or not self._store.has_roots():
            return False
        selected = False
        try:
            if self._sent_folder is None:
                self._sent_lookup = False
                self._sent_folder = _find_sent_folder(imap)
                if self._sent_folder is None:
                    logger.info("[Email] No Sent folder found; replies naming only a provider-assigned Message-ID "
                                "are not matched to their thread (set platforms.email.sent_folder to name it)")
                    return False
            selected = True
            status, _ = imap.select(_quote_mailbox(self._sent_folder), readonly=True)
            if status != "OK":
                if not self._sent_select_failed:
                    self._sent_select_failed = True
                    logger.warning("[Email] Could not select the Sent folder %s; replies naming only a "
                                   "provider-assigned Message-ID are not matched to their thread", self._sent_folder)
                return True
            validity = _uidvalidity(imap)
            mark = self._sent_mark[1] if self._sent_mark and self._sent_mark[0] == validity else None
            status, data = imap.uid("search", None, "ALL" if mark is None else f"UID {mark + 1}:*")
            uids = [u for u in (data[0].split() if status == "OK" and data and data[0] else []) if _uid_int(u) > (mark or 0)]
            if mark is None:
                uids = sorted(uids, key=_uid_int)[-_SENT_BACKFILL_MAX:]
            learned = 0
            if uids:
                status, data = imap.uid("fetch", ",".join(u.decode("ascii") if isinstance(u, bytes) else str(u) for u in uids),
                                        _SENT_HEADERS_FETCH)
                if status != "OK":
                    return True
                learned = self._store.learn_delivered_ids(_sent_copies(data))
            self._sent_mark = (validity, max([mark or 0, *(_uid_int(u) for u in uids)]))
            if learned:
                logger.info("[Email] Indexed %d delivered Message-ID(s) of sent mail from %s", learned, self._sent_folder)
        except (imaplib.IMAP4.error, OSError, ValueError) as exc:  # IMAP refusal or abort, socket/TLS, bad bytes
            logger.warning("[Email] Could not read the Sent folder for delivered Message-IDs: %s", exc)
        except Exception as exc:  # deliberate boundary: a Sent-read bug must never block the INBOX poll after it
            logger.warning("[Email] Could not read the Sent folder for delivered Message-IDs: %s", exc, exc_info=True)
        return selected

    def _message_metadata(self, uid: bytes, msg: email_lib.message.Message) -> Optional[dict[str, Any]]:
        """Parse sender-facing headers and authentication without touching the MIME body."""
        if not (sender_addr := _extract_email_address(msg.get("From", ""))):  # never dispatch an empty identity
            logger.debug("[Email] Dropping message with no parseable From address: %r", msg.get("From", ""))
            return None
        sender_name = _decode_header_value(msg.get("From", ""))
        if "<" in sender_name:
            sender_name = sender_name.split("<")[0].strip().strip('"')
        subject = _decode_header_value(msg.get("Subject", "(no subject)"))
        if _is_automated_sender(sender_addr, dict(msg.items())):
            logger.debug("[Email] Skipping automated sender: %s", sender_addr)
            return None
        sender_authenticated, auth_reason = _verify_sender_authentication(msg, sender_addr, authserv_id=self._authserv_id)
        return {"uid": uid, "sender_addr": sender_addr, "sender_name": sender_name, "subject": subject,
                # str(): a header with raw 8-bit bytes parses to an email.header.Header, not a str (compat32)
                "message_id": str(msg.get("Message-ID") or ""), "in_reply_to": str(msg.get("In-Reply-To") or ""),
                "references": str(msg.get("References") or ""),
                "date": msg.get("Date", ""), "sender_authenticated": sender_authenticated, "auth_reason": auth_reason}

    def _parse_fetched_headers(self, uid: bytes, raw_headers: bytes | bytearray) -> Optional[dict[str, Any]]:
        """Parse the bounded IMAP header preflight without constructing a MIME tree."""
        return self._message_metadata(uid, BytesHeaderParser().parsebytes(bytes(raw_headers)))

    def _parse_fetched_message(self, uid: bytes, raw_email: bytes | bytearray) -> Optional[dict[str, Any]]:
        """Parse an authorized RFC822 payload into a dispatchable dict."""
        msg = email_lib.message_from_bytes(raw_email)
        if (metadata := self._message_metadata(uid, msg)) is None:
            return None
        metadata.update(body=_extract_text_body(msg),
                        attachments=_extract_attachments(msg, skip_attachments=self._skip_attachments))
        return metadata

    @staticmethod
    def _allow_all_senders() -> bool:
        """True when the operator opted into any sender (EMAIL_ or GATEWAY_ALLOW_ALL_USERS).

        Both names go through the scoped reader: under multiplex ``os.environ`` is the DEFAULT
        profile's opt-in, and borrowing it opened every secondary mailbox to any sender."""
        return any(_get_secret(name, "").strip().lower() in _TRUTHY
                   for name in ("EMAIL_ALLOW_ALL_USERS", "GATEWAY_ALLOW_ALL_USERS"))

    def _answers_unknown_senders(self) -> bool:
        """True when ``platforms.email.unauthorized_dm_behavior`` opts into ``pair`` or ``decline``."""
        behavior = (self.config.extra or {}).get("unauthorized_dm_behavior")
        return isinstance(behavior, str) and behavior.strip().lower() in {"pair", "decline"}

    def _sender_accepted(self, sender_addr: str, msg_data: dict[str, Any]) -> bool:
        """Pre-dispatch sender gate: self, automated, authorization, From: authentication."""
        return self._sender_verdict(sender_addr, msg_data)[0]

    def _sender_verdict(self, sender_addr: str, msg_data: dict[str, Any]) -> tuple[bool, bool]:
        """``(accepted, granted)``: accepted mail is dispatched; a sender accepted but not granted access is
        an unknown one the gateway only answers (``unauthorized_dm_behavior`` pair or decline)."""
        if sender_addr == self._address.lower():
            return False, False
        if _is_automated_sender(sender_addr, {}):
            logger.debug("[Email] Dropping automated sender at dispatch: %s", sender_addr)
            return False, False
        allowed_raw = _get_secret("EMAIL_ALLOWED_USERS", "").strip()
        # Parsed like the gateway's allowlists (JSON list literals included), or '["alice"]' would dodge the guard below.
        listed = set()
        for raw in (allowed_raw, _get_secret("GATEWAY_ALLOWED_USERS", "")):
            raw = decode_json_list_literal(raw)
            listed.update(str(a).strip().lower() for a in (raw if isinstance(raw, list) else str(raw).split(","))
                          if str(a).strip())
        is_listed = sender_addr.lower() in listed
        if is_listed:
            granted = True
        elif sender_addr.split("@", 1)[0].lower() in listed:
            # The gateway's check also matches an address by its bare local part (#119446), so an entry like "alice"
            # would admit, or pair, alice@<any domain>; the domain is the sender's to choose.
            logger.debug("[Email] Dropping sender whose local part alone matches an allowlist entry: %s", sender_addr)
            return False, False
        else:
            # Approved pairings grant access too, and only the gateway's own check sees them. Its verdict also decides
            # open access: GATEWAY_ALLOW_ALL_USERS beside a GATEWAY_ALLOWED_USERS list grants a stranger nothing there.
            verdict = self._is_sender_authorized(sender_addr, "dm", sender_addr)
            granted = verdict if verdict is not None else (not allowed_raw and self._allow_all_senders())
        # Drop senders the gateway would neither authorize nor answer (pair/decline) before a MessageEvent (and thread
        # context) exists — otherwise a dispatch/authorization race can send a reply even though the handler returned None.
        if not granted and not self._answers_unknown_senders():
            logger.debug("[Email] Dropping unauthorized sender at dispatch (unknown senders are ignored): %s", sender_addr)
            return False, False
        # Reject spoofed senders (GHSA-rxqh-5572-8m77): every grant keys on the attacker-controlled From:, and a pairing
        # code or decline is mailed back to it; fail-closed. Open access is no exception: the session and every reply
        # key on From:, so a forged one lands in that address's conversation and makes the agent mail it.
        # Warn only where the operator can act: a known misconfiguration, or a listed contact's own mail failing.
        # Forged stranger mail (open access grants everyone) stays at debug.
        if self._require_authenticated_sender and not msg_data.get("sender_authenticated", False):
            auth_reason = msg_data.get("auth_reason", "no verdict")
            hint = _DROP_HINTS.get(auth_reason, "")
            if is_listed or (granted and hint):
                logger.warning("[Email] Dropping sender with unauthenticated From: %s (%s).%s", sender_addr, auth_reason, hint)
            else:
                logger.debug("[Email] Dropping %s sender with unauthenticated From: %s (%s)",
                             "authorized" if granted else "unknown", sender_addr, auth_reason)
            return False, False
        return True, granted

    async def _dispatch_message(self, msg_data: dict[str, Any]) -> None:
        """Convert a fetched email into a MessageEvent and dispatch it."""
        sender_addr = msg_data["sender_addr"]
        accepted, granted = self._sender_verdict(sender_addr, msg_data)
        if not accepted:
            return
        subject, body, attachments = msg_data["subject"], msg_data["body"].strip(), msg_data["attachments"]
        text = f"[Subject: {subject}]\n\n{body}" if subject and not subject.startswith("Re:") else body  # subject unless reply
        # DOCUMENT wins over PHOTO for mixed attachments: run.py keys image handling off the per-path mime type regardless
        # of message_type, but document-context injection gates strictly on MessageType.DOCUMENT — so DOCUMENT surfaces both.
        kinds = {att["type"] for att in attachments}
        self._thread_context[sender_addr] = {"subject": subject, "message_id": msg_data["message_id"]}
        # Only for a granted sender: resolving may write the store, and strangers must not reach it. An unknown
        # sender the gateway answers (pair/decline) gets that answer flat.
        key, header, topic = self._resolve_thread(msg_data) if granted else (None, "", "")
        if key:
            own = msg_data["message_id"]
            self._store.index_message(own, key)
            refs = _message_ids(msg_data.get("references")) or ([header] if header and header != own.strip() else [])
            self._root_context[(sender_addr, key)] = {"subject": topic, "message_id": own, "references": refs}
        name = msg_data["sender_name"] or sender_addr
        event = MessageEvent(
            text=text or "(empty email)", message_id=msg_data["message_id"],
            message_type=MessageType.DOCUMENT if "document" in kinds else MessageType.PHOTO if "image" in kinds else MessageType.TEXT,
            source=self.build_source(chat_id=sender_addr, chat_name=name, chat_type="dm", user_id=sender_addr, user_name=name,
                                     message_id=msg_data["message_id"], thread_id=key,
                                     chat_topic=f"{topic} (thread {_shorten(header)})" if key else None),
            media_urls=[att["path"] for att in attachments], media_types=[att["media_type"] for att in attachments],
            reply_to_message_id=msg_data["in_reply_to"] or None)
        logger.info("[Email] New message from %s: %s", sender_addr, subject)
        await self.handle_message(event)

    def _resolve_thread(self, msg_data: dict[str, Any]) -> tuple[Optional[str], str, str]:
        """``(root key, root Message-ID, thread subject)`` for an accepted inbound mail, or ``(None, "", "")`` for
        today's flat per-sender session. A root the store knows (any id in References, then In-Reply-To) always
        routes, as Hermes only mints roots when cron asked for a thread. With ``thread_sessions`` on, any other
        thread is keyed by its first References entry, else In-Reply-To, else the mail's own Message-ID."""
        subject = msg_data.get("subject") or ""
        refs = _message_ids(msg_data.get("references"))
        irt = [mid for mid in _message_ids(msg_data.get("in_reply_to")) if mid not in refs]
        key = self._store.root_for(refs + irt)
        if key:
            root = self._store.get_root(key) or {}
            return key, root.get("header") or f"<{key}>", root.get("subject") or _strip_reply_prefix(subject)
        if not self._thread_sessions:
            return None, "", ""
        header = refs[0] if refs else irt[0] if irt else (_message_ids(msg_data.get("message_id")) or [""])[0]
        if not header:
            return None, "", ""
        key = _root_key(header)
        self._store.remember_root(key, header, _strip_reply_prefix(subject))
        stored = self._store.get_root(key) or {}
        return key, header, stored.get("subject") or _strip_reply_prefix(subject)

    async def _run_send(self, fn, args: tuple, log_fmt: str, *log_args) -> SendResult:
        """Run a blocking SMTP sender in the executor; wrap its Message-ID in a SendResult."""
        try:
            return SendResult(success=True, message_id=await asyncio.get_running_loop().run_in_executor(None, fn, *args))
        except Exception as e:
            logger.error(log_fmt, *log_args, e)
            return SendResult(success=False, error=str(e))

    async def send(self, chat_id: str, content: str, reply_to: Optional[str] = None, metadata: Optional[dict[str, Any]] = None) -> SendResult:
        """Send an email to the given address. ``metadata["thread_id"]`` (a root from ``create_handoff_thread`` or an
        inbound thread) threads it under that root; ``metadata["job_id"]`` is recorded on a root's first mail."""
        meta = metadata or {}
        job_id = str(meta.get("job_id") or "")
        return await self._run_send(self._send_email, (chat_id, content, reply_to, _thread_key(meta), job_id),
                                    "[Email] Send failed to %s: %s", chat_id)

    async def create_handoff_thread(self, parent_chat_id: str, name: str) -> Optional[str]:
        """Mint a thread root for *parent_chat_id* and record it as pending; no mail is sent here. The first
        ``send`` with this id as ``metadata["thread_id"]`` goes out as a fresh mail whose Message-ID is the root
        and whose subject comes from the content (else *name*). Returns the session-key-safe root, or ``None``."""
        try:
            header = f"<hermes-{uuid.uuid4().hex[:12]}@{self._message_id_domain()}>"
            key = _root_key(header)
            self._store.add_pending_root(key, header, name, (parent_chat_id or "").strip().lower())
            return key
        except (TypeError, ValueError) as exc:  # the root did not serialize; a failed file write is the store's warning
            logger.warning("[Email] Handoff thread: could not record a thread root for %s: %s", parent_chat_id, exc)
            return None

    def _message_id_domain(self) -> str:
        """Domain for generated Message-IDs; ``localhost`` when EMAIL_ADDRESS lacks ``@``."""
        return (self._address.rsplit("@", 1)[-1] if "@" in self._address else "") or "localhost"

    def _new_reply(self, to_addr: str, body: str, reply_to_msg_id: Optional[str] = None, *,
                   attach_empty_body: bool = False, thread_key: Optional[str] = None) -> tuple[MIMEMultipart, str, str]:
        """Build a reply skeleton. Returns ``(msg, msg_id, subject)``.

        A pending thread root goes out fresh (Message-ID = the root, References = the root so a reply carries it
        even if a provider replaces the Message-ID, no In-Reply-To; subject from a leading ``Subject:`` line else the
        job name); any other *thread_key* replies inside that thread with the References chain from the root; no
        *thread_key* threads on the sender's last mail, as before."""
        msg = MIMEMultipart()
        root = self._store.get_root(thread_key) if thread_key else None
        if root and root.get("pending"):
            subject, body = _split_subject(body, _subject_from_thread_name(root.get("name")))
            msg_id = root.get("header") or f"<{thread_key}>"
            thread_headers: tuple = (("References", msg_id),)
        else:
            msg_id = f"<hermes-{uuid.uuid4().hex[:12]}@{self._message_id_domain()}>"
            if thread_key:
                header = (root or {}).get("header") or ("" if thread_key.startswith("h-") else f"<{thread_key}>")
                rctx = self._root_context.get((to_addr.lower(), thread_key), {})
                subject = (rctx.get("subject") or (root or {}).get("subject")
                           or self._thread_context.get(to_addr, {}).get("subject") or "Hermes Agent")
                parent = reply_to_msg_id or rctx.get("message_id") or header
                chain: list[str] = []
                for mid in [*(rctx.get("references") or ([header] if header else [])), parent]:
                    if mid and mid not in chain:
                        chain.append(mid)
                if len(chain) > _REFERENCES_MAX:
                    chain = chain[:1] + chain[-(_REFERENCES_MAX - 1):]
                thread_headers = (("In-Reply-To", parent), ("References", " ".join(chain))) if parent else ()
            else:
                ctx = self._thread_context.get(to_addr, {})
                subject = ctx.get("subject", "Hermes Agent")
                original_msg_id = reply_to_msg_id or ctx.get("message_id")
                thread_headers = (("In-Reply-To", original_msg_id), ("References", original_msg_id)) if original_msg_id else ()
            if not subject.startswith("Re:"):
                subject = f"Re: {subject}"
        for key, value in (("From", self._address), ("To", to_addr), ("Subject", subject), *thread_headers,
                           ("Date", formatdate(localtime=True)), ("Message-ID", msg_id)):
            msg[key] = value
        if body or attach_empty_body:
            _attach_body(msg, body, self._html)
        return msg, msg_id, subject

    def _smtp_send(self, msg: MIMEMultipart) -> None:
        """Login, send, and always release the SMTP connection (quit, else close)."""
        smtp = self._connect_smtp()
        try:
            smtp.login(self._address, self._password)
            smtp.send_message(msg)
        finally:
            try:
                smtp.quit()
            except Exception:
                smtp.close()

    def _after_thread_send(self, thread_key: Optional[str], msg_id: str, subject: str, job_id: str) -> bool:
        """After a confirmed send in a thread: a pending root is marked sent (subject, job id) and the mail is
        indexed to its root. Returns True when this mail was the root itself. Never raises: the mail is out."""
        if not thread_key:
            return False
        try:
            was_root = self._store.is_pending(thread_key)
            if was_root:
                self._store.mark_sent(thread_key, subject, job_id)
            self._store.index_message(msg_id, thread_key)
            return was_root
        except Exception as exc:  # deliberate boundary: the mail is out, so a store error must not report it failed
            logger.warning("[Email] Could not record sent mail %s in thread %s: %s", msg_id, thread_key, exc,
                           exc_info=True)
            return False

    def _send_email(self, to_addr: str, body: str, reply_to_msg_id: Optional[str] = None,
                    thread_key: Optional[str] = None, job_id: str = "") -> str:
        """Send an email via SMTP (in thread *thread_key* when given). Runs in executor thread."""
        msg, msg_id, subject = self._new_reply(to_addr, body, reply_to_msg_id, attach_empty_body=True,
                                               thread_key=thread_key)
        self._smtp_send(msg)
        if self._after_thread_send(thread_key, msg_id, subject, job_id):
            logger.info("[Email] Sent thread root to %s (subject: %s)", to_addr, subject)
        else:
            logger.info("[Email] Sent reply to %s (subject: %s)", to_addr, subject)
        return msg_id

    def _send_with_files(self, to_addr: str, body: str, files: list[tuple[Path, str]], *, lenient: bool,
                         reply_to_msg_id: Optional[str] = None, thread_key: Optional[str] = None) -> str:
        """Send a reply with attachments; *lenient* logs-and-skips unattachable files instead of raising.
        An explicit *reply_to_msg_id* threads the mail like ``_send_email`` does (#10131)."""
        msg, msg_id, subject = self._new_reply(to_addr, body, reply_to_msg_id, thread_key=thread_key)
        for path, name in files:
            try:
                _attach_file(msg, path, name)
            except Exception as e:
                if not lenient:
                    raise
                logger.warning("[Email] Failed to attach %s: %s", path, e)
        self._smtp_send(msg)
        self._after_thread_send(thread_key, msg_id, subject, "")
        return msg_id

    async def send_image(self, chat_id: str, image_url: str, caption: Optional[str] = None,
                         reply_to: Optional[str] = None, metadata: Optional[dict[str, Any]] = None) -> SendResult:
        """Send an image URL as part of an email body (in the thread ``metadata`` names, like ``send``)."""
        return await self.send(chat_id, f"{caption or ''}\n\nImage: {image_url}".strip(), reply_to, metadata)

    async def send_multiple_images(self, chat_id: str, images: list[tuple[str, str]],
                                   metadata: Optional[dict[str, Any]] = None, human_delay: float = 0.0) -> SendResult:
        """One email per batch: local files attached, URL images linked in the body (no remote download); base-class fallback on failure."""
        if not images:
            return SendResult(success=False, error="no images to send")
        from urllib.parse import unquote as _unquote
        body_parts, local_paths = [], []
        for image_url, alt_text in images:
            if alt_text:
                body_parts.append(alt_text)
            if not image_url.startswith("file://"):
                body_parts.append(t("platform.email.image_line", url=image_url))  # parity with send_image
            elif Path(local_path := _unquote(image_url[7:])).exists():
                local_paths.append(local_path)
            else:
                logger.warning("[Email] Skipping missing image: %s", local_path)
        if not local_paths and not body_parts:
            return SendResult(success=False, error="no valid images in batch")
        try:
            send = partial(self._send_email_with_attachments, thread_key=_thread_key(metadata))
            message_id = await asyncio.get_running_loop().run_in_executor(None, send, chat_id, "\n\n".join(body_parts), local_paths)
        except Exception as e:
            logger.error("[Email] Multi-image send failed, falling back: %s", e, exc_info=True)
            return await super().send_multiple_images(chat_id, images, metadata, human_delay)
        return SendResult(success=True, message_id=message_id)

    def _send_email_with_attachments(self, to_addr: str, body: str, file_paths: list[str],
                                     thread_key: Optional[str] = None) -> str:
        """Send an email with multiple file attachments via SMTP (unattachable files are skipped)."""
        msg_id = self._send_with_files(to_addr, body, [(Path(f), Path(f).name) for f in file_paths], lenient=True,
                                       thread_key=thread_key)
        logger.info("[Email] Sent multi-attachment email to %s (%d files)", to_addr, len(file_paths))
        return msg_id

    async def send_document(self, chat_id: str, file_path: str, caption: Optional[str] = None,
                            file_name: Optional[str] = None, reply_to: Optional[str] = None,
                            metadata: Optional[dict[str, Any]] = None, **kwargs) -> SendResult:
        """Send a file as an email attachment (in the thread ``metadata`` names, like ``send``)."""
        return await self._run_send(self._send_email_with_attachment,
                                    (chat_id, caption or "", file_path, file_name, reply_to, _thread_key(metadata)),
                                    "[Email] Send document failed: %s")

    def _send_email_with_attachment(self, to_addr: str, body: str, file_path: str, file_name: Optional[str] = None,
                                    reply_to_msg_id: Optional[str] = None, thread_key: Optional[str] = None) -> str:
        """Send an email with a single file attachment via SMTP (raises if unattachable)."""
        return self._send_with_files(to_addr, body, [(Path(file_path), file_name or Path(file_path).name)], lenient=False,
                                     reply_to_msg_id=reply_to_msg_id, thread_key=thread_key)

    async def get_chat_info(self, chat_id: str) -> dict[str, Any]:
        """Return basic info about the email chat."""
        return {"name": chat_id, "type": "dm", "chat_id": chat_id, "subject": self._thread_context.get(chat_id, {}).get("subject", "")}


# Plugin glue: register() exposes the platform via the registry; EMAIL_* env → PlatformConfig seeding stays in core.
async def _standalone_send(pconfig, chat_id, message, *, thread_id=None, media_files=None, force_document=False):
    """Out-of-process Email delivery via SMTP (one-shot); standalone_sender_fn contract."""
    extra = getattr(pconfig, "extra", {}) or {}
    address, password = extra.get("address") or _get_secret("EMAIL_ADDRESS", ""), _get_secret("EMAIL_PASSWORD", "")
    smtp_host, smtp_port = extra.get("smtp_host") or _get_secret("EMAIL_SMTP_HOST", ""), _esecret_int("EMAIL_SMTP_PORT", 587)
    smtp_security = _normalize_security(_get_secret("EMAIL_SMTP_SECURITY", "") or extra.get("smtp_security"), default="tls" if smtp_port == 465 else "starttls")
    smtp_tls_verify = _esecret_bool("EMAIL_SMTP_TLS_VERIFY", is_truthy_value(extra.get("smtp_tls_verify"), default=True))
    if not all([address, password, smtp_host]):
        return send_error("Email not configured (EMAIL_ADDRESS, EMAIL_PASSWORD, EMAIL_SMTP_HOST required)")
    subject, message = _split_subject(message, t("platform.email.standalone_subject"))
    try:
        rendered = _render_html(message) if is_truthy_value(extra.get("html"), default=False) and message else None
        if rendered is None:
            msg = MIMEText(message, "plain", "utf-8")
        else:
            msg = MIMEMultipart("alternative")
            msg.attach(MIMEText(message, "plain", "utf-8"))
            msg.attach(MIMEText(rendered, "html", "utf-8"))
        for key, value in (("From", address), ("To", chat_id), ("Subject", subject), ("Date", formatdate(localtime=True))):
            msg[key] = value
        server = _open_smtp(smtp_host, smtp_port, smtp_security, _tls_context(smtp_tls_verify, smtp_host), smtplib.SMTP, smtplib.SMTP_SSL)
        server.login(address, password)
        server.send_message(msg)
        server.quit()
        return {"success": True, "platform": "email", "chat_id": chat_id}
    except Exception as e:
        try:
            from tools.send_message_tool import _error as _e
            return _e(f"Email send failed: {e}")
        except Exception:
            return send_error(f"Email send failed: {e}")


def _is_connected(config) -> bool:
    """Connected when an address is configured (PlatformConfig.extra or EMAIL_ADDRESS)."""
    if (getattr(config, "extra", {}) or {}).get("address"):
        return True
    import hermes_cli.gateway as gateway_mod
    return bool((gateway_mod.get_env_value("EMAIL_ADDRESS") or "").strip())


# System-prompt platform hint (PlatformEntry.platform_hint); cached-prompt bytes, keep stable.
_PLATFORM_HINT = (
    "You are communicating via email. Write clear, well-structured responses suitable for email. Use "
    "plain text formatting (no markdown). Keep responses concise but complete. You can send file "
    "attachments — include MEDIA:/absolute/path/to/file in your response. The subject line is preserved "
    "for threading. Do not include greetings or sign-offs unless contextually appropriate."
)


_SEND_TARGET_RE = re.compile(r"^\s*[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\s*$")


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system."""
    from . import gateway_env  # env -> GatewayConfig bridge (gateway.config_env)
    ctx.register_platform(
        target_ref_syntax_fn=regex_stripped_parser(_SEND_TARGET_RE),
        standalone_send=StandaloneSendPolicy(text_direct=True),
        name="email", label="Email", adapter_factory=EmailAdapter, check_fn=check_email_requirements, is_connected=_is_connected,
        dashboard_description="Talk to Hermes through an IMAP/SMTP mailbox.",
        docs_url="https://hermes-agent.nousresearch.com/docs/user-guide/messaging/",
        dashboard_env_vars=("EMAIL_ADDRESS", "EMAIL_PASSWORD", "EMAIL_IMAP_HOST", "EMAIL_SMTP_HOST"),
        dashboard_required_env=("EMAIL_ADDRESS", "EMAIL_PASSWORD", "EMAIL_IMAP_HOST", "EMAIL_SMTP_HOST"),
        required_env=["EMAIL_ADDRESS", "EMAIL_PASSWORD", "EMAIL_SMTP_HOST"],
        install_hint="Email uses the Python stdlib (smtplib/imaplib); the optional html part uses markdown-it-py (a core dependency)", cron_deliver_env_var="EMAIL_HOME_ADDRESS", standalone_sender_fn=_standalone_send,
        max_message_length=50_000, pii_safe=True, emoji="📧", allow_update_command=True,
        platform_hint=_PLATFORM_HINT,
        display_tier="minimal",
        env_overrides=gateway_env.ENV_OVERRIDES,
        env_enable_credentials=gateway_env.ENV_ENABLE_CREDENTIALS, **gateway_env.AUTHZ,
    )
