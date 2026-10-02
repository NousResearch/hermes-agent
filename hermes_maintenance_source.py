"""Host-minted source capabilities and the committed-source delivery outbox.

This is separate from authenticated_inbound_v1, whose exact DM/plain-text
contract remains unchanged. A transcript row is only a projection of a source.
"""

from contextlib import contextmanager, closing
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from uuid import uuid4

_SEAL = object()
_CURRENT = ContextVar("maintenance_original_input", default=None)
CONTRACT = "maintenance-source-v1"
FEISHU_TEXT_CONTRACT = "feishu-text-v2"
FEISHU_CONTROL_CONTRACT = "feishu-control-v1"
FEISHU_CONTROL_KINDS = frozenset({
    "typed_clarify_response", "explicit_queue", "explicit_steer", "retry_previous_input",
})
CLI_CONTROL_CONTRACT = "cli-control-v1"
CLI_CONTROL_KINDS = frozenset({
    "busy_queue", "busy_steer", "busy_interrupt",
    "slash_queue", "slash_steer", "retry_previous_input", "queue_edit",
    "typed_clarify_response",
})
# The v2 event-proof contract is platform-neutral; the historical names stay as
# aliases so existing Feishu rows and callers remain compatible.
PLATFORM_EVIDENCE_AUTHORITIES = {
    "feishu-human": "feishu",
    "qqbot-human": "qqbot",
    "weixin-human": "weixin",
}
PLATFORM_TEXT_CONTRACT = FEISHU_TEXT_CONTRACT
PLATFORM_CONTROL_CONTRACT = FEISHU_CONTROL_CONTRACT
PLATFORM_CONTROL_KINDS = FEISHU_CONTROL_KINDS


def _digest(value):
    return hashlib.sha256(value.encode("utf-8", "surrogatepass")).hexdigest()


_SIGNED_FIELDS = ("source_id", "authority", "origin", "home", "namespace", "actor", "audience",
                  "occurrence", "event_id", "revision", "session_id", "session_key", "message_row_id",
                  "source_time", "received_time", "timezone", "time_fallback", "raw_ref",
                  "raw_digest", "extraction_text")


def _record_digest(row):
    return _digest(json.dumps([row[key] for key in _SIGNED_FIELDS], ensure_ascii=True,
                              separators=(",", ":")))


@dataclass(frozen=True)
class SourceReceipt:
    authority: str
    origin: str
    namespace: str
    actor: str
    audience: str
    raw: str
    extraction: str
    occurrence: str
    event_id: str | None
    revision: str
    source_time: float
    received_time: float
    timezone: str
    time_fallback: bool
    raw_ref: str | None
    home: str | None
    session_key: str | None
    intended_session_id: str | None
    seal: object
    transport: object = field(default=None, repr=False, compare=False)
    chat_id: str | None = None
    chat_type: str | None = None
    display: str | None = None
    contract: str = CONTRACT
    input_kind: str | None = None
    message_type: str | None = None
    thread_id: str | None = None
    reply_to_message_id: str | None = None
    reply_to_author_id: str | None = None
    reply_to_text: str | None = None
    event_structure: str | None = None
    control_kind: str | None = None
    control_binding_json: str | None = None
    receiver: str | None = None


@dataclass(frozen=True)
class SourceBundle:
    """One displayed turn containing distinct authenticated transport occurrences."""
    receipts: tuple[SourceReceipt, ...]
    display: str
    seal: object = field(repr=False, compare=False)
    anchor_event_id: str | None = None


@dataclass(frozen=True)
class FeishuTextReceipt:
    """Adapter-sealed Feishu text/post event before gateway authorization."""
    adapter: object
    identity: tuple[str, str, str, str]
    receiver: str
    raw: str
    extraction: str
    display: str
    event_structure: str
    message_type: str
    thread_id: str | None
    reply_to_message_id: str | None
    reply_to_author_id: str | None
    reply_to_text: str | None
    seal: object = field(repr=False, compare=False)
    control_kind: str | None = None
    control_binding_json: str | None = None


def seal_feishu_text_event(
    event, adapter, *, raw, extraction, display, event_structure, message_type,
    thread_id=None, reply_to_message_id=None, reply_to_author_id=None, reply_to_text=None,
    control_kind=None, control_binding_json=None,
):
    """Adapter-only: seal one real text/post event before any model-facing assembly."""
    source = getattr(event, "source", None)
    identity = (
        getattr(getattr(source, "platform", None), "value", None),
        getattr(source, "chat_id", None), getattr(source, "user_id", None),
        getattr(event, "message_id", None),
    )
    receiver = getattr(adapter, "_bot_open_id", None)
    if (getattr(event, "internal", False) or getattr(source, "is_bot", None) is not False
            or identity[0] != "feishu" or not identity[1] or not identity[2] or not identity[3]
            or not isinstance(receiver, str) or not receiver
            or message_type not in ("text", "post")
            or not all(isinstance(v, str) and v for v in (raw, extraction, display, event_structure))
            or thread_id is not None and not isinstance(thread_id, str)
            or reply_to_message_id is not None and not isinstance(reply_to_message_id, str)
            or reply_to_author_id is not None and not isinstance(reply_to_author_id, str)
            or reply_to_text is not None and not isinstance(reply_to_text, str)
            or control_kind is not None and control_kind not in FEISHU_CONTROL_KINDS
            or control_binding_json is not None and not isinstance(control_binding_json, str)):
        return None
    receipt = FeishuTextReceipt(
        adapter=adapter, identity=identity, receiver=receiver, raw=raw, extraction=extraction,
        display=display, event_structure=event_structure, message_type=message_type,
        thread_id=thread_id, reply_to_message_id=reply_to_message_id,
        reply_to_author_id=reply_to_author_id, reply_to_text=reply_to_text, seal=_SEAL,
        control_kind=control_kind, control_binding_json=control_binding_json,
    )
    if getattr(event, "_maintenance_feishu_receipts", None) not in (None, ()):
        return None
    event._maintenance_feishu_receipts = (receipt,)
    return receipt


def mint_platform_text(*, adapter, authority, receiver, actor, chat_id, chat_type,
                       event_id, raw, source_time=None, timezone="received-time",
                       extraction=None, display=None, event_structure=None,
                       message_type="text", input_kind=None, revision="1",
                       reply_to_message_id=None, reply_to_author_id=None, reply_to_text=None):
    """Adapter-only boundary, after transport/access checks and before text rewriting.

    Omitting ``event_structure`` keeps the original lightweight CONTRACT path.
    Supplying it opts into the shared platform-event proof, used for quoted,
    mixed, and control input so the authored text remains distinct from quote
    or media-derived evidence.
    """
    if (authority not in PLATFORM_EVIDENCE_AUTHORITIES or not adapter
            or not all(isinstance(x, str) and x for x in
                       (receiver, actor, chat_id, chat_type, event_id, raw))
            or chat_type not in ("dm", "group")):
        return None
    structured = event_structure is not None
    extraction = raw if extraction is None else extraction
    display = extraction if display is None else display
    if structured:
        if (not all(isinstance(x, str) and x for x in
                    (extraction, display, event_structure))
                or message_type not in ("text", "post")
                or input_kind not in ("text", "post", "reply_reference", "mixed_items")
                or revision != "1" and not (isinstance(revision, str) and revision)
                or reply_to_message_id is not None and not isinstance(reply_to_message_id, str)
                or reply_to_author_id is not None and not isinstance(reply_to_author_id, str)
                or reply_to_text is not None and not isinstance(reply_to_text, str)):
            return None
    elif not isinstance(extraction, str) or not isinstance(display, str):
        return None
    now = time.time()
    contract = FEISHU_TEXT_CONTRACT if structured else CONTRACT
    return SourceReceipt(
        authority, "human", f"{authority}/{receiver}/{chat_type}/{chat_id}",
        actor, "private" if chat_type == "dm" else "group", raw, extraction,
        event_id, event_id, revision, source_time if source_time is not None else now,
        now, timezone, source_time is None, f"{authority}:{event_id}",
        None, None, None, _SEAL, adapter, chat_id, chat_type, display=display,
        contract=contract, input_kind=input_kind if structured else None,
        message_type=message_type if structured else None,
        reply_to_message_id=reply_to_message_id if structured else None,
        reply_to_author_id=reply_to_author_id if structured else None,
        reply_to_text=reply_to_text if structured else None,
        event_structure=event_structure if structured else None, receiver=receiver)


def _event_receiver_from_namespace(authority, namespace):
    try:
        parts = str(namespace).split("/")
    except Exception:
        return None
    return parts[1] if len(parts) >= 4 and parts[0] == authority and parts[1] else None


def _platform_namespace_matches(receipt):
    namespace = receipt.namespace or ""
    if receipt.authority == "feishu-human":
        return (bool(receipt.session_key) and namespace.startswith("feishu/")
                and namespace.endswith("/" + receipt.session_key))
    receiver = receipt.receiver or _event_receiver_from_namespace(receipt.authority, namespace)
    return bool(receiver and receipt.chat_type and receipt.chat_id
                and namespace == f"{receipt.authority}/{receiver}/{receipt.chat_type}/{receipt.chat_id}")


def _platform_raw_ref(authority, event_id):
    return (("feishu-message:" if authority == "feishu-human" else authority + ":") +
            str(event_id or ""))


def admit_platform_text(event, adapter, home, session_key):
    """Validate adapter-bound capability after gateway authorization and seal its batch."""
    receipts = getattr(event, "_maintenance_sources", ())
    if (getattr(event, "internal", False) or not isinstance(receipts, tuple)
            or not receipts or not isinstance(session_key, str) or not session_key
            or event.text != getattr(event, "_maintenance_display", None)):
        return None
    source = event.source
    platform = getattr(getattr(source, "platform", None), "value", None)
    receiver = (getattr(adapter, "_app_id", None) if platform == "qqbot"
                else getattr(adapter, "_account_id", None))
    if platform not in ("qqbot", "weixin"):
        return None
    for r in receipts:
        if (type(r) is not SourceReceipt or r.seal is not _SEAL or r.transport is not adapter
                or r.authority != platform + "-human"
                or r.receiver != receiver
                or not _platform_namespace_matches(r)
                or r.actor != source.user_id and r.chat_type == "dm"
                or r.chat_id != source.chat_id or r.chat_type != source.chat_type
                or not r.event_id or r.home is not None):
            return None
        if r.contract not in (CONTRACT, FEISHU_TEXT_CONTRACT) or (
                r.contract == FEISHU_TEXT_CONTRACT and not _platform_event_fields_valid(r)):
            return None
    if event.message_id != receipts[0].event_id or receipts[0].actor != source.user_id:
        return None
    expected = "\n".join(
        r.display if r.contract == FEISHU_TEXT_CONTRACT else r.extraction for r in receipts)
    if expected != event.text:
        return None
    bound = tuple(replace(r, home=str(Path(home).resolve()), session_key=session_key)
                  for r in receipts)
    bundle = SourceBundle(bound, event.text, _SEAL, anchor_event_id=receipts[0].event_id)
    _CURRENT.set(bundle)
    return bundle


def mint_local_input(raw, *, home, session_id, extraction=None):
    """Call only at the original local CLI caller boundary, never inside AIAgent."""
    if (not isinstance(raw, str) or not raw.strip() or not isinstance(session_id, str)
            or not session_id or not home):
        raise ValueError("Nonempty local input and trusted receiving scope required")
    now = time.time()
    return SourceReceipt("cli-local", "human", "cli/local", "local-user", "private", raw,
                         extraction if extraction is not None else raw, uuid4().hex,
                         None, "1", now, now, time.strftime("%z"), False, None,
                         str(Path(home).resolve()), None, session_id, _SEAL)


def mint_local_control(raw, *, home, session_id, kind, extraction=None, binding=None):
    """Mint one control occurrence at the local caller boundary, never from a model/tool."""
    if kind not in CLI_CONTROL_KINDS:
        raise ValueError("Invalid local control kind")
    if (not isinstance(raw, str) or not raw.strip() or not isinstance(session_id, str)
            or not session_id or not home):
        raise ValueError("Nonempty local control and trusted receiving scope required")
    extracted = raw if extraction is None else extraction
    if not isinstance(extracted, str) or not extracted.strip():
        raise ValueError("Nonempty local control extraction required")
    encoded = control_binding_json(binding)
    if kind == "typed_clarify_response" and encoded is None:
        raise ValueError("Host binding required for a typed clarify control")
    if kind != "typed_clarify_response" and encoded is not None:
        raise ValueError("Binding is only valid for a typed clarify control")
    now = time.time()
    return SourceReceipt(
        authority="cli-local", origin="human", namespace="cli/local", actor="local-user",
        audience="private", raw=raw, extraction=extracted, occurrence=uuid4().hex,
        event_id=None, revision="1", source_time=now, received_time=now,
        timezone=time.strftime("%z"), time_fallback=False, raw_ref=None,
        home=str(Path(home).resolve()), session_key=None, intended_session_id=session_id,
        seal=_SEAL, display=extracted, contract=CLI_CONTROL_CONTRACT,
        input_kind=kind, control_kind=kind, control_binding_json=encoded,
    )


def _local_control_fields_valid(receipt):
    kind = getattr(receipt, "control_kind", None)
    return (type(receipt) is SourceReceipt and receipt.seal is _SEAL
            and receipt.authority == "cli-local" and receipt.origin == "human"
            and receipt.namespace == "cli/local" and receipt.actor == "local-user"
            and receipt.audience == "private" and receipt.contract == CLI_CONTROL_CONTRACT
            and receipt.event_id is None and receipt.raw_ref is None
            and receipt.input_kind == kind and kind in CLI_CONTROL_KINDS
            and bool(receipt.raw and receipt.extraction)
            and _control_binding_valid(receipt.control_binding_json)
            and (receipt.control_binding_json is not None
                 if kind == "typed_clarify_response" else receipt.control_binding_json is None))


def admit_reused_local_source(receipt, *, home, session_id):
    """Validate an immutable CLI source reused by a queued/retry turn."""
    if (type(receipt) is not SourceReceipt or receipt.seal is not _SEAL
            or receipt.authority != "cli-local"
            or receipt.contract not in (CONTRACT, CLI_CONTROL_CONTRACT)
            or receipt.home != str(Path(home).resolve())
            or receipt.intended_session_id != session_id or receipt.event_id is not None):
        return None
    if receipt.contract == CLI_CONTROL_CONTRACT and not _local_control_fields_valid(receipt):
        return None
    return receipt


def mint_feishu_plain_text(event, adapter, home, session_key):
    """Derive only from the already sealed exact Feishu v1 transport receipt."""
    from hermes_inbound_evidence import _TransportReceipt, _SEAL as v1_seal, _identity
    transport = getattr(event, "_human_transport_receipt", None)
    if (type(transport) is not _TransportReceipt or transport.seal is not v1_seal
            or transport.adapter is not adapter or transport.identity != _identity(event)
            or transport.text != event.text or not session_key):
        return None
    now = time.time()
    s = event.source
    receiver = getattr(adapter, "_bot_open_id", None) or "unknown-receiver"
    return SourceReceipt("feishu-human", "human", "feishu/" + receiver + "/" + session_key, str(s.user_id),
                         "private", transport.text, event.text, "", str(event.message_id),
                         "1", now, now, "received-time", True, "feishu-message:" + str(event.message_id),
                         str(Path(home).resolve()), session_key, None, _SEAL)


def mint_feishu_text(event, adapter, home, session_key):
    """Mint the separate Feishu text-family path from adapter-sealed raw events."""
    seals = getattr(event, "_maintenance_feishu_receipts", ())
    text = getattr(event, "text", None)
    source = getattr(event, "source", None)
    platform = getattr(getattr(source, "platform", None), "value", None)
    receiver = getattr(adapter, "_bot_open_id", None)
    if (getattr(event, "internal", False) or not isinstance(seals, tuple) or not seals
            or not isinstance(text, str) or not text
            or text != getattr(event, "_maintenance_feishu_display", None)
            or platform != "feishu" or not isinstance(session_key, str) or not session_key
            or not isinstance(receiver, str) or not receiver):
        return None
    if any(type(seal) is not FeishuTextReceipt or seal.seal is not _SEAL
           or seal.adapter is not adapter or seal.receiver != receiver
           or seal.identity[0] != platform
           or seal.identity[1] != getattr(source, "chat_id", None)
           or (index == len(seals) - 1 and seal.identity[3] != getattr(event, "message_id", None))
           or seal.thread_id != getattr(source, "thread_id", None)
           for index, seal in enumerate(seals)):
        return None
    last = seals[-1]
    if (last.identity[2] != getattr(source, "user_id", None)
            or "\n".join(seal.display for seal in seals) != text):
        return None
    home_path = str(Path(home).resolve())
    now = time.time()
    receipts = []
    for seal in seals:
        event_id = seal.identity[3]
        chat_type = getattr(source, "chat_type", None)
        input_kind = seal.control_kind or (
            "reply_reference" if seal.reply_to_message_id else seal.message_type)
        contract = FEISHU_CONTROL_CONTRACT if seal.control_kind else FEISHU_TEXT_CONTRACT
        receipts.append(SourceReceipt(
            "feishu-human", "human", f"feishu/{receiver}/{session_key}", seal.identity[2],
            "private" if chat_type == "dm" else "group", seal.raw, seal.extraction,
            event_id, event_id, "1", now, now, "received-time", True,
            f"feishu-message:{event_id}", home_path, session_key, None, _SEAL, adapter,
            seal.identity[1], chat_type, display=seal.display, contract=contract,
            input_kind=input_kind, message_type=seal.message_type, thread_id=seal.thread_id,
            reply_to_message_id=seal.reply_to_message_id,
            reply_to_author_id=seal.reply_to_author_id, reply_to_text=seal.reply_to_text,
            event_structure=seal.event_structure, control_kind=seal.control_kind,
            control_binding_json=seal.control_binding_json, receiver=receiver,
        ))
    if len(receipts) == 1:
        return receipts[0]
    return SourceBundle(tuple(receipts), text, _SEAL, anchor_event_id=getattr(event, "message_id", None))


def with_display(source, display):
    """Attach the exact committed row projection without weakening the sealed source."""
    if not isinstance(display, str):
        return source
    if type(source) is SourceReceipt and source.contract == FEISHU_TEXT_CONTRACT:
        return replace(source, display=display)
    if (type(source) is SourceBundle
            and any(type(r) is SourceReceipt and r.contract == FEISHU_TEXT_CONTRACT
                    for r in source.receipts)):
        return replace(source, display=display)
    return source


@contextmanager
def local_input(receipt):
    if type(receipt) is not SourceReceipt or receipt.seal is not _SEAL or receipt.authority != "cli-local":
        raise ValueError("Host-minted local input required")
    token = _CURRENT.set(receipt)
    try:
        yield
    finally:
        _CURRENT.reset(token)


def current_local_input():
    return _CURRENT.get()


def admit_platform_source(receipt):
    """Admit a sealed human source from any connected platform into the turn context."""
    if (type(receipt) is SourceReceipt and receipt.seal is _SEAL
            and receipt.authority in PLATFORM_EVIDENCE_AUTHORITIES):
        _CURRENT.set(receipt)
        return True
    if (type(receipt) is SourceBundle and receipt.seal is _SEAL and receipt.receipts
          and all(type(r) is SourceReceipt and r.seal is _SEAL
                  and r.authority in PLATFORM_EVIDENCE_AUTHORITIES for r in receipt.receipts)):
        _CURRENT.set(receipt)
        return True
    _CURRENT.set(None)
    return False


def admit_feishu_source(receipt):
    if not (type(receipt) is SourceReceipt and receipt.seal is _SEAL
            and receipt.authority == "feishu-human") and not (
                type(receipt) is SourceBundle and receipt.seal is _SEAL and receipt.receipts
                and all(type(r) is SourceReceipt and r.seal is _SEAL
                        and r.authority == "feishu-human" for r in receipt.receipts)):
        _CURRENT.set(None)
        return False
    return admit_platform_source(receipt)


def reset_admitted_source():
    _CURRENT.set(None)


def current_admitted_source():
    return _CURRENT.get()


def with_extraction(receipt, extraction):
    if type(receipt) is not SourceReceipt or receipt.seal is not _SEAL or not isinstance(extraction, str):
        raise ValueError("Sealed source and extracted text required")
    return replace(receipt, extraction=extraction)


def revise_original_input(receipt, raw, *, extraction=None):
    """Trusted host edit: retain occurrence, mint a distinct immutable revision."""
    if (type(receipt) is not SourceReceipt or receipt.seal is not _SEAL
            or not isinstance(raw, str) or not raw.strip()):
        raise ValueError("Sealed original source and nonempty edited input required")
    now = time.time()
    return replace(receipt, raw=raw, extraction=extraction if extraction is not None else raw,
                   revision=str(int(receipt.revision) + 1), received_time=now)


def _ensure_outbox_columns(conn):
    """Additive migration: old pending rows stay unenrolled, old ACKs stay governed."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(maintenance_outbox_v1)")}
    if "disposition" not in columns:
        conn.execute("ALTER TABLE maintenance_outbox_v1 ADD COLUMN disposition TEXT")
        conn.execute("UPDATE maintenance_outbox_v1 SET disposition=CASE "
                     "WHEN state='ACKED' THEN 'GOVERNED' ELSE 'UNDECIDED' END")
    if "last_attempt_at" not in columns:
        conn.execute("ALTER TABLE maintenance_outbox_v1 ADD COLUMN last_attempt_at REAL NOT NULL DEFAULT 0")
    if "attempt_order" not in columns:
        conn.execute("ALTER TABLE maintenance_outbox_v1 ADD COLUMN attempt_order INTEGER NOT NULL DEFAULT 0")
    if "capture_state" not in columns:
        conn.execute("ALTER TABLE maintenance_outbox_v1 ADD COLUMN capture_state TEXT")
    if "last_error" not in columns:
        conn.execute("ALTER TABLE maintenance_outbox_v1 ADD COLUMN last_error TEXT")
    if "enrollment_enabled" not in columns:
        conn.execute("ALTER TABLE maintenance_outbox_v1 ADD COLUMN enrollment_enabled INTEGER NOT NULL DEFAULT 0")
        conn.execute("UPDATE maintenance_outbox_v1 SET enrollment_enabled=1 WHERE state='ACKED'")


def _intake_enabled_at_commit(home):
    """Persist the intake window at source commit; later enablement cannot enroll old input."""
    try:
        config = json.loads((Path(home) / "mem0.json").read_text(encoding="utf-8-sig"))
        governed = config.get("governed")
        return isinstance(governed, dict) and governed.get("maintenance_intake") is True
    except (OSError, ValueError, TypeError):
        return False


_FEISHU_PROOF_COLUMNS = (
    "contract", "session_id", "session_key", "message_row_id", "event_id", "actor",
    "chat_id", "chat_type", "thread_id", "input_kind", "event_type", "display_text",
    "display_digest", "extraction_digest", "raw_digest", "event_structure",
    "event_structure_digest", "reply_to_message_id", "reply_to_author_id", "reply_to_text",
    "control_kind", "control_binding_json",
)


def _ensure_feishu_text_table(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS maintenance_source_events_v1 (
        source_id TEXT PRIMARY KEY REFERENCES maintenance_sources_v1(source_id),
        contract TEXT NOT NULL, session_id TEXT NOT NULL, session_key TEXT NOT NULL,
        message_row_id INTEGER NOT NULL, event_id TEXT NOT NULL, actor TEXT NOT NULL,
        chat_id TEXT NOT NULL, chat_type TEXT NOT NULL, thread_id TEXT,
        input_kind TEXT NOT NULL, event_type TEXT NOT NULL, display_text TEXT NOT NULL,
        display_digest TEXT NOT NULL, extraction_digest TEXT NOT NULL, raw_digest TEXT NOT NULL,
        event_structure TEXT NOT NULL, event_structure_digest TEXT NOT NULL,
        reply_to_message_id TEXT, reply_to_author_id TEXT, reply_to_text TEXT,
        control_kind TEXT, control_binding_json TEXT, proof_digest TEXT NOT NULL)""")
    columns = {row[1] for row in conn.execute("PRAGMA table_info(maintenance_source_events_v1)")}
    migrated = False
    if "control_kind" not in columns:
        conn.execute("ALTER TABLE maintenance_source_events_v1 ADD COLUMN control_kind TEXT")
        migrated = True
    if "control_binding_json" not in columns:
        conn.execute("ALTER TABLE maintenance_source_events_v1 ADD COLUMN control_binding_json TEXT")
        migrated = True
    if migrated:
        rows = [dict(row) for row in conn.execute("SELECT * FROM maintenance_source_events_v1")]
        for row in rows:
            conn.execute("UPDATE maintenance_source_events_v1 SET proof_digest=? WHERE source_id=?",
                         (_feishu_proof_digest(row), row["source_id"]))


def _feishu_proof_digest(row):
    values = dict(row)
    return _digest(json.dumps([values[key] for key in _FEISHU_PROOF_COLUMNS],
                              ensure_ascii=True, separators=(",", ":")))


def _feishu_proof_values(receipt, session_id, row_id, display_text):
    structure = receipt.event_structure or ""
    return {
        "contract": receipt.contract,
        "session_id": session_id,
        "session_key": receipt.session_key,
        "message_row_id": row_id,
        "event_id": receipt.event_id,
        "actor": receipt.actor,
        "chat_id": receipt.chat_id,
        "chat_type": receipt.chat_type,
        "thread_id": receipt.thread_id,
        "input_kind": receipt.input_kind or receipt.message_type or "text",
        "event_type": receipt.message_type or "text",
        "display_text": display_text,
        "display_digest": _digest(display_text),
        "extraction_digest": _digest(receipt.extraction),
        "raw_digest": _digest(receipt.raw),
        "event_structure": structure,
        "event_structure_digest": _digest(structure),
        "reply_to_message_id": receipt.reply_to_message_id,
        "reply_to_author_id": receipt.reply_to_author_id,
        "reply_to_text": receipt.reply_to_text,
        "control_kind": receipt.control_kind,
        "control_binding_json": receipt.control_binding_json,
    }


def _insert_feishu_text_proof(conn, source_id, receipt, session_id, row_id, display_text):
    values = _feishu_proof_values(receipt, session_id, row_id, display_text)
    values["proof_digest"] = _feishu_proof_digest(values)
    _ensure_feishu_text_table(conn)
    columns = ("source_id", *_FEISHU_PROOF_COLUMNS, "proof_digest")
    conn.execute(
        f"INSERT INTO maintenance_source_events_v1 ({','.join(columns)}) "
        f"VALUES ({','.join('?' for _ in columns)})",
        tuple(source_id if column == "source_id" else values[column] for column in columns),
    )


def _control_binding_valid(encoded):
    if encoded is None:
        return True
    try:
        value = json.loads(encoded)
    except (TypeError, ValueError):
        return False
    return (isinstance(value, dict)
            and set(value) == {"task_id", "item_id", "proposal_revision"}
            and all(isinstance(item, str) and item for item in value.values()))


def control_binding_json(binding):
    """Canonical host-side binding; text/model values are never accepted here."""
    encoded = None if binding is None else json.dumps(
        binding, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    if not _control_binding_valid(encoded):
        raise ValueError("Invalid maintenance control binding")
    return encoded


HOST_QUESTION_STATES = frozenset({"PREPARED", "DELIVERED", "RESOLVED", "CANCELLED"})


def _ensure_host_question_table(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS maintenance_host_questions_v1 (
        prompt_id TEXT PRIMARY KEY,
        owner_id TEXT NOT NULL,
        session_key TEXT,
        question TEXT NOT NULL,
        choices_json TEXT,
        control_binding_json TEXT NOT NULL,
        state TEXT NOT NULL CHECK(state IN ('PREPARED','DELIVERED','RESOLVED','CANCELLED')),
        created_at REAL NOT NULL,
        expires_at REAL NOT NULL,
        channel TEXT,
        message_ref TEXT,
        delivered_at REAL,
        delivery_receipt_json TEXT)""")
    conn.execute("""CREATE INDEX IF NOT EXISTS maintenance_host_questions_session
        ON maintenance_host_questions_v1(session_key,state)""")


def _host_question_id(value):
    return isinstance(value, str) and bool(value) and len(value) <= 256


def begin_host_question_delivery(
    profile_home, prompt_id, owner_id, session_key, question, choices,
    control_binding, *, expires_at,
):
    """Persist a host-owned prompt before any channel send.

    The row is not answerable until ``record_host_question_delivery`` commits a
    receipt from the real host channel. This function is deliberately not
    exposed to model/tool content: callers must supply a host-minted prompt and
    binding.
    """
    if not all((_host_question_id(prompt_id), _host_question_id(owner_id),
                _host_question_id(session_key))):
        raise ValueError("Invalid host question identity")
    if (not isinstance(question, str) or not question.strip()
            or len(question) > 4096):
        raise ValueError("Invalid host question text")
    if (choices is not None and (not isinstance(choices, (list, tuple))
                                  or len(choices) > 16
                                  or any(not isinstance(choice, str) or not choice.strip()
                                         or len(choice) > 256 for choice in choices))):
        raise ValueError("Invalid host question choices")
    encoded_binding = control_binding_json(control_binding)
    if encoded_binding is None:
        raise ValueError("Host question binding required")
    now = time.time()
    if type(expires_at) not in (int, float) or expires_at <= now or expires_at > now + 30 * 24 * 60 * 60:
        raise ValueError("Invalid host question expiry")
    encoded_choices = None if choices is None else json.dumps(
        list(choices), ensure_ascii=True, separators=(",", ":"))
    home = Path(profile_home).resolve()
    with closing(sqlite3.connect((home / "state.db").as_uri() + "?mode=rw", uri=True, timeout=5)) as db:
        db.row_factory = sqlite3.Row
        db.execute("BEGIN IMMEDIATE")
        _ensure_host_question_table(db)
        prior = db.execute(
            "SELECT * FROM maintenance_host_questions_v1 WHERE prompt_id=?", (prompt_id,)).fetchone()
        if prior is not None:
            expected = (owner_id, session_key, question, encoded_choices, encoded_binding)
            actual = (prior["owner_id"], prior["session_key"], prior["question"],
                      prior["choices_json"], prior["control_binding_json"])
            if actual != expected:
                db.rollback()
                raise ValueError("Host question conflict")
            if prior["state"] in ("RESOLVED", "CANCELLED"):
                db.rollback()
                raise ValueError("Host question closed")
            db.commit()
            return prompt_id
        db.execute("""INSERT INTO maintenance_host_questions_v1
            (prompt_id,owner_id,session_key,question,choices_json,control_binding_json,
             state,created_at,expires_at)
            VALUES (?,?,?,?,?,?,?,?,?)""",
            (prompt_id, owner_id, session_key, question, encoded_choices,
             encoded_binding, "PREPARED", now, float(expires_at)))
        db.commit()
    return prompt_id


def record_host_question_delivery(
    profile_home, prompt_id, *, channel, session_key, message_ref, delivered_at=None,
):
    """Commit delivery only after the host channel accepted the real send."""
    if not all((_host_question_id(prompt_id), _host_question_id(channel),
                _host_question_id(session_key), _host_question_id(message_ref))):
        raise ValueError("Invalid host delivery receipt")
    explicit_time = delivered_at is not None
    when = time.time() if delivered_at is None else delivered_at
    if type(when) not in (int, float):
        raise ValueError("Invalid host delivery time")
    receipt = {"channel": channel, "session_key": session_key,
               "message_ref": message_ref, "delivered_at": float(when)}
    encoded = json.dumps(receipt, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    home = Path(profile_home).resolve()
    with closing(sqlite3.connect((home / "state.db").as_uri() + "?mode=rw", uri=True, timeout=5)) as db:
        db.row_factory = sqlite3.Row
        db.execute("BEGIN IMMEDIATE")
        _ensure_host_question_table(db)
        row = db.execute(
            "SELECT * FROM maintenance_host_questions_v1 WHERE prompt_id=?", (prompt_id,)).fetchone()
        if row is None or row["session_key"] != session_key:
            db.rollback()
            raise ValueError("Host question unavailable")
        if row["state"] in ("RESOLVED", "CANCELLED"):
            db.rollback()
            raise ValueError("Host question closed")
        if row["state"] == "DELIVERED":
            prior = json.loads(row["delivery_receipt_json"])
            same_target = (prior.get("channel") == channel
                           and prior.get("session_key") == session_key
                           and prior.get("message_ref") == message_ref)
            same_time = (not explicit_time
                         or float(prior.get("delivered_at", -1)) == float(when))
            if same_target and same_time:
                db.commit()
                return prior
            db.rollback()
            raise ValueError("Host delivery already recorded")
        if float(row["expires_at"]) <= float(when):
            db.rollback()
            raise ValueError("Host question expired")
        db.execute("""UPDATE maintenance_host_questions_v1
            SET state='DELIVERED',channel=?,message_ref=?,delivered_at=?,delivery_receipt_json=?
            WHERE prompt_id=? AND state='PREPARED'""",
            (channel, message_ref, float(when), encoded, prompt_id))
        db.commit()
    return dict(receipt)


def host_question_delivery(profile_home, prompt_id):
    """Read one host delivery row; no channel or answer authority is implied."""
    home = Path(profile_home).resolve()
    try:
        with closing(sqlite3.connect((home / "state.db").as_uri() + "?mode=ro", uri=True)) as db:
            db.row_factory = sqlite3.Row
            row = db.execute(
                "SELECT * FROM maintenance_host_questions_v1 WHERE prompt_id=?", (prompt_id,)).fetchone()
            return dict(row) if row is not None else None
    except sqlite3.OperationalError:
        return None


def host_question_delivery_for_session(profile_home, session_key, *, now=None):
    """Return the one answerable prompt for a session and the candidate count.

    Zero or multiple candidates return ``(None, count)`` so callers fail closed
    rather than guessing which private proposal a bare reply refers to.
    """
    if not _host_question_id(session_key):
        raise ValueError("Invalid host question session")
    current = time.time() if now is None else now
    if type(current) not in (int, float):
        raise ValueError("Invalid host question clock")
    home = Path(profile_home).resolve()
    try:
        with closing(sqlite3.connect((home / "state.db").as_uri() + "?mode=ro", uri=True)) as db:
            db.row_factory = sqlite3.Row
            rows = db.execute("""SELECT * FROM maintenance_host_questions_v1
                WHERE session_key=? AND state='DELIVERED' AND expires_at>?
                ORDER BY delivered_at,prompt_id""", (session_key, float(current))).fetchall()
    except sqlite3.OperationalError:
        return None, 0
    candidates = [dict(row) for row in rows]
    return (candidates[0] if len(candidates) == 1 else None), len(candidates)


def close_host_question_delivery(profile_home, prompt_id, state):
    """Close a host delivery binding without touching plugin task state."""
    if state not in ("RESOLVED", "CANCELLED"):
        raise ValueError("Invalid host question close state")
    home = Path(profile_home).resolve()
    with closing(sqlite3.connect((home / "state.db").as_uri() + "?mode=rw", uri=True, timeout=5)) as db:
        db.row_factory = sqlite3.Row
        db.execute("BEGIN IMMEDIATE")
        _ensure_host_question_table(db)
        row = db.execute(
            "SELECT state FROM maintenance_host_questions_v1 WHERE prompt_id=?", (prompt_id,)).fetchone()
        if row is None or row["state"] in ("RESOLVED", "CANCELLED") and row["state"] != state:
            db.rollback()
            raise ValueError("Host question unavailable")
        db.execute("UPDATE maintenance_host_questions_v1 SET state=? WHERE prompt_id=?",
                   (state, prompt_id))
        db.commit()
    return state


def _platform_event_fields_valid(receipt):
    platform = PLATFORM_EVIDENCE_AUTHORITIES.get(receipt.authority)
    return (type(receipt) is SourceReceipt and receipt.seal is _SEAL
            and platform is not None
            and receipt.contract in (FEISHU_TEXT_CONTRACT, FEISHU_CONTROL_CONTRACT)
            and receipt.origin == "human" and receipt.chat_type in ("dm", "group", "forum", "thread")
            and receipt.audience == ("private" if receipt.chat_type == "dm" else "group")
            and bool(receipt.actor and receipt.chat_id and receipt.event_id)
            and receipt.message_type in ("text", "post")
            and (receipt.input_kind in ("text", "post", "reply_reference", "mixed_items")
                 if receipt.contract == FEISHU_TEXT_CONTRACT else
                 receipt.input_kind == receipt.control_kind
                 and receipt.control_kind in FEISHU_CONTROL_KINDS
                 and _control_binding_valid(receipt.control_binding_json))
            and bool(receipt.raw and receipt.extraction and receipt.display and receipt.event_structure)
            and _platform_namespace_matches(receipt)
            and receipt.raw_ref == _platform_raw_ref(receipt.authority, receipt.event_id))


def _platform_event_binding_matches(receipt, session, row, batch_anchor, display_text):
    platform = PLATFORM_EVIDENCE_AUTHORITIES.get(receipt.authority)
    if not _platform_event_fields_valid(receipt) or display_text is None or platform is None:
        return False
    if (session["source"] != platform or session["session_key"] != receipt.session_key
            or session["chat_id"] != receipt.chat_id or session["chat_type"] != receipt.chat_type
            or (session["thread_id"] or "") != (receipt.thread_id or "")):
        return False
    if receipt.chat_type == "dm" and session["user_id"] != receipt.actor:
        return False
    if row["content"] != display_text or row["platform_message_id"] != (batch_anchor or receipt.event_id):
        return False
    return _platform_namespace_matches(receipt)


def _feishu_proof_matches(conn, source_id, receipt, session_id, row_id, display_text, *, require_row):
    proof = conn.execute("SELECT * FROM maintenance_source_events_v1 WHERE source_id=?",
                         (source_id,)).fetchone()
    if proof is None:
        return False
    proof = dict(proof)
    if (proof["contract"] not in (FEISHU_TEXT_CONTRACT, FEISHU_CONTROL_CONTRACT)
            or _feishu_proof_digest(proof) != proof["proof_digest"]):
        return False
    expected = _feishu_proof_values(receipt, session_id, row_id, display_text)
    for key in _FEISHU_PROOF_COLUMNS:
        if key == "message_row_id" and not require_row:
            continue
        if proof[key] != expected[key]:
            return False
    return True


def _ensure_source_projections(conn):
    if conn.execute("SELECT 1 FROM sqlite_master WHERE name='maintenance_source_projections_v1'").fetchone():
        indexes = conn.execute("PRAGMA index_list(maintenance_source_projections_v1)").fetchall()
        if not any(index[2] and [column[2] for column in conn.execute(
                f"PRAGMA index_info({index[1]})")] == ["message_row_id"] for index in indexes):
            return
        conn.execute("ALTER TABLE maintenance_source_projections_v1 RENAME TO maintenance_source_projections_old")
    conn.execute("""CREATE TABLE maintenance_source_projections_v1 (
        source_id TEXT NOT NULL, session_id TEXT NOT NULL, message_row_id INTEGER NOT NULL,
        PRIMARY KEY(source_id,session_id,message_row_id))""")
    prior = "maintenance_source_projections_old" if conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name='maintenance_source_projections_old'").fetchone() else "maintenance_sources_v1"
    conn.execute("INSERT INTO maintenance_source_projections_v1 "
                 f"SELECT source_id,session_id,message_row_id FROM {prior}")
    if prior == "maintenance_source_projections_old":
        conn.execute("DROP TABLE maintenance_source_projections_old")


def bind_committed_source(conn, db_path, session_id, row_id, receipt):
    """Insert immutable evidence and outbox inside the message writer transaction."""
    if type(receipt) is SourceBundle and receipt.seal is _SEAL:
        if not receipt.receipts or not all(type(r) is SourceReceipt and r.seal is _SEAL
                                           for r in receipt.receipts):
            return None
        feishu_text = any(r.contract == FEISHU_TEXT_CONTRACT for r in receipt.receipts)
        if feishu_text and not all(r.contract == FEISHU_TEXT_CONTRACT for r in receipt.receipts):
            return None
        if not feishu_text and "\n".join(r.extraction for r in receipt.receipts) != receipt.display:
            return None
        anchor = receipt.anchor_event_id or receipt.receipts[0].event_id
        if anchor not in {r.event_id for r in receipt.receipts}:
            return None
        row = conn.execute("SELECT content,platform_message_id FROM messages WHERE id=? AND session_id=?",
                           (row_id, session_id)).fetchone()
        if row is None or row["content"] != receipt.display or row["platform_message_id"] != anchor:
            return None
        ids = []
        conn.execute("SAVEPOINT maintenance_bundle")
        try:
            for item in receipt.receipts:
                ids.append(_bind_one_source(conn, db_path, session_id, row_id, item,
                                            batch_anchor=anchor,
                                            batch_display=receipt.display,
                                            batch_position=len(ids), batch_size=len(receipt.receipts)))
            if not all(ids):
                conn.execute("ROLLBACK TO maintenance_bundle")
                return None
            return ids
        except Exception:
            conn.execute("ROLLBACK TO maintenance_bundle")
            raise
        finally:
            conn.execute("RELEASE maintenance_bundle")
    return _bind_one_source(conn, db_path, session_id, row_id, receipt)


def _project_local_control_source(conn, db_path, session_id, row_id, receipt):
    """Project an already-committed CLI control onto its consumed turn without re-minting it."""
    if not _local_control_fields_valid(receipt):
        return None
    home = str(Path(db_path).resolve().parent)
    if receipt.home != home or receipt.intended_session_id != session_id:
        return None
    row = conn.execute("SELECT role,content,platform_message_id FROM messages "
                       "WHERE id=? AND session_id=?", (row_id, session_id)).fetchone()
    session = conn.execute("SELECT source FROM sessions WHERE id=?", (session_id,)).fetchone()
    if (row is None or session is None or session["source"] != "cli" or row["role"] != "user"
            or not isinstance(row["content"], str) or not row["content"]
            or row["platform_message_id"] is not None):
        return None
    source_id = _digest(json.dumps([home, receipt.namespace, receipt.occurrence, receipt.revision,
                                    "maintenance"], separators=(",", ":"), ensure_ascii=True))
    source = conn.execute("SELECT raw_digest,actor,home FROM maintenance_sources_v1 WHERE source_id=?",
                          (source_id,)).fetchone()
    controls = conn.execute(
        "SELECT control_kind,control_binding_json,control_digest "
        "FROM maintenance_local_controls_v1 WHERE source_id=?", (source_id,)).fetchone()
    if (source is None or controls is None
            or (source["raw_digest"], source["actor"], source["home"]) != (
                _digest(receipt.raw), receipt.actor, home)
            or not _local_control_row_matches(source_id, controls["control_kind"],
                                              controls["control_binding_json"],
                                              controls["control_digest"])
            or controls["control_kind"] != receipt.control_kind
            or controls["control_binding_json"] != receipt.control_binding_json):
        return None
    _ensure_source_projections(conn)
    conn.execute("INSERT OR IGNORE INTO maintenance_source_projections_v1 VALUES (?,?,?)",
                 (source_id, session_id, row_id))
    return source_id


def _bind_one_source(conn, db_path, session_id, row_id, receipt, *,
                     batch_anchor=None, batch_display=None, batch_position=0, batch_size=1):
    if type(receipt) is not SourceReceipt or receipt.seal is not _SEAL:
        return None
    if receipt.contract == FEISHU_CONTROL_CONTRACT:
        return None
    if receipt.contract == CLI_CONTROL_CONTRACT:
        return _project_local_control_source(conn, db_path, session_id, row_id, receipt)
    is_platform_event = (receipt.authority in PLATFORM_EVIDENCE_AUTHORITIES
                         and receipt.contract == FEISHU_TEXT_CONTRACT)
    platform = {"cli-local": "cli", "feishu-human": "feishu", "qqbot-human": "qqbot",
                "weixin-human": "weixin"}.get(receipt.authority)
    if (receipt.origin != "human" or platform is None
            or receipt.audience not in ("private", "group")
            or receipt.audience == "group" and platform not in ("qqbot", "weixin")
            and not is_platform_event):
        return None
    home = str(Path(db_path).resolve().parent)
    if receipt.home != home or is_platform_event and not _platform_event_fields_valid(receipt):
        return None
    display_text = batch_display if batch_display is not None else (
        receipt.display if is_platform_event else receipt.extraction)
    if not isinstance(display_text, str):
        return None
    row = conn.execute("SELECT role,content,display_kind,_compressed_summary,platform_message_id "
                       "FROM messages WHERE id=? AND session_id=?", (row_id, session_id)).fetchone()
    session = conn.execute("SELECT session_key,source,user_id,chat_id,chat_type,thread_id "
                           "FROM sessions WHERE id=?", (session_id,)).fetchone()
    if (row is None or session is None or row["role"] != "user"
            or row["display_kind"] not in (None, "") or row["_compressed_summary"]
            or row["content"] != display_text):
        return None
    occurrence = receipt.event_id or receipt.occurrence
    source_id = _digest(json.dumps([home, receipt.namespace, occurrence, receipt.revision, "maintenance"],
                                   separators=(",", ":"), ensure_ascii=True))
    existing_table = conn.execute("SELECT 1 FROM sqlite_master WHERE name='maintenance_sources_v1'").fetchone()
    if existing_table:
        existing = conn.execute("SELECT raw_digest,actor,home FROM maintenance_sources_v1 WHERE source_id=?",
                                (source_id,)).fetchone()
        if existing is not None:
            if (existing["raw_digest"], existing["actor"], existing["home"]) != (
                    _digest(receipt.raw), receipt.actor, home):
                raise ValueError("Conflicting committed source occurrence")
            if is_platform_event:
                if (not _platform_event_binding_matches(receipt, session, row, batch_anchor, display_text)
                        or not _feishu_proof_matches(conn, source_id, receipt, session_id, row_id,
                                                     display_text, require_row=False)):
                    return None
            else:
                if session["source"] != platform or row["platform_message_id"] != (batch_anchor or receipt.event_id):
                    return None
                if receipt.authority == "feishu-human":
                    original = conn.execute(
                        "SELECT source,user_id,chat_id,chat_type FROM sessions "
                        "WHERE id=(SELECT session_id FROM maintenance_sources_v1 WHERE source_id=?)",
                        (source_id,)).fetchone()
                    if original is None or any(session[key] != original[key] for key in
                                               ("source", "user_id", "chat_id", "chat_type")):
                        return None
            _ensure_source_projections(conn)
            conn.execute("INSERT OR IGNORE INTO maintenance_source_projections_v1 VALUES (?,?,?)",
                         (source_id, session_id, row_id))
            return source_id  # A transcript clone references the original authority.
    if (receipt.session_key is not None and session["session_key"] != receipt.session_key
            or receipt.authority == "cli-local" and (session["source"] != "cli"
                or receipt.namespace != "cli/local" or receipt.actor != "local-user"
                or receipt.intended_session_id != session_id or row["platform_message_id"] is not None)
            or receipt.authority == "feishu-human" and not is_platform_event and (
                session["source"] != "feishu" or session["user_id"] != receipt.actor
                or session["chat_type"] != "dm" or row["platform_message_id"] != receipt.event_id
                or not isinstance(receipt.event_id, str) or not receipt.event_id
                or not receipt.namespace.startswith("feishu/")
                or not receipt.namespace.endswith("/" + session["session_key"])
                or receipt.raw_ref != "feishu-message:" + receipt.event_id
                or not session["chat_id"])
            or is_platform_event and not _platform_event_binding_matches(
                receipt, session, row, batch_anchor, display_text)
            or platform in ("qqbot", "weixin") and not is_platform_event and (
                session["source"] != platform or session["chat_id"] != receipt.chat_id
                or session["chat_type"] != receipt.chat_type
                or receipt.audience != ("group" if receipt.chat_type == "group" else "private")
                or receipt.chat_type == "dm" and session["user_id"] != receipt.actor
                or row["platform_message_id"] != (batch_anchor or receipt.event_id)
                or not receipt.event_id or not receipt.namespace.startswith(receipt.authority + "/")
                or not receipt.namespace.endswith("/" + receipt.chat_type + "/" + receipt.chat_id)
                or receipt.raw_ref != receipt.authority + ":" + receipt.event_id)):
        return None
    if receipt.authority == "feishu-human" and not is_platform_event:
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='authenticated_inbound_v1'").fetchone():
            return None
        proof = conn.execute("SELECT source,chat_id,user_id,platform_message_id,revoked FROM authenticated_inbound_v1 "
                             "WHERE message_row_id=? AND session_id=?", (row_id, session_id)).fetchone()
        if (proof is None or proof["revoked"] or
                (proof["source"], proof["chat_id"], proof["user_id"], proof["platform_message_id"]) !=
                ("feishu", session["chat_id"], receipt.actor, receipt.event_id)):
            return None
    conn.execute("""CREATE TABLE IF NOT EXISTS maintenance_sources_v1 (
        source_id TEXT PRIMARY KEY, authority TEXT NOT NULL, origin TEXT NOT NULL, home TEXT NOT NULL,
        namespace TEXT NOT NULL, actor TEXT NOT NULL, audience TEXT NOT NULL, occurrence TEXT NOT NULL,
        event_id TEXT, revision TEXT NOT NULL, session_id TEXT NOT NULL, session_key TEXT NOT NULL,
        message_row_id INTEGER NOT NULL, source_time REAL NOT NULL, received_time REAL NOT NULL,
        timezone TEXT NOT NULL, time_fallback INTEGER NOT NULL, raw_ref TEXT, raw_text TEXT NOT NULL,
        raw_digest TEXT NOT NULL, extraction_text TEXT NOT NULL, record_digest TEXT NOT NULL,
        revoked INTEGER NOT NULL DEFAULT 0)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS maintenance_outbox_v1 (
        source_id TEXT PRIMARY KEY REFERENCES maintenance_sources_v1(source_id),
        state TEXT NOT NULL DEFAULT 'PENDING' CHECK(state IN ('PENDING','ACKED')),
        task_id TEXT, acknowledged_at REAL)""")
    _ensure_outbox_columns(conn)
    conn.execute("""INSERT INTO maintenance_sources_v1 VALUES
        (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0)""",
        (source_id, receipt.authority, receipt.origin, home, receipt.namespace, receipt.actor,
         receipt.audience, receipt.occurrence, receipt.event_id, receipt.revision, session_id,
         session["session_key"], row_id, receipt.source_time, receipt.received_time, receipt.timezone,
         int(receipt.time_fallback), receipt.raw_ref,
         sqlite3.Binary(receipt.raw.encode("utf-8", "surrogatepass")),
         _digest(receipt.raw), receipt.extraction, ""))
    signed = conn.execute("SELECT * FROM maintenance_sources_v1 WHERE source_id=?", (source_id,)).fetchone()
    conn.execute("UPDATE maintenance_sources_v1 SET record_digest=? WHERE source_id=?",
                 (_record_digest(signed), source_id))
    conn.execute("INSERT INTO maintenance_outbox_v1(source_id,disposition,enrollment_enabled) "
                 "VALUES (?,'UNDECIDED',?)", (source_id, int(_intake_enabled_at_commit(home))))
    if is_platform_event:
        _insert_feishu_text_proof(conn, source_id, receipt, session_id, row_id, display_text)
    if batch_anchor is not None:
        conn.execute("CREATE TABLE IF NOT EXISTS maintenance_source_batches_v1 ("
                     "source_id TEXT PRIMARY KEY, anchor_event_id TEXT NOT NULL, "
                     "display_digest TEXT NOT NULL, position INTEGER NOT NULL, size INTEGER NOT NULL)")
        conn.execute("INSERT INTO maintenance_source_batches_v1 VALUES (?,?,?,?,?)",
                     (source_id, batch_anchor, _digest(batch_display), batch_position, batch_size))
    _ensure_source_projections(conn)
    conn.execute("INSERT OR IGNORE INTO maintenance_source_projections_v1 VALUES (?,?,?)",
                 (source_id, session_id, row_id))
    return source_id


def _local_control_digest(source_id, kind, binding_json):
    return _digest(json.dumps([source_id, kind, binding_json], ensure_ascii=True,
                              separators=(",", ":")))


def _ensure_local_control_table(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS maintenance_local_controls_v1 (
        source_id TEXT PRIMARY KEY REFERENCES maintenance_sources_v1(source_id),
        control_kind TEXT NOT NULL, control_binding_json TEXT, control_digest TEXT NOT NULL)""")


def _local_control_row_matches(source_id, kind, binding_json, digest):
    return (kind in CLI_CONTROL_KINDS and _control_binding_valid(binding_json)
            and (binding_json is not None if kind == "typed_clarify_response" else binding_json is None)
            and digest == _local_control_digest(source_id, kind, binding_json))


def commit_local_control_source(profile_home, session_id, receipt):
    """Commit one CLI-local control without inventing a platform message id or transcript row."""
    if not _local_control_fields_valid(receipt):
        return None
    home = Path(profile_home).resolve()
    if receipt.home != str(home) or receipt.intended_session_id != session_id:
        return None
    occurrence = receipt.occurrence
    source_id = _digest(json.dumps([str(home), receipt.namespace, occurrence, receipt.revision,
                                    "maintenance"], separators=(",", ":"), ensure_ascii=True))
    with closing(sqlite3.connect((home / "state.db").as_uri() + "?mode=rw", uri=True, timeout=5)) as db:
        db.row_factory = sqlite3.Row
        db.execute("BEGIN IMMEDIATE")
        session = db.execute("SELECT session_key,source FROM sessions WHERE id=?", (session_id,)).fetchone()
        if session is None or session["source"] != "cli":
            db.rollback()
            return None
        has_sources = db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='maintenance_sources_v1'").fetchone()
        if has_sources:
            existing = db.execute("SELECT raw_digest,actor,home FROM maintenance_sources_v1 WHERE source_id=?",
                                  (source_id,)).fetchone()
            if existing is not None:
                if (existing["raw_digest"], existing["actor"], existing["home"]) != (
                        _digest(receipt.raw), receipt.actor, str(home)):
                    db.rollback()
                    raise ValueError("Conflicting committed source occurrence")
                has_controls = db.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' "
                    "AND name='maintenance_local_controls_v1'").fetchone()
                control = (db.execute("SELECT control_kind,control_binding_json,control_digest "
                                      "FROM maintenance_local_controls_v1 WHERE source_id=?",
                                      (source_id,)).fetchone() if has_controls else None)
                if (control is None or not _local_control_row_matches(
                        source_id, control["control_kind"], control["control_binding_json"],
                        control["control_digest"])
                        or control["control_kind"] != receipt.control_kind
                        or control["control_binding_json"] != receipt.control_binding_json):
                    db.rollback()
                    return None
                db.commit()
                return source_id
        db.execute("""CREATE TABLE IF NOT EXISTS maintenance_sources_v1 (
            source_id TEXT PRIMARY KEY, authority TEXT NOT NULL, origin TEXT NOT NULL, home TEXT NOT NULL,
            namespace TEXT NOT NULL, actor TEXT NOT NULL, audience TEXT NOT NULL, occurrence TEXT NOT NULL,
            event_id TEXT, revision TEXT NOT NULL, session_id TEXT NOT NULL, session_key TEXT NOT NULL,
            message_row_id INTEGER NOT NULL, source_time REAL NOT NULL, received_time REAL NOT NULL,
            timezone TEXT NOT NULL, time_fallback INTEGER NOT NULL, raw_ref TEXT, raw_text TEXT NOT NULL,
            raw_digest TEXT NOT NULL, extraction_text TEXT NOT NULL, record_digest TEXT NOT NULL,
            revoked INTEGER NOT NULL DEFAULT 0)""")
        db.execute("""CREATE TABLE IF NOT EXISTS maintenance_outbox_v1 (
            source_id TEXT PRIMARY KEY REFERENCES maintenance_sources_v1(source_id),
            state TEXT NOT NULL DEFAULT 'PENDING' CHECK(state IN ('PENDING','ACKED')),
            task_id TEXT, acknowledged_at REAL)""")
        _ensure_outbox_columns(db)
        _ensure_local_control_table(db)
        db.execute("""INSERT INTO maintenance_sources_v1 VALUES
            (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0)""",
            (source_id, receipt.authority, receipt.origin, str(home), receipt.namespace, receipt.actor,
             receipt.audience, receipt.occurrence, receipt.event_id, receipt.revision, session_id,
             session["session_key"] or session_id, 0, receipt.source_time, receipt.received_time,
             receipt.timezone, int(receipt.time_fallback), receipt.raw_ref,
             sqlite3.Binary(receipt.raw.encode("utf-8", "surrogatepass")),
             _digest(receipt.raw), receipt.extraction, ""))
        signed = db.execute("SELECT * FROM maintenance_sources_v1 WHERE source_id=?", (source_id,)).fetchone()
        db.execute("UPDATE maintenance_sources_v1 SET record_digest=? WHERE source_id=?",
                   (_record_digest(signed), source_id))
        db.execute("INSERT INTO maintenance_local_controls_v1 VALUES (?,?,?,?)",
                   (source_id, receipt.control_kind, receipt.control_binding_json,
                    _local_control_digest(source_id, receipt.control_kind, receipt.control_binding_json)))
        disposition = "CONTROL" if receipt.control_kind == "typed_clarify_response" else "UNDECIDED"
        db.execute("INSERT INTO maintenance_outbox_v1(source_id,disposition,enrollment_enabled) "
                   "VALUES (?,?,?)", (source_id, disposition,
                    int(receipt.control_kind != "typed_clarify_response" and _intake_enabled_at_commit(home))))
        db.commit()
    return source_id


def commit_platform_control_source(profile_home, session_id, receipt):
    """Commit one authenticated control occurrence without treating it as transcript text."""
    platform = PLATFORM_EVIDENCE_AUTHORITIES.get(getattr(receipt, "authority", None))
    if (type(receipt) is not SourceReceipt or receipt.seal is not _SEAL
            or receipt.contract != FEISHU_CONTROL_CONTRACT or platform is None
            or not _platform_event_fields_valid(receipt)
            or receipt.control_kind not in FEISHU_CONTROL_KINDS):
        return None
    home = Path(profile_home).resolve()
    if receipt.home != str(home):
        return None
    occurrence = receipt.event_id or receipt.occurrence
    source_id = _digest(json.dumps([str(home), receipt.namespace, occurrence, receipt.revision,
                                    "maintenance"], separators=(",", ":"), ensure_ascii=True))
    with closing(sqlite3.connect((home / "state.db").as_uri() + "?mode=rw", uri=True, timeout=5)) as db:
        db.row_factory = sqlite3.Row
        db.execute("BEGIN IMMEDIATE")
        session = db.execute("SELECT session_key,source,user_id,chat_id,chat_type,thread_id "
                             "FROM sessions WHERE id=?", (session_id,)).fetchone()
        if (session is None or session["source"] != platform
                or session["session_key"] != receipt.session_key
                or session["chat_id"] != receipt.chat_id
                or session["chat_type"] != receipt.chat_type
                or (session["thread_id"] or "") != (receipt.thread_id or "")
                or receipt.chat_type == "dm" and session["user_id"] != receipt.actor
                or not _platform_namespace_matches(receipt)):
            db.rollback()
            return None
        has_sources = db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='maintenance_sources_v1'").fetchone()
        existing = (db.execute("SELECT 1 FROM maintenance_sources_v1 WHERE source_id=?",
                               (source_id,)).fetchone() if has_sources else None)
        if existing is not None:
            if not _feishu_proof_matches(db, source_id, receipt, session_id, 0, receipt.display,
                                         require_row=False):
                db.rollback()
                return None
            db.commit()
            return source_id
        db.execute("""CREATE TABLE IF NOT EXISTS maintenance_sources_v1 (
            source_id TEXT PRIMARY KEY, authority TEXT NOT NULL, origin TEXT NOT NULL, home TEXT NOT NULL,
            namespace TEXT NOT NULL, actor TEXT NOT NULL, audience TEXT NOT NULL, occurrence TEXT NOT NULL,
            event_id TEXT, revision TEXT NOT NULL, session_id TEXT NOT NULL, session_key TEXT NOT NULL,
            message_row_id INTEGER NOT NULL, source_time REAL NOT NULL, received_time REAL NOT NULL,
            timezone TEXT NOT NULL, time_fallback INTEGER NOT NULL, raw_ref TEXT, raw_text TEXT NOT NULL,
            raw_digest TEXT NOT NULL, extraction_text TEXT NOT NULL, record_digest TEXT NOT NULL,
            revoked INTEGER NOT NULL DEFAULT 0)""")
        db.execute("""CREATE TABLE IF NOT EXISTS maintenance_outbox_v1 (
            source_id TEXT PRIMARY KEY REFERENCES maintenance_sources_v1(source_id),
            state TEXT NOT NULL DEFAULT 'PENDING' CHECK(state IN ('PENDING','ACKED')),
            task_id TEXT, acknowledged_at REAL)""")
        _ensure_outbox_columns(db)
        db.execute("""INSERT INTO maintenance_sources_v1 VALUES
            (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0)""",
            (source_id, receipt.authority, receipt.origin, str(home), receipt.namespace, receipt.actor,
             receipt.audience, receipt.occurrence, receipt.event_id, receipt.revision, session_id,
             session["session_key"], 0, receipt.source_time, receipt.received_time, receipt.timezone,
             int(receipt.time_fallback), receipt.raw_ref,
             sqlite3.Binary(receipt.raw.encode("utf-8", "surrogatepass")),
             _digest(receipt.raw), receipt.extraction, ""))
        signed = db.execute("SELECT * FROM maintenance_sources_v1 WHERE source_id=?", (source_id,)).fetchone()
        db.execute("UPDATE maintenance_sources_v1 SET record_digest=? WHERE source_id=?",
                   (_record_digest(signed), source_id))
        disposition = "CONTROL" if receipt.control_kind == "typed_clarify_response" else "UNDECIDED"
        db.execute("INSERT INTO maintenance_outbox_v1(source_id,disposition,enrollment_enabled) "
                   "VALUES (?,?,?)", (source_id, disposition,
                    int(receipt.control_kind != "typed_clarify_response" and _intake_enabled_at_commit(home))))
        _insert_feishu_text_proof(db, source_id, receipt, session_id, 0, receipt.display)
        db.commit()
    return source_id


def commit_feishu_control_source(profile_home, session_id, receipt):
    if getattr(receipt, "authority", None) != "feishu-human":
        return None
    return commit_platform_control_source(profile_home, session_id, receipt)


def read_committed_source(profile_home, source_id):
    if not isinstance(source_id, str) or len(source_id) != 64:
        return None
    home = Path(profile_home).resolve()
    try:
        with closing(sqlite3.connect((home / "state.db").as_uri() + "?mode=ro", uri=True, timeout=2)) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA query_only=ON")
            db.execute("BEGIN")
            source = db.execute("SELECT * FROM maintenance_sources_v1 WHERE source_id=? AND revoked=0",
                                (source_id,)).fetchone()
            if source is None or source["home"] != str(home):
                return None
            raw = source["raw_text"]
            raw = raw.decode("utf-8", "surrogatepass") if isinstance(raw, bytes) else raw
            if (not isinstance(raw, str) or _digest(raw) != source["raw_digest"]
                    or _record_digest(source) != source["record_digest"]):
                return None
            session = db.execute(
                "SELECT session_key,source,user_id,chat_id,chat_type,thread_id FROM sessions WHERE id=?",
                (source["session_id"],)).fetchone()
            platform = {"cli-local": "cli", "feishu-human": "feishu",
                        "qqbot-human": "qqbot", "weixin-human": "weixin"}.get(source["authority"])
            has_batches = db.execute(
                "SELECT 1 FROM sqlite_master WHERE name='maintenance_source_batches_v1'").fetchone()
            batch = (db.execute("SELECT anchor_event_id,display_digest,position,size "
                                "FROM maintenance_source_batches_v1 WHERE source_id=?",
                                (source_id,)).fetchone() if has_batches else None)
            has_events = db.execute(
                "SELECT 1 FROM sqlite_master WHERE name='maintenance_source_events_v1'").fetchone()
            proof = (db.execute("SELECT * FROM maintenance_source_events_v1 WHERE source_id=?",
                                (source_id,)).fetchone() if has_events else None)
            has_local_controls = db.execute(
                "SELECT 1 FROM sqlite_master WHERE name='maintenance_local_controls_v1'").fetchone()
            local_control = (db.execute(
                "SELECT control_kind,control_binding_json,control_digest "
                "FROM maintenance_local_controls_v1 WHERE source_id=?",
                (source_id,)).fetchone() if has_local_controls else None)
            if local_control is not None:
                local_control = dict(local_control)
                kind = local_control["control_kind"]
                if (proof is not None or source["authority"] != "cli-local" or platform != "cli"
                        or source["origin"] != "human" or source["namespace"] != "cli/local"
                        or source["actor"] != "local-user" or source["audience"] != "private"
                        or source["event_id"] is not None or source["raw_ref"] is not None
                        or source["message_row_id"] != 0 or session is None
                        or session["source"] != "cli"
                        or str(session["session_key"] or "") != str(source["session_key"] or "")
                        or not _local_control_row_matches(source_id, kind,
                                                          local_control["control_binding_json"],
                                                          local_control["control_digest"])):
                    return None
                return {"schema": CLI_CONTROL_CONTRACT, **dict(source), "raw_text": raw,
                        "control_kind": kind,
                        "control_binding_json": local_control["control_binding_json"]}
            if proof is not None:
                proof = dict(proof)
                receiver = _event_receiver_from_namespace(source["authority"], source["namespace"])
                receipt = SourceReceipt(
                    source["authority"], source["origin"], source["namespace"], source["actor"],
                    source["audience"], raw, source["extraction_text"], source["occurrence"],
                    source["event_id"], source["revision"], source["source_time"],
                    source["received_time"], source["timezone"], bool(source["time_fallback"]),
                    source["raw_ref"], source["home"], source["session_key"], None, _SEAL, None,
                    proof["chat_id"], proof["chat_type"], display=proof["display_text"],
                    contract=proof["contract"], input_kind=proof["input_kind"],
                    message_type=proof["event_type"], thread_id=proof["thread_id"],
                    reply_to_message_id=proof["reply_to_message_id"],
                    reply_to_author_id=proof["reply_to_author_id"],
                    reply_to_text=proof["reply_to_text"], event_structure=proof["event_structure"],
                    control_kind=proof["control_kind"],
                    control_binding_json=proof["control_binding_json"], receiver=receiver)
                if (not _platform_event_fields_valid(receipt)
                        or _feishu_proof_digest(proof) != proof["proof_digest"]
                        or proof["contract"] not in (FEISHU_TEXT_CONTRACT, FEISHU_CONTROL_CONTRACT)
                        or proof["session_id"] != source["session_id"]
                        or proof["message_row_id"] != source["message_row_id"]
                        or proof["event_id"] != source["event_id"]
                        or proof["actor"] != source["actor"]
                        or proof["raw_digest"] != source["raw_digest"]
                        or proof["extraction_digest"] != _digest(source["extraction_text"])
                        or source["origin"] != "human" or platform is None
                        or source["audience"] != ("private" if proof["chat_type"] == "dm" else "group")):
                    return None
                if proof["contract"] == FEISHU_CONTROL_CONTRACT:
                    if (source["message_row_id"] != 0 or session is None
                            or session["source"] != platform
                            or session["session_key"] != source["session_key"]
                            or session["chat_id"] != proof["chat_id"]
                            or session["chat_type"] != proof["chat_type"]
                            or (session["thread_id"] or "") != (proof["thread_id"] or "")
                            or proof["chat_type"] == "dm" and session["user_id"] != source["actor"]
                            or proof["input_kind"] not in FEISHU_CONTROL_KINDS
                            or proof["control_kind"] != proof["input_kind"]
                            or not _control_binding_valid(proof["control_binding_json"])):
                        return None
                    return {"schema": FEISHU_CONTROL_CONTRACT, **dict(source), "raw_text": raw,
                            **{key: proof[key] for key in _FEISHU_PROOF_COLUMNS
                               if key not in source.keys()},
                            "message_type": proof["event_type"],
                            "control_kind": proof["control_kind"],
                            "control_binding_json": proof["control_binding_json"]}
                row_event_id = batch["anchor_event_id"] if batch is not None else proof["event_id"]
                row = db.execute("SELECT role,content,display_kind,_compressed_summary,platform_message_id "
                                 "FROM messages WHERE id=? AND session_id=?",
                                 (source["message_row_id"], source["session_id"])).fetchone()
                if (session is None or row is None
                        or not _platform_event_binding_matches(receipt, session, row,
                                                               row_event_id, proof["display_text"])
                        or row["role"] != "user"
                        or _digest(row["content"]) != proof["display_digest"]
                        or row["display_kind"] not in (None, "") or row["_compressed_summary"]
                        or batch is not None and (batch["display_digest"] != proof["display_digest"]
                                                  or batch["position"] < 0 or batch["position"] >= batch["size"])):
                    return None
                return {"schema": FEISHU_TEXT_CONTRACT, **dict(source), "raw_text": raw,
                        **{key: proof[key] for key in _FEISHU_PROOF_COLUMNS
                           if key not in source.keys()},
                        "message_type": proof["event_type"],
                        "batch_anchor_event_id": row_event_id if batch is not None else None,
                        "batch_position": batch["position"] if batch is not None else 0,
                        "batch_size": batch["size"] if batch is not None else 1}
            row_event_id = batch[0] if batch else source["event_id"]
            if (session is None or session["session_key"] != source["session_key"]
                    or source["origin"] != "human" or platform is None
                    or session["source"] != platform
                    or source["audience"] not in ("private", "group")
                    or source["audience"] == "group" and platform not in ("qqbot", "weixin")
                    or source["authority"] == "feishu-human" and (
                        session["user_id"] != source["actor"] or session["chat_type"] != "dm"
                        or not source["event_id"]
                        or not session["chat_id"] or not source["namespace"].startswith("feishu/")
                        or not source["namespace"].endswith("/" + session["session_key"])
                        or source["raw_ref"] != "feishu-message:" + source["event_id"])
                    or source["authority"] == "cli-local" and (
                        source["namespace"] != "cli/local" or source["actor"] != "local-user"
                        or source["message_row_id"] <= 0)
                    or platform in ("qqbot", "weixin") and (
                        not source["event_id"] or not session["chat_id"]
                        or session["chat_type"] not in ("dm", "group")
                        or source["audience"] != ("group" if session["chat_type"] == "group" else "private")
                        or session["chat_type"] == "dm" and session["user_id"] != source["actor"]
                        or not source["namespace"].startswith(source["authority"] + "/")
                        or not source["namespace"].endswith("/" + session["chat_type"] + "/" + session["chat_id"])
                        or source["raw_ref"] != source["authority"] + ":" + source["event_id"])):
                return None
            row = db.execute("SELECT role,content,display_kind,_compressed_summary,platform_message_id "
                             "FROM messages WHERE id=? AND session_id=?",
                             (source["message_row_id"], source["session_id"])).fetchone()
            if row is not None and (row["role"] != "user" or
                    (batch is not None and (_digest(row["content"]) != batch["display_digest"]
                     or batch["position"] < 0 or batch["position"] >= batch["size"])
                     or batch is None and row["content"] != source["extraction_text"])
                    or row["display_kind"] not in (None, "") or row["_compressed_summary"]
                    or row["platform_message_id"] != row_event_id):
                return None
            if row is not None and source["authority"] == "feishu-human":
                proof = db.execute("SELECT source,chat_id,user_id,platform_message_id,revoked "
                                   "FROM authenticated_inbound_v1 WHERE message_row_id=? AND session_id=?",
                                   (source["message_row_id"], source["session_id"])).fetchone()
                if (proof is None or proof["revoked"] or
                        (proof["source"], proof["chat_id"], proof["user_id"], proof["platform_message_id"]) !=
                        ("feishu", session["chat_id"], source["actor"], source["event_id"])):
                    return None
            return {"schema": CONTRACT, **dict(source), "raw_text": raw}
    except (sqlite3.Error, OSError, ValueError):
        return None

def current_committed_source(profile_home, session_id):
    receipt = current_admitted_source()
    return committed_source_for_receipt(profile_home, session_id, receipt)


def committed_source_for_receipt(profile_home, session_id, receipt):
    if type(receipt) is not SourceReceipt or receipt.seal is not _SEAL:
        return None
    home = str(Path(profile_home).resolve())
    occurrence = receipt.event_id or receipt.occurrence
    source_id = _digest(json.dumps([home, receipt.namespace, occurrence, receipt.revision, "maintenance"],
                                   separators=(",", ":"), ensure_ascii=True))
    source = read_committed_source(home, source_id)
    if source and source["session_id"] == session_id:
        return source
    return None


def source_for_session_row(profile_home, session_id, row_id):
    """Resolve the immutable source projected onto one session transcript row."""
    if not isinstance(session_id, str) or type(row_id) is not int:
        return None
    home = Path(profile_home).resolve()
    try:
        with closing(sqlite3.connect((home / "state.db").as_uri() + "?mode=ro", uri=True, timeout=2)) as db:
            row = db.execute(
                "SELECT source_id FROM maintenance_sources_v1 "
                "WHERE session_id=? AND message_row_id=?", (session_id, row_id)).fetchone()
            if row is None:
                row = db.execute(
                    "SELECT source_id FROM maintenance_source_projections_v1 "
                    "WHERE session_id=? AND message_row_id=? ORDER BY source_id LIMIT 1",
                    (session_id, row_id)).fetchone()
            return row[0] if row else None
    except sqlite3.Error:
        return None


def source_for_latest_user_message(profile_home, session_id):
    """Source for the latest retryable user projection, skipping bookkeeping user rows."""
    if not isinstance(session_id, str) or not session_id:
        return None
    home = Path(profile_home).resolve()
    try:
        with closing(sqlite3.connect((home / "state.db").as_uri() + "?mode=ro", uri=True, timeout=2)) as db:
            row = db.execute(
                "SELECT id FROM messages WHERE session_id=? AND role='user' AND active=1 "
                "AND content IS NOT NULL AND TRIM(content) != '' "
                "AND (display_kind IS NULL OR display_kind='') "
                "AND (_compressed_summary IS NULL OR _compressed_summary=0) "
                "ORDER BY id DESC LIMIT 1", (session_id,)).fetchone()
    except sqlite3.Error:
        return None
    return source_for_session_row(home, session_id, row[0]) if row else None


def rehydrate_source_receipt(profile_home, source_id):
    """Rebuild a sealed receipt from committed evidence for an idempotent replay."""
    source = read_committed_source(profile_home, source_id)
    if source is None:
        return None
    receiver = _event_receiver_from_namespace(source["authority"], source["namespace"])
    if source["schema"] == CONTRACT and source["authority"] == "cli-local":
        if source["event_id"] is not None or source["message_row_id"] <= 0:
            return None
        return SourceReceipt(
            authority=source["authority"], origin=source["origin"], namespace=source["namespace"],
            actor=source["actor"], audience=source["audience"], raw=source["raw_text"],
            extraction=source["extraction_text"], occurrence=source["occurrence"],
            event_id=None, revision=source["revision"], source_time=source["source_time"],
            received_time=source["received_time"], timezone=source["timezone"],
            time_fallback=bool(source["time_fallback"]), raw_ref=None, home=source["home"],
            session_key=source["session_key"], intended_session_id=source["session_id"],
            seal=_SEAL, display=source["extraction_text"], contract=CONTRACT)
    if source["schema"] == CLI_CONTROL_CONTRACT:
        if (source["authority"] != "cli-local" or source["event_id"] is not None
                or source["message_row_id"] != 0):
            return None
        receipt = SourceReceipt(
            authority=source["authority"], origin=source["origin"], namespace=source["namespace"],
            actor=source["actor"], audience=source["audience"], raw=source["raw_text"],
            extraction=source["extraction_text"], occurrence=source["occurrence"],
            event_id=None, revision=source["revision"], source_time=source["source_time"],
            received_time=source["received_time"], timezone=source["timezone"],
            time_fallback=bool(source["time_fallback"]), raw_ref=None, home=source["home"],
            session_key=source["session_key"], intended_session_id=source["session_id"],
            seal=_SEAL, display=source["extraction_text"], contract=CLI_CONTROL_CONTRACT,
            input_kind=source.get("control_kind"), control_kind=source.get("control_kind"),
            control_binding_json=source.get("control_binding_json"))
        return receipt if _local_control_fields_valid(receipt) else None
    if source["schema"] == CONTRACT and source["authority"] in PLATFORM_EVIDENCE_AUTHORITIES:
        return SourceReceipt(
            source["authority"], source["origin"], source["namespace"], source["actor"],
            source["audience"], source["raw_text"], source["extraction_text"],
            source["occurrence"], source["event_id"], source["revision"],
            source["source_time"], source["received_time"], source["timezone"],
            bool(source["time_fallback"]), source["raw_ref"], source["home"],
            source["session_key"], None, _SEAL, None, source.get("chat_id"), source.get("chat_type"),
            display=source["extraction_text"], contract=CONTRACT, receiver=receiver)
    if source["schema"] in (FEISHU_TEXT_CONTRACT, FEISHU_CONTROL_CONTRACT):
        contract = source["schema"]
        message_type = source.get("event_type") or source.get("message_type") or "text"
        display = source.get("display_text")
        event_structure = source.get("event_structure")
        if not display or not event_structure:
            return None
        receipt = SourceReceipt(
            source["authority"], source["origin"], source["namespace"], source["actor"],
            source["audience"], source["raw_text"], source["extraction_text"],
            source["occurrence"], source["event_id"], source["revision"],
            source["source_time"], source["received_time"], source["timezone"],
            bool(source["time_fallback"]), source["raw_ref"], source["home"],
            source["session_key"], None, _SEAL, None, source["chat_id"], source["chat_type"],
            display=display, contract=contract, input_kind=source.get("input_kind"),
            message_type=message_type, thread_id=source.get("thread_id"),
            reply_to_message_id=source.get("reply_to_message_id"),
            reply_to_author_id=source.get("reply_to_author_id"),
            reply_to_text=source.get("reply_to_text"), event_structure=event_structure,
            control_kind=source.get("control_kind"),
            control_binding_json=source.get("control_binding_json"), receiver=receiver)
        return receipt if _platform_event_fields_valid(receipt) else None
    return None


def derive_platform_control_source(receipt, *, kind, text, binding=None):
    """Derive one new control occurrence from an already sealed event capability."""
    if type(receipt) is SourceBundle and receipt.seal is _SEAL and len(receipt.receipts) == 1:
        receipt = receipt.receipts[0]
    if (type(receipt) is not SourceReceipt or receipt.seal is not _SEAL
            or receipt.authority not in PLATFORM_EVIDENCE_AUTHORITIES
            or kind not in FEISHU_CONTROL_KINDS or not isinstance(text, str) or not text):
        return None
    if receipt.contract not in (CONTRACT, FEISHU_TEXT_CONTRACT, FEISHU_CONTROL_CONTRACT):
        return None
    encoded = control_binding_json(binding)
    derived = replace(receipt, raw=text, extraction=text, display=text,
                      contract=FEISHU_CONTROL_CONTRACT, input_kind=kind, control_kind=kind,
                      control_binding_json=encoded)
    return derived if _platform_event_fields_valid(derived) else None


def derive_feishu_control_source(receipt, *, kind, text, binding=None):
    if getattr(receipt, "authority", None) != "feishu-human":
        return None
    return derive_platform_control_source(receipt, kind=kind, text=text, binding=binding)


def admit_reused_platform_source(receipt, event, adapter, home, session_key):
    """Validate a rehydrated original source against a retry without minting a new one."""
    source = getattr(event, "source", None)
    platform = getattr(getattr(source, "platform", None), "value", None)
    expected_platform = PLATFORM_EVIDENCE_AUTHORITIES.get(getattr(receipt, "authority", None))
    if (type(receipt) is not SourceReceipt or receipt.seal is not _SEAL
            or expected_platform is None
            or receipt.contract not in (CONTRACT, FEISHU_TEXT_CONTRACT, FEISHU_CONTROL_CONTRACT)
            or getattr(event, "internal", False)
            or getattr(source, "is_bot", None) is not False
            or platform != expected_platform or receipt.transport not in (None, adapter)
            or receipt.home != str(Path(home).resolve())
            or receipt.session_key != session_key
            or receipt.actor != getattr(source, "user_id", None)):
        return None
    if receipt.contract in (FEISHU_TEXT_CONTRACT, FEISHU_CONTROL_CONTRACT):
        if (not _platform_event_fields_valid(receipt)
                or receipt.chat_id != getattr(source, "chat_id", None)
                or receipt.chat_type != getattr(source, "chat_type", None)
                or (receipt.thread_id or "") != (getattr(source, "thread_id", None) or "")):
            return None
    elif receipt.authority == "feishu-human":
        if (getattr(source, "chat_type", None) != "dm" or receipt.audience != "private"
                or not receipt.event_id
                or receipt.raw_ref != "feishu-message:" + receipt.event_id):
            return None
    else:
        if (receipt.chat_id != getattr(source, "chat_id", None)
                or receipt.chat_type != getattr(source, "chat_type", None)
                or not _platform_namespace_matches(receipt)
                or not receipt.event_id
                or receipt.raw_ref != _platform_raw_ref(receipt.authority, receipt.event_id)):
            return None
    return receipt


def admit_reused_feishu_source(receipt, event, adapter, home, session_key):
    if getattr(receipt, "authority", None) != "feishu-human":
        return None
    return admit_reused_platform_source(receipt, event, adapter, home, session_key)


def pending_sources(profile_home, limit=100):
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError("Invalid outbox limit")
    home = Path(profile_home).resolve()
    try:
        with closing(sqlite3.connect((home / "state.db").as_uri() + "?mode=ro", uri=True)) as db:
            ids = [r[0] for r in db.execute("SELECT source_id FROM maintenance_outbox_v1 WHERE state='PENDING' LIMIT ?", (limit,))]
    except sqlite3.OperationalError:
        return []
    return ids


def source_disposition(profile_home, session_id, receipt):
    """Inspect the exact sealed occurrence, including revoked evidence, for ownership fencing."""
    home = Path(profile_home).resolve()
    if type(receipt) is not SourceReceipt or receipt.seal is not _SEAL or receipt.home != str(home):
        return None
    source_id = _digest(json.dumps([str(home), receipt.namespace, receipt.event_id or receipt.occurrence,
                                    receipt.revision, "maintenance"], separators=(",", ":"), ensure_ascii=True))
    with closing(sqlite3.connect((home / "state.db").as_uri() + "?mode=rw", uri=True, timeout=5)) as db:
        db.execute("BEGIN IMMEDIATE")
        _ensure_outbox_columns(db)
        _ensure_source_projections(db)
        row = db.execute("SELECT s.session_id,s.raw_digest,o.state,o.disposition,o.capture_state,"
                         "o.enrollment_enabled FROM maintenance_sources_v1 s "
                         "JOIN maintenance_outbox_v1 o USING(source_id) WHERE s.source_id=? "
                         "AND (s.message_row_id=0 OR EXISTS "
                         "(SELECT 1 FROM maintenance_source_projections_v1 p "
                         "WHERE p.source_id=s.source_id AND p.session_id=?))",
                         (source_id, session_id)).fetchone()
        db.commit()
        if row is None or row[1] != _digest(receipt.raw):
            return None
        return source_id, row[3], row[4], bool(row[5])


def claim_source(profile_home, source_id, disposition, *, existing_task_recovery=False):
    """One durable ownership choice before any legacy ADD or journal handoff."""
    if disposition not in ("LEGACY", "GOVERNED"):
        raise ValueError("Invalid source disposition")
    home = Path(profile_home).resolve()
    with closing(sqlite3.connect((home / "state.db").as_uri() + "?mode=rw", uri=True, timeout=5)) as db:
        db.execute("BEGIN IMMEDIATE")
        _ensure_outbox_columns(db)
        row = db.execute("SELECT disposition,enrollment_enabled FROM maintenance_outbox_v1 WHERE source_id=?",
                         (source_id,)).fetchone()
        if row is None:
            raise ValueError("SOURCE_NOT_ENROLLED")
        current = row[0]
        if current == "UNDECIDED":
            if disposition == "GOVERNED" and row[1] != 1 and not existing_task_recovery:
                raise ValueError("SOURCE_NOT_ENROLLED")
            if read_committed_source(home, source_id) is None:
                raise ValueError("COMMITTED_SOURCE_UNAVAILABLE")
            db.execute("UPDATE maintenance_outbox_v1 SET disposition=?,capture_state=? WHERE source_id=?",
                       (disposition, "CLAIMED" if disposition == "LEGACY" else None, source_id))
            current = disposition
        db.commit()
        return current


def begin_legacy_capture(profile_home, source_id):
    home = Path(profile_home).resolve()
    with closing(sqlite3.connect((home / "state.db").as_uri() + "?mode=rw", uri=True, timeout=5)) as db:
        db.execute("BEGIN IMMEDIATE")
        _ensure_outbox_columns(db)
        if read_committed_source(home, source_id) is None:
            db.commit()
            return False
        changed = db.execute("UPDATE maintenance_outbox_v1 SET capture_state='DISPATCHED' "
                             "WHERE source_id=? AND disposition='LEGACY' AND capture_state='CLAIMED'",
                             (source_id,)).rowcount
        db.commit()
        return changed == 1


def finish_legacy_capture(profile_home, source_id, state):
    if state not in ("DONE", "DEFERRED", "UNCERTAIN"):
        raise ValueError("Invalid capture result")
    home = Path(profile_home).resolve()
    with closing(sqlite3.connect((home / "state.db").as_uri() + "?mode=rw", uri=True, timeout=5)) as db:
        db.execute("BEGIN IMMEDIATE")
        changed = db.execute("UPDATE maintenance_outbox_v1 SET capture_state=? "
                             "WHERE source_id=? AND disposition='LEGACY' AND capture_state='DISPATCHED'",
                             (state, source_id)).rowcount
        if changed != 1:
            raise ValueError("Legacy capture state conflict")
        db.commit()


def delivery_sources(profile_home, limit=100):
    """Persisted least-recently-attempted order gives every governed row a turn."""
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError("Invalid outbox limit")
    home = Path(profile_home).resolve()
    with closing(sqlite3.connect((home / "state.db").as_uri() + "?mode=rw", uri=True, timeout=5)) as db:
        db.execute("BEGIN IMMEDIATE")
        if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                      "AND name='maintenance_outbox_v1'").fetchone() is None:
            return []  # Pre-intake databases have no delivery queue yet.
        _ensure_outbox_columns(db)
        has_events = db.execute(
            "SELECT 1 FROM sqlite_master WHERE name='maintenance_source_events_v1'").fetchone()
        event_join = (" LEFT JOIN maintenance_source_events_v1 e ON e.source_id=o.source_id"
                      if has_events else "")
        event_clause = (" OR (e.source_id IS NOT NULL AND o.enrollment_enabled=1)"
                        if has_events else "")
        ids = [r[0] for r in db.execute(
            "SELECT o.source_id FROM maintenance_outbox_v1 o "
            "JOIN maintenance_sources_v1 s ON s.source_id=o.source_id" + event_join +
            " WHERE o.state='PENDING' AND (o.disposition='GOVERNED' "
            "OR (o.disposition='UNDECIDED' AND (o.enrollment_enabled=0 "
            "OR s.authority IN ('cli-local','qqbot-human','weixin-human')" + event_clause + "))) "
            "ORDER BY o.attempt_order, o.rowid LIMIT ?", (limit,))]
        now = time.time()
        rank = db.execute("SELECT COALESCE(MAX(attempt_order),0) FROM maintenance_outbox_v1").fetchone()[0]
        db.executemany("UPDATE maintenance_outbox_v1 SET last_attempt_at=?,attempt_order=? WHERE source_id=?",
                       [(now, rank + index + 1, source_id) for index, source_id in enumerate(ids)])
        db.commit()
        return ids


def control_sources(profile_home, limit=100):
    """Pending control responses that must be bound, never admitted as new facts."""
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError("Invalid control limit")
    home = Path(profile_home).resolve()
    try:
        with closing(sqlite3.connect((home / "state.db").as_uri() + "?mode=ro", uri=True)) as db:
            rows = []
            if db.execute("SELECT 1 FROM sqlite_master WHERE name='maintenance_source_events_v1'").fetchone():
                rows.extend(db.execute(
                    "SELECT o.source_id,o.rowid FROM maintenance_outbox_v1 o "
                    "JOIN maintenance_source_events_v1 e USING(source_id) "
                    "WHERE o.state='PENDING' AND o.disposition='CONTROL' "
                    "AND e.control_kind='typed_clarify_response'").fetchall())
            if db.execute("SELECT 1 FROM sqlite_master WHERE name='maintenance_local_controls_v1'").fetchone():
                rows.extend(db.execute(
                    "SELECT o.source_id,o.rowid FROM maintenance_outbox_v1 o "
                    "JOIN maintenance_local_controls_v1 c USING(source_id) "
                    "WHERE o.state='PENDING' AND o.disposition='CONTROL' "
                    "AND c.control_kind='typed_clarify_response'").fetchall())
            return [row[0] for row in sorted(rows, key=lambda row: row[1])[:limit]]
    except sqlite3.OperationalError:
        return []


def acknowledge_control_source(profile_home, source_id, result_ref):
    """Close a bound control response without creating a maintenance task."""
    if not isinstance(result_ref, str) or not result_ref:
        raise ValueError("Control ACK conflict")
    home = Path(profile_home).resolve()
    with closing(sqlite3.connect((home / "state.db").as_uri() + "?mode=rw", uri=True, timeout=5)) as db:
        db.execute("BEGIN IMMEDIATE")
        changed = db.execute("UPDATE maintenance_outbox_v1 SET state='ACKED',task_id=?,acknowledged_at=? "
                             "WHERE source_id=? AND disposition='CONTROL' "
                             "AND (state='PENDING' OR state='ACKED' AND task_id=?)",
                             (result_ref, time.time(), source_id, result_ref)).rowcount
        if changed != 1:
            db.rollback()
            raise ValueError("Control ACK conflict")
        db.commit()


def record_delivery_failure(profile_home, source_id, reason):
    if reason not in ("COMMITTED_SOURCE_UNAVAILABLE", "SOURCE_HOME_MISMATCH",
                      "SOURCE_OWNER_UNAVAILABLE", "SOURCE_LEGACY_OWNED",
                      "MAINTENANCE_INTAKE_DISABLED", "SOURCE_NOT_ENROLLED"):
        raise ValueError("Invalid delivery rejection")
    home = Path(profile_home).resolve()
    with closing(sqlite3.connect((home / "state.db").as_uri() + "?mode=rw", uri=True, timeout=5)) as db:
        db.execute("BEGIN IMMEDIATE")
        _ensure_outbox_columns(db)
        db.execute("UPDATE maintenance_outbox_v1 SET last_error=? "
                   "WHERE source_id=? AND state='PENDING'", (reason, source_id))
        db.commit()


def delivery_status(profile_home, source_id):
    home = Path(profile_home).resolve()
    try:
        with closing(sqlite3.connect((home / "state.db").as_uri() + "?mode=ro", uri=True)) as db:
            db.row_factory = sqlite3.Row
            row = db.execute("SELECT state,disposition,capture_state,last_error,last_attempt_at,"
                             "attempt_order,enrollment_enabled "
                             "FROM maintenance_outbox_v1 WHERE source_id=?", (source_id,)).fetchone()
            return dict(row) if row else None
    except sqlite3.OperationalError:
        return None


def acknowledge_source(profile_home, source_id, task_id):
    if read_committed_source(profile_home, source_id) is None:
        raise ValueError("Committed source unavailable")
    path = Path(profile_home).resolve() / "state.db"
    with closing(sqlite3.connect(path.as_uri() + "?mode=rw", uri=True, timeout=5)) as db:
        db.execute("BEGIN IMMEDIATE")
        source = db.execute("SELECT revoked FROM maintenance_sources_v1 WHERE source_id=?", (source_id,)).fetchone()
        _ensure_outbox_columns(db)
        row = db.execute("SELECT state,task_id,disposition FROM maintenance_outbox_v1 WHERE source_id=?", (source_id,)).fetchone()
        if (source is None or source[0] or row is None or row[2] != "GOVERNED"
                or row[0] == "ACKED" and row[1] != task_id):
            raise ValueError("Outbox ACK conflict")
        db.execute("UPDATE maintenance_outbox_v1 SET state='ACKED',task_id=?,acknowledged_at=?,last_error=NULL WHERE source_id=?",
                   (task_id, time.time(), source_id))
        db.commit()


def revoke_committed_source(profile_home, source_id):
    """Explicit source revocation; transcript retirement never calls this."""
    if read_committed_source(profile_home, source_id) is None:
        raise ValueError("Committed source unavailable")
    path = Path(profile_home).resolve() / "state.db"
    with closing(sqlite3.connect(path.as_uri() + "?mode=rw", uri=True, timeout=5)) as db:
        db.execute("BEGIN IMMEDIATE")
        db.execute("UPDATE maintenance_sources_v1 SET revoked=1 WHERE source_id=?", (source_id,))
        db.commit()
