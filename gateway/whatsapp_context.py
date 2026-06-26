"""WhatsApp per-chat event log, context bundle, and deterministic task state.

This module intentionally sits below prompting.  Mention-only WhatsApp groups still
need to ingest every message, media filename, and receipt confirmation so a later
mention can build grounded context before the LLM is called.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


_PAYMENT_WORD_RE = re.compile(
    r"\b(payment|paid|receipt|proof|transfer|transaction|comprobante|pago|pagos|pagado|transferencia|recibo|spei|cep)\b",
    re.IGNORECASE,
)
_CONFIRM_RECEIPT_RE = re.compile(
    r"\b(confirmo\s+de\s+recibido|confirmado|recibido|recibida|received|confirmed)\b",
    re.IGNORECASE,
)
_AMOUNT_RE = re.compile(
    r"(?:\$\s*)?(\d{1,3}(?:[,.]\d{3})*(?:[,.]\d{1,2})?|\d+(?:[,.]\d{1,2})?)\s*(k|mil|mxn|usd|pesos|dolares|dólares)?",
    re.IGNORECASE,
)
_CASE_RE = re.compile(
    r"\b(?:Mercantil|Civil|Familiar|Penal|Exp(?:ediente)?\.?|Caso)\s*[#:]?\s*\d{1,6}\s*/\s*\d{2,4}\b",
    re.IGNORECASE,
)


@dataclass
class WhatsAppChatEvent:
    chat_id: str
    message_id: str
    sender_id: str = ""
    sender_name: str = ""
    timestamp: str = ""
    text_body: str = ""
    message_type: str = "text"
    quoted_message_id: str | None = None
    mentioned_users: list[str] = field(default_factory=list)
    media_filename: str | None = None
    media_mime_type: str | None = None
    local_media_path: str | None = None
    extracted_text: str | None = None
    download_extraction_status: str = "none"
    raw: dict[str, Any] = field(default_factory=dict)
    should_reply: bool = False


@dataclass
class WhatsAppPaymentState:
    payment_proof_sent: bool = False
    receipt_confirmed_by_counterparty: bool = False
    amount: str | None = None
    recipient_payee: str | None = None
    sender: str | None = None
    date_time: str | None = None
    filename: str | None = None
    linked_case_project: str | None = None
    proof_message_id: str | None = None
    confirmation_message_id: str | None = None

    def as_prompt_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in (None, "", False)} | {
            "payment_proof_sent": self.payment_proof_sent,
            "receipt_confirmed_by_counterparty": self.receipt_confirmed_by_counterparty,
        }


@dataclass
class WhatsAppContextBundle:
    chat_id: str
    trigger_message_id: str
    current_message: WhatsAppChatEvent
    recent_events: list[WhatsAppChatEvent]
    quoted_chain: list[WhatsAppChatEvent]
    recent_media: list[WhatsAppChatEvent]
    task_state: dict[str, Any]
    guardrails: dict[str, Any]
    context_message_ids: list[str]
    media_paths: list[str]
    unreadable_media: list[WhatsAppChatEvent]
    relevant_entities: list[str]

    def render_for_prompt(self) -> str:
        lines: list[str] = [
            "[WhatsApp Context Bundle — same-chat evidence before answering]",
            f"chat_id: {self.chat_id}",
            f"trigger_message_id: {self.trigger_message_id}",
            f"context_message_ids: {', '.join(self.context_message_ids)}",
        ]
        if self.relevant_entities:
            lines.append("relevant_entities: " + ", ".join(self.relevant_entities))
        if self.task_state:
            lines.append("task_state: " + json.dumps(self.task_state, ensure_ascii=False, sort_keys=True))
        if self.guardrails.get("block_payment_recommendation"):
            lines.extend(
                [
                    "GUARDRAIL: Payment proof and/or receipt confirmation already exists in this chat.",
                    "Do not recommend paying, authorizing payment, or sending the comprobante again.",
                    "Move forward: request date/time of diligence; confirm application of payment to the correct case; request official receipt; ask what action remains pending.",
                ]
            )
        if self.unreadable_media:
            lines.append(
                "Veo que se envió un archivo, pero no pude descargarlo o leerlo. Reenvíalo o respóndeme directamente al archivo."
            )
            lines.append("Do not pretend the unreadable media was read.")
        lines.append("recent_messages:")
        for ev in self.recent_events:
            bits = [ev.message_id, ev.sender_name or ev.sender_id or "unknown", ev.message_type]
            if ev.media_filename:
                bits.append(f"file={ev.media_filename}")
            if ev.local_media_path:
                bits.append(f"path={ev.local_media_path}")
            lines.append("- [" + " | ".join(bits) + f"] {ev.text_body}".rstrip())
            if ev.extracted_text:
                lines.append(f"  extracted_text: {ev.extracted_text[:4000]}")
        if self.quoted_chain:
            lines.append("quoted_chain:")
            for ev in self.quoted_chain:
                lines.append(f"- [{ev.message_id}] {ev.sender_name or ev.sender_id}: {ev.text_body}")
        return "\n".join(lines).strip()


class WhatsAppContextStore:
    """Append-only per-chat WhatsApp event log plus context bundle builder."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.chats_dir = self.root / "chats"
        self.observability_dir = self.root / "reply-observability"
        self.chats_dir.mkdir(parents=True, exist_ok=True)
        self.observability_dir.mkdir(parents=True, exist_ok=True)

    def ingest_bridge_message(self, data: dict[str, Any]) -> WhatsAppChatEvent:
        event = self._event_from_bridge_message(data)
        self._append_event(event)
        return event

    def build_context_bundle(self, chat_id: str, trigger_message_id: str, *, limit: int = 50) -> WhatsAppContextBundle:
        events = self.load_chat_events(chat_id)
        by_id = {ev.message_id: ev for ev in events}
        current = by_id.get(trigger_message_id)
        if current is None:
            raise KeyError(f"No WhatsApp event {trigger_message_id!r} in chat {chat_id!r}")
        idx = events.index(current)
        recent = events[max(0, idx - limit + 1): idx + 1]
        quoted_chain = self._quoted_chain(current, by_id)
        recent_media = [ev for ev in recent if ev.message_type in {"image", "pdf", "document", "audio"} or ev.media_filename]
        unreadable = [ev for ev in recent_media if self._media_unreadable(ev)]
        payment_state = self._extract_payment_state(recent)
        task_state = payment_state.as_prompt_dict()
        entities = self._extract_entities(recent, current)
        guardrails = {
            "block_payment_recommendation": bool(
                payment_state.payment_proof_sent or payment_state.receipt_confirmed_by_counterparty
            )
        }
        context_ids = []
        for ev in [*recent, *quoted_chain]:
            if ev.message_id not in context_ids:
                context_ids.append(ev.message_id)
        media_paths = [ev.local_media_path for ev in recent_media if ev.local_media_path]
        return WhatsAppContextBundle(
            chat_id=chat_id,
            trigger_message_id=trigger_message_id,
            current_message=current,
            recent_events=recent,
            quoted_chain=quoted_chain,
            recent_media=recent_media,
            task_state=task_state,
            guardrails=guardrails,
            context_message_ids=context_ids,
            media_paths=media_paths,
            unreadable_media=unreadable,
            relevant_entities=entities,
        )

    def record_reply_observability(
        self,
        *,
        chat_id: str,
        trigger_message_id: str,
        context_bundle: WhatsAppContextBundle,
        final_prompt_preview: str,
        extraction_results: dict[str, Any] | None = None,
    ) -> Path:
        row = {
            "timestamp": _now_iso(),
            "chat_id": chat_id,
            "trigger_message_id": trigger_message_id,
            "context_message_ids": context_bundle.context_message_ids,
            "media_files_included": context_bundle.media_paths,
            "extracted_state_included": context_bundle.task_state,
            "final_prompt_context_preview": final_prompt_preview[:8000],
            "pdf_ocr_extraction": extraction_results or self._extraction_summary(context_bundle),
        }
        path = self.observability_dir / f"{_safe_name(chat_id)}.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        return path

    def load_chat_events(self, chat_id: str) -> list[WhatsAppChatEvent]:
        path = self._chat_path(chat_id)
        if not path.exists():
            return []
        events: list[WhatsAppChatEvent] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                payload = json.loads(line)
                events.append(WhatsAppChatEvent(**payload))
        return events

    def _append_event(self, event: WhatsAppChatEvent) -> None:
        path = self._chat_path(event.chat_id)
        existing_ids = {ev.message_id for ev in self.load_chat_events(event.chat_id)}
        if event.message_id in existing_ids:
            return
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(event), ensure_ascii=False, sort_keys=True) + "\n")

    def _chat_path(self, chat_id: str) -> Path:
        self.chats_dir.mkdir(parents=True, exist_ok=True)
        return self.chats_dir / f"{_safe_name(chat_id)}.jsonl"

    def _event_from_bridge_message(self, data: dict[str, Any]) -> WhatsAppChatEvent:
        chat_id = str(data.get("chatId") or data.get("from") or "")
        message_id = str(data.get("messageId") or data.get("id") or f"missing-{_now_iso()}")
        media_filename = _first_nonempty(
            data.get("mediaFileName"), data.get("fileName"), data.get("filename"), data.get("documentFileName"),
            *_as_list(data.get("mediaFileNames")), *_as_list(data.get("fileNames")),
        )
        media_type_raw = str(data.get("mediaType") or "").lower()
        has_media = bool(data.get("hasMedia") or media_filename or data.get("mediaUrls"))
        msg_type = "text"
        if has_media:
            if "image" in media_type_raw:
                msg_type = "image"
            elif "audio" in media_type_raw or "ptt" in media_type_raw:
                msg_type = "audio"
            elif media_filename and media_filename.lower().endswith(".pdf"):
                msg_type = "pdf"
            else:
                msg_type = "document"
        local_media_path = self._local_media_path(data)
        extracted_text = _first_nonempty(
            data.get("extractedText"), data.get("extracted_text"), data.get("ocrText"), data.get("ocr_text")
        )
        if not extracted_text and local_media_path:
            extracted_text = self._best_effort_extract_text(local_media_path)
        status = str(
            data.get("mediaDownloadStatus")
            or data.get("downloadStatus")
            or ("extracted" if extracted_text else "downloaded" if local_media_path else "failed" if has_media else "none")
        )
        mentioned = [str(v) for v in _as_list(data.get("mentionedIds"))]
        bot_ids = {str(v).lower() for v in _as_list(data.get("botIds"))}
        text = str(data.get("body") or data.get("text") or "")
        should_reply = bool({m.lower() for m in mentioned} & bot_ids) or bool(re.search(r"@\s*(jack|hermes)\b", text, re.I))
        return WhatsAppChatEvent(
            chat_id=chat_id,
            message_id=message_id,
            sender_id=str(data.get("senderId") or data.get("from") or ""),
            sender_name=str(data.get("senderName") or data.get("pushName") or ""),
            timestamp=str(data.get("timestamp") or _now_iso()),
            text_body=text,
            message_type=msg_type,
            quoted_message_id=_first_nonempty(data.get("quotedMessageId"), data.get("replyToMessageId")),
            mentioned_users=mentioned,
            media_filename=media_filename,
            media_mime_type=_first_nonempty(data.get("mediaMimeType"), data.get("mimeType"), _mime_from_type(msg_type, media_filename)),
            local_media_path=local_media_path,
            extracted_text=extracted_text,
            download_extraction_status=status,
            raw=dict(data),
            should_reply=should_reply,
        )

    def _local_media_path(self, data: dict[str, Any]) -> str | None:
        candidates = [
            data.get("mediaPath"), data.get("localMediaPath"), data.get("documentPath"),
            *_as_list(data.get("mediaUrls")), *_as_list(data.get("media_paths")),
        ]
        for value in candidates:
            if not value:
                continue
            text = str(value)
            if text.startswith("file://"):
                text = text[7:]
            if text.startswith("/"):
                return text
        return None

    def _best_effort_extract_text(self, path: str) -> str | None:
        p = Path(path)
        if not p.exists() or not p.is_file():
            return None
        if p.suffix.lower() in {".txt", ".md", ".csv", ".json", ".xml", ".yaml", ".yml", ".log"}:
            try:
                return p.read_text(encoding="utf-8", errors="replace")[:100_000]
            except OSError:
                return None
        # Optional PDF extraction: use pypdf if present, otherwise report as downloaded but unread.
        if p.suffix.lower() == ".pdf":
            try:
                from pypdf import PdfReader  # type: ignore

                reader = PdfReader(str(p))
                chunks = [(page.extract_text() or "") for page in reader.pages[:20]]
                text = "\n".join(chunks).strip()
                return text[:100_000] or None
            except Exception:
                return None
        return None

    def _quoted_chain(self, current: WhatsAppChatEvent, by_id: dict[str, WhatsAppChatEvent]) -> list[WhatsAppChatEvent]:
        chain: list[WhatsAppChatEvent] = []
        seen: set[str] = set()
        qid = current.quoted_message_id
        while qid and qid not in seen:
            seen.add(qid)
            ev = by_id.get(qid)
            if ev is None:
                break
            chain.append(ev)
            qid = ev.quoted_message_id
        return chain

    def _extract_payment_state(self, events: Iterable[WhatsAppChatEvent]) -> WhatsAppPaymentState:
        state = WhatsAppPaymentState()
        for ev in events:
            haystack = "\n".join(
                part for part in [ev.text_body, ev.media_filename or "", ev.extracted_text or ""] if part
            )
            if (ev.media_filename or ev.message_type in {"pdf", "image", "document"}) and _PAYMENT_WORD_RE.search(haystack):
                state.payment_proof_sent = True
                state.filename = state.filename or ev.media_filename
                state.proof_message_id = state.proof_message_id or ev.message_id
                state.sender = state.sender or ev.sender_name or ev.sender_id
                state.date_time = state.date_time or ev.timestamp
                state.amount = state.amount or _extract_amount(haystack)
                state.recipient_payee = state.recipient_payee or _extract_payee(haystack)
                state.linked_case_project = state.linked_case_project or _extract_case(haystack)
            if _CONFIRM_RECEIPT_RE.search(ev.text_body or ""):
                state.receipt_confirmed_by_counterparty = True
                state.confirmation_message_id = state.confirmation_message_id or ev.message_id
        return state

    def _extract_entities(self, events: Iterable[WhatsAppChatEvent], current: WhatsAppChatEvent) -> list[str]:
        text = "\n".join(
            part
            for ev in events
            for part in [ev.text_body, ev.media_filename or "", ev.extracted_text or "", ev.sender_name]
            if part
        ) + "\n" + current.text_body
        entities: list[str] = []
        for needle in ["Eduardo", "Gestión Inmobiliaria y Legal", "Gestión Inmobiliaria Y Legal"]:
            if needle.lower() in text.lower() and needle not in entities:
                entities.append(needle)
        case = _extract_case(text)
        if case and case not in entities:
            entities.append(case)
        return entities

    @staticmethod
    def _media_unreadable(ev: WhatsAppChatEvent) -> bool:
        if ev.message_type not in {"image", "pdf", "document", "audio"} and not ev.media_filename:
            return False
        status = (ev.download_extraction_status or "").lower()
        if status in {"failed", "download_failed", "extraction_failed", "unreadable"}:
            return True
        if not ev.local_media_path:
            return True
        if ev.message_type in {"pdf", "image", "document"} and not ev.extracted_text:
            return True
        return False

    @staticmethod
    def _extraction_summary(bundle: WhatsAppContextBundle) -> dict[str, Any]:
        return {
            ev.message_id: {
                "filename": ev.media_filename,
                "status": ev.download_extraction_status,
                "has_extracted_text": bool(ev.extracted_text),
            }
            for ev in bundle.recent_media
        }


def _extract_amount(text: str) -> str | None:
    candidates: list[tuple[float, str]] = []
    for match in _AMOUNT_RE.finditer(text):
        raw, suffix = match.group(1), (match.group(2) or "")
        # Avoid treating case numbers/years as payment amounts.
        span = match.span()
        surrounding = text[max(0, span[0] - 15): span[1] + 15]
        if "/" in surrounding and not suffix:
            continue
        normalized = raw.replace(",", "")
        try:
            value = float(normalized)
        except ValueError:
            value = 0.0
        if suffix.lower() in {"k", "mil"} and value < 1000:
            value *= 1000
        if value >= 100:
            display = f"${value:,.0f}" if value.is_integer() else f"${value:,.2f}"
            candidates.append((value, display))
    if not candidates:
        return None
    # Payment proofs usually mention the payment amount as the largest currency-like number.
    return sorted(candidates, reverse=True)[0][1]


def _extract_payee(text: str) -> str | None:
    if re.search(r"gesti[oó]n\s+inmobiliaria\s+y\s+legal", text, re.I):
        return "Gestión Inmobiliaria y Legal"
    m = re.search(r"(?:beneficiario|recipient|payee|a nombre de|para)\s*[:\-]?\s*([^\n.;,]{3,80})", text, re.I)
    if m:
        return m.group(1).strip()
    return None


def _extract_case(text: str) -> str | None:
    m = _CASE_RE.search(text)
    if not m:
        return None
    return re.sub(r"\s*/\s*", "/", m.group(0)).strip()


def _mime_from_type(msg_type: str, filename: str | None) -> str | None:
    if msg_type == "image":
        return "image/jpeg"
    if msg_type == "audio":
        return "audio/ogg"
    if filename and filename.lower().endswith(".pdf"):
        return "application/pdf"
    if msg_type == "document":
        return "application/octet-stream"
    return None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]


def _first_nonempty(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text:
            return text
    return None


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value or "unknown")[:180]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
