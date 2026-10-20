"""
WhatsApp Capture Hook — stores WhatsApp exchanges in Stomaton via MCP.
Fires on agent:end (after the agent responds).
Stores the user message and the agent response as SEPARATE rows
(#1111 slice D): the inbound as direction=INBOUND, the reply as a
first-class direction=OUTBOUND row — never concatenated.
Uploads media files to Stomaton S3 storage for permanent persistence.
Does NOT block the agent — Pedro is allowed to respond.
"""

import hashlib
import json
import os
import base64
import re
import threading
import time
import urllib.request
from datetime import datetime, timezone

STOMATON_MCP_URL = "https://stomaton.sez.ai/api/mcp"

# LID resolution (#1132): WhatsApp privacy LIDs (NNN@lid) hide the phone.
# The bridge maintains whatsapp/session/lid-mapping-*.json; hermes'
# canonical resolver walks them. Without this, wa-events carry a phone-less
# jid and Stomaton's identity gate refuses ref-stub adoption (first live
# widget test, MT-TNS6).
try:
    from gateway.whatsapp_identity import canonical_whatsapp_identifier
except Exception:  # capture must never die on an import
    canonical_whatsapp_identifier = None


def _resolve_lid_jid(user_id: str) -> str:
    """Phone-format jid for NNN@lid inputs when the bridge has a mapping;
    unchanged input otherwise (unmapped LIDs pass through)."""
    if "@lid" not in user_id or canonical_whatsapp_identifier is None:
        return user_id
    try:
        resolved = canonical_whatsapp_identifier(user_id)
    except Exception:
        return user_id
    digits = (resolved or "").split("@")[0]
    if digits.isdigit() and digits != user_id.split("@")[0]:
        return f"{digits}@s.whatsapp.net"
    return user_id


# #1128 — Marmolturco widget ref (repo mirror: lib/crm/marmolturco-ref.ts,
# MARMOLTURCO_REF_PATTERN). Confusable-free alphabet (no 0/O/1/I/L); the
# lookarounds exclude alphanumeric neighbors while letting the site's
# italic `_Ref: MT-XXXX_` format terminate the token.
MARMOLTURCO_REF_PATTERN = re.compile(
    r"(?<![A-Za-z0-9-])MT-[23456789ABCDEFGHJKMNPQRSTUVWXYZ]{4,8}(?![A-Za-z0-9])",
    re.IGNORECASE,
)
# MIME types we recognize for media
KNOWN_MEDIA_TYPES = {
    "image/jpeg", "image/png", "image/webp", "image/gif",
    "application/pdf",
    "audio/ogg", "audio/mpeg", "audio/mp4", "audio/wav",
    "video/mp4", "video/3gpp",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


def _mcp_call(tool_name: str, arguments: dict) -> dict | None:
    """Make a JSON-RPC call to Stomaton MCP."""
    api_key = os.getenv("STOMATON_MCP_API_KEY", "").strip()
    if not api_key:
        print("[whatsapp-capture] STOMATON_MCP_API_KEY is missing; refusing MCP call", flush=True)
        return None
    payload = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    }).encode("utf-8")

    req = urllib.request.Request(
        STOMATON_MCP_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            return result
    except Exception as e:
        print(f"[whatsapp-capture] MCP call {tool_name} failed: {e}", flush=True)
        return None


def _store_email(args: dict) -> bool:
    response = _mcp_call("stomaton_store_email", args)
    try:
        result = response["result"]
        if result.get("isError"):
            return False
        data = json.loads(result["content"][0]["text"])
        return data.get("success") is True and int(data["email"]["id"]) > 0
    except (AttributeError, KeyError, IndexError, TypeError, ValueError):
        return False


def _upload_media(file_path: str, mime_type: str, phone: str) -> tuple[str | None, str | None]:
    """Read a local file, base64 encode, upload via stomaton_upload_media.
    Returns (s3_key, mime_type) or (None, None) on failure."""
    if not os.path.isfile(file_path):
        print(f"[whatsapp-capture] Media file not found: {file_path}", flush=True)
        return None, None

    file_size = os.path.getsize(file_path)
    if file_size > 50 * 1024 * 1024:  # 50MB limit
        print(f"[whatsapp-capture] Media file too large: {file_size} bytes", flush=True)
        return None, None

    try:
        with open(file_path, "rb") as f:
            content_b64 = base64.b64encode(f.read()).decode("ascii")
    except Exception as e:
        print(f"[whatsapp-capture] Failed to read media file: {e}", flush=True)
        return None, None

    filename = os.path.basename(file_path)
    result = _mcp_call("stomaton_upload_media", {
        "content": content_b64,
        "filename": filename,
        "mimeType": mime_type,
        "source": "whatsapp",
        "phoneNumber": phone,
    })

    if result and "result" in result:
        try:
            content = result["result"].get("content", [{}])
            if content:
                data = json.loads(content[0].get("text", "{}"))
                if data.get("success"):
                    print(f"[whatsapp-capture] Uploaded {filename} ({file_size} bytes) → {data['s3Key']}", flush=True)
                    return data["s3Key"], mime_type
        except Exception as e:
            print(f"[whatsapp-capture] Upload response parse error: {e}", flush=True)

    return None, None


def _post_wa_event(user_id: str, message: str, ref: str) -> None:
    """#1128 — POST a ref-carrying inbound message as a wa-events extraction
    event so Stomaton adopts the click-beacon stub (buyer gets phone, no
    duplicate LIVE_WA lead). The route grants this key an adopt-only tier:
    attach yes, create never — a declined event becomes a forced decision
    card. Only fires when a widget ref is present; ordinary messages post
    nothing (store_email below already captures content).

    Confidence 85 / BUYER 95 are deterministic-prior stamps: the ref is only
    ever emitted by the site's quote widget, so its presence IS the buyer
    signal — no LLM guessed these numbers.
    """
    key = os.getenv("STOMATON_WA_MEDIA_KEY", "").strip()
    if not key:
        print("[whatsapp-capture] STOMATON_WA_MEDIA_KEY missing; skipping wa-event", flush=True)
        return
    media_url = os.getenv("STOMATON_WA_MEDIA_URL", "https://stomaton.sez.ai/api/ingest/wa-media")
    events_url = media_url.rstrip("/").replace("/wa-media", "/wa-events")
    jid = user_id if "@" in user_id else f"{user_id.lstrip('+')}@s.whatsapp.net"
    # Idempotency: same buyer + same ref + same text re-fires safely
    # (InboundEvent.externalId unique, createMany skipDuplicates).
    external_id = hashlib.sha256(f"selim-wa|{jid}|{ref}|{message}".encode()).hexdigest()
    event = {
        "source": "WA",
        "externalId": external_id,
        "contactJid": jid,
        "intent": "RFQ",
        "entities": {"reference_code": ref, "summary": message[:300]},
        "confidence": 85,
        "ballInCourt": "US",
        "contactType": "BUYER",
        "contactTypeConfidence": 95,
        "occurredAt": datetime.now(timezone.utc).isoformat(),
    }
    payload = json.dumps({"events": [event]}).encode("utf-8")
    for attempt in (1, 2):
        req = urllib.request.Request(
            events_url,
            data=payload,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode())
                print(
                    f"[whatsapp-capture] wa-event {ref} posted "
                    f"(created={body.get('created')}, duplicates={body.get('duplicates')})",
                    flush=True,
                )
                return
        except Exception as e:
            print(f"[whatsapp-capture] wa-event POST attempt {attempt} failed: {e}", flush=True)
    # Both attempts failed: the externalId is deterministic, so the buyer's
    # NEXT message with the same ref re-fires the adoption — no spool needed.


# ---------------------------------------------------------------- session read
# `agent:end` only carries the triggering message. These two tables carry the
# whole turn, with real timestamps. See module docstring.
WATERMARK_PATH = os.path.join(
    os.getenv("HERMES_HOME", "/root/.hermes/profiles/selim"),
    ".wa-capture-watermark.json",
)


def _state_db_path() -> str:
    return os.path.join(os.getenv("HERMES_HOME", "/root/.hermes/profiles/selim"), "state.db")


def _load_watermark() -> dict:
    try:
        with open(WATERMARK_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _save_watermark(mark: dict) -> None:
    try:
        tmp = WATERMARK_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(mark, fh)
        os.replace(tmp, WATERMARK_PATH)
    except Exception as e:
        print(f"[whatsapp-capture] watermark save failed: {e}", flush=True)


def _session_rows(session_id: str) -> tuple[list, list, dict] | None:
    """(inbound, outbound, new_watermark) since the last capture, or None.

    inbound  = [(row_id, epoch, text)]         from messages(role='user')
    outbound = [(obligation_id, epoch, text, chat_id)]  from delivery_obligations
    """
    import sqlite3

    path = _state_db_path()
    if not os.path.isfile(path):
        return None
    mark = _load_watermark()
    prev = mark.get(session_id) or {}
    last_msg_id = int(prev.get("msg_id") or 0)
    last_out_ts = float(prev.get("out_ts") or 0)
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10)
        try:
            # hook_ctx carries no session_key; the sessions table owns it, and
            # it is what delivery_obligations is keyed by.
            row = conn.execute(
                "SELECT session_key FROM sessions WHERE id=?", (session_id,)
            ).fetchone()
            session_key = (row[0] if row else None) or ""
            inbound = [
                (r[0], float(r[1] or 0), r[2] or "")
                for r in conn.execute(
                    "SELECT id, timestamp, content FROM messages "
                    "WHERE session_id=? AND role='user' AND id>? ORDER BY id",
                    (session_id, last_msg_id),
                )
                if (r[2] or "").strip()
            ]
            outbound = []
            if session_key:
                outbound = [
                    (r[0], float(r[1] or 0), r[2] or "", r[3] or "")
                    for r in conn.execute(
                        "SELECT obligation_id, created_at, content, chat_id FROM delivery_obligations "
                        "WHERE session_key=? AND created_at>? ORDER BY created_at",
                        (session_key, last_out_ts),
                    )
                    if (r[2] or "").strip()
                ]
        finally:
            conn.close()
    except Exception as e:
        print(f"[whatsapp-capture] state.db read failed ({e}); falling back to trigger-only", flush=True)
        return None

    new_mark = dict(mark)
    new_mark[session_id] = {
        "msg_id": inbound[-1][0] if inbound else last_msg_id,
        "out_ts": outbound[-1][1] if outbound else last_out_ts,
    }
    return inbound, outbound, new_mark


# --------------------------------------------------------- delivery (#2463)
# For 36h from 2026-10-17 every Selim reply was lost while being stored as
# delivered: "sent" only meant the bridge's sendMessage resolved. A reply is
# PROVIDER_ACCEPTED only once the bridge has sent something to that chat since
# the reply was ledgered AND WhatsApp's server acked all of it. Until then it
# is stored as UNKNOWN (the nightly self-check flags those), and a background
# check upgrades the row when the ack lands.
BRIDGE_URL = "http://127.0.0.1:3000"
ACK_WAIT_SECONDS = 180.0  # past the bridge's 120s unacked-send alarm
ACK_POLL_SECONDS = 5.0


def _acks_since(chat_id: str, since: float) -> dict | None:
    """{'sent': n, 'acked': m} from the bridge's ack watch, or None when the
    bridge can't tell (not running, or a build without /sends)."""
    from urllib.parse import urlencode

    query = urlencode({"chatId": chat_id, "since": f"{since:.3f}"})
    try:
        with urllib.request.urlopen(f"{BRIDGE_URL}/sends?{query}", timeout=3) as r:
            data = json.load(r)
        return {"sent": int(data["sent"]), "acked": int(data["acked"])}
    except Exception:
        return None


def _accepted(acks: dict) -> bool:
    return acks["sent"] > 0 and acks["acked"] == acks["sent"]


def _confirm_later(args: dict, chat_id: str, since: float) -> None:
    """Re-store the reply as PROVIDER_ACCEPTED once WhatsApp acks it; Stomaton
    upgrades a row still marked UNKNOWN and changes nothing else."""
    import time

    deadline = time.time() + ACK_WAIT_SECONDS
    while time.time() < deadline:
        time.sleep(ACK_POLL_SECONDS)
        acks = _acks_since(chat_id, since)
        if acks is not None and _accepted(acks):
            if _store_email({**args, "deliveryStatus": "PROVIDER_ACCEPTED"}):
                print(f"[whatsapp-capture] OUT {args['whatsappMsgId']} acked by WhatsApp", flush=True)
            return
    print(
        f"[whatsapp-capture] OUT {args['whatsappMsgId']} NOT acked by WhatsApp within "
        f"{int(ACK_WAIT_SECONDS)}s; stays UNKNOWN",
        flush=True,
    )


def _delivery_status(chat_id: str, since: float) -> str | None:
    """PROVIDER_ACCEPTED, UNKNOWN, or None when the bridge doesn't track acks
    (the row is then stored without a delivery status, as before)."""
    if not chat_id:
        return None
    acks = _acks_since(chat_id, since)
    if acks is None:
        return None
    return "PROVIDER_ACCEPTED" if _accepted(acks) else "UNKNOWN"


def _iso(epoch: float) -> str:
    try:
        return datetime.fromtimestamp(epoch, timezone.utc).isoformat()
    except Exception:
        return datetime.now(timezone.utc).isoformat()


def _await_obligation(session_id: str, response: str, timeout: float = 12.0) -> None:
    """Block until this turn's reply has a delivery_obligations row (or give up).

    Best-effort: on timeout we simply fall through and the next turn picks it
    up, which is the pre-fix behaviour rather than a regression.
    """
    import sqlite3
    import time

    path = _state_db_path()
    if not os.path.isfile(path):
        return
    needle = (response or "").strip()[:200]
    if not needle:
        return
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
            try:
                row = conn.execute(
                    "SELECT session_key FROM sessions WHERE id=?", (session_id,)
                ).fetchone()
                key = (row[0] if row else None) or ""
                if key:
                    hit = conn.execute(
                        "SELECT 1 FROM delivery_obligations "
                        "WHERE session_key=? AND substr(content,1,200)=? LIMIT 1",
                        (key, needle),
                    ).fetchone()
                    if hit:
                        return
            finally:
                conn.close()
        except Exception:
            return
        time.sleep(1.0)
    print("[whatsapp-capture] obligation not visible within timeout; next turn will capture it", flush=True)


async def handle(event_type: str, context: dict) -> None:
    platform = context.get("platform", "")
    if platform != "whatsapp":
        return

    message = context.get("message", "")
    response = context.get("response", "")
    user_id = _resolve_lid_jid(str(context.get("user_id", "unknown")))
    chat_id = str(context.get("chat_id", ""))
    session_id = context.get("session_id", "")
    media_urls = context.get("media_urls", [])
    media_types = context.get("media_types", [])

    if not message and not response:
        return

    # Extract phone number
    phone = user_id.split("@")[0] if "@" in user_id else user_id
    if phone and not phone.startswith("+"):
        phone = "+" + phone

    is_group = "@g.us" in chat_id

    # #1128: widget buyers get phone + ref-stub adoption automatically.
    if message and not is_group:
        ref_match = MARMOLTURCO_REF_PATTERN.search(message)
        if ref_match:
            _post_wa_event(user_id, message, ref_match.group(0).upper())

    subject = message[:100].split("\n")[0]
    if is_group:
        subject = f"[Group {chat_id.split('@')[0]}] {subject}"

    body_text = message

    # Upload media files if present
    uploaded_s3_key = None
    uploaded_mime_type = None
    media_descriptions = []

    for i, file_path in enumerate(media_urls):
        mime = media_types[i] if i < len(media_types) else ""
        if not file_path or not os.path.isfile(str(file_path)):
            continue

        # Upload to S3
        s3_key, mtype = _upload_media(str(file_path), mime, phone)
        if s3_key:
            # Use first uploaded file as the primary media
            if uploaded_s3_key is None:
                uploaded_s3_key = s3_key
                uploaded_mime_type = mtype
            filename = os.path.basename(str(file_path))
            media_descriptions.append(f"[Attachment: {filename} ({mime})]")

    # Append media descriptions to body
    if media_descriptions:
        body_text = body_text + "\n\n" + "\n".join(media_descriptions)

    now = datetime.now(timezone.utc).isoformat()
    msg_hash = hash(f"{user_id}-{message}-{now[:16]}") & 0xFFFFFFFF

    def _store_in(wid: str, text: str, when: str, with_media: bool) -> bool:
        subj = text[:100].split("\n")[0]
        if is_group:
            subj = f"[Group {chat_id.split('@')[0]}] {subj}"
        args = {
            "channel": "WHATSAPP",
            "fromEmail": phone,
            "fromName": phone,
            "toEmail": "marmolturco",
            "subject": subj,
            "bodyText": text,
            "account": "whatsapp-marmolturco",
            "phoneNumber": phone,
            "whatsappMsgId": wid,
            "direction": "INBOUND",
            "gmailId": wid,
            "receivedAt": when,
            "scope": "PLATFORM",
        }
        if with_media and uploaded_s3_key:
            args["mediaUrl"] = uploaded_s3_key
            args["mediaType"] = uploaded_mime_type
            args["hasAttachments"] = True
        if _store_email(args):
            print(f"[whatsapp-capture] stored IN {wid}: {phone} — {subj[:50]}", flush=True)
            return True
        print(f"[whatsapp-capture] store IN FAILED {wid} for {phone}", flush=True)
        return False

    def _store_out(wid: str, text: str, when: str, to_chat: str, since: float) -> bool:
        subj = f"Re: {text[:96].split(chr(10))[0]}"
        args = {
            "channel": "WHATSAPP",
            "fromEmail": "whatsapp-marmolturco",
            "fromName": "Selim",
            "toEmail": phone,
            "subject": subj,
            "bodyText": text,
            "account": "whatsapp-marmolturco",
            "phoneNumber": phone,
            "whatsappMsgId": wid,
            "direction": "OUTBOUND",
            "sentAt": when,
            "gmailId": wid,
            "receivedAt": when,
            "scope": "PLATFORM",
        }
        delivery = _delivery_status(to_chat, since)
        if delivery:
            args["deliveryStatus"] = delivery
        if _store_email(args):
            print(f"[whatsapp-capture] stored OUT {wid}: → {phone} ({delivery or 'delivery not tracked'})", flush=True)
            if delivery == "UNKNOWN":
                threading.Thread(
                    target=_confirm_later, args=(args, to_chat, since), daemon=True,
                    name=f"wa-capture-ack-{wid}",
                ).start()
            return True
        print(f"[whatsapp-capture] store OUT FAILED {wid} for {phone}", flush=True)
        return False

    # Preferred path: replay the whole turn from the session, with real
    # timestamps. Media rides the last inbound of the batch, which is the
    # message the gateway actually handed us.
    # delivery_obligations is written a few seconds after agent:end fires, so
    # without this wait the LAST reply of a conversation is only captured by
    # the next turn — and never at all if the buyer doesn't write again.
    if response:
        _await_obligation(str(session_id), response)

    rows = _session_rows(str(session_id))
    if rows is not None:
        inbound, outbound, new_mark = rows
        stored = True
        for idx, (row_id, ts, text) in enumerate(inbound):
            stored = _store_in(f"wa-{session_id}-m{row_id}", text, _iso(ts), idx == len(inbound) - 1) and stored
        for oid, ts, text, to_chat in outbound:
            # The obligation is ledgered just BEFORE the send, so every send
            # for this reply happens after `ts`.
            stored = _store_out(f"wa-do-{oid}", text, _iso(ts), to_chat or chat_id, ts) and stored
        if (inbound or outbound) and stored:
            _save_watermark(new_mark)
        elif not inbound and not outbound:
            print("[whatsapp-capture] nothing new in session since last capture", flush=True)
        return

    # Fallback: the pre-2026-08-07 trigger-only behaviour.
    if message:
        _store_in(f"wa-{session_id}-{msg_hash:08x}", body_text, now, True)
    if response:
        out_hash = hash(f"{user_id}-out-{response}-{now[:16]}") & 0xFFFFFFFF
        _store_out(f"wa-{session_id}-{out_hash:08x}-out", response, now, chat_id or user_id, time.time() - 120)
