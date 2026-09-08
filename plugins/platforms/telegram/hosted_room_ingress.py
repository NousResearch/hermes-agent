"""Reserve a configured Group Chat on the existing Telegram polling connection."""
import hashlib
import json
import logging
import sqlite3
import time
from contextlib import closing
from pathlib import Path

from .hosted_room_transport import hosted_room_binding_path, initialize_queue, load_binding

log = logging.getLogger(__name__)
MAX_TEXT_BYTES = 12000


def _attachment_kind(cached):
    """The canonical Files kind for cached media: image, pdf or file.

    Voice, audio and video are `file` carrying their real MIME. The room's attachment contract
    has exactly these three kinds, and adding one for media would be a second schema.
    """
    mime = (getattr(cached, 'media_type', '') or '').lower()
    if getattr(cached, 'kind', '') == 'image':
        return 'image'
    return 'pdf' if mime == 'application/pdf' else 'file'


async def capture_media(adapter, message):
    """Capture this message's attachment through the NATIVE downloader and media cache.

    Returns ``(attachments, rejected)``: bounded transport references -- never bytes in this
    queue -- for the canonical upload the transport performs later, or a bounded receipt saying
    why media this message really carried could not be captured. Those are different outcomes:
    treating a failed download as "no media" would admit the caption alone as if the picture had
    arrived. No second downloader, cache, size policy or schema; the byte cap is the room's own,
    and the upload id comes from the Telegram identity so a retry stages the same upload.
    """
    from tools.credential_files import from_agent_visible_cache_path

    # Bound capture with the native adapter limit; Files enforces its own cap at admission.
    max_bytes = int(getattr(adapter, '_max_doc_bytes', 20 * 1024 * 1024) or 0) or 20 * 1024 * 1024
    download = getattr(adapter, '_download_observed_media', None)
    if download is None:
        # The message may well carry media this build cannot even look at; say so.
        return [], {'reason': 'downloader_unavailable'}
    status, cached = await download(message, 'hosted room media')
    if status == 'none':
        return [], None
    if status != 'ok' or cached is None:
        return [], {'reason': str(status)}
    # `CachedMedia.path` is the AGENT-visible path and may be container-translated.
    host_path = Path(from_agent_visible_cache_path(str(cached.path)))
    try:
        with host_path.open('rb') as handle:
            data = handle.read(max_bytes + 1)
    except OSError:
        return [], {'reason': 'unreadable'}
    if not 0 < len(data) <= max_bytes:
        return [], {'reason': 'oversized'}
    return [{
        'kind': _attachment_kind(cached),
        'mime': (getattr(cached, 'media_type', '') or 'application/octet-stream').lower(),
        'name': getattr(cached, 'display_name', '') or host_path.name,
        'path': str(host_path),
        'size': len(data),
        'sha256': hashlib.sha256(data).hexdigest(),
        'upload_id': f'telegram:{message.chat_id}:{message.message_id}:0',
    }], None


def wire(application, adapter) -> bool:
    """Wire each native PTB Application; return whether its binding is active."""
    path = hosted_room_binding_path()
    if path is None:
        return False
    config = load_binding(path, include_disabled=True)
    assert config is not None  # include_disabled returns every validated binding.
    chat_id, owner_id = config["chat_id"], config["owner_id"]
    db_path = Path(config["queue_db"])
    initialize_queue(db_path)
    from telegram.ext import ApplicationHandlerStop, MessageHandler, filters

    async def incoming(update, context):
        # Even rejected/disabled room inputs cannot fall through to independent agents.
        try:
            message, sender = update.effective_message, update.effective_user
            if (not message or not sender or sender.is_bot or sender.id != owner_id
                    or message.chat_id != chat_id or update.edited_message is not None):
                return
            text = message.text or message.caption or ""
            if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
                return
            live = load_binding(path)
            if live is None or live != config:
                return  # Disable immediately; routing changes require a gateway restart.
            attachments, rejected = await capture_media(adapter, message)
            if not text and not attachments and not rejected:
                return
            media = ({"attachments": attachments, **({"rejected": rejected} if rejected else {})}
                     if attachments or rejected else None)
            reply = message.reply_to_message
            with closing(sqlite3.connect(db_path, timeout=10)) as db, db:
                db.execute("""INSERT OR IGNORE INTO inbox
                    (event_id,chat_id,message_id,user_id,text,reply_to,topic_id,received_at,media_json)
                    VALUES(?,?,?,?,?,?,?,?,?)""", (
                        f"telegram:{chat_id}:{message.message_id}", chat_id,
                        message.message_id, sender.id, text, reply.message_id if reply else None,
                        message.message_thread_id, time.time(), json.dumps(media) if media else None))
            log.info("Hosted room ingress accepted chat=%s message=%s", chat_id, message.message_id)
        except Exception as exc:
            log.error("Hosted room ingress failed closed: %s", type(exc).__name__)
        finally:
            raise ApplicationHandlerStop

    application.add_handler(MessageHandler(filters.Chat(chat_id), incoming), group=-50)
    log.warning("Canonical Telegram ingress wired chat=%s owner=%s", chat_id, owner_id)
    return config["enabled"]
