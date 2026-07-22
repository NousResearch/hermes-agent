"""Crypto core for the X Chat platform adapter.

A thin, network-free wrapper around the ``chat_xdk`` binding. Everything
that touches the Chat XDK lives here so it can be unit-tested with a fake
``Chat`` object and so the adapter/API layers stay import-light. The SDK is
lazy-installed at first use via ``tools.lazy_deps`` (feature key
``platform.xchat``).

Responsibilities:

* key management        -> :meth:`XChatCrypto.load_keys` /
                           :meth:`XChatCrypto.generate_and_register_payload`
* session identity      -> :meth:`XChatCrypto.set_identity`
* signing-key roster    -> :meth:`XChatCrypto.set_signing_keys`
* message encryption    -> :meth:`XChatCrypto.encrypt_text`
* event decryption      -> :meth:`XChatCrypto.decrypt_batch` (decrypt_events)
                           and :meth:`XChatCrypto.decrypt_one` (decrypt_event)

The decrypted-event dict shape follows the Chat XDK: ``{"type": "Message",
"id": ..., "sender_id": ..., "conversation_id": ..., "content": {"text":
...}}`` for messages, ``{"type": "KeyChange", ...}`` for conversation-key
rotations.
"""
from __future__ import annotations

import base64
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _as_dict(obj: Any) -> dict[str, Any]:
    """Decrypted events come back as native objects; normalise to a dict."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    try:
        return dict(obj)
    except Exception:
        return {}


def _load_chat_class():
    """Import (lazy-installing if needed) and return ``chat_xdk.Chat``."""
    try:
        from chat_xdk import Chat  # type: ignore[import-not-found]
        return Chat
    except ImportError:
        pass
    # Lazy-install path — same pattern as the telegram/matrix platform plugins.
    from tools.lazy_deps import ensure as _lazy_ensure

    _lazy_ensure("platform.xchat", prompt=False)
    from chat_xdk import Chat  # type: ignore[import-not-found]
    return Chat


class XChatCrypto:
    """Wraps a single unlocked ``chat_xdk.Chat`` instance for one bot identity."""

    def __init__(self, chat: Any = None) -> None:
        # ``chat`` injection keeps unit tests free of the native SDK.
        self.chat = chat if chat is not None else _load_chat_class()()
        self.signing_key_version: str = "1"
        self._identity_set = False

    # -- Key management -----------------------------------------------------

    def load_keys(self, private_keys_b64: str, signing_key_version: str = "1") -> None:
        """Import an existing private-key blob (from ``export_keys``) and adopt it.

        ``private_keys_b64`` is the base64 blob produced during registration
        (``hermes xchat setup``). Raises on a malformed blob.
        """
        blob = base64.b64decode(private_keys_b64.strip())
        self.chat.import_keys(blob, version=signing_key_version)
        self.signing_key_version = str(signing_key_version)

    def set_identity(self, user_id: str) -> None:
        """Set the session identity — every later encrypt call signs as this user."""
        self.chat.set_identity(str(user_id), self.signing_key_version)
        self._identity_set = True

    def set_cache_keys(self, enabled: bool = True) -> None:
        """Opt in to the SDK's verified conversation-key cache."""
        self.chat.set_cache_keys(enabled)

    def set_signing_keys(self, signing_keys: list[dict[str, str]]) -> None:
        """Replace the SDK's participant signing-key store (full roster each call)."""
        self.chat.set_signing_keys(signing_keys)

    def generate_and_register_payload(self) -> dict[str, Any]:
        """Generate fresh keypairs for a brand-new bot identity.

        Returns the registration body for ``POST /2/users/{id}/public_keys``
        plus the exported private-key blob (base64) to persist locally.
        Used by ``hermes xchat setup`` only — the adapter never generates keys.
        """
        reg = self.chat.generate_keypairs()
        version = str(reg.version) if getattr(reg, "version", None) is not None else "1"
        body = {
            "public_key": {
                "public_key": reg.public_key.public_key,
                "signing_public_key": reg.public_key.signing_public_key,
                "identity_public_key_signature": reg.public_key.identity_public_key_signature,
                "signing_public_key_signature": reg.public_key.signing_public_key_signature,
                "registration_method": reg.public_key.registration_method,
            },
            "version": version,
            "generate_version": bool(getattr(reg, "generate_version", False)),
        }
        exported = self.chat.export_keys()
        blob_b64 = base64.b64encode(bytes(exported)).decode("ascii") if exported else ""
        return {"registration": body, "version": version, "private_keys_b64": blob_b64}

    # -- Decryption ----------------------------------------------------------

    def decrypt_batch(self, events_b64: list[str]) -> dict[str, Any]:
        """Batch path — initial backlog load and KeyChange processing.

        ``decrypt_events`` extracts conversation keys from any KeyChange
        events in the batch (feeding the SDK's key cache when enabled), then
        decrypts every message. Signing keys come from the
        ``set_signing_keys`` store.
        """
        result = self.chat.decrypt_events(events_b64, None)
        messages = [
            {"event": _as_dict(m.get("event") if isinstance(m, dict) else m)}
            for m in (result.get("messages") or [])
        ]
        return {
            "messages": messages,
            "conversation_keys": result.get("conversation_keys") or {},
            "errors": result.get("errors") or {},
        }

    def decrypt_one(
        self, event_b64: str, conversation_keys: Optional[dict[str, bytes]] = None
    ) -> dict[str, Any]:
        """Single-event path — per-poll decryption with cached conversation keys."""
        return _as_dict(self.chat.decrypt_event(event_b64, conversation_keys, None))

    # -- Encryption ----------------------------------------------------------

    def encrypt_text(self, conversation_id: str, text: str) -> dict[str, str]:
        """Encrypt + sign ``text``, returning the X API send-message body.

        The conversation key is resolved from the SDK's verified-key cache
        (``set_cache_keys``); the sender comes from ``set_identity``. Raises
        ``ValueError`` when no verified key is cached for the conversation.
        """
        payload = self.chat.encrypt_message(str(conversation_id), text)
        return {
            "message_id": payload.message_id,
            "encoded_message_create_event": payload.encrypted_content,
            "encoded_message_event_signature": payload.encoded_event_signature,
        }


def message_text(event: dict[str, Any]) -> Optional[str]:
    """Pull the plain text out of a decrypted Message event, or None."""
    if event.get("type") != "Message":
        return None
    content = event.get("content") or {}
    if isinstance(content, dict):
        return content.get("text")
    return None
