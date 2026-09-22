"""Threema E2E container crypto — key handling, padding, NaCl boxes, callback MACs.

Pure functions over ``bytes``; no network, no I/O beyond reading a key file. The
only third-party import is PyNaCl, whose wheels bundle libsodium (the official
``threema.gateway`` SDK instead depends on ``libnacl``, a ctypes binding that
needs a system ``libsodium`` install, plus ``logbook`` and ``click``).

Container layout (https://gateway.threema.ch/en/developer/e2e):

    box = crypto_box(type-byte || inner || pkcs7-padding, nonce, their_pub, my_priv)

The padding is random 1..255 bytes of PKCS#7, with one wrinkle that is easy to
miss: if ``len(inner) + 1 + pad < 32`` the pad grows so the padded plaintext is
exactly 32 bytes, so that short messages do not leak their length.
"""

from __future__ import annotations

import binascii
import hmac
import os
import secrets
from hashlib import sha256
from typing import Optional, Tuple

try:  # PyNaCl ships libsodium inside the wheel — no system package needed.
    from nacl.exceptions import CryptoError
    from nacl.public import Box, PrivateKey, PublicKey
    from nacl.secret import SecretBox
    from nacl.utils import random as nacl_random
    NACL_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by check_requirements()
    Box = PrivateKey = PublicKey = SecretBox = None  # type: ignore[assignment]
    CryptoError = Exception  # type: ignore[assignment,misc]
    nacl_random = None  # type: ignore[assignment]
    NACL_AVAILABLE = False

KEY_LENGTH = 32
NONCE_LENGTH = 24
MIN_PADDED_LENGTH = 32
MAX_PAD = 255
# Blob symmetric nonces are fixed by the protocol: 23 zero bytes then 0x01 for
# the file itself and 0x02 for its thumbnail.
BLOB_FILE_NONCE = (b"\x00" * 23) + b"\x01"
BLOB_THUMBNAIL_NONCE = (b"\x00" * 23) + b"\x02"

# Container type bytes.
TYPE_TEXT = 0x01
TYPE_LOCATION = 0x10
TYPE_POLL_SETUP = 0x15
TYPE_POLL_VOTE = 0x16
TYPE_FILE = 0x17
TYPE_DELIVERY_RECEIPT = 0x80

# Delivery-receipt statuses.
RECEIPT_RECEIVED = 0x01
RECEIPT_READ = 0x02
RECEIPT_ACKNOWLEDGED = 0x03
RECEIPT_DECLINED = 0x04


class ThreemaCryptoError(Exception):
    """Key, MAC, or container failure. The message is safe to log — it never carries plaintext."""


def _unhex(value: str, field: str, expected_len: Optional[int] = None) -> bytes:
    try:
        raw = binascii.unhexlify(value.strip())
    except (binascii.Error, ValueError) as exc:
        raise ThreemaCryptoError(f"{field} is not valid hex") from exc
    if expected_len is not None and len(raw) != expected_len:
        raise ThreemaCryptoError(f"{field} must be {expected_len} bytes, got {len(raw)}")
    return raw


def parse_key(encoded: str, *, expect: str = "private") -> bytes:
    """Accept ``private:<64 hex>`` / ``public:<64 hex>`` (panel format) or bare hex."""
    text = (encoded or "").strip()
    if not text:
        raise ThreemaCryptoError(f"Empty {expect} key")
    if ":" in text:
        kind, _, body = text.partition(":")
        kind = kind.strip().lower()
        if kind not in ("private", "public"):
            raise ThreemaCryptoError(f"Unknown key type {kind!r}; expected 'private:' or 'public:'")
        if kind != expect:
            raise ThreemaCryptoError(f"Expected a {expect} key but got a {kind} key")
        text = body
    return _unhex(text, f"{expect} key", KEY_LENGTH)


def load_private_key(path_or_value: str) -> bytes:
    """Read the key from a file when the value looks like a path, else parse it directly.

    Threema's panel hands out ``private:<hex>``; users usually save that line to a
    file. Reading the file keeps the secret out of the process environment.
    """
    value = (path_or_value or "").strip()
    if not value:
        raise ThreemaCryptoError("No private key configured")
    expanded = os.path.expanduser(value)
    if os.path.isfile(expanded):
        try:
            with open(expanded, "r", encoding="utf-8") as handle:
                value = handle.read().strip()
        except OSError as exc:
            raise ThreemaCryptoError(f"Cannot read private key file: {exc}") from exc
        if not value:
            raise ThreemaCryptoError(f"Private key file {expanded} is empty")
    return parse_key(value, expect="private")


def derive_public_key(private_key: bytes) -> bytes:
    """Our own public key — handy for `check` output so the user can verify the pair."""
    _require_nacl()
    return bytes(PrivateKey(private_key).public_key)


def _require_nacl() -> None:
    if not NACL_AVAILABLE:
        raise ThreemaCryptoError("PyNaCl is not installed (pip install pynacl)")


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------


def pad(data: bytes, *, pad_length: Optional[int] = None) -> bytes:
    """PKCS#7 with a random 1..255 length, widened so the result is >= 32 bytes."""
    length = secrets.randbelow(MAX_PAD) + 1 if pad_length is None else int(pad_length)
    if not 1 <= length <= MAX_PAD:
        raise ThreemaCryptoError(f"Pad length must be 1..{MAX_PAD}, got {length}")
    if len(data) + length < MIN_PADDED_LENGTH:
        length = MIN_PADDED_LENGTH - len(data)
    return data + bytes([length]) * length


def unpad(padded: bytes) -> bytes:
    """Strip PKCS#7 padding. The trailing byte is the count; it is never zero."""
    if not padded:
        raise ThreemaCryptoError("Empty container")
    length = padded[-1]
    if length == 0 or length > len(padded):
        raise ThreemaCryptoError("Container padding is invalid")
    return padded[:-length]


# ---------------------------------------------------------------------------
# Container encryption
# ---------------------------------------------------------------------------


def encrypt_container(
    message_type: int,
    inner: bytes,
    private_key: bytes,
    recipient_public_key: bytes,
    *,
    nonce: Optional[bytes] = None,
    pad_length: Optional[int] = None,
) -> Tuple[bytes, bytes]:
    """Return ``(nonce, box)`` for one E2E message."""
    _require_nacl()
    if not 0 <= int(message_type) <= 0xFF:
        raise ThreemaCryptoError(f"Message type {message_type} is not a byte")
    nonce = nonce or nacl_random(NONCE_LENGTH)
    if len(nonce) != NONCE_LENGTH:
        raise ThreemaCryptoError(f"Nonce must be {NONCE_LENGTH} bytes, got {len(nonce)}")
    plaintext = pad(bytes([message_type]) + inner, pad_length=pad_length)
    box = Box(PrivateKey(private_key), PublicKey(recipient_public_key))
    return nonce, box.encrypt(plaintext, nonce=nonce).ciphertext


def decrypt_container(
    box_bytes: bytes,
    nonce: bytes,
    private_key: bytes,
    sender_public_key: bytes,
) -> Tuple[int, bytes]:
    """Return ``(type_byte, inner)``. Raises when the box does not authenticate."""
    _require_nacl()
    try:
        plaintext = Box(PrivateKey(private_key), PublicKey(sender_public_key)).decrypt(box_bytes, nonce)
    except CryptoError as exc:
        raise ThreemaCryptoError("Could not decrypt message (wrong key pair or corrupt box)") from exc
    payload = unpad(plaintext)
    if not payload:
        raise ThreemaCryptoError("Container has no type byte")
    return payload[0], payload[1:]


def encrypt_blob(data: bytes, key: bytes, *, thumbnail: bool = False) -> bytes:
    """Symmetric blob encryption with the protocol's fixed nonce."""
    _require_nacl()
    nonce = BLOB_THUMBNAIL_NONCE if thumbnail else BLOB_FILE_NONCE
    return SecretBox(key).encrypt(data, nonce=nonce).ciphertext


def decrypt_blob(data: bytes, key: bytes, *, thumbnail: bool = False) -> bytes:
    _require_nacl()
    nonce = BLOB_THUMBNAIL_NONCE if thumbnail else BLOB_FILE_NONCE
    try:
        return SecretBox(key).decrypt(data, nonce)
    except CryptoError as exc:
        raise ThreemaCryptoError("Could not decrypt blob") from exc


def random_blob_key() -> bytes:
    _require_nacl()
    return nacl_random(KEY_LENGTH)


# ---------------------------------------------------------------------------
# Callback authentication
# ---------------------------------------------------------------------------


def callback_mac(fields: dict, secret: str) -> str:
    """``HMAC-SHA256(from || to || messageId || date || nonce || box, secret)``.

    The parameters are concatenated exactly as POSTed — hex still hex, no URL
    decoding beyond form parsing — which is why this takes the raw form values.
    """
    message = "".join(str(fields.get(name, "")) for name in ("from", "to", "messageId", "date", "nonce", "box"))
    return hmac.new(secret.encode("utf-8"), message.encode("utf-8"), sha256).hexdigest()


def verify_callback(fields: dict, secret: str) -> bool:
    """Constant-time MAC check. Run this BEFORE parsing or decrypting anything."""
    provided = str(fields.get("mac", "") or "").strip().lower()
    if not provided or not secret:
        return False
    return hmac.compare_digest(provided, callback_mac(fields, secret))
