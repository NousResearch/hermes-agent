"""Dependency-free Nostr signing for Buzz WebSocket authentication."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import struct
import time
from typing import Any, Optional


FIELD_ORDER = 2**256 - 2**32 - 977
CURVE_ORDER = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
GENERATOR = (
    0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
    0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8,
)
BECH32_CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"

Point = Optional[tuple[int, int]]


def _bech32_polymod(values: list[int]) -> int:
    generators = (0x3B6A57B2, 0x26508E6D, 0x1EA119FA, 0x3D4233DD, 0x2A1462B3)
    checksum = 1
    for value in values:
        top = checksum >> 25
        checksum = ((checksum & 0x1FFFFFF) << 5) ^ value
        for index, generator in enumerate(generators):
            if (top >> index) & 1:
                checksum ^= generator
    return checksum


def _bech32_hrp_expand(hrp: str) -> list[int]:
    return [ord(char) >> 5 for char in hrp] + [0] + [ord(char) & 31 for char in hrp]


def _decode_nsec(value: str) -> bytes:
    if value.lower() != value and value.upper() != value:
        raise ValueError("nsec cannot mix upper- and lowercase")
    normalized = value.lower()
    separator = normalized.rfind("1")
    if separator < 1 or separator + 7 > len(normalized):
        raise ValueError("invalid nsec encoding")
    hrp = normalized[:separator]
    if hrp != "nsec":
        raise ValueError("private key must use the nsec prefix")
    try:
        data = [BECH32_CHARSET.index(char) for char in normalized[separator + 1 :]]
    except ValueError as exc:
        raise ValueError("invalid character in nsec") from exc
    if _bech32_polymod(_bech32_hrp_expand(hrp) + data) != 1:
        raise ValueError("invalid nsec checksum")

    accumulator = 0
    bits = 0
    decoded = bytearray()
    for value5 in data[:-6]:
        accumulator = (accumulator << 5) | value5
        bits += 5
        while bits >= 8:
            bits -= 8
            decoded.append((accumulator >> bits) & 0xFF)
    if bits and (accumulator & ((1 << bits) - 1)):
        raise ValueError("non-zero nsec padding")
    if len(decoded) != 32:
        raise ValueError("nsec must encode exactly 32 bytes")
    return bytes(decoded)


def decode_private_key(value: str) -> int:
    raw = value.strip()
    if raw.lower().startswith("nsec1"):
        key_bytes = _decode_nsec(raw)
    else:
        try:
            key_bytes = bytes.fromhex(raw)
        except ValueError as exc:
            raise ValueError("private key must be 64 hex characters or nsec") from exc
        if len(key_bytes) != 32:
            raise ValueError("private key must be 32 bytes")
    key = int.from_bytes(key_bytes, "big")
    if not 1 <= key < CURVE_ORDER:
        raise ValueError("private key is outside the secp256k1 range")
    return key


def _point_add(left: Point, right: Point) -> Point:
    if left is None:
        return right
    if right is None:
        return left
    x1, y1 = left
    x2, y2 = right
    if x1 == x2:
        if (y1 + y2) % FIELD_ORDER == 0:
            return None
        slope = (3 * x1 * x1) * pow(2 * y1, FIELD_ORDER - 2, FIELD_ORDER)
    else:
        slope = (y2 - y1) * pow(x2 - x1, FIELD_ORDER - 2, FIELD_ORDER)
    slope %= FIELD_ORDER
    x3 = (slope * slope - x1 - x2) % FIELD_ORDER
    y3 = (slope * (x1 - x3) - y1) % FIELD_ORDER
    return x3, y3


def _point_multiply(scalar: int, point: Point = GENERATOR) -> Point:
    result: Point = None
    addend = point
    while scalar:
        if scalar & 1:
            result = _point_add(result, addend)
        addend = _point_add(addend, addend)
        scalar >>= 1
    return result


def _tagged_hash(tag: str, payload: bytes) -> bytes:
    tag_hash = hashlib.sha256(tag.encode()).digest()
    return hashlib.sha256(tag_hash + tag_hash + payload).digest()


def public_key_hex(private_key: str) -> str:
    point = _point_multiply(decode_private_key(private_key))
    if point is None:  # pragma: no cover - range validation makes this unreachable
        raise ValueError("invalid private key")
    return point[0].to_bytes(32, "big").hex()


def _lift_x(public_key: str) -> tuple[int, int]:
    try:
        x = int(public_key, 16)
    except ValueError as exc:
        raise ValueError("public key must be 64 hex characters") from exc
    if len(public_key) != 64 or x >= FIELD_ORDER:
        raise ValueError("public key must be a valid 32-byte x-only key")
    y_squared = (pow(x, 3, FIELD_ORDER) + 7) % FIELD_ORDER
    y = pow(y_squared, (FIELD_ORDER + 1) // 4, FIELD_ORDER)
    if pow(y, 2, FIELD_ORDER) != y_squared:
        raise ValueError("public key is not on secp256k1")
    return x, y if y % 2 == 0 else FIELD_ORDER - y


def validate_x_only_public_key(public_key: str) -> str:
    """Return a normalized x-only key or raise when it is not on secp256k1."""

    normalized = str(public_key or "").strip().lower()
    _lift_x(normalized)
    return normalized


def _hkdf_expand(prk: bytes, info: bytes, length: int) -> bytes:
    output = bytearray()
    previous = b""
    counter = 1
    while len(output) < length:
        previous = hmac.new(prk, previous + info + bytes([counter]), hashlib.sha256).digest()
        output.extend(previous)
        counter += 1
    return bytes(output[:length])


def _rotate_left(value: int, shift: int) -> int:
    return ((value << shift) & 0xFFFFFFFF) | (value >> (32 - shift))


def _chacha20_xor(key: bytes, nonce: bytes, payload: bytes) -> bytes:
    if len(key) != 32 or len(nonce) != 12:
        raise ValueError("ChaCha20 requires a 32-byte key and 12-byte nonce")

    def quarter_round(state: list[int], a: int, b: int, c: int, d: int) -> None:
        state[a] = (state[a] + state[b]) & 0xFFFFFFFF
        state[d] = _rotate_left(state[d] ^ state[a], 16)
        state[c] = (state[c] + state[d]) & 0xFFFFFFFF
        state[b] = _rotate_left(state[b] ^ state[c], 12)
        state[a] = (state[a] + state[b]) & 0xFFFFFFFF
        state[d] = _rotate_left(state[d] ^ state[a], 8)
        state[c] = (state[c] + state[d]) & 0xFFFFFFFF
        state[b] = _rotate_left(state[b] ^ state[c], 7)

    constants = list(struct.unpack("<4I", b"expand 32-byte k"))
    key_words = list(struct.unpack("<8I", key))
    nonce_words = list(struct.unpack("<3I", nonce))
    encrypted = bytearray()
    for block_index in range((len(payload) + 63) // 64):
        initial = constants + key_words + [block_index] + nonce_words
        state = initial.copy()
        for _ in range(10):
            quarter_round(state, 0, 4, 8, 12)
            quarter_round(state, 1, 5, 9, 13)
            quarter_round(state, 2, 6, 10, 14)
            quarter_round(state, 3, 7, 11, 15)
            quarter_round(state, 0, 5, 10, 15)
            quarter_round(state, 1, 6, 11, 12)
            quarter_round(state, 2, 7, 8, 13)
            quarter_round(state, 3, 4, 9, 14)
        key_stream = struct.pack(
            "<16I",
            *((word + original) & 0xFFFFFFFF for word, original in zip(state, initial)),
        )
        chunk = payload[block_index * 64 : (block_index + 1) * 64]
        encrypted.extend(left ^ right for left, right in zip(chunk, key_stream))
    return bytes(encrypted)


def _nip44_padded_length(length: int) -> int:
    if length <= 32:
        return 32
    next_power = 1 << (length - 1).bit_length()
    chunk = 32 if next_power <= 256 else next_power // 8
    return chunk * ((length - 1) // chunk + 1)


def nip44_encrypt(
    plaintext: str,
    *,
    private_key: str,
    recipient_pubkey: str,
    nonce: Optional[bytes] = None,
) -> str:
    """Encrypt UTF-8 text with NIP-44 v2 for an x-only secp256k1 recipient."""
    encoded = plaintext.encode("utf-8")
    if not 1 <= len(encoded) <= 65_535:
        raise ValueError("NIP-44 plaintext must contain 1 to 65535 bytes")
    nonce = secrets.token_bytes(32) if nonce is None else nonce
    if len(nonce) != 32:
        raise ValueError("NIP-44 nonce must be 32 bytes")

    shared_point = _point_multiply(decode_private_key(private_key), _lift_x(recipient_pubkey))
    if shared_point is None:  # pragma: no cover - validated nonzero keys make this unreachable
        raise ValueError("invalid NIP-44 shared point")
    shared_x = shared_point[0].to_bytes(32, "big")
    conversation_key = hmac.new(b"nip44-v2", shared_x, hashlib.sha256).digest()
    message_keys = _hkdf_expand(conversation_key, nonce, 76)
    chacha_key = message_keys[:32]
    chacha_nonce = message_keys[32:44]
    hmac_key = message_keys[44:]

    prefix = len(encoded).to_bytes(2, "big")
    padded = prefix + encoded + bytes(_nip44_padded_length(len(encoded)) - len(encoded))
    ciphertext = _chacha20_xor(chacha_key, chacha_nonce, padded)
    mac = hmac.new(hmac_key, nonce + ciphertext, hashlib.sha256).digest()
    return base64.b64encode(b"\x02" + nonce + ciphertext + mac).decode("ascii")


def build_observer_event(
    *,
    private_key: str,
    owner_pubkey: str,
    payload: dict[str, Any],
    created_at: Optional[int] = None,
    nonce: Optional[bytes] = None,
    auxiliary_randomness: Optional[bytes] = None,
) -> dict[str, Any]:
    """Build a signed, owner-encrypted NIP-AO telemetry event (kind 24200)."""
    plaintext = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    if len(plaintext.encode("utf-8")) > 65_535:
        raise ValueError("observer plaintext exceeds 65535 bytes")
    content = nip44_encrypt(
        plaintext,
        private_key=private_key,
        recipient_pubkey=owner_pubkey,
        nonce=nonce,
    )
    pubkey = public_key_hex(private_key)
    timestamp = int(time.time()) if created_at is None else int(created_at)
    tags = [
        ["p", owner_pubkey.lower()],
        ["agent", pubkey],
        ["frame", "telemetry"],
    ]
    serialized = json.dumps(
        [0, pubkey, timestamp, 24200, tags, content],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    event_id = hashlib.sha256(serialized).digest()
    return {
        "id": event_id.hex(),
        "pubkey": pubkey,
        "created_at": timestamp,
        "kind": 24200,
        "tags": tags,
        "content": content,
        "sig": schnorr_sign(
            event_id,
            private_key,
            auxiliary_randomness=auxiliary_randomness,
        ).hex(),
    }


def schnorr_sign(
    message: bytes,
    private_key: str,
    *,
    auxiliary_randomness: Optional[bytes] = None,
) -> bytes:
    if len(message) != 32:
        raise ValueError("BIP-340 signs a 32-byte message")
    secret = decode_private_key(private_key)
    public_point = _point_multiply(secret)
    if public_point is None:  # pragma: no cover
        raise ValueError("invalid private key")
    public_x = public_point[0].to_bytes(32, "big")
    adjusted_secret = secret if public_point[1] % 2 == 0 else CURVE_ORDER - secret

    aux = (
        auxiliary_randomness
        if auxiliary_randomness is not None
        else secrets.token_bytes(32)
    )
    if len(aux) != 32:
        raise ValueError("auxiliary randomness must be 32 bytes")
    masked = bytes(
        left ^ right
        for left, right in zip(
            adjusted_secret.to_bytes(32, "big"),
            _tagged_hash("BIP0340/aux", aux),
        )
    )
    nonce = (
        int.from_bytes(
            _tagged_hash("BIP0340/nonce", masked + public_x + message), "big"
        )
        % CURVE_ORDER
    )
    if nonce == 0:
        raise RuntimeError("BIP-340 produced a zero nonce")
    nonce_point = _point_multiply(nonce)
    if nonce_point is None:  # pragma: no cover
        raise RuntimeError("BIP-340 produced an invalid nonce point")
    adjusted_nonce = nonce if nonce_point[1] % 2 == 0 else CURVE_ORDER - nonce
    nonce_x = nonce_point[0].to_bytes(32, "big")
    challenge = (
        int.from_bytes(
            _tagged_hash("BIP0340/challenge", nonce_x + public_x + message), "big"
        )
        % CURVE_ORDER
    )
    signature_scalar = (adjusted_nonce + challenge * adjusted_secret) % CURVE_ORDER
    return nonce_x + signature_scalar.to_bytes(32, "big")


def build_auth_event(
    *,
    private_key: str,
    challenge: str,
    relay_url: str,
    auth_tag_json: str = "",
    created_at: Optional[int] = None,
    auxiliary_randomness: Optional[bytes] = None,
) -> dict[str, Any]:
    tags: list[list[str]] = [
        ["relay", relay_url],
        ["challenge", challenge],
    ]
    if auth_tag_json.strip():
        try:
            auth_tag = json.loads(auth_tag_json)
        except json.JSONDecodeError as exc:
            raise ValueError("BUZZ_AUTH_TAG is not valid JSON") from exc
        if (
            not isinstance(auth_tag, list)
            or len(auth_tag) != 4
            or auth_tag[0] != "auth"
            or not all(isinstance(part, str) for part in auth_tag)
        ):
            raise ValueError("BUZZ_AUTH_TAG must be a four-string auth tag")
        tags.append(auth_tag)

    pubkey = public_key_hex(private_key)
    timestamp = int(time.time()) if created_at is None else int(created_at)
    serialized = json.dumps(
        [0, pubkey, timestamp, 22242, tags, ""],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    event_id = hashlib.sha256(serialized).digest()
    return {
        "id": event_id.hex(),
        "pubkey": pubkey,
        "created_at": timestamp,
        "kind": 22242,
        "tags": tags,
        "content": "",
        "sig": schnorr_sign(
            event_id,
            private_key,
            auxiliary_randomness=auxiliary_randomness,
        ).hex(),
    }
