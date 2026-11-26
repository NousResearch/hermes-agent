"""python-olm compatible API backed by vodozemac, for mautrix's E2EE stack.

mautrix 0.21 (``mautrix.crypto``) is written against python-olm, the CFFI binding to
libolm. libolm was deprecated by the Matrix.org Foundation in July 2024 in favour of
vodozemac (Rust, audited), is no longer maintained, and python-olm ships no wheels for
macOS, Windows or CPython >= 3.13 -- which is what forces Hermes' ``matrix`` extra to be
Linux-only and to build C/C++ from source on many machines.

This module implements the subset of the python-olm API that mautrix 0.21 and the Hermes
Matrix adapter use, on top of the ``vodozemac`` Python bindings (pre-built wheels for
Linux glibc/musl, macOS x86_64/arm64 and Windows x64/arm64, CPython 3.10-3.14).
:func:`install` registers it as the ``olm`` module (plus an empty ``_libolm`` stub that
``mautrix.crypto.sessions`` imports) *before* ``mautrix.crypto`` is imported, so mautrix
runs unchanged.

Compatibility notes
-------------------
* Wire format: identical (Olm v1 / Megolm v1). Peers cannot tell the difference.
* Pickles: new pickles are vodozemac pickles encrypted with ``sha256(passphrase)``
  (same scheme matrix-nio >= 0.26 uses). Existing libolm pickles (Account, Olm
  ``Session``, ``InboundGroupSession``) are read transparently via vodozemac's
  ``from_libolm_pickle`` and re-written in vodozemac format the next time they are
  saved. libolm *outbound* group-session pickles cannot be imported by vodozemac;
  :class:`OutboundGroupSession.from_pickle` raises :class:`OlmGroupSessionError` for them
  and callers must discard them (a new outbound session is simply created and shared).
  The migration is one-way: a store written by this backend cannot be read by libolm.
* ``InboundSession``: vodozemac decrypts the pre-key message while creating the session,
  whereas libolm creates the session and decrypts in a second call. The plaintext of that
  first message is kept in a small one-shot cache keyed by (session id, ciphertext) and
  handed back on the first ``decrypt()`` of the identical ciphertext.
* ``Account.remove_one_time_keys`` is a no-op: vodozemac removes the used one-time key
  when the inbound session is created.
* ``PkSigning`` (cross-signing keys) uses Ed25519 from pycryptodome, which mautrix already
  requires; keys and signatures are byte-identical to libolm's for the same seed.
* Base64 input is accepted padded or unpadded (the spec allows both); libolm rejected
  padded Megolm ciphertext (see hermes-agent#138108).
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import importlib.machinery
import os
import sys
import threading
import time
import types
from collections import OrderedDict
from typing import Any, Optional

import vodozemac as _vz

__all__ = [
    "Account",
    "InboundGroupSession",
    "InboundSession",
    "OlmAccountError",
    "OlmGroupSessionError",
    "OlmHashError",
    "OlmMessage",
    "OlmPreKeyMessage",
    "OlmSessionError",
    "OlmVerifyError",
    "OutboundGroupSession",
    "OutboundSession",
    "PkSigning",
    "PkSigningError",
    "Session",
    "ed25519_verify",
    "install",
    "sha256",
]

BACKEND_NAME = "vodozemac"
VODOZEMAC_VERSION = getattr(_vz, "__version__", "unknown")

_BytesLike = str | bytes | bytearray | memoryview


# --------------------------------------------------------------------------- errors


class OlmAccountError(Exception):
    """Raised by :class:`Account` operations (python-olm compatible)."""


class OlmSessionError(Exception):
    """Raised by Olm :class:`Session` operations (python-olm compatible)."""


class OlmGroupSessionError(Exception):
    """Raised by Megolm group session operations (python-olm compatible)."""


class PkSigningError(Exception):
    """Raised by :class:`PkSigning` (python-olm compatible)."""


class OlmVerifyError(Exception):
    """Raised by :func:`ed25519_verify` (python-olm compatible)."""


class OlmHashError(Exception):
    """Kept for API compatibility with python-olm; never raised."""


_VZ_ERRORS: tuple[type, ...] = tuple(
    getattr(_vz, name)
    for name in (
        "KeyException",
        "SignatureException",
        "DecodeException",
        "LibolmPickleException",
        "SessionKeyDecodeException",
        "PickleException",
        "SessionCreationException",
        "OlmEncryptionException",
        "OlmDecryptionException",
        "MegolmDecryptionException",
    )
    if hasattr(_vz, name)
) + (ValueError,)


# --------------------------------------------------------------------------- helpers


def _to_bytes(value: _BytesLike) -> bytes:
    if isinstance(value, str):
        return value.encode("utf-8")
    return bytes(value)


def _to_text(value: _BytesLike) -> str:
    if isinstance(value, str):
        return value
    return bytes(value).decode("utf-8")


def _b64decode(value: _BytesLike) -> bytes:
    """Decode standard base64, padded or unpadded (the Matrix spec permits both)."""
    text = _to_text(value).strip().rstrip("=")
    try:
        return base64.b64decode(text + "=" * (-len(text) % 4), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("INVALID_BASE64") from exc


def _b64encode(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii").rstrip("=")


def _pickle_key(passphrase: _BytesLike) -> bytes:
    # vodozemac wants a 32-byte key; it applies its own HKDF + MAC when (un)pickling.
    # Same derivation as matrix-nio >= 0.26.
    return hashlib.sha256(_to_bytes(passphrase or "")).digest()


def _libolm_key(passphrase: _BytesLike) -> bytes:
    # python-olm used the raw UTF-8 passphrase as the libolm pickle key.
    return _to_bytes(passphrase or "")


def _unpickle(vz_cls: Any, pickle: _BytesLike, passphrase: _BytesLike, error: type) -> Any:
    """Load a vodozemac pickle, falling back to a legacy libolm pickle."""
    if not pickle:
        raise ValueError("Pickle can't be empty")
    text = _to_text(pickle)
    try:
        return vz_cls.from_pickle(text, _pickle_key(passphrase))
    except _VZ_ERRORS as vz_exc:
        legacy = getattr(vz_cls, "from_libolm_pickle", None)
        if legacy is None:
            raise error(f"BAD_PICKLE: not a vodozemac pickle ({vz_exc})") from vz_exc
        try:
            return legacy(text, _libolm_key(passphrase))
        except _VZ_ERRORS as olm_exc:
            # python-olm reported a wrong passphrase as BAD_ACCOUNT_KEY / BAD_SESSION_KEY.
            raise error(f"BAD_PICKLE_KEY: {olm_exc}") from olm_exc


def _curve25519(key: _BytesLike) -> Any:
    try:
        return _vz.Curve25519PublicKey.from_base64(_to_text(key).rstrip("="))
    except _VZ_ERRORS as exc:
        raise OlmSessionError(f"INVALID_BASE64: bad Curve25519 key ({exc})") from exc


# ------------------------------------------------- one-shot plaintext of a pre-key message

# Decrypted to-device payloads (typically room keys) wait here only until mautrix's immediate
# decrypt() of the same pre-key message, so entries that are never taken (that flow failed
# between session creation and decrypt) expire instead of staying in memory for the process
# lifetime.
_FIRST_PLAINTEXT_MAX = 1024
_FIRST_PLAINTEXT_TTL = 60.0  # seconds
_first_plaintext: OrderedDict[tuple[str, bytes], tuple[float, bytes]] = OrderedDict()
_first_plaintext_lock = threading.Lock()


def _expire_first_plaintext(now: float) -> None:
    while _first_plaintext:
        stored_at, _ = next(iter(_first_plaintext.values()))
        if now - stored_at < _FIRST_PLAINTEXT_TTL and len(_first_plaintext) <= _FIRST_PLAINTEXT_MAX:
            return
        _first_plaintext.popitem(last=False)  # insertion order == age order


def _remember_first_plaintext(session_id: str, ciphertext: bytes, plaintext: bytes) -> None:
    with _first_plaintext_lock:
        now = time.monotonic()
        _first_plaintext[(session_id, ciphertext)] = (now, plaintext)
        _first_plaintext.move_to_end((session_id, ciphertext))  # keep insertion order == age order
        _expire_first_plaintext(now)


def _take_first_plaintext(session_id: str, ciphertext: bytes) -> Optional[bytes]:
    with _first_plaintext_lock:
        _expire_first_plaintext(time.monotonic())
        entry = _first_plaintext.pop((session_id, ciphertext), None)
        return None if entry is None else entry[1]


# --------------------------------------------------------------------------- utility


def sha256(input_string: _BytesLike) -> str:
    """Unpadded base64 SHA-256, like ``olm.sha256``."""
    return _b64encode(hashlib.sha256(_to_bytes(input_string)).digest())


def ed25519_verify(key: _BytesLike, message: _BytesLike, signature: _BytesLike) -> None:
    """Verify an Ed25519 signature; raises :class:`OlmVerifyError` like ``olm.ed25519_verify``."""
    try:
        public_key = _vz.Ed25519PublicKey.from_base64(_to_text(key).rstrip("="))
        sig = _vz.Ed25519Signature.from_base64(_to_text(signature).rstrip("="))
        public_key.verify_signature(_to_bytes(message), sig)
    except _VZ_ERRORS as exc:
        raise OlmVerifyError(f"BAD_MESSAGE_MAC: {exc}") from exc


# --------------------------------------------------------------------------- Account


class Account:
    """Olm account (identity keys, one-time keys, fallback key)."""

    _vz: Any

    def __new__(cls, *args: Any, **kwargs: Any) -> Account:
        obj = super().__new__(cls)
        obj._vz = None
        return obj

    def __init__(self) -> None:
        self._vz = _vz.Account()

    def pickle(self, passphrase: _BytesLike = "") -> bytes:
        return self._vz.pickle(_pickle_key(passphrase)).encode("utf-8")

    @classmethod
    def from_pickle(cls, pickle: _BytesLike, passphrase: _BytesLike = "") -> Account:
        obj = cls.__new__(cls)
        obj._vz = _unpickle(_vz.Account, pickle, passphrase, OlmAccountError)
        return obj

    @property
    def identity_keys(self) -> dict:
        return {
            "curve25519": self._vz.curve25519_key.to_base64(),
            "ed25519": self._vz.ed25519_key.to_base64(),
        }

    def sign(self, message: _BytesLike) -> str:
        return self._vz.sign(_to_bytes(message)).to_base64()

    @property
    def max_one_time_keys(self) -> int:
        return self._vz.max_number_of_one_time_keys

    def mark_keys_as_published(self) -> None:
        self._vz.mark_keys_as_published()

    def generate_one_time_keys(self, count: int) -> None:
        self._vz.generate_one_time_keys(count)

    @property
    def one_time_keys(self) -> dict:
        return {"curve25519": {kid: key.to_base64() for kid, key in self._vz.one_time_keys.items()}}

    def remove_one_time_keys(self, session: Session) -> None:
        """No-op: vodozemac removes the used one-time key when the inbound session is created."""

    def generate_fallback_key(self) -> None:
        self._vz.generate_fallback_key()

    @property
    def fallback_key(self) -> dict:
        keys = self._vz.fallback_key or {}
        return {"curve25519": {kid: key.to_base64() for kid, key in keys.items()}}

    def forget_old_fallback_key(self) -> None:
        """No-op: vodozemac manages fallback-key rotation itself."""


# --------------------------------------------------------------------------- Olm messages


class _OlmMessage:
    message_type: int

    def __init__(self, ciphertext: _BytesLike, message_type: int) -> None:
        if not ciphertext:
            raise ValueError("Ciphertext can't be empty")
        self.ciphertext = _to_text(ciphertext)
        self.message_type = message_type

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.ciphertext})"


class OlmPreKeyMessage(_OlmMessage):
    MESSAGE_TYPE = 0

    def __init__(self, ciphertext: _BytesLike) -> None:
        super().__init__(ciphertext, self.MESSAGE_TYPE)

    def __repr__(self) -> str:
        return f"OlmPreKeyMessage({self.ciphertext})"


class OlmMessage(_OlmMessage):
    MESSAGE_TYPE = 1

    def __init__(self, ciphertext: _BytesLike) -> None:
        super().__init__(ciphertext, self.MESSAGE_TYPE)

    def __repr__(self) -> str:
        return f"OlmMessage({self.ciphertext})"


# --------------------------------------------------------------------------- Olm sessions


class Session:
    """Olm 1:1 session. Like python-olm, only created via InboundSession,
    OutboundSession or :meth:`from_pickle`."""

    _vz: Any

    def __new__(cls, *args: Any, **kwargs: Any) -> Session:
        obj = super().__new__(cls)
        obj._vz = None
        return obj

    def __init__(self) -> None:
        if type(self) is Session:
            raise TypeError("Session class may not be instantiated.")

    def pickle(self, passphrase: _BytesLike = "") -> bytes:
        return self._vz.pickle(_pickle_key(passphrase)).encode("utf-8")

    @classmethod
    def from_pickle(cls, pickle: _BytesLike, passphrase: _BytesLike = "") -> Session:
        obj = cls.__new__(cls)
        obj._vz = _unpickle(_vz.Session, pickle, passphrase, OlmSessionError)
        return obj

    @property
    def id(self) -> str:
        return self._vz.session_id

    def encrypt(self, plaintext: _BytesLike) -> _OlmMessage:
        try:
            message_type, raw = self._vz.encrypt(_to_bytes(plaintext)).to_parts()
        except _VZ_ERRORS as exc:
            raise OlmSessionError(str(exc)) from exc
        cls = OlmPreKeyMessage if message_type == OlmPreKeyMessage.MESSAGE_TYPE else OlmMessage
        return cls(_b64encode(bytes(raw)))

    def decrypt(self, message: _OlmMessage, unicode_errors: str = "replace") -> str:
        if not message.ciphertext:
            raise ValueError("Ciphertext can't be empty")
        try:
            raw = _b64decode(message.ciphertext)
        except ValueError as exc:
            raise OlmSessionError("INVALID_BASE64") from exc
        if message.message_type == OlmPreKeyMessage.MESSAGE_TYPE:
            first = _take_first_plaintext(self.id, raw)
            if first is not None:
                return first.decode("utf-8", errors=unicode_errors)
            vz_message = _vz.AnyOlmMessage.pre_key(raw)
        else:
            vz_message = _vz.AnyOlmMessage.normal(raw)
        try:
            plaintext = self._vz.decrypt(vz_message)
        except _VZ_ERRORS as exc:
            raise OlmSessionError(str(exc)) from exc
        return bytes(plaintext).decode("utf-8", errors=unicode_errors)

    def matches(self, message: _OlmMessage, identity_key: Optional[_BytesLike] = None) -> bool:
        if not isinstance(message, OlmPreKeyMessage):
            raise TypeError("Matches can only be called with prekey messages.")
        try:
            pre_key = _vz.AnyOlmMessage.pre_key(_b64decode(message.ciphertext)).to_pre_key()
        except _VZ_ERRORS as exc:
            raise OlmSessionError(str(exc)) from exc
        return pre_key is not None and bool(self._vz.session_matches(pre_key))

    def describe(self, buffer_length: int = 600) -> str:
        return f"vodozemac olm session {self.id}"


class InboundSession(Session):
    """Olm session created from a received pre-key message."""

    def __new__(cls, account: Account, message: OlmPreKeyMessage,
                identity_key: Optional[_BytesLike] = None) -> InboundSession:
        return super().__new__(cls)

    def __init__(self, account: Account, message: OlmPreKeyMessage,
                 identity_key: Optional[_BytesLike] = None) -> None:
        if not message.ciphertext:
            raise ValueError("Ciphertext can't be empty")
        super().__init__()
        if not identity_key:
            # libolm could create a session without checking the sender key; vodozemac needs
            # the sender's Curve25519 key. mautrix always passes it.
            raise OlmSessionError("identity_key (sender Curve25519 key) is required")
        try:
            raw = _b64decode(message.ciphertext)
            pre_key = _vz.AnyOlmMessage.pre_key(raw).to_pre_key()
            if pre_key is None:
                raise OlmSessionError("BAD_MESSAGE_FORMAT: not a pre-key message")
            session, plaintext = account._vz.create_inbound_session(_curve25519(identity_key), pre_key)
        except _VZ_ERRORS as exc:
            raise OlmSessionError(str(exc)) from exc
        self._vz = session
        _remember_first_plaintext(session.session_id, raw, bytes(plaintext))


class OutboundSession(Session):
    """Olm session we initiate towards another device's identity + one-time key."""

    def __new__(cls, account: Account, identity_key: _BytesLike,
                one_time_key: _BytesLike) -> OutboundSession:
        return super().__new__(cls)

    def __init__(self, account: Account, identity_key: _BytesLike, one_time_key: _BytesLike) -> None:
        if not identity_key:
            raise ValueError("Identity key can't be empty")
        if not one_time_key:
            raise ValueError("One-time key can't be empty")
        super().__init__()
        try:
            self._vz = account._vz.create_outbound_session(
                _curve25519(identity_key), _curve25519(one_time_key))
        except _VZ_ERRORS as exc:
            raise OlmSessionError(str(exc)) from exc


# --------------------------------------------------------------------------- Megolm


class InboundGroupSession:
    """Megolm session used to decrypt a sender's room messages."""

    _vz: Any

    def __new__(cls, *args: Any, **kwargs: Any) -> InboundGroupSession:
        obj = super().__new__(cls)
        obj._vz = None
        return obj

    def __init__(self, session_key: _BytesLike) -> None:
        if not session_key:
            raise ValueError("Session key can't be empty")
        try:
            self._vz = _vz.InboundGroupSession(_vz.SessionKey(_to_text(session_key)))
        except _VZ_ERRORS as exc:
            raise OlmGroupSessionError(str(exc)) from exc

    def pickle(self, passphrase: _BytesLike = "") -> bytes:
        return self._vz.pickle(_pickle_key(passphrase)).encode("utf-8")

    @classmethod
    def from_pickle(cls, pickle: _BytesLike, passphrase: _BytesLike = "") -> InboundGroupSession:
        obj = cls.__new__(cls)
        obj._vz = _unpickle(_vz.InboundGroupSession, pickle, passphrase, OlmGroupSessionError)
        return obj

    def decrypt(self, ciphertext: _BytesLike, unicode_errors: str = "replace") -> tuple[str, int]:
        if not ciphertext:
            raise ValueError("Ciphertext can't be empty.")
        try:
            message = _vz.MegolmMessage.from_bytes(_b64decode(ciphertext))
            decrypted = self._vz.decrypt(message)
        except _VZ_ERRORS as exc:
            raise OlmGroupSessionError(str(exc)) from exc
        return bytes(decrypted.plaintext).decode("utf-8", errors=unicode_errors), decrypted.message_index

    @property
    def id(self) -> str:
        return self._vz.session_id

    @property
    def first_known_index(self) -> int:
        return self._vz.first_known_index

    def export_session(self, message_index: int) -> str:
        exported = self._vz.export_at(message_index)
        if exported is None:
            raise OlmGroupSessionError("OLM_UNKNOWN_MESSAGE_INDEX")
        return exported.to_base64()

    @classmethod
    def import_session(cls, session_key: _BytesLike) -> InboundGroupSession:
        if not session_key:
            raise ValueError("Session key can't be empty")
        obj = cls.__new__(cls)
        try:
            obj._vz = _vz.InboundGroupSession.import_session(_vz.ExportedSessionKey(_to_text(session_key)))
        except _VZ_ERRORS as exc:
            raise OlmGroupSessionError(str(exc)) from exc
        return obj


class OutboundGroupSession:
    """Megolm session used to encrypt our own room messages."""

    _vz: Any

    def __new__(cls, *args: Any, **kwargs: Any) -> OutboundGroupSession:
        obj = super().__new__(cls)
        obj._vz = None
        return obj

    def __init__(self) -> None:
        self._vz = _vz.GroupSession()

    def pickle(self, passphrase: _BytesLike = "") -> bytes:
        return self._vz.pickle(_pickle_key(passphrase)).encode("utf-8")

    @classmethod
    def from_pickle(cls, pickle: _BytesLike, passphrase: _BytesLike = "") -> OutboundGroupSession:
        # vodozemac cannot import libolm outbound group-session pickles; callers discard those
        # (a fresh outbound session is created and shared on the next send).
        obj = cls.__new__(cls)
        obj._vz = _unpickle(_vz.GroupSession, pickle, passphrase, OlmGroupSessionError)
        return obj

    def encrypt(self, plaintext: _BytesLike) -> str:
        try:
            return self._vz.encrypt(_to_bytes(plaintext)).to_base64()
        except _VZ_ERRORS as exc:
            raise OlmGroupSessionError(str(exc)) from exc

    @property
    def id(self) -> str:
        return self._vz.session_id

    @property
    def message_index(self) -> int:
        return self._vz.message_index

    @property
    def session_key(self) -> str:
        return self._vz.session_key.to_base64()


# --------------------------------------------------------------------------- PkSigning


def _ed25519_backend() -> tuple[Any, Any]:
    try:
        from Crypto.PublicKey import ECC
        from Crypto.Signature import eddsa
    except ImportError:  # pycryptodomex
        from Cryptodome.PublicKey import ECC  # type: ignore[no-redef]
        from Cryptodome.Signature import eddsa  # type: ignore[no-redef]
    return ECC, eddsa


class PkSigning:
    """Ed25519 signing key derived from a 32-byte seed (cross-signing keys).

    Byte-identical to libolm's ``olm_pk_signing_key_from_seed`` / ``olm_pk_sign``.
    """

    SEED_LENGTH = 32

    def __init__(self, seed: bytes) -> None:
        if not seed:
            raise ValueError("seed can't be empty")
        seed = bytes(seed)
        if len(seed) != self.SEED_LENGTH:
            raise PkSigningError("OLM_INPUT_BUFFER_TOO_SMALL")
        ecc, eddsa = _ed25519_backend()
        self._key = ecc.construct(curve="Ed25519", seed=seed)
        self._signer = eddsa.new(self._key, "rfc8032")
        self.public_key = _b64encode(self._key.public_key().export_key(format="raw"))

    @classmethod
    def generate_seed(cls) -> bytes:
        return os.urandom(cls.SEED_LENGTH)

    def sign(self, message: _BytesLike) -> str:
        return _b64encode(self._signer.sign(_to_bytes(message)))


# --------------------------------------------------------------------------- install


def install() -> types.ModuleType:
    """Register this module as ``olm`` (and an empty ``_libolm`` stub) in ``sys.modules``.

    Must run before ``mautrix.crypto`` is first imported. Idempotent. Returns this module.
    Raises RuntimeError if a different ``olm`` (python-olm) was already imported, because
    mautrix would then already be bound to it.
    """
    this = sys.modules[__name__]
    current = sys.modules.get("olm")
    if current is not None and current is not this:
        raise RuntimeError(
            f"python-olm ({getattr(current, '__file__', current)!r}) is already imported; "
            "the vodozemac backend must be installed before mautrix.crypto is imported")
    sys.modules["olm"] = this
    if "_libolm" not in sys.modules:
        stub = types.ModuleType("_libolm")
        stub.__doc__ = "Stub: vodozemac backend in use, libolm is not loaded."
        stub.__spec__ = importlib.machinery.ModuleSpec("_libolm", None)
        stub.ffi = None  # type: ignore[attr-defined]
        stub.lib = types.SimpleNamespace()  # type: ignore[attr-defined]
        sys.modules["_libolm"] = stub
    return this


def installed() -> bool:
    return sys.modules.get("olm") is sys.modules[__name__]
