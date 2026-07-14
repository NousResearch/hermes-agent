"""Strict ``keychain://service/account`` references and the macOS Keychain
adapter used by the OAuth broker and credential-pool secret references.

Security policy (see docs/design/oauth-broker.md §十):

* Secret values travel only between this process and Security.framework via
  in-process C calls — never through external command argv, environment
  variables, ordinary files, stdout, or stderr.
* Exception text carries service/account identifiers plus a normalized
  OSStatus code and category, never secret bytes.
* On non-Darwin platforms every read/write fails closed with
  ``KeychainUnavailable``; there is no fallback secret source.
"""

from __future__ import annotations

import ctypes
import platform
import re
from dataclasses import dataclass
from typing import Optional, Protocol

KEYCHAIN_URI_SCHEME = "keychain://"

# OSStatus values from Security/SecBase.h.
ERR_SEC_SUCCESS = 0
ERR_SEC_DUPLICATE_ITEM = -25299
ERR_SEC_ITEM_NOT_FOUND = -25300

_SECURITY_FRAMEWORK = "/System/Library/Frameworks/Security.framework/Security"
_COREFOUNDATION_FRAMEWORK = (
    "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
)

# Keychain service/account segments: printable, no separators, no escapes.
_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class KeychainRef:
    service: str
    account: str


class KeychainError(RuntimeError):
    """Keychain failure carrying only identifiers, category, and OSStatus."""

    default_category = "keychain_error"

    def __init__(
        self,
        *,
        service: str = "",
        account: str = "",
        category: Optional[str] = None,
        os_status: Optional[int] = None,
        message: Optional[str] = None,
    ) -> None:
        self.service = service
        self.account = account
        self.category = category or type(self).default_category
        self.os_status = os_status
        if message is None:
            detail = (
                f"OSStatus {os_status}" if os_status is not None else "no OSStatus"
            )
            message = (
                f"keychain {self.category} for service={service!r} "
                f"account={account!r} ({detail})"
            )
        super().__init__(message)


class KeychainUnavailable(KeychainError):
    default_category = "unavailable"


class KeychainNotFound(KeychainError):
    default_category = "not_found"

    def __init__(self, *, service: str = "", account: str = "", **kwargs) -> None:
        kwargs.setdefault("os_status", ERR_SEC_ITEM_NOT_FOUND)
        super().__init__(service=service, account=account, **kwargs)


def parse_keychain_uri(uri: str) -> KeychainRef:
    """Parse a strict ``keychain://<service>/<account>`` reference.

    Both segments must match ``[A-Za-z0-9._-]+`` — this rejects empty
    segments, extra path segments, query strings, fragments, any percent
    escapes, whitespace, and control characters in one rule.
    """
    if not isinstance(uri, str):
        raise ValueError("keychain uri must be a string")
    if not uri.startswith(KEYCHAIN_URI_SCHEME):
        raise ValueError("keychain uri must start with 'keychain://'")
    rest = uri[len(KEYCHAIN_URI_SCHEME):]
    service, sep, account = rest.partition("/")
    if not sep:
        raise ValueError("keychain uri must be keychain://<service>/<account>")
    for segment in (service, account):
        if not segment or _SEGMENT_RE.fullmatch(segment) is None:
            raise ValueError(
                "keychain uri segments must be non-empty and match [A-Za-z0-9._-]"
            )
    return KeychainRef(service=service, account=account)


def _configure_security_prototypes(security, corefoundation) -> None:
    """Set explicit argtypes/restype on the real framework functions.

    OSStatus is a signed 32-bit int. Applied ONLY to libraries this module
    loads itself — injected test fakes (whose bound methods reject attribute
    assignment) are never touched.
    """
    os_status = ctypes.c_int32
    p_u32 = ctypes.POINTER(ctypes.c_uint32)
    p_void = ctypes.POINTER(ctypes.c_void_p)

    find = security.SecKeychainFindGenericPassword
    find.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_char_p,
        ctypes.c_uint32,
        ctypes.c_char_p,
        p_u32,
        p_void,
        p_void,
    ]
    find.restype = os_status
    add = security.SecKeychainAddGenericPassword
    add.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_char_p,
        ctypes.c_uint32,
        ctypes.c_char_p,
        ctypes.c_uint32,
        ctypes.c_char_p,
        p_void,
    ]
    add.restype = os_status
    modify = security.SecKeychainItemModifyAttributesAndData
    modify.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_char_p]
    modify.restype = os_status
    security.SecKeychainItemDelete.argtypes = [ctypes.c_void_p]
    security.SecKeychainItemDelete.restype = os_status
    security.SecKeychainItemFreeContent.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    security.SecKeychainItemFreeContent.restype = os_status
    corefoundation.CFRelease.argtypes = [ctypes.c_void_p]
    corefoundation.CFRelease.restype = None


class KeychainBackend(Protocol):
    def read(self, ref: KeychainRef) -> str: ...

    def write(self, ref: KeychainRef, secret: str) -> None: ...

    def delete(self, ref: KeychainRef) -> None: ...


class DarwinSecurityFrameworkBackend:
    """Generic-password operations through Security.framework via ctypes.

    ``security`` and ``corefoundation`` are injectable for tests. Production
    loads both system frameworks and applies explicit signed OSStatus/pointer
    prototypes; injected duck-typed fakes stay unmodified.
    """

    def __init__(self, *, security=None, corefoundation=None) -> None:
        if security is None or corefoundation is None:
            if platform.system() != "Darwin":
                raise KeychainUnavailable(
                    message=(
                        "macOS Keychain is only available on Darwin; "
                        "refusing to fall back to any other secret source"
                    ),
                )
            loaded_both = security is None and corefoundation is None
            security = security or ctypes.CDLL(_SECURITY_FRAMEWORK)
            corefoundation = corefoundation or ctypes.CDLL(
                _COREFOUNDATION_FRAMEWORK
            )
            if loaded_both:
                _configure_security_prototypes(security, corefoundation)
        self._security = security
        self._corefoundation = corefoundation

    def _find(self, ref: KeychainRef, *, want_data: bool, want_item: bool):
        service = ref.service.encode("utf-8")
        account = ref.account.encode("utf-8")
        length = ctypes.c_uint32(0)
        data = ctypes.c_void_p(None)
        item = ctypes.c_void_p(None)
        status = self._security.SecKeychainFindGenericPassword(
            None,
            len(service),
            service,
            len(account),
            account,
            ctypes.byref(length) if want_data else None,
            ctypes.byref(data) if want_data else None,
            ctypes.byref(item) if want_item else None,
        )
        if status == ERR_SEC_ITEM_NOT_FOUND:
            raise KeychainNotFound(service=ref.service, account=ref.account)
        if status != ERR_SEC_SUCCESS:
            raise KeychainError(
                service=ref.service,
                account=ref.account,
                category="os_error",
                os_status=int(status),
            )
        return length, data, item

    def read(self, ref: KeychainRef) -> str:
        length, data, _item = self._find(ref, want_data=True, want_item=False)
        try:
            if data.value and length.value:
                raw = ctypes.string_at(data.value, length.value)
            else:
                raw = b""
        finally:
            self._security.SecKeychainItemFreeContent(None, data)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            # ``from None``: a chained UnicodeDecodeError would embed secret
            # bytes in the traceback.
            raise KeychainError(
                service=ref.service,
                account=ref.account,
                category="invalid_encoding",
            ) from None

    def write(self, ref: KeychainRef, secret: str) -> None:
        payload = secret.encode("utf-8")
        service = ref.service.encode("utf-8")
        account = ref.account.encode("utf-8")
        status = self._security.SecKeychainAddGenericPassword(
            None,
            len(service),
            service,
            len(account),
            account,
            len(payload),
            payload,
            None,
        )
        if status == ERR_SEC_DUPLICATE_ITEM:
            _length, _data, item = self._find(ref, want_data=False, want_item=True)
            try:
                status = self._security.SecKeychainItemModifyAttributesAndData(
                    item, None, len(payload), payload
                )
                if status != ERR_SEC_SUCCESS:
                    raise KeychainError(
                        service=ref.service,
                        account=ref.account,
                        category="os_error",
                        os_status=int(status),
                    )
            finally:
                if item.value:
                    self._corefoundation.CFRelease(item)
            return
        if status != ERR_SEC_SUCCESS:
            raise KeychainError(
                service=ref.service,
                account=ref.account,
                category="os_error",
                os_status=int(status),
            )

    def delete(self, ref: KeychainRef) -> None:
        _length, _data, item = self._find(ref, want_data=False, want_item=True)
        try:
            status = self._security.SecKeychainItemDelete(item)
            if status != ERR_SEC_SUCCESS:
                raise KeychainError(
                    service=ref.service,
                    account=ref.account,
                    category="os_error",
                    os_status=int(status),
                )
        finally:
            if item.value:
                self._corefoundation.CFRelease(item)


def _default_backend() -> KeychainBackend:
    if platform.system() != "Darwin":
        raise KeychainUnavailable(
            message=(
                "macOS Keychain is only available on Darwin; "
                "refusing to fall back to any other secret source"
            ),
        )
    return DarwinSecurityFrameworkBackend()


def read_keychain_secret(
    ref: KeychainRef, *, backend: Optional[KeychainBackend] = None
) -> str:
    backend = backend if backend is not None else _default_backend()
    return backend.read(ref)


def write_keychain_secret(
    ref: KeychainRef, secret: str, *, backend: Optional[KeychainBackend] = None
) -> None:
    if not isinstance(secret, str) or not secret:
        raise ValueError("keychain secret must be a non-empty string")
    backend = backend if backend is not None else _default_backend()
    backend.write(ref, secret)


def delete_keychain_secret(
    ref: KeychainRef, *, backend: Optional[KeychainBackend] = None
) -> None:
    backend = backend if backend is not None else _default_backend()
    backend.delete(ref)


__all__ = [
    "ERR_SEC_DUPLICATE_ITEM",
    "ERR_SEC_ITEM_NOT_FOUND",
    "ERR_SEC_SUCCESS",
    "DarwinSecurityFrameworkBackend",
    "KeychainBackend",
    "KeychainError",
    "KeychainNotFound",
    "KeychainRef",
    "KeychainUnavailable",
    "delete_keychain_secret",
    "parse_keychain_uri",
    "read_keychain_secret",
    "write_keychain_secret",
]
