"""Tests for agent/keychain_secret.py — strict keychain:// refs and the
injectable macOS Keychain adapter.

Every test uses fake backends or fake Security.framework libraries with
synthetic values only. No test touches the real Keychain, and the adapter
under test must never shell out to /usr/bin/security.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import pytest

from agent.keychain_secret import (
    ERR_SEC_DUPLICATE_ITEM,
    ERR_SEC_ITEM_NOT_FOUND,
    DarwinSecurityFrameworkBackend,
    KeychainError,
    KeychainNotFound,
    KeychainRef,
    KeychainUnavailable,
    delete_keychain_secret,
    parse_keychain_uri,
    read_keychain_secret,
    write_keychain_secret,
)

SYNTHETIC_SECRET = "synthetic-local-client-key-000"


# ---------------------------------------------------------------------------
# parse_keychain_uri
# ---------------------------------------------------------------------------


def test_parse_canonical_client_uri():
    ref = parse_keychain_uri("keychain://ai.hermes.oauth-broker.client/local")
    assert ref == KeychainRef(
        service="ai.hermes.oauth-broker.client", account="local"
    )
    assert ref.service == "ai.hermes.oauth-broker.client"
    assert ref.account == "local"


def test_parse_grant_service_uri():
    ref = parse_keychain_uri("keychain://ai.hermes.oauth-broker.openai-codex/A")
    assert ref == KeychainRef(
        service="ai.hermes.oauth-broker.openai-codex", account="A"
    )


@pytest.mark.parametrize(
    "bad_uri",
    [
        "",
        "keychain://",
        "keychain:///local",  # empty service
        "keychain://svc/",  # empty account
        "keychain://svc",  # missing account segment
        "keychain://svc/acct/extra",  # extra path segment
        "keychain://svc/acct?query=1",  # query string
        "keychain://svc/acct#frag",  # fragment
        "keychain://svc/acc%2Ft",  # percent-encoding smuggling '/'
        "keychain://svc/ac%41ct",  # any percent-encoding rejected
        "secret://svc/acct",  # wrong scheme
        "http://svc/acct",  # wrong scheme
        "KEYCHAIN://svc/acct",  # scheme is case-sensitive
        "keychain://svc/ac\nct",  # control character
        "keychain://svc/acct\n",  # final newline must not match `$`
        "keychain://svc\n/acct",  # newline in service
        "keychain://sv\x00c/acct",  # NUL byte
        "keychain://svc/acct ",  # trailing whitespace
    ],
)
def test_parse_rejects_malformed_uris(bad_uri):
    with pytest.raises(ValueError):
        parse_keychain_uri(bad_uri)


def test_parse_rejects_non_string_input():
    with pytest.raises(ValueError):
        parse_keychain_uri(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Injectable backend plumbing (read/write/delete helpers)
# ---------------------------------------------------------------------------


class FakeBackend:
    """In-memory KeychainBackend recording (service, account) call tuples."""

    def __init__(self, items=None):
        self.items = dict(items or {})
        self.read_calls = []
        self.write_calls = []
        self.delete_calls = []

    def read(self, ref):
        self.read_calls.append((ref.service, ref.account))
        try:
            return self.items[(ref.service, ref.account)]
        except KeyError:
            raise KeychainNotFound(
                service=ref.service, account=ref.account
            ) from None

    def write(self, ref, secret):
        self.write_calls.append((ref.service, ref.account))
        self.items[(ref.service, ref.account)] = secret

    def delete(self, ref):
        self.delete_calls.append((ref.service, ref.account))
        if (ref.service, ref.account) not in self.items:
            raise KeychainNotFound(service=ref.service, account=ref.account)
        del self.items[(ref.service, ref.account)]


CLIENT_REF = KeychainRef(service="ai.hermes.oauth-broker.client", account="local")


def test_read_returns_synthetic_secret_from_fake_backend():
    backend = FakeBackend({("ai.hermes.oauth-broker.client", "local"): SYNTHETIC_SECRET})
    assert read_keychain_secret(CLIENT_REF, backend=backend) == SYNTHETIC_SECRET
    # Service and account travel as separate values, never concatenated.
    assert backend.read_calls == [("ai.hermes.oauth-broker.client", "local")]


def test_read_missing_item_raises_keychain_not_found():
    backend = FakeBackend()
    with pytest.raises(KeychainNotFound):
        read_keychain_secret(CLIENT_REF, backend=backend)


def test_write_then_read_round_trip():
    backend = FakeBackend()
    write_keychain_secret(CLIENT_REF, SYNTHETIC_SECRET, backend=backend)
    assert backend.write_calls == [("ai.hermes.oauth-broker.client", "local")]
    assert read_keychain_secret(CLIENT_REF, backend=backend) == SYNTHETIC_SECRET


def test_delete_removes_item():
    backend = FakeBackend({("ai.hermes.oauth-broker.client", "local"): SYNTHETIC_SECRET})
    delete_keychain_secret(CLIENT_REF, backend=backend)
    assert backend.delete_calls == [("ai.hermes.oauth-broker.client", "local")]
    with pytest.raises(KeychainNotFound):
        read_keychain_secret(CLIENT_REF, backend=backend)


def test_write_rejects_empty_secret():
    backend = FakeBackend()
    with pytest.raises(ValueError):
        write_keychain_secret(CLIENT_REF, "", backend=backend)
    assert backend.write_calls == []


def test_default_backend_fails_closed_off_darwin(monkeypatch):
    import agent.keychain_secret as mod

    monkeypatch.setattr(mod.platform, "system", lambda: "Linux")
    with pytest.raises(KeychainUnavailable):
        read_keychain_secret(CLIENT_REF)


def test_keychain_error_text_never_contains_secret():
    err = KeychainError(
        service="svc", account="acct", category="os_error", os_status=-25293
    )
    text = str(err)
    assert "svc" in text and "acct" in text
    assert "-25293" in text
    assert SYNTHETIC_SECRET not in text


# ---------------------------------------------------------------------------
# Darwin Security.framework backend against a fake ctypes library
# ---------------------------------------------------------------------------


class FakeSecurityLib:
    """Duck-typed stand-in for the Security.framework CDLL.

    Stores synthetic passwords in a dict and honours the C calling
    convention the backend uses (byref out-params, string_at reads).
    """

    def __init__(self, items=None, *, find_status=None):
        self.items = dict(items or {})
        self.find_status = find_status  # forced OSStatus for find, or None
        self.freed = []
        self.deleted = []
        self.modified = []
        self.added = []
        self._buffers = []  # keep C buffers alive until FreeContent

    def _key(self, service_len, service, account_len, account):
        return (
            ctypes.string_at(service, service_len).decode(),
            ctypes.string_at(account, account_len).decode(),
        )

    def SecKeychainFindGenericPassword(
        self, keychain, service_len, service, account_len, account,
        length_ref, data_ref, item_ref,
    ):
        if self.find_status is not None:
            return self.find_status
        key = self._key(service_len, service, account_len, account)
        if key not in self.items:
            return ERR_SEC_ITEM_NOT_FOUND
        if length_ref is not None and data_ref is not None:
            raw = self.items[key].encode()
            buf = ctypes.create_string_buffer(raw, len(raw))
            self._buffers.append(buf)
            length_ref._obj.value = len(raw)
            data_ref._obj.value = ctypes.addressof(buf)
        if item_ref is not None:
            item_ref._obj.value = id(key) & 0xFFFFFFFF or 1
        return 0

    def SecKeychainAddGenericPassword(
        self, keychain, service_len, service, account_len, account,
        secret_len, secret, item_ref,
    ):
        key = self._key(service_len, service, account_len, account)
        if key in self.items:
            return ERR_SEC_DUPLICATE_ITEM
        value = ctypes.string_at(secret, secret_len).decode()
        self.items[key] = value
        self.added.append(key)
        return 0

    def SecKeychainItemModifyAttributesAndData(
        self, item, attrs, secret_len, secret
    ):
        value = ctypes.string_at(secret, secret_len).decode()
        # The fake stores one pending modify target: the last found key.
        self.modified.append(value)
        for key in list(self.items):
            self.items[key] = value
        return 0

    def SecKeychainItemDelete(self, item):
        self.deleted.append(True)
        self.items.clear()
        return 0

    def SecKeychainItemFreeContent(self, attrs, data):
        self.freed.append(data)
        return 0


class FakeCoreFoundationLib:
    def __init__(self):
        self.released = []

    def CFRelease(self, ref):
        self.released.append(ref)


def _darwin_backend(items=None, *, find_status=None):
    sec = FakeSecurityLib(items, find_status=find_status)
    cf = FakeCoreFoundationLib()
    backend = DarwinSecurityFrameworkBackend(security=sec, corefoundation=cf)
    return backend, sec, cf


def test_darwin_read_decodes_password_and_frees_buffer():
    backend, sec, _cf = _darwin_backend(
        {("ai.hermes.oauth-broker.openai-codex", "A"): SYNTHETIC_SECRET}
    )
    ref = KeychainRef(service="ai.hermes.oauth-broker.openai-codex", account="A")
    assert backend.read(ref) == SYNTHETIC_SECRET
    assert len(sec.freed) == 1  # SecKeychainItemFreeContent always called


def test_darwin_read_maps_not_found_status():
    backend, _sec, _cf = _darwin_backend()
    with pytest.raises(KeychainNotFound):
        backend.read(KeychainRef(service="svc", account="A"))


def test_darwin_read_normalizes_other_os_status():
    backend, _sec, _cf = _darwin_backend(find_status=-25293)
    with pytest.raises(KeychainError) as excinfo:
        backend.read(KeychainRef(service="svc", account="A"))
    assert excinfo.value.os_status == -25293
    assert SYNTHETIC_SECRET not in str(excinfo.value)


def test_darwin_write_adds_new_item():
    backend, sec, _cf = _darwin_backend()
    backend.write(KeychainRef(service="svc", account="A"), SYNTHETIC_SECRET)
    assert sec.items[("svc", "A")] == SYNTHETIC_SECRET
    assert sec.added == [("svc", "A")]


def test_darwin_write_duplicate_modifies_in_place_and_releases_item():
    backend, sec, cf = _darwin_backend({("svc", "A"): "synthetic-old-value"})
    backend.write(KeychainRef(service="svc", account="A"), SYNTHETIC_SECRET)
    assert sec.items[("svc", "A")] == SYNTHETIC_SECRET
    assert sec.modified == [SYNTHETIC_SECRET]
    assert len(cf.released) == 1  # item ref from the find is released


def test_darwin_delete_deletes_item_and_releases_ref():
    backend, sec, cf = _darwin_backend({("svc", "A"): SYNTHETIC_SECRET})
    backend.delete(KeychainRef(service="svc", account="A"))
    assert sec.items == {}
    assert sec.deleted == [True]
    assert len(cf.released) == 1


def test_darwin_delete_missing_raises_not_found():
    backend, _sec, _cf = _darwin_backend()
    with pytest.raises(KeychainNotFound):
        backend.delete(KeychainRef(service="svc", account="A"))


# ---------------------------------------------------------------------------
# Explicit ctypes prototypes for the real frameworks
# ---------------------------------------------------------------------------


class _ProtoFn:
    """Attribute bag standing in for a ctypes function pointer."""


class _ProtoLib:
    def __init__(self, names):
        for name in names:
            setattr(self, name, _ProtoFn())


def test_prototype_configuration_sets_argtypes_and_restype():
    from agent.keychain_secret import _configure_security_prototypes

    security = _ProtoLib(
        [
            "SecKeychainFindGenericPassword",
            "SecKeychainAddGenericPassword",
            "SecKeychainItemModifyAttributesAndData",
            "SecKeychainItemDelete",
            "SecKeychainItemFreeContent",
        ]
    )
    corefoundation = _ProtoLib(["CFRelease"])
    _configure_security_prototypes(security, corefoundation)

    find = security.SecKeychainFindGenericPassword
    assert find.restype is ctypes.c_int32  # OSStatus
    assert find.argtypes == [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_char_p,
        ctypes.c_uint32,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    add = security.SecKeychainAddGenericPassword
    assert add.restype is ctypes.c_int32
    assert add.argtypes == [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_char_p,
        ctypes.c_uint32,
        ctypes.c_char_p,
        ctypes.c_uint32,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    modify = security.SecKeychainItemModifyAttributesAndData
    assert modify.restype is ctypes.c_int32
    assert modify.argtypes == [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_char_p,
    ]
    assert security.SecKeychainItemDelete.argtypes == [ctypes.c_void_p]
    assert security.SecKeychainItemDelete.restype is ctypes.c_int32
    assert security.SecKeychainItemFreeContent.argtypes == [
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    assert security.SecKeychainItemFreeContent.restype is ctypes.c_int32
    assert corefoundation.CFRelease.argtypes == [ctypes.c_void_p]
    assert corefoundation.CFRelease.restype is None


def test_injected_fake_backends_are_never_prototype_configured():
    """Injected duck-typed fakes (bound methods reject attribute assignment)
    must keep working — prototypes apply only to libraries the backend
    loads itself."""
    backend, sec, _cf = _darwin_backend(
        {("svc", "A"): SYNTHETIC_SECRET}
    )
    assert backend.read(KeychainRef(service="svc", account="A")) == SYNTHETIC_SECRET
    assert not hasattr(sec.SecKeychainFindGenericPassword, "argtypes")


# ---------------------------------------------------------------------------
# Source-level safety: never subprocess, never /usr/bin/security
# ---------------------------------------------------------------------------


def test_adapter_source_never_uses_subprocess_or_security_cli():
    import agent.keychain_secret as mod

    source = Path(mod.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "subprocess",
        "Popen",
        "os.system",
        "/usr/bin/security",
        "find-generic-password",
        "add-generic-password",
        "delete-generic-password",
    ):
        assert forbidden not in source, f"adapter must not use {forbidden!r}"
