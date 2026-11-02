"""HOTP consumption must persist each counter before returning its code."""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import pytest

from agent.vault_backends.local import LocalLoginBackend
from agent.vault_store import VaultError, VaultStore

# Public RFC 4226 seed and all ten reference outputs, not real credentials.
SEED = "GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ"
CODES = ["755224", "287082", "359152", "969429", "338314",
         "254676", "287922", "162583", "399871", "520489"]


def _login(store, otp):
    return store.add_item("login", "Example", {
        "identifier_type": "username", "identifier": "user", "password": "dummy",
        "otp_secret": otp,
    }, origin="https://example.com")


def test_backend_hotp_vectors_survive_reopening(tmp_path, monkeypatch):
    """Exercise the same add/resolve path as Desktop/CLI and the browser backend."""
    base = tmp_path / "vault"
    meta = _login(VaultStore(base), f"otpauth://hotp/Example?secret={SEED}&counter=0")
    monkeypatch.setattr("agent.vault_store.get_vault_store", lambda: VaultStore(base))
    backend = LocalLoginBackend()
    assert [backend.resolve_otp(meta.id) for _ in CODES] == CODES
    assert backend.get_meta(meta.id) == meta
    assert meta.has_otp
    assert "otp_secret" not in meta.to_dict() and SEED not in str(meta.to_dict())
    assert SEED.encode() not in (base / "vault.json.enc").read_bytes()
    assert backend.resolve_password(meta.id) == "dummy"

    # Non-default digits/algorithms use RFC 6238's HMAC(counter=1) reference values.
    import base64

    for raw_seed, options, expected in [
        (b"12345678901234567890", "digits=7", "4287082"),
        (b"12345678901234567890123456789012", "digits=8&algorithm=SHA256", "46119246"),
        (b"1234567890123456789012345678901234567890123456789012345678901234",
         "digits=8&algorithm=SHA512", "90693936"),
    ]:
        seed = base64.b32encode(raw_seed).decode().rstrip("=")
        item = _login(VaultStore(base), f"otpauth://hotp/Example?secret={seed}&counter=1&{options}")
        assert backend.resolve_otp(item.id) == expected


def test_hotp_enrollment_rejects_ambiguous_or_invalid_parameters(tmp_path):
    store = VaultStore(tmp_path / "vault")
    for query in [
        f"secret={SEED}",  # no implicit counter
        f"secret={SEED}&counter=-1",
        f"secret={SEED}&counter={1 << 64}",
        f"secret={SEED}&counter=0&COUNTER=1",
        f"secret={SEED}&counter=0&secret={SEED}",
        f"secret={SEED}&counter=0&period=30",
        f"secret={SEED}&counter=0&digits=9",
        f"secret={SEED}&counter=0&algorithm=MD5",
        "secret=A&counter=0",  # syntactically base32, but not decodable
    ]:
        with pytest.raises(VaultError) as error:
            _login(store, f"otpauth://hotp/Example?{query}")
        assert SEED not in str(error.value) and "otpauth://" not in str(error.value)
    assert store.list_items() == []


def test_hotp_does_not_return_a_code_if_counter_persistence_fails(tmp_path, monkeypatch):
    store = VaultStore(tmp_path / "vault")
    meta = _login(store, f"otpauth://hotp/Example?secret={SEED}&counter=0")
    before = store._vault_path.read_bytes()

    def fail_write(*args, **kwargs):
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr("agent.vault_store.atomic_write_bytes", fail_write)
        with pytest.raises(VaultError, match="persist HOTP counter"):
            store.consume_otp(meta.id)
    assert store._vault_path.read_bytes() == before
    assert store.consume_otp(meta.id) == CODES[0]
    assert VaultStore(store._base).consume_otp(meta.id) == CODES[1]


def _consume_in_process(base, handle, barrier):
    store = VaultStore(base)
    barrier.wait(timeout=30)
    return [store.consume_otp(handle) for _ in range(5)]


def test_hotp_consumers_in_separate_processes_do_not_reuse_counters(tmp_path):
    base = tmp_path / "vault"
    meta = _login(VaultStore(base), f"otpauth://hotp/Example?secret={SEED}&counter=0")
    context = multiprocessing.get_context("spawn")
    with context.Manager() as manager:
        barrier = manager.Barrier(2)
        with ProcessPoolExecutor(max_workers=2, mp_context=context) as pool:
            futures = [pool.submit(_consume_in_process, base, meta.id, barrier) for _ in range(2)]
            codes = [code for future in futures for code in future.result(timeout=60)]
    assert sorted(codes) == sorted(CODES)
    assert VaultStore(base).resolve_secret(meta.id)["otp_secret"].endswith("|10")


def test_hotp_last_counter_is_consumed_once_then_refused(tmp_path):
    store = VaultStore(tmp_path / "vault")
    meta = _login(store, f"otpauth://hotp/Example?secret={SEED}&counter={(1 << 64) - 1}")
    assert store.consume_otp(meta.id) == "094451"
    with pytest.raises(VaultError, match="exhausted"):
        VaultStore(store._base).consume_otp(meta.id)


def test_backend_legacy_totp_and_absent_otp_do_not_rewrite_vault(tmp_path, monkeypatch):
    import time

    store = VaultStore(tmp_path / "vault")
    # Existing stored forms: bare seed and seed|digits|period|algorithm.
    plain = _login(store, SEED)
    custom = _login(store, f"otpauth://totp/Example?secret={SEED}&digits=8&period=60")
    absent = _login(store, "")
    monkeypatch.setattr("agent.vault_store.get_vault_store", lambda: store)
    monkeypatch.setattr(time, "time", lambda: 119)
    before = store._vault_path.read_bytes()
    backend = LocalLoginBackend()
    assert backend.resolve_otp(plain.id) == "969429"
    assert backend.resolve_otp(custom.id) == "94287082"
    assert backend.resolve_otp(absent.id) is None
    assert store._vault_path.read_bytes() == before
