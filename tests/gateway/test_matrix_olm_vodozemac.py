"""vodozemac backend for mautrix E2EE (replaces deprecated python-olm/libolm).

Covers the python-olm-compatible layer (``plugins/platforms/matrix/olm_vodozemac.py``), the
libolm -> vodozemac crypto-store migration (``crypto_migration.py``) against golden libolm
pickles (so CI needs no python-olm), and, when python-olm happens to be installed,
cross-implementation interop in a subprocess (``matrix_olm_interop_check.py``).
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("vodozemac")

from plugins.platforms.matrix import olm_vodozemac as vz

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "matrix_libolm_pickles.json").read_text())
PASS = FIXTURES["passphrase"]
_TS = "2026-10-01 12:00:00"  # timestamp columns as mautrix/aiosqlite store them


def _otk(account: vz.Account) -> str:
    account.generate_one_time_keys(1)
    key = next(iter(account.one_time_keys["curve25519"].values()))
    account.mark_keys_as_published()
    return key


# ------------------------------------------------------------------ python-olm compatible API


def test_pk_signing_rfc8032_vector():
    """Cross-signing keys: RFC 8032 section 7.1, test 1 (libolm produced the same bytes)."""
    seed = bytes.fromhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")
    signer = vz.PkSigning(seed)
    assert vz._b64decode(signer.public_key).hex() == (
        "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a")
    assert vz._b64decode(signer.sign(b"")).hex() == (
        "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065"
        "224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b")
    vz.ed25519_verify(signer.public_key, "", signer.sign(""))
    with pytest.raises(vz.OlmVerifyError):
        vz.ed25519_verify(signer.public_key, "tampered", signer.sign(""))
    with pytest.raises(vz.PkSigningError):
        vz.PkSigning(b"short")


def test_olm_round_trip_and_first_prekey_plaintext():
    alice, bob = vz.Account(), vz.Account()
    out = vz.OutboundSession(alice, bob.identity_keys["curve25519"], _otk(bob))
    first = out.encrypt("hello bob")
    assert isinstance(first, vz.OlmPreKeyMessage)
    inbound = vz.InboundSession(bob, vz.OlmPreKeyMessage(first.ciphertext), alice.identity_keys["curve25519"])
    # mautrix pickles the new session and reloads it before decrypting the same pre-key message
    reloaded = vz.Session.from_pickle(inbound.pickle("k"), "k")
    assert reloaded.matches(vz.OlmPreKeyMessage(first.ciphertext))
    assert reloaded.decrypt(vz.OlmPreKeyMessage(first.ciphertext)) == "hello bob"
    with pytest.raises(vz.OlmSessionError):  # consumed once; a replay is rejected like libolm
        reloaded.decrypt(vz.OlmPreKeyMessage(first.ciphertext))
    reply = reloaded.encrypt("hi alice")
    assert isinstance(reply, vz.OlmMessage)
    assert out.decrypt(reply) == "hi alice"
    assert out.id == reloaded.id
    with pytest.raises(TypeError):
        vz.Session()


def test_inbound_session_requires_sender_key():
    alice, bob = vz.Account(), vz.Account()
    first = vz.OutboundSession(alice, bob.identity_keys["curve25519"], _otk(bob)).encrypt("x")
    with pytest.raises(vz.OlmSessionError):
        vz.InboundSession(bob, vz.OlmPreKeyMessage(first.ciphertext))


def test_megolm_round_trip_padding_and_tamper():
    out = vz.OutboundGroupSession()
    inbound = vz.InboundGroupSession(out.session_key)
    ciphertexts = [out.encrypt("x" * (150 + i)) for i in range(4)]
    for i, ct in enumerate(ciphertexts):
        padded = ct + "=" * (-len(ct) % 4)  # padded base64 is valid per spec (#138108)
        assert inbound.decrypt(padded) == ("x" * (150 + i), i)
    raw = bytearray(vz._b64decode(ciphertexts[0]))
    raw[-1] ^= 1
    with pytest.raises(vz.OlmGroupSessionError):
        inbound.decrypt(vz._b64encode(bytes(raw)))
    exported = inbound.export_session(2)
    late = vz.InboundGroupSession.import_session(exported)
    assert late.first_known_index == 2
    assert late.decrypt(ciphertexts[3]) == ("x" * 153, 3)
    with pytest.raises(vz.OlmGroupSessionError):
        late.decrypt(ciphertexts[0])


def test_pickles_round_trip_and_reject_wrong_key():
    acct = vz.Account()
    assert vz.Account.from_pickle(acct.pickle(PASS), PASS).identity_keys == acct.identity_keys
    with pytest.raises(vz.OlmAccountError):
        vz.Account.from_pickle(acct.pickle(PASS), "wrong")
    out = vz.OutboundGroupSession()
    assert vz.OutboundGroupSession.from_pickle(out.pickle(PASS), PASS).id == out.id
    inbound = vz.InboundGroupSession(out.session_key)
    assert vz.InboundGroupSession.from_pickle(inbound.pickle(PASS), PASS).id == inbound.id


# ------------------------------------------------------------------ golden libolm pickles


def test_libolm_account_pickle_keeps_identity():
    account = vz.Account.from_pickle(FIXTURES["account"]["pickle"].encode(), PASS)
    assert account.identity_keys == FIXTURES["account"]["identity_keys"]
    assert account.sign("hello") == FIXTURES["account"]["signature_of_hello"]


def test_libolm_olm_session_pickle_continues_conversation():
    data = FIXTURES["olm_session"]
    session = vz.Session.from_pickle(data["pickle"].encode(), PASS)
    assert session.id == data["session_id"]
    msg_cls = vz.OlmPreKeyMessage if data["next_message"]["type"] == 0 else vz.OlmMessage
    assert session.decrypt(msg_cls(data["next_message"]["body"])) == data["next_plaintext"]


def test_libolm_inbound_group_session_pickle_keeps_history():
    data = FIXTURES["megolm_inbound"]
    session = vz.InboundGroupSession.from_pickle(data["pickle"].encode(), PASS)
    assert session.id == data["session_id"]
    for key in ("before", "after"):
        assert session.decrypt(data[key]["ciphertext"]) == (data[key]["plaintext"], data[key]["index"])


def test_libolm_outbound_group_session_pickle_is_rejected():
    with pytest.raises(vz.OlmGroupSessionError):
        vz.OutboundGroupSession.from_pickle(FIXTURES["megolm_outbound"]["pickle"].encode(), PASS)


# ------------------------------------------------------------------ crypto.db migration


def _seed_libolm_store(db_path: Path, account_id: str) -> None:
    """A mautrix 0.21 crypto.db as python-olm left it (schema created by mautrix itself)."""
    async def create_schema():
        from mautrix.crypto.store.asyncpg import PgCryptoStore
        from mautrix.util.async_db import Database
        db = Database.create(f"sqlite:///{db_path}", upgrade_table=PgCryptoStore.upgrade_table)
        await db.start()
        await db.stop()
    asyncio.run(create_schema())
    import sqlite3
    with sqlite3.connect(db_path) as conn:
        conn.execute("INSERT INTO crypto_account (account_id, device_id, shared, sync_token, account) "
                     "VALUES (?, 'HERMESDEV', 1, '', ?)", (account_id, FIXTURES["account"]["pickle"].encode()))
        olm_session = FIXTURES["olm_session"]
        conn.execute("INSERT INTO crypto_olm_session (account_id, session_id, sender_key, session, created_at, "
                     "last_encrypted, last_decrypted) VALUES (?, ?, ?, ?, ?, ?, ?)",
                     (account_id, olm_session["session_id"], olm_session["peer_curve25519"],
                      olm_session["pickle"].encode(), _TS, _TS, _TS))
        inbound = FIXTURES["megolm_inbound"]
        conn.execute("INSERT INTO crypto_megolm_inbound_session (account_id, session_id, sender_key, signing_key, "
                     "room_id, session, forwarding_chains) VALUES (?, ?, ?, ?, '!room:example.org', ?, '')",
                     (account_id, inbound["session_id"], inbound["sender_curve25519"], inbound["sender_ed25519"],
                      inbound["pickle"].encode()))
        outbound = FIXTURES["megolm_outbound"]
        conn.execute("INSERT INTO crypto_megolm_outbound_session (account_id, room_id, session_id, session, shared, "
                     "max_messages, message_count, max_age, created_at, last_used) VALUES "
                     "(?, '!room:example.org', ?, ?, 1, 100, 1, 604800000, ?, ?)",
                     (account_id, outbound["session_id"], outbound["pickle"].encode(), _TS, _TS))


def test_crypto_store_migration_backs_up_and_preserves_keys(tmp_path):
    pytest.importorskip("mautrix")
    pytest.importorskip("aiosqlite")
    from plugins.platforms.matrix import crypto_migration

    vz.install()  # what crypto_backend.select_backend() does before mautrix.crypto is imported
    from mautrix.crypto.store.asyncpg import PgCryptoStore
    from mautrix.types import IdentityKey, RoomID, SessionID
    from mautrix.util.async_db import Database

    account_id = "@hermes:example.org"
    db_path = tmp_path / "crypto.db"
    _seed_libolm_store(db_path, account_id)

    backup = crypto_migration.backup_if_libolm_store(db_path, (PASS,))
    assert backup is not None and backup.exists()
    assert stat.S_IMODE(os.stat(backup).st_mode) == 0o600
    assert backup.read_bytes()[:16] == b"SQLite format 3\x00"

    async def migrate_and_use():
        db = Database.create(f"sqlite:///{db_path}", upgrade_table=PgCryptoStore.upgrade_table)
        await db.start()
        try:
            report = await crypto_migration.migrate_store_to_vodozemac(db, account_id, PASS)
            again = await crypto_migration.migrate_store_to_vodozemac(db, account_id, PASS)
            store = PgCryptoStore(account_id=account_id, pickle_key=PASS, db=db)
            await store.open()
            account = await store.get_account()
            inbound = await store.get_group_session(
                RoomID("!room:example.org"), SessionID(FIXTURES["megolm_inbound"]["session_id"]))
            sessions = await store.get_sessions(IdentityKey(FIXTURES["olm_session"]["peer_curve25519"]))
            outbound = await store.get_outbound_group_session(RoomID("!room:example.org"))
            return report, again, account, inbound, sessions, outbound
        finally:
            await db.stop()

    report, again, account, inbound, sessions, outbound = asyncio.run(migrate_and_use())
    assert (report.account, report.olm_sessions, report.inbound_group_sessions,
            report.outbound_group_sessions_dropped, report.unreadable) == (True, 1, 1, 1, 0)
    assert not again.changed  # idempotent
    assert account.identity_keys == FIXTURES["account"]["identity_keys"]
    for key in ("before", "after"):
        data = FIXTURES["megolm_inbound"][key]
        assert inbound.decrypt(data["ciphertext"]) == (data["plaintext"], data["index"])
    assert len(sessions) == 1 and sessions[0].id == FIXTURES["olm_session"]["session_id"]
    assert outbound is None  # rotated: mautrix creates and shares a new one on the next send
    # Second start: nothing left to back up.
    assert crypto_migration.backup_if_libolm_store(db_path, (PASS,)) is None


# ------------------------------------------------------------------ backend selection


def test_select_backend_prefers_vodozemac(monkeypatch):
    from plugins.platforms.matrix import crypto_backend
    monkeypatch.setattr(crypto_backend, "_selected", None)
    monkeypatch.delenv(crypto_backend.ENV_VAR, raising=False)
    if "olm" in sys.modules and sys.modules["olm"] is not vz:
        pytest.skip("real python-olm already imported in this process")
    assert crypto_backend.select_backend() == crypto_backend.VODOZEMAC
    assert sys.modules["olm"] is vz
    assert sys.modules["_libolm"].ffi is None
    assert crypto_backend.select_backend() == crypto_backend.VODOZEMAC  # idempotent


def _fresh_backend(monkeypatch, requested: str | None, *, vodozemac: bool, python_olm: bool):
    """crypto_backend in a state where only the given implementations are importable."""
    import importlib.util
    from plugins.platforms.matrix import crypto_backend
    monkeypatch.setattr(crypto_backend, "_selected", None)
    if requested is None:
        monkeypatch.delenv(crypto_backend.ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(crypto_backend.ENV_VAR, requested)
    monkeypatch.delitem(sys.modules, "olm", raising=False)
    monkeypatch.delitem(sys.modules, "_libolm", raising=False)
    real_find_spec = importlib.util.find_spec
    present = {"vodozemac": vodozemac, "olm": python_olm}
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: (
        (real_find_spec("json") if present[name] else None) if name in present else real_find_spec(name, *a)))
    return crypto_backend


def test_explicit_vodozemac_never_falls_back_to_libolm(monkeypatch):
    backend = _fresh_backend(monkeypatch, "vodozemac", vodozemac=False, python_olm=True)
    assert backend.select_backend() == backend.NONE
    assert "olm" not in sys.modules


def test_explicit_libolm_never_installs_the_vodozemac_layer(monkeypatch):
    backend = _fresh_backend(monkeypatch, "libolm", vodozemac=True, python_olm=True)
    assert backend.select_backend() == backend.LIBOLM
    assert "olm" not in sys.modules  # the real python-olm is imported by mautrix later, not us


def test_auto_falls_back_to_libolm_only_without_vodozemac(monkeypatch):
    backend = _fresh_backend(monkeypatch, None, vodozemac=False, python_olm=True)
    assert backend.select_backend() == backend.LIBOLM
    backend = _fresh_backend(monkeypatch, "auto", vodozemac=False, python_olm=False)
    assert backend.select_backend() == backend.NONE


def test_unknown_backend_selects_nothing(monkeypatch):
    backend = _fresh_backend(monkeypatch, "olm-please", vodozemac=True, python_olm=True)
    assert backend.select_backend() == backend.NONE
    assert "olm" not in sys.modules


# ------------------------------------------------------------------ fail-closed store handling


def test_backup_failure_raises_and_leaves_no_partial_copy(tmp_path, monkeypatch):
    pytest.importorskip("mautrix")
    pytest.importorskip("aiosqlite")
    from plugins.platforms.matrix import crypto_migration

    db_path = tmp_path / "crypto.db"
    _seed_libolm_store(db_path, "@hermes:example.org")

    import sqlite3
    import types

    class _FailingBackup:  # a connection whose copy fails mid-way (disk full, I/O error)
        def __init__(self, conn):
            self._conn = conn

        def __getattr__(self, name):
            return getattr(self._conn, name)

        def backup(self, target, *args, **kwargs):
            raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(crypto_migration, "sqlite3", types.SimpleNamespace(
        connect=lambda *a, **k: _FailingBackup(sqlite3.connect(*a, **k)), Error=sqlite3.Error))
    with pytest.raises(crypto_migration.CryptoStoreBackupError):
        crypto_migration.backup_if_libolm_store(db_path, (PASS,))
    assert crypto_migration.latest_backup(db_path) is None


def test_backup_is_never_world_readable_even_with_open_umask(tmp_path):
    pytest.importorskip("mautrix")
    pytest.importorskip("aiosqlite")
    from plugins.platforms.matrix import crypto_migration

    db_path = tmp_path / "crypto.db"
    _seed_libolm_store(db_path, "@hermes:example.org")
    old = os.umask(0)
    try:
        first = crypto_migration.backup_if_libolm_store(db_path, (PASS,))
        second = crypto_migration.backup_if_libolm_store(db_path, (PASS,))  # same second: no clobber
    finally:
        os.umask(old)
    assert first is not None and second is not None and first != second
    for backup in (first, second):
        assert stat.S_IMODE(os.stat(backup).st_mode) == 0o600


def test_first_prekey_plaintext_expires(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(vz.time, "monotonic", lambda: clock[0])
    vz._remember_first_plaintext("sid", b"ct", b"room key")
    clock[0] += vz._FIRST_PLAINTEXT_TTL + 1
    assert vz._take_first_plaintext("sid", b"ct") is None
    vz._remember_first_plaintext("sid", b"ct", b"room key")
    assert vz._take_first_plaintext("sid", b"ct") == b"room key"
    assert vz._take_first_plaintext("sid", b"ct") is None  # one-shot


# ------------------------------------------------------------------ interop with real libolm


def test_interop_with_real_python_olm_if_installed():
    """Byte-level interop vs libolm. Runs in a subprocess so the real `olm` package never meets
    the vodozemac backend registered as `olm` here; skipped when python-olm is not installed
    (e.g. `uv run --with python-olm==3.2.16 pytest ...` to enable)."""
    script = Path(__file__).parent / "matrix_olm_interop_check.py"
    proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=120,
                          check=False)
    if proc.returncode == 77:
        pytest.skip(proc.stdout.strip() or "python-olm not installed")
    assert proc.returncode == 0, proc.stdout + proc.stderr
