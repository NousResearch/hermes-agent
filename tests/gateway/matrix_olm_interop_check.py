"""Cross-implementation check: the vodozemac backend vs real python-olm/libolm.

Run as a script (``python matrix_olm_interop_check.py``) in an interpreter where BOTH
python-olm and vodozemac are installed. It is executed in a subprocess by
``test_matrix_olm_vodozemac.py`` so that importing the real ``olm`` package can never
collide with the vodozemac backend registered as ``olm`` in the test process.

Exit codes: 0 = all checks passed, 1 = a check failed, 77 = python-olm not installed (skip).
"""

from __future__ import annotations

import importlib.util
import os
import sys
import traceback
from pathlib import Path

try:
    import olm as libolm  # the real python-olm package
except ImportError:
    print("SKIP: python-olm is not installed")
    sys.exit(77)

if not hasattr(libolm, "__file__") or "olm_vodozemac" in str(getattr(libolm, "__file__", "")):
    print("SKIP: 'olm' is not python-olm")
    sys.exit(77)

_SHIM_PATH = Path(__file__).resolve().parents[2] / "plugins" / "platforms" / "matrix" / "olm_vodozemac.py"
_spec = importlib.util.spec_from_file_location("olm_vodozemac_under_test", _SHIM_PATH)
vz = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = vz
_spec.loader.exec_module(vz)  # imported under its own name; NOT installed as `olm`

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


def _otk(account) -> str:
    account.generate_one_time_keys(1)
    key = next(iter(account.one_time_keys["curve25519"].values()))
    account.mark_keys_as_published()
    return key


def _olm_conversation(sender, receiver, sender_lib, receiver_lib):
    """sender opens an Olm session to receiver, then they exchange messages both ways."""
    out = sender_lib.OutboundSession(sender, receiver.identity_keys["curve25519"], _otk(receiver))
    first = out.encrypt("hello from sender")
    assert isinstance(first, sender_lib.OlmPreKeyMessage), type(first)
    pre = receiver_lib.OlmPreKeyMessage(first.ciphertext)
    inbound = receiver_lib.InboundSession(receiver, pre, sender.identity_keys["curve25519"])
    receiver.remove_one_time_keys(inbound)
    assert inbound.matches(receiver_lib.OlmPreKeyMessage(first.ciphertext))
    assert inbound.decrypt(pre) == "hello from sender"
    assert inbound.id == out.id
    for i in range(5):
        reply = inbound.encrypt(f"reply {i}")
        cls = sender_lib.OlmPreKeyMessage if reply.message_type == 0 else sender_lib.OlmMessage
        assert out.decrypt(cls(reply.ciphertext)) == f"reply {i}"
        again = out.encrypt(f"again {i}")
        cls = receiver_lib.OlmPreKeyMessage if again.message_type == 0 else receiver_lib.OlmMessage
        assert inbound.decrypt(cls(again.ciphertext)) == f"again {i}"
    return out, inbound


@check
def olm_libolm_to_vodozemac():
    _olm_conversation(libolm.Account(), vz.Account(), libolm, vz)


@check
def olm_vodozemac_to_libolm():
    _olm_conversation(vz.Account(), libolm.Account(), vz, libolm)


@check
def megolm_libolm_to_vodozemac_incl_padding():
    out = libolm.OutboundGroupSession()
    inbound = vz.InboundGroupSession(out.session_key)
    assert inbound.id == out.id
    for i in range(20):
        ct = out.encrypt("x" * (100 + i))
        assert inbound.decrypt(ct) == ("x" * (100 + i), i)
        padded = ct + "=" * (-len(ct) % 4)
        assert inbound.decrypt(padded) == ("x" * (100 + i), i)


@check
def megolm_vodozemac_to_libolm():
    out = vz.OutboundGroupSession()
    inbound = libolm.InboundGroupSession(out.session_key)
    assert inbound.id == out.id
    for i in range(20):
        assert inbound.decrypt(out.encrypt(f"msg {i}")) == (f"msg {i}", i)
    assert out.message_index == 20


@check
def megolm_tampered_signature_rejected():
    out = libolm.OutboundGroupSession()
    inbound = vz.InboundGroupSession(out.session_key)
    ct = out.encrypt("authentic")
    raw = bytearray(vz._b64decode(ct))
    raw[-1] ^= 0x01  # flip a bit in the Ed25519 signature
    try:
        inbound.decrypt(vz._b64encode(bytes(raw)))
    except vz.OlmGroupSessionError:
        return
    raise AssertionError("tampered Megolm message was accepted")


@check
def megolm_export_import_both_directions():
    out = libolm.OutboundGroupSession()
    for i in range(3):
        out.encrypt(f"early {i}")
    lib_in = libolm.InboundGroupSession(out.session_key)
    exported = lib_in.export_session(3)
    vz_in = vz.InboundGroupSession.import_session(exported)
    assert vz_in.first_known_index == 3
    assert vz_in.decrypt(out.encrypt("late"))[0] == "late"
    lib_again = libolm.InboundGroupSession.import_session(vz_in.export_session(4))
    assert lib_again.decrypt(out.encrypt("later")) == ("later", 4)


@check
def account_pickle_migration_keeps_identity():
    old = libolm.Account()
    old.generate_one_time_keys(3)
    pickled = old.pickle("hermes-pickle-key")
    new = vz.Account.from_pickle(pickled, "hermes-pickle-key")
    assert new.identity_keys == old.identity_keys
    # Same one-time key values. Key *IDs* are opaque and re-encoded (libolm used 32-bit IDs,
    # vodozemac 64-bit); peers address OTKs by value when building sessions, so this is harmless.
    assert sorted(new.one_time_keys["curve25519"].values()) == sorted(old.one_time_keys["curve25519"].values())
    msg = '{"canonical":"json"}'
    assert new.sign(msg) == old.sign(msg)  # same Ed25519 key => identical deterministic signature
    repickled = new.pickle("hermes-pickle-key")
    assert vz.Account.from_pickle(repickled, "hermes-pickle-key").identity_keys == old.identity_keys
    try:
        vz.Account.from_pickle(pickled, "wrong-key")
    except vz.OlmAccountError:
        pass
    else:
        raise AssertionError("wrong pickle key accepted")


@check
def pre_upgrade_one_time_keys_still_work():
    """OTKs a libolm bot published before upgrading are claimed by peers after the upgrade."""
    bot = libolm.Account()
    bot.generate_one_time_keys(bot.max_one_time_keys // 2)  # what mautrix keeps on the server
    published = list(bot.one_time_keys["curve25519"].values())
    bot.mark_keys_as_published()
    migrated = vz.Account.from_pickle(bot.pickle("k"), "k")
    for otk in published:
        peer = libolm.Account()
        first = libolm.OutboundSession(peer, migrated.identity_keys["curve25519"], otk).encrypt("hi")
        inbound = vz.InboundSession(migrated, vz.OlmPreKeyMessage(first.ciphertext),
                                    peer.identity_keys["curve25519"])
        assert inbound.decrypt(vz.OlmPreKeyMessage(first.ciphertext)) == "hi"


@check
def olm_session_pickle_migration_continues_conversation():
    alice, bob = libolm.Account(), libolm.Account()
    out, inbound = _olm_conversation(alice, bob, libolm, libolm)
    migrated = vz.Session.from_pickle(inbound.pickle("k"), "k")  # Bob upgrades to vodozemac
    assert migrated.id == out.id
    m = out.encrypt("after upgrade")
    cls = vz.OlmPreKeyMessage if m.message_type == 0 else vz.OlmMessage
    assert migrated.decrypt(cls(m.ciphertext)) == "after upgrade"
    r = migrated.encrypt("bob still here")
    cls = libolm.OlmPreKeyMessage if r.message_type == 0 else libolm.OlmMessage
    assert out.decrypt(cls(r.ciphertext)) == "bob still here"


@check
def inbound_group_session_pickle_migration():
    out = libolm.OutboundGroupSession()
    lib_in = libolm.InboundGroupSession(out.session_key)
    assert lib_in.decrypt(out.encrypt("before"))[0] == "before"
    migrated = vz.InboundGroupSession.from_pickle(lib_in.pickle("k"), "k")
    assert migrated.id == out.id
    assert migrated.decrypt(out.encrypt("after")) == ("after", 1)
    # history sent before the upgrade is still readable
    assert migrated.first_known_index == 0


@check
def outbound_group_session_libolm_pickle_is_rejected_cleanly():
    pickled = libolm.OutboundGroupSession().pickle("k")
    try:
        vz.OutboundGroupSession.from_pickle(pickled, "k")
    except vz.OlmGroupSessionError:
        return
    raise AssertionError("expected OlmGroupSessionError for a libolm outbound pickle")


@check
def pk_signing_byte_identical():
    for _ in range(25):
        seed = os.urandom(32)
        a, b = libolm.PkSigning(seed), vz.PkSigning(seed)
        assert a.public_key == b.public_key
        for msg in ("", "x", '{"a":1,"b":[true,null]}', "ü€😀" * 50):
            assert a.sign(msg) == b.sign(msg)
            libolm.ed25519_verify(a.public_key, msg, b.sign(msg))
            vz.ed25519_verify(b.public_key, msg, a.sign(msg))


@check
def account_signature_verifies_with_libolm():
    acct = vz.Account()
    sig = acct.sign("payload")
    libolm.ed25519_verify(acct.identity_keys["ed25519"], "payload", sig)


@check
def utility_sha256_matches():
    for data in ("", "abc", "ü" * 100):
        assert vz.sha256(data) == libolm.sha256(data)


def main() -> int:
    failed = 0
    for fn in CHECKS:
        try:
            fn()
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
        else:
            print(f"ok   {fn.__name__}")
    from importlib.metadata import version
    print(f"python-olm {version('python-olm')} vs vodozemac {version('vodozemac')}: "
          f"{len(CHECKS) - failed}/{len(CHECKS)} interop checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
