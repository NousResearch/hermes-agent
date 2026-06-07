"""Media-ref auth (per-run HMAC), safe_root confinement, and cleanup.

Harvests #18510's safe_root / strict-auth path-confinement idea
(credit: ayoahha + Donmeusi).
"""

import pytest

from gateway.media_spool import MediaSpool, sign_ref, verify_ref, confine_to_safe_root


def test_ref_token_is_scoped_to_run():
    secret = "worker-bearer"
    token = sign_ref("runA", "ref1", secret)
    assert verify_ref("runA", "ref1", token, secret) is True
    # Same ref, different run → rejected (no cross-run reuse).
    assert verify_ref("runB", "ref1", token, secret) is False
    # Tampered token → rejected.
    assert verify_ref("runA", "ref1", "deadbeef", secret) is False


def test_safe_root_rejects_traversal(tmp_path):
    root = tmp_path / "spool"
    root.mkdir()
    with pytest.raises(ValueError):
        confine_to_safe_root("../../etc/passwd", root)
    inside = confine_to_safe_root("ok.bin", root)
    assert str(inside).startswith(str(root.resolve()))


def test_resolve_rejects_ref_escaping_root(tmp_path):
    spool = MediaSpool(tmp_path / "spool")
    with pytest.raises((KeyError, ValueError)):
        spool.resolve("../../../etc/passwd")


def test_unlink_is_idempotent(tmp_path):
    spool = MediaSpool(tmp_path / "spool")
    ref = spool.mint(b"d", filename="a.png", mime="image/png", kind="image")
    spool.unlink(ref.ref)
    spool.unlink(ref.ref)  # second unlink must not raise
