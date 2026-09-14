"""Regression: the profile re-scan watcher must detect file *replacements*.

``profile_serve_signature`` keyed files on ``(st_mtime_ns, st_size)`` only, so a
replacement preserving both (``cp -p``, ``rsync -t``, timestamp-pinning) never
triggered a profile re-scan and adapters were not rebuilt until restart.

See #111105.
"""
import os
import shutil


def test_profile_serve_signature_detects_replacement_with_pinned_mtime(tmp_path):
    from gateway.run_profile_reconcile import profile_serve_signature

    cfg = tmp_path / "config.yaml"
    cfg.write_text("x" * 64, encoding="utf-8")
    before = profile_serve_signature(tmp_path)
    st = cfg.stat()
    other = tmp_path / "other"
    other.write_text("y" * 64, encoding="utf-8")
    shutil.copy2(other, cfg)
    os.utime(cfg, ns=(st.st_atime_ns, st.st_mtime_ns))
    after = cfg.stat()
    assert after.st_mtime_ns == st.st_mtime_ns
    assert after.st_size == st.st_size
    assert profile_serve_signature(tmp_path) != before
