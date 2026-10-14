"""#111105: the profile re-scan watcher must rebuild adapters after a config.yaml replacement
that keeps mtime and size (``cp -p``, ``rsync -t``, a timestamp-pinning writer)."""
import os
import shutil

from gateway.run_profile_reconcile import profile_serve_signature


def test_profile_serve_signature_changes_on_replacement_with_pinned_mtime(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("x" * 64, encoding="utf-8")
    before = profile_serve_signature(tmp_path)
    assert profile_serve_signature(tmp_path) == before  # unchanged file: stable
    st = cfg.stat()
    other = tmp_path / "other"
    other.write_text("y" * 64, encoding="utf-8")
    shutil.copy2(other, cfg)
    os.utime(cfg, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert (cfg.stat().st_mtime_ns, cfg.stat().st_size) == (st.st_mtime_ns, st.st_size)
    assert profile_serve_signature(tmp_path) != before
