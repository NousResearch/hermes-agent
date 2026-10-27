"""#111105: the profile re-scan watcher must rebuild adapters after a config.yaml replacement
that keeps mtime and size (``cp -p``, ``rsync -t``, a timestamp-pinning writer)."""
import os
import shutil

from gateway.run_profile_reconcile import profile_serve_signature
from tests.waiting import wait_for_ctime_tick


def test_profile_serve_signature_changes_on_replacement_with_pinned_mtime(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("x" * 64, encoding="utf-8")
    before = profile_serve_signature(tmp_path)
    assert profile_serve_signature(tmp_path) == before  # unchanged file: stable
    st = cfg.stat()
    wait_for_ctime_tick(tmp_path, st.st_ctime_ns)
    other = tmp_path / "other"
    other.write_text("y" * 64, encoding="utf-8")
    shutil.copy2(other, cfg)
    os.utime(cfg, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert (cfg.stat().st_mtime_ns, cfg.stat().st_size) == (st.st_mtime_ns, st.st_size)
    assert profile_serve_signature(tmp_path) != before


def test_profile_serve_signature_follows_the_config_backend_version(tmp_path, monkeypatch):
    """Remote Config (#122509): a remote layer changes with no local config.yaml change, so the
    signature must come from the config backend's version, not a stat of the local file."""
    from hermes_cli import config_backend

    class _Backend:
        name = "remote"
        version_now = ("etag-1",)

        def version(self, home):
            return self.version_now

    backend = _Backend()
    monkeypatch.setattr(config_backend, "get_config_backend", lambda: backend)
    before = profile_serve_signature(tmp_path)
    assert profile_serve_signature(tmp_path) == before
    backend.version_now = ("etag-2",)
    assert not (tmp_path / "config.yaml").exists()
    assert profile_serve_signature(tmp_path) != before
