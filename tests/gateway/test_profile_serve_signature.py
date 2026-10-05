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


def test_profile_serve_signature_tracks_shell_hook_allowlist(tmp_path):
    """#132993: consent recorded after the scan (``shell-hooks-allowlist.json`` appearing or
    changing in the profile home) must flip the signature, so the multiplex re-scan rebuilds
    the adapters and registers the previously-skipped hook instead of waiting for a restart."""
    before = profile_serve_signature(tmp_path)  # no allowlist yet
    assert profile_serve_signature(tmp_path) == before

    allowlist = tmp_path / "shell-hooks-allowlist.json"
    allowlist.write_text(
        '{"approvals": [{"event": "pre_tool_call", "command": "/bin/true", '
        '"approved_at": "2026-10-05T00:00:00Z"}]}',
        encoding="utf-8",
    )
    after_consent = profile_serve_signature(tmp_path)
    assert after_consent != before  # re-scan fires, hook registers

    assert profile_serve_signature(tmp_path) == after_consent  # stable until changed
    allowlist.write_text('{"approvals": []}', encoding="utf-8")  # consent revoked
    assert profile_serve_signature(tmp_path) != after_consent


def test_signature_files_cover_the_real_allowlist_filename():
    """The signature must watch the file agent.shell_hooks actually reads, or consent changes
    stay invisible to the re-scan (#132993)."""
    from agent.shell_hooks import ALLOWLIST_FILENAME
    from gateway.run_profile_reconcile import _PROFILE_SIGNATURE_FILES

    assert ALLOWLIST_FILENAME in _PROFILE_SIGNATURE_FILES
