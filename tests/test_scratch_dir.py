"""Scratch dir contract: TMPDIR/TMP/TEMP follow HERMES_HOME/cache/scratch unless the user set them."""

import os
import subprocess
import sys
import time

from hermes_constants import apply_scratch_tmp_env, get_scratch_dir, prune_scratch_dir


def test_scratch_env_follows_home_and_respects_user_tmpdir(tmp_path):
    """Unset temp vars → scratch of env HERMES_HOME; a Hermes-exported value re-derives for a routed
    home; a user/OS-set value (macOS ``/var/folders``, ``%TEMP%``) is never touched."""
    home_a, home_b = tmp_path / "a", tmp_path / "b"
    env = {"HERMES_HOME": str(home_a)}
    assert apply_scratch_tmp_env(env) is True
    scratch_a = str(home_a / "cache" / "scratch")
    assert env["TMPDIR"] == env["TMP"] == env["TEMP"] == env["HERMES_SCRATCH_DIR"] == scratch_a
    assert os.path.isdir(scratch_a)

    env["HERMES_HOME"] = str(home_b)  # child served under another profile's home
    assert apply_scratch_tmp_env(env) is True
    assert env["TMPDIR"] == str(home_b / "cache" / "scratch")

    user_env = {"HERMES_HOME": str(home_a), "TMPDIR": "/var/folders/zz"}
    assert apply_scratch_tmp_env(user_env) is False
    assert user_env["TMPDIR"] == "/var/folders/zz" and "HERMES_SCRATCH_DIR" not in user_env
    assert "TMP" not in user_env  # a partially user-set triple is left exactly as found


def test_bootstrap_import_exports_scratch_to_process_and_children(tmp_path):
    """``import hermes_bootstrap`` alone makes ``tempfile`` (this process AND a child) land in scratch."""
    env = {k: v for k, v in os.environ.items() if k not in ("TMPDIR", "TMP", "TEMP", "HERMES_SCRATCH_DIR")}
    env["HERMES_HOME"] = str(tmp_path)
    code = ("import tempfile, os, subprocess, sys; import hermes_bootstrap; "
            "print(tempfile.gettempdir()); "
            "print(subprocess.run([sys.executable, '-c', 'import tempfile;print(tempfile.gettempdir())'],"
            " capture_output=True, text=True).stdout.strip())")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                         cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))), check=True)
    expected = str(tmp_path / "cache" / "scratch")
    assert out.stdout.split() == [expected, expected]


def test_prune_removes_only_stale_top_level_entries(tmp_path):
    scratch = get_scratch_dir(tmp_path, prune=False)
    stale, fresh = scratch / "stale", scratch / "fresh.txt"
    stale.mkdir()
    (stale / "f").write_text("x", encoding="utf-8")
    fresh.write_text("y", encoding="utf-8")
    ancient = time.time() - 100 * 3600
    os.utime(stale, (ancient, ancient))
    assert prune_scratch_dir(scratch) == 1
    assert not stale.exists() and fresh.exists()
