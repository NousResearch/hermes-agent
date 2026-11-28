"""Login shells in the image keep Hermes' own dirs on PATH.

Debian's ``/etc/profile`` resets PATH for every login shell, so without
``/etc/profile.d/hermes.sh`` a ``bash -l`` (``docker exec <c> bash -l``, ssh, the
terminal tool's background ``bash -lic``) lost ``/opt/hermes/bin`` and bare
``hermes`` exited 127 (#56634).
"""
from __future__ import annotations

import subprocess


def _login_path(image: str, user: str) -> list[str]:
    r = subprocess.run(
        ["docker", "run", "--rm", "-u", user, "--entrypoint", "bash", image, "-lc", 'printf "%s" "$PATH"'],
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert r.returncode == 0, r.stderr
    return r.stdout.split(":")


def test_login_shell_resolves_hermes_to_the_image_shim(built_image: str) -> None:
    r = subprocess.run(
        ["docker", "run", "--rm", "-u", "hermes", "--entrypoint", "bash", built_image,
         "-lc", 'echo "hermes=$(command -v hermes)"; echo "path=$PATH"'],
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert r.returncode == 0, r.stderr
    lines = dict(line.split("=", 1) for line in r.stdout.splitlines() if "=" in line)
    assert lines["hermes"] == "/opt/hermes/bin/hermes"
    entries = lines["path"].split(":")
    assert entries.index("/opt/hermes/bin") < entries.index("/opt/hermes/.venv/bin")
    assert entries.count("/opt/hermes/bin") == 1


def test_root_login_shell_lists_no_data_volume_dir(built_image: str) -> None:
    """Root shells resolve commands from root-owned dirs only; /opt/data is hermes-writable."""
    entries = _login_path(built_image, "root")
    assert "/opt/hermes/bin" in entries
    assert not [e for e in entries if e.startswith("/opt/data")], entries


def test_hermes_login_shell_appends_its_local_bin_after_system_dirs(built_image: str) -> None:
    entries = _login_path(built_image, "hermes")
    assert entries.count("/opt/data/.local/bin") == 1, entries
    assert entries.index("/opt/data/.local/bin") > entries.index("/usr/bin"), entries
