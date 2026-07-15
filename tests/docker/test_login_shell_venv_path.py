"""Integration test for the login-shell venv PATH drop-in (#56634).

Builds the image (or reuses ``HERMES_TEST_IMAGE``) and runs a real login shell
(``bash -l``, which sources Debian's /etc/profile and only then
/etc/profile.d/*.sh) as the unprivileged ``hermes`` user -- the same shell
``init_session`` snapshots via ``bash -l -c ... export -p``. Without the
/etc/profile.d drop-in the login shell inherits /etc/profile's PATH reset, so a
bare ``python3`` resolves to /usr/local/bin/python3 (the system interpreter,
which lacks Pillow). These tests prove the drop-in restores the venv end-to-end
in the built image.

Auto-skips when Docker is unavailable (see tests/docker/conftest.py).
"""
from __future__ import annotations

import subprocess


def _login_shell(image: str, script: str) -> subprocess.CompletedProcess[str]:
    """Run ``script`` in a login shell as the hermes user, entrypoint bypassed."""
    return subprocess.run(
        [
            "docker", "run", "--rm",
            "--entrypoint", "bash",
            "--user", "hermes",
            image,
            "-lc", script,
        ],
        capture_output=True, text=True, timeout=120,
    )


def test_login_shell_resolves_venv_python(built_image: str) -> None:
    r = _login_shell(built_image, "command -v python3")
    assert r.returncode == 0, r.stderr
    python3 = r.stdout.strip()
    # The venv (and the privilege-drop shim dir ahead of it) live under
    # /opt/hermes; the system interpreter the /etc/profile reset would leave in
    # front is /usr/local/bin/python3 or /usr/bin/python3.
    assert python3.startswith("/opt/hermes/"), (
        f"login-shell python3 resolved to {python3!r}; expected the Hermes venv "
        "under /opt/hermes -- the /etc/profile.d drop-in did not restore PATH."
    )


def test_login_shell_python_imports_venv_only_dependency(built_image: str) -> None:
    # Pillow is installed only in the venv, so a bare ``import PIL`` succeeding
    # proves the login shell selected the venv interpreter, not the system one.
    r = _login_shell(built_image, 'python3 -c "import PIL; print(PIL.__name__)"')
    assert r.returncode == 0, (
        f"`import PIL` failed in a login shell (rc={r.returncode}): "
        f"{(r.stderr.strip() or r.stdout.strip())!r}"
    )
    assert "PIL" in r.stdout
