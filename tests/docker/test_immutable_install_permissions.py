"""Docker smoke tests for immutable install permissions."""
from __future__ import annotations

import subprocess
import textwrap

from tests.docker.conftest import docker_exec, docker_exec_sh, start_container


def test_container_sets_hosted_write_policy_env(built_image: str) -> None:
    script = (
        'test "$HERMES_HOME" = "/opt/data" && '
        'test "$HERMES_WRITE_SAFE_ROOT" = "/opt/data:/tmp/hermes-files" && '
        'test "$TMPDIR" = "/tmp/hermes-files" && '
        # Opt-in extras install into PM generations under $HERMES_HOME, never
        # the sealed /opt/hermes tree, so the image must not refuse them.
        'test -z "${HERMES_DISABLE_LAZY_INSTALLS:-}" && '
        'test "$PYTHONDONTWRITEBYTECODE" = "1"'
    )
    result = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "sh", built_image, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr[-2000:]


def test_container_initializes_private_file_scratch_dir(
    built_image: str, container_name: str,
) -> None:
    start_container(built_image, container_name)
    scratch = docker_exec_sh(
        container_name,
        'test -d /tmp/hermes-files && test ! -L /tmp/hermes-files && '
        'test "$(stat -c %u /tmp/hermes-files)" = "$(id -u)" && '
        'test "$(stat -c %a /tmp/hermes-files)" = "700" && '
        'touch /tmp/hermes-files/write-probe',
    )
    assert scratch.returncode == 0, scratch.stderr[-2000:]
    policy = docker_exec(
        container_name,
        "/opt/hermes/.venv/bin/python",
        "-c",
        "from agent.file_safety import get_write_denied_error as denied; "
        "assert denied('/tmp/hermes-files/helper.py') is None; "
        "assert denied('/tmp/hermes-runtime/hermes-bot-desktop-alloc.lock') is not None; "
        "assert denied('/tmp/.X11-unix/X1') is not None",
    )
    assert policy.returncode == 0, policy.stdout + policy.stderr


def test_hermes_user_cannot_modify_install_but_can_write_data(built_image: str) -> None:
    script = textwrap.dedent(
        r"""
        set -eu
        /opt/hermes/.venv/bin/python - <<'PY'
        from pathlib import Path

        install_file = Path("/opt/hermes/agent/message_sanitization.py")
        try:
            with install_file.open("a", encoding="utf-8") as handle:
                handle.write("\n# unexpected hosted mutation\n")
        except PermissionError:
            pass
        else:
            raise SystemExit("install source write unexpectedly succeeded")

        skill_dir = Path("/opt/data/skills/permission-smoke")
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text("# Permission smoke\n", encoding="utf-8")
        if skill_file.read_text(encoding="utf-8") != "# Permission smoke\n":
            raise SystemExit("data write verification failed")
        PY
        """
    ).strip()
    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "su",
            built_image,
            "hermes",
            "-s",
            "/bin/sh",
            "-c",
            script,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]
