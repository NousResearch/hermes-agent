"""Docker smoke tests for immutable install permissions."""
from __future__ import annotations

import subprocess
import textwrap


def test_container_sets_hosted_write_policy_env(built_image: str) -> None:
    script = (
        'test "$HERMES_HOME" = "/opt/data" && '
        'test "$HERMES_WRITE_SAFE_ROOT" = "/opt/data" && '
        'test "$HERMES_DISABLE_LAZY_INSTALLS" = "1" && '
        'test "$PYTHONDONTWRITEBYTECODE" = "1"'
    )
    result = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "sh", built_image, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr[-2000:]


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
