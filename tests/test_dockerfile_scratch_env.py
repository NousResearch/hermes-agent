"""The Docker image environment must leave the scratch boot hook free to set the temp vars."""

import shlex
from pathlib import Path

from hermes_constants import apply_scratch_tmp_env

DOCKERFILE = Path(__file__).resolve().parents[1] / "Dockerfile"
ENV_INSTRUCTION = "ENV "


def _dockerfile_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for line in DOCKERFILE.read_text(encoding="utf-8").splitlines():
        if line.startswith(ENV_INSTRUCTION):
            for pair in shlex.split(line[len(ENV_INSTRUCTION):]):
                key, _, value = pair.partition("=")
                env[key] = value
    return env


def test_image_env_lets_scratch_hook_own_temp_vars(tmp_path):
    env = _dockerfile_env()
    env["HERMES_HOME"] = str(tmp_path)
    assert apply_scratch_tmp_env(env) is True
    assert env["TMPDIR"] == str(tmp_path / "cache" / "scratch")
