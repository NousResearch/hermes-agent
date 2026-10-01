"""Bundled transcription weights must load offline as the runtime user."""
import subprocess

from tests.docker.conftest import docker_exec, wait_for_container_ready


def test_boot_seeds_whisper_for_runtime_user(built_image, container_name):
    subprocess.run(
        ["docker", "run", "-d", "--name", container_name,
         "--tmpfs", "/opt/data", "-e", "HOME=/root",
         built_image, "sleep", "infinity"],
        check=True, capture_output=True, timeout=60,
    )
    wait_for_container_ready(container_name, deadline_s=180)
    result = docker_exec(
        container_name, "env", "HOME=/opt/data", "HF_HUB_OFFLINE=1",
        "/opt/hermes/.venv/bin/python", "-c",
        "from faster_whisper import WhisperModel; "
        "WhisperModel('base', device='cpu', compute_type='int8', local_files_only=True)",
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
