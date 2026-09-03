"""Opt-in native lifecycle validation for Apple's Container runtime."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess

import pytest


pytestmark = pytest.mark.skipif(
    os.getenv("HERMES_RUN_APPLE_CONTAINER_INTEGRATION") != "1",
    reason="set HERMES_RUN_APPLE_CONTAINER_INTEGRATION=1 on macOS 26 ARM64",
)


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _terminal_result(raw: str) -> dict:
    result = json.loads(raw)
    assert result.get("exit_code") == 0, result
    return result


def _list_json_contains_identity(raw: str, identity: str) -> bool:
    """Check parsed native list JSON for an exact container identity."""
    payload = json.loads(raw or "[]")

    def contains(value) -> bool:
        if isinstance(value, str):
            return value == identity
        if isinstance(value, dict):
            return any(contains(item) for item in value.values())
        if isinstance(value, list):
            return any(contains(item) for item in value)
        return False

    return contains(payload)


def test_native_apple_container_cross_tool_lifecycle(monkeypatch, tmp_path):
    if platform.system() != "Darwin" or platform.machine().lower() != "arm64":
        pytest.skip("Apple Container native test requires Darwin arm64")
    try:
        major = int(platform.mac_ver()[0].split(".", 1)[0])
    except (ValueError, IndexError):
        pytest.skip("could not confirm the required macOS 26+ version")
    if major < 26:
        pytest.skip(f"Apple Container native test requires macOS 26+ (found {major})")

    import tools.credential_files as credential_files
    import tools.file_tools as file_tools
    import tools.terminal_tool as terminal
    from tools.code_execution_tool import execute_code
    from tools.environments import apple_container

    apple_container._container_executable = None
    executable = apple_container.find_container_cli()
    if not executable:
        pytest.skip("Apple Container CLI is not installed or discoverable")
    running, detail = apple_container.container_system_status(executable)
    if not running:
        pytest.skip(
            "Apple Container system is not running; run `container system start` manually "
            f"before the opt-in test (status: {detail or 'unknown'})"
        )

    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    credential = hermes_home / "native-readonly-token.txt"
    credential.write_bytes(b"native-readonly-fixture\n")
    before_hash = _sha256(credential)
    print(f"credential fixture SHA-256 before: {before_hash}")
    (hermes_home / "config.yaml").write_text(
        "terminal:\n"
        "  backend: apple_container\n"
        "  credential_files:\n"
        "    - native-readonly-token.txt\n",
        encoding="utf-8",
    )

    task_id = f"apple-native-{os.getpid()}-{tmp_path.name}"
    environment = None
    container_name = None
    container_names = []
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("TERMINAL_ENV", "apple_container")
    monkeypatch.setenv("TERMINAL_APPLE_CONTAINER_IMAGE", "python:3.11-slim-bookworm")
    monkeypatch.setenv("TERMINAL_APPLE_CONTAINER_VOLUMES", "[]")
    monkeypatch.setenv("TERMINAL_CONTAINER_CPU", "1")
    monkeypatch.setenv("TERMINAL_CONTAINER_MEMORY", "1024")
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "true")
    monkeypatch.setattr(terminal, "_terminal_config_bridge_attempted", True)
    credential_files._config_files = None
    terminal.register_task_env_overrides(
        task_id, {"apple_container_image": "python:3.11-slim-bookworm"}
    )

    try:
        identity_raw = terminal.terminal_tool(
            command="uname -s && uname -m", task_id=task_id
        )
        environment = terminal.get_active_env(task_id)
        assert isinstance(environment, apple_container.AppleContainerEnvironment)
        container_name = environment._container_name
        assert container_name and container_name.startswith("hermes-")
        container_names.append(container_name)
        identity = _terminal_result(identity_raw)["output"]
        assert "Linux" in identity
        assert any(machine in identity for machine in ("aarch64", "arm64"))

        _terminal_result(
            terminal.terminal_tool(
                command="printf 'from-terminal' > /workspace/shared.txt",
                task_id=task_id,
            )
        )
        read_result = json.loads(
            file_tools.read_file_tool("/workspace/shared.txt", task_id=task_id)
        )
        assert "from-terminal" in read_result["content"]

        write_result = json.loads(
            file_tools.write_file_tool(
                "/workspace/shared.txt", "from-file-tool", task_id=task_id
            )
        )
        assert not write_result.get("error"), write_result

        execute_result = json.loads(
            execute_code(
                "print(open('/workspace/shared.txt', encoding='utf-8').read())",
                task_id=task_id,
            )
        )
        assert execute_result["status"] == "success", execute_result
        assert "from-file-tool" in execute_result["output"]

        readonly_result = json.loads(
            terminal.terminal_tool(
                command=(
                    "printf 'changed' > /root/.hermes/native-readonly-token.txt"
                ),
                task_id=task_id,
            )
        )
        assert readonly_result.get("exit_code") != 0, readonly_result
        assert credential.read_bytes() == b"native-readonly-fixture\n"
    finally:
        if environment is None:
            environment = terminal.get_active_env(task_id)
            if isinstance(environment, apple_container.AppleContainerEnvironment):
                container_name = environment._container_name
        if environment is not None:
            environment.cleanup()
        terminal._active_environments.pop(task_id, None)
        terminal._last_activity.pop(task_id, None)
        terminal.clear_task_env_overrides(task_id)
        file_tools.clear_file_ops_cache(task_id)

    # Persistent root/workspace directories are reused for the same task. A
    # second lifecycle verifies credential exposure leaves no stale container
    # state (for example, symlinks) that prevents a restart.
    restart_environment = None
    try:
        restart_environment = apple_container.AppleContainerEnvironment(
            image="python:3.11-slim-bookworm",
            cpu=1,
            memory=1024,
            persistent_filesystem=True,
            task_id=task_id,
        )
        restart_name = restart_environment._container_name
        assert restart_name and restart_name.startswith("hermes-")
        container_names.append(restart_name)
        restart_read = restart_environment.execute(
            "cat /root/.hermes/native-readonly-token.txt", cwd="/"
        )
        assert "native-readonly-fixture" in restart_read.get("output", "")
    finally:
        if restart_environment is not None:
            restart_environment.cleanup()

    after_hash = _sha256(credential)
    print(f"credential fixture SHA-256 after:  {after_hash}")
    assert after_hash == before_hash

    listing = subprocess.run(
        [executable, "list", "--all", "--format", "json"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert listing.returncode == 0, listing.stderr
    assert container_names
    for name in container_names:
        assert not _list_json_contains_identity(listing.stdout, name)
