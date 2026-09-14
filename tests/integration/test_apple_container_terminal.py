"""Opt-in native lifecycle validation for Apple's Container runtime."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import psutil
import pytest


pytestmark = [
    pytest.mark.macos_only,
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("HERMES_RUN_APPLE_CONTAINER_INTEGRATION") != "1",
        reason="set HERMES_RUN_APPLE_CONTAINER_INTEGRATION=1 on macOS 26 ARM64",
    ),
]


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


@pytest.fixture
def native_container_cli():
    from tools.environments import apple_container as apple

    assert apple.is_apple_container_supported_host(), "requires macOS 26+ arm64"
    apple._container_executable = None
    executable = apple.find_container_cli()
    assert executable, "Apple Container CLI is required for explicitly opted-in tests"
    running, detail = apple.container_system_status(executable)
    assert running, f"Apple Container service must already be running: {detail}"
    return executable


def _native_command(executable, *arguments):
    return subprocess.run(
        [executable, *arguments], capture_output=True, text=True, timeout=15,
    )


def _native_present(executable, name):
    result = _native_command(executable, "list", "--all", "--format", "json")
    assert result.returncode == 0, result.stderr
    assert isinstance(json.loads(result.stdout), list), result.stdout
    return _list_json_contains_identity(result.stdout, name)


def _native_wait_absent(executable, name):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if not _native_present(executable, name):
            return
        time.sleep(0.2)
    pytest.fail(f"owner is dead but container still exists: {name}")


def _fixture_process_alive(process):
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _recorded_process(record, prefix):
    created = record.get(f"{prefix}_created")
    if created is None:
        return None
    try:
        process = psutil.Process(record[f"{prefix}_pid"])
        if process.create_time() == created:
            return process
    except psutil.NoSuchProcess:
        pass
    return None


@contextmanager
def _native_owner(executable, root, home, mode):
    root.mkdir(parents=True)
    home.mkdir(parents=True, exist_ok=True)
    child_env = {
        key: os.environ[key]
        for key in ("PATH", "HOME", "TMPDIR", "TZ", "LANG", "LC_ALL")
        if key in os.environ
    }
    child_env.update(
        HERMES_HOME=str(home), HERMES_TEST_ISOLATION=str(home),
        HERMES_DISABLE_LAZY_INSTALLS="1", PYTHONDONTWRITEBYTECODE="1",
    )
    helper = Path(__file__).with_name("apple_container_owner.py")
    started_path = root / "started.json"
    ready_path = root / "ready.json"
    children = {}
    name = None
    log_path = root / "owner.log"

    def collect_client():
        if not started_path.exists():
            return None
        started = json.loads(started_path.read_text(encoding="utf-8"))
        client = _recorded_process(started, "client")
        if client is not None:
            children[client.pid] = client
        return started["name"]

    with log_path.open("wb") as log:
        owner = subprocess.Popen(
            [sys.executable, str(helper), mode, executable, str(root), str(home)],
            env=child_env, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
            close_fds=True, start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 330
            while True:
                name = collect_client() or name
                if ready_path.exists():
                    ready = json.loads(ready_path.read_text(encoding="utf-8"))
                    spectator = _recorded_process(ready, "spectator")
                    assert spectator is not None, "fixture spectator exited before readiness"
                    children[spectator.pid] = spectator
                    break
                assert owner.poll() is None, log_path.read_text(
                    encoding="utf-8", errors="replace"
                )
                assert time.monotonic() < deadline, log_path.read_text(
                    encoding="utf-8", errors="replace"
                )
                time.sleep(0.1)
            yield {**ready, "owner": owner, "spectator": spectator}
        finally:
            if owner.poll() is None:
                owner.kill()
            owner.wait(timeout=10)
            name = collect_client() or name
            # A failed assertion must not leave fixture processes behind.
            # Match recorded creation times as well as PIDs before signaling.
            if ready_path.exists():
                ready = json.loads(ready_path.read_text(encoding="utf-8"))
                spectator = _recorded_process(ready, "spectator")
                if spectator is not None:
                    children[spectator.pid] = spectator
            for process in children.values():
                try:
                    if _fixture_process_alive(process):
                        process.kill()
                    process.wait(timeout=10)
                except psutil.NoSuchProcess:
                    pass
            if name is not None:
                _native_command(executable, "delete", "--force", name)
                assert not _native_present(executable, name), "fixture cleanup failed"


# Killing the owner deliberately reparents its helpers outside pytest's tree.
# Teardown still matches recorded PID + creation time before signaling them.
@pytest.mark.live_system_guard_bypass
def test_native_raw_pipe_owner_sigkill_removes_container(native_container_cli, tmp_path):
    with _native_owner(
        native_container_cli, tmp_path / "raw-owner", tmp_path / "raw-home", "raw"
    ) as victim:
        assert _native_present(native_container_cli, victim["name"])
        assert victim["pipe_inheritable"] is False
        victim["owner"].kill()
        victim["owner"].wait(timeout=10)
        _native_wait_absent(native_container_cli, victim["name"])
        assert _fixture_process_alive(victim["spectator"])


@pytest.mark.live_system_guard_bypass
@pytest.mark.parametrize("same_profile", [True, False])
def test_native_backend_owner_sigkill_preserves_other_owner(
    native_container_cli, tmp_path, same_profile
):
    victim_home = tmp_path / "victim-home"
    survivor_home = victim_home if same_profile else tmp_path / "survivor-home"
    with _native_owner(
        native_container_cli, tmp_path / "victim", victim_home, "backend"
    ) as victim, _native_owner(
        native_container_cli, tmp_path / "survivor", survivor_home, "backend"
    ) as survivor:
        victim["owner"].kill()
        victim["owner"].wait(timeout=10)
        _native_wait_absent(native_container_cli, victim["name"])
        assert victim["pipe_inheritable"] is False
        assert _fixture_process_alive(victim["spectator"])
        assert Path(victim["persistent_file"]).read_text(encoding="utf-8") == "preserved"
        assert survivor["owner"].poll() is None
        result = _native_command(
            native_container_cli, "exec", survivor["name"],
            "bash", "-c", "cat /workspace/owner.txt",
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout == "preserved"


@pytest.mark.live_system_guard_bypass
def test_native_lifetime_client_sigkill_removes_container(native_container_cli, tmp_path):
    with _native_owner(
        native_container_cli, tmp_path / "client-victim", tmp_path / "client-home", "backend"
    ) as victim:
        client = _recorded_process(victim, "client")
        assert client is not None, "fixture lifetime client exited before SIGKILL"
        client.kill()
        # Only the still-live owner can reap this child. A zombie has already
        # closed its descriptors, so verify VM removal, not PID disappearance.
        _native_wait_absent(native_container_cli, victim["name"])
        assert not _fixture_process_alive(client)
        assert victim["owner"].poll() is None
        assert Path(victim["persistent_file"]).read_text(encoding="utf-8") == "preserved"


@pytest.mark.parametrize(
    "extra_args",
    [
        ["python:3.11-slim-bookworm", "-c", "exec sleep infinity"],
        ["-c1"], ["-iv/tmp:/mnt"], ["--network="],
        ["-iv=/tmp:/mnt"], ["-ic=1"],
    ],
)
def test_native_unsafe_extra_args_are_rejected(
    native_container_cli, monkeypatch, tmp_path, extra_args
):
    import tools.credential_files as credential_files
    from tools.environments import apple_container as apple

    monkeypatch.setattr(credential_files, "get_credential_file_mounts", lambda: [])
    monkeypatch.setattr(credential_files, "get_skills_directory_mount", lambda: [])
    monkeypatch.setattr(credential_files, "get_cache_directory_mounts", lambda: [])
    monkeypatch.setattr(apple, "get_sandbox_dir", lambda: tmp_path / "sandboxes")
    env = None
    try:
        with pytest.raises(ValueError):
            env = apple.AppleContainerEnvironment(
                cpu=1, memory=1024, extra_args=extra_args,
            )
    finally:
        if env is not None:
            name = env._container_name
            env.cleanup()
            assert not _native_present(native_container_cli, name)


@pytest.mark.parametrize("volume_flag", ["-v", "-v=", "-iv"])
def test_native_explicit_option_values_preserve_lifetime(
    native_container_cli, monkeypatch, tmp_path, volume_flag
):
    import tools.credential_files as credential_files
    from tools.environments import apple_container as apple

    monkeypatch.setattr(credential_files, "get_credential_file_mounts", lambda: [])
    monkeypatch.setattr(credential_files, "get_skills_directory_mount", lambda: [])
    monkeypatch.setattr(credential_files, "get_cache_directory_mounts", lambda: [])
    monkeypatch.setattr(apple, "get_sandbox_dir", lambda: tmp_path / "sandboxes")
    source = tmp_path / "fixture data"
    source.mkdir()
    (source / "marker").write_text("fixture", encoding="utf-8")
    spec = f"{source}:/workspace/extra:ro"
    volume = [volume_flag + spec] if volume_flag.endswith("=") else [volume_flag, spec]
    env = apple.AppleContainerEnvironment(
        cpu=1, memory=1024,
        extra_args=["--network=none", "-c=1", *volume],
    )
    name = env._container_name
    try:
        result = env.execute("cat /workspace/extra/marker")
        assert result.get("returncode") == 0, result
        assert "fixture" in result.get("output", "")
        client = env._run_process
        assert client is not None and client.stdin is not None
        client.stdin.close()
        _native_wait_absent(native_container_cli, name)
        assert (source / "marker").read_text(encoding="utf-8") == "fixture"
    finally:
        env.cleanup()
        assert not _native_present(native_container_cli, name)
