"""Real Python/TypeScript socket integration with a synthetic OMP hook source."""

import json
import os
import select
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import venv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux-only skill")
pytest.importorskip("fcntl")
SCRIPTS = (
    Path(__file__).resolve().parents[2]
    / "optional-skills/autonomous-ai-agents/omp-session-supervision/scripts"
)
sys.path.insert(0, str(SCRIPTS))
from omp_supervisor import tui  # noqa: E402


@pytest.fixture(scope="module")
def node():
    executable = shutil.which("node")
    if executable is None:
        pytest.skip("Node 22.6+ is required for TypeScript socket integration")
    probe = subprocess.run(
        [executable, "--experimental-strip-types", "--version"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if probe.returncode:
        pytest.skip("Node with --experimental-strip-types is required")
    return executable


def test_extension_lifecycle_and_socket_regressions(node):
    result = subprocess.run(
        [
            node,
            "--experimental-strip-types",
            "--test",
            str(Path(__file__).with_name("test_omp_session_supervision_extension.mjs")),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def driver_reply(process):
    assert process.stdout is not None
    pending = bytearray()
    deadline = time.monotonic() + 5
    while b"\n" not in pending:
        readable, _, _ = select.select(
            [process.stdout], [], [], max(0, deadline - time.monotonic())
        )
        assert readable, "Timed out waiting for the synthetic hook driver"
        chunk = os.read(process.stdout.fileno(), 4096)
        assert chunk, "Synthetic hook driver closed stdout before replying"
        pending.extend(chunk)
        assert len(pending) <= 4096, "Oversized synthetic hook reply"
    return pending.decode().strip()


def test_installed_cli_observes_installed_extension(node):
    with tempfile.TemporaryDirectory(
        prefix="i", dir=os.environ.get("TMPDIR")
    ) as directory:
        root = Path(directory).resolve()
        installed = root / "installed"
        shutil.copytree(
            SCRIPTS.parent, installed, ignore=shutil.ignore_patterns("__pycache__")
        )
        venv.EnvBuilder(with_pip=False).create(root / "python")
        command = [
            str(root / "python/bin/python"),
            str(installed / "scripts/omp_supervise.py"),
        ]
        env = {
            "PATH": os.defpath,
            "HERMES_HOME": str(root),
            "HERMES_SESSION_PLATFORM": "discord",
            "HERMES_SESSION_KEY": "installed-key",
            "HERMES_SESSION_ID": "installed-generation",
            "HERMES_SESSION_CHAT_ID": "test-chat",
            "HERMES_SESSION_THREAD_ID": "",
        }
        prepared = subprocess.run(
            [
                *command,
                "prepare",
                "--workspace",
                str(root),
                "--tmux-session",
                "installed",
                "--state-root",
                str(root),
            ],
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            check=True,
            timeout=10,
        )
        run = Path(json.loads(prepared.stdout)["run_dir"])
        driver = Path(__file__).parent / "fixtures/omp_tui_extension_host.mjs"
        extension = installed / "scripts/omp_supervisor/tui_extension.ts"
        with subprocess.Popen(
            [node, "--experimental-strip-types", str(driver), str(extension)],
            env={**env, "OMP_HERMES_BINDING_FILE": str(run / "binding.json")},
            cwd=root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ) as process:
            try:
                assert driver_reply(process) == "READY"
                for expected_sequence in (2, 4):
                    for event in (
                        {"type": "agent_start"},
                        {"type": "agent_end", "messages": []},
                    ):
                        process.stdin.write(json.dumps(event) + "\n")
                        process.stdin.flush()
                        assert driver_reply(process) == "ACK"
                    receipt = subprocess.run(
                        [*command, "watch", "--run-dir", str(run), "--timeout", "2"],
                        cwd=root,
                        env=env,
                        text=True,
                        capture_output=True,
                        check=True,
                        timeout=5,
                    )
                    observed = json.loads(receipt.stdout)
                    assert (observed["kind"], observed["seq"]) == (
                        "turn_settled",
                        expected_sequence,
                    )
                process.stdin.write('{"type":"session_shutdown"}\n')
                process.stdin.flush()
                assert driver_reply(process) == "ACK"
                process.stdin.close()
                assert process.wait(timeout=3) == 0
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=3)


@pytest.mark.parametrize("platform", ["discord", "slack"])
def test_socket_bridge_replays_rearms_and_revokes(monkeypatch, platform, node):
    with tempfile.TemporaryDirectory(
        prefix="x", dir=os.environ.get("TMPDIR")
    ) as directory:
        root = Path(directory)
        for key, value in {
            "HERMES_HOME": str(root),
            "HERMES_SESSION_PLATFORM": platform,
            "HERMES_SESSION_KEY": "synthetic-integration-key",
            "HERMES_SESSION_ID": "synthetic-integration-generation",
            "HERMES_SESSION_CHAT_ID": "100",
            "HERMES_SESSION_THREAD_ID": "101",
        }.items():
            monkeypatch.setenv(key, value)
        prepared = tui.prepare(root, "synthetic-integration", root)
        run = Path(prepared["run_dir"])
        driver = Path(__file__).parent / "fixtures" / "omp_tui_extension_host.mjs"
        env = {**os.environ, "OMP_HERMES_BINDING_FILE": str(run / "binding.json")}
        with subprocess.Popen(
            [node, "--experimental-strip-types", str(driver)],
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ) as process:
            try:
                assert driver_reply(process) == "READY"

                def emit(kind, **fields):
                    process.stdin.write(json.dumps({"type": kind, **fields}) + "\n")
                    process.stdin.flush()
                    assert driver_reply(process) == "ACK"

                emit("agent_start")
                emit("agent_end", messages=[])
                first = tui.watch(run, timeout=2)
                assert (first["kind"], first["seq"]) == ("turn_settled", 2)
                expired = tui.watch(run, timeout=2)
                assert (expired["kind"], expired["reason"]) == (
                    "observation_lost",
                    "timeout",
                )
                assert tui.status(run)["cursor"]["terminal"] is False
                connected = threading.Event()
                real_frames = tui._frames

                def witnessed_frames(client, deadline):
                    for message in real_frames(client, deadline):
                        if message["type"] == "hello":
                            connected.set()
                        yield message

                monkeypatch.setattr(tui, "_frames", witnessed_frames)
                with ThreadPoolExecutor(max_workers=1) as executor:
                    following = executor.submit(tui.watch, run, timeout=10)
                    assert connected.wait(timeout=5), (
                        "Expected a real socket subscription"
                    )
                    emit("agent_start")
                    emit("agent_end", messages=[])
                    second = following.result(timeout=5)
                assert (second["kind"], second["seq"]) == ("turn_settled", 4)
                assert first["epoch"] == second["epoch"]
                emit("session_switch", sessionId="synthetic-replacement-session")
                revoked = tui.watch(run, timeout=2)
                assert revoked["kind"] == "session_revoked"
                observed = tui.status(run)
                assert observed["cursor"]["terminal"] is True
                assert observed["journal"]["state"] == "revoked"
                assert (
                    observed["journal"]["omp_session_id"]
                    == "synthetic-cross-language-session"
                )
                emit("session_shutdown")
                process.stdin.close()
                assert process.wait(timeout=3) == 0
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=3)
