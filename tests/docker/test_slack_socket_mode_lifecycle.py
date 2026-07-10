"""Exact-image Slack Socket Mode lifecycle regression.

The harness executes with the image's own Python, Slack dependencies, and
``SlackAdapter`` as the non-root ``hermes`` user. No Slack network call occurs.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import textwrap


_LIFECYCLE_HARNESS = textwrap.dedent(
    r"""
    import asyncio
    import json
    import os
    from importlib.metadata import version
    from pathlib import Path

    from slack_bolt.app.async_app import AsyncApp
    from slack_sdk.socket_mode.aiohttp import SocketModeClient

    from gateway.config import PlatformConfig
    from plugins.platforms.slack import adapter as slack_adapter_module
    from plugins.platforms.slack.adapter import SlackAdapter


    class ControlledSession:
        def __init__(self, order):
            self.closed = False
            self.order = order

        async def close(self):
            self.closed = True
            self.order.append("session:closed")


    async def owned_task(name, started, cancelling, release, order):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelling.set()
            await release.wait()
            order.append(f"{name}:settled")
            raise


    async def run():
        assert os.geteuid() != 0
        assert Path("/opt/hermes/.install_method").read_text(encoding="utf-8").strip() == "docker"
        assert not os.environ.get("PYTHONPATH")
        expected_sha = os.environ["HERMES_EXPECTED_BUILD_SHA"]
        assert Path("/opt/hermes/.hermes_build_sha").read_text(encoding="utf-8").strip() == expected_sha
        assert version("slack-sdk") == "3.40.1"
        assert version("slack-bolt") == "1.27.0"
        assert version("aiohttp") == "3.14.1"

        baseline = {
            task
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task() and not task.done()
        }
        order = []
        handler = slack_adapter_module.AsyncSocketModeHandler(
            AsyncApp(token="xoxb-image-test-token"),
            "xapp-image-test-token",
        )
        client = handler.client
        assert isinstance(client, SocketModeClient)
        client.message_processor.cancel()
        await asyncio.gather(client.message_processor, return_exceptions=True)
        await client.aiohttp_client_session.close()
        session = ControlledSession(order)
        client.aiohttp_client_session = session

        release = asyncio.Event()
        started = [asyncio.Event() for _ in range(4)]
        cancelling = [asyncio.Event() for _ in range(4)]
        names = (
            "outer",
            "message_processor",
            "current_session_monitor",
            "message_receiver",
        )
        tasks = [
            asyncio.create_task(
                owned_task(name, started[index], cancelling[index], release, order)
            )
            for index, name in enumerate(names)
        ]
        outer, client.message_processor, client.current_session_monitor, client.message_receiver = tasks
        for event in started:
            await event.wait()

        adapter = SlackAdapter(
            PlatformConfig(enabled=True, token="xoxb-image-test-token")
        )
        adapter._handler = handler
        adapter._socket_mode_task = outer
        stop_task = asyncio.create_task(adapter._stop_socket_mode_handler())
        for event in cancelling:
            await asyncio.wait_for(event.wait(), timeout=0.5)

        assert not stop_task.done()
        assert not session.closed
        release.set()
        await asyncio.wait_for(stop_task, timeout=0.5)
        assert session.closed
        assert all(task.done() for task in tasks)
        assert order[-1] == "session:closed"
        assert adapter._handler is None
        assert adapter._socket_mode_task is None

        await asyncio.sleep(0)
        leaked = {
            task
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task()
            and not task.done()
            and task not in baseline
        }
        assert leaked == set()
        print(json.dumps({"lifecycle": "pass", "sha": expected_sha}))


    asyncio.run(run())
    """
)


def _repo_head() -> str:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_slack_socket_mode_lifecycle_uses_exact_immutable_image(
    built_image: str,
) -> None:
    expected_sha = _repo_head()
    inspect = subprocess.run(
        ["docker", "image", "inspect", built_image],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert inspect.returncode == 0, inspect.stderr
    image_config = json.loads(inspect.stdout)[0]["Config"]
    assert image_config["Entrypoint"] == [
        "/init",
        "/opt/hermes/docker/main-wrapper.sh",
    ]
    env_keys = {entry.split("=", 1)[0] for entry in image_config.get("Env") or []}
    assert "PYTHONPATH" not in env_keys

    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--user",
            "hermes",
            "--env",
            f"HERMES_EXPECTED_BUILD_SHA={expected_sha}",
            "--entrypoint",
            "/opt/hermes/.venv/bin/python",
            built_image,
            "-c",
            _LIFECYCLE_HARNESS,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"image lifecycle harness failed ({result.returncode}):\n"
        f"stdout={result.stdout[-2000:]}\nstderr={result.stderr[-2000:]}"
    )
    result_payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert result_payload == {"lifecycle": "pass", "sha": expected_sha}
