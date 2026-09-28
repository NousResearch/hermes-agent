"""Disposable Synapse, Linux gateway, and independent Matrix client fixtures."""

from __future__ import annotations

import asyncio
import os
import subprocess
import time
import urllib.request
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import docker
import pytest
from docker import errors as docker_errors
from nio import AsyncClient, LoginResponse, RegisterResponse, RoomCreateResponse
from testcontainers.core import testcontainers_config
from testcontainers.core.container import DockerContainer, Reaper
from testcontainers.core.labels import LABEL_SESSION_ID, SESSION_ID
from testcontainers.core.network import Network

from tests.fakes.fake_llm_provider import FakeLLMServer, Text, write_hermes_home


REPO_ROOT = Path(__file__).resolve().parents[3]
SYNAPSE_IMAGE = "matrixdotorg/synapse:v1.158.0@sha256:5f868df1f5772907c6dbe973a9b69ab530a5d6bb317c011a3788f7ad78eb1292"
RYUK_IMAGE = "testcontainers/ryuk:0.8.1@sha256:bf3f74a47dee0acda89aba4b2fc9c7fdcf994a084db02a2d06566f07baae022e"


@dataclass(frozen=True)
class MatrixAccount:
    user_id: str
    device_id: str
    access_token: str

    def client(self, homeserver: str) -> AsyncClient:
        client = AsyncClient(homeserver, self.user_id)
        client.restore_login(self.user_id, self.device_id, self.access_token)
        return client


@dataclass(frozen=True)
class LiveRoom:
    homeserver: str
    room_id: str
    bot: MatrixAccount
    observer: MatrixAccount


@dataclass(frozen=True)
class LiveGateway:
    container: DockerContainer
    model: FakeLLMServer


@dataclass(frozen=True)
class LinuxNioObserver:
    container: DockerContainer
    account: MatrixAccount

    def run_python(self, code: str) -> str:
        result = self.container.exec(["/opt/hermes/.venv/bin/python", "-c", code])
        output = result.output.decode(errors="replace")
        assert result.exit_code == 0, f"Linux matrix-nio client failed:\n{output}"
        return output


def _wait_for(
    predicate: Callable[[], bool], description: str, *, timeout: float = 60.0,
    details: Callable[[], str] | None = None,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.25)
    suffix = f"\n{details()}" if details is not None else ""
    pytest.fail(f"Timed out waiting for {description}{suffix}")


@pytest.fixture(scope="module")
def docker_engine(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    docker_config = tmp_path_factory.mktemp("matrix-docker-config")
    (docker_config / "config.json").write_text("{}", encoding="utf-8")
    previous_config = os.environ.get("DOCKER_CONFIG")
    os.environ["DOCKER_CONFIG"] = str(docker_config)
    testcontainers_config.ryuk_disabled = False
    testcontainers_config.ryuk_image = RYUK_IMAGE
    client = None
    try:
        client = docker.from_env()
        client.ping()
    except docker_errors.DockerException as exc:
        message = f"Matrix live tests require a running Docker daemon: {exc}"
        if os.environ.get("CI"):
            pytest.fail(message, pytrace=False)
        pytest.skip(message)
    finally:
        if client is not None:
            client.close()
    try:
        yield
    finally:
        if previous_config is None:
            os.environ.pop("DOCKER_CONFIG", None)
        else:
            os.environ["DOCKER_CONFIG"] = previous_config


@pytest.fixture(scope="module")
def gateway_image(docker_engine: None) -> Iterator[str]:
    # Start Ryuk before building so it can remove the image if the worker is killed.
    Reaper.get_instance()
    image = f"hermes-matrix-live:{uuid.uuid4().hex}"
    try:
        result = subprocess.run(
            [
                "docker", "buildx", "build", "--load", "--progress=plain",
                "-f", str(REPO_ROOT / "tests" / "integration" / "matrix_live" / "Dockerfile"),
                "--label", f"{LABEL_SESSION_ID}={SESSION_ID}", "-t", image, str(REPO_ROOT),
            ],
            capture_output=True,
            text=True,
            timeout=1800,
        )
        assert result.returncode == 0, f"Linux gateway image build failed:\n{result.stdout[-6000:]}\n{result.stderr[-6000:]}"

        with DockerContainer(image, entrypoint="/opt/hermes/.venv/bin/python").with_command(
            ["-c", "import olm; import mautrix.crypto"]
        ) as crypto_probe:
            status = crypto_probe.get_wrapped_container().wait(timeout=30)
            assert status["StatusCode"] == 0, (
                "Linux gateway image lacks Matrix crypto support:\n"
                + crypto_probe.get_wrapped_container().logs().decode(errors="replace")
            )
        yield image
    finally:
        client = docker.from_env()
        try:
            client.images.remove(image=image, force=True)
        except docker_errors.ImageNotFound:
            pass
        finally:
            client.close()


@pytest.fixture
def synapse(docker_engine: None) -> Iterator[tuple[DockerContainer, str, Network]]:
    # Start Ryuk before creating the volume so a killed worker cannot leave it behind.
    Reaper.get_instance()
    client = docker.from_env()
    volume = client.volumes.create(
        name=f"hermes-matrix-live-{uuid.uuid4().hex}",
        labels={LABEL_SESSION_ID: SESSION_ID},
    )
    try:
        with DockerContainer(SYNAPSE_IMAGE, command="generate").with_env(
            "SYNAPSE_SERVER_NAME", "matrix.test"
        ).with_env("SYNAPSE_REPORT_STATS", "no").with_volume_mapping(volume.name, "/data", "rw") as generator:
            exit_state = generator.get_wrapped_container().wait(timeout=90)
            assert exit_state["StatusCode"] == 0, generator.get_wrapped_container().logs().decode(errors="replace")

        with DockerContainer(
            SYNAPSE_IMAGE,
            entrypoint="/bin/sh",
        ).with_command([
            "-c",
            "printf '\\nenable_registration: true\\nenable_registration_without_verification: true\\n' >> /data/homeserver.yaml",
        ]).with_volume_mapping(volume.name, "/data", "rw") as configure:
            exit_state = configure.get_wrapped_container().wait(timeout=30)
            assert exit_state["StatusCode"] == 0, configure.get_wrapped_container().logs().decode(errors="replace")

        with Network() as network:
            with DockerContainer(SYNAPSE_IMAGE, network=network, network_aliases=["synapse"]).with_volume_mapping(
                volume.name, "/data", "rw"
            ).with_exposed_ports(8008) as container:
                url = f"http://{container.get_container_host_ip()}:{container.get_exposed_port(8008)}"

                def ready() -> bool:
                    try:
                        with urllib.request.urlopen(f"{url}/_matrix/client/versions", timeout=2) as response:
                            return response.status == 200
                    except OSError:
                        return False

                _wait_for(ready, "Synapse client API", timeout=120)
                yield container, url, network
    finally:
        volume.remove(force=True)
        client.close()


async def _register(url: str, localpart: str) -> MatrixAccount:
    client = AsyncClient(url, f"@{localpart}:matrix.test")
    try:
        registration = await client.register(localpart, "matrix-test-password", device_name=f"{localpart}-device")
        assert isinstance(registration, RegisterResponse), registration
        login = await client.login("matrix-test-password")
        assert isinstance(login, LoginResponse), login
        return MatrixAccount(login.user_id, login.device_id, login.access_token)
    finally:
        await client.close()


@pytest.fixture
def live_room(synapse: tuple[DockerContainer, str, Network]) -> LiveRoom:
    _, url, _ = synapse

    async def create() -> LiveRoom:
        bot = await _register(url, "hermes")
        alice = await _register(url, "alice")
        client = alice.client(url)
        try:
            response = await client.room_create(name="Matrix live test", invite=[bot.user_id])
            assert isinstance(response, RoomCreateResponse), response
            return LiveRoom(url, response.room_id, bot, alice)
        finally:
            await client.close()

    return asyncio.run(create())


@pytest.fixture
def linux_nio_observer(
    gateway_image: str,
    synapse: tuple[DockerContainer, str, Network],
    live_room: LiveRoom,
) -> Iterator[LinuxNioObserver]:
    _, _, network = synapse
    with DockerContainer(
        gateway_image,
        network=network,
        entrypoint="/bin/sleep",
        command="infinity",
    ).with_volume_mapping(
        REPO_ROOT / "tests" / "integration" / "matrix_live", "/matrix_live", "ro"
    ).with_env("PYTHONPATH", "/matrix_live").with_env(
        "NIO_HOMESERVER", "http://synapse:8008"
    ).with_env("NIO_USER_ID", live_room.observer.user_id).with_env(
        "NIO_DEVICE_ID", live_room.observer.device_id
    ).with_env("NIO_ACCESS_TOKEN", live_room.observer.access_token).with_env(
        "NIO_STORE_PATH", "/opt/data/matrix-nio-store"
    ) as container:
        observer = LinuxNioObserver(container, live_room.observer)
        observer.run_python(
            "from client import open_encrypted_client; "
            "client = open_encrypted_client(); "
            "assert client.olm is not None; "
            "print(client.user_id, client.device_id)"
        )
        yield observer


@pytest.fixture
def gateway(
    tmp_path: Path,
    gateway_image: str,
    synapse: tuple[DockerContainer, str, Network],
    live_room: LiveRoom,
    request: pytest.FixtureRequest,
) -> Iterator[LiveGateway]:
    _, _, network = synapse
    room_id = live_room.room_id
    context_pause = getattr(request, "param", None) in {"pause-context", "pause-image-context"}
    native_images = getattr(request, "param", None) == "pause-image-context"
    home = tmp_path / "hermes"
    home.mkdir(mode=0o777)
    with FakeLLMServer([Text("Matrix live reply")], bind_host="0.0.0.0") as model:
        write_hermes_home(
            home,
            f"http://host.docker.internal:{model.port}/v1",
            extra_config=(
                ("  image_input_mode: native\n" if native_images else "")
                + "platforms:\n  matrix:\n    enabled: true\nupdates:\n  check: false\n"
                + ("plugins:\n  enabled:\n    - matrix-live-context\n"
                   if context_pause else "")
            ),
        )
        if native_images:
            config_path = home / "config.yaml"
            config_path.write_text(
                config_path.read_text(encoding="utf-8").replace(
                    "model:\n", "model:\n  supports_vision: true\n", 1,
                ),
                encoding="utf-8",
            )
        with (home / ".env").open("a", encoding="utf-8") as stream:
            stream.write(
                "MATRIX_HOMESERVER=http://synapse:8008\n"
                f"MATRIX_ACCESS_TOKEN={live_room.bot.access_token}\n"
                f"MATRIX_ALLOWED_USERS={live_room.observer.user_id}\n"
                f"MATRIX_HOME_ROOM={room_id}\n"
                "MATRIX_E2EE_MODE=optional\nMATRIX_REACTIONS=false\nMATRIX_AUTO_THREAD=false\n"
            )
        if context_pause:
            plugin = home / "plugins" / "matrix-live-context"
            plugin.mkdir(parents=True)
            (plugin / "plugin.yaml").write_text(
                "name: matrix-live-context\nversion: 1.0.0\ndescription: Matrix live context pause\n",
                encoding="utf-8",
            )
            (plugin / "__init__.py").write_text(
                "import asyncio\n"
                "from agent.context_references import ContextReferenceProvider\n"
                "from hermes_constants import get_hermes_home\n"
                "class ContextPause(ContextReferenceProvider):\n"
                "    prefix = 'matrix-live'\n"
                "    async def autocomplete(self, query, *, limit=10):\n"
                "        return []\n"
                "    async def expand(self, target):\n"
                "        (get_hermes_home() / 'context-started').write_text('started', encoding='utf-8')\n"
                "        while not (get_hermes_home() / 'context-release').exists():\n"
                "            await asyncio.sleep(0.01)\n"
                "        return 'Live enrichment completed'\n"
                "def register(ctx):\n"
                "    ctx.register_context_reference(ContextPause())\n",
                encoding="utf-8",
            )
        if native_images:
            with (plugin / "__init__.py").open("a", encoding="utf-8") as stream:
                stream.write(
                    "from plugins.platforms.matrix.reply_context import MatrixEventContextCache\n"
                    "original_store = MatrixEventContextCache.store\n"
                    "def observed_store(self, room_id, event_id, entry):\n"
                    "    result = original_store(self, room_id, event_id, entry)\n"
                    "    expected = get_hermes_home() / 'expected-media-change'\n"
                    "    if expected.exists() and expected.read_text(encoding='utf-8') == event_id:\n"
                    "        if (get_hermes_home() / 'evict-media-state').exists():\n"
                    "            from plugins.platforms.matrix.reply_context import MatrixEventContext\n"
                    "            for index in range(self.max_entries):\n"
                    "                original_store(self, room_id, f'$unrelated{index}', MatrixEventContext('', 'unrelated'))\n"
                    "            import gc\n"
                    "            gc.collect()\n"
                    "        (get_hermes_home() / 'media-change-observed').write_text(event_id, encoding='utf-8')\n"
                    "    return result\n"
                    "MatrixEventContextCache.store = observed_store\n"
                    "from gateway.platforms.base import BasePlatformAdapter\n"
                    "original_init = BasePlatformAdapter.__init__\n"
                    "def observed_init(self, *args, **kwargs):\n"
                    "    original_init(self, *args, **kwargs)\n"
                    "    if self.platform.value != 'matrix':\n"
                    "        return\n"
                    "    original_message = self._on_room_message\n"
                    "    async def observed_message(event):\n"
                    "        await original_message(event)\n"
                    "        expected = get_hermes_home() / 'expected-media-change'\n"
                    "        from plugins.platforms.matrix.effective_event import event_content\n"
                    "        relation = event_content(event).get('m.relates_to', {})\n"
                    "        if expected.exists() and relation.get('event_id') == expected.read_text(encoding='utf-8'):\n"
                    "            (get_hermes_home() / 'media-change-observed').write_text(relation['event_id'], encoding='utf-8')\n"
                    "    self._on_room_message = observed_message\n"
                    "BasePlatformAdapter.__init__ = observed_init\n"
                )
        home.chmod(0o777)

        with DockerContainer(
            gateway_image,
            network=network,
            entrypoint="/opt/hermes/.venv/bin/python",
            user="10000:10000",
            working_dir="/opt/hermes",
            extra_hosts={"host.docker.internal": "host-gateway"},
        ).with_command("-m hermes_cli.main gateway run").with_volume_mapping(home, "/opt/data", "rw") as container:
            def connected() -> bool:
                output = container.get_wrapped_container().logs().decode(errors="replace")
                gateway_log = home / "logs" / "gateway.log"
                if gateway_log.exists() and f"Matrix: joined {room_id}" in gateway_log.read_text(errors="replace"):
                    return True
                if container.get_wrapped_container().status == "exited":
                    pytest.fail(f"Gateway exited before Matrix connected:\n{output}")
                return False

            _wait_for(
                connected, "Matrix gateway initial sync", timeout=120,
                details=lambda: container.get_wrapped_container().logs().decode(errors="replace")[-6000:],
            )
            yield LiveGateway(container, model)
