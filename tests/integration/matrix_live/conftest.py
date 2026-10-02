"""Disposable Synapse, Linux gateway, and independent Matrix client fixtures."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import shutil
import socket
import subprocess
import time
import urllib.request
import uuid
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import docker
import pytest
from docker import errors as docker_errors
from docker.context import ContextAPI
from docker.context.config import get_current_context_name
from nio import (
    AsyncClient,
    AsyncClientConfig,
    LoginResponse,
    RegisterResponse,
    RoomCreateResponse,
)
from testcontainers.core import testcontainers_config
from testcontainers.core.container import DockerContainer, Reaper
from testcontainers.core.labels import LABEL_SESSION_ID, SESSION_ID
from testcontainers.core.network import Network

from hermes_platform.host import facts
from tests.fakes.fake_llm_provider import FakeLLMServer, Response, Text, write_hermes_home
from tests.integration.matrix_live.image_build import REPO_ROOT, build_command


PREBUILT_IMAGE_VARIABLE = "HERMES_TEST_MATRIX_GATEWAY_IMAGE"
SYNAPSE_IMAGE = "matrixdotorg/synapse:v1.158.0@sha256:5f868df1f5772907c6dbe973a9b69ab530a5d6bb317c011a3788f7ad78eb1292"
RYUK_IMAGE = "testcontainers/ryuk:0.8.1@sha256:bf3f74a47dee0acda89aba4b2fc9c7fdcf994a084db02a2d06566f07baae022e"


@dataclass(frozen=True)
class MatrixAccount:
    user_id: str
    device_id: str
    access_token: str = field(repr=False)

    def client(self, homeserver: str) -> AsyncClient:
        client = AsyncClient(
            homeserver,
            self.user_id,
            config=AsyncClientConfig(
                request_timeout=15, max_limit_exceeded=0, max_timeouts=0
            ),
        )
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
    home: Path

    def restart(self, *, wait_for_checkpoint: bool = True) -> None:
        gateway_log = self.home / "logs" / "gateway.log"
        log_offset = len(gateway_log.read_text(encoding="utf-8"))
        self.container.get_wrapped_container().restart(timeout=5)
        if not wait_for_checkpoint:
            return

        def connected() -> bool:
            log = gateway_log.read_text(encoding="utf-8")[log_offset:]
            return (
                "Matrix: connected after initial dispatch checkpoint" in log
                and "Press Ctrl+C to stop" in log
            )

        _wait_for(
            connected,
            "Matrix gateway sync and startup restoration after restart",
            timeout=30,
            details=lambda: self.container.get_wrapped_container()
            .logs()
            .decode(errors="replace")[-6000:],
        )

    def log_tail(self, lines: int = 200) -> str:
        path = self.home / "logs" / "gateway.log"
        if not path.exists():
            return f"{path} does not exist"
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])


@dataclass(frozen=True)
class GatewaySettings:
    reply: str = "Matrix live reply"
    max_message_length: int | None = None
    mode: str | None = None
    reply_to_mode: str | None = None


@dataclass(frozen=True)
class LinuxNioObserver:
    container: DockerContainer
    account: MatrixAccount

    def run_python(self, code: str) -> str:
        result = self.container.exec(["/opt/hermes/.venv/bin/python", "-c", code])
        output = result.output.decode(errors="replace")
        assert result.exit_code == 0, f"Linux matrix-nio client failed:\n{output}"
        return output


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None],
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    report = yield
    gateways = [value for value in getattr(item, "funcargs", {}).values() if isinstance(value, LiveGateway)]
    if report.when == "call" and report.failed:
        report.sections.extend(("gateway.log", gateway.log_tail()) for gateway in gateways)
    return report


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
    with _docker_connection_scope(tmp_path_factory.mktemp("matrix-docker-config")):
        yield


@contextmanager
def _docker_connection_scope(docker_config: Path) -> Iterator[None]:
    with pytest.MonkeyPatch.context() as environment:
        client = None
        docker_host = "the selected Docker context"
        try:
            docker_host = _selected_docker_host()
            _write_isolated_docker_config(docker_config)
            environment.setenv("DOCKER_HOST", docker_host)
            environment.delenv("DOCKER_CONTEXT", raising=False)
            environment.setenv("DOCKER_CONFIG", str(docker_config))
            if facts.os_family() == "darwin":
                # The daemon resolves Ryuk's socket mount inside its Linux VM, where the socket is /var/run/docker.sock.
                environment.setenv("TESTCONTAINERS_DOCKER_SOCKET_OVERRIDE", "/var/run/docker.sock")
            environment.setattr(testcontainers_config, "ryuk_disabled", False)
            environment.setattr(testcontainers_config, "ryuk_image", RYUK_IMAGE)
            client = docker.from_env()
            client.ping()
        except docker_errors.DockerException as exc:
            message = f"Matrix live tests require a running Docker daemon at {docker_host}: {exc}"
            if os.environ.get("CI"):
                pytest.fail(message, pytrace=False)
            pytest.skip(message)
        finally:
            if client is not None:
                client.close()
        yield


def _selected_docker_host() -> str:
    if host := os.environ.get("DOCKER_HOST"):
        return host
    name = os.environ.get("DOCKER_CONTEXT") or get_current_context_name()
    context = ContextAPI.get_context(name)
    if context is None:
        raise docker_errors.ContextNotFound(name)
    return context.Host


def _write_isolated_docker_config(destination: Path) -> None:
    source = Path(os.environ.get("DOCKER_CONFIG") or Path.home() / ".docker")
    source_file = source / "config.json"
    settings = json.loads(source_file.read_text(encoding="utf-8")) if source_file.is_file() else {}
    plugin_dirs = [str(source / "cli-plugins"), *settings.get("cliPluginsExtraDirs", [])]
    (destination / "config.json").write_text(json.dumps({"cliPluginsExtraDirs": plugin_dirs}), encoding="utf-8")


class _ImageStore(Protocol):
    def get(self, tag: str) -> object: ...

    def remove(self, image: str, force: bool) -> None: ...


@contextmanager
def _gateway_image_tag(images: _ImageStore, prebuilt: str | None, build: Callable[[str], None]) -> Iterator[str]:
    if prebuilt:
        try:
            images.get(prebuilt)
        except docker_errors.ImageNotFound:
            pytest.fail(f"{PREBUILT_IMAGE_VARIABLE}={prebuilt} is not loaded in the Docker daemon", pytrace=False)
        yield prebuilt
        return

    image = f"hermes-matrix-live:{uuid.uuid4().hex}"
    try:
        build(image)
        yield image
    finally:
        try:
            images.remove(image=image, force=True)
        except docker_errors.ImageNotFound:
            pass


def _build_gateway_image(tag: str) -> None:
    result = subprocess.run(build_command(tag, {LABEL_SESSION_ID: SESSION_ID}), capture_output=True, text=True)
    assert result.returncode == 0, f"Linux gateway image build failed:\n{result.stdout[-6000:]}\n{result.stderr[-6000:]}"


@pytest.fixture(scope="module")
def gateway_image(docker_engine: None) -> Iterator[str]:
    # Start Ryuk before building so it can remove the image if the worker is killed.
    Reaper.get_instance()
    client = docker.from_env()
    try:
        with _gateway_image_tag(client.images, os.environ.get(PREBUILT_IMAGE_VARIABLE), _build_gateway_image) as image:
            yield image
    finally:
        client.close()


@contextmanager
def _synapse_server(*, extra_config: str = "") -> Iterator[tuple[DockerContainer, str, Network]]:
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

        with (
            DockerContainer(
                SYNAPSE_IMAGE,
                entrypoint="/bin/sh",
            )
            .with_command([
                "-c",
                "printf '%s' \"$1\" >> /data/homeserver.yaml",
                "matrix-test-config",
                "\nenable_registration: true\nenable_registration_without_verification: true\n"
                "rc_message:\n  per_second: 100\n  burst_count: 100\n" + extra_config,
            ])
            .with_volume_mapping(volume.name, "/data", "rw") as configure
        ):
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


@pytest.fixture
def synapse(docker_engine: None) -> Iterator[tuple[DockerContainer, str, Network]]:
    with _synapse_server() as server:
        yield server


async def _register(url: str, localpart: str, *, unique: bool = False) -> MatrixAccount:
    if unique:
        localpart = f"{localpart}-{uuid.uuid4().hex}"

    client = AsyncClient(
        url,
        f"@{localpart}:matrix.test",
        config=AsyncClientConfig(
            request_timeout=15, max_limit_exceeded=0, max_timeouts=0
        ),
    )
    try:
        registration = await client.register(localpart, "matrix-test-password", device_name=f"{localpart}-device")
        assert isinstance(registration, RegisterResponse), registration
        login = await client.login("matrix-test-password")
        assert isinstance(login, LoginResponse), login
        return MatrixAccount(login.user_id, login.device_id, login.access_token)
    finally:
        await client.close()


def _create_live_room(synapse: tuple[DockerContainer, str, Network], *, unique_accounts: bool = False) -> LiveRoom:
    _, url, _ = synapse

    async def create() -> LiveRoom:
        bot = await _register(url, "hermes", unique=unique_accounts)
        alice = await _register(url, "alice", unique=unique_accounts)
        client = alice.client(url)
        try:
            response = await client.room_create(name="Matrix live test", invite=[bot.user_id])
            assert isinstance(response, RoomCreateResponse), response
            return LiveRoom(url, response.room_id, bot, alice)
        finally:
            await client.close()

    return asyncio.run(create())


@pytest.fixture
def live_room(synapse: tuple[DockerContainer, str, Network]) -> LiveRoom:
    return _create_live_room(synapse)


@pytest.fixture
def linux_nio_observer(
    gateway_image: str,
    synapse: tuple[DockerContainer, str, Network],
    live_room: LiveRoom,
    gateway: LiveGateway,
) -> Iterator[LinuxNioObserver]:
    _, _, network = synapse
    with DockerContainer(
        gateway_image,
        network=network,
        entrypoint="/bin/sleep",
        command="infinity",
    ).with_volume_mapping(
        REPO_ROOT / "tests" / "integration" / "matrix_live", "/matrix_live", "ro"
    ).with_volume_mapping(
        gateway.home, "/gateway-barriers", "rw"
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


@dataclass(frozen=True)
class HostRoute:
    bind_host: str
    container_address: str


def _host_route(network: Network) -> HostRoute:
    """Choose where a server on this host listens and what ``host.docker.internal`` resolves to.

    With a daemon on this host (Linux), the network's bridge gateway is a local address. A daemon
    in a VM (Docker Desktop, OrbStack) does not expose that address to the host, and its
    ``host-gateway`` forwards to the host's loopback interface instead.
    """
    client = docker.from_env()
    try:
        configs = client.networks.get(network.id).attrs["IPAM"]["Config"]
    finally:
        client.close()
    gateway = next(
        config["Gateway"] for config in configs
        if config.get("Gateway") and ipaddress.ip_address(config["Gateway"]).version == 4
    )
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind((gateway, 0))
    except OSError:
        return HostRoute("127.0.0.1", "host-gateway")
    return HostRoute(gateway, gateway)


def _host_user() -> str:
    """Return the container user that lets this user delete what a container writes to a bind mount."""
    if facts.os_family() == "win32":
        return "10000:10000"
    return f"{os.getuid()}:{os.getgid()}"


def _gateway_ready(log: str, room_id: str) -> bool:
    """Whether the gateway has joined the room and dispatches its messages directly.

    While startup restore runs, the gateway queues inbound messages and replays them when it
    finishes. The gateway logs "Press Ctrl+C to stop" after startup restore has finished.
    """
    return f"Matrix: joined {room_id}" in log and "Press Ctrl+C to stop" in log


@pytest.fixture
def gateway_extra_config() -> str:
    return ""


@pytest.fixture
def gateway_config() -> str:
    return "platforms:\n  matrix:\n    enabled: true\nupdates:\n  check: false\n"


@pytest.fixture
def gateway_home_setup() -> Callable[[Path], None]:
    return lambda _home: None


@pytest.fixture
def gateway(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    gateway_extra_config: str,
    gateway_config: str,
    gateway_image: str,
    synapse: tuple[DockerContainer, str, Network],
    live_room: LiveRoom,
    gateway_home_setup: Callable[[Path], None],
) -> Iterator[LiveGateway]:
    param = getattr(request, "param", GatewaySettings())
    settings = GatewaySettings(mode=param) if isinstance(param, str) else param
    _, _, network = synapse
    room_id = live_room.room_id
    mode = settings.mode
    resolution_pause = mode == "pause-resolution"
    context_pause = mode in {"pause-context", "pause-image-context", "pause-image-conversion", "pause-queued-context"}
    native_images = mode in {"pause-image-context", "pause-image-conversion", "image-packs"}
    home = tmp_path / "hermes"
    home.mkdir()
    route = _host_route(network)
    if settings.reply_to_mode is not None:
        gateway_config = gateway_config.replace(
            "    enabled: true\n",
            "    enabled: true\n" + f'    reply_to_mode: "{settings.reply_to_mode}"\n',
            1,
        )
    script = [] if mode == "inspection" else [Text(settings.reply)]
    with FakeLLMServer(
        script, bind_host=route.bind_host, default_text=settings.reply if mode == "inspection" else "ok",
    ) as model:
        write_hermes_home(
            home,
            f"http://host.docker.internal:{model.port}/v1",
            extra_config=(
                ("  image_input_mode: native\n" if native_images else "")
                + gateway_config.replace(
                    "    enabled: true\n",
                    "    enabled: true\n"
                    + ("    thread_require_mention: true\n" if mode == "pause-context" or resolution_pause else "")
                    + (f"    free_response_rooms:\n      - {room_id!r}\n" if resolution_pause else ""),
                    1,
                )
                + ("auxiliary:\n  background_review:\n    enabled: false\n  title_generation:\n    model_upgrade_enabled: false\n"
                   if mode in {"inspection", "pause-image-context", "image-packs"} else "")
                + ("display:\n  busy_input_mode: queue\n  busy_ack_enabled: false\n"
                   if mode == "pause-queued-context" else "")
                + ("plugins:\n  enabled:\n    - matrix-live-context\n"
                   if context_pause else "")
                + ("plugins:\n  enabled:\n    - matrix-live-resolution\n" if resolution_pause else "")
                + gateway_extra_config
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
            if settings.max_message_length is not None:
                stream.write(f"MATRIX_MAX_MESSAGE_LENGTH={settings.max_message_length}\n")
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
                "def signal(name, value):\n"
                "    path = get_hermes_home() / name\n"
                "    partial = path.with_name(f'.{name}.partial')\n"
                "    partial.write_text(value, encoding='utf-8')\n"
                "    partial.replace(path)\n"
                "class ContextPause(ContextReferenceProvider):\n"
                "    prefix = 'matrix-live'\n"
                "    async def autocomplete(self, query, *, limit=10):\n"
                "        return []\n"
                "    async def expand(self, target):\n"
                "        phase = 'queued-' if target == 'queued' else ''\n"
                "        signal(f'{phase}context-started', 'started')\n"
                "        while not (get_hermes_home() / f'{phase}context-release').exists():\n"
                "            request = get_hermes_home() / 'read-effective-event'\n"
                "            done = get_hermes_home() / 'effective-event-read'\n"
                "            if request.exists() and not done.exists():\n"
                "                import json\n"
                "                from plugins.platforms.matrix.read_context import read_matrix_context\n"
                "                target = json.loads(request.read_text(encoding='utf-8'))\n"
                "                result = await read_matrix_context(matrix_adapter, 'event', target['room'], target['event'], 1, requester=target['sender'])\n"
                "                signal('effective-event-read', json.dumps(result))\n"
                "            await asyncio.sleep(0.01)\n"
                "        return 'Live enrichment completed'\n"
                "def register(ctx):\n"
                "    ctx.register_context_reference(ContextPause())\n",
                encoding="utf-8",
            )
        if context_pause:
            with (plugin / "__init__.py").open("a", encoding="utf-8") as stream:
                stream.write(
                    "from plugins.platforms.matrix.reply_context import MatrixEventContextCache\n"
                    "original_store = MatrixEventContextCache.store\n"
                    "def observed_store(self, room_id, event_id, entry):\n"
                    "    result = original_store(self, room_id, event_id, entry)\n"
                    "    if entry.redacted:\n"
                    "        with (get_hermes_home() / 'redacted-events-observed').open('a', encoding='utf-8') as stream:\n"
                    "            stream.write(event_id + '\\n')\n"
                    "    expected = get_hermes_home() / 'expected-media-change'\n"
                    "    # A typed edit stores this placeholder and then fetches the target again;\n"
                    "    # observed_message signals that edit once the fetch has finished.\n"
                    "    refetching = entry.state_error == 'event content changed'\n"
                    "    if expected.exists() and expected.read_text(encoding='utf-8') == event_id and not refetching:\n"
                    "        if (get_hermes_home() / 'evict-media-state').exists():\n"
                    "            from plugins.platforms.matrix.reply_context import MatrixEventContext\n"
                    "            for index in range(self.max_entries):\n"
                    "                original_store(self, room_id, f'$unrelated{index}', MatrixEventContext('', 'unrelated'))\n"
                    "            import gc\n"
                    "            gc.collect()\n"
                    "        signal('media-change-observed', event_id)\n"
                    "    return result\n"
                    "MatrixEventContextCache.store = observed_store\n"
                    "from gateway.platforms.base import BasePlatformAdapter\n"
                    "original_init = BasePlatformAdapter.__init__\n"
                    "def observed_init(self, *args, **kwargs):\n"
                    "    original_init(self, *args, **kwargs)\n"
                    "    if self.platform.value != 'matrix':\n"
                    "        return\n"
                    "    global matrix_adapter\n"
                    "    matrix_adapter = self\n"
                    "    original_message = self._on_room_message\n"
                    "    async def observed_message(event):\n"
                    "        from plugins.platforms.matrix.effective_event import event_content\n"
                    "        queued = 'queued question' in str(event_content(event).get('body', ''))\n"
                    "        if queued:\n"
                    "            self._event_context_cache._entries.clear()\n"
                    "            import gc\n"
                    "            gc.collect()\n"
                    "        await original_message(event)\n"
                    "        if 'queued sticker' in str(event_content(event).get('body', '')):\n"
                    "            marker = get_hermes_home() / 'rich-events-queued'\n"
                    "            event_id = event.get('event_id') if isinstance(event, dict) else str(event.event_id)\n"
                    "            with marker.open('a', encoding='utf-8') as stream:\n"
                    "                stream.write(event_id + '\\n')\n"
                    "        if queued:\n"
                    "            signal('reply-queued', 'queued')\n"
                    "        expected = get_hermes_home() / 'expected-media-change'\n"
                    "        from plugins.platforms.matrix.effective_event import event_content\n"
                    "        relation = event_content(event).get('m.relates_to', {})\n"
                    "        if expected.exists() and relation.get('event_id') == expected.read_text(encoding='utf-8'):\n"
                    "            signal('media-change-observed', relation['event_id'])\n"
                    "    self._on_room_message = observed_message\n"
                    "BasePlatformAdapter.__init__ = observed_init\n"
                )
        if mode == "pause-image-conversion":
            with (plugin / "__init__.py").open("a", encoding="utf-8") as stream:
                stream.write(
                    "import time\n"
                    "from pathlib import Path\n"
                    "async def expanded(self, target):\n"
                    "    return 'Live enrichment completed'\n"
                    "ContextPause.expand = expanded\n"
                    "original_read_bytes = Path.read_bytes\n"
                    "def paused_read_bytes(path):\n"
                    "    home = get_hermes_home()\n"
                    "    if path.suffix == '.png' and home in path.parents and not (home / 'context-started').exists():\n"
                    "        signal('context-started', 'conversion')\n"
                    "        deadline = time.monotonic() + 10\n"
                    "        while not (home / 'context-release').exists():\n"
                    "            if time.monotonic() >= deadline:\n"
                    "                raise TimeoutError('Matrix file conversion was not released')\n"
                    "            time.sleep(0.01)\n"
                    "    return original_read_bytes(path)\n"
                    "Path.read_bytes = paused_read_bytes\n"
                )
        if resolution_pause:
            plugin = home / "plugins" / "matrix-live-resolution"
            plugin.mkdir(parents=True)
            (plugin / "plugin.yaml").write_text(
                "name: matrix-live-resolution\nversion: 1.0.0\ndescription: Matrix live resolution barrier\n",
                encoding="utf-8",
            )
            (plugin / "__init__.py").write_text(
                (Path(__file__).parent / "resolution_probe.py").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        gateway_home_setup(home)

        with DockerContainer(
            gateway_image,
            network=network,
            entrypoint="/opt/hermes/.venv/bin/python",
            user=_host_user(),
            working_dir="/opt/hermes",
            extra_hosts={"host.docker.internal": route.container_address},
        ).with_command("-m hermes_cli.main gateway run").with_volume_mapping(
            home, "/opt/data", "rw"
        ).with_volume_mapping(
            REPO_ROOT / "tests/integration/matrix_live", "/matrix_live", "ro"
        ).with_env("HOME", "/opt/data") as container:
            def connected() -> bool:
                output = container.get_wrapped_container().logs().decode(errors="replace")
                gateway_log = home / "logs" / "gateway.log"
                if gateway_log.exists():
                    log = gateway_log.read_text(errors="replace")
                    if (_gateway_ready(log, room_id)
                            and "Matrix: connected after initial dispatch checkpoint" in log):
                        return True
                if container.get_wrapped_container().status == "exited":
                    pytest.fail(f"Gateway exited before Matrix connected:\n{output}")
                return False

            _wait_for(
                connected, "Matrix gateway start-up", timeout=120,
                details=lambda: container.get_wrapped_container().logs().decode(errors="replace")[-6000:],
            )
            yield LiveGateway(container, model, home)

    # The test runner ignores errors when it deletes its temporary directory, so files that the
    # container wrote and this user cannot delete would remain on the host. Removing the home here
    # makes such files fail the test.
    shutil.rmtree(home)


@dataclass(frozen=True)
class ApprovalGateway(LiveGateway):
    other_user: MatrixAccount
    encrypted: bool
    decision: str

    def diagnostics(self) -> str:
        from plugins.platforms.matrix.approval_cards import force_redact_command

        requests = [
            {"kind": record["kind"], "model": request.get("model"), "tools": [tool["function"]["name"] for tool in request.get("tools", [])],
             "messages": [message for message in request.get("messages", []) if message.get("role") != "system"][-8:]}
            for record in self.model.requests
            for request in [record["body"]]
        ]
        logs = {}
        for filename in ("gateway.log", "agent.log", "errors.log"):
            path = self.home / "logs" / filename
            if path.exists():
                logs[filename] = path.read_text(encoding="utf-8", errors="replace")[-8000:]
        stdout = self.container.get_wrapped_container().logs().decode(errors="replace")[-8000:]
        return force_redact_command(json.dumps({"model_requests": requests, "logs": logs, "stdout": stdout}, indent=2))


@pytest.fixture
def approval_gateway(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    gateway_image: str,
    synapse: tuple[DockerContainer, str, Network],
    live_room: LiveRoom,
) -> Iterator[ApprovalGateway]:
    from tests.fakes.fake_llm_provider import ToolCall
    from nio import RoomPutStateResponse, RoomInviteResponse, JoinResponse

    encrypted, decision = request.param
    _, url, network = synapse

    async def prepare() -> MatrixAccount:
        other = await _register(url, "other", unique=True)
        alice = live_room.observer.client(url)
        other_client = other.client(url)
        try:
            invited = await alice.room_invite(live_room.room_id, other.user_id)
            assert isinstance(invited, RoomInviteResponse), invited
            joined = await other_client.join(live_room.room_id)
            assert isinstance(joined, JoinResponse), joined
            if encrypted:
                response = await alice.room_put_state(
                    live_room.room_id, "m.room.encryption", {"algorithm": "m.megolm.v1.aes-sha2"},
                )
                assert isinstance(response, RoomPutStateResponse), response
            return other
        finally:
            await alice.close()
            await other_client.close()

    other = asyncio.run(prepare())
    script: list[Response] = [
        ToolCall("terminal", {"command": "python3 -c \"print('approval-first-ran')\""}),
    ]
    if decision == "once":
        script.append(ToolCall("terminal", {"command": "python3 -c \"print('approval-second-ran')\""}))
    script.extend(Text("Matrix approval live reply") for _ in range(2 if decision == "once" else 1))
    home = tmp_path / "approval-home"
    home.mkdir()
    route = _host_route(network)
    with FakeLLMServer(script, bind_host=route.bind_host, aux=lambda _req: Text("Prints a test marker through Python.")) as model:
        summary = decision == "summarized"
        write_hermes_home(
            home, f"http://host.docker.internal:{model.port}/v1",
            extra_config=(
                "platforms:\n  matrix:\n    enabled: true\n"
                "updates:\n  check: false\n"
                "approvals:\n  mode: manual\n"
                f"  timeout: {3 if decision == 'expired' else 60}\n"
                "auxiliary:\n  title_generation:\n    enabled: false\n    model_upgrade_enabled: false\n"
                "  background_review:\n    enabled: false\n"
                "memory:\n  memory_enabled: false\n  user_profile_enabled: false\n"
                "compression:\n  enabled: false\n"
                "matrix:\n  approvals:\n    llm_summary:\n"
                f"      enabled: {str(summary).lower()}\n"
                "      provider_policy: remote_redacted\n"
            ),
        )
        with (home / ".env").open("a", encoding="utf-8") as stream:
            stream.write(
                "MATRIX_HOMESERVER=http://synapse:8008\n"
                f"MATRIX_ACCESS_TOKEN={live_room.bot.access_token}\n"
                f"MATRIX_ALLOWED_USERS={live_room.observer.user_id},{other.user_id}\n"
                f"MATRIX_HOME_ROOM={live_room.room_id}\n"
                "MATRIX_E2EE_MODE=optional\nMATRIX_REACTIONS=false\nMATRIX_AUTO_THREAD=false\nMATRIX_REQUIRE_MENTION=false\nMATRIX_THREAD_REQUIRE_MENTION=false\n"
            )
        with DockerContainer(
            gateway_image, network=network,
            entrypoint="/opt/hermes/.venv/bin/python", user=_host_user(), working_dir="/opt/hermes",
            extra_hosts={"host.docker.internal": route.container_address},
        ).with_command("-m hermes_cli.main gateway run").with_volume_mapping(
            home, "/opt/data", "rw"
        ).with_env("HOME", "/opt/data") as container:
            launcher = container.get_wrapped_container().exec_run(["python3", "-c", "import sys; print(sys.executable)"])
            assert (launcher.exit_code, launcher.output) == (0, b"/usr/local/bin/python3\n"), launcher
            def connected() -> bool:
                gateway_log = home / "logs" / "gateway.log"
                return gateway_log.exists() and _gateway_ready(gateway_log.read_text(errors="replace"), live_room.room_id)

            _wait_for(
                connected, "approval gateway start-up", timeout=120,
                details=lambda: container.get_wrapped_container().logs().decode(errors="replace")[-6000:],
            )
            yield ApprovalGateway(container, model, home, other, encrypted, decision)

    shutil.rmtree(home)
