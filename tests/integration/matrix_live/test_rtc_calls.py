"""Real MatrixRTC signalling, OpenID verification and decoded duplex audio on Linux."""

from __future__ import annotations

import asyncio
import json
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

import pytest
from testcontainers.core.container import DockerContainer

from plugins.platforms.matrix.adapter import _STARTUP_GRACE_SECONDS
from tests.fakes.fake_llm_provider import FakeLLMServer, Text, write_hermes_home
from tests.integration.matrix_live.conftest import (
    REPO_ROOT, _gateway_ready, _host_route, _host_user, _register, _wait_for,
)
from tests.integration.matrix_live.live_gateway import LiveGateway

SFU_IMAGE = "livekit/livekit-server:v1.13.6@sha256:e37d68f172556d02aa77968b9fc55ef481468c0315fa38e4fa6c56ce72e3a815"
AUTH_IMAGE = "ghcr.io/element-hq/lk-jwt-service:sha-7991c1f@sha256:b2eb41f06d9d7425781c96399f29758638daac574463b5ce5847df55ae70b83b"
TEST_SECRET = "rtc-native-test-secret-0123456789abcdef"


@pytest.fixture
def matrix_synapse_overrides():
    return {"public_baseurl": "http://synapse:8008/", "serve_client_wellknown": True,
            "extra_well_known_client_content": {"org.matrix.msc4143.rtc_foci": [
                {"type": "livekit", "livekit_service_url": "http://rtc-auth:8080"}]},
            "max_event_delay_duration": "30s",
            "listeners": [{"port": 8008, "tls": False, "type": "http", "bind_addresses": ["0.0.0.0"],
                           "resources": [{"names": ["client", "federation"], "compress": False}]}]}


def _http_ready(url):
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return response.status == 200
    except OSError:
        return False


@pytest.fixture
def rtc_services(tmp_path, gateway_image, synapse):
    _, _, network = synapse
    config = tmp_path / "livekit.yaml"
    config.write_text("port: 7880\nbind_addresses: [0.0.0.0]\nrtc:\n  tcp_port: 7881\n"
                      "  udp_port: 7882\n  use_external_ip: false\nroom:\n  auto_create: false\n"
                      f"keys:\n  nativekey: {TEST_SECRET}\n")
    with ExitStack() as stack:
        sfu = stack.enter_context(DockerContainer(SFU_IMAGE, network=network, network_aliases=["rtc-sfu"])
                                  .with_command(["--config", "/livekit.yaml"])
                                  .with_volume_mapping(config, "/livekit.yaml", "ro").with_exposed_ports(7880))
        _wait_for(lambda: _http_ready(f"http://{sfu.get_container_host_ip()}:{sfu.get_exposed_port(7880)}/"),
                  "LiveKit SFU", timeout=30, details=lambda: sfu.get_wrapped_container().logs().decode(errors="replace")[-4000:])
        proxy = stack.enter_context(DockerContainer(gateway_image, network=network, network_aliases=["matrix.test"],
                                    entrypoint="/opt/hermes/.venv/bin/python")
                                    .with_command("/matrix_live/rtc_tls_proxy.py")
                                    .with_volume_mapping(REPO_ROOT / "tests/integration/matrix_live", "/matrix_live", "ro"))
        _wait_for(lambda: b"RTC federation proxy ready" in proxy.get_wrapped_container().logs(), "OpenID federation TLS", timeout=20)
        auth = stack.enter_context(DockerContainer(AUTH_IMAGE, network=network, network_aliases=["rtc-auth"])
                                   .with_env("LIVEKIT_URL", "ws://rtc-sfu:7880")
                                   .with_env("LIVEKIT_KEY", "nativekey").with_env("LIVEKIT_SECRET", TEST_SECRET)
                                   .with_env("LIVEKIT_FULL_ACCESS_HOMESERVERS", "matrix.test")
                                   .with_env("LIVEKIT_INSECURE_SKIP_VERIFY_TLS", "YES_I_KNOW_WHAT_I_AM_DOING")
                                   .with_env("LIVEKIT_LOG_LEVEL", "debug")
                                   .with_exposed_ports(8080))
        _wait_for(lambda: _http_ready(f"http://{auth.get_container_host_ip()}:{auth.get_exposed_port(8080)}/healthz"),
                  "pinned MatrixRTC authorisation service", timeout=30,
                  details=lambda: auth.get_wrapped_container().logs().decode(errors="replace")[-4000:])
        yield sfu, auth, proxy


@pytest.fixture
def rtc_gateway(tmp_path, gateway_image, synapse, live_room, rtc_services):
    _, _, network = synapse
    home = tmp_path / "rtc-home"
    route = _host_route(network)
    with FakeLLMServer([Text("Typed context reply"), Text("RTC audio reply")], bind_host=route.bind_host) as model:
        write_hermes_home(home, f"http://host.docker.internal:{model.port}/v1", extra_config=
                          "platforms:\n  matrix:\n    enabled: true\nupdates:\n  check: false\n"
                          "auxiliary:\n  title_generation:\n    enabled: false\n"
                          "matrix:\n  rtc:\n    silence_threshold: 0.4\n    leave_delay_seconds: 6\n")
        with (home / ".env").open("a") as stream:
            stream.write("MATRIX_HOMESERVER=http://synapse:8008\n"
                         f"MATRIX_ACCESS_TOKEN={live_room.bot.access_token}\n"
                         f"MATRIX_ALLOWED_USERS={live_room.observer.user_id}\n"
                         f"MATRIX_HOME_ROOM={live_room.room_id}\n"
                         "MATRIX_E2EE_MODE=optional\nMATRIX_REACTIONS=false\nMATRIX_AUTO_THREAD=false\n"
                         "MATRIX_REQUIRE_MENTION=false\n")
        with DockerContainer(gateway_image, network=network, entrypoint="/opt/hermes/.venv/bin/python",
                             user=_host_user(), working_dir="/opt/hermes",
                             extra_hosts={"host.docker.internal": route.container_address}) \
                .with_command("/matrix_live/rtc_gateway.py") \
                .with_env("PYTHONPATH", "/opt/hermes:/matrix_live").with_env("HOME", "/opt/data") \
                .with_volume_mapping(home, "/opt/data", "rw") \
                .with_volume_mapping(REPO_ROOT / "tests/integration/matrix_live", "/matrix_live", "ro") as container:
            _wait_for(lambda: (home / "logs/gateway.log").exists()
                      and _gateway_ready((home / "logs/gateway.log").read_text(errors="replace"), live_room.room_id),
                      "MatrixRTC gateway start-up", timeout=45,
                      details=lambda: container.get_wrapped_container().logs().decode(errors="replace")[-6000:])
            yield LiveGateway(container, model, home)
    shutil.rmtree(home)


@pytest.fixture
def gateway(rtc_gateway: LiveGateway) -> LiveGateway:
    return rtc_gateway


@pytest.mark.parametrize("mode", ["leave", "shutdown", "crash", "restart"])
def test_matrix_rtc_client_visible_lifecycle_and_duplex_audio(
        mode, rtc_gateway, rtc_services, live_room, linux_nio_observer, synapse):
    container, model, home = rtc_gateway.container, rtc_gateway.model, rtc_gateway.home

    def logs():
        named = {"gateway": container, "sfu": rtc_services[0], "auth": rtc_services[1], "proxy": rtc_services[2]}
        return "\n".join(f"--- {name}\n{c.get_wrapped_container().logs().decode(errors='replace')[-6000:]}"
                         for name, c in named.items())

    observer = linux_nio_observer
    async def mallory_account():
        account = await _register(live_room.homeserver, "mallory")
        alice = live_room.observer.client(live_room.homeserver)
        mallory = account.client(live_room.homeserver)
        try:
            await alice.room_invite(live_room.room_id, account.user_id)
            await mallory.join(live_room.room_id)
            return account
        finally:
            await alice.close()
            await mallory.close()
    mallory = asyncio.run(mallory_account())
    args = repr((live_room.room_id, live_room.bot.user_id, mode, asdict(mallory), _STARTUP_GRACE_SECONDS))
    code = f"import asyncio; from rtc_peer import run; print(asyncio.run(run(*{args})))"
    compile(code, "rtc-peer-command", "exec")
    def peer_json(path):
        result = observer.container.exec(["/opt/hermes/.venv/bin/python", "-c",
                    f"from pathlib import Path; p=Path({path!r}); print(p.read_text() if p.exists() else '')"])
        output = result.output.decode(errors="replace").strip()
        return json.loads(output) if output else None
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(observer.run_python, code)
        _wait_for(lambda: peer_json("/opt/data/rtc-peer-ready.json"), "independent RTC participant ready", timeout=75,
                  details=lambda: f"peer: {future.exception() if future.done() else 'running'}\n{logs()}")
        if mode in ("shutdown", "crash", "restart"):
            container.get_wrapped_container().kill(signal="SIGTERM" if mode == "shutdown" else "SIGKILL")
        if mode == "restart":
            container.get_wrapped_container().start()
        control = json.dumps({"mode": mode})
        observer.run_python(f"from pathlib import Path; Path('/opt/data/rtc-peer-control.json').write_text({control!r})")
        try:
            output = future.result(timeout=90)
        except Exception:
            print(logs())
            raise
        result = peer_json("/opt/data/rtc-peer-result.json")
    assert {"mode": result["mode"], "left": result["left"]} == {"mode": mode, "left": {}}, output
    if mode == "shutdown":
        status = container.get_wrapped_container().wait(timeout=30)
        gateway_logs = container.get_wrapped_container().logs().decode(errors="replace")
        aborted = [marker for marker in ("panicked", "fatal runtime error", "Aborted") if marker in gateway_logs]
        # An unplanned SIGTERM exits 1 so that a service manager restarts the gateway. A native
        # abort during LiveKit teardown would exit 134 instead.
        assert (status["StatusCode"], aborted) == (1, []), gateway_logs[-6000:]
    requests = model.main_requests()
    assert (len(requests), model.aux_requests()) == (2 if mode == "leave" else 0, [])
    if mode == "leave":
        records = json.loads((home / "rtc-stt.json").read_text())
        assert len(records) == 1, records
        assert {"sample_rate": records[0]["sample_rate"], "channels": records[0]["channels"], "home": records[0]["home"]} == {
            "sample_rate": 16000, "channels": 1, "home": "/opt/data"}
        assert abs(records[0]["frequency"] - 660) < 12 and records[0]["duration"] >= 0.5, records
        assert result["received"]["duration"] >= 0.5 and abs(result["received"]["frequency"] - 880) < 12, result
        assert requests[0]["tools"] == requests[1]["tools"]
        prefix = [message for message in requests[0]["messages"] if message["role"] == "system"]
        assert [message for message in requests[1]["messages"] if message["role"] == "system"] == prefix
        assert requests[1]["messages"][:len(requests[0]["messages"])] == requests[0]["messages"]
    print(json.dumps({"mode": mode, "decoded_audio": result["received"], "main_calls": len(requests), "auxiliary_calls": 0}))
