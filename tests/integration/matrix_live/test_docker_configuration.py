"""The live fixtures use the Docker endpoint that the user selected, without registry credentials,
and build the gateway image only when no prebuilt image is supplied."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import docker
import pytest
from docker import errors as docker_errors
from testcontainers.core import testcontainers_config
from testcontainers.core.docker_client import get_docker_host

from tests.integration.matrix_live.conftest import _docker_connection_scope, _gateway_image_tag


def _write_context(config_dir: Path, name: str, host: str) -> None:
    metadata = config_dir / "contexts" / "meta" / hashlib.sha256(name.encode()).hexdigest()
    metadata.mkdir(parents=True)
    (metadata / "meta.json").write_text(
        json.dumps({"Name": name, "Metadata": {}, "Endpoints": {"docker": {"Host": host, "SkipTLSVerify": False}}}),
        encoding="utf-8",
    )


@pytest.mark.platforms("posix")
@pytest.mark.parametrize(
    ("current_context", "environment", "selected"),
    [
        ("current", {}, "current"),
        ("current", {"DOCKER_CONTEXT": "override"}, "override"),
        ("current", {"DOCKER_CONTEXT": "override", "DOCKER_HOST": "host"}, "host"),
        (None, {}, "default"),
    ],
)
def test_scope_uses_selected_endpoint_and_drops_registry_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    current_context: str | None,
    environment: dict[str, str],
    selected: str,
) -> None:
    endpoints = {name: f"unix://{tmp_path / name}.sock" for name in ("current", "override", "host")}
    endpoints["default"] = "unix:///var/run/docker.sock"
    user_config = tmp_path / "user-docker-config"
    extra_plugins = str(tmp_path / "extra-plugins")
    settings = {
        "cliPluginsExtraDirs": [extra_plugins],
        "auths": {"registry.test": {"auth": "private-registry-token"}},
        "credsStore": "private-helper",
        "credHelpers": {"registry.test": "other-private-helper"},
    }
    if current_context is not None:
        settings["currentContext"] = current_context
    user_config.mkdir()
    (user_config / "config.json").write_text(json.dumps(settings), encoding="utf-8")
    for name in ("current", "override"):
        _write_context(user_config, name, endpoints[name])

    for variable in ("DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("DOCKER_CONFIG", str(user_config))
    for variable, value in environment.items():
        monkeypatch.setenv(variable, endpoints[value] if variable == "DOCKER_HOST" else value)
    from_env = docker.from_env
    monkeypatch.setattr(docker, "from_env", lambda: from_env(version="1.47"))
    monkeypatch.setattr(docker.DockerClient, "ping", lambda client: True)
    environment_before = dict(os.environ)

    isolated_config = tmp_path / "isolated-docker-config"
    isolated_config.mkdir()
    with _docker_connection_scope(isolated_config):
        client = docker.from_env()
        try:
            observed = {
                "sdk_socket": client.api.adapters["http+docker://"].socket_path,
                "testcontainers": get_docker_host(),
                "cli_environment": {name: os.environ.get(name) for name in ("DOCKER_HOST", "DOCKER_CONTEXT")},
                "config": json.loads((isolated_config / "config.json").read_text(encoding="utf-8")),
            }
        finally:
            client.close()

    assert observed == {
        "sdk_socket": endpoints[selected].removeprefix("unix://"),
        "testcontainers": endpoints[selected],
        "cli_environment": {"DOCKER_HOST": endpoints[selected], "DOCKER_CONTEXT": None},
        "config": {"cliPluginsExtraDirs": [str(user_config / "cli-plugins"), extra_plugins]},
    }
    assert dict(os.environ) == environment_before


@pytest.mark.platforms("macos")
def test_scope_mounts_the_vm_socket_into_ryuk_on_macos(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    user_config = tmp_path / "user-docker-config"
    user_config.mkdir()
    (user_config / "config.json").write_text(json.dumps({"currentContext": "local"}), encoding="utf-8")
    _write_context(user_config, "local", f"unix://{tmp_path / 'local.sock'}")
    for variable in ("DOCKER_HOST", "DOCKER_CONTEXT", "TESTCONTAINERS_DOCKER_SOCKET_OVERRIDE"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("DOCKER_CONFIG", str(user_config))
    monkeypatch.setattr(testcontainers_config, "_ryuk_docker_socket", "")
    from_env = docker.from_env
    monkeypatch.setattr(docker, "from_env", lambda: from_env(version="1.47"))
    monkeypatch.setattr(docker.DockerClient, "ping", lambda client: True)

    isolated_config = tmp_path / "isolated-docker-config"
    isolated_config.mkdir()
    with _docker_connection_scope(isolated_config):
        ryuk_socket = testcontainers_config.ryuk_docker_socket

    assert ryuk_socket == "/var/run/docker.sock"


class _ImageStore:
    def __init__(self, tags: set[str]) -> None:
        self.tags = tags
        self.removed: list[str] = []

    def get(self, tag: str) -> str:
        if tag not in self.tags:
            raise docker_errors.ImageNotFound(tag)
        return tag

    def remove(self, image: str, force: bool) -> None:
        self.removed.append(image)
        self.tags.discard(image)


@pytest.mark.parametrize("prebuilt", [None, "hermes-matrix-live:ci"])
def test_gateway_image_reuses_a_prebuilt_image_and_removes_only_its_own_build(prebuilt: str | None) -> None:
    store = _ImageStore({"hermes-matrix-live:ci"})
    built: list[str] = []

    def build(tag: str) -> None:
        built.append(tag)
        store.tags.add(tag)

    with _gateway_image_tag(store, prebuilt, build) as image:
        present_during_use = image in store.tags

    own_build = [] if prebuilt else [image]
    assert {"built": built, "present_during_use": present_during_use, "removed": store.removed, "tags": store.tags} == {
        "built": own_build,
        "present_during_use": True,
        "removed": own_build,
        "tags": {"hermes-matrix-live:ci"},
    }
    assert image == prebuilt or image.startswith("hermes-matrix-live:")
