"""Provider registry."""

import pytest

from sandbox.errors import SandboxConfigError
from sandbox.providers.docker import DockerSandboxProvider
from sandbox.providers.gvisor import GVisorSandboxProvider
from sandbox.providers.local import LocalSandboxProvider
from sandbox.registry import get_provider


def test_get_local():
    p = get_provider("local", {})
    assert isinstance(p, LocalSandboxProvider)


def test_get_docker():
    p = get_provider("docker", {"image": "alpine:3.19"})
    assert isinstance(p, DockerSandboxProvider)


def test_get_gvisor():
    p = get_provider("gvisor", {"image": "alpine:3.19", "gvisor": {"docker_runtime": "runsc"}})
    assert isinstance(p, GVisorSandboxProvider)


def test_unknown_raises():
    with pytest.raises(SandboxConfigError):
        get_provider("unknown-backend", {})
