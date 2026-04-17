"""Docker argv construction (no real Docker)."""

import sys

import pytest

from sandbox.docker_cmd import build_docker_run_argv
from sandbox.errors import SandboxConfigError, SandboxNotSupportedError
from sandbox.providers.docker import DockerSandboxProvider
from sandbox.types import IsolationProfile


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX bind-mount paths")
def test_build_docker_run_argv_includes_mount_and_image():
    prof = IsolationProfile(network_policy="none", cpu_quota=0.5, mem_limit_mb=128)
    argv = build_docker_run_argv(
        image="alpine:3.19",
        workdir="/tmp/w",
        inner_cmd=["python", "script.py"],
        child_env={"FOO": "bar"},
        profile=prof,
    )
    assert argv[0:2] == ["docker", "run"]
    assert "--network" in argv and "none" in argv
    assert "-v" in argv and "/tmp/w:/tmp/w" in argv
    assert argv[-3:] == ["alpine:3.19", "python", "script.py"]


@pytest.mark.skipif(sys.platform == "win32", reason="docker provider argv Linux-oriented")
def test_build_popen_argv_requires_image():
    prov = DockerSandboxProvider({})
    with pytest.raises(SandboxConfigError):
        prov.build_popen_argv(
            workdir="/tmp/h",
            inner_cmd=["python", "x.py"],
            child_env={},
        )


@pytest.mark.skipif(sys.platform == "win32", reason="host docker argv not supported")
def test_build_popen_argv_shape():
    prov = DockerSandboxProvider({"image": "alpine:3.19", "profiles": {"default": {}}})
    argv = prov.build_popen_argv(
        workdir="/tmp/h",
        inner_cmd=["python", "x.py"],
        child_env={"HERMES_RPC_SOCKET": "/tmp/h/s.sock"},
    )
    assert "docker" in argv[0]
    assert "alpine:3.19" in argv


def test_windows_raises_not_supported():
    if sys.platform != "win32":
        pytest.skip("Windows-only assertion")
    prov = DockerSandboxProvider({"image": "alpine:3.19"})
    with pytest.raises(SandboxNotSupportedError):
        prov.build_popen_argv(
            workdir="C:\\\\temp\\\\w",
            inner_cmd=["python", "x.py"],
            child_env={},
        )
