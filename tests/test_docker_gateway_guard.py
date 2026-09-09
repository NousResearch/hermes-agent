"""Exercise the real subprocess guard with a harmless external Docker stand-in."""

import os
import subprocess
import sys

import pytest

from tests.docker.conftest import container_name, docker_exec_sh, start_container  # noqa: F401
from tests.docker.ownership import container_target, is_owned_container_exec


@pytest.fixture
def fake_docker(tmp_path, monkeypatch):
    docker = tmp_path / "docker"
    docker.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "if sys.argv[1] == 'run': print('a' * 64)\n"
        "elif sys.argv[1] == 'exec': print('profile=default')\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))


@pytest.mark.linux_only
def test_gateway_exec_uses_owned_id_and_loses_authority_on_fixture_cleanup(fake_docker, request):
    # Enter the actual cleanup fixture as pytest does, then exercise its finalizer
    # before checking that the ID no longer grants the guard's narrow exception.
    from tests.docker.conftest import container_name as fixture

    lifecycle = fixture.__wrapped__(request)
    name = next(lifecycle)
    try:
        start_container("test-image", name)
        target = container_target(name)
        assert target == "a" * 64 and target != name
        assert docker_exec_sh(name, "hermes -p voice gateway start").returncode == 0
    finally:
        with pytest.raises(StopIteration):
            next(lifecycle)
    command = ["docker", "exec", "-u", "hermes", target, "hermes", "gateway", "start"]
    assert not is_owned_container_exec(command)
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.run(command, check=True)


@pytest.mark.linux_only
@pytest.mark.parametrize("kind", ["host", "unowned", "shell"])
def test_container_exception_does_not_allow_host_or_unowned_gateway(fake_docker, container_name, kind):
    start_container("test-image", container_name)
    target = container_target(container_name)
    command = {
        "host": ["hermes", "gateway", "start"],
        "unowned": ["docker", "exec", "-u", "hermes", "b" * 64, "hermes", "gateway", "start"],
        "shell": f"docker exec -u hermes {target} hermes gateway start",
    }[kind]
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.run(command, shell=isinstance(command, str), check=True)
