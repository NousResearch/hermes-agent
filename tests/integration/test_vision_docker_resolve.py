"""Docker integration tests for the vision image-source resolver.

Exercises the in-sandbox exec-read path that unit tests can only mock. Two
scenarios that ONLY the container filesystem can serve:

  * a tmpfs ``/workspace`` file with no host path at all, and
  * a root-owned mode-600 file the host user cannot read,

both delivered by the resolver's ``base64`` exec-read fallback
(``tools/image_source._resolve_container_fallback``). This is the same path
that provides terminal-backend confinement for GHSA-gpxw-6wxv-w3qq: under a
non-local backend a non-cache path is read INSIDE the container, never on the
host.

Gating follows the repo convention: the ``integration`` marker excludes these
from the default suite (``addopts = -m 'not integration'`` in pyproject.toml);
they run under ``pytest -m integration`` when a Docker daemon is available and
skip cleanly when it is not. Container spin-up exceeds the 30s suite default,
so the timeout is bumped to 180s.

Run:  pytest -m integration tests/integration/test_vision_docker_resolve.py
"""
import base64
import shutil
import subprocess

import pytest


def _docker_available() -> bool:
    """True iff a docker CLI is on PATH and the daemon answers."""
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(
            ["docker", "info"], capture_output=True, timeout=5
        ).returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.timeout(180),
    pytest.mark.skipif(not _docker_available(), reason="Docker daemon not available"),
]

# A real 1x1 PNG.
_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


@pytest.fixture
def docker_backend(request, monkeypatch):
    """A live DockerEnvironment registered so ``get_active_env`` finds it.

    Unique task_id per test: DockerEnvironment derives the container name from
    the task_id, so a shared id would make one test's teardown remove the
    other's container.
    """
    from tools import terminal_tool
    from tools.environments import docker as docker_env

    # The resolver keys the exec-read off TERMINAL_ENV=docker.
    monkeypatch.setenv("TERMINAL_ENV", "docker")

    task_id = f"vision-docker-resolve-{request.node.name}"
    env = docker_env.DockerEnvironment(
        image="python:3.11-slim",
        cwd="/workspace",
        timeout=120,
        task_id=task_id,
        volumes=[],
        network=False,
    )
    # Register under the raw task_id; get_active_env() falls back to the raw
    # key after trying the _resolve_container_task_id() mapping.
    with terminal_tool._env_lock:
        terminal_tool._active_environments[task_id] = env
    try:
        env._task_id = task_id
        yield env
    finally:
        with terminal_tool._env_lock:
            terminal_tool._active_environments.pop(task_id, None)
        try:
            env.cleanup()
        except Exception:
            pass


def _write_png_in_container(env, path, *, mode=None):
    b64 = base64.b64encode(_TINY_PNG).decode()
    cmd = f"printf %s {b64} | base64 -d > {path}"
    if mode:
        cmd += f" && chmod {mode} {path}"
    res = env.execute(cmd)
    assert res.get("returncode", 1) == 0, res


@pytest.mark.asyncio
async def test_resolves_tmpfs_workspace_file(docker_backend):
    """A container-only path (no host file) is delivered via exec-read."""
    from tools.image_source import ResolveContext, resolve_image_source

    _write_png_in_container(docker_backend, "/workspace/shot.png")
    res = await resolve_image_source(
        "/workspace/shot.png", ResolveContext(task_id=docker_backend._task_id))
    assert res.origin == "container"
    assert res.mime == "image/png"
    assert res.data == _TINY_PNG


@pytest.mark.asyncio
async def test_resolves_root_owned_mode600_file(docker_backend):
    """Root-owned mode-600 (host user can't read it) is served in-container."""
    from tools.image_source import ResolveContext, resolve_image_source

    _write_png_in_container(docker_backend, "/workspace/secret.png", mode="600")
    res = await resolve_image_source(
        "/workspace/secret.png", ResolveContext(task_id=docker_backend._task_id))
    assert res.origin == "container"
    assert res.mime == "image/png"
    assert res.data == _TINY_PNG


@pytest.mark.asyncio
async def test_host_secret_path_reads_container_not_host(docker_backend, tmp_path):
    """The GHSA-gpxw invariant, end-to-end against real Docker.

    A path that exists on the HOST with secret bytes but does NOT exist in the
    container must resolve to the container read (which fails to find it) —
    never to the host bytes. Proves vision cannot exfiltrate a host file under
    a sandbox backend even when the exact path is real on the host.
    """
    from tools.image_source import (
        ImageResolutionError,
        ResolveContext,
        resolve_image_source,
    )

    # A real host file outside any media cache, holding a secret.
    host_secret = tmp_path / "id_rsa"
    host_secret.write_bytes(b"HOST-PRIVATE-KEY-DO-NOT-LEAK")

    # That path does not exist inside the fresh container, so the exec-read
    # finds nothing and the resolver refuses. The exact failure shape depends
    # on the container's base64 build (some exit non-zero -> SourceNotFound,
    # some print an error to stderr and exit 0 with empty stdout -> NotAnImage);
    # both are ImageResolutionError. What matters for GHSA-gpxw: the resolver
    # never returns the HOST bytes.
    with pytest.raises(ImageResolutionError) as excinfo:
        await resolve_image_source(
            str(host_secret), ResolveContext(task_id=docker_backend._task_id))
    # Belt and suspenders: the host secret must not appear anywhere in the error.
    assert b"HOST-PRIVATE-KEY".decode() not in str(excinfo.value)
