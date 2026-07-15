"""Runtime smoke test for Docker image license-file presence.

Build the real image and verify the LICENSE file is present inside the
container (PEP 639 license-files metadata must resolve inside the
Docker image).
"""
from __future__ import annotations

from tests.docker.conftest import docker_exec


def test_docker_image_contains_license_file(shared_container: str) -> None:
    """The LICENSE file must be present inside the built Docker image.

    PEP 639 license-files metadata references LICENSE, and the Docker
    build context must not exclude it.
    """
    r = docker_exec(shared_container, "test", "-f", "/opt/hermes/LICENSE")
    assert r.returncode == 0, (
        f"LICENSE file not found at /opt/hermes/LICENSE inside the Docker "
        f"image: {r.stderr[-500:]}"
    )
