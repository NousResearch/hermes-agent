"""Runtime smoke test for Docker image license-file presence.

Build the real image and verify the LICENSE file is present inside the
container (PEP 639 license-files metadata must resolve inside the
Docker image).
"""
from __future__ import annotations

import subprocess


def test_docker_image_contains_license_file(built_image: str) -> None:
    """The LICENSE file must be present inside the built Docker image.

    PEP 639 license-files metadata references LICENSE, and the Docker
    build context must not exclude it.
    """
    r = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "test",
         built_image, "-f", "/opt/hermes/LICENSE"],
        capture_output=True, text=True, timeout=60,
    )
    assert r.returncode == 0, (
        f"LICENSE file not found at /opt/hermes/LICENSE inside the Docker "
        f"image: {r.stderr[-500:]}"
    )