"""Build the Linux gateway image for the Matrix live tests.

The ``gateway_image`` fixture runs this build for each test module unless
``HERMES_TEST_MATRIX_GATEWAY_IMAGE`` specifies an image that is already loaded.
CI builds the image once with
``python -m tests.integration.matrix_live.image_build TAG`` before the tests
start, so the build does not count towards the per-file test timeout.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DOCKERFILE = Path(__file__).with_name("Dockerfile")


def build_command(tag: str, labels: Mapping[str, str]) -> list[str]:
    label_arguments = [argument for name, value in labels.items() for argument in ("--label", f"{name}={value}")]
    return [
        "docker", "buildx", "build", "--load", "--progress=plain",
        "-f", str(DOCKERFILE), *label_arguments, "-t", tag, str(REPO_ROOT),
    ]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.executable} -m tests.integration.matrix_live.image_build TAG")
    sys.exit(subprocess.run(build_command(sys.argv[1], {}), check=False).returncode)
