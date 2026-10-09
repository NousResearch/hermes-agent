"""Prepare the image environment and record its first-generation baseline."""
from __future__ import annotations

from pathlib import Path

from pm import build_environment
from pm.features import installed_extras, write_features


IMAGE_EXTRAS = (
    "all", "messaging", "otlp", "anthropic", "bedrock",
    "azure-identity", "matrix", "google-chat",
)


def build_image_dependencies(root: Path, python: Path) -> None:
    build_environment(
        source=root, python=python, out=root / ".venv",
        extras=list(IMAGE_EXTRAS), no_install_project=True,
        frozen=True, sealed=True, explicit=True,
    )
    inventory = installed_extras(root, root / ".venv", python_exe=python)
    write_features(inventory, root)


if __name__ == "__main__":
    build_image_dependencies(Path("/opt/hermes"), Path("/usr/local/bin/python3"))
