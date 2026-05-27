"""Regression tests for Docker entrypoint config.yaml ownership repair."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = REPO_ROOT / "docker" / "entrypoint.sh"


def test_config_ownership_repair_rejects_symlinks() -> None:
    text = ENTRYPOINT.read_text()

    assert '[ ! -L "$config_path" ]' in text, (
        "entrypoint must not chown/chmod HERMES_HOME/config.yaml when it is "
        "a symlink, because root-side chmod/chown would affect the target"
    )
    assert 'chown hermes:hermes "$HERMES_HOME/config.yaml"' not in text
    assert 'chmod 640 "$HERMES_HOME/config.yaml"' not in text
