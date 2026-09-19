"""Deterministic files for the edit-tool shape audit."""

from __future__ import annotations

import tempfile
from pathlib import Path


def build_workspace() -> tempfile.TemporaryDirectory[str]:
    """Create a small, realistic source tree and return its owner."""
    workspace = tempfile.TemporaryDirectory(prefix="edittool-")
    root = Path(workspace.name)
    (root / "src").mkdir()
    (root / "config").mkdir()
    (root / "src" / "service.py").write_text(
        "def normalize(value):\n    return value.strip()\n", encoding="utf-8"
    )
    (root / "src" / "retry.py").write_text(
        "def retry_delay():\n    return 250\n", encoding="utf-8"
    )
    (root / "config" / "settings.ini").write_text(
        "[api]\nenabled = false\n[worker]\nenabled = false\n", encoding="utf-8"
    )
    (root / "src" / "handlers.py").write_text(
        "def create_user():\n    # TODO: validate input\n    pass\n\n"
        "def delete_user():\n    # TODO: validate input\n    pass\n",
        encoding="utf-8",
    )
    (root / "src" / "already.py").write_text(
        "STATUS = 'new'\n", encoding="utf-8"
    )
    return workspace
