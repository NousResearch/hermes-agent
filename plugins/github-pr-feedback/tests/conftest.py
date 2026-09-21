"""Make plugin sources importable under the repository's hermetic test runner."""

from __future__ import annotations

import sys
from pathlib import Path


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))


def pytest_configure(config):
    # Plugin tests are outside tests/, so its isolation conftest never runs.
    # Clear task identity before collection: handoff tests spawn real CLI children.
    import os
    import tempfile
    import pytest

    sandbox = tempfile.TemporaryDirectory(prefix="hermes-feedback-tests-")
    patch = pytest.MonkeyPatch()
    for name in tuple(os.environ):
        if name.startswith("HERMES_KANBAN_") or name == "HERMES_PROFILE":
            patch.delenv(name, raising=False)
    patch.setenv("HOME", sandbox.name)
    patch.setenv("HERMES_HOME", str(Path(sandbox.name) / ".hermes"))
    patch.setenv("HERMES_TEST_ISOLATION", str(Path(sandbox.name) / ".hermes"))
    config.add_cleanup(sandbox.cleanup)
    config.add_cleanup(patch.undo)
