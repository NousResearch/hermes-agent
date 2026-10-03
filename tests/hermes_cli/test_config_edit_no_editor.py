"""Tests for the actionable hint ``hermes config edit`` prints when no editor
is found.

On a minimal container (no ``nano``/``vim``/``code`` and no ``$EDITOR``) the
command used to print a two-line dead end. It now names the editors it tried
and the cheap non-interactive alternatives.
"""

import os
from unittest.mock import patch

import pytest

from hermes_cli.config import edit_config


@pytest.fixture(autouse=True)
def _isolated_hermes_home(tmp_path):
    (tmp_path / ".env").touch()
    # Pre-create config.yaml so edit_config goes straight to editor resolution.
    (tmp_path / "config.yaml").write_text("model: hermes\n", encoding="utf-8")
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}, clear=False):
        # Ensure neither EDITOR nor VISUAL is set.
        os.environ.pop("EDITOR", None)
        os.environ.pop("VISUAL", None)
        yield tmp_path


def test_no_editor_message_is_actionable(_isolated_hermes_home, capsys):
    # No editor on PATH and no $EDITOR / $VISUAL.
    with patch("shutil.which", return_value=None):
        edit_config()
    out = capsys.readouterr().out
    assert "No editor found" in out
    # Names the editors it tried…
    assert "Tried:" in out
    # …and each of the three documented next steps.  Assert only the ``EDITOR``
    # variable name: the hint must not pin shell-specific ``VAR=value`` syntax,
    # nor name a concrete editor — every plausible one is a probed candidate, so
    # reaching this branch proves it is absent.
    assert "EDITOR" in out
    assert "hermes config set" in out
    assert "hermes config path" in out


def test_no_editor_does_not_launch_subprocess(_isolated_hermes_home):
    with patch("shutil.which", return_value=None), \
            patch("subprocess.run") as run:
        edit_config()
    run.assert_not_called()
