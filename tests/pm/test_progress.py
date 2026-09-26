"""Contained child output: CI keeps every line, interactive runs never hide a failure."""

import io
import subprocess
import sys

import pytest

from pm.progress import run_contained, verbose_output


def test_ci_streams_and_an_explicit_choice_overrides_it():
    assert verbose_output({"CI": "true"}) and verbose_output({"GITHUB_ACTIONS": "true"})
    assert not verbose_output({}) and not verbose_output({"CI": "0"})
    assert not verbose_output({"CI": "1", "HERMES_VERBOSE": "0"})
    assert verbose_output({"HERMES_VERBOSE": "1"})


@pytest.mark.parametrize("fails", [False, True])
def test_contained_run_shows_output_only_when_the_child_fails(monkeypatch, fails):
    monkeypatch.setenv("HERMES_VERBOSE", "0")
    stream = io.StringIO()
    script = "import sys\nfor i in range(200): print('noise', i)\nprint('the real error')\nsys.exit(%d)" % fails
    command = [sys.executable, "-c", script]
    if fails:
        with pytest.raises(subprocess.CalledProcessError):
            run_contained(command, "Installing things", stream=stream)
    else:
        run_contained(command, "Installing things", stream=stream)
    text = stream.getvalue()
    assert "→ Installing things" in text
    assert ("the real error" in text) is fails
    assert "noise 0" not in text, "the failure tail stays bounded"
