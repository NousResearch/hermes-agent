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
    script = f"import sys\nfor i in range(200): print('noise', i)\nprint('the real error')\nsys.exit({fails:d})"
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

def test_contained_run_does_not_hang_when_child_waits_on_stdin(monkeypatch):
    """Contained mode hides stdout, so a stdin prompt would hang forever (#138154)."""
    monkeypatch.setenv("HERMES_VERBOSE", "0")
    stream = io.StringIO()
    # Prints a prompt then blocks on stdin.readline(). With inherited stdin this
    # waits forever while the prompt sits unread in the capture pipe.
    script = "import sys\nprint('Continue?', flush=True)\nsys.stdin.readline()\n"
    run_contained([sys.executable, "-c", script], "Preparing Node dependencies", stream=stream)
    assert "→ Preparing Node dependencies" in stream.getvalue()


def test_contained_run_honors_explicit_stdin(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_VERBOSE", "0")
    fed = tmp_path / "in.txt"
    fed.write_text("yes\n", encoding="utf-8")
    stream = io.StringIO()
    script = "import sys\nline = sys.stdin.readline().strip()\nraise SystemExit(0 if line == 'yes' else 1)\n"
    with fed.open(encoding="utf-8") as fh:
        run_contained([sys.executable, "-c", script], "Reading input", stream=stream, stdin=fh)
    assert "→ Reading input" in stream.getvalue()

