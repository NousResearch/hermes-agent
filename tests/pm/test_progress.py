"""Contained child output: CI keeps every line, interactive runs never hide a failure."""

import io
import subprocess
import sys
import threading
import time

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


class _MemoryStream:
    def __init__(self):
        self._parts = []
        self._lock = threading.Lock()

    def write(self, text):
        with self._lock:
            self._parts.append(text)
        return len(text)

    def flush(self):
        return None

    def isatty(self):
        return False

    def getvalue(self):
        with self._lock:
            return "".join(self._parts)


def test_contained_progress_reaches_the_log_before_the_child_exits(tmp_path, monkeypatch):
    """A non-interactive build must move the log while the child is still running.

    The Windows desktop hand-off cancels a step when update.log is unchanged
    for 10 minutes. Buffering the child until it exits makes a healthy rebuild
    look stalled.
    """
    monkeypatch.setenv("HERMES_VERBOSE", "0")
    ready = tmp_path / "ready"
    release = tmp_path / "release"
    script = (
        "import pathlib, sys, time\n"
        "print('chunk-progress', flush=True)\n"
        f"pathlib.Path({str(ready)!r}).write_text('1')\n"
        "deadline = time.time() + 15\n"
        f"while not pathlib.Path({str(release)!r}).exists():\n"
        "    if time.time() > deadline:\n"
        "        sys.exit(3)\n"
        "    time.sleep(0.05)\n"
    )
    stream = _MemoryStream()
    holder = {}

    def run():
        try:
            run_contained(
                [sys.executable, "-c", script],
                "Building desktop packaged app",
                stream=stream,
                heartbeat_seconds=0,
            )
        except BaseException as exc:
            holder["error"] = exc

    thread = threading.Thread(target=run)
    thread.start()
    try:
        deadline = time.time() + 10
        while "chunk-progress" not in stream.getvalue() and time.time() < deadline:
            if not thread.is_alive() and "error" in holder:
                break
            time.sleep(0.05)
        assert "chunk-progress" in stream.getvalue()
        assert thread.is_alive()
    finally:
        release.write_text("1")
        thread.join(timeout=10)
    assert "error" not in holder
    assert not thread.is_alive()
