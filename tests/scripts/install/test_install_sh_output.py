"""install.sh child output: one status line on a terminal, the full stream in CI."""
import os
from pathlib import Path
import pty
import re
import shlex
import subprocess

import pytest

pytestmark = pytest.mark.platforms("posix")
INSTALL_SH = Path(__file__).resolve().parents[3] / "scripts" / "install.sh"
NOISY = "echo noisy-first; echo noisy-second >&2; exit {code}"


def _env(tmp_path: Path, **extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in ("CI", "GITHUB_ACTIONS", "HERMES_INSTALL_VERBOSE")}
    env.update(HOME=str(tmp_path), HERMES_HOME=str(tmp_path / "home"), NO_COLOR="1", TERM="dumb", **extra)
    return env


def _script(code: int) -> str:
    return (f"source {shlex.quote(INSTALL_SH.as_posix())} --manifest\n"
            f"run_logged 'Noisy step' sh -c {shlex.quote(NOISY.format(code=code))}\n"
            "echo \"rc=$?\"\n")


def _on_terminal(tmp_path: Path, code: int) -> tuple[str, str]:
    """Run under a pseudo-terminal; return the transcript as the screen shows it."""
    controller, terminal = pty.openpty()
    proc = subprocess.Popen(["bash", "-c", _script(code)], stdin=terminal, stdout=terminal, stderr=terminal,
                            env=_env(tmp_path), close_fds=True)
    os.close(terminal)
    chunks = []
    while True:
        try:
            chunk = os.read(controller, 4096)
        except OSError:  # EIO: the child closed the terminal
            break
        if not chunk:
            break
        chunks.append(chunk)
    os.close(controller)
    assert proc.wait(timeout=30) == 0
    raw = b"".join(chunks).decode("utf-8", "replace").replace("\r\n", "\n")
    # A carriage return redraws its line: the screen keeps what follows the last one.
    screen = "\n".join(re.sub(r"\x1b\[K", "", line).rsplit("\r", 1)[-1] for line in raw.split("\n"))
    return screen, (tmp_path / "home" / "logs" / "install.log").read_text(encoding="utf-8")


def test_terminal_run_keeps_output_in_the_log_and_shows_it_only_on_failure(tmp_path):
    screen, log = _on_terminal(tmp_path / "ok", 0)
    assert "noisy" not in screen and "rc=0" in screen
    assert "noisy-first" in log and "noisy-second" in log

    screen, log = _on_terminal(tmp_path / "failed", 3)
    assert "rc=3" in screen
    assert "Noisy step failed (exit 3)" in screen
    assert "noisy-first" in screen and "noisy-second" in screen
    assert str(tmp_path / "failed" / "home" / "logs" / "install.log") in screen


def test_ci_streams_every_line_and_writes_no_log(tmp_path):
    result = subprocess.run(["bash", "-c", _script(0)], env=_env(tmp_path, CI="true"),
                            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "noisy-first" in result.stdout and "noisy-second" in result.stderr
    assert "Noisy step" in result.stdout
    assert not (tmp_path / "home" / "logs" / "install.log").exists()
