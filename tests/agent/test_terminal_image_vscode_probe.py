"""VS Code image-support probe (agent/terminal_image.py::probe_vscode_images) over a real PTY.

The far end of the PTY plays the terminal: it answers the probe the way VS Code's xterm.js was
measured to answer with ``terminal.integrated.enableImages`` on (sixel reply + DA1) and off
(DA1 only), optionally late. After the probe the child reads its tty the way the prompt would:
any reply byte left there would have been typed into the user's prompt.
"""

from __future__ import annotations

import os
import select
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.platforms("posix")

_REPO = str(Path(__file__).resolve().parents[2])
_CHILD = textwrap.dedent("""
    import os, select, sys, termios, tty
    from agent import terminal_image
    result = terminal_image.probe_vscode_images()
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    leaked = b""
    while select.select([fd], [], [], 0.3)[0]:
        leaked += os.read(fd, 256)
    termios.tcsetattr(fd, termios.TCSADRAIN, old)
    print("RESULT", result, terminal_image.supports_iterm_images(),
          terminal_image.resolve_inline_mode("auto", overlays=True), "leaked=%d" % len(leaked), flush=True)
""")

_ON = b"\x1b[?1;0;4096S\x1b[?62;4;9;22c"
_OFF = b"\x1b[?1;2c"


def _run(reply: bytes, delay: float) -> str:
    import pty

    master, slave = pty.openpty()
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("SSH_", "KITTY_", "ITERM_", "WEZTERM_", "KONSOLE_", "TMUX"))}
    env.update(TERM_PROGRAM="vscode", TERM="xterm-256color", PYTHONPATH=_REPO)
    proc = subprocess.Popen([sys.executable, "-c", _CHILD], stdin=slave, stdout=slave,
                            stderr=subprocess.PIPE, env=env, cwd=_REPO)
    os.close(slave)
    out = b""
    answered = False
    try:
        while select.select([master], [], [], 15)[0]:
            try:
                data = os.read(master, 4096)
            except OSError:  # EIO: the child closed its end
                break
            if not data:
                break
            out += data
            if not answered and b"\x1b[c" in out:
                time.sleep(delay)  # health: allow HX013 -- the late reply IS the scenario under test
                os.write(master, reply)
                answered = True
    finally:
        os.close(master)
        proc.wait(timeout=15)
    assert proc.returncode == 0 and proc.stderr is not None, proc.stderr and proc.stderr.read()
    text = out.decode(errors="replace")
    return text[text.index("RESULT"):].splitlines()[0].strip() if "RESULT" in text else text


@pytest.mark.parametrize(
    ("reply", "delay", "expected"),
    [
        (_ON, 0, "RESULT True True iterm leaked=0"),  # enableImages on
        (_OFF, 0, "RESULT False False unicode leaked=0"),  # enableImages off: DA1 only
        (_ON, 0.08, "RESULT True True iterm leaked=0"),  # past the 50 ms deadline, inside the drain
    ],
)
def test_vscode_probe_reads_its_reply_and_leaves_nothing_for_the_prompt(reply, delay, expected):
    assert _run(reply, delay) == expected
