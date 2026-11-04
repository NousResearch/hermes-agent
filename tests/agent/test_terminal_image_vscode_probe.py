"""VS Code image-support probe (agent/terminal_image.py::probe_vscode_images) over a real PTY.

The far end of the PTY plays the terminal: it answers the probe the way VS Code's xterm.js was
measured to answer with ``terminal.integrated.enableImages`` on (sixel reply + DA1) and off
(DA1 only).
"""

from __future__ import annotations

import os
import select
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.platforms("posix")

_REPO = str(Path(__file__).resolve().parents[2])
_CHILD = textwrap.dedent("""
    from agent import terminal_image
    print("RESULT", terminal_image.probe_vscode_images(), terminal_image.supports_iterm_images(),
          terminal_image.resolve_inline_mode("auto", overlays=True), terminal_image.resolve_inline_mode("auto"),
          flush=True)
""")


def _run(reply: bytes) -> str:
    import pty

    master, slave = pty.openpty()
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("SSH_", "KITTY_", "ITERM_", "WEZTERM_", "KONSOLE_", "TMUX"))}
    env.update(TERM_PROGRAM="vscode", TERM="xterm-256color", PYTHONPATH=_REPO)
    proc = subprocess.Popen([sys.executable, "-c", _CHILD], stdin=slave, stdout=slave,
                            stderr=subprocess.PIPE, env=env, cwd=_REPO, text=False)
    os.close(slave)
    out = b""
    answered = False
    try:
        while True:
            if not select.select([master], [], [], 15)[0]:
                break
            try:
                data = os.read(master, 4096)
            except OSError:  # EIO: the child closed its end
                break
            if not data:
                break
            out += data
            if not answered and b"\x1b[c" in out:
                os.write(master, reply)
                answered = True
    finally:
        os.close(master)
        proc.wait(timeout=15)
    text = out.decode(errors="replace")
    assert proc.returncode == 0 and proc.stderr is not None, proc.stderr and proc.stderr.read()
    return next((ln.strip() for ln in text.splitlines() if "RESULT" in ln), text)


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (b"\x1b[?1;0;4096S\x1b[?62;4;9;22c", "RESULT True True iterm unicode"),  # enableImages on
        (b"\x1b[?1;2c", "RESULT False False unicode unicode"),  # enableImages off: DA1 only
    ],
)
def test_vscode_probe_picks_full_images_only_when_vscode_answers_for_them(reply, expected):
    assert _run(reply).endswith(expected)
