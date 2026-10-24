"""A venv re-entry keeps the application argv visible to process-identity readers.

``relaunch_command`` re-execs Hermes as ``python -I -c "<source that assigns sys.argv>"``. Before
this fix the application argv lived only inside that source, so ``/proc/<pid>/cmdline`` and ``ps``
showed four elements and nothing else. The Desktop's SSH ownership proof
(``apps/desktop/electron/remote-lifecycle.ts`` ``pidIsOurDashboard``) looks for ``serve``,
``--isolated`` and ``--ssh-owner-nonce <nonce>`` as separate argv elements, so it judged every
re-entered backend FOREIGN: each reconnect dropped the lock and spawned a second backend beside a
live one, and that orphan kept holding its open chats until it went idle.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import venv_sync

NONCE = "a" * 16
APP = ["--profile", "academic", "serve", "--isolated", "--host", "127.0.0.1", "--port", "0",
       "--ssh-session-token-file", "/h/desktop-ssh/x/t.token", "--ssh-owner-nonce", NONCE]


def _desktop_owned(args: list[str], nonce: str, profile: str) -> bool:
    """The element-wise checks of the Desktop's POSIX ownership proof (pidIsOurDashboard)."""
    try:
        p = args.index("--profile")
        serve_pos = [i for i, a in enumerate(args) if a == "serve" and i != p + 1]
        serve = serve_pos[0]
        owner = args.index("--ssh-owner-nonce", serve + 1)
        isolated = args.index("--isolated", serve + 1)
    except (ValueError, IndexError):
        return False
    return (len(serve_pos) == 1 and isolated > serve and args.count("--isolated") == 1
            and args.count("--ssh-owner-nonce") == 1 and args[owner + 1] == nonce
            and args.count("--profile") == 1 and args[p + 1] == profile)


@pytest.mark.parametrize("mode", ["script", "module", "command"])
def test_reentry_argv_is_visible_and_unchanged_for_the_app(tmp_path, mode):
    root = tmp_path / "src"
    root.mkdir()
    entry = root / "entry.py"
    entry.write_text("import sys, json\nprint(json.dumps(sys.argv[1:]))\n")
    argv, orig, module = [str(entry), *APP], [sys.executable, str(entry), *APP], None
    if mode == "module":
        module, orig = "entry", [sys.executable, "-m", "entry", *APP]
    elif mode == "command":
        argv, orig = ["-c", *APP], [sys.executable, "-c", "import entry", *APP]

    command = venv_sync.relaunch_command(Path(sys.executable), root, argv, orig, module)

    # What /proc/<pid>/cmdline and `ps` expose: the application argv as separate elements.
    assert command[-len(APP):] == APP
    assert _desktop_owned(command, NONCE, "academic")
    # What the application sees: exactly its argv, not a doubled copy.
    out = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=30, check=True)
    assert json.loads(out.stdout) == APP


@pytest.mark.skipif(not Path("/proc/self/cmdline").exists(), reason="needs /proc")
def test_running_reentered_process_proves_ownership(tmp_path):
    root = tmp_path / "src"
    root.mkdir()
    entry = root / "entry.py"
    entry.write_text("import os\nprint(open(f'/proc/{os.getpid()}/cmdline','rb').read().hex())\n")
    command = venv_sync.relaunch_command(
        Path(sys.executable), root, [str(entry), *APP], [sys.executable, str(entry), *APP], None)
    out = subprocess.run(command, capture_output=True, text=True, timeout=30, check=True, env=os.environ)
    live = [x.decode() for x in bytes.fromhex(out.stdout.strip()).split(b"\0") if x]
    assert _desktop_owned(live, NONCE, "academic")
    assert not _desktop_owned(live, "b" * 16, "academic")
