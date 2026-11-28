"""``hermes config edit``: open the local ``config.yaml`` in the user's editor, seeding it first."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys


def editor_argv(editor: str) -> list[str]:
    """``$EDITOR`` as argv, since it may carry flags (``code --wait``, ``omarchy-launch-editor --inline``).

    Windows keeps the whole value: POSIX shlex eats the backslashes of ``C:\\...\\notepad.exe``.
    Unbalanced quotes give ``[]``, so the caller reports it instead of retrying through a shell."""
    if sys.platform == "win32":
        return [editor]
    try:
        return shlex.split(editor)
    except ValueError:
        return []


def edit_config():
    """Open config file in user's editor."""
    from hermes_cli.config import (
        _exit_invalid, get_config_path, is_managed, managed_error, seed_config_file, supports_file_tooling,
    )

    if is_managed():
        managed_error("edit configuration")
        return
    if not supports_file_tooling():
        # Exits 1 (unlike the managed refusal above, which every managed_error caller shares):
        # a script must not read this refusal as a successful edit.
        _exit_invalid("`hermes config edit` opens the local config file, and this config backend has none. "
                      "Use `hermes config set <key> <value>` instead.")
    config_path = get_config_path()
    if not config_path.exists():  # config-reader: ok — file tooling, gated on supports_file_tooling()
        seed_config_file(config_path)
        print(f"Created {config_path}")

    # Windows lands on notepad even without Git Bash/nano; POSIX prefers nano/vim, which headless
    # servers are more likely to have.
    candidates = (['notepad', 'code', 'vim', 'vi', 'nano'] if sys.platform == "win32"
                  else ['nano', 'vim', 'vi', 'code', 'notepad'])
    editor = os.getenv('EDITOR') or os.getenv('VISUAL') or next(
        (cmd for cmd in candidates if shutil.which(cmd)), None)
    if not editor:
        print("No editor found. Config file is at:")
        print(f"  {config_path}")
        return
    argv = editor_argv(editor)
    if not argv:
        print(f"Cannot run $EDITOR ({editor!r}). Config file is at:")
        print(f"  {config_path}")
        return

    print(f"Opening {config_path} in {editor}...")
    subprocess.run([*argv, str(config_path)], check=False)  # health: allow HX006 -- interactive editor, runs until the user closes it
