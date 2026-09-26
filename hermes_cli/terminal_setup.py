"""Safe terminal-side setup for classic Hermes CLI multiline input."""

from __future__ import annotations

import errno
import inspect
import os
import secrets
import stat
import sys
from pathlib import Path

_START = "# >>> hermes terminal-setup >>>"
_END = "# <<< hermes terminal-setup <<<"
_SEQUENCE = r"\x1b[13;2u"


class SecurePersistenceUnavailable(RuntimeError):
    """Raised when this platform cannot persist managed config safely."""


def detect_terminal() -> str:
    """Return a best-effort terminal emulator identifier from its environment."""
    term_program = os.environ.get("TERM_PROGRAM", "").casefold()
    terminal = os.environ.get("TERM", "").casefold()

    if os.environ.get("WT_SESSION"):
        return "windows-terminal"
    if os.environ.get("VSCODE_PID") or "vscode" in term_program:
        return "vscode"
    if (
        "iterm" in term_program
        or "iterm" in os.environ.get("LC_TERMINAL", "").casefold()
    ):
        return "iterm2"
    if term_program == "apple_terminal":
        return "apple-terminal"
    if "kitty" in terminal or os.environ.get("KITTY_WINDOW_ID"):
        return "kitty"
    if (
        "wezterm" in term_program
        or "wezterm" in os.environ.get("TERM_EMULATOR", "").casefold()
    ):
        return "wezterm"
    if "ghostty" in term_program or os.environ.get("GHOSTTY_RESOURCES_DIR"):
        return "ghostty"
    return "unknown"


def is_native_windows() -> bool:
    """Whether Hermes is running with prompt_toolkit's Win32 input backend."""
    return os.name == "nt"


def managed_config_path(terminal: str) -> Path:
    """Return the user-owned config path for terminals we can edit safely."""
    config_root = managed_config_root(terminal)
    if terminal == "kitty":
        return config_root / "kitty" / "kitty.conf"
    if terminal == "ghostty":
        if sys.platform == "darwin" and not os.environ.get("XDG_CONFIG_HOME"):
            return config_root / "config"
        return config_root / "ghostty" / "config"
    raise ValueError(f"{terminal!r} has no managed configuration path")


def managed_config_root(terminal: str) -> Path:
    """Return the absolute user config root for a managed terminal."""
    if (
        terminal == "ghostty"
        and sys.platform == "darwin"
        and not os.environ.get("XDG_CONFIG_HOME")
    ):
        return Path.home() / "Library/Application Support/com.mitchellh.ghostty"
    xdg_home = os.environ.get("XDG_CONFIG_HOME")
    if xdg_home:
        root = Path(xdg_home)
        if not root.is_absolute():
            raise ValueError("XDG_CONFIG_HOME must be an absolute path")
        if root.is_symlink():
            raise ValueError(
                "XDG_CONFIG_HOME config root is a symlink; refusing to modify it"
            )
        return root
    return Path.home() / ".config"


def managed_block(terminal: str) -> str:
    """Return the exact, deliberately narrow mapping managed by Hermes."""
    mapping = {
        "kitty": f"map shift+enter send_text all {_SEQUENCE}",
        "ghostty": f"keybind = shift+enter=text:{_SEQUENCE}",
    }[terminal]
    return f"{_START}\n{mapping}\n{_END}"


def update_managed_block(content: str, terminal: str) -> str:
    """Replace Hermes' block, leaving all unrelated user configuration intact."""
    block = managed_block(terminal)
    marker_lines = [
        line for line in content.splitlines() if _START in line or _END in line
    ]
    if not marker_lines:
        separator = "" if not content or content.endswith("\n") else "\n"
        return f"{content}{separator}{block}\n"
    if len(marker_lines) != 2 or marker_lines != [_START, _END]:
        raise ValueError(
            "invalid Hermes terminal-setup markers; refusing to modify config"
        )
    start = content.find(_START)
    end = content.find(_END, start)
    end += len(_END)
    return f"{content[:start]}{block}{content[end:]}"


def _snippet(terminal: str) -> str | None:
    snippets = {
        "wezterm": """local wezterm = require 'wezterm'

return {
  keys = {
    { key = 'Enter', mods = 'SHIFT', action = wezterm.action.SendString '\\x1b[13;2u' },
  },
}""",
        "iterm2": """iTerm2 → Settings → Profiles → Keys → Key Mappings → +
Keyboard Shortcut: Shift+Return
Action: Send Escape Sequence
Esc+ field (paste this value): [13;2u""",
        "windows-terminal": """// settings.json: add this object to the existing "actions" array
{
  "command": { "action": "sendInput", "input": "\\u001b[13;2u" },
  "keys": "shift+enter"
}""",
        "vscode": """// keybindings.json
{
  "key": "shift+enter",
  "command": "workbench.action.terminal.sendSequence",
  "args": { "text": "\\u001b[13;2u" },
  "when": "terminalFocus"
}""",
    }
    return snippets.get(terminal)


def _print_intro(terminal: str) -> None:
    labels = {
        "windows-terminal": "Windows Terminal",
        "vscode": "VS Code-family integrated terminal",
        "iterm2": "iTerm2",
        "apple-terminal": "Apple Terminal",
        "kitty": "kitty",
        "wezterm": "WezTerm",
        "ghostty": "Ghostty",
        "unknown": "an unrecognised terminal",
    }
    print("Hermes terminal setup (classic CLI)")
    print(f"Detected: {labels[terminal]}.")
    print(
        "Shift+Enter needs a terminal mapping to ESC [ 13 ; 2 u. Hermes never enables"
    )
    print(
        "a global keyboard protocol: only this chord is mapped, preserving Ctrl+C, Esc, and Alt keys."
    )


def _print_windows_notice() -> None:
    print()
    print(
        "Native Windows limitation: prompt_toolkit Win32 backend cannot enable app-side Shift+Enter."
    )
    print("Use Ctrl+Enter (or Ctrl+J) for a newline.")
    print(
        "Use a mapping only in WSL or a remote Unix host, where the Unix input decoder"
    )
    print("receives the sequence.")


def _safe_managed_config_path(terminal: str) -> Path:
    """Resolve a managed path without crossing config-root or symlink boundaries."""
    root = managed_config_root(terminal)
    path = managed_config_path(terminal)
    if not root.is_absolute() or not path.is_absolute():
        raise ValueError("managed config paths must be absolute")
    _reject_symlink_components(root, "config root")
    _reject_symlink_components(path, "managed config path")
    root = root.resolve()
    path = path.resolve(strict=False)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "managed config path resolves outside its config root"
        ) from exc
    return path


def _reject_symlink_components(path: Path, label: str) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{label} contains a symlink; refusing to modify it")


def _secure_persistence_available() -> bool:
    """Whether this runtime has the descriptor APIs needed for safe writes."""
    required_flags = ("O_DIRECTORY", "O_NOFOLLOW", "O_CREAT", "O_EXCL")
    required_operations = (os.open, os.mkdir, os.unlink)
    required_functions = ("fchmod", "fsync")
    try:
        replace_parameters = inspect.signature(os.replace).parameters
    except (TypeError, ValueError):
        return False
    return (
        os.name != "nt"
        and all(hasattr(os, flag) for flag in required_flags)
        and all(hasattr(os, function) for function in required_functions)
        and all(operation in os.supports_dir_fd for operation in required_operations)
        and all(
            parameter in replace_parameters
            and replace_parameters[parameter].kind
            is not inspect.Parameter.POSITIONAL_ONLY
            for parameter in ("src_dir_fd", "dst_dir_fd")
        )
    )


def _open_directory_no_follow(path: Path) -> int:
    """Open or create an absolute directory path without traversing symlinks."""
    if not _secure_persistence_available():
        raise SecurePersistenceUnavailable(
            "descriptor-relative no-follow config persistence is unavailable"
        )
    if not path.is_absolute():
        raise ValueError("managed config directories must be absolute")

    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_fd = os.open(path.anchor, flags)
    try:
        for component in path.parts[1:]:
            try:
                os.mkdir(component, 0o700, dir_fd=directory_fd)
            except FileExistsError:
                pass
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
            child_fd = os.open(component, flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = child_fd
        return directory_fd
    except BaseException:
        os.close(directory_fd)
        raise


def _read_regular_file_no_follow(directory_fd: int, basename: str) -> tuple[str, int]:
    """Read a regular config file relative to an already-open directory."""
    try:
        file_fd = os.open(basename, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
    except FileNotFoundError:
        return "", 0o600
    try:
        metadata = os.fstat(file_fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("managed config target is not a regular file")
        with os.fdopen(file_fd, "r", encoding="utf-8") as config_file:
            file_fd = -1
            return config_file.read(), stat.S_IMODE(metadata.st_mode)
    finally:
        if file_fd != -1:
            os.close(file_fd)


def _secure_update_managed_config(path: Path, terminal: str) -> bool:
    """Atomically update ``path`` without following any directory or target symlink."""
    basename = path.name
    if basename in {"", ".", ".."} or len(path.parts) < 2:
        raise ValueError("managed config target must be a file beneath its config root")

    directory_fd = _open_directory_no_follow(path.parent)
    temporary_name = ""
    try:
        existing, mode = _read_regular_file_no_follow(directory_fd, basename)
        updated = update_managed_block(existing, terminal)
        if updated == existing:
            return False

        temporary_name = f".{basename}.hermes-{secrets.token_hex(16)}.tmp"
        temporary_fd = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            mode,
            dir_fd=directory_fd,
        )
        try:
            os.fchmod(temporary_fd, mode)
            with os.fdopen(temporary_fd, "w", encoding="utf-8") as temporary_file:
                temporary_fd = -1
                temporary_file.write(updated)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
        finally:
            if temporary_fd != -1:
                os.close(temporary_fd)

        os.replace(
            temporary_name,
            basename,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        temporary_name = ""
        return True
    finally:
        if temporary_name:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        os.close(directory_fd)


def _apply_managed_config(terminal: str, *, dry_run: bool, print_config: bool) -> None:
    path = _safe_managed_config_path(terminal)
    block = managed_block(terminal)

    if print_config:
        print(f"\nPaste into {path}:\n{block}")
    if dry_run:
        print(f"\nWould update {path} with:\n{block}")
        return
    if print_config:
        return
    if not _secure_persistence_available():
        # Preserve marker validation even when this runtime must fail closed.
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        update_managed_block(existing, terminal)
    try:
        changed = _secure_update_managed_config(path, terminal)
    except SecurePersistenceUnavailable as exc:
        print(
            f"\nCannot safely update {path}: {exc}. "
            "No files were written; rerun with --print and paste the mapping manually."
        )
        return
    if not changed:
        print(f"\nAlready configured: {path}")
        return
    print(f"\nUpdated {path}; only the marked Hermes block was changed.")


def run_terminal_setup(args=None) -> None:
    """Configure one safe terminal mapping or print a terminal-specific snippet."""
    dry_run = bool(getattr(args, "dry_run", False))
    print_config = bool(getattr(args, "print_config", False))
    terminal = detect_terminal()
    _print_intro(terminal)

    if is_native_windows():
        _print_windows_notice()
        return

    if terminal in {"kitty", "ghostty"}:
        _apply_managed_config(terminal, dry_run=dry_run, print_config=print_config)
    elif snippet := _snippet(terminal):
        print(
            "\nPaste this single-chord mapping into your existing configuration; Hermes will not edit it:"
        )
        print(snippet)
        if terminal == "wezterm":
            print(
                "Merge the keys entry with existing keys rather than adding a second return table."
            )
        elif terminal == "vscode":
            print(
                "For VS Code forks, use the equivalent keybindings JSON command if its command ID differs."
            )
        elif terminal == "windows-terminal":
            print(
                "This requires a Windows Terminal version with the sendInput action; open a new tab afterward."
            )
    elif terminal == "apple-terminal":
        print(
            "\nApple Terminal cannot distinguish Shift+Enter. Use Option+Enter or Ctrl+J for a newline."
        )
    else:
        print(
            "\nThis terminal is not recognised. If it supports a per-key mapping, map only Shift+Enter"
        )
        print(r"to \x1b[13;2u; otherwise use Option+Enter or Ctrl+J for a newline.")

    if dry_run or print_config:
        print(
            "\nNo files were written (--dry-run and --print are compatible preview modes)."
        )
    print("Open a new terminal session after changing terminal configuration.")
