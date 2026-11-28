"""Interactive and explicit per-user installation of static shell completions."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import sys
from typing import Literal

from hermes_cli import completion
from utils import atomic_write_bytes


Shell = Literal["bash", "zsh", "fish"]
Mode = Literal["install", "manual", "raw"]
_SHELLS: tuple[Shell, ...] = ("bash", "zsh", "fish")
_GENERATORS = {"bash": completion.generate_bash, "zsh": completion.generate_zsh,
               "fish": completion.generate_fish}
_START = b"# >>> hermes completion >>>"
_END = b"# <<< hermes completion <<<"


@dataclass(frozen=True)
class CompletionRequest:
    shell: Shell
    mode: Mode
    confirm: bool


@dataclass(frozen=True)
class FileChange:
    path: Path
    destination: Path
    before: bytes | None
    content: bytes


@dataclass(frozen=True)
class CompletionPlan:
    shell: Shell
    script: Path
    rc: Path | None
    initialization: str


def _choose(title: str, items: list[str]) -> int | None:
    from hermes_cli.curses_ui import curses_single_select

    return curses_single_select(title, items)


def _request(args) -> CompletionRequest | None:
    positional = getattr(args, "shell", None)
    selected = getattr(args, "shell_option", None)
    mode = getattr(args, "mode", None)
    if positional and selected and positional != selected:
        raise ValueError("conflicting positional shell and --shell; choose the same shell or use only --shell")
    shell = selected or positional
    if positional and mode is None:
        mode = "raw"
    if mode == "raw" and shell is None:
        raise ValueError("--raw requires --shell bash|zsh|fish (or a positional shell)")
    needs_menu = shell is None or mode is None
    if needs_menu and not (sys.stdin.isatty() and sys.stdout.isatty()):
        raise ValueError("choose --shell bash|zsh|fish and --install, --manual, or --raw; "
                         "interactive setup requires both stdin and stdout to be terminals")
    if shell is None:
        index = _choose("Shell completion: choose your shell", ["Bash", "Zsh", "Fish"])
        if index is None:
            return None
        shell = _SHELLS[index]
    if mode is None:
        index = _choose(f"Set up {shell} completions", ["Automatic installation", "Manual setup instructions"])
        if index is None:
            return None
        mode = ("install", "manual")[index]
    return CompletionRequest(shell, mode, needs_menu and mode == "install")


def _environment_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if not value:
        return default
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute path for completion setup; unset it or use an absolute path")
    return path


def _quote(path: Path, shell: Shell) -> str:
    value = str(path)
    if shell == "fish":
        return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"
    return shlex.quote(value)


def _plan(shell: Shell) -> CompletionPlan:
    home = Path.home()
    if shell == "fish":
        config = _environment_path("XDG_CONFIG_HOME", home / ".config")
        return CompletionPlan(shell, config / "fish/completions/hermes.fish", None, "")
    data = _environment_path("XDG_DATA_HOME", home / ".local/share")
    if shell == "bash":
        script = data / "bash-completion/completions/hermes"
        path = _quote(script, shell)
        return CompletionPlan(shell, script, home / ".bashrc",
                              f"if [ -r {path} ]; then\n    source {path}\nfi\n")
    script = data / "zsh/completions/_hermes"
    path = _quote(script, shell)
    rc = _environment_path("ZDOTDIR", home) / ".zshrc"
    # The generator ends in compdef, so source it after compinit, not via fpath autoload.
    initialization = (f"if [ -r {path} ]; then\n"
                      "    if (( ! $+functions[compdef] )); then\n"
                      "        autoload -Uz compinit\n"
                      "        compinit\n"
                      "    fi\n"
                      f"    source {path}\nfi\n")
    return CompletionPlan(shell, script, rc, initialization)


def _merge_initialization(before: bytes, initialization: str, rc: Path) -> bytes:
    lines = before.splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if _START in line]
    ends = [i for i, line in enumerate(lines) if _END in line]
    if starts or ends:
        valid = (len(starts) == len(ends) == 1 and starts[0] < ends[0]
                 and lines[starts[0]].rstrip(b"\r\n") == _START
                 and lines[ends[0]].rstrip(b"\r\n") == _END)
        if not valid:
            raise ValueError(f"malformed Hermes completion markers in {rc}; "
                             "repair or remove that managed block before retrying --install")
    # Shell control syntax requires LF, even when unrelated comments use CRLF.
    newline = b"\n"
    block = newline.join((_START, *initialization.rstrip("\n").encode().split(b"\n"), _END, b""))
    if starts:
        return b"".join(lines[:starts[0]]) + block + b"".join(lines[ends[0] + 1:])
    separator = newline if before and not before.endswith(b"\n") else b""
    return before + separator + block


def _read_existing(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None


def _changes(plan: CompletionPlan, script: str) -> tuple[FileChange, ...]:
    destination = plan.script.resolve()
    changes = [FileChange(plan.script, destination, _read_existing(destination), script.encode())]
    if plan.rc is not None:
        destination = plan.rc.resolve()
        before = _read_existing(destination)
        merged = _merge_initialization(before or b"", plan.initialization, plan.rc)
        changes.append(FileChange(plan.rc, destination, before, merged))
    if len({change.destination for change in changes}) != len(changes):
        raise ValueError("completion script and shell rc resolve to the same file; "
                         "fix the symlink or XDG/ZDOTDIR paths before retrying")
    return tuple(changes)


def _plan_text(plan: CompletionPlan, changes: tuple[FileChange, ...]) -> str:
    lines = [f"{plan.shell} completion installation plan:"]
    for change in changes:
        state = "unchanged" if change.before == change.content else "write"
        lines.append(f"  {state}: {change.path}")
        if change.destination != change.path:
            lines.append(f"    resolved destination: {change.destination}")
    if plan.rc is not None:
        lines.extend(("Managed initialization (unrelated shell configuration is preserved):",
                      _START.decode(), plan.initialization.rstrip("\n"), _END.decode()))
    return "\n".join(lines)


def _confirm_install(description: str) -> bool:
    from hermes_cli.curses_ui import curses_radiolist

    return curses_radiolist("Apply this completion installation plan?", ["Install completions", "Cancel"],
                            selected=1, cancel_returns=1, description=description) == 0


def _activation(plan: CompletionPlan) -> str:
    if plan.shell == "zsh":
        return plan.initialization.rstrip("\n")
    return f"source {_quote(plan.script, plan.shell)}"


def _manual(plan: CompletionPlan) -> None:
    print(f"Manual {plan.shell} completion setup (no files have been changed).")
    print("Run these commands to save a static script:")
    print(f"    mkdir -p {_quote(plan.script.parent, plan.shell)}")
    print(f"    hermes completion --shell {plan.shell} --raw > {_quote(plan.script, plan.shell)}")
    if plan.rc is not None:
        print(f"Add this initialization to {plan.rc} once (do not append the whole generated script):")
        print(_START.decode())
        print(plan.initialization, end="")
        print(_END.decode())
    print("Start a new shell, or activate in the current shell:")
    print(_activation(plan))
    print("After updating Hermes, rerun completion setup to refresh the static script.")


def _install(plan: CompletionPlan, script: str, confirm: bool) -> None:
    changes = _changes(plan, script)
    description = _plan_text(plan, changes)
    print(description, flush=True)
    if confirm and not _confirm_install(description):
        print("Completion setup cancelled; no shell files changed.")
        return
    # Validate every destination before writing any: a malformed rc or an edit made
    # while the confirmation was open must not leave a half-installed setup.
    for change in changes:
        if change.path.resolve() != change.destination or _read_existing(change.destination) != change.before:
            raise ValueError(f"{change.path} changed while planning; rerun --install to review the new plan")
    for change in changes:
        if change.before != change.content:
            atomic_write_bytes(change.destination, change.content)
            print(f"Updated: {change.path}")
    print("Completions installed. Start a new shell, or activate in the current shell:")
    print(_activation(plan))
    print("After updating Hermes, rerun completion setup to refresh the static script.")


def run_completion(args, parser) -> int:
    """Dispatch without touching shell files or opening menus in raw mode."""
    try:
        request = _request(args)
        if request is None:
            print("Completion setup cancelled; no shell files changed.")
            return 0
        if request.mode == "raw":
            sys.stdout.write(_GENERATORS[request.shell](parser))
            return 0
        plan = _plan(request.shell)
        if request.mode == "manual":
            _manual(plan)
        else:
            _install(plan, _GENERATORS[request.shell](parser), request.confirm)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"hermes completion: {exc}", file=sys.stderr)
        return 1
    return 0
