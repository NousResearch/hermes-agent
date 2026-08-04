"""Safe guidance for multiline input in the classic Hermes CLI.

``hermes terminal-setup`` is intentionally informational: terminal keyboard
shortcuts belong to the user's terminal emulator, so this helper never edits
terminal, shell, or system configuration.
"""

from __future__ import annotations

import os


def detect_terminal() -> str:
    """Return a best-effort terminal emulator identifier from its environment."""
    term_program = os.environ.get("TERM_PROGRAM", "").casefold()
    term = os.environ.get("TERM", "").casefold()

    if os.environ.get("WT_SESSION"):
        return "windows-terminal"
    if os.environ.get("VSCODE_PID") or "vscode" in term_program:
        return "vscode"
    if "iterm" in term_program or "iterm" in os.environ.get("LC_TERMINAL", "").casefold():
        return "iterm2"
    if "kitty" in term or os.environ.get("KITTY_WINDOW_ID"):
        return "kitty"
    if "wezterm" in term_program or "wezterm" in os.environ.get("TERM_EMULATOR", "").casefold():
        return "wezterm"
    if "ghostty" in term_program or os.environ.get("GHOSTTY_RESOURCES_DIR"):
        return "ghostty"
    return "unknown"


def _terminal_label(terminal: str) -> str:
    return {
        "windows-terminal": "Windows Terminal",
        "vscode": "VS Code-family integrated terminal",
        "iterm2": "iTerm2",
        "kitty": "kitty",
        "wezterm": "WezTerm",
        "ghostty": "Ghostty",
        "unknown": "an unrecognised terminal",
    }[terminal]


def run_terminal_setup(args=None) -> None:  # noqa: ARG001
    """Print non-destructive classic-CLI multiline-input guidance."""
    terminal = detect_terminal()
    print("Hermes terminal setup (classic CLI)")
    print("This command only prints guidance; it does not change terminal, shell, or system configuration.")
    print(f"Detected: {_terminal_label(terminal)}.")
    print()
    print("For multiline input in the classic `hermes` CLI:")
    print("  • Shift+Enter works when the terminal emits a distinct modified-Enter sequence.")
    print("  • Hermes recognises Kitty CSI-u (ESC [ 13 ; 2 u) and xterm modifyOtherKeys")
    print("    (ESC [ 27 ; 2 ; 13 ~) Shift+Enter sequences.")

    if terminal == "windows-terminal":
        print("  • In Windows Terminal, use Ctrl+Enter for a newline if Alt+Enter is intercepted.")
    else:
        print("  • Alt+Enter is the fallback newline shortcut when the terminal passes it through.")

    print()
    print("If Shift+Enter submits instead, configure your terminal emulator to send one of")
    print("the sequences above, then open a new terminal session and retry. No Hermes")
    print("configuration change is required.")
