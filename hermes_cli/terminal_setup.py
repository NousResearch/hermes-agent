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


def _terminal_guidance(terminal: str) -> tuple[str, ...]:
    """Return safe, terminal-specific steps without mutating user settings."""
    guidance = {
        "iterm2": (
            "iTerm2: open Settings → Profiles → Keys and enable ‘Report modifiers using CSI u’.",
            "If you use a custom key mapping, ensure Shift+Return is not mapped to plain Return.",
        ),
        "vscode": (
            "VS Code: open Preferences: Open User Settings (JSON) and add:",
            '  "terminal.integrated.enableKittyKeyboardProtocol": true',
            "Open a new integrated terminal after saving the setting.",
        ),
        "windows-terminal": (
            "Windows Terminal: Kitty keyboard protocol support requires Windows Terminal Preview 1.25+.",
            "Update to Preview and open a new tab; stable Windows Terminal cannot distinguish Shift+Enter.",
        ),
        "kitty": ("kitty: Kitty keyboard reporting is enabled by default; open a new shell if needed.",),
        "wezterm": ("WezTerm: Kitty keyboard reporting is enabled by default; open a new pane if needed.",),
        "ghostty": ("Ghostty: Kitty keyboard reporting is enabled by default; open a new terminal if needed.",),
        "unknown": (
            "Check the terminal’s keyboard settings for Kitty keyboard protocol or xterm modifyOtherKeys support.",
            "If it cannot emit modified Enter sequences, use Alt+Enter or Ctrl+J instead.",
        ),
    }
    return guidance[terminal]


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

    print()
    print("Terminal-specific setup:")
    for line in _terminal_guidance(terminal):
        print(f"  • {line}")

    if terminal == "windows-terminal":
        print("  • Use Ctrl+Enter (delivered as Ctrl+J) or Ctrl+J directly for a newline.")
    else:
        print("  • Alt+Enter is the fallback newline shortcut when the terminal passes it through.")

    print()
    print("After changing settings, open a new terminal session and retry. No Hermes")
    print("configuration change is required.")
