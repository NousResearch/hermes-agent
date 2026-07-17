"""Interactive terminal setup for Shift+Enter newline support."""

from __future__ import annotations

import os
import subprocess
import sys


def _detect_terminal() -> str:
    """Return a short canonical token for the running terminal emulator.

    Checks common environment variables in priority order.  Returns one of:
    ``iterm2``, ``terminal_app``, ``kitty``, ``wezterm``, ``ghostty``,
    ``vscode``, or ``unknown``.
    """
    term_program = os.environ.get("TERM_PROGRAM", "").lower()
    lc_terminal = os.environ.get("LC_TERMINAL", "").lower()
    term = os.environ.get("TERM", "").lower()
    term_emulator = os.environ.get("TERM_EMULATOR", "").lower()

    # iTerm2 sets TERM_PROGRAM=iTerm.app
    if "iterm" in term_program or "iterm" in lc_terminal:
        return "iterm2"

    # macOS Terminal.app
    if term_program == "apple_terminal":
        return "terminal_app"

    # kitty sets TERM=xterm-kitty
    if "kitty" in term or os.environ.get("KITTY_WINDOW_ID"):
        return "kitty"

    # WezTerm
    if "wezterm" in term_program or "wezterm" in term_emulator:
        return "wezterm"

    # Ghostty
    if "ghostty" in term_program or os.environ.get("GHOSTTY_RESOURCES_DIR"):
        return "ghostty"

    # VS Code integrated terminal
    if "vscode" in term_program or os.environ.get("VSCODE_PID"):
        return "vscode"

    return "unknown"


_ITERM2_DOMAIN = "com.googlecode.iterm2"


def _iterm2_has_shift_return_keybinding() -> bool:
    """Return True if iTerm2 has a GlobalKeyMap binding that intercepts Shift+Return.

    Checks the raw ``defaults read`` output for the key code pattern
    ``"0x-0xd-0x20000"`` (Shift+Return in iTerm2's key encoding).
    """
    try:
        result = subprocess.run(
            ["defaults", "read", _ITERM2_DOMAIN, "GlobalKeyMap"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return False
        # iTerm2 encodes Shift+Return as "0xd-0x20000" or "0x-0xd-0x20000"
        raw = result.stdout
        # Look for entries containing Return (keycode 0xd) + Shift modifier (0x20000)
        shift_return_patterns = [
            "0xd-0x20000",   # Shift+Return
            "0x-0xd-0x20000",  # alternate encoding
        ]
        return any(p in raw for p in shift_return_patterns)
    except Exception:
        return False


_RESET = "\033[0m"
_BOLD = "\033[1m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_RED = "\033[31m"
_DIM = "\033[2m"

def _plain() -> bool:
    """Return True when colour output should be suppressed."""
    return not sys.stdout.isatty() or bool(os.environ.get("NO_COLOR"))


def _p(text: str, colour: str = "") -> None:
    if _plain() or not colour:
        print(text)
    else:
        print(f"{colour}{text}{_RESET}")


def _header(text: str) -> None:
    print()
    _p(f"  {text}", _BOLD + _CYAN)
    _p("  " + "─" * len(text), _DIM)


def _ok(text: str) -> None:
    _p(f"  ✓  {text}", _GREEN)


def _warn(text: str) -> None:
    _p(f"  ⚠  {text}", _YELLOW)


def _info(text: str) -> None:
    _p(f"     {text}")


def _err(text: str) -> None:
    _p(f"  ✗  {text}", _RED)


def _ask(prompt: str, default: str = "y") -> str:
    """Prompt the user for a y/n answer.  Returns 'y' or 'n'."""
    hint = "[Y/n]" if default.lower() == "y" else "[y/N]"
    try:
        answer = input(f"  {_CYAN}?{_RESET}  {prompt} {hint}: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return default.lower()
    return answer if answer in ("y", "n") else default.lower()


def _wizard_iterm2() -> None:
    """Full iTerm2 setup wizard for Shift+Enter → newline."""
    _header("iTerm2 detected")
    _info("Hermes supports native Shift+Enter → newline in iTerm2 via the Kitty")
    _info("keyboard protocol (CSI-u sequences).  Two things must be in place:")
    _info("")
    _info("  1. 'Report modifiers using CSI u' must be ON  (Profiles → Keys)")
    _info("  2. No GlobalKeyMap entry must intercept Shift+Return")
    _info("")

    _header("Step 1 — Checking GlobalKeyMap for conflicting Shift+Return binding")
    has_conflict = _iterm2_has_shift_return_keybinding()
    if has_conflict:
        _warn("Found a GlobalKeyMap entry that intercepts Shift+Return.")
        _info("This binding captures the key before iTerm2 can emit the CSI-u")
        _info("sequence, so Hermes never sees it. Remove only the conflicting")
        _info("Shift+Return mapping; do not delete the entire GlobalKeyMap.")
        _info("")
        _info("Manual fix:")
        _info("  1. Open iTerm2 → Settings/Preferences → Keys → Key Mappings")
        _info("  2. Look for a global Shift+Return / Shift+Enter entry; also")
        _info("     check Profiles → Keys for an overriding profile mapping")
        _info("  3. Remove that single mapping, or change it so it no longer")
        _info("     intercepts Return")
        _info("  4. If you manage settings via defaults, inspect first with:")
        _info("     defaults read com.googlecode.iterm2 GlobalKeyMap")
        _info("")
        _warn("Shift+Enter will not work until that conflicting mapping is removed.")
    else:
        _ok("No conflicting GlobalKeyMap binding found.")

    _header("Step 2 — Enable 'Report modifiers using CSI u' in iTerm2")
    _info("This setting cannot be changed from the command line; you must")
    _info("toggle it in iTerm2 Preferences:")
    _info("")
    _info("  1. Open iTerm2 → Preferences  (Cmd+,)")
    _info("  2. Click  Profiles")
    _info("  3. Select the profile you use  (usually 'Default')")
    _info("  4. Click the  Keys  tab")
    _info("  5. At the bottom, find  'Report modifiers using CSI u'")
    _info("  6. Check the box if it is not already checked")
    _info("  7. Close Preferences — no restart needed")
    _info("")
    _info("  Tip: the option is near the bottom of the Keys tab,")
    _info("  below the 'Key Mappings' table and 'Presets' dropdown.")
    _info("")

    # ── Final validation advice ─────────────────────────────────────────────
    _header("Validation")
    _info("After completing Step 2:")
    _info("")
    _info("  • Open a new iTerm2 tab after changing the key mapping.")
    _info("  • Run  hermes  and press Shift+Enter inside the prompt.")
    _info("  • A newline should be inserted without submitting the message.")
    _info("")
    _info("If Shift+Enter still submits, double-check that 'Report modifiers")
    _info("using CSI u' is enabled for your active profile.")
    _info("")
    _info("Alt+Enter always works as an unconditional fallback.")
    _info("")


def _wizard_terminal_app() -> None:
    """Inform the user that macOS Terminal.app is unsupported."""
    _header("macOS Terminal.app detected")
    _warn("Terminal.app does not support the Kitty keyboard protocol (CSI-u).")
    _info("")
    _info("Terminal.app sends the same byte sequence for Enter and Shift+Enter,")
    _info("so Hermes cannot distinguish the two at the application level.")
    _info("There is no terminal-side fix for this limitation.")
    _info("")
    _ok("Alt+Enter always works — press Escape then Enter, or hold Option")
    _ok("and press Enter, to insert a newline without submitting.")
    _info("")
    _info("If native Shift+Enter newlines are important to you, consider")
    _info("switching to a terminal that supports the Kitty keyboard protocol:")
    _info("")
    _info("  • iTerm2   https://iterm2.com  (macOS, free)")
    _info("  • kitty    https://sw.kovidgoyal.net/kitty/  (macOS/Linux, free)")
    _info("  • WezTerm  https://wezfurlong.org/wezterm/  (macOS/Linux/Win, free)")
    _info("  • Ghostty  https://ghostty.org  (macOS/Linux, free)")
    _info("")


def _wizard_modern_terminal(name: str) -> None:
    """Confirm that a modern terminal already works."""
    _header(f"{name} detected")
    _ok(f"{name} supports the Kitty keyboard protocol (CSI-u) natively.")
    _info("")
    _info("Shift+Enter should already insert a newline in Hermes without")
    _info("any additional configuration.")
    _info("")
    _info("If it does not work:")
    _info("  • Ensure Hermes is on a recent version  (hermes update)")
    _info("  • Verify the keyboard protocol is not disabled in your config")
    _info("  • Alt+Enter is always available as a fallback")
    _info("")


def _wizard_vscode() -> None:
    """VS Code integrated terminal guidance."""
    _header("VS Code / Cursor / Windsurf integrated terminal detected")
    _info("The classic CLI can use Shift+Enter here only when the integrated")
    _info("terminal is configured to emit a distinct modified Enter sequence.")
    _info("")
    _info("Per the CLI guide, VS Code-family terminals support this once the")
    _info("Kitty keyboard protocol is enabled in terminal settings.")
    _info("If Shift+Enter still submits, enable the terminal's keyboard")
    _info("protocol setting and open a fresh integrated terminal tab.")
    _info("")
    _info("Alt+Enter is the general classic-CLI fallback for inserting a")
    _info("newline. In Windows Terminal, Alt+Enter toggles fullscreen, so")
    _info("use Ctrl+Enter or Ctrl+J there instead.")
    _info("")


def _wizard_unknown() -> None:
    """Generic guidance for unrecognised terminals."""
    _header("Terminal emulator not recognised")
    _info(f"TERM_PROGRAM={os.environ.get('TERM_PROGRAM', '(unset)')!r}  "
          f"TERM={os.environ.get('TERM', '(unset)')!r}")
    _info("")
    _info("Hermes uses the Kitty keyboard protocol (CSI-u) to detect")
    _info("Shift+Enter.  For this to work your terminal must:")
    _info("")
    _info("  • Emit \\x1b[13;2u  (Kitty CSI-u Shift+Enter) or")
    _info("    \\x1b[27;2;13~  (xterm modifyOtherKeys Shift+Enter)")
    _info("")
    _info("Terminals known to support this out of the box:")
    _info("  kitty, foot, WezTerm, Ghostty")
    _info("")
    _info("Supported once terminal keyboard-protocol settings are enabled:")
    _info("  iTerm2, Alacritty, VS Code terminal, Warp")
    _info("")
    _info("Alt+Enter is always available as a fallback.")
    _info("")


def run_terminal_setup(args=None) -> None:
    """Run the interactive terminal-setup wizard."""
    print()
    _p("  Hermes terminal-setup — Shift+Enter newline configuration", _BOLD)
    _p("  " + "═" * 54, _DIM)

    terminal = _detect_terminal()

    if terminal == "iterm2":
        _wizard_iterm2()
    elif terminal == "terminal_app":
        _wizard_terminal_app()
    elif terminal == "kitty":
        _wizard_modern_terminal("kitty")
    elif terminal == "wezterm":
        _wizard_modern_terminal("WezTerm")
    elif terminal == "ghostty":
        _wizard_modern_terminal("Ghostty")
    elif terminal == "vscode":
        _wizard_vscode()
    else:
        _wizard_unknown()

    _p("  Done.", _BOLD)
    print()
