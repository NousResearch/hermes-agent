"""Guard: every DETACHED Chrome/Chromium launcher in the harness must suppress the
macOS "Keychain Not Found" Safe-Storage modal.

A Chromium-family browser launched detached on macOS (no Aqua login session — i.e.
from the gateway, a kanban/subagent worker, or a cron) tries to read its "Safe
Storage" key from the macOS Keychain and, when it can't, throws a BLOCKING on-screen
"A keychain cannot be found to store \"Chrome\"" system modal that leaks onto the
user's display and stalls automation. Launching with
``--password-store=basic --use-mock-keychain`` routes password storage off the OS
Keychain and suppresses the modal entirely (harmless no-ops on Linux/Windows).

This recurred on the fleet because the original guard (the tabs-outliner e2e
``lint-browser-launchers.sh``) only scanned ``extension-build/e2e`` and never saw
``hermes_cli/browser_connect.py`` — the harness's own debug-Chrome auto-launch path.
This test closes that coverage gap inside the harness repo so a future launcher that
omits the flags fails CI.

Scope (DETACHED automation launches only):
  - INCLUDED: code that builds a Chrome argv / ``open -a "Google Chrome" --args ...``
    command for a detached debug/automation browser (the modal-throwing path).
  - EXCLUDED: ``webbrowser.open(url)`` interactive OAuth/portal flows — those open the
    user's EXISTING default browser in their real login session, never spawn a
    detached profile, and so never throw the modal (and must NOT carry the flags).
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# A "detached Chrome launcher" line builds an automation Chrome invocation. We key on
# the two unambiguous tells: a --remote-debugging-port arg (CDP automation) or an
# `open -a "Google Chrome" --args` shell command. Both are detached-spawn paths.
_LAUNCHER_SIGNATURES = (
    re.compile(r"--remote-debugging-port"),
    re.compile(r'open -a "Google Chrome" --args'),
)
_REQUIRED_FLAGS = ("--password-store=basic", "--use-mock-keychain")

# Directories that contain harness launch code worth scanning.
_SCAN_DIRS = ("hermes_cli", "tools")


def _python_sources() -> list[Path]:
    files: list[Path] = []
    for d in _SCAN_DIRS:
        root = REPO_ROOT / d
        if root.is_dir():
            files.extend(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)
    return files


def _detached_chrome_launchers() -> list[Path]:
    """Files that contain at least one detached-Chrome launch signature."""
    hits: list[Path] = []
    for path in _python_sources():
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(sig.search(text) for sig in _LAUNCHER_SIGNATURES):
            hits.append(path)
    return hits


def _launch_constructs(text: str) -> list[tuple[int, str]]:
    """Split a source file into per-launcher windows so a SECOND launch path that
    omits a flag can't be masked by a FIRST path that carries it (a whole-file
    substring check has exactly that blind spot). We anchor each construct on a
    launch signature line and take a window of surrounding lines large enough to
    span an argv list / multi-line command string but small enough not to bleed
    into the next construct.

    Returns (line_number, window_text) for each distinct launch signature hit.
    """
    lines = text.splitlines()
    anchors = [
        i for i, line in enumerate(lines)
        if any(sig.search(line) for sig in _LAUNCHER_SIGNATURES)
    ]
    windows: list[tuple[int, str]] = []
    for idx, i in enumerate(anchors):
        # Forward-only window from this anchor to just before the NEXT anchor (capped),
        # so an argv list whose flags sit 15-20 lines below a comment-padded
        # --remote-debugging-port line are still captured, without bleeding into the
        # next distinct launch construct.
        next_anchor = anchors[idx + 1] if idx + 1 < len(anchors) else len(lines)
        hi = min(next_anchor, i + 30)
        windows.append((i + 1, "\n".join(lines[i:hi])))
    return windows


def test_browser_connect_is_discovered_as_a_launcher():
    """Sanity: the scanner actually finds the known launcher. If this fails the
    signature regexes drifted and the guard below is vacuously green."""
    launchers = {p.name for p in _detached_chrome_launchers()}
    assert "browser_connect.py" in launchers, (
        "browser_connect.py should be detected as a detached-Chrome launcher; "
        f"found: {sorted(launchers)}"
    )


def test_browser_connect_has_both_launch_constructs():
    """Sanity: browser_connect.py has BOTH detached launch paths (the
    _chrome_debug_args argv list AND the macOS `open -a` fallback string). If this
    drops below 2 the per-construct guard below may be silently checking fewer
    paths than exist."""
    text = (REPO_ROOT / "hermes_cli" / "browser_connect.py").read_text(encoding="utf-8")
    assert len(_launch_constructs(text)) >= 2, (
        "expected >=2 launch constructs in browser_connect.py "
        f"(argv list + open -a fallback); found {len(_launch_constructs(text))}"
    )


def test_every_detached_chrome_launch_construct_suppresses_macos_keychain_modal():
    """Every INDIVIDUAL detached-Chrome launch construct MUST carry both GUI-silence
    flags — checked per-construct, not per-file, so a second launch path that omits a
    flag is caught even when a sibling path carries it. RED-prove by deleting either
    flag from EITHER the _chrome_debug_args list OR the macOS `open -a` string."""
    offenders: list[str] = []
    for path in _detached_chrome_launchers():
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, window in _launch_constructs(text):
            missing = [f for f in _REQUIRED_FLAGS if f not in window]
            if missing:
                rel = path.relative_to(REPO_ROOT)
                offenders.append(f"{rel}:{lineno} missing {missing}")
    assert not offenders, (
        "Detached-Chrome launch construct(s) omit a macOS Keychain-modal suppression "
        "flag (--password-store=basic / --use-mock-keychain):\n  " + "\n  ".join(offenders)
    )
