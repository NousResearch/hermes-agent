"""Tests for the subprocess console-window rule in check-windows-footguns.py.

These assert behavior contracts of the AST rule — which call shapes get
flagged and which are correctly exempt — NOT a snapshot of how many sites
the repo currently has. The rule's job: flag subprocess calls that can spawn
a NEW Windows console window, ignore the ones that physically cannot.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# The checker lives at scripts/check-windows-footguns.py (hyphenated, not a
# normal importable module name) — load it by path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHECKER_PATH = _REPO_ROOT / "scripts" / "check-windows-footguns.py"


@pytest.fixture(scope="module")
def checker():
    spec = importlib.util.spec_from_file_location("_wf_checker", _CHECKER_PATH)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so the module's dataclasses can resolve their
    # __module__ via sys.modules (dataclasses._is_type looks it up there).
    sys.modules["_wf_checker"] = mod
    spec.loader.exec_module(mod)
    return mod


def _flag(checker, src: str) -> list[int]:
    """Return the line numbers the subprocess rule flags for a source string."""
    hits = checker.scan_subprocess_window_footguns(Path("x.py"), src)
    return [lineno for (lineno, _line, _fg) in hits]


# --- Calls that SHOULD be flagged (can pop a Windows console) --------------


@pytest.mark.parametrize(
    "src",
    [
        'subprocess.run(["git", "status"])',
        'subprocess.Popen(["node", "x.js"])',
        'subprocess.call(["npm", "run", "build"])',
        'subprocess.check_call(["python", "setup.py"])',
        "subprocess.run(cmd)",  # dynamic argv, no redirection
        'sp.run(["foo"])',  # `sp` alias
    ],
)
def test_flags_bare_window_spawning_calls(checker, src):
    assert _flag(checker, src) == [1], src


def test_flags_multiline_call_without_redirection(checker):
    src = (
        "subprocess.run(\n"
        "    [npm, 'run', 'build'],\n"
        "    cwd=desktop_dir,\n"
        "    check=False,\n"
        ")\n"
    )
    assert _flag(checker, src) == [1]


# --- Calls that should NOT be flagged (no new console possible) ------------


@pytest.mark.parametrize(
    "src",
    [
        # captured/redirected AND not a known Windows-flashing program -> safe.
        # (espeak-ng / a non-console-exe; capture inherits the parent console.)
        'subprocess.run(["espeak-ng", "hi"], capture_output=True)',
        'subprocess.run(["mytool", "x"], stdout=subprocess.PIPE)',
        'subprocess.check_output(["mytool", "rev-parse"])',
        # already managing the console
        'subprocess.run(["git", "x"], creationflags=windows_hide_flags())',
        # ** spread may carry a helper -> not penalised
        "subprocess.Popen(argv, **windows_detach_popen_kwargs())",
        "subprocess.run(cmd, **run_kwargs)",
        # routed through the chokepoint wrapper -> different prefix, never flagged
        "_subprocess_compat.run(['git', 'status'])",
    ],
)
def test_exempts_window_safe_calls(checker, src):
    assert _flag(checker, src) == [], src


@pytest.mark.parametrize(
    "src",
    [
        # Cross-platform console exes that allocate a Windows console even when
        # captured — capture is NOT a safety boundary for these (Gille review,
        # PR #53791 follow-up). They must be flagged despite capture/redirect.
        'subprocess.run(["git", "status"], capture_output=True)',
        'subprocess.run(["git", "x"], stdout=subprocess.PIPE)',
        'subprocess.run(["gh", "pr", "list"], stderr=subprocess.DEVNULL)',
        'subprocess.check_output(["git", "rev-parse", "HEAD"])',
        'subprocess.run(["npm", "ci"], capture_output=True)',
        'subprocess.run(["ffmpeg", "-i", "x"], capture_output=True)',
        'subprocess.run(["docker", "info"], capture_output=True, timeout=10)',
        'subprocess.run(["uv", "pip", "install"], capture_output=True)',
    ],
)
def test_flags_flashing_programs_even_when_captured(checker, src):
    assert _flag(checker, src) == [1], src


@pytest.mark.parametrize(
    "src",
    [
        'subprocess.run(["launchctl", "bootout", target])',
        'subprocess.run(["systemctl", "status", svc])',
        'subprocess.run(["brew", "install", "espeak-ng"])',
        'subprocess.run(["codesign", "--sign", "-", app])',
        'subprocess.run(["/usr/bin/sudo", "chmod", "4755", p])',  # path-qualified
    ],
)
def test_exempts_posix_only_programs(checker, src):
    """launchctl/systemctl/brew/etc. don't exist on Windows -> can't pop a
    Windows console, so they must not require a creationflag or suppression."""
    assert _flag(checker, src) == [], src


def test_inline_suppression_marker(checker):
    src = 'subprocess.run(["git", "x"])  # windows-footgun: ok\n'
    assert _flag(checker, src) == []


def test_inline_suppression_on_multiline_closing_paren(checker):
    src = (
        "subprocess.run(\n"
        "    [npm, 'run', 'build'],\n"
        "    cwd=d,\n"
        ")  # windows-footgun: ok\n"
    )
    assert _flag(checker, src) == []


def test_non_subprocess_calls_ignored(checker):
    # A .run() on something that isn't the subprocess module is not our concern.
    src = "loop.run(coro)\nclient.run()\n"
    assert _flag(checker, src) == []


def test_syntax_error_returns_empty(checker):
    assert _flag(checker, "def (:\n") == []


def test_repo_is_clean_of_window_footguns(checker):
    """Full-repo invariant: no unsuppressed window-spawning subprocess calls
    remain in shippable Python packages. This is the chokepoint the rule
    exists to hold."""
    roots = [
        _REPO_ROOT / d
        for d in (
            "hermes_cli",
            "gateway",
            "tools",
            "cron",
            "agent",
            "plugins",
            "scripts",
            "acp_adapter",
            "acp_registry",
        )
    ]
    roots = [r for r in roots if r.exists()]
    offenders: list[str] = []
    for path in checker.iter_files(roots):
        if path.suffix not in {".py", ".pyw", ".pyi"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, _line, _fg in checker.scan_subprocess_window_footguns(path, text):
            offenders.append(f"{path.relative_to(_REPO_ROOT)}:{lineno}")
    assert not offenders, "Unsuppressed Windows console footguns:\n" + "\n".join(
        offenders
    )
