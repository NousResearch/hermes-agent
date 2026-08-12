"""Tests for the light-mode terminal detection + color remap in cli.py.

Covers the env-override path and the SkinConfig.get_color() wrapper that
the resize / light-mode salvage installs at module import time.  We don't
try to fake an OSC 11 reply — the env-override branch short-circuits
before the terminal query, which is the path most users hit.
"""

from __future__ import annotations


import pytest


@pytest.fixture
def cli_mod(monkeypatch):
    """Import cli with the light-mode cache cleared each test."""
    import cli as _cli

    # The module-level _install_skin_light_mode_hook() and import-time
    # _detect_light_mode() prime ran once at first import.  We just reset
    # the detection cache so the per-test env override takes effect.
    monkeypatch.setattr(_cli, "_LIGHT_MODE_CACHE", None)
    return _cli


class TestLightModeDetection:
    def test_hermes_light_env_true_forces_light(self, cli_mod, monkeypatch):
        monkeypatch.setenv("HERMES_LIGHT", "1")
        assert cli_mod._detect_light_mode() is True

    def test_hermes_light_env_false_forces_dark(self, cli_mod, monkeypatch):
        monkeypatch.setenv("HERMES_LIGHT", "0")
        # Also blank out other signals so nothing else flips it light.
        monkeypatch.delenv("HERMES_TUI_LIGHT", raising=False)
        monkeypatch.delenv("HERMES_TUI_THEME", raising=False)
        monkeypatch.delenv("HERMES_TUI_BACKGROUND", raising=False)
        monkeypatch.delenv("COLORFGBG", raising=False)
        assert cli_mod._detect_light_mode() is False

    def test_theme_hint_light(self, cli_mod, monkeypatch):
        monkeypatch.delenv("HERMES_LIGHT", raising=False)
        monkeypatch.delenv("HERMES_TUI_LIGHT", raising=False)
        monkeypatch.setenv("HERMES_TUI_THEME", "light")
        assert cli_mod._detect_light_mode() is True

    def test_background_hex_hint_light(self, cli_mod, monkeypatch):
        monkeypatch.delenv("HERMES_LIGHT", raising=False)
        monkeypatch.delenv("HERMES_TUI_LIGHT", raising=False)
        monkeypatch.delenv("HERMES_TUI_THEME", raising=False)
        monkeypatch.setenv("HERMES_TUI_BACKGROUND", "#FFFFFF")
        assert cli_mod._detect_light_mode() is True

    def test_background_hex_hint_dark(self, cli_mod, monkeypatch):
        monkeypatch.delenv("HERMES_LIGHT", raising=False)
        monkeypatch.delenv("HERMES_TUI_LIGHT", raising=False)
        monkeypatch.delenv("HERMES_TUI_THEME", raising=False)
        monkeypatch.setenv("HERMES_TUI_BACKGROUND", "#1a1a2e")
        monkeypatch.delenv("COLORFGBG", raising=False)
        assert cli_mod._detect_light_mode() is False

    def test_colorfgbg_light_bg_slot(self, cli_mod, monkeypatch):
        monkeypatch.delenv("HERMES_LIGHT", raising=False)
        monkeypatch.delenv("HERMES_TUI_LIGHT", raising=False)
        monkeypatch.delenv("HERMES_TUI_THEME", raising=False)
        monkeypatch.delenv("HERMES_TUI_BACKGROUND", raising=False)
        monkeypatch.setenv("COLORFGBG", "0;15")  # bg slot 15 = light
        assert cli_mod._detect_light_mode() is True

    def test_cache_is_sticky(self, cli_mod, monkeypatch):
        monkeypatch.setenv("HERMES_LIGHT", "1")
        assert cli_mod._detect_light_mode() is True
        # Even if the env flips, the cached result wins until reset.
        monkeypatch.setenv("HERMES_LIGHT", "0")
        assert cli_mod._detect_light_mode() is True


class TestOsc11Probe:
    """The OSC 11 background probe must never run where its reply can leak
    into prompt_toolkit's input (a late BEL-terminated reply reads as Ctrl+G
    = open-editor, trapping the user in a stray editor). Guard the cases we
    refuse to probe in.
    """

    @pytest.mark.parametrize("var", ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY"))
    def test_skips_over_ssh(self, cli_mod, monkeypatch, var):
        monkeypatch.setattr(cli_mod.sys.stdin, "isatty", lambda: True, raising=False)
        monkeypatch.setattr(cli_mod.sys.stdout, "isatty", lambda: True, raising=False)
        for v in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY"):
            monkeypatch.delenv(v, raising=False)
        monkeypatch.setenv(var, "1.2.3.4 5555 22")
        assert cli_mod._query_osc11_background() is None

    def test_skips_when_not_a_tty(self, cli_mod, monkeypatch):
        monkeypatch.setattr(cli_mod.sys.stdin, "isatty", lambda: False, raising=False)
        assert cli_mod._query_osc11_background() is None


class TestLightModeRemap:
    def test_remap_no_op_in_dark_mode(self, cli_mod, monkeypatch):
        monkeypatch.setenv("HERMES_LIGHT", "0")
        # Cache is None from the fixture; first call sticks at False.
        assert cli_mod._maybe_remap_for_light_mode("#FFF8DC") == "#FFF8DC"

    def test_remap_known_dark_color(self, cli_mod, monkeypatch):
        monkeypatch.setenv("HERMES_LIGHT", "1")
        # Force the detect cache to True for this test.
        cli_mod._LIGHT_MODE_CACHE = True
        assert cli_mod._maybe_remap_for_light_mode("#FFF8DC") == "#1A1A1A"
        assert cli_mod._maybe_remap_for_light_mode("#FFD700") == "#9A6B00"

    def test_remap_case_insensitive(self, cli_mod, monkeypatch):
        cli_mod._LIGHT_MODE_CACHE = True
        # Lowercase input should still remap.
        assert cli_mod._maybe_remap_for_light_mode("#fff8dc") == "#1A1A1A"

    def test_remap_unknown_color_passthrough(self, cli_mod, monkeypatch):
        cli_mod._LIGHT_MODE_CACHE = True
        # A color not in the remap table is returned unchanged.
        assert cli_mod._maybe_remap_for_light_mode("#ABCDEF") == "#ABCDEF"

    def test_remap_skips_statusbar_paired_colors(self, cli_mod, monkeypatch):
        """Colors that live on a dark bg (status bar fg) MUST NOT be
        remapped — otherwise they go dark-on-dark and disappear.

        Regression guard for the patch-11 fix (intentional table omission).
        """
        cli_mod._LIGHT_MODE_CACHE = True
        for fg in ("#C0C0C0", "#888888", "#555555", "#8B8682"):
            assert cli_mod._maybe_remap_for_light_mode(fg) == fg, (
                f"{fg} is a status-bar fg paired with dark bg; remapping it "
                "would produce dark-on-dark"
            )


class TestSkinConfigHook:
    """The salvage wraps SkinConfig.get_color at module import time so
    every skin color read goes through the light-mode remap.  Verify
    the hook installed and functions correctly.
    """

    def test_hook_installed(self, cli_mod):
        from hermes_cli.skin_engine import SkinConfig

        assert getattr(SkinConfig, "_hermes_light_mode_hook_installed", False) is True

    def test_hook_is_idempotent(self, cli_mod):
        # Calling the installer twice must not double-wrap (the marker
        # attribute is the guard).
        from hermes_cli.skin_engine import SkinConfig

        before = SkinConfig.get_color
        cli_mod._install_skin_light_mode_hook()
        after = SkinConfig.get_color
        assert before is after

    def test_skin_color_remaps_through_wrapper_in_light_mode(
        self, cli_mod, monkeypatch
    ):
        from hermes_cli.skin_engine import SkinConfig

        cli_mod._LIGHT_MODE_CACHE = True
        skin = SkinConfig(
            name="test",
            colors={"banner_text": "#FFF8DC", "response_border": "#FFD700"},
        )
        # The wrapper kicks in at get_color, not at construction time.
        assert skin.get_color("banner_text") == "#1A1A1A"
        assert skin.get_color("response_border") == "#9A6B00"

    def test_skin_color_passthrough_in_dark_mode(self, cli_mod, monkeypatch):
        from hermes_cli.skin_engine import SkinConfig

        cli_mod._LIGHT_MODE_CACHE = False
        skin = SkinConfig(name="test", colors={"banner_text": "#FFF8DC"})
        assert skin.get_color("banner_text") == "#FFF8DC"


class TestOsc11DrainGuard:
    """Regression: a late-arriving OSC 11 reply must not leak into
    prompt_toolkit's input buffer (#40250).

    Two layers guard against this: the DA1 fence keeps the main read loop
    listening until the terminal proves it has processed our query, and
    the drain loop in the ``finally`` block reads (and discards) any
    stragglers that slip past TCSAFLUSH.
    """

    def test_late_reply_is_consumed_not_leaked(self, cli_mod, monkeypatch):
        """Simulate a terminal that sends the OSC 11 reply 150ms after the
        query.  With the DA1 fence the main loop is still listening, so the
        reply is consumed AND used; nothing remains for prompt_toolkit."""
        import os, termios, tty as _tty

        # Create a pipe pair to fake stdin
        read_fd, write_fd = os.pipe()

        # Set up fake termios on the read end
        # We'll monkeypatch tcgetattr/tcsetattr to no-op
        fake_attrs = [0, 0, 0, 0, 0, 0, [b'\x00'] * 32]
        monkeypatch.setattr(termios, "tcgetattr", lambda fd: fake_attrs)
        monkeypatch.setattr(termios, "tcsetattr", lambda fd, when, attrs: None)
        monkeypatch.setattr(_tty, "setcbreak", lambda fd: None)

        # Make stdin.isatty / stdout.isatty return True
        monkeypatch.setattr(cli_mod.sys.stdin, "isatty", lambda: True, raising=False)
        monkeypatch.setattr(cli_mod.sys.stdout, "isatty", lambda: True, raising=False)
        monkeypatch.setattr(cli_mod.sys.stdin, "fileno", lambda: read_fd, raising=False)

        # Clear SSH env vars
        for v in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY"):
            monkeypatch.delenv(v, raising=False)

        # Write a delayed OSC 11 reply (then the DA1 fence reply) — the
        # fenced main loop must still be listening and consume both.
        import threading

        def delayed_write():
            import time
            time.sleep(0.15)
            os.write(write_fd, b"\x1b]11;rgb:0c0c/0c0c/0c0c\x1b\\\x1b[?62;22c")

        t = threading.Thread(target=delayed_write, daemon=True)
        t.start()

        # The late reply is consumed by the fenced read loop and used.
        result = cli_mod._query_osc11_background()
        assert result == "#0C0C0C"

        # Verify the pipe is drained — a non-blocking read should return empty
        import select
        r, _, _ = select.select([read_fd], [], [], 0)
        assert not r, "late OSC 11 bytes must be consumed, not left to leak"

        os.close(read_fd)
        os.close(write_fd)

    def test_post_deadline_straggler_is_drained(self, cli_mod, monkeypatch):
        """Bytes that arrive after the main loop has already finished (DA1
        answered instantly, reply straggles in during teardown) are eaten
        by the post-flush drain window instead of leaking (#40250)."""
        import os, termios, tty as _tty

        read_fd, write_fd = os.pipe()
        fake_attrs = [0, 0, 0, 0, 0, 0, [b'\x00'] * 32]
        monkeypatch.setattr(termios, "tcgetattr", lambda fd: fake_attrs)
        monkeypatch.setattr(termios, "tcsetattr", lambda fd, when, attrs: None)
        monkeypatch.setattr(_tty, "setcbreak", lambda fd: None)
        monkeypatch.setattr(cli_mod.sys.stdin, "isatty", lambda: True, raising=False)
        monkeypatch.setattr(cli_mod.sys.stdout, "isatty", lambda: True, raising=False)
        monkeypatch.setattr(cli_mod.sys.stdin, "fileno", lambda: read_fd, raising=False)
        for v in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY"):
            monkeypatch.delenv(v, raising=False)

        # DA1 answered immediately (herdr-style: OSC 11 swallowed) — main
        # loop exits fast — then a straggler payload lands during teardown.
        os.write(write_fd, b"\x1b[?62;22c")

        import threading

        def straggler():
            import time
            time.sleep(0.02)  # inside the 50ms drain window
            os.write(write_fd, b"\x1b]11;rgb:0c0c/0c0c/0c0c\x1b\\")

        t = threading.Thread(target=straggler, daemon=True)
        t.start()

        result = cli_mod._query_osc11_background()
        assert result is None  # OSC 11 was swallowed; only DA1 answered

        import select
        r, _, _ = select.select([read_fd], [], [], 0)
        assert not r, "drain loop should have consumed the straggler bytes"

        os.close(read_fd)
        os.close(write_fd)


# ────────────────────────────────────────────────────────────────────────
# OSC 11 query — DA1 fence behavior.
#
# The query is fenced with a DA1 sentinel so a terminal manager that
# swallows OSC 11 (herdr) or relays it slowly (SSH bridges, some tmux
# setups) can never leave reply bytes in the tty buffer for
# prompt_toolkit to read as typed input.  These tests run the real
# function in a child on a real PTY and play the terminal's role from
# the parent side.

import os as _os
import sys as _sys


_CHILD_SRC = r"""
import sys, os
sys.path.insert(0, os.environ["HERMES_REPO"])
import cli
bg = cli._query_osc11_background()
print("RESULT:" + repr(bg), flush=True)
# Drain anything left in the tty buffer — must be empty (no leak).
import termios, tty, select, time
fd = sys.stdin.fileno()
old = termios.tcgetattr(fd)
tty.setcbreak(fd)
buf = b""
deadline = time.monotonic() + 0.6
while time.monotonic() < deadline:
    r, _, _ = select.select([fd], [], [], 0.1)
    if r:
        buf += os.read(fd, 256)
termios.tcsetattr(fd, termios.TCSADRAIN, old)
print("LEFTOVER:" + repr(buf), flush=True)
"""


def _run_osc11_child(reply_fn, repo_root, timeout=8.0):
    """Fork a PTY child running _query_osc11_background().

    reply_fn(query_age_seconds) -> bytes to write once, or None to wait.
    Returns (result_line, leftover_line).
    """
    import pty
    import time as _time

    env = dict(_os.environ, HERMES_REPO=str(repo_root))
    for var in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY"):
        env.pop(var, None)
    pid, master = pty.fork()
    if pid == 0:  # child
        _os.execvpe(_sys.executable, [_sys.executable, "-c", _CHILD_SRC], env)
    _os.set_blocking(master, False)
    out = b""
    answered = 0
    query_at = None
    t0 = _time.monotonic()
    try:
        while _time.monotonic() - t0 < timeout:
            try:
                chunk = _os.read(master, 1024)
                if chunk:
                    out += chunk
            except (BlockingIOError, OSError):
                pass
            # importing cli primes _detect_light_mode() which issues its own
            # OSC 11 query before the explicit call — answer each query.
            if query_at is None and out.count(b"\x1b]11;?") > answered:
                query_at = _time.monotonic()
            if query_at is not None:
                payload = reply_fn(_time.monotonic() - query_at)
                if payload is not None:
                    _os.write(master, payload)
                    answered += 1
                    query_at = None
            if b"LEFTOVER:" in out and out.rstrip().endswith(b"'"):
                break
            _time.sleep(0.005)
    finally:
        try:
            _os.close(master)
        except OSError:
            pass
        try:
            _os.waitpid(pid, 0)
        except ChildProcessError:
            pass
    text = out.decode("utf-8", "replace")
    print("child raw output:", repr(text))  # aids debugging on failure
    result = leftover = None
    for line in text.splitlines():
        # The PTY echoes the query bytes onto the same line as the first
        # print, so match anywhere in the line rather than at the start.
        if "RESULT:" in line and result is None:
            result = line.split("RESULT:", 1)[1]
        elif "LEFTOVER:" in line and leftover is None:
            leftover = line.split("LEFTOVER:", 1)[1]
    return result, leftover


@pytest.fixture
def repo_root():
    import pathlib
    return pathlib.Path(__file__).resolve().parents[2]


@pytest.mark.skipif(_sys.platform == "win32", reason="POSIX PTY test")
class TestOsc11Da1Fence:
    def test_herdr_style_da1_only_returns_none_without_leak(self, repo_root):
        """Terminal answers DA1 instantly but swallows OSC 11 (herdr)."""
        result, leftover = _run_osc11_child(
            lambda age: b"\x1b[?62;22c", repo_root
        )
        assert result == "None"
        assert leftover == "b''"

    def test_slow_inorder_reply_is_consumed_not_leaked(self, repo_root):
        """OSC 11 reply arrives at +300ms (past the old 100ms budget),
        DA1 right behind it.  The fence keeps us listening, so the color
        is detected and nothing leaks into the tty buffer."""
        result, leftover = _run_osc11_child(
            lambda age: (
                b"\x1b]11;rgb:1e1e/1e1e/2e2e\x1b\\\x1b[?62;22c"
                if age > 0.3 else None
            ),
            repo_root,
        )
        assert result == "'#1E1E2E'"
        assert leftover == "b''"

    def test_mute_terminal_times_out_clean(self, repo_root):
        """Terminal that answers nothing: give up at the safety-net
        deadline with no leftovers."""
        result, leftover = _run_osc11_child(lambda age: None, repo_root)
        assert result == "None"
        assert leftover == "b''"
