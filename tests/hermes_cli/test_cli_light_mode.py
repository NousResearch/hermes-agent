"""Tests for the light-mode terminal detection + color remap in cli.py.

Covers the env-override path and the SkinConfig.get_color() wrapper that
the resize / light-mode salvage installs at module import time.  We don't
try to fake an OSC 11 reply — the env-override branch short-circuits
before the terminal query, which is the path most users hit.
"""

from __future__ import annotations

import time

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
    """Exercise the installed color hook, including self-painted badge colors."""

    @pytest.mark.parametrize("skin_name", ["default", "sisyphus"])
    def test_badge_preserves_its_paired_colors_in_light_mode(
        self, cli_mod, monkeypatch, skin_name
    ):
        from hermes_cli.skin_engine import (
            get_active_skin, get_prompt_toolkit_style_overrides, set_active_skin,
        )

        monkeypatch.setenv("HERMES_LIGHT", "1")
        previous = get_active_skin().name
        try:
            set_active_skin(skin_name)
            skin = get_active_skin()
            background = skin.colors.get("status_bar_strong", skin.colors.get("banner_title", "#FFD700"))
            foreground = skin.colors.get("status_bar_bg", "#1a1a2e")
            assert get_prompt_toolkit_style_overrides()["status-bar-session-title"] == (
                f"bg:{background} {foreground} bold"
            )
        finally:
            set_active_skin(previous)



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


@pytest.mark.platforms("linux")
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


@pytest.mark.platforms("posix")  # POSIX PTY test
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


# ────────────────────────────────────────────────────────────────────────
# Silent-terminal verdict cache (#144550).
#
# A terminal that answers neither the OSC 11 query nor the DA1 fence (Apple
# Terminal replies with zero bytes) cannot convert the 1 s budget into a
# colour, yet it paid that budget on every launch. Two consecutive misses are
# remembered per terminal, TTL-bounded, and ONLY for the negative case: a terminal that
# answers must keep being probed so a theme switch lands on the next start.


class _FakeClock:
    """Deterministic stand-in for the ``time`` module: advances on every read."""

    def __init__(self, step: float = 0.25, wall: float | None = None):
        self.step = step
        self.t = 0.0
        self.wall = time.time() if wall is None else wall

    def monotonic(self) -> float:
        self.t += self.step
        return self.t

    def time(self) -> float:
        return self.wall


@pytest.fixture
def probe_env(monkeypatch, cli_mod, tmp_path):
    """Drive the real probe against a fake tty whose terminal never answers.

    Returns a callback ``arm(reply=b"", late_reply=b"")`` that reinstalls the tty fakes and clock, so each
    call to ``_query_osc11_background`` is measured on simulated time instead of wall-clock seconds.
    """
    import os
    import select
    import termios
    import tty as _tty
    from types import SimpleNamespace

    import hermes_cli.cli_render as render

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for var in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")
    monkeypatch.delenv("COLORTERM", raising=False)

    read_fd, write_fd = os.pipe()
    real_select = select.select
    fake_attrs = [0, 0, 0, 0, 0, 0, [b"\x00"] * 32]
    monkeypatch.setattr(termios, "tcgetattr", lambda fd: fake_attrs)
    monkeypatch.setattr(termios, "tcsetattr", lambda fd, when, attrs: None)
    monkeypatch.setattr(_tty, "setcbreak", lambda fd: None)
    monkeypatch.setattr(cli_mod.sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(cli_mod.sys.stdout, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(cli_mod.sys.stdin, "fileno", lambda: read_fd, raising=False)

    def arm(*, reply: bytes = b"", late_reply: bytes = b"") -> _FakeClock:
        def restore(fd, when, attrs):
            if late_reply:
                os.write(write_fd, late_reply)

        monkeypatch.setattr(termios, "tcsetattr", restore)
        if reply:
            os.write(write_fd, reply)
        if reply or late_reply:
            monkeypatch.setattr(select, "select", lambda r, w, x, timeout: real_select(r, w, x, 0))
        else:
            # Nothing ever arrives: the read loop must fall through to its deadline.
            monkeypatch.setattr(select, "select", lambda r, w, x, timeout: ([], [], []))
        clock = _FakeClock(step=0.005 if late_reply else 0.25)
        monkeypatch.setattr(render, "time", SimpleNamespace(monotonic=clock.monotonic, time=clock.time))
        return clock

    yield arm
    os.close(read_fd)
    os.close(write_fd)


def _verdict_path():
    from hermes_constants import get_hermes_dir

    return get_hermes_dir("cache/terminal", "terminal_cache") / "osc11_silent.json"


@pytest.mark.platforms("posix")
class TestSilentTerminalProbeCache:
    def test_interrupted_probe_preserves_one_miss(self, cli_mod, probe_env):
        import json
        from unittest.mock import patch

        probe_env()
        assert cli_mod._query_osc11_background() is None
        path = _verdict_path()
        record = path.read_bytes()
        assert json.loads(record)["misses"] == 1

        probe_env()
        with patch("select.select", side_effect=[KeyboardInterrupt("probe interrupted"), ([], [], [])]):
            with pytest.raises(KeyboardInterrupt, match="probe interrupted"):
                cli_mod._query_osc11_background()

        assert json.loads(path.read_bytes())["misses"] == 1
        assert path.read_bytes() == record

    def test_uninterrupted_silent_probe_increments_one_miss(self, cli_mod, probe_env):
        import json

        probe_env()
        assert cli_mod._query_osc11_background() is None
        assert json.loads(_verdict_path().read_text())["misses"] == 1

        probe_env()
        assert cli_mod._query_osc11_background() is None
        assert json.loads(_verdict_path().read_text())["misses"] == 2

    def test_drain_reply_resets_count(self, cli_mod, probe_env):
        """Bytes arriving only after TCSAFLUSH still break a silent streak."""
        import json

        probe_env()
        assert cli_mod._query_osc11_background() is None
        assert json.loads(_verdict_path().read_text())["misses"] == 1

        probe_env(late_reply=b"\x1b[?62;22c")
        assert cli_mod._query_osc11_background() is None
        assert json.loads(_verdict_path().read_text())["misses"] == 0

        clock = probe_env()
        assert cli_mod._query_osc11_background() is None
        assert clock.t >= 1.0, "a late answer must not suppress the next probe"

    @pytest.mark.parametrize("misses", ['"2"', '2.9', 'true', '1e309', 'NaN', '-1', '[]', '{}', 'null'])
    def test_invalid_miss_count_still_probes(self, cli_mod, probe_env, misses):
        import hermes_cli.cli_render as render

        clock = probe_env()
        path = _verdict_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            '{"terminal":"xterm-256color|apple_terminal|","at":'
            + str(clock.wall) + ',"misses":' + misses + '}', encoding="utf-8",
        )
        assert render._osc11_terminal_silent_misses() == 0
        assert cli_mod._query_osc11_background() is None
        assert clock.t >= 1.0, "only a whole-number miss count may suppress probing"

    @pytest.mark.parametrize("recorded", ['1e309', '-1e309', 'NaN', '1' + '0' * 400, '"bad"', '[]', '{}', 'null'])
    def test_invalid_timestamp_still_probes(self, cli_mod, probe_env, recorded):
        clock = probe_env()
        path = _verdict_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            '{"terminal":"xterm-256color|apple_terminal|","at":'
            + recorded + ',"misses":2}', encoding="utf-8",
        )
        assert cli_mod._query_osc11_background() is None
        assert clock.t >= 1.0, "an invalid timestamp must not suppress probing"

    def test_fresh_whole_number_count_still_skips(self, cli_mod, probe_env):
        import json

        clock = probe_env()
        path = _verdict_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"terminal": "xterm-256color|apple_terminal|", "at": clock.wall, "misses": 2}),
            encoding="utf-8",
        )
        assert cli_mod._query_osc11_background() is None
        assert clock.t < 0.1, "two fresh integer misses must still suppress the probe"

    def test_silent_terminal_does_not_pay_the_budget_again(self, cli_mod, probe_env):
        """Two silent launches spend the budget; the third skips it."""
        clock = probe_env()
        assert cli_mod._query_osc11_background() is None
        assert clock.t >= 1.0, "the first probe must still use its full budget"

        clock2 = probe_env()
        assert cli_mod._query_osc11_background() is None
        assert clock2.t >= 1.0, "one missed window must not suppress the next probe"

        clock3 = probe_env()
        assert cli_mod._query_osc11_background() is None
        assert clock3.t < 0.1, (
            "a terminal already known to answer nothing must be skipped, not probed for another ~1 s"
        )

    @pytest.mark.parametrize("reply", [
        b"\x1b]11;rgb:ffff/ffff/ffff\x1b\\\x1b[?62;22c",
        b"\x1b[?62;22c",
        b"x",
    ])
    def test_answer_between_silent_launches_resets_count(self, cli_mod, probe_env, reply):
        """Any received byte breaks the sequence, even without an OSC 11 reply."""
        probe_env()
        assert cli_mod._query_osc11_background() is None

        probe_env(reply=reply)
        result = cli_mod._query_osc11_background()
        assert result == ("#FFFFFF" if b"rgb:" in reply else None)

        probe_env()
        assert cli_mod._query_osc11_background() is None
        clock = probe_env()
        assert cli_mod._query_osc11_background() is None
        assert clock.t >= 1.0, "silence separated by an answer must not count as consecutive"

    def test_answering_terminal_is_not_cached_as_silent(self, cli_mod, probe_env):
        """A terminal that answers keeps being probed, so a theme switch still lands."""
        probe_env(reply=b"\x1b]11;rgb:0c0c/0c0c/0c0c\x1b\\\x1b[?62;22c")
        assert cli_mod._query_osc11_background() == "#0C0C0C"
        assert not _verdict_path().exists(), "an answering terminal must never be recorded as silent"

        probe_env(reply=b"\x1b]11;rgb:ffff/ffff/ffff\x1b\\\x1b[?62;22c")
        assert cli_mod._query_osc11_background() == "#FFFFFF", "detection must re-run on the next launch"

    def test_verdict_expires_after_the_ttl(self, cli_mod, probe_env):
        """A stale verdict (terminal upgraded, emulator now answers) is re-probed."""
        import json

        clock = probe_env()
        path = _verdict_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"terminal": "xterm-256color|apple_terminal|", "at": time.time() - 2 * 24 * 3600, "misses": 2}),
            encoding="utf-8",
        )
        assert cli_mod._query_osc11_background() is None
        assert clock.t >= 1.0, "an expired verdict must not suppress the probe"
        clock2 = probe_env()
        assert cli_mod._query_osc11_background() is None
        assert clock2.t >= 1.0, "an expired count must restart at one miss"

    def test_verdict_is_per_terminal(self, cli_mod, probe_env):
        """Another terminal (TERM/TERM_PROGRAM) gets its own probe."""
        import json

        clock = probe_env()
        path = _verdict_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"terminal": "xterm-ghostty|ghostty|", "at": time.time(), "misses": 2}),
            encoding="utf-8",
        )
        assert cli_mod._query_osc11_background() is None
        assert clock.t >= 1.0, "a verdict recorded for another terminal must not apply here"
        clock2 = probe_env()
        assert cli_mod._query_osc11_background() is None
        assert clock2.t >= 1.0, "another terminal's count must restart at one miss"

