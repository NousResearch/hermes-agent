"""Regression tests for ANSI in startup config notices.

`hermes_cli/config.py`'s bootstrap notices (`print_config_warnings`,
`warn_deprecated_cwd_env_vars`) used to write hardcoded SGR codes straight to
``sys.stderr``. Two consumers cannot render those:

* a redirected stderr (launchd's ``gateway.error.log``, systemd, pipes) — the
  escapes land in the log as noise;
* prompt_toolkit's ``StdoutProxy``, which is what ``patch_stdout()`` installs as
  ``sys.stdout`` *and* ``sys.stderr`` while the interactive CLI owns the
  terminal. Its ``isatty()`` forwards the terminal's ``True``, but it feeds text
  through ``Vt100_Output.write()``, which replaces every ESC byte with ``?`` —
  so the user sees ``?[33m⚠ Deprecated .env settings detected:?[0m``.

Both are covered by ``hermes_cli.colors.color(..., stream=...)``: the TTY check
follows the stream being written to, and a prompt_toolkit proxy is treated as
"no color" because it mangles raw SGR.
"""

import io

import pytest


class _FakeStream:
    """Stream stand-in reporting a fixed TTY status and recording writes."""

    def __init__(self, tty: bool):
        self._tty = tty
        self.written = io.StringIO()

    def isatty(self):
        return self._tty

    def write(self, s):
        self.written.write(s)
        return len(s)

    def flush(self):
        pass

    def getvalue(self):
        return self.written.getvalue()


class StdoutProxy(_FakeStream):
    """Same shape as prompt_toolkit's ``patch_stdout`` proxy: TTY *and* mangling.

    ``hermes_cli.colors`` matches that class by name (it must not hard-import
    prompt_toolkit), so the fake carries the real name.
    """

    def __init__(self):
        super().__init__(tty=True)


class TestColorsStreamParam:
    def test_default_stream_is_stdout(self, monkeypatch):
        from hermes_cli.colors import color

        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("TERM", "xterm-256color")
        monkeypatch.setattr("sys.stdout", _FakeStream(tty=True))
        monkeypatch.setattr("sys.stderr", _FakeStream(tty=False))
        assert "\033[" in color("x", "\033[1m")

    def test_stderr_stream_gates_on_stderr(self, monkeypatch):
        from hermes_cli.colors import color

        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("TERM", "xterm-256color")
        monkeypatch.setattr("sys.stdout", _FakeStream(tty=True))
        monkeypatch.setattr("sys.stderr", _FakeStream(tty=False))
        import sys

        assert "\033[" not in color("x", "\033[1m", stream=sys.stderr)

    def test_no_color_disables(self, monkeypatch):
        from hermes_cli.colors import color

        monkeypatch.setenv("NO_COLOR", "1")
        monkeypatch.setenv("TERM", "xterm-256color")
        monkeypatch.setattr("sys.stdout", _FakeStream(tty=True))
        import sys

        assert "\033[" not in color("x", "\033[1m", stream=sys.stdout)

    def test_term_dumb_disables(self, monkeypatch):
        from hermes_cli.colors import color

        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("TERM", "dumb")
        monkeypatch.setattr("sys.stdout", _FakeStream(tty=True))
        import sys

        assert "\033[" not in color("x", "\033[1m", stream=sys.stdout)

    def test_prompt_toolkit_proxy_is_never_colored(self, monkeypatch):
        """isatty() is True here, but SGR written to it comes out as `?[33m`."""
        from hermes_cli.colors import color, should_use_color

        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("TERM", "xterm-256color")
        proxy = StdoutProxy()
        assert proxy.isatty() is True
        assert should_use_color(proxy) is False
        assert "\033[" not in color("x", "\033[1m", stream=proxy)

    def test_default_path_keeps_zero_arg_hook(self, monkeypatch):
        """Existing sites patch should_use_color with a no-arg callable (e.g.
        tests/hermes_cli/test_cron.py) — the default color() path must keep
        calling it that way."""
        import hermes_cli.colors as colors_mod

        monkeypatch.setattr(colors_mod, "should_use_color", lambda: True)
        assert "\033[" in colors_mod.color("x", "\033[1m")


class TestStartupNoticesArePlain:
    """The two bootstrap notices never emit ESC when the stream mangles or pipes."""

    def _env_home(self, monkeypatch, tmp_path):
        home = tmp_path / ".hermes"
        home.mkdir()
        (home / ".env").write_text("MESSAGING_CWD=/legacy/path\n", encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.delenv("MESSAGING_CWD", raising=False)
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        return home

    def test_deprecation_notice_plain_under_toolkit_proxy(self, monkeypatch, tmp_path):
        self._env_home(monkeypatch, tmp_path)
        proxy = StdoutProxy()
        monkeypatch.setattr("sys.stderr", proxy)

        from hermes_cli.config import warn_deprecated_cwd_env_vars

        warn_deprecated_cwd_env_vars()

        out = proxy.getvalue()
        assert "Deprecated .env settings detected" in out
        assert "\033" not in out
        assert "?[33m" not in out

    def test_deprecation_notice_plain_on_pipe(self, monkeypatch, tmp_path):
        self._env_home(monkeypatch, tmp_path)
        monkeypatch.setattr("sys.stderr", _FakeStream(tty=False))

        import sys
        from hermes_cli.config import warn_deprecated_cwd_env_vars

        warn_deprecated_cwd_env_vars()

        assert "\033" not in sys.stderr.getvalue()

    def test_deprecation_notice_keeps_color_on_real_tty(self, monkeypatch, tmp_path):
        self._env_home(monkeypatch, tmp_path)
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("TERM", "xterm-256color")
        tty = _FakeStream(tty=True)
        monkeypatch.setattr("sys.stderr", tty)

        from hermes_cli.config import warn_deprecated_cwd_env_vars

        warn_deprecated_cwd_env_vars()

        assert "\033[33m" in tty.getvalue()

    def test_config_warnings_plain_under_toolkit_proxy(self, monkeypatch):
        proxy = StdoutProxy()
        monkeypatch.setattr("sys.stderr", proxy)

        from hermes_cli.config import print_config_warnings

        print_config_warnings({
            "terminal": {"cwd": "x"},
            "custom_providers": {"a": {}},
            "fallback_model": {"provider": "x", "default": "y", "base_url": "z"},
        })

        out = proxy.getvalue()
        assert "Config issues detected" in out
        assert "\033" not in out
        assert "?[33m" not in out


class TestToolkitProxyContract:
    def test_patch_stdout_class_is_named_stdout_proxy(self):
        """Guards the duck-type in hermes_cli.colors._is_sgr_mangling_proxy()."""
        patch_stdout = pytest.importorskip("prompt_toolkit.patch_stdout")
        import sys

        try:
            with patch_stdout.patch_stdout():
                name = type(sys.stderr).__name__
        except Exception as exc:  # no usable app/output under this pytest run
            pytest.skip(f"patch_stdout unavailable here: {exc}")
        assert name == "StdoutProxy"
