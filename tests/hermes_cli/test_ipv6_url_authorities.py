"""IPv6 binds must render as bracketed URL authorities everywhere a URL is built.

An IPv6 literal followed by ``:port`` is only a valid URL authority inside
brackets (``http://[::1]:9119``); unbracketed ``http://::1:9119`` is not a URL a
browser or ``urllib`` can parse. Covers the dashboard attach URL, the
browser auto-open URL, the startup banner, and the shared host formatter
(#67367).
"""
import contextlib
import types

import pytest
import uvicorn

from hermes_cli import main_dashboard, web_server, web_server_lifecycle


def _stub_uvicorn(monkeypatch):
    """Replace uvicorn.Config/Server with no-op fakes so start_server returns."""
    captured: dict = {}

    class _FakeConfig:
        loaded = True
        host = "127.0.0.1"
        port = 8000
        _loop_factory = None

        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

        def load(self):
            pass

        def get_loop_factory(self):
            return self._loop_factory

        class lifespan_class:
            should_exit = False
            state: dict = {}

            def __init__(self, *a, **kw):
                pass

            async def startup(self):
                pass

            async def shutdown(self):
                pass

    class _FakeServer:
        should_exit = False
        started = True
        servers: list = []
        lifespan = None

        @staticmethod
        def capture_signals():
            return contextlib.nullcontext()

        async def startup(self, sockets=None):
            pass

        async def main_loop(self):
            pass

        async def shutdown(self, sockets=None):
            pass

    monkeypatch.setattr(uvicorn, "Config", _FakeConfig)
    monkeypatch.setattr(uvicorn, "Server", lambda config: _FakeServer())
    return captured


class TestFormatUrlHost:

    def test_brackets_ipv6_literal_with_port_following(self):
        from hermes_cli.url_utils import format_url_host

        assert format_url_host("::1") == "[::1]"
        assert format_url_host("fe80::1") == "[fe80::1]"

    def test_preserves_ipv4_and_names(self):
        from hermes_cli.url_utils import format_url_host

        assert format_url_host("127.0.0.1") == "127.0.0.1"
        assert format_url_host("localhost") == "localhost"

    def test_already_bracketed_stays_bracketed(self):
        from hermes_cli.url_utils import format_url_host

        assert format_url_host("[::1]") == "[::1]"

    def test_escapes_ipv6_zone_identifier(self):
        from hermes_cli.url_utils import format_url_host

        assert format_url_host("fe80::1%eth0") == "[fe80::1%25eth0]"
        assert format_url_host("[fe80::1%eth0]") == "[fe80::1%25eth0]"


class TestBrowserOpenUrl:

    def test_maybe_open_browser_brackets_ipv6_loopback(self, monkeypatch):
        """The browser auto-open URL needs brackets around an IPv6 authority."""
        import webbrowser

        opened = []

        class _ImmediateThread:
            def __init__(self, *, target, daemon):
                self._target = target

            def start(self):
                self._target()

        monkeypatch.setattr(web_server_lifecycle.threading, "Thread", _ImmediateThread)
        monkeypatch.setattr(web_server_lifecycle.time, "sleep", lambda _s: None)
        monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url))
        monkeypatch.setenv("DISPLAY", ":0")

        web_server_lifecycle._maybe_open_browser("::1", 9119, True, "worker_x")

        assert opened == ["http://[::1]:9119/?profile=worker_x"]

    def test_start_server_banner_prints_bracketed_ipv6_url(self, monkeypatch, capsys):
        """The no-browser startup banner must show a valid IPv6 URL."""
        _stub_uvicorn(monkeypatch)

        web_server.start_server(host="::1", port=9119, open_browser=False)

        assert "Hermes Web UI → http://[::1]:9119" in capsys.readouterr().err


class TestAttachUrl:

    def test_attach_brackets_ipv6_host_in_browser_url(self, monkeypatch):
        """Named-profile attach to a running IPv6 dashboard must open a valid URL."""
        import webbrowser
        from gateway import host_rendezvous as hr

        opened = []
        monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url))
        record = types.SimpleNamespace(host="::1", port=9119, pid=4242, role="dashboard")
        monkeypatch.setattr(main_dashboard, "_host_backend_attachment", lambda: record)
        monkeypatch.setattr(main_dashboard, "_is_desktop_owned_backend", lambda: False)
        monkeypatch.setattr(main_dashboard, "_explicit_endpoint_flags", lambda: set())
        monkeypatch.setattr(hr, "probe_owner", lambda _r: {"servesSpa": True, "pid": 4242})
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", lambda: "worker_x"
        )

        args = types.SimpleNamespace(
            host="::1", port=9119, no_open=False, isolated=False, open_profile="", ui_surface="dashboard"
        )
        with pytest.raises(SystemExit) as exc:
            main_dashboard._attach_to_host_backend(args)

        assert exc.value.code == 0
        assert opened == ["http://[::1]:9119/?profile=worker_x"]
