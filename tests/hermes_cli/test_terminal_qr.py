"""Tests for hermes_cli/terminal_qr.py and the ``hermes dashboard --qr`` gate."""

import builtins
import socket

import pytest

from hermes_cli import terminal_qr


class TestResolveAdvertisedUrl:
    def test_public_url_wins_and_is_normalized(self, monkeypatch):
        monkeypatch.setattr(terminal_qr, "lan_ip", lambda: "192.168.1.20")
        assert (
            terminal_qr.resolve_advertised_url("0.0.0.0", 9119, "https://h.example.com/")
            == "https://h.example.com"
        )

    def test_wildcard_bind_advertises_lan_ip(self, monkeypatch):
        monkeypatch.setattr(terminal_qr, "lan_ip", lambda: "192.168.1.20")
        assert (
            terminal_qr.resolve_advertised_url("0.0.0.0", 9119)
            == "http://192.168.1.20:9119"
        )

    def test_wildcard_bind_without_lan_falls_back_to_bind_host(self, monkeypatch):
        monkeypatch.setattr(terminal_qr, "lan_ip", lambda: None)
        assert (
            terminal_qr.resolve_advertised_url("0.0.0.0", 9119)
            == "http://0.0.0.0:9119"
        )

    def test_concrete_bind_advertises_itself(self, monkeypatch):
        monkeypatch.setattr(
            terminal_qr, "lan_ip", lambda: pytest.fail("must not probe LAN")
        )
        assert (
            terminal_qr.resolve_advertised_url("192.168.1.20", 9119)
            == "http://192.168.1.20:9119"
        )


class TestLanIp:
    def test_offline_machine_returns_none(self, monkeypatch):
        class _DeadSocket:
            def __init__(self, *a, **k):
                raise OSError("network unreachable")

        monkeypatch.setattr(socket, "socket", _DeadSocket)
        assert terminal_qr.lan_ip() is None

    def test_loopback_source_address_is_rejected(self, monkeypatch):
        class _LoopbackSocket:
            def __init__(self, *a, **k):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def settimeout(self, *_a):
                pass

            def connect(self, *_a):
                pass

            def getsockname(self):
                return ("127.0.0.1", 54321)

        monkeypatch.setattr(socket, "socket", _LoopbackSocket)
        assert terminal_qr.lan_ip() is None


class TestRenderer:
    def test_missing_qrcode_returns_false(self, monkeypatch):
        real_import = builtins.__import__

        def _no_qrcode(name, *args, **kwargs):
            if name == "qrcode":
                raise ImportError("no module named qrcode")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_qrcode)
        assert terminal_qr.render_qr_to_terminal("http://example.com") is False

    def test_renders_half_block_qr_when_available(self, capsys):
        pytest.importorskip("qrcode")
        assert terminal_qr.render_qr_to_terminal("http://192.168.1.20:9119") is True
        out = capsys.readouterr().out
        assert any(ch in out for ch in ("█", "▀", "▄"))


class TestDashboardQrGate:
    def test_loopback_bind_prints_guidance_not_qr(self, capsys):
        from hermes_cli.web_server import _print_connect_qr

        _print_connect_qr("127.0.0.1", 9119, "")
        out = capsys.readouterr().out
        assert "phone cannot" in out
        assert "Scan to open" not in out

    def test_public_url_prints_scannable_url(self, capsys, monkeypatch):
        pytest.importorskip("qrcode")
        from hermes_cli import web_server

        _print_connect_qr = web_server._print_connect_qr
        _print_connect_qr("127.0.0.1", 9119, "https://hermes.example.com")
        out = capsys.readouterr().out
        assert "Scan to open on your phone → https://hermes.example.com" in out
        assert any(ch in out for ch in ("█", "▀", "▄"))

    def test_configured_public_url_overrides_loopback_bind(self, capsys, monkeypatch):
        from hermes_cli import web_server
        from hermes_cli.dashboard_auth import prefix

        rendered = []
        monkeypatch.setattr(
            prefix,
            "resolve_public_url",
            lambda: "https://configured.example.com/dashboard",
        )
        monkeypatch.setattr(
            terminal_qr,
            "render_qr_to_terminal",
            lambda url: rendered.append(url) or True,
        )

        web_server._print_connect_qr("127.0.0.1", 9119, "")

        out = capsys.readouterr().out
        assert "Scan to open on your phone → https://configured.example.com/dashboard" in out
        assert "phone cannot" not in out
        assert rendered == ["https://configured.example.com/dashboard"]

    def test_invalid_explicit_url_falls_back_to_config(self, capsys, monkeypatch):
        from hermes_cli import web_server
        from hermes_cli.dashboard_auth import prefix

        rendered = []
        monkeypatch.setattr(
            prefix,
            "resolve_public_url",
            lambda: "https://configured.example.com",
        )
        monkeypatch.setattr(
            terminal_qr,
            "render_qr_to_terminal",
            lambda url: rendered.append(url) or True,
        )

        web_server._print_connect_qr("127.0.0.1", 9119, "file:///tmp/dashboard")

        assert "https://configured.example.com" in capsys.readouterr().out
        assert rendered == ["https://configured.example.com"]

    def test_missing_qrcode_prints_guidance_without_installing(
        self, capsys, monkeypatch
    ):
        from hermes_cli import web_server

        monkeypatch.setattr(
            terminal_qr,
            "ensure_qrcode_installed",
            lambda: pytest.fail("dashboard startup must not install qrcode"),
        )
        monkeypatch.setattr(
            terminal_qr,
            "render_qr_to_terminal",
            lambda _url: False,
        )

        web_server._print_connect_qr(
            "127.0.0.1",
            9119,
            "https://hermes.example.com",
        )

        out = capsys.readouterr().out
        assert "Scan to open on your phone → https://hermes.example.com" in out
        assert "pip install qrcode" in out
