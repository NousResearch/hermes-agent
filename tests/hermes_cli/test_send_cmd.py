"""Tests for the ``hermes send`` CLI subcommand.

Covers the argument parsing / stdin / file / list behavior of
``hermes_cli.send_cmd``. The underlying ``send_message_tool`` is stubbed so
no network I/O or gateway is required.
"""

from __future__ import annotations

import io
import json

import pytest

from hermes_cli import send_cmd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse(argv):
    """Build the top-level parser and return the parsed args for ``argv``."""
    import argparse

    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    send_cmd.register_send_subparser(subparsers)
    return parser.parse_args(["send", *argv])


class _FakeTool:
    """Replacement for ``tools.send_message_tool.send_message_tool``."""

    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def __call__(self, args, **_kw):
        self.calls.append(dict(args))
        return json.dumps(self.payload)


@pytest.fixture
def fake_tool(monkeypatch):
    """Install a fake send_message_tool and return the stub for inspection."""
    import sys
    import types

    fake = _FakeTool({"success": True, "message_id": "m123"})

    mod = types.ModuleType("tools.send_message_tool")
    mod.send_message_tool = fake
    # Register the stub so ``from tools.send_message_tool import ...`` inside
    # cmd_send resolves to our fake. Also patch the parent ``tools`` package
    # entry so attribute lookup works.
    monkeypatch.setitem(sys.modules, "tools.send_message_tool", mod)
    return fake


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_whatsapp_mentions_validate_and_reach_native_text_and_media_payloads(monkeypatch, tmp_path, capsys):
    """The CLI-only option stays scoped, normalizes JIDs, and pings once per logical send."""
    import asyncio
    from types import SimpleNamespace

    import aiohttp

    from gateway.config import Platform
    from gateway.platform_registry import platform_registry
    from hermes_cli.plugins import discover_plugins

    calls = []
    bridge_state = {"supports_mentions": True}

    class BridgeResponse:
        status = 200

        def __init__(self, *, health=False):
            self.health = health

        async def json(self):
            if self.health:
                return {"capabilities": {"outboundMentions": bridge_state["supports_mentions"]}}
            return {"messageId": f"m{len(calls)}"}

        async def text(self):
            return ""

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    class BridgeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        def get(self, url, *, timeout):
            return BridgeResponse(health=True)

        def post(self, url, *, json, timeout):
            calls.append((url, json))
            return BridgeResponse()

    discover_plugins()
    whatsapp_config = SimpleNamespace(enabled=True, token=None, extra={"bridge_port": 3000})
    config = SimpleNamespace(
        platforms={Platform.WHATSAPP: whatsapp_config},
        get_home_channel=lambda _platform: None,
    )
    monkeypatch.setattr(send_cmd, "_load_hermes_env", lambda: None)
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)
    monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)
    monkeypatch.setattr("model_tools._run_async", lambda coro: asyncio.run(coro))
    monkeypatch.setattr("tools.send_message_tool._mirror_sent_message", lambda *_args: False)
    monkeypatch.setattr(aiohttp, "ClientSession", lambda *_args, **_kwargs: BridgeSession())

    for argv in (
        ["--to", "telegram", "--mention", "15550000001", "hello"],
        ["--to", "whatsapp:120363000000000000@g.us", "--mention", "not-a-phone", "hello"],
        ["--to", "whatsapp:120363000000000000@g.us", "--mention", "١٥٥٥٠٠٠٠٠٠١", "hello"],
        [
            "--to", "whatsapp:120363000000000000@g.us",
            "--mention", "١٥٥٥٠٠٠٠٠٠١@s.whatsapp.net", "hello",
        ],
    ):
        with pytest.raises(SystemExit) as exc:
            send_cmd.cmd_send(_parse(argv))
        assert exc.value.code == 2
    assert calls == []
    capsys.readouterr()

    async def legacy_sender(_config, _chat_id, _message, *, thread_id=None):
        return {"success": True}

    whatsapp_entry = platform_registry.get("whatsapp")
    assert whatsapp_entry is not None
    current_sender = whatsapp_entry.standalone_sender_fn
    whatsapp_entry.standalone_sender_fn = legacy_sender
    try:
        with pytest.raises(SystemExit) as exc:
            send_cmd.cmd_send(_parse([
                "--to", "whatsapp:120363000000000000@g.us",
                "--mention", "15550000001", "hello @15550000001",
            ]))
        assert exc.value.code == 1
        assert "does not support native mentions" in capsys.readouterr().err
    finally:
        whatsapp_entry.standalone_sender_fn = current_sender
    assert calls == []

    bridge_state["supports_mentions"] = False
    with pytest.raises(SystemExit) as exc:
        send_cmd.cmd_send(_parse([
            "--to", "whatsapp:120363000000000000@g.us",
            "--mention", "15550000001", "hello @15550000001",
        ]))
    assert exc.value.code == 1
    assert "does not support native mentions" in capsys.readouterr().err
    bridge_state["supports_mentions"] = True
    calls.clear()

    image = tmp_path / "photo.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    long_message = "@15550000001 " + "word " * 1000
    text_args = _parse([
        "--to", "whatsapp:120363000000000000@g.us",
        "--mention", "+1 (555) 000-0001",
        "--mention", "15550000001@s.whatsapp.net",
        f"{long_message} MEDIA:{image}",
    ])
    with pytest.raises(SystemExit) as exc:
        send_cmd.cmd_send(text_args)
    assert exc.value.code == 0
    text_payloads = [payload for url, payload in calls if url.endswith("/send")]
    assert len(text_payloads) >= 2
    assert text_payloads[0]["mentions"] == ["15550000001@s.whatsapp.net"]
    assert calls[-1][0].endswith("/send-media")
    assert all("mentions" not in payload for _, payload in calls[1:])

    calls.clear()
    media_args = _parse([
        "--to", "whatsapp:120363000000000000@g.us",
        "--mention", "15550000001",
        f"hello @15550000001 MEDIA:{image}",
    ])
    with pytest.raises(SystemExit) as exc:
        send_cmd.cmd_send(media_args)
    assert exc.value.code == 0
    assert len(calls) == 1 and calls[0][0].endswith("/send-media")
    assert calls[0][1]["mentions"] == ["15550000001@s.whatsapp.net"]










# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------




def test_file_decode_error_suggests_media_directive(fake_tool, capsys, monkeypatch, tmp_path):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    bad = tmp_path / "bad-bytes.bin"
    bad.write_bytes(b"\xff\xfe\x00")

    args = _parse(["--to", "telegram", "--file", str(bad)])
    with pytest.raises(SystemExit) as exc:
        send_cmd.cmd_send(args)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "not a text file" in err.lower()
    assert f"MEDIA:{bad}" in err
    assert "[[as_document]]" in err






# ---------------------------------------------------------------------------
# --list
# ---------------------------------------------------------------------------


def test_list_includes_configured_platform_without_discovered_channels(
    monkeypatch, capsys
):
    """A configured platform absent from the channel directory must still be
    listed (with a no-channels hint) instead of silently omitted."""
    import types
    import sys

    class _FakePlatform:
        def __init__(self, value):
            self.value = value

    class _FakeGwConfig:
        def get_connected_platforms(self):
            return [_FakePlatform("simplex")]

    fake_gw_config = types.ModuleType("gateway.config")
    fake_gw_config.load_gateway_config = lambda: _FakeGwConfig()
    monkeypatch.setitem(sys.modules, "gateway.config", fake_gw_config)

    fake_dir = types.ModuleType("gateway.channel_directory")
    fake_dir.load_directory = lambda: {"updated_at": None, "platforms": {}}

    def _format(platforms=None):
        lines = []
        for name, channels in sorted((platforms or {}).items()):
            lines.append(f"{name}:")
            if not channels:
                lines.append("  (no channels discovered yet)")
        return "\n".join(lines)

    fake_dir.format_directory_for_display = _format
    monkeypatch.setitem(sys.modules, "gateway.channel_directory", fake_dir)

    rc = send_cmd._list_targets(None, json_mode=False)
    out = capsys.readouterr().out
    assert rc == 0
    assert "simplex" in out
    assert "no channels discovered yet" in out


def test_list_json_includes_configured_platform(monkeypatch, capsys):
    import types
    import sys

    class _FakePlatform:
        def __init__(self, value):
            self.value = value

    class _FakeGwConfig:
        def get_connected_platforms(self):
            return [_FakePlatform("simplex"), _FakePlatform("local")]

    fake_gw_config = types.ModuleType("gateway.config")
    fake_gw_config.load_gateway_config = lambda: _FakeGwConfig()
    monkeypatch.setitem(sys.modules, "gateway.config", fake_gw_config)

    fake_dir = types.ModuleType("gateway.channel_directory")
    fake_dir.load_directory = lambda: {
        "updated_at": None,
        "platforms": {"telegram": [{"id": "1", "name": "home"}]},
    }
    fake_dir.format_directory_for_display = lambda platforms=None: ""
    monkeypatch.setitem(sys.modules, "gateway.channel_directory", fake_dir)

    rc = send_cmd._list_targets(None, json_mode=True)
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["platforms"]["simplex"] == []
    assert "local" not in payload["platforms"]  # infra pseudo-platform skipped
    assert payload["platforms"]["telegram"]  # discovered entries preserved


# ---------------------------------------------------------------------------
# Parser registration contract
# ---------------------------------------------------------------------------


def test_register_send_subparser_is_reusable():
    """Sanity check: the registrar returns a parser and wires ``cmd_send``."""
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    send_parser = send_cmd.register_send_subparser(subparsers)
    assert send_parser is not None
    args = parser.parse_args(["send", "--to", "telegram", "hi"])
    assert args.func is send_cmd.cmd_send
    assert args.to == "telegram"
    assert args.message == "hi"


# ---------------------------------------------------------------------------
# Env loader
# ---------------------------------------------------------------------------


def test_load_hermes_env_bridges_config_yaml_scalars(tmp_path, monkeypatch):
    """Top-level config.yaml scalars should be bridged into os.environ.

    This mirrors the gateway/run.py bootstrap behavior: without this, running
    ``hermes send`` from a fresh shell cannot resolve the home channel
    because ``TELEGRAM_HOME_CHANNEL`` (saved by ``hermes config set``) lives
    in config.yaml, not in .env — and the gateway's config loader reads via
    ``os.getenv(...)``.
    """
    import os

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / ".env").write_text("SOME_TOKEN=abc123\n", encoding="utf-8")
    (hermes_home / "config.yaml").write_text(
        "TELEGRAM_HOME_CHANNEL: '5550001111'\nnested:\n  ignored: true\n"
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("TELEGRAM_HOME_CHANNEL", raising=False)
    monkeypatch.delenv("SOME_TOKEN", raising=False)

    # Force get_hermes_home() to re-resolve under the patched env.
    from importlib import reload

    import hermes_cli.config as _hc_config
    reload(_hc_config)

    send_cmd._load_hermes_env()

    assert os.environ.get("SOME_TOKEN") == "abc123"
    assert os.environ.get("TELEGRAM_HOME_CHANNEL") == "5550001111"


def test_load_hermes_env_utf8_bom_preserves_first_key(tmp_path, monkeypatch):
    """A leading UTF-8 BOM must not mangle the first .env key name.

    PowerShell 5.1 `Set-Content -Encoding UTF8` and Notepad prepend a BOM
    (EF BB BF). With encoding=utf-8, python-dotenv kept U+FEFF on the first
    key, so the credential never appeared under its canonical name and
    `hermes send` failed to authenticate.
    """
    import os

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / ".env").write_bytes(
        b"\xef\xbb\xbfSEND_BOM_BOT_TOKEN=tok-first\nSEND_BOM_SECOND=two\n"
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("SEND_BOM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("SEND_BOM_SECOND", raising=False)

    from importlib import reload
    import hermes_cli.config as _hc_config
    reload(_hc_config)

    send_cmd._load_hermes_env()

    assert os.environ.get("SEND_BOM_BOT_TOKEN") == "tok-first"
    assert os.environ.get("SEND_BOM_SECOND") == "two"
    assert "\ufeff" + "SEND_BOM_BOT_TOKEN" not in os.environ

def test_load_hermes_env_bomless_utf8_still_loads(tmp_path, monkeypatch):
    """BOM-less UTF-8 .env files must keep loading after the utf-8-sig switch."""
    import os

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / ".env").write_bytes(b"SEND_PLAIN_TOKEN=plain-val\n")

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("SEND_PLAIN_TOKEN", raising=False)

    from importlib import reload
    import hermes_cli.config as _hc_config
    reload(_hc_config)

    send_cmd._load_hermes_env()

    assert os.environ.get("SEND_PLAIN_TOKEN") == "plain-val"

def test_load_hermes_env_latin1_fallback_still_loads(tmp_path, monkeypatch):
    """Invalid UTF-8 bytes must still load via the latin-1 fallback path,
    and a leading BOM must be stripped before the latin-1 decode so the
    first key keeps its canonical name."""
    import os

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    # BOM + valid first key + latin-1 é (0xE9, invalid UTF-8 alone) in a
    # later value — forces the UnicodeDecodeError → latin-1 stream path.
    (hermes_home / ".env").write_bytes(
        b"\xef\xbb\xbfSEND_L1_TOKEN=tok-l1\nSEND_L1_NOTE=caf\xe9\n"
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("SEND_L1_TOKEN", raising=False)
    monkeypatch.delenv("SEND_L1_NOTE", raising=False)

    from importlib import reload
    import hermes_cli.config as _hc_config
    reload(_hc_config)

    send_cmd._load_hermes_env()

    assert os.environ.get("SEND_L1_TOKEN") == "tok-l1"
    assert os.environ.get("SEND_L1_NOTE") == "caf\xe9"
    assert "\ufeff" + "SEND_L1_TOKEN" not in os.environ

def test_load_hermes_env_latin1_fallback_overrides_shell(tmp_path, monkeypatch):
    """The stream-based latin-1 fallback must keep override=True semantics:
    the .env value wins over a stale shell export, same as the primary path."""
    import os

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    # 0xE9 forces the UnicodeDecodeError \u2192 latin-1 stream fallback.
    (hermes_home / ".env").write_bytes(b"SEND_OVR_TOKEN=caf\xe9-file\n")

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("SEND_OVR_TOKEN", "stale-shell-value")

    from importlib import reload
    import hermes_cli.config as _hc_config
    reload(_hc_config)

    send_cmd._load_hermes_env()

    assert os.environ.get("SEND_OVR_TOKEN") == "caf\xe9-file"

def test_load_hermes_env_fallback_read_error_is_swallowed(tmp_path, monkeypatch):
    """An I/O error inside the latin-1 fallback must not escape \u2014 the send
    path is best-effort by design and must never crash on a broken .env."""
    from pathlib import Path

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    # Invalid UTF-8 so the fallback (and its read_bytes call) is reached.
    (hermes_home / ".env").write_bytes(b"SEND_ERR_TOKEN=caf\xe9\n")

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    def _boom(self):
        raise OSError("disk went away")

    monkeypatch.setattr(Path, "read_bytes", _boom)

    from importlib import reload
    import hermes_cli.config as _hc_config
    reload(_hc_config)

    # Should not raise.
    send_cmd._load_hermes_env()

def test_load_hermes_env_bom_only_env_is_noop(tmp_path, monkeypatch):
    """A .env containing only a BOM must load zero vars without error."""
    import os

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / ".env").write_bytes(b"\xef\xbb\xbf")

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from importlib import reload
    import hermes_cli.config as _hc_config
    reload(_hc_config)

    before = dict(os.environ)
    send_cmd._load_hermes_env()

    added = {k: v for k, v in os.environ.items() if k not in before}
    assert "\ufeff" not in "".join(added)
