"""Tests for the ``hermes send`` CLI subcommand.

Covers the argument parsing / stdin / file / list behavior of
``hermes_cli.send_cmd``. The underlying ``send_message_tool`` is stubbed so
no network I/O or gateway is required.
"""

from __future__ import annotations

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


@pytest.fixture
def whatsapp_bridge(monkeypatch):
    """Route ``hermes send --to whatsapp:...`` through the real plugin standalone sender into a fake
    bridge; returns the recorded ``(path, payload)`` posts and a mutable ``supports_mentions`` flag."""
    import asyncio
    from types import SimpleNamespace

    import aiohttp

    from gateway.config import Platform
    from hermes_cli.plugins import discover_plugins

    calls = []
    state = {"supports_mentions": True}

    class BridgeResponse:
        status = 200

        def __init__(self, *, health=False):
            self.health = health

        async def json(self):
            if self.health:
                return {"capabilities": {"outboundMentions": state["supports_mentions"]}}
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
            calls.append((url.rsplit("/", 1)[-1], json))
            return BridgeResponse()

    discover_plugins()
    config = SimpleNamespace(
        platforms={Platform.WHATSAPP: SimpleNamespace(enabled=True, token=None, extra={"bridge_port": 3000})},
        get_home_channel=lambda _platform: None,
    )
    monkeypatch.setattr(send_cmd, "_load_hermes_env", lambda: None)
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)
    monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)
    monkeypatch.setattr("model_tools._run_async", lambda coro: asyncio.run(coro))
    monkeypatch.setattr("tools.send_message_tool._mirror_sent_message", lambda *_args: False)
    monkeypatch.setattr(aiohttp, "ClientSession", lambda *_args, **_kwargs: BridgeSession())
    return SimpleNamespace(calls=calls, state=state)


_GROUP = "whatsapp:120363000000000000@g.us"


@pytest.mark.parametrize("argv", [
    ["--to", "telegram", "--mention", "15550000001", "hello"],
    ["--to", _GROUP, "--mention", "not-a-phone", "hello"],
    ["--to", _GROUP, "--mention", "\u0661\u0665\u0665\u0665\u0660\u0660\u0660\u0660\u0660\u0660\u0661", "hello"],
    ["--to", _GROUP, "--mention", "\u0661\u0665\u0665\u0665\u0660\u0660\u0660\u0660\u0660\u0660\u0661@s.whatsapp.net", "hello"],
])
def test_whatsapp_mention_rejections_never_reach_the_bridge(whatsapp_bridge, capsys, argv):
    """Non-WhatsApp targets and non-ASCII / non-numeric mention values are usage errors (exit 2)
    raised before any delivery attempt."""
    with pytest.raises(SystemExit) as exc:
        send_cmd.cmd_send(_parse(argv))
    assert exc.value.code == 2
    assert "mention" in capsys.readouterr().err.lower()
    assert whatsapp_bridge.calls == []


def test_whatsapp_mentions_ride_the_first_bridge_payload_only(whatsapp_bridge, tmp_path, capsys):
    """Across chunked text and text+media, exactly one bridge payload carries the normalized,
    deduplicated JIDs; a captioned single-media send carries them on the media payload; a bridge
    without native-mention support fails closed instead of silently sending an unmentioned message."""
    calls = whatsapp_bridge.calls
    image = tmp_path / "photo.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    long_message = "@15550000001 " + "word " * 1000

    with pytest.raises(SystemExit) as exc:
        send_cmd.cmd_send(_parse([
            "--to", _GROUP, "--mention", "+1 (555) 000-0001", "--mention", "15550000001@s.whatsapp.net",
            f"{long_message} MEDIA:{image}",
        ]))
    assert exc.value.code == 0
    text_payloads = [payload for path, payload in calls if path == "send"]
    assert len(text_payloads) >= 2 and calls[-1][0] == "send-media"
    assert text_payloads[0]["mentions"] == ["15550000001@s.whatsapp.net"]
    assert all("mentions" not in payload for _, payload in calls[1:])

    calls.clear()
    with pytest.raises(SystemExit) as exc:
        send_cmd.cmd_send(_parse(["--to", _GROUP, "--mention", "15550000001", f"hello @15550000001 MEDIA:{image}"]))
    assert exc.value.code == 0
    assert [path for path, _ in calls] == ["send-media"]
    assert calls[0][1]["mentions"] == ["15550000001@s.whatsapp.net"]

    calls.clear()
    whatsapp_bridge.state["supports_mentions"] = False
    with pytest.raises(SystemExit) as exc:
        send_cmd.cmd_send(_parse(["--to", _GROUP, "--mention", "15550000001", "hello @15550000001"]))
    assert exc.value.code == 1
    assert "does not support native mentions" in capsys.readouterr().err
    assert calls == []








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
    fake_dir.merge_session_channels = lambda platforms: None

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
    fake_dir.merge_session_channels = lambda platforms: None
    monkeypatch.setitem(sys.modules, "gateway.channel_directory", fake_dir)

    rc = send_cmd._list_targets(None, json_mode=True)
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["platforms"]["simplex"] == []
    assert "local" not in payload["platforms"]  # infra pseudo-platform skipped
    assert payload["platforms"]["telegram"]  # discovered entries preserved


def test_list_merges_session_dm_into_stale_directory(tmp_path, monkeypatch, capsys):
    """A DM whose session row exists but whose directory entry is a rebuild away must
    still appear in `hermes send --list` (unfiltered text, and filtered --json).

    Real gateway.channel_directory end-to-end (#48303): the CLI builds its own
    platforms dict, so it has to run the session merge itself.
    """
    dm_chat_id = "1180000000000000001"
    directory = tmp_path / "channel_directory.json"
    directory.write_text(json.dumps({"updated_at": None, "platforms": {
        "discord": [{"id": "999000111222333444", "name": "general",
                     "guild": "Some Server", "type": "channel"}],
    }}))
    sessions = tmp_path / "sessions" / "sessions.json"
    sessions.parent.mkdir(parents=True)
    sessions.write_text(json.dumps({
        "s_discord_new_dm": {
            "origin": {"platform": "discord", "chat_id": dm_chat_id,
                       "chat_name": "new-dm-contact"},
            "chat_type": "dm",
        },
    }))

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("gateway.channel_directory.DIRECTORY_PATH", directory)
    monkeypatch.setattr("gateway.channel_directory.CHANNEL_ALIASES_PATH",
                        tmp_path / "no-aliases.json")

    class _FakeGwConfig:
        def get_connected_platforms(self):
            return []

    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: _FakeGwConfig())

    rc = send_cmd._list_targets(None, json_mode=False)
    out = capsys.readouterr().out
    assert rc == 0
    assert "discord:#general" in out  # stale entry preserved
    assert "discord:new-dm-contact" in out  # the gap this closes

    rc = send_cmd._list_targets("discord", json_mode=True)
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert any(entry["id"] == dm_chat_id for entry in payload["platforms"]["discord"])


# ---------------------------------------------------------------------------
# Parser registration contract
# ---------------------------------------------------------------------------




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
    the .env value wins over a stale shell export, same as the primary path. (A non-credential
    key name: ``*_TOKEN`` values are ASCII-sanitized by the shared loader, by design.)"""
    import os

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    # 0xE9 forces the UnicodeDecodeError \u2192 latin-1 stream fallback.
    (hermes_home / ".env").write_bytes(b"SEND_OVR_LABEL=caf\xe9-file\n")

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("SEND_OVR_LABEL", "stale-shell-value")

    from importlib import reload
    import hermes_cli.config as _hc_config
    reload(_hc_config)

    send_cmd._load_hermes_env()

    assert os.environ.get("SEND_OVR_LABEL") == "caf\xe9-file"

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


def test_help_and_empty_list_hint_name_the_resolved_home(tmp_path, monkeypatch, capsys):
    """``--help`` and the ``--list`` empty-state hint derive their paths from the resolved home instead of a
    hardcoded ``~/.hermes`` (absent on a Windows install or under a profile home)."""
    import argparse
    import sys
    import types

    home = tmp_path / "AppData" / "Local" / "hermes"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))

    parser = argparse.ArgumentParser(prog="hermes")
    send_parser = send_cmd.register_send_subparser(parser.add_subparsers(dest="command"))
    help_text = send_parser.format_help()
    assert str(home / ".env") in help_text and str(home / "config.yaml") in help_text
    assert "~/.hermes" not in help_text

    fake_gw_config = types.ModuleType("gateway.config")
    fake_gw_config.load_gateway_config = lambda: types.SimpleNamespace(get_connected_platforms=lambda: [])
    monkeypatch.setitem(sys.modules, "gateway.config", fake_gw_config)
    fake_dir = types.ModuleType("gateway.channel_directory")
    fake_dir.load_directory = lambda: {"updated_at": None, "platforms": {}}
    fake_dir.format_directory_for_display = lambda platforms=None: ""
    fake_dir.merge_session_channels = lambda platforms: None
    monkeypatch.setitem(sys.modules, "gateway.channel_directory", fake_dir)

    assert send_cmd._list_targets(None, json_mode=False) == 0
    out = capsys.readouterr().out
    assert str(home / "channel_directory.json") in out
    assert "~/.hermes" not in out


