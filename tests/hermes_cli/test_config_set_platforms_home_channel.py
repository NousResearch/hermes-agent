"""Regression tests for #33141: ``platforms.<name>.home_channel`` write shape.

`hermes config set platforms.<name>.home_channel <chat_id>` used to persist a bare
string, but ``PlatformConfig.from_dict`` (gateway/config.py) drops a non-mapping
``home_channel`` silently — the write reported success while the gateway runtime
never saw it. The set path now wraps a bare chat id into the mapping
``HomeChannel.from_dict`` reads.
"""

from pathlib import Path

import yaml


def _write_config(hermes_home: Path, data: dict) -> Path:
    hermes_home.mkdir(parents=True, exist_ok=True)
    config_path = hermes_home / "config.yaml"
    config_path.write_text(yaml.dump(data))
    return config_path


def _set(monkeypatch, hermes_home, key, value, force=False):
    """Isolated call to set_config_value against a temp HERMES_HOME."""
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    from hermes_cli.config import set_config_value

    set_config_value(key, value, force=force)


def _read_back(hermes_home, platform):
    result = yaml.safe_load((hermes_home / "config.yaml").read_text())
    return result["platforms"][platform]


def _home_channel_for(platform_config):
    from gateway.config import PlatformConfig

    return PlatformConfig.from_dict(platform_config).home_channel


def _make_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    _write_config(
        home,
        {
            "model": {"default": "test-model", "provider": "openrouter"},
            "platforms": {
                "telegram": {"token": "secret-bot-token"},
            },
        },
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


class TestBareChatIdWrapped:
    def test_string_chat_id_persists_as_mapping(self, tmp_path, monkeypatch):
        home = _make_home(tmp_path, monkeypatch)
        _set(
            monkeypatch,
            home,
            "platforms.telegram.home_channel",
            "oc_b0b897cce6e70190959898f94fe8f9a8",
        )

        stored = _read_back(home, "telegram")["home_channel"]
        assert stored == {
            "platform": "telegram",
            "chat_id": "oc_b0b897cce6e70190959898f94fe8f9a8",
            "name": "Home",
        }
        # Sibling connection keys survive the write.
        assert _read_back(home, "telegram")["token"] == "secret-bot-token"

    def test_gateway_reads_wrapped_home_channel(self, tmp_path, monkeypatch):
        """The persisted shape must survive the gateway's PlatformConfig.from_dict."""
        home = _make_home(tmp_path, monkeypatch)
        _set(
            monkeypatch,
            home,
            "platforms.telegram.home_channel",
            "oc_b0b897cce6e70190959898f94fe8f9a8",
        )

        home_channel = _home_channel_for(_read_back(home, "telegram"))
        assert home_channel is not None
        assert home_channel.chat_id == "oc_b0b897cce6e70190959898f94fe8f9a8"
        assert home_channel.name == "Home"

    def test_numeric_chat_id_wrapped_as_string(self, tmp_path, monkeypatch):
        """Coercion turns an all-digit chat id into an int before the wrap; it must
        still land as the exact string chat_id HomeChannel expects. A real 19-digit
        Discord snowflake pins this: a float-mediated round-trip would zero its tail
        (``1532136816336044092`` -> ``...000``) and address a channel that does not
        exist, while a short id survives such a round-trip and cannot tell the two
        apart."""
        home = _make_home(tmp_path, monkeypatch)
        _set(monkeypatch, home, "platforms.discord.home_channel", "1532136816336044092")

        stored = _read_back(home, "discord")["home_channel"]
        assert stored == {
            "platform": "discord",
            "chat_id": "1532136816336044092",
            "name": "Home",
        }
        assert (
            _home_channel_for(_read_back(home, "discord")).chat_id
            == "1532136816336044092"
        )

    def test_unrelated_platform_key_not_wrapped(self, tmp_path, monkeypatch):
        home = _make_home(tmp_path, monkeypatch)
        _set(monkeypatch, home, "platforms.telegram.token", "new-token")

        stored = _read_back(home, "telegram")
        assert stored["token"] == "new-token"
        assert "home_channel" not in stored


class TestExplicitMappingPassthrough:
    def test_mapping_value_untouched(self, tmp_path, monkeypatch):
        """An explicit mapping (already the shape from_dict reads) passes through
        verbatim — including thread_id and a custom name."""
        home = _make_home(tmp_path, monkeypatch)
        _set(
            monkeypatch,
            home,
            "platforms.telegram.home_channel",
            "{platform: telegram, chat_id: '42', name: Ops, thread_id: '7'}",
        )

        stored = _read_back(home, "telegram")["home_channel"]
        assert stored == {
            "platform": "telegram",
            "chat_id": "42",
            "name": "Ops",
            "thread_id": "7",
        }
        home_channel = _home_channel_for(_read_back(home, "telegram"))
        assert home_channel.chat_id == "42"
        assert home_channel.name == "Ops"
        assert home_channel.thread_id == "7"


class TestRootLevelTwoSegmentSpelling:
    """``hermes config set <platform>.home_channel <id>`` writes a root-level ``<platform>:``
    block that no reader picks up (``merge_platform_sections`` never copies a root block while
    ``home_channel`` is a typed key) — the same silent dead end as the bare three-segment write.
    The set path now redirects the key onto ``platforms.<name>.home_channel`` first."""

    def test_two_segment_key_redirects_to_platforms_block(self, tmp_path, monkeypatch):
        home = _make_home(tmp_path, monkeypatch)
        _set(monkeypatch, home, "discord.home_channel", "1532136816336044092")

        raw = yaml.safe_load((home / "config.yaml").read_text())
        # No root-level dead-end block is created...
        assert "discord" not in raw
        # ...and the value lands wrapped on the path the gateway reads.
        assert raw["platforms"]["discord"]["home_channel"] == {
            "platform": "discord",
            "chat_id": "1532136816336044092",
            "name": "Home",
        }

    def test_two_segment_redirect_survives_gateway_read(self, tmp_path, monkeypatch):
        home = _make_home(tmp_path, monkeypatch)
        _set(monkeypatch, home, "discord.home_channel", "1532136816336044092")

        home_channel = _home_channel_for(_read_back(home, "discord"))
        assert home_channel is not None
        assert home_channel.chat_id == "1532136816336044092"

    def test_unknown_two_segment_prefix_not_redirected(self, tmp_path, monkeypatch):
        """A two-segment key whose first segment is not a platform keeps its plain
        meaning — only recognized platforms redirect."""
        home = _make_home(tmp_path, monkeypatch)
        _set(monkeypatch, home, "display.home_channel", "oops")

        raw = yaml.safe_load((home / "config.yaml").read_text())
        assert raw["display"] == {"home_channel": "oops"}
        assert "platforms" not in raw or "display" not in raw["platforms"]
