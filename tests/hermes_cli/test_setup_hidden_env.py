"""Setup surfaces ask only for what a platform can't start without.

The knobs in SETUP_HIDDEN_ENV_SUFFIXES are self-configuring (/sethome) or
already correct by default, so they don't belong on a setup form. They must
still be reachable everywhere else — this is a presentation change, not a
feature removal.
"""

import pytest

from hermes_cli.setup_hidden_env import is_setup_hidden_env


class TestIsSetupHiddenEnv:
    @pytest.mark.parametrize(
        "key",
        [
            "DISCORD_HOME_CHANNEL",
            "DISCORD_HOME_CHANNEL_NAME",
            "DISCORD_ALLOW_ALL_USERS",
            "DISCORD_REPLY_TO_MODE",
            "MATTERMOST_REPLY_MODE",
            "TELEGRAM_PROXY",
        ],
    )
    def test_self_configuring_knobs_are_hidden(self, key):
        assert is_setup_hidden_env(key)


    def test_applies_to_plugin_platforms_nobody_enumerated(self):
        """Suffix matching is the point — IRC/SimpleX/ntfy get this for free."""
        for key in (
            "IRC_ALLOW_ALL_USERS",
            "SIMPLEX_HOME_CHANNEL",
            "NTFY_HOME_CHANNEL_NAME",
            "LINE_ALLOW_ALL_USERS",
        ):
            assert is_setup_hidden_env(key), key


class TestChannelCards:
    def test_discord_card_asks_for_token_and_allowlist_only(self):
        """The reported bug: the Discord card asked five questions for a
        one-credential platform."""
        from hermes_cli.web_server import _build_catalog_entry

        assert set(_build_catalog_entry("discord")["env_vars"]) == {
            "DISCORD_BOT_TOKEN",
            "DISCORD_ALLOWED_USERS",
        }

    def test_no_card_shows_a_hidden_knob(self):
        from hermes_cli.web_server import _messaging_platform_catalog

        for entry in _messaging_platform_catalog():
            for key in entry["env_vars"]:
                assert not is_setup_hidden_env(key), f"{entry['id']}: {key}"


    def test_hidden_knobs_move_to_the_keys_page_not_into_a_void(self):
        """Keys hides what a Channels card owns. Dropping these from the card
        must hand them back to Keys, not orphan them from every surface."""
        from hermes_cli.web_server import _channel_managed_env_keys

        managed = _channel_managed_env_keys()
        for key in (
            "DISCORD_HOME_CHANNEL",
            "DISCORD_ALLOW_ALL_USERS",
            "DISCORD_REPLY_TO_MODE",
        ):
            assert key not in managed, f"{key} is hidden from both surfaces"


class TestCliWizard:
    """`hermes setup gateway` drops the same knobs. Real wizard, scripted
    stdin, real .env writes under a temp HERMES_HOME."""

    @pytest.fixture
    def home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        return tmp_path

    def _run(self, answers, monkeypatch):
        import io

        from hermes_cli import gateway as gw

        platform = next(p for p in gw._PLATFORMS if p["key"] == "mattermost")
        monkeypatch.setattr("sys.stdin", io.StringIO("\n".join(answers) + "\n"))
        gw._setup_standard_platform(dict(platform))

    def _env(self, home):
        path = home / ".env"
        if not path.exists():
            return {}
        out = {}
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k] = v
        return out

    def test_wizard_stops_asking_about_hidden_knobs(self, home, monkeypatch, capsys):
        # Only the three real answers. If the wizard still prompted for the
        # home channel it would read past them.
        self._run(
            ["https://mm.example.com", "tok-abc", "user26charid"], monkeypatch
        )

        written = self._env(home)
        assert written.get("MATTERMOST_URL") == "https://mm.example.com"
        assert written.get("MATTERMOST_TOKEN") == "tok-abc"
        assert "MATTERMOST_HOME_CHANNEL" not in written
        assert "MATTERMOST_REPLY_MODE" not in written

        out = capsys.readouterr().out
        assert "Home channel" not in out
        assert "Reply mode" not in out

    def test_required_token_still_gates_setup(self, home, monkeypatch):
        """Proves the token wasn't swept out along with the knobs."""
        self._run(["https://mm.example.com", ""], monkeypatch)

        assert "MATTERMOST_TOKEN" not in self._env(home)


class TestStillConfigurable:
    def test_gateway_still_honors_the_env_vars(self, tmp_path, monkeypatch):
        """Nothing was removed from the product — only from the setup form."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "t")
        monkeypatch.setenv("DISCORD_HOME_CHANNEL", "999")
        monkeypatch.setenv("DISCORD_REPLY_TO_MODE", "off")

        from gateway.config import Platform, load_gateway_config

        discord = load_gateway_config().platforms[Platform.DISCORD]
        assert discord.home_channel is not None
        assert discord.home_channel.chat_id == "999"
        assert discord.reply_to_mode == "off"
