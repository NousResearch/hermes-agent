"""Tests for #20591 follow-up: prefer .env on Nous-subscription credential probes.

``hermes_cli.nous_subscription`` aliases ``get_env_value`` to
``get_env_value_prefer_dotenv``.  Every ``get_env_value`` call in that
module is a credential probe gating subscription-feature availability —
a stale shell-exported value masking a freshly rotated ``.env`` key
would silently steer the user to managed-only behaviour even after they
rotate the credential.

We test ``_get_gateway_direct_credentials`` because it exercises the full
set of probed keys (web/image_gen/tts/browser) through the alias.
"""

from pathlib import Path

import pytest


@pytest.fixture
def isolated_hermes_home(tmp_path, monkeypatch):
    """Point HERMES_HOME at a temp dir and clear known probe env vars."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    for key in [
        "FIRECRAWL_API_KEY", "FIRECRAWL_API_URL",
        "PARALLEL_API_KEY", "TAVILY_API_KEY", "EXA_API_KEY",
        "ELEVENLABS_API_KEY", "BROWSER_USE_API_KEY",
        "BROWSERBASE_API_KEY", "BROWSERBASE_PROJECT_ID",
        "FAL_KEY", "OPENAI_API_KEY", "VOICE_TOOLS_OPENAI_KEY",
    ]:
        monkeypatch.delenv(key, raising=False)
    return home


def _write_env_file(home: Path, **kwargs) -> None:
    lines = [f"{k}={v}" for k, v in kwargs.items()]
    (home / ".env").write_text("\n".join(lines) + "\n")


class TestGatewayProbeDotEnvPriority:
    """Probes must report direct credentials present when only .env has them."""

    def test_web_probe_finds_dotenv_only_key(self, isolated_hermes_home):
        _write_env_file(isolated_hermes_home, EXA_API_KEY="sk-exa-dotenv")
        from hermes_cli.nous_subscription import _get_gateway_direct_credentials

        creds = _get_gateway_direct_credentials()
        assert creds["web"] is True

    def test_browser_probe_finds_dotenv_only_keys(self, isolated_hermes_home):
        _write_env_file(
            isolated_hermes_home,
            BROWSERBASE_API_KEY="bb-dotenv",
            BROWSERBASE_PROJECT_ID="prj-dotenv",
        )
        from hermes_cli.nous_subscription import _get_gateway_direct_credentials

        creds = _get_gateway_direct_credentials()
        assert creds["browser"] is True

    def test_tts_probe_finds_dotenv_only_key(self, isolated_hermes_home):
        _write_env_file(isolated_hermes_home, ELEVENLABS_API_KEY="el-dotenv")
        from hermes_cli.nous_subscription import _get_gateway_direct_credentials

        creds = _get_gateway_direct_credentials()
        assert creds["tts"] is True

    def test_no_credentials_anywhere_yields_all_false(self, isolated_hermes_home):
        from hermes_cli.nous_subscription import _get_gateway_direct_credentials

        creds = _get_gateway_direct_credentials()
        assert creds == {
            "web": False,
            "image_gen": False,
            "tts": False,
            "browser": False,
        }


class TestProbeUsesPreferDotenvAlias:
    """The module-level ``get_env_value`` alias must be the prefer-dotenv variant.

    Guards against accidental revert of the import-aliasing in the
    follow-up rebase against PR #20602.  If somebody re-imports the
    plain ``get_env_value`` here, every probe in the file silently
    starts ignoring fresh ``.env`` rotations again (bug class #20591).
    """

    def test_alias_resolves_to_prefer_dotenv_helper(self):
        from hermes_cli import nous_subscription
        from hermes_cli.config import get_env_value_prefer_dotenv

        assert nous_subscription.get_env_value is get_env_value_prefer_dotenv
