"""Tests for GatewayRunner._format_session_info — session config surfacing."""

import pytest
from unittest.mock import patch

from gateway.run import GatewayRunner


@pytest.fixture()
def runner():
    """Create a bare GatewayRunner without __init__."""
    return GatewayRunner.__new__(GatewayRunner)


def _patch_info(tmp_path, config_yaml, model, runtime):
    """Return a context-manager stack that patches _format_session_info deps."""
    cfg_path = tmp_path / "config.yaml"
    if config_yaml is not None:
        cfg_path.write_text(config_yaml)
    return (
        patch("gateway.run._hermes_home", tmp_path),
        patch("gateway.run._resolve_gateway_model", return_value=model),
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value=runtime),
    )


class TestFormatSessionInfo:

    def test_includes_model_name(self, runner, tmp_path):
        p1, p2, p3 = _patch_info(tmp_path, "model:\n  default: anthropic/claude-opus-4.6\n  provider: openrouter\n",
                                  "anthropic/claude-opus-4.6",
                                  {"provider": "openrouter", "base_url": "https://openrouter.ai/api/v1", "api_key": "k"})
        with p1, p2, p3:
            info = runner._format_session_info()
        assert "claude-opus-4.6" in info


    def test_config_context_length(self, runner, tmp_path):
        p1, p2, p3 = _patch_info(tmp_path, "model:\n  default: test-model\n  context_length: 32768\n",
                                  "test-model",
                                  {"provider": "custom", "base_url": "", "api_key": ""})
        with p1, p2, p3:
            info = runner._format_session_info()
        assert "32K" in info
        assert "config" in info

    def test_default_fallback_hint(self, runner, tmp_path):
        p1, p2, p3 = _patch_info(tmp_path, "model:\n  default: unknown-model-xyz\n",
                                  "unknown-model-xyz",
                                  {"provider": "", "base_url": "", "api_key": ""})
        with p1, p2, p3:
            info = runner._format_session_info()
        assert "256K" in info
        assert "model.context_length" in info

    def test_local_endpoint_shown(self, runner, tmp_path):
        p1, p2, p3 = _patch_info(
            tmp_path,
            "model:\n  default: qwen3:8b\n  provider: custom\n  base_url: http://localhost:11434/v1\n  context_length: 8192\n",
            "qwen3:8b",
            {"provider": "custom", "base_url": "http://localhost:11434/v1", "api_key": ""})
        with p1, p2, p3:
            info = runner._format_session_info()
        assert "localhost:11434" in info
        assert "8K" in info

    def test_named_custom_provider_keeps_context_pin_without_model_base_url(
        self, runner, tmp_path
    ):
        """Session-reset banner must honor model.context_length for named custom providers.

        Repro: /status shows 262144 from config while the reset banner said
        ``131K tokens (detected)`` because empty model.base_url + runtime URL
        falsely cleared the pin and fell through to the Qwen family default.
        """
        model = "custom-local-agentw/Qwen-AgentWorld-35B-A3B-Q5_K_XL"
        config_yaml = (
            "model:\n"
            f"  default: {model}\n"
            "  provider: custom-local-agentw\n"
            "  context_length: 262144\n"
            "custom_providers:\n"
            "  - name: custom-local-agentw\n"
            "    base_url: http://127.0.0.1:8080/v1\n"
            "    models: {}\n"
        )
        p1, p2, p3 = _patch_info(
            tmp_path,
            config_yaml,
            model,
            {
                "provider": "custom-local-agentw",
                "base_url": "http://127.0.0.1:8080/v1",
                "api_key": "",
            },
        )
        with p1, p2, p3, patch(
            "hermes_cli.config.get_compatible_custom_providers",
            return_value=[
                {
                    "name": "custom-local-agentw",
                    "base_url": "http://127.0.0.1:8080/v1",
                    "models": {},
                }
            ],
        ), patch(
            "agent.model_metadata.get_model_context_length",
            side_effect=lambda *args, **kwargs: (
                kwargs.get("config_context_length")
                if kwargs.get("config_context_length")
                else 131072
            ),
        ):
            info = runner._format_session_info()
        assert "262K" in info
        assert "config" in info
        assert "131K" not in info


class TestResetNoticeSessionInfo:
    """#59003: the auto-reset banner must report the serving profile's config,
    not the multiplexer's base config."""

    _RUNTIME = {"provider": "", "base_url": "", "api_key": ""}

    def _source(self):
        from gateway.config import Platform
        from gateway.session import SessionSource
        return SessionSource(
            platform=Platform.TELEGRAM, chat_id="123", user_id="u1",
            profile="planner",
        )

    def _homes(self, tmp_path):
        base = tmp_path / "base"
        profile = tmp_path / "profiles" / "planner"
        profile.mkdir(parents=True)
        base.mkdir()
        base.joinpath("config.yaml").write_text(
            "model:\n  default: base-model\n  provider: custom\n  context_length: 1000\n")
        profile.joinpath("config.yaml").write_text(
            "model:\n  default: profile-model\n  provider: anthropic\n  context_length: 2000\n")
        return base, profile

    def test_multiplex_uses_profile_config(self, runner, tmp_path):
        from types import SimpleNamespace
        base, profile = self._homes(tmp_path)
        runner.config = SimpleNamespace(multiplex_profiles=True)
        with patch("gateway.run._hermes_home", base), \
             patch.object(GatewayRunner, "_resolve_profile_home_for_source", return_value=profile), \
             patch("gateway.run._resolve_runtime_agent_kwargs", return_value=self._RUNTIME):
            info = runner._reset_notice_session_info(self._source())
        assert "profile-model" in info
        assert "anthropic" in info
        assert "base-model" not in info



class TestSessionInfoLanguage:
    """#57324: the reset banner must render in the language of the profile it serves.

    ``agent.i18n`` memoizes ``display.language`` for the whole process and
    ``_profile_runtime_scope`` neither scopes nor clears that cache, so letting ``t()``
    resolve the language on its own renders every multiplexed profile in whichever
    language happened to warm the cache first.
    """

    _RUNTIME = {"provider": "", "base_url": "", "api_key": ""}

    @pytest.fixture(autouse=True)
    def _clean_language_cache(self, monkeypatch):
        """Start and finish on a cold cache so ordering inside a test is what decides."""
        from agent import i18n
        monkeypatch.delenv("HERMES_LANGUAGE", raising=False)
        i18n.reset_language_cache()
        yield
        i18n.reset_language_cache()

    def _source(self, profile: str):
        from gateway.config import Platform
        from gateway.session import SessionSource
        return SessionSource(
            platform=Platform.TELEGRAM, chat_id="123", user_id="u1", profile=profile,
        )

    def _homes(self, tmp_path, profiles):
        """Base home plus one profile home per ``(name, language)`` pair."""
        base = tmp_path / "base"
        base.mkdir()
        base.joinpath("config.yaml").write_text(
            "model:\n  default: base-model\n  provider: custom\n  context_length: 1000\n")
        homes = {}
        for name, lang in profiles:
            home = tmp_path / "profiles" / name
            home.mkdir(parents=True)
            home.joinpath("config.yaml").write_text(
                f"model:\n  default: {name}-model\n  provider: custom\n"
                f"  context_length: 2000\ndisplay:\n  language: {lang}\n")
            homes[name] = home
        return base, homes

    def _patches(self, base, homes):
        return (
            patch("gateway.run._hermes_home", base),
            patch.object(GatewayRunner, "_resolve_profile_home_for_source",
                         side_effect=lambda source: homes[source.profile]),
            patch("gateway.run._resolve_runtime_agent_kwargs", return_value=self._RUNTIME),
        )

    def test_profile_language_localizes_session_info(self, runner, tmp_path):
        """A profile on a non-English locale gets localized labels, not the English ones."""
        from types import SimpleNamespace
        base, homes = self._homes(tmp_path, [("planner", "ja")])
        runner.config = SimpleNamespace(multiplex_profiles=True)
        p1, p2, p3 = self._patches(base, homes)
        with p1, p2, p3:
            info = runner._reset_notice_session_info(self._source("planner"))
        assert "◆ モデル: `planner-model`" in info
        assert "◆ プロバイダー: custom" in info
        assert "◆ Model:" not in info
        assert "◆ Provider:" not in info

    def test_multiplex_profiles_do_not_share_one_language(self, runner, tmp_path):
        """Serving a Japanese profile first must not pin Japanese onto the French one."""
        from types import SimpleNamespace
        base, homes = self._homes(tmp_path, [("planner", "ja"), ("scribe", "fr")])
        runner.config = SimpleNamespace(multiplex_profiles=True)
        rendered = {}
        p1, p2, p3 = self._patches(base, homes)
        with p1, p2, p3:
            # Order is the whole point: the Japanese profile renders first, so a
            # process-global language cache already holds "ja" when French renders.
            for name in ("planner", "scribe"):
                rendered[name] = runner._reset_notice_locale_and_info(self._source(name))

        assert rendered["planner"][0] == "ja"
        assert rendered["scribe"][0] == "fr"
        assert "◆ モデル: `planner-model`" in rendered["planner"][1]
        assert "◆ Modèle : `scribe-model`" in rendered["scribe"][1]
        assert "モデル" not in rendered["scribe"][1]
        assert "Modèle" not in rendered["planner"][1]

    @pytest.mark.asyncio
    async def test_auto_reset_notice_follows_the_served_profile(self, runner, tmp_path):
        """The notice itself, not just the info block, must follow the serving profile.

        This is the delivery path the review flagged: the notice text used to be built
        outside the profile-scoped helper, so ``t()`` resolved it against the shared cache.
        """
        from types import SimpleNamespace
        from unittest.mock import AsyncMock
        from agent.i18n import t

        base, homes = self._homes(tmp_path, [("planner", "ja"), ("scribe", "fr")])
        runner.config = SimpleNamespace(multiplex_profiles=True)
        adapter = SimpleNamespace(send=AsyncMock())
        sent = {}
        p1, p2, p3 = self._patches(base, homes)
        with p1, p2, p3, \
                patch.object(GatewayRunner, "_adapter_for_source", return_value=adapter), \
                patch.object(GatewayRunner, "_thread_metadata_for_source", return_value=None):
            for name in ("planner", "scribe"):
                entry = SimpleNamespace(auto_reset_reason="suspended")
                await runner._hmwa_deliver_auto_reset_notice(entry, self._source(name), [])
                sent[name] = adapter.send.await_args.args[1]

        assert sent["planner"].startswith(t("gateway.auto_reset.notice", lang="ja"))
        assert sent["scribe"].startswith(t("gateway.auto_reset.notice", lang="fr"))
        assert sent["planner"] != sent["scribe"]
        assert "◆ モデル: `planner-model`" in sent["planner"]
        assert "◆ Modèle : `scribe-model`" in sent["scribe"]
