"""``/model`` with no ``model.provider`` names the provider the resolver actually routes through (#136801).

``_ModelSwitchContext.read_config`` leaves ``current_provider`` on its ``openrouter`` placeholder when
config.yaml has no ``model.provider``. The listing used to show that placeholder, so a dry API-key
provider billing every turn ("Insufficient Balance") hid behind a route the gateway was not using.
Real config, real env key, real ``resolve_provider``: nothing patched but the home and the env.
"""

import asyncio
import os

import pytest


def _listing_first_line(home, config: str) -> str:
    from gateway.slash_commands_model import GatewayModelCommandsMixin, _ModelSwitchContext

    (home / "config.yaml").write_text(config, encoding="utf-8")

    class Host(GatewayModelCommandsMixin):
        def _delivery_adapter_for(self, source):
            return None  # no picker adapter: /model falls through to the text listing

    ctx = _ModelSwitchContext(session_key="s", source=None, config_path=home / "config.yaml", persist_global=False)
    ctx.read_config()
    reply = asyncio.run(Host()._model_listing_reply(None, ctx, None))
    return (reply or "").splitlines()[0]


@pytest.fixture
def deepseek_only_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for key in [k for k in os.environ if k.endswith("_API_KEY")]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-not-real")
    return home


def test_unset_provider_lists_the_route_the_resolver_picks(deepseek_only_home):
    line = _listing_first_line(deepseek_only_home, "model:\n  default: deepseek-v4-flash\n")
    assert line.lower() == "current: `deepseek-v4-flash` on deepseek", line


def test_configured_provider_is_listed_as_configured(deepseek_only_home):
    line = _listing_first_line(deepseek_only_home, "model:\n  provider: anthropic\n  default: claude-x\n")
    assert line.lower() == "current: `claude-x` on anthropic", line
