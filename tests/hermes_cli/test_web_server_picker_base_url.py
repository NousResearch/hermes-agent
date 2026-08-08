"""Dashboard model picker (``POST /api/model/set``) × ``custom_providers``.

The picker sends provider + model only — no ``base_url``
(``apps/desktop/src/hermes.ts`` ``setGlobalModel``) — so
``_apply_main_model_assignment`` alone decides what happens to
``model.base_url`` on a provider switch.

It used to blank the URL on every provider change, on the assumption that the
resolver would supply "the new provider's own default". That holds for built-ins
(their endpoint lives in the resolver registry) but not for providers the user
declared themselves: for those the user-supplied URL *is* the default. A bare
``custom`` target has no registry entry and no name to look up, so a blanked
``model.base_url`` left the resolver falling through to OpenRouter's default host
with no key, and the next agent call returned
``HTTP 401: Missing Authentication header``.

These tests pin the contract:
  * switching to a user-declared provider writes THAT provider's endpoint
    (canonical ``custom:<name>`` slug, raw display name, and bare ``custom``);
  * switching to a built-in still clears ``base_url``;
  * a same-provider re-pick still preserves it;
  * and the resolver really does route to the user's gateway afterwards.
"""

import pytest
import yaml

web_server = pytest.importorskip(
    "hermes_cli.web_server", reason="fastapi/starlette not installed"
)
_apply_main_model_assignment = web_server._apply_main_model_assignment

LITELLM_URL = "http://192.168.1.10:4000"
VLLM_URL = "http://192.168.1.11:8000"

CONFIG = f"""\
custom_providers:
- name: my-litellm
  base_url: {LITELLM_URL}
  api_key: sk-test-litellm
  api_format: openai
- name: my-vllm
  base_url: {VLLM_URL}
  api_key: sk-test-vllm
  api_format: openai

model:
  default: gpt-4o-mini
  provider: {{provider}}
  base_url: {{base_url}}
  api_key: sk-test-endpoint
"""


@pytest.fixture
def config_home(tmp_path, monkeypatch):
    """HERMES_HOME holding a two-custom-provider config; returns a writer."""
    home = tmp_path / "picker_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    def _write(provider: str, base_url: str) -> dict:
        (home / "config.yaml").write_text(
            CONFIG.format(provider=provider, base_url=base_url or "''")
        )
        return yaml.safe_load((home / "config.yaml").read_text())

    _write("anthropic", "https://api.anthropic.com")
    return _write


def _pick(cfg: dict, provider: str, model: str = "model-b") -> dict:
    """Apply a picker assignment the way ``POST /api/model/set`` does."""
    return _apply_main_model_assignment(cfg.get("model") or {}, provider, model)


class TestSwitchingToUserDeclaredProvider:
    """The target's OWN endpoint is written, whatever spelling arrives."""

    @pytest.mark.parametrize(
        "provider",
        [
            "custom:my-litellm",  # canonical slug — what the picker rows use
            "my-litellm",  # raw display name
            "custom",  # bare: self-heals to the first valid entry
        ],
    )
    def test_switch_writes_target_endpoint(self, config_home, provider):
        cfg = config_home("anthropic", "https://api.anthropic.com")

        model_cfg = _pick(cfg, provider)

        assert model_cfg["provider"] == provider
        assert model_cfg["default"] == "model-b"
        assert model_cfg.get("base_url") == LITELLM_URL, (
            f"picker left base_url={model_cfg.get('base_url')!r} for {provider!r}; "
            "an empty URL makes the resolver fall through to OpenRouter and the "
            "next agent call 401s"
        )

    def test_switch_between_custom_providers_uses_the_new_one(self, config_home):
        """custom-A → custom-B must not leave A's host behind.

        Preserving the old URL would point B's model at A's gateway — the
        opposite failure from clearing it.
        """
        cfg = config_home("custom:my-litellm", LITELLM_URL)

        model_cfg = _pick(cfg, "custom:my-vllm")

        assert model_cfg.get("base_url") == VLLM_URL

    def test_explicit_base_url_still_wins(self, config_home):
        cfg = config_home("custom:my-litellm", LITELLM_URL)

        model_cfg = _apply_main_model_assignment(
            cfg["model"], "custom:my-litellm", "model-b", "http://127.0.0.1:9999"
        )

        assert model_cfg.get("base_url") == "http://127.0.0.1:9999"


class TestBuiltInAndSameProviderBehaviourUnchanged:
    """Pre-existing rules this fix must not disturb."""

    def test_switch_to_built_in_clears_stale_base_url(self, config_home):
        cfg = config_home("custom:my-litellm", LITELLM_URL)

        model_cfg = _pick(cfg, "anthropic", "claude-haiku-4-5")

        assert model_cfg["provider"] == "anthropic"
        assert not model_cfg.get("base_url"), (
            "a built-in provider carries its endpoint in the resolver registry; "
            "a pinned URL from the previous provider must not survive"
        )

    def test_same_provider_repick_preserves_base_url(self, config_home):
        cfg = config_home("custom:my-litellm", LITELLM_URL)

        model_cfg = _pick(cfg, "custom:my-litellm")

        assert model_cfg.get("base_url") == LITELLM_URL

    def test_empty_base_url_stays_empty_for_built_ins(self, config_home):
        cfg = config_home("openai", "")

        model_cfg = _pick(cfg, "anthropic", "claude-haiku-4-5")

        assert not model_cfg.get("base_url")


def test_resolution_after_bare_custom_switch_reaches_the_users_gateway(config_home):
    """End-to-end: the picker write must leave routing intact.

    This is the assertion that actually encodes the bug report — before the fix
    the resolved endpoint was ``https://openrouter.ai/api/v1`` with no key.
    """
    from hermes_cli.config import get_config_path
    from hermes_cli.runtime_provider import resolve_runtime_provider

    cfg = config_home("anthropic", "https://api.anthropic.com")
    cfg["model"] = _pick(cfg, "custom")
    get_config_path().write_text(yaml.safe_dump(cfg, sort_keys=False))

    runtime = resolve_runtime_provider(
        requested=cfg["model"]["provider"], target_model="model-b"
    )

    assert runtime.get("base_url") == LITELLM_URL
    assert "openrouter.ai" not in (runtime.get("base_url") or "")
