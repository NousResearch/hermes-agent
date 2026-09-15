"""A configured custom (OpenAI-compatible) endpoint is explicit provider intent.

Regression for #108383: ``resolve_provider("auto")`` recognised only registry providers from
``model.provider``, so the boot inventory (``free_tier_bootstrap``) read a llama.cpp / vLLM /
ollama install as "nothing configured" and ``setup.status`` reported ``provider_configured:
False`` — the dashboard's Ink chat parked every new session on "Setup Required" while
``hermes chat`` (which resolves the runtime directly) worked against the same config.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    (home / ".env").write_text("", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_GUEST_ONBOARDING", raising=False)
    for var in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_BASE_URL",
                "OPENROUTER_BASE_URL", "HERMES_INFERENCE_PROVIDER", "NOUS_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("agent.bedrock_adapter.has_aws_credentials", lambda: False)
    from hermes_cli import free_tier_bootstrap as fb
    fb.reset_for_tests()
    return home


@pytest.mark.parametrize(
    "model_block",
    [
        pytest.param(
            "model:\n  default: nvidia/Nemotron\n  provider: custom\n"
            "  base_url: http://127.0.0.1:8000/v1\n  api_key: dummy\n",
            id="provider-custom",
        ),
        pytest.param(
            "model:\n  default: qwen3\n  provider: vllm\n  base_url: http://127.0.0.1:8000/v1\n",
            id="local-server-alias",
        ),
        pytest.param(
            "model:\n  default: qwen3\n  base_url: http://localhost:8080/v1\n",
            id="loopback-base-url-only",
        ),
    ],
)
def test_configured_custom_endpoint_resolves_as_a_provider(isolated_home, model_block):
    (isolated_home / "config.yaml").write_text(model_block, encoding="utf-8")
    from hermes_cli.auth import resolve_provider
    from hermes_cli.free_tier_bootstrap import run_bootstrap

    assert resolve_provider("auto") == "custom"
    record = run_bootstrap(announce=False)
    assert record.provider_configured is True
    assert record.other_providers is True
    assert record.inference_provider == "custom"


def test_stale_remote_base_url_without_a_custom_pin_is_not_a_provider(isolated_home):
    """The URL rung follows the runtime's own trust rule: a non-loopback ``base_url`` left behind
    under another provider's pin is not custom intent (#14676), so a blank machine still reads
    as unconfigured."""
    (isolated_home / "config.yaml").write_text(
        "model:\n  default: some/model\n  provider: openrouter\n  base_url: https://api.z.ai/v1\n",
        encoding="utf-8",
    )
    from hermes_cli.auth import AuthError, resolve_provider
    from hermes_cli.free_tier_bootstrap import run_bootstrap

    with pytest.raises(AuthError):
        resolve_provider("auto")
    assert run_bootstrap(announce=False).provider_configured is False
