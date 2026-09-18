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


def test_auto_provider_with_loopback_base_url_resolves_without_recursing(isolated_home, monkeypatch):
    """A fresh setup keeps ``provider: auto`` until the picker stores its choice (#110926)."""
    (isolated_home / "config.yaml").write_text(
        "model:\n  provider: auto\n  base_url: http://127.0.0.1:8000/v1\n",
        encoding="utf-8",
    )
    from hermes_cli import runtime_provider
    from hermes_cli.auth import resolve_provider

    def unexpected_provider_resolution(_name):
        raise AssertionError("the bare custom trust check must not resolve model.provider=auto")

    monkeypatch.setattr(runtime_provider, "_resolves_to_custom", unexpected_provider_resolution)

    assert resolve_provider("auto") == "custom"


def test_a_provider_configured_after_boot_flips_the_stale_setup_record(isolated_home, monkeypatch):
    """A serve process whose boot inventory found nothing keeps that record for its lifetime;
    ``setup.status`` reads it (``wait_for_record``), so the dashboard chat stayed on "Setup
    Required" after the user configured a provider — from the Models page, a picker key, or
    ``hermes setup`` in another process. A ``False`` record is reconciled with the config files
    on read: it flips (+ one ``setup.ready``) once something carries inference, and a blank
    machine stays ``False`` with no broadcast."""
    from hermes_cli import free_tier_bootstrap as fb

    broadcasts = []
    monkeypatch.setattr(fb, "_broadcast", broadcasts.append)
    boot = fb.run_bootstrap(announce=False)
    assert boot.provider_configured is False
    assert fb.wait_for_record(timeout=0) is boot and broadcasts == [], "blank machine: nothing to reconcile"

    # The record is the launch profile's: a write scoped to another profile (the dashboard's
    # ``?profile=b``) must not let THAT profile's provider open the launch gate.
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    profile_b = isolated_home / "profiles" / "b"
    profile_b.mkdir(parents=True)
    (profile_b / "config.yaml").write_text("model:\n  default: qwen3\n  provider: custom\n  base_url: http://127.0.0.1:8000/v1\n  api_key: dummy\n", encoding="utf-8")
    token = set_hermes_home_override(str(profile_b))
    try:
        assert fb.reconcile_record() is boot and broadcasts == [], "another profile's provider is not ours"
    finally:
        reset_hermes_home_override(token)

    (isolated_home / "config.yaml").write_text(
        "model:\n  default: qwen3\n  provider: custom\n  base_url: http://127.0.0.1:8000/v1\n  api_key: dummy\n",
        encoding="utf-8",
    )
    fresh = fb.wait_for_record(timeout=0)
    assert fresh.provider_configured is True and fresh.other_providers is True
    assert fresh.inference_provider == "custom"
    assert fresh.has_identity is boot.has_identity and fresh.failure == boot.failure, "the mint verdict is kept"
    assert broadcasts == [fresh]
    assert fb.wait_for_record(timeout=0) is fresh and broadcasts == [fresh], "settled: no re-inventory, no re-announce"
