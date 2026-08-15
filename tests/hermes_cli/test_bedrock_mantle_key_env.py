"""Bedrock API-key setup must produce a config that actually authenticates.

The wizard used to stash the Bedrock bearer token in ``OPENAI_API_KEY`` and set
a bare ``provider: custom``. Since the cross-provider credential gate landed
(#28660) that variable is only honoured for ``openai.com`` hosts, so the token
was silently dropped and every request went out as ``no-key-required``.

These tests lock the seam the bug lived in: what the wizard writes, and what the
runtime resolver then makes of it.
"""

import os

import yaml

import hermes_cli.runtime_provider as rp
from hermes_cli.model_setup_flows import _model_flow_bedrock_api_key


REGION = "us-east-1"
TOKEN = "test-bedrock-bearer-token"


def _run_wizard(monkeypatch, selected="openai.gpt-5.6-terra"):
    """Drive the real setup flow non-interactively and return the saved config."""
    import hermes_cli.auth as auth_mod

    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", TOKEN)
    monkeypatch.setattr(
        auth_mod, "_prompt_model_selection", lambda *a, **k: selected
    )
    monkeypatch.setattr(auth_mod, "_save_model_choice", lambda *a, **k: None)
    monkeypatch.setattr(auth_mod, "deactivate_provider", lambda *a, **k: None)

    _model_flow_bedrock_api_key({}, REGION)

    from hermes_constants import get_hermes_home

    return get_hermes_home(), yaml.safe_load(
        (get_hermes_home() / "config.yaml").read_text(encoding="utf-8")
    )


def test_wizard_writes_named_provider_carrying_the_key_env(monkeypatch):
    home, cfg = _run_wizard(monkeypatch)

    # The credential must travel via a named provider entry: that is the only
    # resolution branch that reads key_env.
    entry = cfg["providers"]["bedrock-mantle"]
    assert entry["key_env"] == "AWS_BEARER_TOKEN_BEDROCK"
    assert entry["base_url"].startswith(f"https://bedrock-mantle.{REGION}.api.aws")
    assert cfg["model"]["provider"] == "custom:bedrock-mantle"

    # A bare ``custom`` provider plus model.base_url is the shape that could not
    # carry the token; make sure we did not leave it behind.
    assert "base_url" not in cfg["model"]


def test_wizard_does_not_park_the_bedrock_token_in_openai_api_key(monkeypatch):
    home, _cfg = _run_wizard(monkeypatch)

    env_file = home / ".env"
    written = env_file.read_text(encoding="utf-8") if env_file.exists() else ""
    names = {
        line.split("=", 1)[0].strip()
        for line in written.splitlines()
        if "=" in line and not line.lstrip().startswith("#")
    }

    # Writing a Bedrock credential into another vendor's variable is what the
    # #28660 gate exists to prevent. (The token itself already came from the
    # environment here, so the flow has no reason to re-write it.)
    assert "OPENAI_API_KEY" not in names
    assert "OPENAI_BASE_URL" not in names


def test_saved_config_resolves_the_bearer_token_not_a_placeholder(monkeypatch):
    """The regression contract: the token must reach the resolved runtime."""
    _home, cfg = _run_wizard(monkeypatch)

    # Nothing else may supply a credential for this host.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", TOKEN)
    monkeypatch.setattr(rp, "load_config", lambda: cfg)

    resolved = rp.resolve_runtime_provider(
        requested=cfg["model"]["provider"],
    )

    assert resolved["api_key"] == TOKEN
    assert resolved["api_key"] != "no-key-required"
    assert "bedrock-mantle" in resolved["base_url"]


def test_resolution_fails_closed_when_the_token_is_absent(monkeypatch):
    """No token in the environment must not silently resolve to a placeholder key."""
    _home, cfg = _run_wizard(monkeypatch)

    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(rp, "load_config", lambda: cfg)

    resolved = rp.resolve_runtime_provider(requested=cfg["model"]["provider"])

    assert resolved["api_key"] != TOKEN
