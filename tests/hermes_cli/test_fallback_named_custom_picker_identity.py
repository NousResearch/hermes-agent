"""A named list-form custom provider keeps its identity in the model snapshot.

Regression for NousResearch/hermes-agent#149320: the custom-endpoint picker wrote a bare
``custom`` for a ``custom_providers:`` entry that has no ``provider_key``, so the name never
reached the fallback chain. Bare ``custom`` resolves its key from ``OPENAI_API_KEY`` or from a
base_url that matches the primary's; a chain entry pointing at a *different* configured
endpoint matches neither, so the request went out keyless (401).

The picker snapshot is deliberately stripped of ``base_url`` here: a snapshot may lack it, and
the provider identity alone has to be enough to resolve the endpoint and its key.
"""

from unittest.mock import patch

import pytest

import hermes_yaml as yaml
from hermes_cli.model_setup_flows_custom import _model_flow_named_custom


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    (home / ".env").write_text("", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(
        "model:\n"
        "  provider: custom\n"
        "  default: primary-model\n"
        "  base_url: https://primary.example/v1\n"
        "  api_key: sk-primary\n"
        "custom_providers:\n"
        "  - name: Named Endpoint\n"
        "    base_url: https://secondary.example/v1\n"
        "    api_key: sk-secondary\n"
        "    model: secondary-model\n"
        "    discover_models: false\n"
        "    models: [secondary-model]\n",
        encoding="utf-8",
    )
    return home


def test_named_list_provider_keeps_its_identity_without_snapshot_base_url(isolated_home):
    from agent.auxiliary_client import resolve_provider_client
    from hermes_cli.fallback_cmd import _extract_fallback_from_model_cfg

    provider_info = {
        "name": "Named Endpoint",
        "base_url": "https://secondary.example/v1",
        "api_key": "",
        "model": "secondary-model",
        "discover_models": False,
        "models": ["secondary-model"],
    }

    with patch(
        "hermes_cli.model_setup_flows_custom._pick_named_custom_model",
        return_value="secondary-model",
    ):
        _model_flow_named_custom({}, provider_info)

    picked = yaml.safe_load((isolated_home / "config.yaml").read_text(encoding="utf-8"))["model"]
    assert picked["provider"] == "custom:named-endpoint"

    # A picker snapshot need not carry base_url; the identity alone must be enough.
    snapshot = {k: v for k, v in picked.items() if k != "base_url"}
    fallback = _extract_fallback_from_model_cfg(snapshot)
    assert fallback is not None
    assert fallback["provider"] == "custom:named-endpoint"
    assert "base_url" not in fallback

    client, _ = resolve_provider_client(
        fallback["provider"],
        model=fallback["model"],
        raw_codex=True,
        explicit_base_url=None,
        explicit_api_key=None,
    )
    assert str(client.base_url).rstrip("/") == "https://secondary.example/v1"
    assert str(client.api_key) == "sk-secondary"
