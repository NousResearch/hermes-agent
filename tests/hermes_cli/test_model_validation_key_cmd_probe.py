"""`/model` validation probes a `key_cmd` provider with its minted token, not the token source's repr.

A `key_cmd` provider's runtime credential is a callable ``CommandTokenSource``
(``hermes_cli/model_switch_providers._entry_credentials`` returns it as the "unminted credential"),
and the model pickers handle that shape explicitly: they branch on ``callable(api_key)`` and mint
with ``agent.command_token_source.materialize_probe_api_key`` before probing. The ``/model``
validation ladder had no such branch, so every probe below it built ``Authorization: Bearer
<CommandTokenSource object at 0x…>``; the endpoint answered 401 and the switch reported the
provider's listing as unreachable (#154004).

The credential is materialized once, at the ladder's request boundary, so all of its probe sites
(the generic ``probe_api_models`` probe, the Anthropic native fetcher, the live listing) get the
token instead of the object.
"""

import pytest

from agent.command_token_source import CommandTokenSource
from hermes_cli.models_validate import validate_requested_model

PROBE_PAYLOAD = {
    "models": ["model-b"],
    "probed_url": "https://proxy.example.test/v1/models",
    "resolved_base_url": "https://proxy.example.test/v1",
    "suggested_base_url": None,
    "used_fallback": False,
}


@pytest.fixture(autouse=True)
def hermes_home(tmp_path, monkeypatch):
    """A key_cmd helper runs for the profile whose provider is being minted: keep it off the real home."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.fixture
def key_cmd():
    """The runtime credential a `key_cmd` provider hands the model switch."""
    return CommandTokenSource("printf tok-abc", "myproxy")


@pytest.fixture
def probed_keys(monkeypatch):
    """The ``api_key`` each catalog probe was handed, in call order."""
    keys: list = []

    def fake_probe(api_key, base_url=None, **kwargs):
        keys.append(api_key)
        return dict(PROBE_PAYLOAD)

    monkeypatch.setattr("hermes_cli.models.probe_api_models", fake_probe)
    return keys


def test_custom_provider_probe_gets_the_minted_token(key_cmd, probed_keys):
    verdict = validate_requested_model(
        "model-b", "custom:myproxy", api_key=key_cmd,
        base_url="https://proxy.example.test/v1", api_mode="chat_completions",
    )
    assert probed_keys == ["tok-abc"]
    assert verdict["accepted"] is True


def test_anthropic_native_probe_gets_the_minted_token(key_cmd, monkeypatch):
    seen: list = []

    def fake_fetch(*, base_url=None, api_key=None):
        seen.append(api_key)
        return ["claude-opus-5"]

    monkeypatch.setattr("hermes_cli.models._fetch_anthropic_models", fake_fetch)
    verdict = validate_requested_model(
        "claude-opus-5", "anthropic", api_key=key_cmd, base_url="https://api.anthropic.com",
    )
    assert seen == ["tok-abc"]
    assert verdict["accepted"] is True


def test_a_static_string_key_is_probed_unchanged(probed_keys):
    validate_requested_model(
        "model-b", "custom:myproxy", api_key="  sk-abc  ",
        base_url="https://proxy.example.test/v1", api_mode="chat_completions",
    )
    assert probed_keys == ["sk-abc"]


def test_no_key_probes_without_a_credential(probed_keys):
    validate_requested_model(
        "model-b", "custom:myproxy", base_url="https://proxy.example.test/v1",
        api_mode="chat_completions",
    )
    assert probed_keys == [None]
