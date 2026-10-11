"""/model must accept the Discord-rendered ``name:<provider>/<model>`` form on
every surface and switch BOTH provider and model.

Symptom: Discord's native /model shows its option as ``name:<provider>/<model>``.
Pasted into Telegram, the whole string became the model id under the CURRENT
provider (``<current>`` + ``name:<provider>/<model>``) and broke the session.

Pinned here:
  * the shared parser strips a leading ``name:``/``name=``/``model:``/``model=``;
  * ``/model name:X``, ``/model X``, ``/model X --provider P`` and ``/model P/X``
    all resolve to the same switch;
  * an explicit ``--provider`` wins over an inferred one;
  * aggregator ``vendor/model`` slugs stay intact;
  * ``unknown/model`` is refused instead of persisting a broken pair.

Hermetic: resolution chain mocked like test_model_switch_configured_provider_routing;
``provider/model`` detection runs the real ``_resolve_provider_prefix`` over the test providers.
"""

from unittest.mock import patch

import pytest

from hermes_cli.model_switch import parse_model_switch_args, switch_model
from hermes_cli.models import _resolve_provider_prefix

_ACCEPTED = {"accepted": True, "persist": True, "recognized": True, "message": None}
# Endpoint could not list the id: soft-accept with a note (the path that used
# to persist ``bogus/model`` silently under the current provider).
_SOFT = {"accepted": True, "persist": True, "recognized": False, "message": "Note: could not verify"}

_USER_PROVIDERS = {
    "my-apr": {"base_url": "http://apr.invalid/v1", "models": {"claude-opus-5": {}}},
    "my-bpx": {"base_url": "http://bpx.invalid/v1", "models": {"claude-fable-5-1": {}}},
}


def _switch(raw_input, *, explicit_provider="", current_provider="my-apr",
            validation=_ACCEPTED, user_providers=_USER_PROVIDERS):
    with patch("hermes_cli.model_switch.resolve_alias", return_value=None), \
         patch("hermes_cli.model_switch.list_provider_models", return_value=[]), \
         patch("hermes_cli.model_switch.normalize_model_for_provider", side_effect=lambda model, provider: model), \
         patch("hermes_cli.models_validate.validate_requested_model", return_value=validation), \
         patch("hermes_cli.models._configured_provider_ids", return_value=set(user_providers or {})), \
         patch("hermes_cli.models.detect_provider_for_model",
               side_effect=lambda name, _current: _resolve_provider_prefix(name)), \
         patch("hermes_cli.model_switch.get_model_info", return_value=None), \
         patch("hermes_cli.model_switch.get_model_capabilities", return_value=None), \
         patch(
             "hermes_cli.runtime_provider.resolve_runtime_provider",
             return_value={"api_key": "***", "base_url": "http://resolved/v1", "api_mode": ""},
         ):
        return switch_model(
            raw_input=raw_input,
            current_provider=current_provider,
            current_model="claude-opus-5",
            explicit_provider=explicit_provider,
            user_providers=user_providers,
            custom_providers=[],
        )


# --- parser: option-label stripping ----------------------------------------

@pytest.mark.parametrize("raw,target", [
    ("name:my-bpx/claude-fable-5-1", "my-bpx/claude-fable-5-1"),
    ("NAME=my-bpx/claude-fable-5-1", "my-bpx/claude-fable-5-1"),
    ("model:claude-fable-5-1", "claude-fable-5-1"),
    ("model=claude-fable-5-1", "claude-fable-5-1"),
    ("name: claude-fable-5-1", "claude-fable-5-1"),
    ("name:claude-fable-5-1 --provider my-bpx", "claude-fable-5-1"),
    ("claude-fable-5-1", "claude-fable-5-1"),
    # a colon that is not an option label is left for switch_model
    ("my-bpx:claude-fable-5-1", "my-bpx:claude-fable-5-1"),
    ("anthropic/claude-sonnet-4.5:extended", "anthropic/claude-sonnet-4.5:extended"),
])
def test_parser_strips_discord_option_label(raw, target):
    assert parse_model_switch_args(raw).target == target


def test_parser_keeps_flags_with_label():
    req = parse_model_switch_args("name:claude-fable-5-1 --provider my-bpx --global")
    assert req.target == "claude-fable-5-1"
    assert req.explicit_provider == "my-bpx"
    assert req.is_global


# --- all four forms converge -------------------------------------------------

@pytest.mark.parametrize("raw", [
    "name:my-bpx/claude-fable-5-1",      # Discord-rendered, pasted anywhere
    "my-bpx/claude-fable-5-1",           # provider/model
    "claude-fable-5-1 --provider my-bpx",
    "name:claude-fable-5-1 --provider my-bpx",
])
def test_every_form_switches_provider_and_model(raw):
    req = parse_model_switch_args(raw)
    result = _switch(req.target, explicit_provider=req.explicit_provider)
    assert result.success, result.error_message
    assert (result.target_provider, result.new_model) == ("my-bpx", "claude-fable-5-1")


def test_explicit_provider_wins_over_inferred_prefix():
    req = parse_model_switch_args("my-bpx/claude-fable-5-1 --provider my-apr")
    result = _switch(req.target, explicit_provider=req.explicit_provider)
    assert result.success, result.error_message
    assert result.target_provider == "my-apr"
    # the slash id is taken verbatim under the explicit provider
    assert result.new_model == "my-bpx/claude-fable-5-1"


def test_openrouter_vendor_slug_stays_intact():
    result = _switch("anthropic/claude-sonnet-4.5", current_provider="openrouter",
                     user_providers={})
    assert result.success, result.error_message
    assert result.target_provider == "openrouter"
    assert result.new_model == "anthropic/claude-sonnet-4.5"


# --- unknown provider prefix is refused --------------------------------------

def test_unknown_provider_prefix_is_refused():
    req = parse_model_switch_args("name:no-such-prov/claude-fable-5-1")
    result = _switch(req.target, validation=_SOFT)
    assert not result.success
    assert "Unknown provider 'no-such-prov'" in result.error_message
    assert "No model switch was made" in result.error_message


def test_unknown_explicit_provider_is_refused():
    result = _switch("claude-fable-5-1", explicit_provider="no-such-prov")
    assert not result.success
    assert "Unknown provider 'no-such-prov'" in result.error_message


def test_vendor_namespace_slug_is_not_refused_off_aggregator():
    # meta-llama is a model publisher, not a provider: never an unknown-provider error.
    result = _switch("meta-llama/llama-4-scout", validation=_SOFT)
    assert result.success, result.error_message


def test_slug_the_endpoint_recognises_is_not_refused():
    # HF-style ids on a private endpoint: validation recognised the full slug.
    result = _switch("someorg/custom-model", validation=_ACCEPTED)
    assert result.success, result.error_message
    assert result.new_model == "someorg/custom-model"


def test_slug_declared_on_current_provider_is_not_refused():
    providers = dict(_USER_PROVIDERS)
    providers["my-apr"] = {"base_url": "http://apr.invalid/v1",
                           "models": {"someorg/custom-model": {}}}
    result = _switch("someorg/custom-model", validation=_SOFT, user_providers=providers)
    assert result.success, result.error_message
    assert result.target_provider == "my-apr"


def test_custom_endpoint_hf_slug_is_not_refused():
    # A custom/local endpoint serves its own ids (``Qwen/Qwen3-8B`` on vLLM): never refused.
    result = _switch("someorg/custom-model", current_provider="custom", validation=_SOFT,
                     user_providers={})
    assert result.success, result.error_message
    assert result.new_model == "someorg/custom-model"
