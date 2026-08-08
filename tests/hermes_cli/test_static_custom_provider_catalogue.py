"""Regression tests for the static custom-provider catalogue fix.

Problem: when a custom provider's ``/v1/models`` discovery fails,
``switch_model`` already soft-accepts models declared in config but carries
a misleading "could not reach this custom endpoint's model listing" warning
and reports ``recognized=False``.

Fix: treat an explicit non-empty ``models`` list/dict catalogue as valid
verification — suppress the warning and return ``recognized=True`` when the
selected model is in it.  Preserve the existing soft-accept-with-warning
behaviour for a bare ``model:`` scalar (no catalogue).

Three scenarios are pinned:

1. Named custom provider (``custom:<name>``) with a static ``models`` list —
   warning suppressed, ``recognized=True``.
2. Bare custom provider matched by ``base_url`` with a static ``models`` dict —
   warning suppressed, ``recognized=True``.
3. Bare ``model:`` scalar only (no ``models`` catalogue) — old behaviour
   preserved: warning present, ``recognized=False``.

Hermetic: the model-resolution chain is fully mocked (no network), mirroring
``tests/hermes_cli/test_model_switch_configured_provider_routing.py``.
"""

from unittest.mock import patch

from hermes_cli.model_switch import switch_model


# What validate_requested_model returns when /v1/models discovery FAILS for a
# custom endpoint — the exact "could not reach" warning we want to suppress
# when a static catalogue is present.
_DISCOVERY_FAILED = {
    "accepted": False,
    "persist": True,
    "recognized": False,
    "message": (
        "Note: could not reach this custom endpoint's model listing at "
        "`http://relay.test/v1/models`. Hermes will still save `qwen3.5-4b`, "
        "but the endpoint should expose `/models` for verification."
    ),
}


def _run_switch(
    *,
    raw_input,
    current_provider,
    user_providers=None,
    custom_providers=None,
    validation=_DISCOVERY_FAILED,
    current_model="old-model",
    current_base_url="",
):
    """Drive ``switch_model`` with the resolution chain mocked out.

    Every external lookup that would otherwise hit catalogs/network is patched,
    isolating the config-override + warning-suppression logic.
    """
    with patch("hermes_cli.model_switch.resolve_alias", return_value=None), \
         patch("hermes_cli.model_switch.list_provider_models", return_value=[]), \
         patch(
             "hermes_cli.model_switch.normalize_model_for_provider",
             side_effect=lambda model, provider: model,
         ), \
         patch(
             "hermes_cli.models.validate_requested_model",
             return_value=validation,
         ), \
         patch("hermes_cli.models.detect_provider_for_model", return_value=None), \
         patch("hermes_cli.model_switch.get_model_info", return_value=None), \
         patch("hermes_cli.model_switch.get_model_capabilities", return_value=None), \
         patch(
             "hermes_cli.runtime_provider.resolve_runtime_provider",
             return_value={
                 "api_key": "relay-key",
                "base_url": current_base_url or "http://relay.test/v1",
                "api_mode": "",
            },
         ):
        return switch_model(
            raw_input=raw_input,
            current_provider=current_provider,
            current_model=current_model,
            current_base_url=current_base_url,
            user_providers=user_providers or {},
            custom_providers=custom_providers or [],
        )


# ---------------------------------------------------------------------------
# 1. Named custom provider with a static models list — warning suppressed
# ---------------------------------------------------------------------------

def test_named_custom_provider_static_list_suppresses_warning():
    """A named custom provider (``custom:<name>``) whose ``models`` list
    explicitly contains the requested model is treated as verified: no
    warning, ``recognized`` True — even when live ``/v1/models`` discovery
    failed."""
    custom_providers = [
        {
            "name": "myrelay",
            "base_url": "http://relay.test/v1",
            "api_key": "relay-key",
            "models": ["qwen3.5-4b", "kimi-k2.5"],
        }
    ]
    result = _run_switch(
        raw_input="qwen3.5-4b",
        current_provider="custom:myrelay",
        current_model="kimi-k2.5",
        current_base_url="http://relay.test/v1",
        custom_providers=custom_providers,
    )
    assert result.success is True, result.error_message
    assert result.target_provider == "custom:myrelay"
    assert result.new_model == "qwen3.5-4b"
    # The misleading "could not reach" warning must be suppressed.
    assert result.warning_message == ""
    # Recognized: the static catalogue is verification enough.
    assert result.error_message == ""


# ---------------------------------------------------------------------------
# 2. Bare custom provider matched by base_url with a static models dict
# ---------------------------------------------------------------------------

def test_bare_custom_provider_matched_by_url_static_dict_suppresses_warning():
    """A bare ``provider: custom`` endpoint matched by ``base_url`` whose
    ``models`` dict explicitly contains the requested model is treated as
    verified: no warning — even when live ``/v1/models`` discovery failed."""
    custom_providers = [
        {
            "name": "relaytwo",
            "base_url": "http://relay-two.test/v1",
            "api_key": "relay-two-key",
            "models": {
                "qwen3.5-4b": {"context_length": 32768},
                "glm-5.1": {},
            },
        }
    ]
    result = _run_switch(
        raw_input="qwen3.5-4b",
        current_provider="custom",
        current_model="glm-5.1",
        current_base_url="http://relay-two.test/v1",
        custom_providers=custom_providers,
    )
    assert result.success is True, result.error_message
    assert result.new_model == "qwen3.5-4b"
    assert result.warning_message == ""
    assert result.error_message == ""


# ---------------------------------------------------------------------------
# 3. Bare model: scalar only — old soft-accept-with-warning preserved
# ---------------------------------------------------------------------------

def test_bare_model_scalar_preserves_warning():
    """A custom provider with only a bare ``model:`` scalar (no ``models``
    catalogue) must keep the old behaviour: soft-accept WITH the discovery
    warning so the user knows the endpoint could not be verified."""
    custom_providers = [
        {
            "name": "solo",
            "base_url": "http://solo.test/v1",
            "api_key": "solo-key",
            "model": "qwen3.5-4b",
        }
    ]
    result = _run_switch(
        raw_input="qwen3.5-4b",
        current_provider="custom:solo",
        current_model="old-model",
        current_base_url="http://solo.test/v1",
        custom_providers=custom_providers,
    )
    assert result.success is True, result.error_message
    assert result.new_model == "qwen3.5-4b"
    # Bare scalar is NOT a catalogue — warning must still be present.
    assert result.warning_message != ""
    assert "could not reach" in result.warning_message


# ---------------------------------------------------------------------------
# 4. user_providers (providers: dict) static list — also suppressed
# ---------------------------------------------------------------------------

def test_user_providers_static_list_suppresses_warning():
    """A ``providers:`` dict entry whose ``models`` list contains the
    requested model is treated as verified: no warning — even when live
    ``/v1/models`` discovery failed."""
    user_providers = {
        "my-relay": {
            "name": "My Relay",
            "base_url": "http://relay-three.test/v1",
            "api_key": "relay-three-key",
            "models": ["qwen3.5-4b", "kimi-k2.5"],
        }
    }
    result = _run_switch(
        raw_input="qwen3.5-4b",
        current_provider="my-relay",
        current_model="kimi-k2.5",
        current_base_url="http://relay-three.test/v1",
        user_providers=user_providers,
    )
    assert result.success is True, result.error_message
    assert result.target_provider == "my-relay"
    assert result.new_model == "qwen3.5-4b"
    assert result.warning_message == ""


# ---------------------------------------------------------------------------
# 5. Empty models list is NOT a catalogue — warning preserved
# ---------------------------------------------------------------------------

def test_empty_models_list_is_not_a_catalogue():
    """An empty ``models: []`` is not a catalogue — the old soft-accept-with-
    warning behaviour is preserved (defensive: should not normally match
    anyway, but if it does, don't claim verification)."""
    custom_providers = [
        {
            "name": "emptyrelay",
            "base_url": "http://empty.test/v1",
            "api_key": "empty-key",
            "model": "qwen3.5-4b",
            "models": [],
        }
    ]
    # Match is via the bare ``model`` scalar, so override fires but
    # catalogue_verified is False (models list is empty).
    result = _run_switch(
        raw_input="qwen3.5-4b",
        current_provider="custom:emptyrelay",
        current_model="old-model",
        current_base_url="http://empty.test/v1",
        custom_providers=custom_providers,
    )
    assert result.success is True, result.error_message
    # Matched via bare scalar -> warning preserved.
    assert result.warning_message != ""
    assert "could not reach" in result.warning_message
