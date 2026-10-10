"""Regression tests for #130100: ``providers:`` entries double-counted in
``_configured_provider_matches``.

Repro: with any ``providers:`` entry and the session on another provider, switching to a bare
model name (``/model glm-5.3-flash``) failed with "declared by multiple configured providers".
``get_compatible_custom_providers()`` projects every ``providers.<slug>`` row into the
custom-provider view (``custom:<name>``, stamped ``provider_key=<slug>``); collecting BOTH the
``user_providers`` slugs and the projection made one declaration match twice (``zhipu-coding``
+ ``custom:zhipu-coding``), tripping the multiple-guard. The error vanished once the session was
already on the target (short-circuit), masking the bug.

The fix drops pure compat projections (``provider_key`` already a ``user_providers`` slug, same
endpoint/credential, no extra models) in ``_configured_provider_matches`` while preserving
genuine multi-source conflicts across distinct providers.

Hermetic: the model-resolution chain is fully mocked (no network), mirroring
``tests/hermes_cli/test_model_switch_configured_provider_routing.py``.
"""

from unittest.mock import patch

from hermes_cli.model_switch import (
    _configured_provider_matches,
    _duplicates_configured_row,
    switch_model,
)

_ACCEPTED = {"accepted": True, "persist": True, "recognized": True, "message": None}


def _run_switch(
    *,
    raw_input,
    current_provider,
    user_providers=None,
    custom_providers=None,
    current_model="old-model",
    current_base_url="",
):
    """Drive ``switch_model`` with the resolution chain mocked out."""
    with patch("hermes_cli.model_switch.resolve_alias", return_value=None), \
         patch("hermes_cli.model_switch.list_provider_models", return_value=[]), \
         patch("hermes_cli.model_switch.normalize_model_for_provider", side_effect=lambda model, provider: model), \
         patch("hermes_cli.models_validate.validate_requested_model", return_value=_ACCEPTED), \
         patch("hermes_cli.models.detect_provider_for_model", return_value=None), \
         patch("hermes_cli.model_switch.get_model_info", return_value=None), \
         patch("hermes_cli.model_switch.get_model_capabilities", return_value=None), \
         patch(
             "hermes_cli.runtime_provider.resolve_runtime_provider",
             return_value={
                 "api_key": "***",
                 "base_url": current_base_url or "http://resolved/v1",
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


def _compat(user_providers, extra_custom=None):
    from hermes_cli.config import get_compatible_custom_providers

    cfg = {"providers": user_providers}
    if extra_custom is not None:
        cfg["custom_providers"] = extra_custom
    return get_compatible_custom_providers(cfg)


def test_bare_name_single_provider_entry_matches_once():
    """One ``providers.<slug>`` declaration must match once, not as ``slug`` +
    ``custom:<slug>`` (#130100 root cause)."""
    user_providers = {
        "zhipu-coding": {
            "base_url": "https://open.bigmodel.cn/api/coding/paas/v4",
            "api_key": "k",
            "models": ["glm-5.3-flash"],
        }
    }
    matches = _configured_provider_matches(
        "glm-5.3-flash", user_providers, _compat(user_providers))
    assert matches == {"zhipu-coding": "glm-5.3-flash"}


def test_bare_name_switch_from_other_provider_succeeds():
    """End-to-end #130100 repro: session on another provider, bare ``/model <name>`` routes to
    the declaring provider instead of failing as 'multiple configured providers'."""
    user_providers = {
        "zhipu-coding": {
            "base_url": "https://open.bigmodel.cn/api/coding/paas/v4",
            "api_key": "k",
            "models": ["glm-5.3-flash"],
        }
    }
    result = _run_switch(
        raw_input="glm-5.3-flash",
        current_provider="openrouter",
        user_providers=user_providers,
        custom_providers=_compat(user_providers),
    )
    assert result.success is True, result.error_message
    assert result.target_provider == "zhipu-coding"
    assert result.new_model == "glm-5.3-flash"


def test_projection_with_distinct_display_name_still_collapses():
    """A ``providers`` row whose ``name`` differs from its slug (``Zhipu Coding`` vs
    ``zhipu-coding``) is still one declaration."""
    user_providers = {
        "zhipu-coding": {
            "name": "Zhipu Coding",
            "base_url": "https://x.example/v1",
            "api_key": "k",
            "models": ["glm-5.3-flash"],
        }
    }
    matches = _configured_provider_matches(
        "glm-5.3-flash", user_providers, _compat(user_providers))
    assert list(matches) == ["zhipu-coding"]
    result = _run_switch(
        raw_input="glm-5.3-flash",
        current_provider="openrouter",
        user_providers=user_providers,
        custom_providers=_compat(user_providers),
    )
    assert result.success is True, result.error_message
    assert result.target_provider == "zhipu-coding"


def test_genuine_conflict_across_distinct_providers_preserved():
    """Two DISTINCT ``providers`` rows declaring the same model stay ambiguous — the
    multiple-guard must still fire."""
    user_providers = {
        "a": {"base_url": "https://a.example/v1", "api_key": "k1", "models": ["shared-model"]},
        "b": {"base_url": "https://b.example/v1", "api_key": "k2", "models": ["shared-model"]},
    }
    matches = _configured_provider_matches(
        "shared-model", user_providers, _compat(user_providers))
    assert sorted(matches) == ["a", "b"]
    result = _run_switch(
        raw_input="shared-model",
        current_provider="openrouter",
        user_providers=user_providers,
        custom_providers=_compat(user_providers),
    )
    assert result.success is False
    assert "multiple configured providers" in (result.error_message or "")


def test_provider_key_aimed_elsewhere_stays_distinct():
    """A raw-list entry stamping ``provider_key`` of a ``providers`` slug but pointing at a
    different endpoint/credential is NOT that row's projection — ambiguity preserved."""
    user_providers = {
        "relay": {
            "name": "relay", "api": "https://relay.example/v1",
            "key_env": "RELAY_KEY", "default_model": "claude-opus-4-7",
        }
    }
    raw = [{"name": "relay", "provider_key": "relay", "base_url": "https://backup.example/v1",
            "key_env": "BACKUP_KEY", "model": "claude-opus-4-7"}]
    matches = _configured_provider_matches("claude-opus-4-7", user_providers, raw)
    assert sorted(matches) == ["custom:relay", "relay"]
    result = _run_switch(
        raw_input="claude-opus-4-7",
        current_provider="openrouter",
        user_providers=user_providers,
        custom_providers=raw,
    )
    assert result.success is False
    assert "multiple configured providers" in (result.error_message or "")


def test_case_variant_sibling_rows_resolve_exact_first():
    """Case-variant sibling rows (``a`` vs ``A``) sharing an endpoint must not collapse
    last-wins: ``provider_key='a'`` resolves to row ``a``, not ``A``."""
    from hermes_cli.model_switch import _configured_provider_identity

    cfg_a = {"base_url": "https://x.example/v1", "api_key": "k"}
    cfg_upper = {"base_url": "https://x.example/v1", "api_key": "k"}
    rows = {
        "a": _configured_provider_identity("a", cfg_a),
        "A": _configured_provider_identity("A", cfg_upper),
    }
    # Sanity: siblings share endpoint/credential so the fast-path guard passes for both.
    assert rows["a"][1:3] == rows["A"][1:3]

    entry_a = {
        "name": "a", "base_url": "https://x.example/v1", "api_key": "k",
        "provider_key": "a",
    }
    entry_upper = {
        "name": "A", "base_url": "https://x.example/v1", "api_key": "k",
        "provider_key": "A",
    }
    assert _duplicates_configured_row("custom:a", entry_a, rows) == "a"
    assert _duplicates_configured_row("custom:A", entry_upper, rows) == "A"

    # Fallback still folds when no exact row exists (different-case key, same endpoint).
    assert _duplicates_configured_row("custom:A", entry_upper, {"a": rows["a"]}) == "a"
