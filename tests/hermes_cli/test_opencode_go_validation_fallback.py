"""Tests for the static-catalog fallback in validate_requested_model.

OpenCode Go and OpenCode Zen publish an OpenAI-compatible API at paths that do
NOT expose ``/models`` (the path returns the marketing site's HTML 404).  This
caused ``validate_requested_model`` to return ``accepted=False`` for every
model on those providers, which in turn made ``switch_model()`` fail and the
gateway's ``/model <name> --provider opencode-go`` command never write to
``_session_model_overrides``.

These tests cover the catalog-fallback path: when ``fetch_api_models`` returns
``None``, the validator must consult ``provider_model_ids()`` for the provider
(populated from ``_PROVIDER_MODELS``) rather than rejecting outright.
"""

from unittest.mock import patch

from hermes_cli.models import validate_requested_model


_UNREACHABLE_PROBE = {
    "models": None,
    "probed_url": "https://opencode.ai/zen/go/v1/models",
    "resolved_base_url": "https://opencode.ai/zen/go/v1",
    "suggested_base_url": None,
    "used_fallback": False,
}


def _patched(func):
    """Decorator: force fetch_api_models / probe_api_models to simulate an
    unreachable /models endpoint, proving the catalog path is used."""
    def wrapper(*args, **kwargs):
        with patch("hermes_cli.models.fetch_api_models", return_value=None), \
             patch("hermes_cli.models.probe_api_models", return_value=_UNREACHABLE_PROBE):
            return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    return wrapper


# ---------------------------------------------------------------------------
# opencode-go: curated catalog in _PROVIDER_MODELS
# ---------------------------------------------------------------------------


@_patched
def test_opencode_go_known_model_accepted():
    """A model present in the opencode-go curated catalog must be accepted
    even when /models is unreachable."""
    result = validate_requested_model("kimi-k2.6", "opencode-go")
    assert result["accepted"] is True
    assert result["persist"] is True
    assert result["recognized"] is True
    assert result["message"] is None


@_patched
def test_opencode_go_known_model_case_insensitive():
    """Catalog lookup is case-insensitive."""
    result = validate_requested_model("KIMI-K2.6", "opencode-go")
    assert result["accepted"] is True
    assert result["recognized"] is True


@_patched
def test_opencode_go_typo_auto_corrected():
    """A close typo (>= 0.9 similarity) is auto-corrected to the catalog
    entry."""
    # 'kimi-k2.55' vs 'kimi-k2.5' ratio ≈ 0.95 — within the 0.9 cutoff.
    result = validate_requested_model("kimi-k2.55", "opencode-go")
    assert result["accepted"] is True
    assert result["recognized"] is True
    assert result.get("corrected_model") == "kimi-k2.5"


@_patched
def test_opencode_go_unknown_model_accepted_with_suggestion():
    """An unknown model that has a medium-similarity match (>= 0.5 but < 0.9)
    is accepted with recognized=False and a 'similar models' hint.  The key
    invariant: the gateway MUST be able to persist this override, so
    accepted/persist must both be True."""
    # 'kimi-k3-preview' vs 'kimi-k2.6' — similar enough to suggest, not to auto-correct.
    result = validate_requested_model("kimi-k3-preview", "opencode-go")
    assert result["accepted"] is True
    assert result["persist"] is True
    assert result["recognized"] is False
    assert "kimi-k3-preview" in result["message"]
    assert "curated catalog" in result["message"]


@_patched
def test_opencode_go_totally_unknown_model_still_accepted():
    """A model with zero similarity to the catalog is still accepted (no
    suggestion line) so the user can try a model that hasn't made it into the
    curated list yet."""
    result = validate_requested_model("some-brand-new-model", "opencode-go")
    assert result["accepted"] is True
    assert result["persist"] is True
    assert result["recognized"] is False
    # No suggestion text (no close matches)
    assert "Similar models" not in result["message"]
    assert "opencode" in result["message"].lower() or "opencode go" in result["message"].lower()


# ---------------------------------------------------------------------------
# opencode-zen: same pattern as opencode-go
# ---------------------------------------------------------------------------


@_patched
def test_opencode_zen_known_model_accepted():
    """opencode-zen also uses _PROVIDER_MODELS; kimi-k2 is in its catalog."""
    result = validate_requested_model("kimi-k2", "opencode-zen")
    assert result["accepted"] is True
    assert result["recognized"] is True


# ---------------------------------------------------------------------------
# Unknown provider with no catalog: soft-accept (honors the comment's intent)
# ---------------------------------------------------------------------------


@_patched
def test_provider_without_catalog_accepts_with_warning():
    """When a provider has no entry in _PROVIDER_MODELS and /models is
    unreachable, accept the model with a 'Note:' warning rather than reject.
    This matches the in-code comment: 'Accept and persist, but warn so typos
    don't silently break things.'"""
    # Use a made-up provider name that won't resolve to any catalog.
    result = validate_requested_model("some-model", "provider-that-does-not-exist")
    assert result["accepted"] is True
    assert result["persist"] is True
    assert result["recognized"] is False
    assert "Note:" in result["message"]
