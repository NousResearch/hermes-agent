"""Regression: auxiliary runtime must never pair a model name with a
foreign endpoint.

Root cause recap
----------------
A cross-provider fallback rewrites the live runtime's ``provider`` +
``base_url`` but the model name reaching the auxiliary layer could still be the
PRIMARY model, from either of two independent sources:

1. ``_normalize_main_runtime`` merges fields individually and skips empty ones,
   so a snapshot could carry a new endpoint with a stale ``model``.
2. ``_resolve_auto`` backfilled a missing model from config.yaml's
   ``model.default`` — valid only on the *configured* provider — without
   checking that the live runtime still IS that provider.

Either path produced a request like::

    base_url=https://integrate.api.nvidia.com/v1/  model=claude-opus-5
    → 400 "The supported API model names are deepseek-v4-pro or
           deepseek-v4-flash, but you passed claude-opus-5."

Why it mattered beyond one failed call: compression is the main consumer and a
failed summary is SILENT. It logs, then inserts a degraded placeholder marker
instead of a real handoff, so the session never sheds tokens and the
compression threshold re-trips on the very next turn.

These tests assert the invariant (a model is only ever sent to the endpoint it
was resolved for), not the current model catalog, so they stay valid as
providers change.
"""

from agent.auxiliary_client import _normalize_main_runtime

NVIDIA_BASE = "https://integrate.api.nvidia.com/v1/"
PRIMARY_BASE = "https://aicodelink.top/v1"


def test_stale_model_dropped_when_endpoint_was_swapped():
    """The exact 400 shape: fallback endpoint + model from the old endpoint."""
    runtime = _normalize_main_runtime({
        "model": "claude-opus-5",
        "provider": "nvidia",
        "base_url": NVIDIA_BASE,
        "api_key": "k",
        "api_mode": "chat_completions",
        "model_endpoint": ("aicodelink-claude", PRIMARY_BASE),
    })
    # The orphaned model must be gone: downstream treats a missing model as
    # "use this endpoint's default", which is always valid on that endpoint.
    assert "model" not in runtime
    # ...while the endpoint and its credentials survive untouched.
    assert runtime["base_url"] == NVIDIA_BASE
    assert runtime["provider"] == "nvidia"
    assert runtime["api_key"] == "k"


def test_coherent_runtime_keeps_its_model():
    """A model matching its own endpoint must never be stripped."""
    runtime = _normalize_main_runtime({
        "model": "claude-opus-5",
        "provider": "aicodelink-claude",
        "base_url": PRIMARY_BASE,
        "api_key": "k",
        "model_endpoint": ("aicodelink-claude", PRIMARY_BASE),
    })
    assert runtime["model"] == "claude-opus-5"


def test_trailing_slash_and_case_are_not_a_mismatch():
    """Endpoint identity is compared normalized — no false positives."""
    runtime = _normalize_main_runtime({
        "model": "claude-opus-5",
        "provider": "AICodeLink-Claude",
        "base_url": PRIMARY_BASE + "/",
        "model_endpoint": ("aicodelink-claude", PRIMARY_BASE),
    })
    assert runtime["model"] == "claude-opus-5"


def test_snapshot_without_provenance_is_left_alone():
    """Absence of evidence is not evidence of mismatch (back-compat).

    Legacy callers that don't stamp ``model_endpoint`` must keep working:
    stripping models from every unstamped snapshot would break correct
    single-endpoint routing everywhere.
    """
    runtime = _normalize_main_runtime({
        "model": "claude-opus-5",
        "provider": "nvidia",
        "base_url": NVIDIA_BASE,
    })
    assert runtime["model"] == "claude-opus-5"


def test_empty_provenance_is_not_treated_as_mismatch():
    """A producer that didn't know its endpoint yet must not trigger a strip."""
    runtime = _normalize_main_runtime({
        "model": "some-model",
        "provider": "p",
        "base_url": "https://example.invalid/v1",
        "model_endpoint": ("", ""),
    })
    assert runtime["model"] == "some-model"


def test_provenance_field_never_leaks_downstream():
    """``model_endpoint`` is internal: it must not reach provider kwargs.

    It is also deliberately kept out of _MAIN_RUNTIME_CONTEXT_FIELDS because
    those fields feed the auxiliary client cache-key discriminator.
    """
    for snapshot in (
        {
            "model": "claude-opus-5",
            "provider": "nvidia",
            "base_url": NVIDIA_BASE,
            "model_endpoint": ("aicodelink-claude", PRIMARY_BASE),
        },
        {
            "model": "claude-opus-5",
            "provider": "aicodelink-claude",
            "base_url": PRIMARY_BASE,
            "model_endpoint": ("aicodelink-claude", PRIMARY_BASE),
        },
    ):
        assert "model_endpoint" not in _normalize_main_runtime(snapshot)


def test_compressor_stamps_endpoint_provenance():
    """The guard is only meaningful if the producer actually stamps it.

    Without this the field is always absent, every snapshot takes the
    back-compat path, and the guard silently checks nothing.
    """
    import inspect

    from agent import context_compressor

    source = inspect.getsource(context_compressor)
    # Both auxiliary call sites in the compressor must carry the stamp.
    assert source.count('"model_endpoint"') >= 2
