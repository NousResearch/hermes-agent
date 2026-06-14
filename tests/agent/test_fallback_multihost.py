"""Regression tests for multi-host fallback de-dup (_fallback_targets_current_backend).

Bug: the prior skip logic in try_activate_fallback skipped any fallback whose
provider+model matched the current backend, IGNORING base_url. That silently
dropped a same-provider/same-model fallback pointing at a DIFFERENT HOST — i.e.
multi-host failover (primary router on host A, fallback router on host B) never
engaged; the chain jumped straight past the second host.

The predicate must treat "same backend" as same model AND same endpoint, where
the endpoint is the fallback's explicit base_url when set, else the provider
default.
"""

from agent.chat_completion_helpers import _fallback_targets_current_backend as same_backend

# All args are pre-normalised (lower-cased, trailing slash stripped) by the caller.
PRIMARY_PROV = "nebius-tofa"
PRIMARY_MODEL = "hermes/standard"
PRIMARY_URL = "http://127.0.0.1:7831/v1"


def test_multihost_same_provider_model_different_host_is_NOT_current_backend():
    # THE FIX: rob-beast router — same provider+model as primary, different host.
    assert (
        same_backend(
            "nebius-tofa", "hermes/standard", "http://100.74.162.102:7831/v1",
            PRIMARY_PROV, PRIMARY_MODEL, PRIMARY_URL,
        )
        is False
    )


def test_same_provider_model_no_base_url_inherits_current_backend():
    # No explicit base_url -> inherits the provider default == current endpoint -> skip.
    assert (
        same_backend("nebius-tofa", "hermes/standard", "", PRIMARY_PROV, PRIMARY_MODEL, PRIMARY_URL)
        is True
    )


def test_same_base_url_same_model_is_current_backend_even_cross_provider():
    # Two distinct providers pointing at the SAME shim/proxy URL + same model -> skip.
    assert (
        same_backend("custom", "hermes/standard", PRIMARY_URL, PRIMARY_PROV, PRIMARY_MODEL, PRIMARY_URL)
        is True
    )


def test_exact_same_backend_is_skipped():
    assert (
        same_backend("nebius-tofa", "hermes/standard", PRIMARY_URL, PRIMARY_PROV, PRIMARY_MODEL, PRIMARY_URL)
        is True
    )


def test_different_model_is_not_current_backend():
    # The direct-ToFa seatbelt entry: different model -> always a distinct backend.
    assert (
        same_backend(
            "nebius-tofa", "qwen/qwen3-235b-a22b-instruct-2507",
            "https://api.tokenfactory.nebius.com/v1",
            PRIMARY_PROV, PRIMARY_MODEL, PRIMARY_URL,
        )
        is False
    )


def test_different_host_different_model_not_current_backend():
    assert (
        same_backend("nebius-tofa", "other/model", "http://100.74.162.102:7831/v1",
                     PRIMARY_PROV, PRIMARY_MODEL, PRIMARY_URL)
        is False
    )
