"""Both Sol context choices survive offline discovery and an older fallback cache."""
import json
import time

import pytest

from hermes_cli import codex_models, models

PAIR = ("gpt-6.1-sol", "gpt-6.1-sol-900k")


def assert_sol_choices(ids):
    assert all(model in ids for model in PAIR)
    assert all(ids.count(model) == 1 for model in PAIR)
    assert ids.index(PAIR[1]) == ids.index(PAIR[0]) + 1


def test_offline_codex_catalog_offers_standard_and_extended_sol(monkeypatch, tmp_path):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    assert_sol_choices(codex_models.get_codex_model_ids(access_token=None))


def test_forward_compatible_catalog_offers_both_sol_choices():
    assert_sol_choices(codex_models._finalize_codex_models(["gpt-6-sol", "gpt-6-luna"]))


@pytest.mark.parametrize("fallback", [True, False])
def test_cached_catalog_distinguishes_fallback_from_account_entitlement(monkeypatch, tmp_path, fallback):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(models, "_credential_fingerprint", lambda _: "test-principal")
    monkeypatch.setattr(models, "_spawn_swr_refresh", lambda _: None)
    ids = ["gpt-6-sol", "gpt-6-sol-900k", "gpt-6-luna", "gpt-6-luna-900k"]
    row = {"fp": "test-principal", "at": time.time(), "models": ids}
    if fallback:
        row["fallback"] = True
    cache = tmp_path / "provider_models_cache.json"
    cache.write_text(json.dumps({"openai-codex": row}))
    returned = models.cached_provider_model_ids("openai-codex", non_blocking=True)
    if fallback:
        assert_sol_choices(returned)
    else:
        # An account's actual catalog is not an offline fallback: do not invent entitlement.
        assert returned == ids
