"""First accounted route must keep the model column and model_config coherent (#118969).

A session row stores the live route twice: the ``model`` column (rewritten by the
first accounted usage when the requested route failed and fallback succeeded) and
the ``model_config`` JSON (what resume recombines the model with — provider,
base_url). Rewriting only the column left ``model_config`` pinned to the original
provider, so a resumed session recombined an impossible pair such as
``provider=xai + model=deepseek-v4-flash`` and 404'd on every reopened turn until
the user manually re-picked a model.
"""

import json

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _row_config(db, session_id):
    row = db.get_session(session_id)
    assert row is not None
    return row, (json.loads(row["model_config"]) if row["model_config"] else {})


class TestFirstAccountedRouteModelConfigSync:
    def test_fallback_route_updates_model_config_provider(self, db):
        # The desktop picker's selection as the fresh row records it: requested
        # route (xai, grok-imagine-image-2.0) with its endpoint pinned.
        db.create_session(
            "s1", source="cli", model="grok-imagine-image-2.0",
            model_config={
                "model": "grok-imagine-image-2.0", "provider": "xai",
                "base_url": "https://api.x.ai/v1",
            })
        # First accounted usage arrives on the fallback destination.
        db.update_token_counts(
            "s1", input_tokens=100, api_call_count=1,
            model="deepseek-v4-flash", billing_provider="deepseek",
            billing_base_url="https://api.deepseek.com/v1")

        row, config = _row_config(db, "s1")
        assert row["model"] == "deepseek-v4-flash"
        assert config["model"] == "deepseek-v4-flash"
        assert config["provider"] == "deepseek"
        assert config["base_url"] == "https://api.deepseek.com/v1"

    def test_falsy_base_url_drops_stale_endpoint_key(self, db):
        db.create_session(
            "s1", source="cli", model="primary-model",
            model_config={"model": "primary-model", "provider": "xai",
                          "base_url": "https://api.x.ai/v1"})
        db.update_token_counts(
            "s1", input_tokens=10, api_call_count=1,
            model="fallback-model", billing_provider="deepseek",
            billing_base_url=None)

        _, config = _row_config(db, "s1")
        assert config["provider"] == "deepseek"
        assert "base_url" not in config

    def test_lineage_markers_survive_the_rewrite(self, db):
        db.create_session(
            "s1", source="cli", model="primary-model",
            model_config={"model": "primary-model", "provider": "xai",
                          "_branched_from": "root-session"})
        db.update_token_counts(
            "s1", input_tokens=10, api_call_count=1,
            model="fallback-model", billing_provider="deepseek",
            billing_base_url="https://api.deepseek.com/v1")

        _, config = _row_config(db, "s1")
        assert config["_branched_from"] == "root-session"
        assert config["provider"] == "deepseek"

    def test_second_accounted_route_leaves_config_alone(self, db):
        db.create_session(
            "s1", source="cli", model="primary-model",
            model_config={"model": "primary-model", "provider": "xai"})
        db.update_token_counts(
            "s1", input_tokens=10, api_call_count=1,
            model="fallback-model", billing_provider="deepseek",
            billing_base_url="https://api.deepseek.com/v1")
        # After the first accounted call the row route is authoritative; a later
        # different route (one row cannot represent mixed usage) must not rewrite it.
        db.update_token_counts(
            "s1", input_tokens=10, api_call_count=1,
            model="other-model", billing_provider="zai",
            billing_base_url="https://api.z.ai/api/paas/v4")

        row, config = _row_config(db, "s1")
        assert row["model"] == "fallback-model"
        assert config["model"] == "fallback-model"
        assert config["provider"] == "deepseek"
