"""A pinned Codex route remains on its endpoint across real pool rotation and persistence."""

import json

import pytest

from tests.gateway import test_codex_effective_model_routing as fixtures
from tests.gateway.test_codex_effective_model_routing import BASE, TARGET
from tests.gateway.test_independent_rereview_boundaries import FOREIGN, row, session, store

home = fixtures.home


@pytest.mark.parametrize("path", ["fresh", "cached"])
def test_equivalent_endpoint_spelling_does_not_make_a_blocked_key_independent(home, path):
    pin = "https://CHATGPT.com:443/backend-api/codex/"
    store(home, [row(model_cooldowns={TARGET: 4_000_000_000}), row("synthetic-healthy")])
    _, runtime, wire = session(home, path, base=pin)
    assert wire["api_key"] == "synthetic-healthy"
    assert wire["base_url"] == pin
    assert runtime["credential_pool"].select(model=TARGET).runtime_api_key == "synthetic-healthy"


@pytest.mark.parametrize("path", ["fresh", "cached"])
def test_pinned_pool_rotation_never_borrows_foreign_endpoint(home, path):
    foreign = row("synthetic-foreign")
    foreign["base_url"] = FOREIGN
    own = row()
    own["priority"] = 1
    sibling = row("synthetic-sibling")
    sibling["priority"] = 2
    store(home, [foreign, own, sibling])
    model, runtime, wire = session(home, path)
    assert wire["api_key"] == "synthetic-blocked"
    assert wire["base_url"] == BASE
    pool = runtime["credential_pool"]
    selected = pool.mark_exhausted_and_rotate(status_code=400, api_key_hint=wire["api_key"],
                                              model=model, failure_reason="model_entitlement")
    assert selected.runtime_api_key == "synthetic-sibling"
    assert selected.base_url == BASE
    assert pool.mark_exhausted_and_rotate(status_code=400, api_key_hint=selected.runtime_api_key,
                                          model=model, failure_reason="model_entitlement") is None
    assert pool.select(model=TARGET) is None
    persisted = json.loads((home / "auth.json").read_text())["credential_pool"]["openai-codex"]
    assert next(entry for entry in persisted if entry["id"] == foreign["id"]) == foreign
    assert {entry["id"] for entry in persisted} == {foreign["id"], own["id"], sibling["id"]}
