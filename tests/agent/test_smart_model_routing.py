from agent.smart_model_routing import apply_route


# ----- apply_route (match-based model.routes) -----

def test_apply_route_no_op_when_context_empty():
    cfg = {"routes": [{"match": {"platform": "hub"}, "model": "hub-model"}]}
    m, r = apply_route("base-model", {"api_key": "k1"}, cfg, {})
    assert m == "base-model" and r == {"api_key": "k1"}


def test_apply_route_no_op_when_no_routes():
    m, r = apply_route("base-model", {"api_key": "k1"}, {"default": "base-model"}, {"platform": "hub"})
    assert m == "base-model" and r == {"api_key": "k1"}


def test_apply_route_matches_by_platform():
    cfg = {
        "routes": [
            {"match": {"platform": "hub"}, "model": "hub-model", "api_key": "sk-hub"},
        ],
    }
    m, r = apply_route("base-model", {"api_key": "k1"}, cfg, {"platform": "hub", "source_kind": "hub_peer"})
    assert m == "hub-model"
    assert r["api_key"] == "sk-hub"


def test_apply_route_matches_by_source_kind():
    cfg = {
        "routes": [
            {"match": {"source_kind": "owner"}, "model": "strong-model"},
        ],
    }
    m, r = apply_route("base-model", {"api_key": "k1"}, cfg, {"platform": "telegram", "source_kind": "owner"})
    assert m == "strong-model"
    assert r == {"api_key": "k1"}  # base runtime preserved


def test_apply_route_requires_all_match_keys():
    # Compound match: both platform AND source_kind must match
    cfg = {
        "routes": [
            {"match": {"platform": "discord", "source_kind": "stranger"}, "model": "edge-case-model"},
        ],
    }
    m, _ = apply_route("base-model", {}, cfg, {"platform": "discord", "source_kind": "owner"})
    assert m == "base-model"  # source_kind mismatched → no match
    m, _ = apply_route("base-model", {}, cfg, {"platform": "telegram", "source_kind": "stranger"})
    assert m == "base-model"  # platform mismatched → no match
    m, _ = apply_route("base-model", {}, cfg, {"platform": "discord", "source_kind": "stranger"})
    assert m == "edge-case-model"  # both matched → route fires


def test_apply_route_first_match_wins():
    cfg = {
        "routes": [
            {"match": {"source_kind": "owner"}, "model": "first-route-model"},
            {"match": {"source_kind": "owner"}, "model": "second-route-model"},
        ],
    }
    m, _ = apply_route("base-model", {}, cfg, {"platform": "telegram", "source_kind": "owner"})
    assert m == "first-route-model"


def test_apply_route_legacy_platforms_shim_string():
    # ``model.platforms.hub: "hub-model"`` is legacy shorthand
    cfg = {"platforms": {"hub": "hub-model"}}
    m, _ = apply_route("base-model", {}, cfg, {"platform": "hub"})
    assert m == "hub-model"


def test_apply_route_legacy_platforms_shim_dict():
    cfg = {"platforms": {"hub": {"model": "hub-model", "base_url": "https://hub.example/v1"}}}
    m, r = apply_route("base-model", {}, cfg, {"platform": "hub"})
    assert m == "hub-model"
    assert r["base_url"] == "https://hub.example/v1"


def test_apply_route_legacy_by_source_shim():
    cfg = {"by_source": {"owner": {"model": "strong-model", "api_key": "sk-owner"}}}
    m, r = apply_route("base-model", {"api_key": "k1"}, cfg, {"source_kind": "owner"})
    assert m == "strong-model"
    assert r["api_key"] == "sk-owner"


def test_apply_route_explicit_routes_beat_legacy():
    # Explicit routes come first in the effective list → they match before legacy shims.
    cfg = {
        "routes": [{"match": {"source_kind": "owner"}, "model": "new-explicit"}],
        "by_source": {"owner": {"model": "old-legacy"}},
    }
    m, _ = apply_route("base-model", {}, cfg, {"source_kind": "owner"})
    assert m == "new-explicit"


def test_apply_route_partial_override_preserves_base_runtime():
    cfg = {"routes": [{"match": {"source_kind": "cron"}, "model": "cheap-model"}]}
    m, r = apply_route(
        "base-model",
        {"api_key": "k1", "base_url": "https://base.example/v1", "provider": "custom"},
        cfg,
        {"source_kind": "cron"},
    )
    assert m == "cheap-model"
    assert r == {"api_key": "k1", "base_url": "https://base.example/v1", "provider": "custom"}


def test_apply_route_empty_fields_dont_overwrite():
    cfg = {"routes": [{"match": {"source_kind": "owner"}, "api_key": "", "base_url": None, "args": []}]}
    m, r = apply_route(
        "base-model",
        {"api_key": "k1", "base_url": "https://base.example/v1"},
        cfg,
        {"source_kind": "owner"},
    )
    # Route matched but all fields are empty/nullish → inputs preserved
    assert m == "base-model"
    assert r["api_key"] == "k1"
    assert r["base_url"] == "https://base.example/v1"


def test_apply_route_context_missing_match_key_does_not_match():
    cfg = {"routes": [{"match": {"source_kind": "owner"}, "model": "strong-model"}]}
    m, _ = apply_route("base-model", {}, cfg, {"platform": "telegram"})  # no source_kind in context
    assert m == "base-model"


def test_apply_route_null_model_config():
    m, r = apply_route("base-model", {"api_key": "k1"}, None, {"source_kind": "owner"})
    assert m == "base-model" and r == {"api_key": "k1"}
