"""Config-shape tolerance for platforms.webhook.extra.routes (2026-09-24 RCA).

A LIST-form routes config crashed the gateway at startup with a cryptic
ValueError('dictionary update sequence element #0 has length 4; 2 is required')
— 571 crash-loop restarts on 2026-09-24 (12:53-17:56 CEST) before anyone
noticed. The adapter now normalizes the list form into the named mapping it
needs and raises an actionable error only for unrecoverable shapes.
"""

import logging

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.webhook import WebhookAdapter


def _config(routes):
    return PlatformConfig(
        enabled=True, extra={"host": "127.0.0.1", "port": 0, "routes": routes}
    )


def test_list_form_routes_normalized_to_named_mapping(caplog):
    cfg = _config([{"path": "/events", "secret": "s1", "script": "ingest.py"}])
    with caplog.at_level(logging.WARNING, logger="gateway.platforms.webhook"):
        adapter = WebhookAdapter(cfg)
    assert adapter._routes == {
        "events": {"path": "/events", "secret": "s1", "script": "ingest.py"}
    }
    assert any(
        "normalized" in (r.getMessage() or "") for r in caplog.records
    ), "list-form routes must log a warning naming the normalization"


def test_list_form_name_collision_gets_suffixes():
    adapter = WebhookAdapter(_config([{"path": "/a"}, {"path": "/a"}]))
    assert set(adapter._routes) == {"a", "a_2"}


def test_list_form_nested_path_becomes_underscore_name():
    adapter = WebhookAdapter(_config([{"path": "/hooks/github/pr"}]))
    assert set(adapter._routes) == {"hooks_github_pr"}


def test_list_form_non_dict_entry_raises_actionable():
    with pytest.raises(ValueError, match=r"routes.*list entries must be mappings.*got str"):
        WebhookAdapter(_config(["not-a-dict"]))


def test_non_dict_routes_raises_actionable():
    with pytest.raises(ValueError, match=r"routes must be a mapping of route name.*got int"):
        WebhookAdapter(_config(42))


def test_mapping_with_non_dict_value_raises_actionable():
    with pytest.raises(ValueError, match=r"non-mapping value\(s\).*broken"):
        WebhookAdapter(_config({"broken": "not-a-dict"}))


def test_valid_mapping_passes_through_unchanged():
    # Today's live shape (multi-secret array included) must be untouched.
    routes = {"events": {"path": "/events", "secret": ["s1", "s2"], "script": "ingest.py"}}
    adapter = WebhookAdapter(_config(routes))
    assert adapter._routes == routes


def test_null_routes_normalizes_to_empty():
    # A bare `routes:` YAML key (null) is indistinguishable from an absent key —
    # must not crash the gateway (review follow-up 24/9).
    adapter = WebhookAdapter(_config(None))
    assert adapter._routes == {}


def test_list_of_pairs_rejected_actionable():
    # Deliberate behavior change: dict() previously accepted [['name', {...}]] pair
    # sequences; the list form is now strictly route-config mappings (documented
    # in the commit message — exotic, undocumented shape).
    with pytest.raises(ValueError, match=r"list entries must be mappings.*got list"):
        WebhookAdapter(_config([["events", {"path": "/events"}]]))
