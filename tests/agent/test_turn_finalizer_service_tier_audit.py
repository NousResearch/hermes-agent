"""The billing-audit `service_tier` field must see fast-mode tiers.

`hermes_cli/oneshot.py` advertises this field as the way batch pipelines
"verify the tier they think they're paying for actually went out on the wire",
citing a July 2026 incident where a silently-dropped flex caused 2.3x billing.

It only ever read `request_overrides["extra_body"]["service_tier"]` — the
custom-provider config shape. `resolve_fast_mode_overrides()` (the `/fast`
toggle and `agent.service_tier`) writes the tier TOP-LEVEL in
`request_overrides`, which is the shape the transport actually sends. So the
audit field reported None for every fast-mode turn, recreating exactly the
blind spot it exists to prevent.
"""

from __future__ import annotations

from types import SimpleNamespace

from agent.turn_finalizer import _requested_service_tier


def _agent(overrides):
    return SimpleNamespace(request_overrides=overrides)


def test_reports_top_level_tier_from_fast_mode_resolver():
    assert _requested_service_tier(_agent({"service_tier": "flex"})) == "flex"
    assert _requested_service_tier(_agent({"service_tier": "priority"})) == "priority"


def test_still_reports_custom_provider_extra_body_shape():
    """Back-compat: the original shape must keep working."""
    assert _requested_service_tier(_agent({"extra_body": {"service_tier": "flex"}})) == "flex"


def test_top_level_wins_when_both_present():
    """The resolver's tier is what the transport sends, so it is what we audit."""
    agent = _agent({"service_tier": "flex", "extra_body": {"service_tier": "priority"}})

    assert _requested_service_tier(agent) == "flex"


def test_none_when_no_tier_requested():
    assert _requested_service_tier(_agent({})) is None
    assert _requested_service_tier(_agent(None)) is None
    assert _requested_service_tier(SimpleNamespace()) is None


def test_anthropic_speed_is_not_a_service_tier():
    """`speed: fast` is a different knob and must not be reported as a tier."""
    assert _requested_service_tier(_agent({"speed": "fast"})) is None
