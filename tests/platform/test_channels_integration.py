"""NOVA's channel output, fed to the runtime's own configuration and routing engine.

Not a stub anywhere. `parse_profile_routes`, `match_profile_route` and `load_gateway_config`
are the runtime's, and what they receive is whatever NOVA actually compiled. That matters
because the entire architecture rests on one claim — *a NOVA agent id is a runtime profile
name, so a declared channel routes with no glue* — and the only honest way to check a claim
like that is to hand the output to the thing that will consume it.

They skip when the runtime is not importable, which is the normal state of NOVA's own test
environment: the layer is meant to install without it.
"""

from __future__ import annotations

import pytest

from nova.channels import parse_channels
from nova.runtime.hermes.channels import apply, merge_into, plan, supports_serve_allowlist


@pytest.fixture(scope="module", autouse=True)
def _requires_runtime():
    pytest.importorskip(
        "gateway.profile_routing", reason="routing checks need the runtime's own engine"
    )


@pytest.fixture
def declared():
    return parse_channels(
        {
            "channels": [
                {
                    "id": "support-telegram",
                    "provider": "telegram",
                    "allowed_agents": ["customer-support", "operations"],
                    "routes": [
                        {"name": "escalations", "agent": "operations", "conversation": -100123},
                        {
                            "name": "escalation-thread",
                            "agent": "customer-support",
                            "conversation": -100123,
                            "thread": "77",
                        },
                        {"name": "everything-else", "agent": "customer-support"},
                    ],
                }
            ]
        }
    )


def test_the_runtime_parses_every_route_nova_compiles(declared):
    from gateway.profile_routing import parse_profile_routes

    compiled = [dict(route) for route in plan(declared).routes]
    parsed = parse_profile_routes(compiled)
    assert len(parsed) == len(compiled), "the runtime silently dropped a route NOVA compiled"


@pytest.mark.parametrize(
    "label, kwargs, expected",
    [
        ("a thread inside the escalations group", {"chat_id": "-100123", "thread_id": "77"}, "customer-support"),
        ("the escalations group itself", {"chat_id": "-100123"}, "operations"),
        ("any other conversation", {"chat_id": "55501"}, "customer-support"),
    ],
)
def test_inbound_resolves_to_the_agent_nova_declared(declared, label, kwargs, expected):
    from gateway.profile_routing import match_profile_route, parse_profile_routes

    routes = parse_profile_routes([dict(r) for r in plan(declared).routes])
    matched = match_profile_route(routes, platform="telegram", **kwargs)
    assert matched is not None, f"{label} matched nothing"
    assert matched.profile == expected, label


def test_a_platform_nobody_connected_routes_nowhere(declared):
    from gateway.profile_routing import match_profile_route, parse_profile_routes

    routes = parse_profile_routes([dict(r) for r in plan(declared).routes])
    assert match_profile_route(routes, platform="slack", chat_id="C123") is None


def test_every_route_target_is_inside_the_grant(declared):
    """The runtime rejects a route whose profile it does not serve. Compiling the grant into
    the served set is what turns NOVA's authorization into the runtime's own fail-closed
    check rather than only a parser opinion."""
    compiled = plan(declared)
    served = set(compiled.served_agents)
    assert served, "a connected channel must serve somebody"
    for route in compiled.routes:
        assert route["profile"] in served, (
            f"route {route['name']!r} targets {route['profile']!r}, which the runtime would "
            f"refuse to serve — the grant and the routes have diverged"
        )


def test_the_real_gateway_loader_accepts_what_nova_writes(declared, tmp_path, monkeypatch):
    """The end of the compile path: the runtime's own loader, on NOVA's own file."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("NOVA_HOME", str(tmp_path))

    # An operator-owned setting that must survive.
    import yaml

    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"gateway": {"max_concurrent_sessions": 7}}), encoding="utf-8"
    )
    apply(tmp_path, declared)

    from gateway.config import Platform, load_gateway_config

    config = load_gateway_config()
    assert config.multiplex_profiles is True
    assert config.max_concurrent_sessions == 7, "an operator setting was lost to an apply"
    assert len(config.profile_routes) == 3

    # The served-profile allowlist is asserted only where the runtime still has one. Upstream
    # removed the field and now computes the served set from the profile directory, so on
    # those versions NOVA must NOT write the key — an ignored key in a config file reads as a
    # live restriction to whoever opens it next.
    if supports_serve_allowlist():
        assert set(config.multiplex_profile_allowlist) == {"customer-support", "operations"}
    else:
        import yaml as _yaml

        written = _yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
        assert "multiplex_profile_allowlist" not in written["gateway"], (
            "NOVA wrote an allowlist this runtime ignores, which would overstate what is enforced"
        )

    telegram = config.platforms.get(Platform("telegram"))
    assert telegram is not None and telegram.enabled is True
    assert not telegram.token, "NOVA must never put a credential in the runtime's config"


def test_the_written_file_contains_no_credential(declared, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    apply(tmp_path, declared)
    text = (tmp_path / "config.yaml").read_text(encoding="utf-8").lower()
    for marker in ("token:", "api_key:", "secret:", "password:"):
        assert marker not in text, f"{marker} reached a file NOVA writes"


def test_revoking_every_channel_stops_delivery_in_the_runtime(declared, tmp_path, monkeypatch):
    """A removed channel that still delivers is the one outcome this layer must never
    produce, so this checks the runtime's view, not NOVA's."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    apply(tmp_path, declared)
    apply(tmp_path, ())

    from gateway.config import load_gateway_config

    config = load_gateway_config()
    # The routing table is what stops delivery on every runtime version, so this is the
    # assertion that matters and it is unconditional.
    assert config.profile_routes == []
    if supports_serve_allowlist():
        assert not config.multiplex_profile_allowlist


def test_nova_never_writes_an_allowlist_this_runtime_would_ignore(declared, tmp_path, monkeypatch):
    """The honesty rule, asserted directly rather than as a side effect of another test.

    `multiplex_profile_allowlist` was removed from `GatewayConfig` upstream; the served set is
    now computed in one chokepoint from the profile directory. Writing the key anyway would
    put a restriction in a config file that nothing reads — the exact shape of overclaim this
    project exists to avoid. So the key appears if and only if the runtime has the field.
    """
    import yaml

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("NOVA_HOME", str(tmp_path))
    plan_ = apply(tmp_path, declared)

    written = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    present = "multiplex_profile_allowlist" in written["gateway"]
    assert present is supports_serve_allowlist()

    # And when it is not written, the operator is told why rather than left to notice.
    if not present:
        assert any("no longer reads" in w for w in plan_.warnings), (
            "the grant quietly dropped a tier of enforcement and said nothing"
        )
