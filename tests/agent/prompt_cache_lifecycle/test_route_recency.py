"""A route the session just used stays in the remembered window."""

from types import SimpleNamespace

from agent.prompt_cache_clock import CacheEvent, _ROUTES_REMEMBERED, _set_event


def _event(name: str) -> CacheEvent:
    route = (name, "https://example.test/v1", "model")
    return CacheEvent(
        route=route, destination=route, route_class="openai_spec", requested_ttl=None,
        sent_at=1.0, read_tokens=0, write_tokens=0, tier=None, confidence="unknown", reason="test",
    )


def test_reused_route_is_kept_when_a_newer_route_arrives():
    agent = SimpleNamespace()
    for name in "abcde":
        _set_event(agent, _event(name))
    _set_event(agent, _event("b"))
    _set_event(agent, _event("f"))

    kept = {route[0] for route in agent._prompt_cache_clock_by_route}
    assert len(agent._prompt_cache_clock_by_route) == _ROUTES_REMEMBERED
    assert kept == {"b", "d", "e", "f"}
