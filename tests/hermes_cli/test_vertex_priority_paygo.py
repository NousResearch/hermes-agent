"""Vertex Priority PayGo is opt-in: the header rides only when the user asked for it.

Vertex answers capacity 429s on the shared Standard PayGo pool. Priority PayGo is
Google's opt-in alternative: the request carries
``X-Vertex-AI-LLM-Shared-Request-Type: priority`` and is billed at a higher per-token rate.

The invariant that matters: the header is NEVER sent unless ``vertex.priority_paygo`` is
explicitly enabled. It changes what the user pays.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def vertex_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _write_config(home, priority_paygo) -> None:
    (home / "config.yaml").write_text(
        "vertex:\n"
        f"  project_id: proj-abc\n"
        f"  region: global\n"
        f"  priority_paygo: {str(priority_paygo).lower()}\n"
    )


def _profile():
    """The registered Vertex profile, loaded through the real discovery path."""
    from providers import get_provider_profile

    profile = get_provider_profile("vertex")
    assert profile is not None, "the vertex provider profile must be registered"
    return profile


def test_priority_paygo_header_is_sent_when_enabled(vertex_home):
    """Opting in adds the request header."""
    _write_config(vertex_home, True)

    _extra_body, top_level = _profile().build_api_kwargs_extras(reasoning_config=None)

    headers = top_level.get("extra_headers") or {}
    assert headers.get("X-Vertex-AI-LLM-Shared-Request-Type") == "priority"


def test_priority_paygo_header_is_absent_by_default(vertex_home):
    """The default is OFF: no config, no header — a billing change must never ride along."""
    _profile().build_api_kwargs_extras(reasoning_config=None)

    _extra_body, top_level = _profile().build_api_kwargs_extras(reasoning_config=None)

    assert not top_level.get("extra_headers"), (
        "the Priority PayGo header must not be sent unless the user enabled it")


def test_priority_paygo_header_is_absent_when_explicitly_disabled(vertex_home):
    """An explicit ``false`` is still off."""
    _write_config(vertex_home, False)

    _extra_body, top_level = _profile().build_api_kwargs_extras(reasoning_config=None)

    assert not top_level.get("extra_headers")


def test_the_header_never_leaks_to_another_provider(vertex_home):
    """Only the Vertex profile may carry the header."""
    _write_config(vertex_home, True)

    from providers import get_provider_profile

    for slug in ("openrouter", "nous", "gemini", "openai"):
        profile = get_provider_profile(slug)
        if profile is None:
            continue
        _extra_body, top_level = profile.build_api_kwargs_extras(reasoning_config=None)
        assert not (top_level.get("extra_headers") or {}).get("X-Vertex-AI-LLM-Shared-Request-Type"), (
            f"the Priority PayGo header leaked onto the {slug!r} provider")
