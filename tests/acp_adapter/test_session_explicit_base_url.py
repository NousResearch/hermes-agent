"""ACP threads explicit_base_url INTO resolution + provider-ownership rule.

Regression for #13489 (cpf-zkw.6): the ACP adapter used to resolve credentials
*before* applying the session base_url and then patch it on afterward
(``base_url or runtime.get("base_url")``). That skipped the bare-``custom``
resolution branch, so the API key was matched against the wrong provider and
the post-patch only corrected the displayed URL.

Fix: pass ``explicit_base_url`` into ``resolve_runtime_provider`` so the
resolver owns the routing decision, and use ``runtime["base_url"]`` verbatim.
The resolver then enforces provider-ownership: a custom provider honors the
explicit base_url; a built-in provider re-derives its own fixed endpoint and
never carries a stale session base_url forward.
"""

from __future__ import annotations

import pytest

import acp_adapter.session as acp_session


class _FakeAgent:
    """Records the kwargs AIAgent was constructed with."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        # _make_agent reads agent.model / sets agent._print_fn afterward.
        self.model = kwargs.get("model")


@pytest.fixture
def patch_agent(monkeypatch):
    """Stub run_agent.AIAgent so _make_agent doesn't build a real agent."""
    import run_agent

    created = {}

    def _factory(**kwargs):
        agent = _FakeAgent(**kwargs)
        created["agent"] = agent
        return agent

    monkeypatch.setattr(run_agent, "AIAgent", _factory)
    return created


def test_explicit_base_url_threaded_into_resolution(monkeypatch, patch_agent):
    """The session base_url is passed as explicit_base_url INTO the resolver,
    and the resolver's returned base_url is used verbatim (no post-patch)."""
    import hermes_cli.runtime_provider as rp
    import hermes_cli.config as config

    monkeypatch.setattr(config, "load_config",
                        lambda: {"model": {"provider": "custom", "default": "m"}})

    seen = {}

    def _spy(*, requested=None, explicit_api_key=None, explicit_base_url=None,
             target_model=None, **extra):
        seen.update(requested=requested, explicit_base_url=explicit_base_url,
                    target_model=target_model)
        return {
            "provider": "custom",
            "api_mode": "chat_completions",
            # Deliberately DIFFERENT from the session base_url to prove the
            # resolver's result wins (old `base_url or ...` patch would have
            # forced the session value instead).
            "base_url": "http://resolver-decided/v1",
            "api_key": "sk-resolved",
        }

    monkeypatch.setattr(rp, "resolve_runtime_provider", _spy)

    mgr = acp_session.SessionManager()
    mgr._make_agent(
        session_id="s1",
        cwd=".",
        model="m",
        requested_provider="custom",
        base_url="http://session-custom/v1",
    )

    # Resolver SAW the session base_url as explicit_base_url.
    assert seen["explicit_base_url"] == "http://session-custom/v1"
    assert seen["requested"] == "custom"

    # The agent uses the resolver's authoritative base_url — NOT a post-patched
    # `session or resolved` value.
    agent = patch_agent["agent"]
    assert agent.kwargs["base_url"] == "http://resolver-decided/v1"
    assert agent.kwargs["api_key"] == "sk-resolved"


def test_custom_provider_preserves_explicit_base_url(monkeypatch, patch_agent):
    """End-to-end with the REAL resolver: a custom provider honors the
    explicit session base_url (same-provider → base_url persists)."""
    import hermes_cli.config as config

    monkeypatch.setattr(config, "load_config",
                        lambda: {"model": {"provider": "custom", "default": "m"}})
    monkeypatch.setenv("OPENAI_API_KEY", "sk-custom-key")

    mgr = acp_session.SessionManager()
    mgr._make_agent(
        session_id="s2",
        cwd=".",
        model="m",
        requested_provider="custom",
        base_url="http://my-local-llm:1234/v1",
    )

    agent = patch_agent["agent"]
    assert agent.kwargs["provider"] == "custom"
    assert agent.kwargs["base_url"] == "http://my-local-llm:1234/v1"


def test_builtin_provider_resolves_its_own_endpoint(monkeypatch, patch_agent):
    """End-to-end with the REAL resolver: a built-in provider (openrouter)
    with no session base_url re-derives its own fixed endpoint.

    This is the cross-provider ownership rule in practice: a built-in
    provider's base_url comes from resolution, not from a carried-over
    session value. (The old `base_url or runtime.get("base_url")` post-patch
    would have force-overridden whatever resolution chose — #13489.)"""
    import hermes_cli.config as config
    from hermes_cli.runtime_provider import base_url_host_matches

    monkeypatch.setattr(config, "load_config",
                        lambda: {"model": {"provider": "openrouter", "default": "x/y"}})
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-key")

    mgr = acp_session.SessionManager()
    mgr._make_agent(
        session_id="s3",
        cwd=".",
        model="x/y",
        requested_provider="openrouter",
        base_url=None,  # a real openrouter session carries no custom base_url
    )

    agent = patch_agent["agent"]
    assert agent.kwargs["provider"] == "openrouter"
    assert base_url_host_matches(agent.kwargs.get("base_url") or "", "openrouter.ai"), \
        f"openrouter must resolve its own endpoint, got {agent.kwargs.get('base_url')!r}"


def test_custom_session_resolves_credential_for_threaded_base_url(monkeypatch, patch_agent):
    """#13489 (the credential path, end-to-end with the REAL resolver): the
    api_key is resolved FOR the session's threaded endpoint, not against the
    config default.

    Config declares ``provider: custom`` but carries NO base_url of its own, so
    the only endpoint information is what the ACP session threads in. A vendor
    key matching that endpoint's host is the only credential present; the
    resolver must select it (host-gated, #28660). The pre-fix resolve-then-patch
    flow resolved credentials against the config default first, so it could
    never have picked this host-matched key — proving the credential now follows
    the threaded URL."""
    import hermes_cli.config as config
    import hermes_cli.runtime_provider as rp

    rp.clear_resolution_memo()
    monkeypatch.setattr(config, "load_config",
                        lambda: {"model": {"provider": "custom", "default": "m"}})
    for var in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "CUSTOM_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-correct")

    mgr = acp_session.SessionManager()
    mgr._make_agent(
        session_id="s4",
        cwd=".",
        model="m",
        requested_provider="custom",
        base_url="https://api.deepseek.com/v1",
    )

    agent = patch_agent["agent"]
    assert agent.kwargs["provider"] == "custom"
    assert agent.kwargs["base_url"] == "https://api.deepseek.com/v1"
    # The credential is the one matching the THREADED endpoint host — not an
    # empty/openrouter key from resolving against the config default.
    assert agent.kwargs["api_key"] == "sk-deepseek-correct"


def test_custom_session_does_not_send_nonmatching_host_key(monkeypatch, patch_agent):
    """Negative companion to the above: a vendor key whose host does NOT match
    the session endpoint must NOT be attached (host-gating, #28660). With only a
    non-matching DEEPSEEK key set, the deepseek key must not leak to an
    unrelated local endpoint — the resolver falls back to no-key-required."""
    import hermes_cli.config as config
    import hermes_cli.runtime_provider as rp

    rp.clear_resolution_memo()
    monkeypatch.setattr(config, "load_config",
                        lambda: {"model": {"provider": "custom", "default": "m"}})
    for var in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "CUSTOM_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-correct")

    mgr = acp_session.SessionManager()
    mgr._make_agent(
        session_id="s5",
        cwd=".",
        model="m",
        requested_provider="custom",
        base_url="http://192.168.1.50:1234/v1",  # LAN host: no vendor label
    )

    agent = patch_agent["agent"]
    assert agent.kwargs["base_url"] == "http://192.168.1.50:1234/v1"
    assert agent.kwargs["api_key"] != "sk-deepseek-correct"
