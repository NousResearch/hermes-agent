"""Tests for the Anthropic-on-Vertex runtime-provider integration.

Design contract: Anthropic Claude on Google Vertex AI does NOT have its
own provider name — it shares the ``vertex`` provider with Gemini-on-Vertex
because they run on the same GCP platform under the same ADC auth. The
wire transport is chosen at ``resolve_runtime_provider`` time based on the
requested model:

* ``anthropic/claude-*`` (or bare ``claude-*``) → ``anthropic_messages``
  runtime, backed by the ``AnthropicVertex`` SDK client.
* ``google/gemini-*`` (or anything else) → ``chat_completions`` runtime,
  backed by Vertex's OpenAI-compat aggregator.

The tests below cover:

1. The classifier (``is_anthropic_vertex_model``) recognises both prefixed
   and bare Claude forms and rejects Gemini + empty strings.
2. ``resolve_runtime_provider(requested="vertex", target_model=<claude>)``
   returns the expected ``anthropic_messages`` runtime dict shape that
   ``agent_init``, ``agent_runtime_helpers``, and ``run_agent`` consume.
3. ``resolve_runtime_provider(requested="vertex", target_model=<gemini>)``
   returns the existing ``chat_completions`` runtime and does NOT touch
   the Anthropic-Vertex adapter (regression guard against dispatch bleed).
4. All existing ``vertex`` aliases (``google-vertex``, ``vertex-ai``,
   ``gcp-vertex``, ``vertexai``) still resolve to the ``vertex`` provider
   AND still route Claude models through the Anthropic path.
5. Friendly ``AuthError`` messages when Vertex credentials or project_id
   cannot be resolved.

Distinct from ``test_anthropic_vertex_adapter.py``, which mocks at the
SDK seam to test client construction; this file exercises the
runtime-resolution branch that maps
``(requested_provider, target_model) → runtime dict``.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "model_id, expected",
    [
        # Vendor-prefixed Anthropic is the ONLY accepted form.
        ("anthropic/claude-opus-4-8", True),
        ("anthropic/claude-sonnet-4-5", True),
        ("anthropic/claude-haiku-4-5", True),
        # Case-insensitive + whitespace-tolerant.
        ("ANTHROPIC/Claude-Opus-4-8", True),
        ("  anthropic/claude-opus-4-8  ", True),
        # Bare Claude names DO NOT match — vendor prefix is required.
        # See ``is_anthropic_vertex_model`` docstring: strict form is
        # deliberate because Vertex is multi-vendor.
        ("claude-opus-4-8", False),
        ("claude-fable-5", False),
        # Gemini and other models are False for the same reason (wrong
        # vendor prefix or no prefix at all).
        ("google/gemini-3.1-pro-preview", False),
        ("gemini-3.1-pro-preview", False),
        ("google/gemma-3-27b-it", False),
        # Empty / non-string inputs are safely rejected.
        ("", False),
        (None, False),
        (123, False),
    ],
)
def test_is_anthropic_vertex_model(model_id, expected):
    from agent.anthropic_vertex_adapter import is_anthropic_vertex_model

    assert is_anthropic_vertex_model(model_id) is expected


# ---------------------------------------------------------------------------
# Dispatch: Claude on Vertex → anthropic_messages runtime
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "model",
    [
        "anthropic/claude-opus-4-8",
        "anthropic/claude-sonnet-4-5",
        # Version-suffixed IDs (Vertex's ``@YYYYMMDD`` form).
        "anthropic/claude-opus-4-5@20250929",
    ],
)
def test_vertex_provider_dispatches_claude_to_anthropic_messages(model, monkeypatch):
    """``requested=vertex`` + ``anthropic/`` model → anthropic_messages runtime.

    Confirms the runtime dict has every field ``agent_init._is_vertex_anthropic``
    and ``run_agent._rebuild_anthropic_client`` read to construct the
    AnthropicVertex client.
    """
    import agent.anthropic_vertex_adapter as ava
    from hermes_cli import runtime_provider as rp

    monkeypatch.setattr(ava, "has_anthropic_vertex_credentials", lambda: True)
    monkeypatch.setattr(
        ava, "get_anthropic_vertex_config", lambda: ("test-project-42", "global")
    )

    rt = rp.resolve_runtime_provider(requested="vertex", target_model=model)

    assert rt["provider"] == "vertex"
    assert rt["api_mode"] == "anthropic_messages"
    assert rt["source"] == "vertex-anthropic-oauth"
    # Placeholder key — AnthropicVertex mints its own OAuth token per request.
    assert rt["api_key"] == "vertex-adc"
    assert rt["anthropic_api_key"] == "vertex-adc"
    # Fields the client-construction sites read to build AnthropicVertex(project_id, region).
    assert rt["vertex_project_id"] == "test-project-42"
    assert rt["vertex_region"] == "global"
    assert rt["vertex_anthropic"] is True
    # Display-only base_url (real request URL is built inside the SDK).
    assert "aiplatform.googleapis.com" in rt["base_url"]
    assert "publishers/anthropic" in rt["base_url"]


@pytest.mark.parametrize(
    "alias",
    ["vertex", "google-vertex", "vertex-ai", "gcp-vertex", "vertexai"],
)
def test_all_vertex_aliases_route_claude_through_anthropic(alias, monkeypatch):
    """Every ``vertex`` alias must still route Claude to the Anthropic path
    — no alias regression when the ``anthropic-vertex`` provider name was
    removed."""
    import agent.anthropic_vertex_adapter as ava
    from hermes_cli import runtime_provider as rp

    monkeypatch.setattr(ava, "has_anthropic_vertex_credentials", lambda: True)
    monkeypatch.setattr(
        ava, "get_anthropic_vertex_config", lambda: ("proj", "global")
    )

    rt = rp.resolve_runtime_provider(
        requested=alias, target_model="anthropic/claude-opus-4-8"
    )
    assert rt["api_mode"] == "anthropic_messages"
    assert rt["provider"] == "vertex"


# ---------------------------------------------------------------------------
# Dispatch: Gemini on Vertex → chat_completions runtime (unchanged)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "cfg_default, expect_api_mode",
    [
        # Claude in config.yaml + no explicit target_model → anthropic path.
        ("anthropic/claude-opus-4-8", "anthropic_messages"),
        # Gemini in config.yaml + no explicit target_model → chat_completions.
        ("google/gemini-3.1-pro-preview", "chat_completions"),
    ],
)
def test_vertex_dispatch_when_target_model_is_none(
    cfg_default, expect_api_mode, monkeypatch
):
    """Regression: cron scheduler + gateway per-turn agent resolution
    call ``resolve_runtime_provider(requested="vertex", target_model=None)``.
    In that path the model must be sourced from ``_get_model_config()``,
    not from a local ``model_cfg`` reference that would trip Python's
    static scoping (``UnboundLocalError`` — ``model_cfg`` is assigned
    later in the same function body inside the auto-detect branch, so
    referencing it in the ``vertex`` branch without a distinct local
    name blows up on target_model=None call paths).

    Failure mode this test guards: prod's ``cron.scheduler.run_job``
    surfaced ``UnboundLocalError: cannot access local variable
    'model_cfg' where it is not associated with a value`` after the
    initial refactor landed, because CLI probes always pass a truthy
    ``target_model`` and the bug only fires when the short-circuit
    ``target_model or ...`` falls through to evaluate ``model_cfg``.
    """
    import agent.anthropic_vertex_adapter as ava
    import agent.vertex_adapter as va
    from hermes_cli import runtime_provider as rp

    monkeypatch.setattr(
        rp, "_get_model_config", lambda: {"provider": "vertex", "default": cfg_default}
    )
    monkeypatch.setattr(ava, "has_anthropic_vertex_credentials", lambda: True)
    monkeypatch.setattr(
        ava, "get_anthropic_vertex_config", lambda: ("proj", "global")
    )
    monkeypatch.setattr(
        va, "get_vertex_config",
        lambda: ("stub-token", "https://aiplatform.googleapis.com/v1beta1/projects/proj/locations/global/endpoints/openapi"),
    )

    rt = rp.resolve_runtime_provider(requested="vertex", target_model=None)

    assert rt["provider"] == "vertex"
    assert rt["api_mode"] == expect_api_mode


def test_vertex_provider_rejects_bare_claude_to_openai_compat(monkeypatch):
    """Regression: bare ``claude-*`` under ``provider=vertex`` must go
    through the OpenAI-compat aggregator, NOT the Anthropic path.

    Vertex is a multi-vendor surface, so a bare Claude name is ambiguous
    intent. The design deliberately routes it through the aggregator so
    Vertex 404s with an actionable error ("publisher google — model
    claude-opus-4-8 not found") rather than silently guessing at the
    intended wire protocol. This test locks that behavior in.
    """
    import agent.anthropic_vertex_adapter as ava
    import agent.vertex_adapter as va
    from hermes_cli import runtime_provider as rp

    monkeypatch.setattr(
        ava, "has_anthropic_vertex_credentials",
        lambda: pytest.fail("anthropic-vertex creds must not be checked for bare claude names"),
    )
    monkeypatch.setattr(
        ava, "get_anthropic_vertex_config",
        lambda: pytest.fail("anthropic-vertex config must not be read for bare claude names"),
    )
    monkeypatch.setattr(
        va, "get_vertex_config",
        lambda: (
            "stub-oauth-token",
            "https://aiplatform.googleapis.com/v1beta1/projects/proj/locations/global/endpoints/openapi",
        ),
    )

    rt = rp.resolve_runtime_provider(
        requested="vertex", target_model="claude-opus-4-8"
    )
    assert rt["api_mode"] == "chat_completions"
    assert rt["provider"] == "vertex"
    assert rt.get("vertex_anthropic") is not True


def test_vertex_provider_still_dispatches_gemini_to_chat_completions(monkeypatch):
    """Regression: Gemini-on-Vertex must NOT be touched by the Anthropic
    dispatch branch. Runtime resolution must call ``get_vertex_config``
    (the OpenAI-compat path) and never invoke the anthropic-vertex adapter.
    """
    import agent.anthropic_vertex_adapter as ava
    import agent.vertex_adapter as va
    from hermes_cli import runtime_provider as rp

    calls = {"anthropic_vertex_creds": 0, "anthropic_vertex_cfg": 0}
    monkeypatch.setattr(
        ava,
        "has_anthropic_vertex_credentials",
        lambda: (calls.__setitem__("anthropic_vertex_creds", calls["anthropic_vertex_creds"] + 1), True)[1],
    )
    monkeypatch.setattr(
        ava,
        "get_anthropic_vertex_config",
        lambda: (calls.__setitem__("anthropic_vertex_cfg", calls["anthropic_vertex_cfg"] + 1), ("proj", "global"))[1],
    )
    monkeypatch.setattr(
        va, "get_vertex_config",
        lambda: (
            "stub-oauth-token",
            "https://aiplatform.googleapis.com/v1beta1/projects/proj/locations/global/endpoints/openapi",
        ),
    )

    rt = rp.resolve_runtime_provider(
        requested="vertex", target_model="google/gemini-3.1-pro-preview"
    )
    assert rt["api_mode"] == "chat_completions"
    assert rt["provider"] == "vertex"
    assert rt.get("vertex_anthropic") is not True
    assert rt["source"] == "vertex-oauth"
    # The Anthropic-Vertex adapter must not have been consulted at all.
    assert calls == {"anthropic_vertex_creds": 0, "anthropic_vertex_cfg": 0}


# ---------------------------------------------------------------------------
# Removed provider name: back-compat guard
# ---------------------------------------------------------------------------

def test_no_standalone_anthropic_vertex_provider():
    """The old ``anthropic-vertex`` ProviderProfile was intentionally
    removed as part of the refactor to a single ``vertex`` provider —
    dispatch is now model-driven. Confirm the name doesn't resolve so we
    don't accidentally re-introduce a duplicate registration."""
    from providers import get_provider_profile

    # The ``vertex`` provider still exists.
    assert get_provider_profile("vertex").name == "vertex"
    # The old ``anthropic-vertex`` name no longer resolves.
    assert get_provider_profile("anthropic-vertex") is None


@pytest.mark.parametrize(
    "old_alias", ["claude-vertex", "anthropic-gcp", "vertex-anthropic"],
)
def test_removed_aliases_no_longer_resolve(old_alias):
    """Old aliases from the standalone-provider era must not resolve —
    surface a clear error rather than silently routing to the wrong thing."""
    from providers import get_provider_profile

    assert get_provider_profile(old_alias) is None


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_missing_credentials_raises_actionable_autherror(monkeypatch):
    import agent.anthropic_vertex_adapter as ava
    from hermes_cli import runtime_provider as rp
    from hermes_cli.auth import AuthError

    monkeypatch.setattr(ava, "has_anthropic_vertex_credentials", lambda: False)

    with pytest.raises(AuthError) as exc:
        rp.resolve_runtime_provider(
            requested="vertex", target_model="anthropic/claude-opus-4-8"
        )
    msg = str(exc.value)
    assert "OAuth2" in msg
    # Actionable next step: Vertex Model Garden enablement.
    assert "Model Garden" in msg


def test_missing_project_id_raises_autherror(monkeypatch):
    """Credentials resolved, but no project_id inferable — surface a
    project-specific error message so the user knows exactly what to set."""
    import agent.anthropic_vertex_adapter as ava
    from hermes_cli import runtime_provider as rp
    from hermes_cli.auth import AuthError

    monkeypatch.setattr(ava, "has_anthropic_vertex_credentials", lambda: True)
    monkeypatch.setattr(ava, "get_anthropic_vertex_config", lambda: (None, None))

    with pytest.raises(AuthError) as exc:
        rp.resolve_runtime_provider(
            requested="vertex", target_model="anthropic/claude-opus-4-8"
        )
    msg = str(exc.value)
    assert "project_id" in msg


# ---------------------------------------------------------------------------
# Model normalization
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "input_model, expected",
    [
        # Claude on Vertex: strip ``anthropic/`` so the AnthropicVertex SDK
        # gets the bare model name it needs for URL construction.
        ("anthropic/claude-opus-4-8", "claude-opus-4-8"),
        ("anthropic/claude-sonnet-4-5", "claude-sonnet-4-5"),
        ("ANTHROPIC/claude-opus-4-8", "claude-opus-4-8"),
        # Bare Claude passes through unchanged.
        ("claude-opus-4-8", "claude-opus-4-8"),
        # Gemini keeps its ``google/`` prefix (required by the OpenAI-compat
        # aggregator wire).
        ("google/gemini-3.1-pro-preview", "google/gemini-3.1-pro-preview"),
        # Bare Gemini too — Vertex will 404 with a clear hint, that's the
        # user-facing failure mode.
        ("gemini-3.1-pro-preview", "gemini-3.1-pro-preview"),
    ],
)
def test_normalize_model_for_vertex_strips_anthropic_prefix_only(
    input_model, expected
):
    """Normalization contract for ``vertex``: strip ``anthropic/`` (needed
    for AnthropicVertex SDK URL construction), preserve everything else."""
    from hermes_cli.model_normalize import normalize_model_for_provider

    assert normalize_model_for_provider(input_model, "vertex") == expected
