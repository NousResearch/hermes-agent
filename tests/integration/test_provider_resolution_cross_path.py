"""Cross-path provider-resolution invariants.

These tests assert that the independent resolution entry points (CLI/auth,
auxiliary client, and — in later tasks — gateway/ACP) share a single source of
truth. They are deliberately NOT marked ``integration``: they require no
external services and must run in the default suite, because their whole job is
to hard-fail the moment the paths drift apart again.

Task 2 (cpf-zkw.2): both alias tables now route through
``hermes_cli.provider_resolution.canonicalize_provider``. → #12146
"""
import pytest

from hermes_cli.provider_resolution import canonicalize_provider


# Comprehensive alias oracle — the union of the two legacy literals that used to
# live inline in auth.resolve_provider and auxiliary_client._PROVIDER_ALIASES.
# If a future change drifts either call site, these tests fail.
_ALIAS_ORACLE = {
    "glm": "zai", "z-ai": "zai", "z.ai": "zai", "zhipu": "zai",
    "google": "gemini", "google-gemini": "gemini", "google-ai-studio": "gemini",
    "x-ai": "xai", "x.ai": "xai", "grok": "xai",
    "kimi": "kimi-coding", "moonshot": "kimi-coding",
    "kimi-cn": "kimi-coding-cn", "moonshot-cn": "kimi-coding-cn",
    "gmi-cloud": "gmi", "gmicloud": "gmi",
    "minimax-china": "minimax-cn", "minimax_cn": "minimax-cn",
    "claude": "anthropic", "claude-code": "anthropic",
    "github": "copilot", "github-copilot": "copilot",
    "github-model": "copilot", "github-models": "copilot",
    "github-copilot-acp": "copilot-acp", "copilot-acp-agent": "copilot-acp",
    "tencent": "tencent-tokenhub", "tokenhub": "tencent-tokenhub",
    "tencent-cloud": "tencent-tokenhub", "tencentmaas": "tencent-tokenhub",
    # Local-server aliases the aux table used to OMIT — the #12146 bug.
    "ollama": "custom", "vllm": "custom", "llamacpp": "custom",
    "llama.cpp": "custom", "llama-cpp": "custom",
}


@pytest.mark.parametrize("alias,expected", sorted(_ALIAS_ORACLE.items()))
def test_alias_tables_in_sync(alias, expected):
    """auth and aux now canonicalize every legacy alias identically. → #12146"""
    from agent.auxiliary_client import _normalize_aux_provider

    # The unified function is the single source of truth.
    assert canonicalize_provider(alias) == expected
    # The auxiliary client delegates to it (no private table, no drift).
    assert _normalize_aux_provider(alias) == expected


def test_aux_resolves_local_server_aliases_to_custom():
    """Regression for #12146: aux used to drop ollama/vllm/llamacpp, yielding
    "unknown provider 'ollama'" → OpenRouter fallback. Now they resolve to
    the generic custom provider on the aux path."""
    from agent.auxiliary_client import _normalize_aux_provider

    for alias in ("ollama", "vllm", "llamacpp", "llama.cpp", "llama-cpp"):
        assert _normalize_aux_provider(alias) == "custom"


def test_auth_resolve_provider_routes_through_canonicalize():
    """auth.resolve_provider's alias step is now canonicalize_provider.

    Aliases that map to a registry provider (or the special "custom" id) are
    returned directly, so these assertions are deterministic without any auth
    store / credentials.
    """
    from hermes_cli.auth import resolve_provider

    assert resolve_provider("glm") == "zai"
    assert resolve_provider("claude") == "anthropic"
    assert resolve_provider("ollama") == "custom"
    assert resolve_provider("vllm") == "custom"


# ---------------------------------------------------------------------------
# Task 3 (cpf-zkw.3): #5358 precedence + OpenRouter-fallback unreachability
# ---------------------------------------------------------------------------

def test_configured_vendor_key_wins_over_ambient_openai_key(monkeypatch):
    """#5358: a configured vendor key (DEEPSEEK_API_KEY) must win over an
    ambient OPENAI_API_KEY. The OPENAI/OPENROUTER short-circuit now runs at the
    BOTTOM of the auto-detect scan, so it no longer hijacks the selection."""
    from hermes_cli import auth

    monkeypatch.setattr(auth, "_load_auth_store", lambda: {})
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-key")
    monkeypatch.setenv("OPENAI_API_KEY", "oai-key")

    assert auth.resolve_provider("auto") == "deepseek"


def test_ambient_openrouter_key_is_last_resort(monkeypatch):
    """With no configured vendor key, the ambient OPENAI/OPENROUTER short-circuit
    still resolves to openrouter (preserved as the final fallback)."""
    from hermes_cli import auth

    monkeypatch.setattr(auth, "_load_auth_store", lambda: {})
    monkeypatch.setenv("OPENAI_API_KEY", "oai-key")

    assert auth.resolve_provider("auto") == "openrouter"


def test_openrouter_fallback_unreachable_for_configured_vendor(monkeypatch):
    """Global invariant (§5): a configured non-openrouter provider must never
    reach the terminal _resolve_openrouter_runtime fallback. Monkeypatch it to
    explode and prove a configured vendor still resolves to itself."""
    from hermes_cli import runtime_provider as rp

    def _boom(**kwargs):
        raise AssertionError("terminal OpenRouter fallback must be unreachable")

    monkeypatch.setattr(rp, "_resolve_openrouter_runtime", _boom)
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "anthropic")
    monkeypatch.setattr(rp, "_get_model_config", lambda: {"provider": "anthropic"})
    monkeypatch.setattr(rp, "load_pool", lambda provider: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")

    resolved = rp.resolve_runtime_provider(requested="anthropic")
    assert resolved["provider"] == "anthropic"
    assert "openrouter.ai" not in resolved["base_url"]


def test_bare_custom_without_base_url_fails_closed(monkeypatch):
    """Fail-closed invariant (§1 decision #1): ``provider: custom`` with no
    resolvable base_url anywhere must raise ``custom_provider_unresolved`` —
    NOT silently route custom intent to the openrouter.ai registry default with
    an empty key (the relabeled silent fallback this epic kills)."""
    from hermes_cli import runtime_provider as rp
    from hermes_cli.auth import AuthError

    monkeypatch.setattr(rp, "load_config", lambda: {"model": {"provider": "custom"}})
    monkeypatch.setattr(rp, "load_pool", lambda provider: None)
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "custom")
    for var in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "OPENAI_BASE_URL", "CUSTOM_BASE_URL", "OPENROUTER_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    rp.clear_resolution_memo()

    with pytest.raises(AuthError) as exc:
        rp.resolve_runtime_provider(requested="custom")
    assert exc.value.code == "custom_provider_unresolved"


def test_custom_with_explicit_base_url_does_not_fail_closed(monkeypatch):
    """The fail-closed guard must trip ONLY on the registry-default terminus —
    a real user-supplied base_url (here via explicit override) is honored."""
    from hermes_cli import runtime_provider as rp

    monkeypatch.setattr(rp, "load_config", lambda: {"model": {"provider": "custom"}})
    monkeypatch.setattr(rp, "load_pool", lambda provider: None)
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "custom")
    for var in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "OPENAI_BASE_URL", "CUSTOM_BASE_URL", "OPENROUTER_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    rp.clear_resolution_memo()

    resolved = rp.resolve_runtime_provider(
        requested="custom", explicit_base_url="http://localhost:8080"
    )
    assert resolved["provider"] == "custom"
    assert resolved["base_url"] == "http://localhost:8080/v1"
    assert "openrouter.ai" not in resolved["base_url"]


# ---------------------------------------------------------------------------
# Task 4 (cpf-zkw.4): offline-resolver invariant (decision #2)
# ---------------------------------------------------------------------------

def test_no_socket_during_resolve(monkeypatch):
    """Decision #2 tripwire: resolve_runtime_provider performs ZERO network I/O.

    A local custom base_url with no configured default used to trigger
    _auto_detect_local_model (a GET to <base_url>/v1/models) inside
    _get_model_config; that probe now lives only in the interactive/startup
    layer. Opening any socket during resolution fails this test."""
    import socket

    from hermes_cli import runtime_provider as rp

    monkeypatch.setattr(rp, "load_config", lambda: {
        "model": {"provider": "custom", "base_url": "http://localhost:11434"},
    })
    monkeypatch.setattr(rp, "load_pool", lambda provider: None)
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "openrouter")

    def _no_socket(*args, **kwargs):
        raise AssertionError("resolve_runtime_provider opened a socket (must be offline)")

    monkeypatch.setattr(socket, "socket", _no_socket)

    resolved = rp.resolve_runtime_provider(requested="custom")
    assert resolved["provider"] == "custom"
    # bare local host still normalized to /v1 — purely offline, no probe.
    assert resolved["base_url"] == "http://localhost:11434/v1"


# ---------------------------------------------------------------------------
# Task 9 (cpf-zkw.9): gateway resolves deterministically per context (#5358)
# ---------------------------------------------------------------------------

def test_gateway_threads_explicit_base_url(tmp_path, monkeypatch):
    """The gateway's per-message resolver threads the configured custom
    base_url through verbatim and NEVER falls back to OpenRouter (#5358).

    The gateway path (_resolve_runtime_agent_kwargs) must reach the same
    ResolvedProvider the CLI does — a configured custom provider stays custom.
    """
    from hermes_cli import runtime_provider as rp

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "model:\n"
        "  default: my-local-model\n"
        "  provider: custom\n"
        "  base_url: http://localhost:1234\n"
        "  api_key: sk-cfg\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    # An ambient OpenRouter key must NOT hijack the configured custom provider.
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-should-be-ignored")
    rp.clear_resolution_memo()

    from gateway.run import _resolve_runtime_agent_kwargs

    kwargs = _resolve_runtime_agent_kwargs()
    assert kwargs["provider"] == "custom"
    assert kwargs["base_url"] == "http://localhost:1234/v1"
    assert "openrouter.ai" not in (kwargs["base_url"] or "")

    # And it matches the CLI/typed resolver exactly — one source of truth.
    cli_obj = rp.resolve_runtime_provider_object(requested="custom")
    assert kwargs["base_url"] == cli_obj.base_url
    assert kwargs["provider"] == cli_obj.provider


def test_gateway_resolution_is_deterministic(tmp_path, monkeypatch):
    """Repeated gateway resolutions on unchanged inputs are byte-identical —
    the determinism that makes per-message re-resolution safe and lets the
    signature-gated agent cache hit (the real 'second half' of #5358)."""
    from hermes_cli import runtime_provider as rp

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "model:\n  default: m\n  provider: custom\n  base_url: http://localhost:9999\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    rp.clear_resolution_memo()

    from gateway.run import _resolve_runtime_agent_kwargs

    first = _resolve_runtime_agent_kwargs()
    second = _resolve_runtime_agent_kwargs()
    assert first == second
    assert first["base_url"] == "http://localhost:9999/v1"


# ---------------------------------------------------------------------------
# Follow-up (cpf-zkw.21): models.py picker aliases route through
# canonicalize_provider — the 4th (last) divergent alias surface. → #12146
# ---------------------------------------------------------------------------

def test_models_picker_aliases_match_canonical():
    """Every model-picker alias resolves identically to canonicalize_provider,
    except the documented picker-specific overrides (catalog routing)."""
    from hermes_cli.models import _PROVIDER_ALIASES, _PICKER_PROVIDER_ALIASES

    for alias, mapped in _PROVIDER_ALIASES.items():
        if alias in _PICKER_PROVIDER_ALIASES:
            # Intentional picker override (e.g. qwen→alibaba) — skip.
            continue
        assert mapped == canonicalize_provider(alias), (
            f"picker alias {alias!r}→{mapped!r} drifted from "
            f"canonicalize_provider→{canonicalize_provider(alias)!r}"
        )


def test_models_picker_override_qwen_diverges_intentionally():
    """The one genuine divergence is documented: picker qwen→alibaba (catalog),
    runtime canon qwen→qwen-oauth."""
    from hermes_cli.models import _PROVIDER_ALIASES

    assert _PROVIDER_ALIASES["qwen"] == "alibaba"
    assert canonicalize_provider("qwen") == "qwen-oauth"
