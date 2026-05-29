"""Unit tests for the pure provider-resolution primitives.

Foundation layer for the unified resolver — see
docs/plans/2026-05-28-custom-provider-fallback-resolution.md (Task 1, cpf-zkw.1).

Everything tested here is a *pure, offline* computation: no network I/O, no
disk reads, no env lookups. These functions are the single source of truth for
provider alias canonicalization, base_url ``/v1`` normalization, and api_mode
selection.
"""
import dataclasses

import pytest

from hermes_cli import provider_resolution as pr


# ---------------------------------------------------------------------------
# ResolvedProvider dataclass
# ---------------------------------------------------------------------------

def test_resolved_provider_is_frozen():
    rp = pr.ResolvedProvider(
        provider="custom",
        requested_provider="ollama",
        api_mode="chat_completions",
        base_url="http://localhost:11434/v1",
        api_key="no-key-required",
        base_url_source="config.base_url",
        key_source="no-key-required",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        rp.provider = "openrouter"  # type: ignore[misc]


def test_resolved_provider_defaults():
    rp = pr.ResolvedProvider(
        provider="openrouter",
        requested_provider="auto",
        api_mode="chat_completions",
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-xxx",
        base_url_source="registry-default",
        key_source="env:OPENROUTER_API_KEY",
    )
    assert rp.model is None
    assert rp.credential_pool is None
    assert rp.extra == {}


def test_resolved_provider_extra_is_independent_per_instance():
    a = pr.ResolvedProvider(
        provider="custom", requested_provider="custom", api_mode="chat_completions",
        base_url="http://h/v1", api_key="x", base_url_source="explicit", key_source="explicit",
    )
    b = pr.ResolvedProvider(
        provider="custom", requested_provider="custom", api_mode="chat_completions",
        base_url="http://h/v1", api_key="x", base_url_source="explicit", key_source="explicit",
    )
    assert a.extra is not b.extra


# ---------------------------------------------------------------------------
# canonicalize_provider
# ---------------------------------------------------------------------------

def test_canonicalize_lowercases_and_strips():
    assert pr.canonicalize_provider("  ZAI  ") == "zai"


def test_canonicalize_none_and_empty_are_auto():
    assert pr.canonicalize_provider(None) == "auto"
    assert pr.canonicalize_provider("") == "auto"
    assert pr.canonicalize_provider("   ") == "auto"


def test_canonicalize_passthrough_for_unknown_and_canonical_ids():
    # Already-canonical registry ids and unknown names are returned verbatim
    # (lowercased) — canonicalize never invents a mapping.
    assert pr.canonicalize_provider("anthropic") == "anthropic"
    assert pr.canonicalize_provider("openrouter") == "openrouter"
    assert pr.canonicalize_provider("custom") == "custom"
    assert pr.canonicalize_provider("totally-unknown-xyz") == "totally-unknown-xyz"


def test_canonicalize_known_aliases():
    assert pr.canonicalize_provider("glm") == "zai"
    assert pr.canonicalize_provider("claude") == "anthropic"
    assert pr.canonicalize_provider("github") == "copilot"
    assert pr.canonicalize_provider("ollama") == "custom"
    assert pr.canonicalize_provider("vllm") == "custom"
    assert pr.canonicalize_provider("llamacpp") == "custom"
    assert pr.canonicalize_provider("moonshot") == "kimi-coding"


# --- Alias parity with the two legacy tables (the whole point of #12146) ---

# Mirror of the inline literal in hermes_cli/auth.py:1500 (resolve_provider).
# Task 2 (cpf-zkw.2) routes that call site through canonicalize_provider and
# deletes the literal; until then this copy is the parity oracle. If auth.py's
# table changes, update this and canonicalize_provider together.
_AUTH_ALIASES_AT_1500 = {
    "glm": "zai", "z-ai": "zai", "z.ai": "zai", "zhipu": "zai",
    "google": "gemini", "google-gemini": "gemini", "google-ai-studio": "gemini",
    "x-ai": "xai", "x.ai": "xai", "grok": "xai",
    "xai-oauth": "xai-oauth", "x-ai-oauth": "xai-oauth",
    "grok-oauth": "xai-oauth", "xai-grok-oauth": "xai-oauth",
    "kimi": "kimi-coding", "kimi-for-coding": "kimi-coding", "moonshot": "kimi-coding",
    "kimi-cn": "kimi-coding-cn", "moonshot-cn": "kimi-coding-cn",
    "step": "stepfun", "stepfun-coding-plan": "stepfun",
    "arcee-ai": "arcee", "arceeai": "arcee",
    "gmi-cloud": "gmi", "gmicloud": "gmi",
    "minimax-china": "minimax-cn", "minimax_cn": "minimax-cn",
    "minimax-portal": "minimax-oauth", "minimax-global": "minimax-oauth", "minimax_oauth": "minimax-oauth",
    "alibaba_coding": "alibaba-coding-plan", "alibaba-coding": "alibaba-coding-plan",
    "alibaba_coding_plan": "alibaba-coding-plan",
    "claude": "anthropic", "claude-code": "anthropic",
    "github": "copilot", "github-copilot": "copilot",
    "github-models": "copilot", "github-model": "copilot",
    "github-copilot-acp": "copilot-acp", "copilot-acp-agent": "copilot-acp",
    "opencode": "opencode-zen", "zen": "opencode-zen",
    "qwen-portal": "qwen-oauth", "qwen-cli": "qwen-oauth", "qwen-oauth": "qwen-oauth",
    "google-gemini-cli": "google-gemini-cli", "gemini-cli": "google-gemini-cli", "gemini-oauth": "google-gemini-cli",
    "hf": "huggingface", "hugging-face": "huggingface", "huggingface-hub": "huggingface",
    "mimo": "xiaomi", "xiaomi-mimo": "xiaomi",
    "tencent": "tencent-tokenhub", "tokenhub": "tencent-tokenhub",
    "tencent-cloud": "tencent-tokenhub", "tencentmaas": "tencent-tokenhub",
    "aws": "bedrock", "aws-bedrock": "bedrock", "amazon-bedrock": "bedrock", "amazon": "bedrock",
    "go": "opencode-go", "opencode-go-sub": "opencode-go",
    "kilo": "kilocode", "kilo-code": "kilocode", "kilo-gateway": "kilocode",
    "lmstudio": "lmstudio", "lm-studio": "lmstudio", "lm_studio": "lmstudio",
    "ollama": "custom", "ollama_cloud": "ollama-cloud",
    "vllm": "custom", "llamacpp": "custom",
    "llama.cpp": "custom", "llama-cpp": "custom",
}


def test_alias_parity_with_auth_table():
    """Every alias in auth.py:1500 canonicalizes identically. → #12146"""
    mismatches = {
        alias: (pr.canonicalize_provider(alias), expected)
        for alias, expected in _AUTH_ALIASES_AT_1500.items()
        if pr.canonicalize_provider(alias) != expected
    }
    assert not mismatches, f"auth.py alias drift: {mismatches}"


# Mirror of the legacy auxiliary_client._PROVIDER_ALIASES literal (deleted in
# cpf-zkw.2 when the aux client began delegating to canonicalize_provider).
# Retained as a parity oracle: canonicalize must still cover everything the aux
# table used to, AND the local-server aliases (ollama/vllm/llamacpp) it omitted
# — the omission was the root of #12146.
_LEGACY_AUX_ALIASES = {
    "google": "gemini", "google-gemini": "gemini", "google-ai-studio": "gemini",
    "x-ai": "xai", "x.ai": "xai", "grok": "xai",
    "glm": "zai", "z-ai": "zai", "z.ai": "zai", "zhipu": "zai",
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
}


def test_alias_parity_with_legacy_auxiliary_client_table():
    """Every alias the legacy aux table declared canonicalizes identically. → #12146"""
    mismatches = {
        alias: (pr.canonicalize_provider(alias), expected)
        for alias, expected in _LEGACY_AUX_ALIASES.items()
        if pr.canonicalize_provider(alias) != expected
    }
    assert not mismatches, f"auxiliary_client alias drift: {mismatches}"


# ---------------------------------------------------------------------------
# normalize_base_url
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["chat_completions", "codex_responses"])
def test_normalize_appends_v1_only_for_bare_host(mode):
    assert pr.normalize_base_url("http://localhost:1234", mode) == "http://localhost:1234/v1"
    assert pr.normalize_base_url("http://localhost:1234/", mode) == "http://localhost:1234/v1"
    assert pr.normalize_base_url("https://api.example.com", mode) == "https://api.example.com/v1"


@pytest.mark.parametrize("mode", ["chat_completions", "codex_responses"])
@pytest.mark.parametrize(
    "url",
    [
        "https://api.example.com/v1",
        "https://api.example.com/v2",
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "https://api.groq.com/openai/v1",
        "https://api.example.com/anthropic",
        "https://api.kimi.com/coding",
        "https://open.bigmodel.cn/api/paas/v4",
    ],
)
def test_normalize_never_touches_url_with_path(mode, url):
    """The /v1 append rule self-disables on any non-bare path. → #4600"""
    assert pr.normalize_base_url(url, mode) == url


@pytest.mark.parametrize("mode", ["chat_completions", "codex_responses"])
def test_normalize_preserves_query_string(mode):
    assert (
        pr.normalize_base_url("http://localhost:1234?key=v", mode)
        == "http://localhost:1234/v1?key=v"
    )
    # A pathful URL with a query is left entirely alone.
    assert (
        pr.normalize_base_url("https://api.example.com/v1?key=v", mode)
        == "https://api.example.com/v1?key=v"
    )


def test_normalize_anthropic_strips_trailing_v1():
    assert (
        pr.normalize_base_url("https://api.example.com/v1", "anthropic_messages")
        == "https://api.example.com"
    )
    assert (
        pr.normalize_base_url("https://api.example.com/v1/", "anthropic_messages")
        == "https://api.example.com"
    )


def test_normalize_anthropic_preserves_anthropic_suffix():
    # /anthropic does not end in /v1 → untouched. ZAI's /api/anthropic special
    # case is preserved (it is rewritten by the aux transport layer, not here).
    assert (
        pr.normalize_base_url("https://api.minimax.io/anthropic", "anthropic_messages")
        == "https://api.minimax.io/anthropic"
    )
    assert (
        pr.normalize_base_url("https://open.bigmodel.cn/api/anthropic", "anthropic_messages")
        == "https://open.bigmodel.cn/api/anthropic"
    )


def test_normalize_anthropic_does_not_append_v1_to_bare_host():
    assert (
        pr.normalize_base_url("https://api.anthropic.com", "anthropic_messages")
        == "https://api.anthropic.com"
    )


@pytest.mark.parametrize(
    "mode",
    ["chat_completions", "codex_responses", "anthropic_messages", "bedrock_converse", None, ""],
)
@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:1234",
        "http://localhost:1234/",
        "https://api.example.com/v1",
        "https://api.example.com/anthropic",
        "https://api.groq.com/openai/v1",
        "https://open.bigmodel.cn/api/paas/v4",
        "http://localhost:1234?key=v",
    ],
)
def test_normalize_is_idempotent(mode, url):
    once = pr.normalize_base_url(url, mode)
    assert pr.normalize_base_url(once, mode) == once


def test_normalize_empty_and_none_passthrough():
    assert pr.normalize_base_url("", "chat_completions") == ""
    assert pr.normalize_base_url(None, "chat_completions") == ""


def test_normalize_host_less_query_only_input_not_corrupted():
    # Regression (cpf-zkw.11.1): a scheme-less, host-less input yields no "//"
    # from urlunsplit, so a blind [2:] slice used to eat the leading "/v" of
    # the appended "/v1". Output must not silently mangle characters.
    assert pr.normalize_base_url("?q=1", "chat_completions") == "/v1?q=1"


def test_normalize_unknown_mode_is_passthrough():
    # bedrock_converse / unknown modes have no /v1 rule.
    assert (
        pr.normalize_base_url("http://localhost:1234", "bedrock_converse")
        == "http://localhost:1234"
    )


# ---------------------------------------------------------------------------
# select_api_mode
# ---------------------------------------------------------------------------

def test_select_api_mode_explicit_override_wins():
    assert (
        pr.select_api_mode(explicit_api_mode="anthropic_messages", base_url="https://api.openai.com")
        == "anthropic_messages"
    )


def test_select_api_mode_invalid_override_is_ignored():
    assert (
        pr.select_api_mode(explicit_api_mode="garbage", base_url="http://localhost:1234")
        == "chat_completions"
    )


def test_select_api_mode_override_is_case_insensitive():
    assert (
        pr.select_api_mode(explicit_api_mode="  Anthropic_Messages ", base_url=None)
        == "anthropic_messages"
    )


def test_select_api_mode_detects_openai_codex_responses():
    assert pr.select_api_mode(base_url="https://api.openai.com/v1") == "codex_responses"


def test_select_api_mode_detects_xai_codex_responses():
    assert pr.select_api_mode(base_url="https://api.x.ai/v1") == "codex_responses"


def test_select_api_mode_detects_anthropic_suffix():
    assert pr.select_api_mode(base_url="https://api.minimax.io/anthropic") == "anthropic_messages"


def test_select_api_mode_detects_kimi_coding():
    assert pr.select_api_mode(base_url="https://api.kimi.com/coding") == "anthropic_messages"


def test_select_api_mode_defaults_to_chat_completions():
    assert pr.select_api_mode(base_url="http://localhost:1234/v1") == "chat_completions"
    assert pr.select_api_mode(base_url=None) == "chat_completions"
    assert pr.select_api_mode() == "chat_completions"


def test_select_api_mode_ignores_anthropic_in_query_string():
    # Regression (cpf-zkw.11.2): detection runs on the path only, so a query
    # value ending in /anthropic must NOT trigger anthropic_messages.
    assert (
        pr.select_api_mode(base_url="https://evil.test/path?x=/anthropic")
        == "chat_completions"
    )


def test_select_api_mode_ignores_coding_in_query_string():
    # api.kimi.com /coding detection is path-scoped too.
    assert (
        pr.select_api_mode(base_url="https://api.kimi.com/v1?next=/coding")
        == "chat_completions"
    )


def test_select_api_mode_resists_lookalike_host():
    # Substring spoof must NOT be detected as the OpenAI Responses endpoint.
    assert (
        pr.select_api_mode(base_url="https://api.openai.com.attacker.test/v1")
        == "chat_completions"
    )
