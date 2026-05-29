"""Pure, offline provider-resolution primitives.

This module is the foundation for the unified provider resolver described in
``docs/plans/2026-05-28-custom-provider-fallback-resolution.md`` (Task 1).

**Invariant: everything here is a pure, offline computation.** No network I/O,
no disk reads, no environment lookups, no mutation of shared state. These
functions are the single source of truth for three concerns that today are
re-implemented (and have drifted) across ``hermes_cli/auth.py``,
``hermes_cli/runtime_provider.py`` and ``agent/auxiliary_client.py``:

* :func:`canonicalize_provider` — one registry-driven alias table (fixes the
  divergence behind #12146).
* :func:`normalize_base_url` — one ``/v1`` normalizer (fixes #4600).
* :func:`select_api_mode` — one api_mode selection ladder.

:class:`ResolvedProvider` is the frozen value object the staged resolver will
produce (Task 3 onwards). Call sites are migrated in later tasks; this module
introduces no call-site changes on its own.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional
from urllib.parse import urlsplit, urlunsplit

from utils import base_url_hostname

__all__ = [
    "ResolvedProvider",
    "VALID_API_MODES",
    "canonicalize_provider",
    "normalize_base_url",
    "select_api_mode",
]


# ---------------------------------------------------------------------------
# ResolvedProvider
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResolvedProvider:
    """The single authoritative result of provider/credential resolution.

    Produced once per lifecycle (CLI init / gateway agent-build / ACP
    session-start) and carried on the agent — never re-derived per request.
    """

    provider: str            # canonical registry id (post-canonicalization)
    requested_provider: str  # what the caller asked for, pre-canonicalization
    api_mode: str            # chat_completions | codex_responses | anthropic_messages | ...
    base_url: str            # FULLY NORMALIZED, transport-correct
    api_key: Any             # str | Callable | "no-key-required" | "aws-sdk"
    base_url_source: str     # explicit | config.base_url | config.api_base | registry-default | ...
    key_source: str          # explicit | config | pool | oauth | env:<VAR> | no-key-required | none
    model: Optional[str] = None
    # Typed as Any to keep this module import-light and side-effect-free;
    # the concrete type is hermes_cli.credential_pool.CredentialPool.
    credential_pool: Optional[Any] = None
    # External-process providers (copilot-acp, local launchers) carry a launch
    # command + args. First-class fields (promoted from ``extra``) so consumers
    # read ``resolved.command`` / ``resolved.args`` with type safety instead of
    # an untyped, typo-prone ``extra.get("command")``.
    command: Optional[str] = None
    args: tuple = ()
    extra: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        """Reconstruct the legacy ``Dict[str, Any]`` runtime shape.

        Compatibility export for the handful of consumers that still want a
        plain dict. The legacy dict used a single ``"source"`` key and carried
        provider-specific keys (region, command, args, bedrock_anthropic,
        expires_at, …) inline; non-promoted ones live in :attr:`extra` and are
        re-emitted verbatim. ``base_url_source`` / ``key_source`` are NOT
        emitted here so this stays byte-compatible with the historical dict
        shape — the typed object's mapping reads (``get``/``[]``/``in``) DO
        expose them (see :meth:`_view`).
        """
        # Start from extra (legacy "source" + provider-specific keys), then let
        # the canonical named fields win — so a stray same-named key in `extra`
        # can never shadow the authoritative field value.
        out: dict = dict(self.extra)
        out["provider"] = self.provider
        out["requested_provider"] = self.requested_provider
        out["api_mode"] = self.api_mode
        out["base_url"] = self.base_url
        out["api_key"] = self.api_key
        if self.model is not None:
            out["model"] = self.model
        if self.credential_pool is not None:
            out["credential_pool"] = self.credential_pool
        # command/args were historically carried inline in the runtime dict
        # (only when set) — preserve that for legacy dict consumers.
        if self.command is not None:
            out["command"] = self.command
        if self.args:
            # Re-emit as a list: the legacy runtime dict carried ``args`` as a
            # list, so as_dict() must stay byte-compatible for consumers that
            # index it. ResolvedProvider stores a tuple internally (the value
            # object is frozen/hashable); the conversion happens only here.
            out["args"] = list(self.args)
        return out

    def _view(self) -> dict:
        """Full read view: the legacy dict shape PLUS the provenance fields.

        Backs the mapping reads so ``resolved.get("base_url_source")`` returns
        the real value instead of silently ``None`` (the dict/object dualism
        footgun). ``as_dict`` deliberately stays narrower for byte-compat."""
        out = self.as_dict()
        out["base_url_source"] = self.base_url_source
        out["key_source"] = self.key_source
        return out

    # --- dict-style read compatibility: lets the frozen value object stand in
    # for the legacy runtime dict. Backed by _view() (NOT as_dict) so every
    # field — including provenance — reads consistently with attribute access. ---
    def __getitem__(self, key: str) -> Any:
        return self._view()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._view().get(key, default)

    def __contains__(self, key: object) -> bool:
        return key in self._view()


# ---------------------------------------------------------------------------
# api_mode constants
# ---------------------------------------------------------------------------

# Mirror of hermes_cli.runtime_provider._VALID_API_MODES. Kept in sync via
# tests; this is the offline copy so this module never imports the heavy
# runtime_provider module.
VALID_API_MODES = frozenset(
    {
        "chat_completions",
        "codex_responses",
        "anthropic_messages",
        "bedrock_converse",
        "codex_app_server",
    }
)

_DEFAULT_API_MODE = "chat_completions"
# Modes whose SDK appends ``/chat/completions`` (or ``/responses``) to the base
# URL — these want a ``/v1`` segment when the user supplied only a bare host.
_OPENAI_STYLE_MODES = frozenset({"chat_completions", "codex_responses"})


# ---------------------------------------------------------------------------
# canonicalize_provider
# ---------------------------------------------------------------------------

# The one authoritative alias table. This is the union of the two legacy
# tables — the inline literal in ``auth.resolve_provider`` (auth.py:1500) and
# ``auxiliary_client._PROVIDER_ALIASES`` (auxiliary_client.py:131) — with the
# local-server aliases (ollama/vllm/llamacpp → custom) the aux table was
# missing. Test ``test_alias_parity_*`` hard-fails on any future drift.
_STATIC_PROVIDER_ALIASES: dict[str, str] = {
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
    # Local server aliases — route through the generic custom provider (#12146).
    "ollama": "custom", "ollama_cloud": "ollama-cloud",
    "vllm": "custom", "llamacpp": "custom",
    "llama.cpp": "custom", "llama-cpp": "custom",
}

# Lazily-built table = static aliases + any provider-plugin aliases not already
# mapped. Building it imports the ``providers`` registry (a local module import,
# no network); the static dict stays authoritative for existing keys, matching
# the precedence the legacy auth.py table used.
_extended_alias_cache: Optional[dict[str, str]] = None


def _alias_table(*, include_registry: bool = True) -> dict[str, str]:
    global _extended_alias_cache
    if not include_registry:
        return _STATIC_PROVIDER_ALIASES
    if _extended_alias_cache is not None:
        return _extended_alias_cache
    table = dict(_STATIC_PROVIDER_ALIASES)
    try:
        from providers import list_providers as _lp

        for _pp in _lp():
            for _alias in getattr(_pp, "aliases", ()):  # type: ignore[union-attr]
                key = str(_alias).strip().lower()
                if key and key not in table:
                    table[key] = _pp.name
    except Exception:
        # Registry unavailable / partial — fall back to the static table.
        # Never let alias canonicalization depend on a working plugin import.
        return _STATIC_PROVIDER_ALIASES
    _extended_alias_cache = table
    return table


def canonicalize_provider(name: Optional[str], *, include_registry: bool = True) -> str:
    """Map a requested provider name to its canonical registry id.

    Lowercases and strips ``name``, then resolves aliases via the one
    authoritative table (optionally extended with provider-plugin aliases).
    Unknown names and already-canonical ids are returned verbatim (lowercased)
    — this function never raises and never invents a mapping; validation
    against the registry happens in a later stage. ``None``/empty → ``"auto"``.
    """
    normalized = (name or "auto").strip().lower()
    if not normalized:
        return "auto"
    return _alias_table(include_registry=include_registry).get(normalized, normalized)


# ---------------------------------------------------------------------------
# normalize_base_url
# ---------------------------------------------------------------------------

def _append_v1_if_bare(path: str) -> str:
    # Only a bare host (no path, or just "/") gets a "/v1". Any existing path
    # segment — /v1, /v2, /v1beta, /openai/v1, /anthropic, /coding, /paas/v4 —
    # is left untouched, so the rule self-disables and stays idempotent.
    if path in ("", "/"):
        return "/v1"
    return path


def _strip_trailing_v1(path: str) -> str:
    # The Anthropic SDK appends "/v1/messages"; a base URL that already ends in
    # "/v1" would produce "/v1/v1/messages". Strip exactly one trailing "/v1"
    # (tolerating a trailing slash). "/anthropic" and other suffixes are kept.
    trimmed = path.rstrip("/")
    if trimmed.lower().endswith("/v1"):
        return trimmed[: -len("/v1")]
    return path


def normalize_base_url(url: Optional[str], api_mode: Optional[str]) -> str:
    """Return a transport-correct base URL for ``api_mode``.

    * ``chat_completions`` / ``codex_responses``: append exactly one ``/v1`` —
      but **only** when ``url`` is a bare host (empty path or ``/``). URLs that
      already carry a path/version segment are returned unchanged.
    * ``anthropic_messages``: strip a single trailing ``/v1`` (the SDK adds
      ``/v1/messages``); ``/anthropic`` and other suffixes are preserved.
    * Any other / unknown mode: returned unchanged.

    The query string is preserved. The function is idempotent:
    ``normalize_base_url(normalize_base_url(u, m), m) == normalize_base_url(u, m)``.
    Empty/``None`` input returns ``""``.
    """
    raw = (url or "").strip()
    if not raw:
        return raw

    mode = (api_mode or "").strip().lower()
    if mode not in _OPENAI_STYLE_MODES and mode != "anthropic_messages":
        return raw

    has_scheme = "://" in raw
    # urlsplit misreads a scheme-less "host:port" as scheme="host"; prefix "//"
    # so the host lands in netloc, then drop it again when rebuilding.
    split = urlsplit(raw if has_scheme else f"//{raw}")

    if mode == "anthropic_messages":
        new_path = _strip_trailing_v1(split.path)
    else:  # chat_completions / codex_responses
        new_path = _append_v1_if_bare(split.path)

    if new_path == split.path:
        # No change → return the original string verbatim (preserves trailing
        # slashes and any formatting the caller chose).
        return raw

    rebuilt = urlunsplit(split._replace(path=new_path))
    # We prepended "//" before parsing a scheme-less input; drop it on rebuild,
    # but only if urlunsplit actually re-emitted it (it omits "//" when the
    # netloc is empty, e.g. a host-less "?q=1" input — slicing blindly would
    # eat real characters).
    if not has_scheme and rebuilt.startswith("//"):
        return rebuilt[2:]
    return rebuilt


# ---------------------------------------------------------------------------
# select_api_mode
# ---------------------------------------------------------------------------

def _parse_api_mode(raw: Any) -> Optional[str]:
    """Validate an explicit api_mode override; return None if invalid."""
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in VALID_API_MODES:
            return normalized
    return None


def _detect_api_mode_for_url(base_url: Optional[str]) -> Optional[str]:
    """Auto-detect api_mode from the resolved base URL.

    Host matches are exact (via :func:`base_url_hostname`) so lookalike hosts
    like ``api.openai.com.attacker.test`` are NOT treated as native endpoints.
    Mirrors hermes_cli.runtime_provider._detect_api_mode_for_url.
    """
    raw = (base_url or "").strip()
    if not raw:
        return None
    hostname = base_url_hostname(raw)
    if hostname in ("api.x.ai", "api.openai.com"):
        # GPT-5.x / Grok tool calls with reasoning need the Responses API.
        return "codex_responses"
    # Match on the PATH only (query/fragment excluded) so a query value like
    # "?x=/anthropic" can't spoof the protocol.
    path = urlsplit(raw if "://" in raw else f"//{raw}").path.lower().rstrip("/")
    if path.endswith("/anthropic"):
        # Third-party Anthropic-compatible gateways expose the native protocol
        # under /anthropic.
        return "anthropic_messages"
    if hostname == "api.kimi.com" and "/coding" in path:
        return "anthropic_messages"
    return None


def select_api_mode(
    explicit_api_mode: Optional[str] = None,
    base_url: Optional[str] = None,
) -> str:
    """Select the wire protocol for a request.

    Precedence: a valid explicit override → URL-based detection → the
    ``chat_completions`` default. Invalid overrides are ignored (fall through).
    """
    override = _parse_api_mode(explicit_api_mode)
    if override:
        return override
    return _detect_api_mode_for_url(base_url) or _DEFAULT_API_MODE
