"""Gateway startup disclosure for implicit paid-cloud credentials.

Motivation — issue #32524
-------------------------

A user reported that ``hermes gateway run`` silently adopted
``ANTHROPIC_API_KEY`` from their shell environment and billed roughly
**$80** of Claude usage against their account before they noticed.
There was no startup line announcing which provider had been chosen,
no warning that the active credential came from an environment
variable (rather than from a deliberate ``hermes auth add`` step or
an explicit ``providers:`` config entry), and no indication that the
target was a paid cloud API.

This module fills the disclosure gap: every gateway startup runs the
resolved runtime credentials through :func:`classify_runtime_credential`
and emits a single, prominent banner via :func:`build_disclosure_lines`.
The banner always names the active provider + model + credential
source.  When the active provider is a paid cloud API **and** its
credential came implicitly from an environment variable (no auth
store entry, no explicit config wiring), the banner upgrades to a
``WARNING``-level multi-line block so operators can't miss it scrolling
past in ``gateway.log``.

The module is intentionally pure-Python and side-effect free — the
gateway is responsible for actually emitting the lines through its
logger.  This keeps the helper trivially unit-testable and lets
non-gateway callers (a future ``hermes status`` enhancement, the
dashboard, etc.) reuse the same classification without taking on a
``gateway.run`` import.

What counts as "implicit env"?
------------------------------

Heuristic (kept deliberately conservative — false negatives are fine,
false positives erode trust):

1. ``runtime["source"]`` resolves to a value indicating env-var pickup
   (``"env"``, ``"environment"``, ``"env-var"``, ``"env_fallback"``).
2. The :data:`PAID_CLOUD_PROVIDERS` allowlist contains the resolved
   provider id.
3. Neither the on-disk auth store nor the user's ``providers:`` /
   ``custom_providers:`` config sections explicitly mention the
   provider — this is checked via :func:`_provider_is_explicitly_configured`
   so any of those three "I meant to use this" signals suppresses the
   warning to a routine INFO disclosure.

When all three hold, the banner is the WARNING flavour; otherwise it's
the routine INFO flavour (or suppressed entirely when neither provider
nor model is known — typically because the gateway is failing closed
before any credentials resolved).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


# ─── Paid cloud allowlist ───────────────────────────────────────────────


# Providers whose env-var-only adoption should emit the loud (WARNING)
# disclosure variant. The criterion is "real money flows the moment we
# send a request" — local servers (LM Studio, Ollama, llama.cpp), free
# OAuth subscriptions (xai-oauth Premium+, Claude Code OAuth via
# ``ANTHROPIC_TOKEN`` — those are subscription-billed, not metered), and
# Nous Portal (subscription) are intentionally excluded.
#
# Aggregators (OpenRouter, Vercel AI Gateway, ai-gateway) are paid: even
# with prepaid credit, surprise usage drains the wallet. Same for
# OpenAI/Anthropic raw API keys, Gemini AI Studio, Z.AI/GLM, DeepSeek,
# xAI direct, Kimi/Moonshot, GMI, NVIDIA, MiniMax, Alibaba Dashscope.
#
# Bedrock and Vertex are excluded because their credential chain is
# governed by the cloud provider's IAM/console (the user did
# ``aws configure`` / ``gcloud auth``) — discovering AWS keys in a shell
# env is qualitatively different from discovering a raw ``ANTHROPIC_API_KEY``.
PAID_CLOUD_PROVIDERS: frozenset[str] = frozenset(
    {
        "anthropic",
        "openai",
        "openai-api",
        "openrouter",
        "gemini",
        "deepseek",
        "xai",
        "zai",
        "kimi-coding",
        "kimi-coding-cn",
        "minimax",
        "minimax-cn",
        "alibaba",
        "alibaba-coding-plan",
        "stepfun",
        "arcee",
        "gmi",
        "nvidia",
        "ai-gateway",
        "groq",
        "cerebras",
        "sambanova",
        "fireworks",
        "together",
        "perplexity",
        "mistral",
        "cohere",
        "huggingface",
    }
)


_ENV_SOURCE_TOKENS: frozenset[str] = frozenset(
    {"env", "environment", "env-var", "env_var", "env_fallback"}
)


# ─── Public API ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CredentialDisclosure:
    """Structured view of a resolved runtime credential.

    Returned by :func:`classify_runtime_credential` and consumed by
    :func:`build_disclosure_lines`.  Frozen so callers can safely cache
    or compare instances across requests.
    """

    provider: str
    model: str
    base_url: str
    source: str
    env_var: Optional[str]
    is_paid_cloud: bool
    is_implicit_env: bool

    @property
    def is_warning_worthy(self) -> bool:
        return self.is_paid_cloud and self.is_implicit_env


def classify_runtime_credential(
    runtime: Mapping[str, Any],
    *,
    model: str = "",
    auth_store: Optional[Mapping[str, Any]] = None,
    user_config: Optional[Mapping[str, Any]] = None,
    env: Optional[Mapping[str, str]] = None,
) -> CredentialDisclosure:
    """Classify a resolved runtime dict for startup disclosure.

    ``runtime`` is the shape returned by
    ``hermes_cli.runtime_provider.resolve_runtime_provider`` (and
    re-exposed by ``gateway.run._resolve_runtime_agent_kwargs``).  The
    only fields read are ``provider``, ``base_url``, ``source``, and
    optionally ``api_key`` (used together with ``env`` to identify the
    *specific* env var that supplied the key, so the warning can name
    it back to the user instead of saying "some env var").

    All other arguments are optional injection points for testing:

    * ``auth_store`` — pre-loaded ``hermes_cli.auth._load_auth_store()``
      dict.  When omitted, the store is loaded lazily so callers don't
      pay the I/O cost when the classifier short-circuits.
    * ``user_config`` — pre-loaded ``hermes_cli.config.load_config()``
      dict.  Same lazy-load semantics.
    * ``env`` — process environment mapping; defaults to ``os.environ``.
    """
    provider = str(runtime.get("provider") or "").strip().lower()
    source = str(runtime.get("source") or "").strip().lower()
    base_url = str(runtime.get("base_url") or "").strip()
    api_key = str(runtime.get("api_key") or "")

    env_map = env if env is not None else os.environ

    is_paid_cloud = provider in PAID_CLOUD_PROVIDERS
    env_var = _identify_env_var_for_key(provider, api_key, env_map) if api_key else None

    # Only resolve auth_store / config when we actually need them — the
    # common case (non-paid local provider, or env-only source false) is
    # decided in two attribute reads.
    explicit = False
    if is_paid_cloud and source in _ENV_SOURCE_TOKENS:
        explicit = _provider_is_explicitly_configured(
            provider, auth_store=auth_store, user_config=user_config
        )

    is_implicit_env = (
        is_paid_cloud
        and source in _ENV_SOURCE_TOKENS
        and not explicit
    )

    return CredentialDisclosure(
        provider=provider,
        model=str(model or "").strip(),
        base_url=base_url,
        source=source or "(unknown)",
        env_var=env_var,
        is_paid_cloud=is_paid_cloud,
        is_implicit_env=is_implicit_env,
    )


def build_disclosure_lines(info: CredentialDisclosure) -> List[Tuple[int, str]]:
    """Build the log lines for the gateway startup disclosure banner.

    Returns a list of ``(log_level, message)`` tuples so the caller can
    forward each line through ``logger.log(level, msg)``.  The first
    line is always an informational summary (``logging.INFO``); when
    :attr:`CredentialDisclosure.is_warning_worthy` is True, additional
    ``logging.WARNING`` lines explain the billing exposure and point at
    the remediation path (``hermes auth add ...`` or
    ``hermes auth pool ...``).
    """
    if not info.provider:
        return []

    summary = _format_summary(info)
    lines: List[Tuple[int, str]] = [(logging.INFO, summary)]

    if not info.is_warning_worthy:
        return lines

    # The reporter's pain point: no disclosure of the fact that a paid
    # cloud key was being used, no warning that it came from an env
    # var rather than a deliberate hermes auth step. Spell every piece
    # of that out so an operator scanning gateway.log can immediately
    # confirm the bill is going to land on this provider.
    env_var_hint = (
        f"environment variable {info.env_var}"
        if info.env_var
        else "an environment variable"
    )
    lines.append(
        (
            logging.WARNING,
            (
                f"⚠ {info.provider}: paid cloud API selected via {env_var_hint}. "
                "Every gateway message will be billed against this provider."
            ),
        )
    )
    lines.append(
        (
            logging.WARNING,
            (
                "  This account was not added via `hermes auth add` and is not "
                "pinned in config.yaml — Hermes inferred it from the environment."
            ),
        )
    )
    lines.append(
        (
            logging.WARNING,
            (
                "  To silence this warning intentionally: run "
                f"`hermes auth add {info.provider}` (or pin the provider in "
                "config.yaml `providers:`). To disable disclosure entirely: "
                "set `gateway.warn_implicit_paid_credentials: false` in "
                "config.yaml. See issue #32524 for context."
            ),
        )
    )
    return lines


def should_emit_disclosure(cfg: Optional[Mapping[str, Any]]) -> bool:
    """Return True when the gateway startup banner should be emitted.

    Reads ``gateway.warn_implicit_paid_credentials`` from the merged
    config.  Defaults to True when the key is missing — disclosure is
    on by default (the whole point of #32524) and operators must opt
    out, never opt in.
    """
    if not isinstance(cfg, Mapping):
        return True
    gw = cfg.get("gateway")
    if not isinstance(gw, Mapping):
        return True
    flag = gw.get("warn_implicit_paid_credentials")
    if flag is None:
        return True
    if isinstance(flag, bool):
        return flag
    if isinstance(flag, str):
        return flag.strip().lower() not in {"0", "false", "no", "off"}
    return bool(flag)


# ─── Internals ──────────────────────────────────────────────────────────


def _format_summary(info: CredentialDisclosure) -> str:
    parts = [f"Gateway active provider: {info.provider}"]
    if info.model:
        parts.append(f"model={info.model}")
    if info.base_url:
        parts.append(f"base_url={info.base_url}")
    parts.append(f"credential_source={info.source}")
    if info.env_var:
        parts.append(f"env_var={info.env_var}")
    return " | ".join(parts)


def _identify_env_var_for_key(
    provider: str, api_key: str, env: Mapping[str, str]
) -> Optional[str]:
    """Best-effort lookup: which env var supplied this api_key?

    We can't always know — the credential resolver may have munged the
    raw env value (e.g. trimming whitespace) — but the common case is a
    verbatim match against one of the provider's known env-var slots.
    """
    if not api_key or not provider:
        return None
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY  # heavy import — keep lazy
    except Exception:
        return None
    pcfg = PROVIDER_REGISTRY.get(provider)
    candidates: Iterable[str]
    if pcfg and pcfg.api_key_env_vars:
        candidates = pcfg.api_key_env_vars
    else:
        # Fall back to common conventions when the provider isn't in the
        # registry (custom user providers, plugin-defined providers).
        candidates = (
            f"{provider.upper()}_API_KEY",
            f"{provider.replace('-', '_').upper()}_API_KEY",
        )
    for name in candidates:
        val = env.get(name)
        if val and val.strip() == api_key.strip():
            return name
    return None


def _provider_is_explicitly_configured(
    provider: str,
    *,
    auth_store: Optional[Mapping[str, Any]] = None,
    user_config: Optional[Mapping[str, Any]] = None,
) -> bool:
    """True when the user has signalled deliberate use of ``provider``.

    Three explicit signals suppress the warning:

    1. An entry in ``~/.hermes/auth.json`` ``providers`` map — i.e. the
       user ran ``hermes auth add <provider>``.
    2. An entry in the auth store's ``credential_pool`` for the
       provider — i.e. ``hermes auth pool add <provider> ...``.
    3. A matching entry in the merged config's ``providers:`` or
       ``custom_providers:`` sections — i.e. the user hand-edited
       ``config.yaml`` and named the provider via ``key_env`` or
       ``base_url``.
    """
    provider_norm = (provider or "").strip().lower()
    if not provider_norm:
        return False

    if auth_store is None:
        try:
            from hermes_cli.auth import _load_auth_store

            auth_store = _load_auth_store()
        except Exception:
            auth_store = None

    if isinstance(auth_store, Mapping):
        providers = auth_store.get("providers")
        if isinstance(providers, Mapping):
            for key in providers.keys():
                if str(key).strip().lower() == provider_norm:
                    return True
        pool = auth_store.get("credential_pool")
        if isinstance(pool, Mapping):
            for key in pool.keys():
                if str(key).strip().lower() == provider_norm:
                    return True

    if user_config is None:
        try:
            from hermes_cli.config import load_config

            user_config = load_config()
        except Exception:
            user_config = None

    if isinstance(user_config, Mapping):
        # providers: (keyed dict) — name match counts.
        providers_cfg = user_config.get("providers")
        if isinstance(providers_cfg, Mapping):
            for key in providers_cfg.keys():
                if str(key).strip().lower() == provider_norm:
                    return True
        # custom_providers: (list of dicts) — name match counts.
        cps = user_config.get("custom_providers")
        if isinstance(cps, list):
            for entry in cps:
                if not isinstance(entry, Mapping):
                    continue
                name = str(entry.get("name") or "").strip().lower()
                if name == provider_norm:
                    return True
        # model.provider pinned to this provider counts too — the user
        # explicitly told the gateway to route here.
        model_cfg = user_config.get("model")
        if isinstance(model_cfg, Mapping):
            pinned = str(model_cfg.get("provider") or "").strip().lower()
            if pinned == provider_norm:
                return True

    return False
