"""The matter host's model is product-controlled: refuse a profile that routes around it.

Ana's model and fallback come from LitCo's product model settings, which LitKit sends on every turn
(``model``, ``fallbackModels``). A host must never carry a route of its own: no Hermes profile
fallback chain, no CLIProxyAPI or other proxy on a tailnet, no Claude Max or Codex subscription
account. On 2026-10-06 nothing checked that, and the incident review could not rule such a route out.

:func:`assert_product_controlled` is the one check. ``plugins/platforms/litco_turn/adapter.py`` runs it
before the turn server starts (the platform does not connect, so ``/health`` is never served) and
``deploy/host/litco-agent-init`` runs it on the values it renders (the unit fails with EX_CONFIG).

Standard library only, and no import of the ``litco`` package: litco-agent-init loads this file by
path under the system python3.
"""

from __future__ import annotations

import re
from typing import Any, List, Mapping, Optional

# Names of non-product model routes seen on LitCo machines (a CLIProxyAPI on the owner's tailnet,
# a Claude Max subscription). Matched case-insensitively anywhere in config text or endpoint values.
FORBIDDEN_ROUTE_TOKENS = ("cliproxy", "tail999258", "claude-max", "ncc1701d", "santacruz", ".ts.net:8317")
# Endpoint variables inspected by value (secrets never are): these name where model traffic goes.
ENDPOINT_KEYS = ("LITCO_MODEL_BASE_URL", "OPENAI_BASE_URL", "ANTHROPIC_BASE_URL")
# Hermes providers whose auth.json entries are subscription accounts, not product API keys.
SUBSCRIPTION_PROVIDERS = ("anthropic", "openai-codex")
_FALLBACK_KEYS = ("fallback_providers", "fallback_model")
# A top-level fallback key in YAML text whose value is not explicitly empty.
_FALLBACK_LINE = re.compile(r"^(fallback_providers|fallback_model)[ \t]*:[ \t]*(?P<value>[^#\n]*)", re.MULTILINE)
_EMPTY_YAML = ("", "[]", "{}", "null", "~", '""', "''")


class ProductModelGuardError(Exception):
    """The host's profile routes model traffic outside the product's settings."""


def _token_in(text: str) -> Optional[str]:
    lowered = (text or "").lower()
    return next((token for token in FORBIDDEN_ROUTE_TOKENS if token in lowered), None)


def _text_fallback_keys(config_text: str) -> List[str]:
    found = []
    for match in _FALLBACK_LINE.finditer(config_text or ""):
        value = match.group("value").strip()
        rest = config_text[match.end():]
        # ``key:`` alone opens a block; it is a chain when an indented item follows.
        block = value == "" and re.match(r"\n[ \t]+\S", rest) is not None
        if value not in _EMPTY_YAML or block:
            found.append(match.group(1))
    return found


def _subscription_accounts(auth: Optional[Mapping[str, Any]]) -> List[str]:
    if not isinstance(auth, Mapping):
        return []
    found = []
    providers = auth.get("providers")
    pool = auth.get("credential_pool")
    for name in SUBSCRIPTION_PROVIDERS:
        if isinstance(providers, Mapping) and providers.get(name):
            found.append(name)
            continue
        entries = pool.get(name) if isinstance(pool, Mapping) else None
        for entry in entries if isinstance(entries, list) else []:
            if not isinstance(entry, Mapping):
                continue
            token = str(entry.get("access_token") or "")
            if name == "openai-codex" or entry.get("auth_type") == "oauth" or token.startswith("sk-ant-oat"):
                found.append(name)
                break
    return found


def problems(config: Optional[Mapping[str, Any]], config_text: str, env: Mapping[str, str],
             auth: Optional[Mapping[str, Any]] = None) -> List[str]:
    """Every reason the profile is not product-controlled; empty when it is. Never quotes a value."""
    out: List[str] = []
    token = _token_in(config_text)
    if token:
        out.append(f"config.yaml names a non-product model route ({token})")
    for key in sorted(env):
        if key in ENDPOINT_KEYS or key.endswith("_BASE_URL"):
            token = _token_in(str(env.get(key) or ""))
            if token:
                out.append(f"{key} points at a non-product model route ({token})")
    config = config if isinstance(config, Mapping) else {}
    chains = sorted({key for key in _FALLBACK_KEYS if config.get(key)} | set(_text_fallback_keys(config_text)))
    if chains:
        out.append(f"config.yaml carries a Hermes fallback chain ({', '.join(chains)}); the product sends "
                   "fallbackModels on each turn")
    model = config.get("model")
    base_url = model.get("base_url") if isinstance(model, Mapping) else None
    token = _token_in(str(base_url or ""))
    if token:
        out.append(f"model.base_url points at a non-product model route ({token})")
    for name in _subscription_accounts(auth):
        out.append(f"auth.json holds a {name} subscription account; the matter host uses product API keys only")
    return out


def assert_product_controlled(config: Optional[Mapping[str, Any]], config_text: str, env: Mapping[str, str],
                              auth: Optional[Mapping[str, Any]] = None) -> None:
    """Raise :class:`ProductModelGuardError` naming every violation (see the module docstring)."""
    found = problems(config, config_text, env, auth)
    if found:
        raise ProductModelGuardError("the model route is not product-controlled: " + "; ".join(found))
