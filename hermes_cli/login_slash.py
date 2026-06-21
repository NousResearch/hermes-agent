"""Interactive CLI slash command wrapper for provider login.

The canonical credential plumbing lives under ``hermes auth``.  This module
keeps ``/login`` as a small, cache-safe CLI convenience layer so users can add,
inspect, and remove provider credentials without leaving a Hermes session.
"""

from __future__ import annotations

import argparse
import shlex
from types import SimpleNamespace

from hermes_cli.auth import PROVIDER_REGISTRY
from hermes_cli.auth_commands import (
    auth_add_command,
    auth_list_command,
    auth_remove_command,
    auth_reset_command,
)

_PROVIDER_ALIASES = {
    "claude": "anthropic",
    "claude-code": "anthropic",
    "anthropic": "anthropic",
    "codex": "openai-codex",
    "openai": "openai-codex",
    "openai-codex": "openai-codex",
    "github": "copilot",
    "github-copilot": "copilot",
    "github-models": "copilot",
    "copilot": "copilot",
    "grok": "xai-oauth",
    "xai": "xai-oauth",
    "xai-oauth": "xai-oauth",
    "gemini": "google-gemini-cli",
    "google": "google-gemini-cli",
    "google-gemini-cli": "google-gemini-cli",
    "qwen": "qwen-oauth",
    "qwen-oauth": "qwen-oauth",
    "nous": "nous",
    "minimax": "minimax-oauth",
    "minimax-oauth": "minimax-oauth",
}

_OAUTH_DEFAULT_PROVIDERS = {
    "anthropic",
    "copilot",
    "google-gemini-cli",
    "minimax-oauth",
    "nous",
    "openai-codex",
    "qwen-oauth",
    "xai-oauth",
}


def normalize_login_provider(raw: str) -> str:
    """Normalize user-facing ``/login`` provider aliases."""
    provider = (raw or "").strip().lower().replace("_", "-")
    return _PROVIDER_ALIASES.get(provider, provider)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="/login",
        description="Authenticate provider-backed Hermes sessions.",
    )
    parser.add_argument(
        "action_or_provider",
        nargs="?",
        default="status",
        help="provider alias, or one of: add, status, list, logout, remove, refresh",
    )
    parser.add_argument("provider", nargs="?", help="provider for subcommands")
    parser.add_argument("target", nargs="?", help="credential index/id/label for logout/remove")
    parser.add_argument("--type", dest="auth_type", choices=["oauth", "api-key", "api_key"])
    parser.add_argument("--label", help="credential label/account name")
    parser.add_argument("--api-key", help="API key value for --type api-key")
    parser.add_argument("--no-browser", action="store_true", help="do not open browser automatically")
    parser.add_argument("--manual-paste", action="store_true", help="paste OAuth callback URL/code manually when supported")
    parser.add_argument("--timeout", type=float, default=None, help="login timeout in seconds")
    parser.add_argument("--portal-url", help="Nous portal override")
    parser.add_argument("--inference-url", help="provider inference URL override")
    parser.add_argument("--client-id", help="OAuth client id override")
    parser.add_argument("--scope", help="OAuth scope override")
    parser.add_argument("--ca-bundle", help="CA bundle path")
    parser.add_argument("--insecure", action="store_true", help="disable TLS verification for testing")
    return parser


def _known_provider(provider: str) -> bool:
    return provider in PROVIDER_REGISTRY or provider == "openrouter" or provider.startswith("custom:")


def _add_args(provider: str, ns: argparse.Namespace) -> SimpleNamespace:
    auth_type = (ns.auth_type or "").strip().lower().replace("-", "_")
    if auth_type == "api_key":
        auth_type = "api_key"
    elif auth_type == "oauth":
        auth_type = "oauth"
    elif provider in _OAUTH_DEFAULT_PROVIDERS:
        auth_type = "oauth"
    else:
        auth_type = "api_key"

    return SimpleNamespace(
        provider=provider,
        auth_type=auth_type,
        label=ns.label,
        api_key=ns.api_key,
        portal_url=ns.portal_url,
        inference_url=ns.inference_url,
        client_id=ns.client_id,
        scope=ns.scope,
        no_browser=bool(ns.no_browser),
        manual_paste=bool(ns.manual_paste),
        timeout=ns.timeout,
        insecure=bool(ns.insecure),
        ca_bundle=ns.ca_bundle,
        min_key_ttl_seconds=5 * 60,
    )


def run_login_slash(command: str) -> None:
    """Execute a ``/login`` command body.

    ``command`` may include or omit the leading slash command token.  Raises
    ``SystemExit`` from argparse/auth helpers just like normal CLI subcommands.
    """
    text = (command or "").strip()
    if text.startswith("/login"):
        text = text[len("/login"):].strip()
    argv = shlex.split(text) if text else []
    ns = _build_parser().parse_args(argv)

    first = (ns.action_or_provider or "status").strip().lower().replace("_", "-")
    action_aliases = {
        "ls": "status",
        "list": "status",
        "status": "status",
        "add": "add",
        "login": "add",
        "remove": "remove",
        "logout": "remove",
        "rm": "remove",
        "refresh": "refresh",
        "reset": "refresh",
    }

    if first in action_aliases:
        action = action_aliases[first]
        provider = normalize_login_provider(ns.provider or "")
    else:
        action = "add"
        provider = normalize_login_provider(first)

    if action == "status":
        provider = normalize_login_provider(ns.provider or "")
        auth_list_command(SimpleNamespace(provider=provider or None))
        if provider and _known_provider(provider):
            # auth_list_command is intentionally quiet for empty pools; give an
            # explicit line so `/login status anthropic` is legible.
            from agent.credential_pool import load_pool

            if not load_pool(provider).has_credentials():
                print(f"{provider}: no pooled credentials")
        return

    if not provider:
        raise SystemExit("Provider required. Examples: /login anthropic, /login codex, /login github")
    if not _known_provider(provider):
        raise SystemExit(f"Unknown provider: {provider}")

    if action == "add":
        auth_add_command(_add_args(provider, ns))
        return

    if action == "remove":
        target = ns.target or ""
        if not target:
            raise SystemExit(
                "Credential target required. Usage: /login logout <provider> <index|id|label>"
            )
        auth_remove_command(SimpleNamespace(provider=provider, target=target))
        return

    if action == "refresh":
        # This resets local cooldown/error state. Runtime OAuth refresh still
        # happens through the provider-specific credential resolver on next use.
        auth_reset_command(SimpleNamespace(provider=provider))
        print(f"{provider}: credential cooldowns reset; runtime token will refresh on next use if needed")
        return

    raise SystemExit(f"Unknown /login action: {action}")
