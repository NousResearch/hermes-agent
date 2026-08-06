"""Billing-source guard for the Claude subscription runtime.

This module answers one question before a subscription turn is allowed to
start: **which account pays for it?**

The Claude Code CLI picks a credential in a fixed order (see
``docs/design/claude-subscription-via-agent-sdk.md`` §5, from
<https://code.claude.com/docs/en/iam>) and the user's subscription is *last*.
A developer with ``ANTHROPIC_API_KEY`` exported — the common case for Hermes
users, because the setup wizard asks for one — selects "Claude subscription"
and gets billed as metered API usage instead, silently, because that ordering
is correct behavior for the CLI.

Two mechanisms, deliberately overlapping:

* :func:`sanitized_child_env` builds the environment the CLI subprocess is
  launched from, with every higher-precedence credential **removed**.  This is
  structural: it is what
  :mod:`agent.transports.claude_sanitized_transport` hands to
  ``anyio.open_process``.  ``ClaudeAgentOptions.env`` cannot do this — the SDK
  merges it *over* ``os.environ`` (``subprocess_cli.py``: ``{**inherited_env,
  ..., **options.env, ...}``), so it can override a key but never delete one.
* :func:`static_billing_refusal` and :func:`probe_claude_billing_source`
  **refuse the turn** when a higher-precedence credential is present at all.
  Refusing costs the user a message; guessing wrong costs them money, so the
  guard errs toward refusing and names the exact variable and the exact fix.

Nothing here reads a credential *value*.  Presence is the only signal, and no
value is ever logged or included in a message.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

# T3 Code found 8s too short because a Bedrock-configured CLI is slow to
# initialize, and raised their equivalent probe to 25s.
CLAUDE_INIT_PROBE_TIMEOUT_SECONDS = 25.0


# ---------------------------------------------------------------------------
# Credential precedence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CredentialSlot:
    """One credential the CLI would prefer over the user's subscription.

    ``rank`` mirrors the documented precedence order; 6 (subscription OAuth
    from ``/login``) is the only rank we want to win, so every slot recorded
    here outranks it.
    """

    rank: int
    name: str
    kind: str  # "env" | "setting"
    bills: str
    fix: str


# Ranks 1-5 of the documented precedence list. Rank 6 — the subscription OAuth
# session from `claude auth login` — is the one we want to win and therefore
# has no entry.
#
# The rank-1 entries are the *selectors*: the CLI only reaches for cloud
# credentials when one of these is set, so removing the selector is what
# removes the whole slot. The list is wider than the three the docs name
# (BEDROCK / VERTEX / FOUNDRY) because the shipped CLI (2.1.220) also honors
# ANTHROPIC_AWS, ANTHROPIC_GOOGLE_CLOUD, MANTLE, and GATEWAY.
CLAUDE_CREDENTIAL_PRECEDENCE: Tuple[CredentialSlot, ...] = (
    *(
        CredentialSlot(
            rank=1,
            name=name,
            kind="env",
            bills="a cloud provider account (Bedrock / Vertex / Foundry / gateway)",
            fix=f"unset {name}",
        )
        for name in (
            "CLAUDE_CODE_USE_BEDROCK",
            "CLAUDE_CODE_USE_VERTEX",
            "CLAUDE_CODE_USE_FOUNDRY",
            "CLAUDE_CODE_USE_ANTHROPIC_AWS",
            "CLAUDE_CODE_USE_ANTHROPIC_GOOGLE_CLOUD",
            "CLAUDE_CODE_USE_MANTLE",
            "CLAUDE_CODE_USE_GATEWAY",
        )
    ),
    CredentialSlot(
        rank=2,
        name="ANTHROPIC_AUTH_TOKEN",
        kind="env",
        bills="whatever account issued that bearer token",
        fix="unset ANTHROPIC_AUTH_TOKEN",
    ),
    CredentialSlot(
        rank=3,
        name="ANTHROPIC_API_KEY",
        kind="env",
        bills="your Anthropic Console account, as metered API usage",
        fix="unset ANTHROPIC_API_KEY",
    ),
    # Not honored by the shipped CLI (2.1.220 reads ANTHROPIC_API_KEY and
    # ANTHROPIC_AUTH_TOKEN, not this name), but wrappers and older builds do.
    # A credential-shaped variable we cannot rule out is treated as one.
    CredentialSlot(
        rank=3,
        name="ANTHROPIC_TOKEN",
        kind="env",
        bills="whatever account issued that token",
        fix="unset ANTHROPIC_TOKEN",
    ),
    CredentialSlot(
        rank=4,
        name="apiKeyHelper",
        kind="setting",
        bills="whatever account the helper script's key belongs to",
        fix="remove `apiKeyHelper` from your Claude settings.json",
    ),
    CredentialSlot(
        rank=5,
        name="CLAUDE_CODE_OAUTH_TOKEN",  # claude-boundary: ok — variable name only, never read as a credential
        kind="env",
        # This one does draw on the plan, but through the extra-usage meter
        # rather than plan limits — a different bucket, and not what the user
        # picked when they picked "Claude subscription".
        bills="your plan's extra-usage credits, not its included limits",
        fix="unset CLAUDE_CODE_OAUTH_TOKEN",  # claude-boundary: ok — variable name only, never read as a credential
    ),
)

# Companions that carry the same credential in another form. They are stripped
# from the child environment but do not, on their own, trigger a refusal: none
# of them selects a billing source without its primary above also being set.
_COMPANION_ENV_VARS: Tuple[str, ...] = (
    "CLAUDE_CODE_OAUTH_REFRESH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR",  # claude-boundary: ok — variable name only, never read as a credential
    "ANTHROPIC_AWS_API_KEY",
    "ANTHROPIC_FOUNDRY_API_KEY",
    "ANTHROPIC_FOUNDRY_AUTH_TOKEN",
    "ANTHROPIC_IDENTITY_TOKEN_FILE",
)

# Everything removed from the child environment in subscription mode.
BLOCKED_CHILD_ENV_VARS: frozenset = frozenset(
    [slot.name for slot in CLAUDE_CREDENTIAL_PRECEDENCE if slot.kind == "env"]
    + list(_COMPANION_ENV_VARS)
)

# Named so the intent survives a future edit to the blocklist. CLAUDE_CONFIG_DIR
# is how a caller isolates Claude's config; HOME must never be rewritten,
# because on macOS that relocates the login-keychain lookup
# ($HOME/Library/Keychains) and the CLI then cannot find its stored OAuth
# credentials and reports "Not logged in".
PASS_THROUGH_ENV_VARS: Tuple[str, ...] = ("CLAUDE_CONFIG_DIR", "HOME")


def sanitized_child_env(
    base: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """Return a filtered copy of *base* (default ``os.environ``) for the child.

    A copy — ``os.environ`` itself is never mutated.  Hermes is multi-threaded
    and other providers run concurrently; a process-global mutation around a
    spawn would leak into them.
    """
    source = os.environ if base is None else base
    return {
        key: value
        for key, value in source.items()
        if key not in BLOCKED_CHILD_ENV_VARS
    }


def blocking_credentials(
    env: Optional[Mapping[str, str]] = None,
    *,
    settings_keys: Optional[Mapping[str, Any]] = None,
) -> List[CredentialSlot]:
    """Credential slots present in *env* that would outrank the subscription.

    Presence only: a value is never read, compared, or logged.  An
    empty-valued variable does not count — the shipped CLI treats ``FOO=`` as
    absent, and refusing on it would block users whose shell exports a blank
    placeholder.
    """
    source = os.environ if env is None else env
    found: List[CredentialSlot] = []
    for slot in CLAUDE_CREDENTIAL_PRECEDENCE:
        if slot.kind == "env":
            if (source.get(slot.name) or "").strip():
                found.append(slot)
        elif settings_keys is not None and settings_keys.get(slot.name):
            found.append(slot)
    return sorted(found, key=lambda s: s.rank)


def credential_refusal_message(slots: List[CredentialSlot]) -> str:
    """Refusal text naming each offending variable and its exact fix."""
    lines = [
        "Refusing to start the Claude subscription runtime: this turn would "
        "not be billed to your Claude plan."
    ]
    for slot in slots:
        noun = "is set" if slot.kind == "env" else "is configured"
        lines.append(
            f"  • {slot.name} {noun}, and the Claude Code CLI prefers it over "
            f"your subscription — requests would bill {slot.bills}."
        )
        lines.append(f"    Fix: {slot.fix}")
    lines.append("  Check what the CLI would use with: claude auth status")
    return "\n".join(lines)


def static_billing_refusal(
    env: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """Refusal string when *env* carries a higher-precedence credential.

    ``None`` means nothing in the environment outranks the subscription.
    """
    slots = blocking_credentials(env)
    if not slots:
        return None
    return credential_refusal_message(slots)


# ---------------------------------------------------------------------------
# Billing-source classification
# ---------------------------------------------------------------------------


def _normalize(value: Any) -> str:
    """Lowercase and strip separators, so ``claude.ai`` == ``claudeAi``."""
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


# ``tokenSource`` values that mean "a key/token resolved from configuration",
# not "the signed-in plan".
_API_TOKEN_SOURCES = frozenset(
    {"apikey", "anthropicapikey", "anthropicauthtoken", "apikeyhelper"}
)
_OAUTH_TOKEN_SOURCES = frozenset({"claudecodeoauthtoken"})
# Normalized ``apiProvider`` values that are Anthropic's own first-party
# service. Anything else is a cloud reseller.
_FIRST_PARTY_PROVIDERS = frozenset({"firstparty", "anthropic", ""})


@dataclass(frozen=True)
class BillingSource:
    """What the CLI says it would bill, as reported by its own init payload."""

    kind: str  # subscription | api_key | oauth_token | cloud | unauthenticated | unknown
    detail: str = ""  # the variable / provider that decided it
    plan: str = ""
    account: str = ""

    @property
    def is_subscription(self) -> bool:
        return self.kind == "subscription"


def classify_account(account: Optional[Mapping[str, Any]]) -> BillingSource:
    """Classify the ``account`` block of the CLI's initialize response.

    Observed shapes (Claude Code 2.1.220), in order of the checks below:

    * plan login — ``{"email", "organization", "subscriptionType",
      "apiProvider": "firstParty"}``
    * ``ANTHROPIC_API_KEY`` set — ``{"tokenSource": "claude.ai",
      "apiKeySource": "ANTHROPIC_API_KEY", "apiProvider": "firstParty"}``.
      Note ``tokenSource`` still says ``claude.ai`` here, so ``apiKeySource``
      is the load-bearing field and must be checked first.
    * ``ANTHROPIC_AUTH_TOKEN`` set — ``{"tokenSource": "ANTHROPIC_AUTH_TOKEN"}``
    * nothing signed in — ``{"tokenSource": "none"}``
    """
    if not isinstance(account, Mapping):
        return BillingSource(kind="unknown")

    provider = _normalize(account.get("apiProvider"))
    if provider not in _FIRST_PARTY_PROVIDERS:
        return BillingSource(
            kind="cloud", detail=str(account.get("apiProvider") or "").strip()
        )

    api_key_source = str(account.get("apiKeySource") or "").strip()
    if api_key_source:
        return BillingSource(kind="api_key", detail=api_key_source)

    token_source = str(account.get("tokenSource") or "").strip()
    normalized_token = _normalize(token_source)
    if normalized_token in _API_TOKEN_SOURCES:
        return BillingSource(kind="api_key", detail=token_source)
    if normalized_token in _OAUTH_TOKEN_SOURCES:
        return BillingSource(kind="oauth_token", detail=token_source)
    if normalized_token == "none":
        return BillingSource(kind="unauthenticated")

    plan = str(account.get("subscriptionType") or "").strip()
    email = str(account.get("email") or "").strip()
    if plan:
        return BillingSource(kind="subscription", plan=plan, account=email)
    if normalized_token and email:
        # A signed-in session whose tier the CLI did not name.
        return BillingSource(kind="subscription", detail=token_source, account=email)
    return BillingSource(kind="unknown", detail=token_source)


def billing_source_refusal(source: BillingSource) -> Optional[str]:
    """Refusal string for a classified billing source, or None when it pays."""
    if source.is_subscription:
        return None
    if source.kind == "api_key":
        named = source.detail
        if named == "apiKeyHelper":
            fix = "remove `apiKeyHelper` from your Claude settings.json"
        elif named:
            fix = f"unset {named}"
        else:
            fix = (
                "unset ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN, or remove "
                "`apiKeyHelper` from your Claude settings.json"
            )
        return (
            "Refusing to start the Claude subscription runtime: the Claude Code "
            f"CLI resolved an API key from {named or 'your configuration'}, so "
            "this turn would be billed as metered Anthropic API usage rather "
            "than to your Claude plan.\n"
            f"  Fix: {fix}, then start Hermes again.\n"
            "  Check with: claude auth status"
        )
    if source.kind == "oauth_token":
        return (
            "Refusing to start the Claude subscription runtime: the Claude Code "
            f"CLI is using {source.detail or 'CLAUDE_CODE_OAUTH_TOKEN'}, which "  # claude-boundary: ok — variable name only, never read as a credential
            "draws on your plan's extra-usage credits instead of its included "
            "limits.\n"
            f"  Fix: unset {source.detail or 'CLAUDE_CODE_OAUTH_TOKEN'} and sign "  # claude-boundary: ok — variable name only, never read as a credential
            "in with `claude auth login`.\n"
            "  Check with: claude auth status"
        )
    if source.kind == "cloud":
        return (
            "Refusing to start the Claude subscription runtime: the Claude Code "
            f"CLI is pointed at the '{source.detail}' provider, which bills a "
            "cloud account, not your Claude plan.\n"
            "  Fix: unset CLAUDE_CODE_USE_BEDROCK / CLAUDE_CODE_USE_VERTEX / "
            "CLAUDE_CODE_USE_FOUNDRY, then start Hermes again.\n"
            "  Check with: claude auth status"
        )
    if source.kind == "unauthenticated":
        return (
            "Refusing to start the Claude subscription runtime: the Claude Code "
            "CLI has no signed-in account.\n"
            "  Fix: run `claude auth login`."
        )
    return (
        "Refusing to start the Claude subscription runtime: could not confirm "
        "that this turn would be billed to your Claude plan.\n"
        "  Check with: claude auth status"
    )


# ---------------------------------------------------------------------------
# Zero-cost billing probe
# ---------------------------------------------------------------------------


def probe_options() -> Any:
    """Options for a probe-only session: no tools, no MCP servers, no prompt.

    ``setting_sources=[]`` is load-bearing beyond context isolation: it is also
    what stops the CLI running an ``apiKeyHelper`` from the user's settings
    (verified against Claude Code 2.1.220 — with ``["user"]`` the helper runs
    and the account reports ``tokenSource: apiKeyHelper``; with ``[]`` it never
    runs and the account reports the plan).
    """
    from claude_agent_sdk import ClaudeAgentOptions

    return ClaudeAgentOptions(tools=[], setting_sources=[], allowed_tools=[])


def _probe_account_payload(
    *,
    timeout: float,
    client_factory: Optional[Callable[..., Any]] = None,
    options_factory: Optional[Callable[[], Any]] = None,
    transport_factory: Optional[Callable[[Any], Any]] = None,
) -> Optional[Mapping[str, Any]]:
    """Read the CLI's initialize response without issuing a model request.

    The SDK's ``initialize`` control request is answered from the subprocess's
    *local* startup: the CLI reports the account it resolved before any prompt
    exists.  We connect, read that response, and disconnect — no user message
    is ever written to the child's stdin, so there is nothing for it to send
    to Anthropic and the probe costs no tokens and no quota.

    The three factories are injection points for tests; each defaults to the
    real SDK path and is imported only when it is used, so this module still
    imports without the optional extra.
    """
    import asyncio

    if client_factory is None:
        from claude_agent_sdk import ClaudeSDKClient

        client_factory = ClaudeSDKClient
    if transport_factory is None:
        from agent.transports.claude_sanitized_transport import (
            build_sanitized_transport,
        )

        transport_factory = build_sanitized_transport

    factory = client_factory
    options = (options_factory or probe_options)()

    result: Dict[str, Any] = {}

    async def _run() -> None:
        client = factory(options=options, transport=transport_factory(options))
        try:
            await client.connect()
            info = await client.get_server_info()
            if isinstance(info, Mapping):
                account = info.get("account")
                if isinstance(account, Mapping):
                    result["account"] = dict(account)
        finally:
            try:
                await client.disconnect()
            except Exception:
                logger.debug("claude billing probe disconnect failed", exc_info=True)

    def _thread() -> None:
        try:
            asyncio.run(asyncio.wait_for(_run(), timeout))
        except Exception as exc:
            result["error"] = str(exc)

    # Its own loop on its own thread: the caller is Hermes' synchronous turn
    # thread, which may already be inside another runtime's event loop.
    thread = threading.Thread(target=_thread, name="hermes-claude-billing-probe")
    thread.start()
    thread.join(timeout + 10.0)
    if thread.is_alive():
        logger.warning("claude billing probe did not finish within %.0fs", timeout)
        return None
    if "error" in result:
        logger.info("claude billing probe failed: %s", result["error"])
        return None
    return result.get("account")


def probe_claude_billing_source(
    *,
    timeout: float = CLAUDE_INIT_PROBE_TIMEOUT_SECONDS,
    client_factory: Optional[Callable[..., Any]] = None,
    options_factory: Optional[Callable[[], Any]] = None,
    transport_factory: Optional[Callable[[Any], Any]] = None,
) -> BillingSource:
    """Ask the CLI which account it would bill. Never raises.

    Falls back to ``claude auth status`` (:func:`hermes_cli.claude_code.
    probe_claude_auth`) when the SDK probe cannot run or reports nothing.
    """
    try:
        account = _probe_account_payload(
            timeout=timeout,
            client_factory=client_factory,
            options_factory=options_factory,
            transport_factory=transport_factory,
        )
    except Exception:
        logger.debug("claude billing probe raised", exc_info=True)
        account = None

    if account is not None:
        source = classify_account(account)
        if source.kind != "unknown":
            return source

    return _fallback_billing_source()


def _fallback_billing_source() -> BillingSource:
    """Classify from ``claude auth status`` when the SDK probe is unavailable."""
    try:
        from hermes_cli.claude_code import probe_claude_auth

        probe = probe_claude_auth()
    except Exception:
        logger.debug("claude auth status fallback failed", exc_info=True)
        return BillingSource(kind="unknown")

    if not probe.get("logged_in"):
        return BillingSource(kind="unauthenticated")
    if probe.get("auth_method") == "api-key":
        # `claude auth status` says "api-key" without naming the source, so the
        # message stays generic rather than guessing a variable.
        return BillingSource(kind="api_key")
    return BillingSource(
        kind="subscription",
        plan=str(probe.get("subscription_type") or ""),
        account=str(probe.get("account") or ""),
    )


# ---------------------------------------------------------------------------
# The gate the runtime calls
# ---------------------------------------------------------------------------


def verify_claude_billing_source(
    *,
    env: Optional[Mapping[str, str]] = None,
    probe: bool = True,
    timeout: float = CLAUDE_INIT_PROBE_TIMEOUT_SECONDS,
    client_factory: Optional[Callable[..., Any]] = None,
) -> Optional[str]:
    """Return a refusal string, or None when the turn bills the user's plan.

    Static first (free, and it names the variable the user has to act on),
    then the subprocess probe (exact, and catches anything the static list
    does not know about).
    """
    refusal = static_billing_refusal(env)
    if refusal is not None:
        return refusal
    if not probe:
        return None
    source = probe_claude_billing_source(timeout=timeout, client_factory=client_factory)
    return billing_source_refusal(source)


__all__ = [
    "BLOCKED_CHILD_ENV_VARS",
    "CLAUDE_CREDENTIAL_PRECEDENCE",
    "CLAUDE_INIT_PROBE_TIMEOUT_SECONDS",
    "PASS_THROUGH_ENV_VARS",
    "BillingSource",
    "CredentialSlot",
    "billing_source_refusal",
    "blocking_credentials",
    "classify_account",
    "credential_refusal_message",
    "probe_options",
    "probe_claude_billing_source",
    "sanitized_child_env",
    "static_billing_refusal",
    "verify_claude_billing_source",
]
