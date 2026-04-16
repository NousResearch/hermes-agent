"""Slash-command preprocessor for the OpenAI-compatible API server adapter.

The API server's ``/v1/runs`` and ``/v1/chat/completions`` endpoints are
intentionally stateless — each request spins up a fresh ``AIAgent`` with
no session cache, no model override dict, and no reference to the
``GatewayRouter`` that platforms like Discord and the CLI route through.
See the comment at ``gateway/run.py:3148`` which confirms api_server is
excluded from the router notification path.

Without an interceptor, a message like ``/model claude-sonnet-4-6`` sent
to ``/v1/runs`` passes through to the LLM verbatim.  The model then
hallucinates a plausible-sounding reply such as *"``/model`` is a
client-side command, I can't execute it"*, because it doesn't know the
command was supposed to be handled by the gateway.  Frontends have no
way to tell the command was swallowed.

This module is a small preprocessor that runs before the LLM is invoked.
It checks the user text for a leading ``/``, resolves the first token
against ``COMMAND_REGISTRY`` from :mod:`hermes_cli.commands`, and either:

* **Executes** the command, for the subset that is genuinely stateless
  (``/help``, ``/commands``, ``/profile``, ``/provider``) using the same
  helpers that :class:`GatewayRouter` uses.  These exist so API-server
  frontends can reach the canonical help/config surface without having
  to hardcode command lists.
* **Declines politely** for commands that require router-owned session
  state (``/model``, ``/new``, ``/retry``, ``/yolo``, …) by returning a
  short notice explaining that the endpoint is stateless.  This replaces
  the LLM-hallucination path with a deterministic, helpful response.
* **Falls through** (``return None``) for non-slash text, unknown slash
  commands, and strictly ``cli_only`` commands — the caller then runs
  the normal LLM path.

The preprocessor is wrapped in a top-level ``try/except`` so that any
bug inside it logs at WARNING and falls through to the LLM path.  A
preprocessor fault must never take down a normal chat request.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from hermes_cli.commands import resolve_command

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SlashCommandResult:
    """Result of a handled gateway slash command.

    ``text`` is the plain-text body to surface to the caller; the caller
    is responsible for wrapping it in whatever transport envelope the
    endpoint needs (OpenAI chunk, ``message.delta``, JSON, …).

    ``command`` is the canonical command name (without the leading
    slash) — useful for logging and for echoing the name back to the
    user in decline notices.
    """

    text: str
    command: str


# ---------------------------------------------------------------------------
# Stateless handlers — each returns the textual body of a synthetic reply.
# They must NOT touch router-owned state (session store, agent cache,
# model override dict, per-adapter pickers).
# ---------------------------------------------------------------------------


def _handle_help(_args: str) -> str:
    """Return the canonical gateway help listing.

    Delegates to :func:`hermes_cli.commands.gateway_help_lines` so the
    output stays in sync with the central command registry and honors
    ``gateway_config_gate`` overrides automatically.
    """
    from hermes_cli.commands import gateway_help_lines

    lines = ["**Hermes Commands**", ""]
    lines.extend(gateway_help_lines())
    lines.append("")
    lines.append(
        "Note: commands that mutate session state (for example `/model`, "
        "`/new`, `/retry`, `/yolo`) are not available on the stateless "
        "`/v1/runs` and `/v1/chat/completions` endpoints.  Use a channel "
        "with persistent session state (CLI, Discord, Telegram, Slack) "
        "for those."
    )
    return "\n".join(lines)


def _handle_commands(args: str) -> str:
    """Return the paginated ``/commands`` listing.

    Mirrors :meth:`GatewayRouter._handle_commands_command` but without
    the platform-specific page size and without the skill-commands
    section (which depends on router-side skill loading that api_server
    does not do).
    """
    from hermes_cli.commands import gateway_help_lines

    entries = list(gateway_help_lines())
    if not entries:
        return "No commands available."

    raw = (args or "").strip()
    if raw:
        try:
            requested_page = int(raw)
        except ValueError:
            return "Usage: `/commands [page]`"
    else:
        requested_page = 1

    page_size = 20
    total_pages = max(1, (len(entries) + page_size - 1) // page_size)
    page = max(1, min(requested_page, total_pages))
    start = (page - 1) * page_size
    page_entries = entries[start:start + page_size]

    lines = [
        f"**Commands** ({len(entries)} total, page {page}/{total_pages})",
        "",
        *page_entries,
    ]
    if total_pages > 1:
        nav_parts = []
        if page > 1:
            nav_parts.append(f"`/commands {page - 1}` ← prev")
        if page < total_pages:
            nav_parts.append(f"next → `/commands {page + 1}`")
        lines.extend(["", " | ".join(nav_parts)])
    if page != requested_page:
        lines.append(
            f"_(Requested page {requested_page} was out of range, "
            f"showing page {page}.)_"
        )
    return "\n".join(lines)


def _handle_profile(_args: str) -> str:
    """Return the active profile name and home directory.

    Mirrors :meth:`GatewayRouter._handle_profile_command`.  Reads only
    environment / filesystem state; no router dependency.
    """
    from pathlib import Path

    from hermes_constants import display_hermes_home, get_hermes_home

    home = get_hermes_home()
    display = display_hermes_home()

    profiles_parent = Path.home() / ".hermes" / "profiles"
    try:
        rel = home.relative_to(profiles_parent)
        profile_name = str(rel).split("/")[0]
    except ValueError:
        profile_name = None

    if profile_name:
        return f"**Profile:** `{profile_name}`\n**Home:** `{display}`"
    return f"**Profile:** default\n**Home:** `{display}`"


def _handle_provider(_args: str) -> str:
    """Return the current provider plus the list of available providers.

    Mirrors the read-only branch of
    :meth:`GatewayRouter._handle_provider_command`.  Reads config.yaml
    and the provider registry directly — no router dependency.
    """
    import yaml

    from hermes_cli.models import _PROVIDER_LABELS, list_available_providers, normalize_provider

    # Resolve current provider from config
    current_provider = "openrouter"
    model_cfg: dict = {}
    try:
        from hermes_constants import get_hermes_home
        config_path = get_hermes_home() / "config.yaml"
        if config_path.exists():
            with open(config_path, encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
            raw_model_cfg = cfg.get("model", {})
            if isinstance(raw_model_cfg, dict):
                model_cfg = raw_model_cfg
                current_provider = model_cfg.get("provider", current_provider)
    except Exception:
        pass

    current_provider = normalize_provider(current_provider)
    if current_provider == "auto":
        try:
            from hermes_cli.auth import resolve_provider as _resolve_provider
            current_provider = _resolve_provider(current_provider)
        except Exception:
            current_provider = "openrouter"

    # Detect custom endpoint from config base_url
    if current_provider == "openrouter":
        cfg_base = model_cfg.get("base_url", "") if isinstance(model_cfg, dict) else ""
        if cfg_base and "openrouter.ai" not in cfg_base:
            current_provider = "custom"

    current_label = _PROVIDER_LABELS.get(current_provider, current_provider)

    lines = [
        f"**Current provider:** {current_label} (`{current_provider}`)",
        "",
        "**Available providers:**",
    ]
    try:
        providers = list_available_providers()
    except Exception:
        providers = []
    for provider in providers:
        marker = " ← active" if provider["id"] == current_provider else ""
        auth = "authenticated" if provider["authenticated"] else "not authenticated"
        aliases = (
            f"  _(also: {', '.join(provider['aliases'])})_"
            if provider["aliases"]
            else ""
        )
        lines.append(
            f"- `{provider['id']}` — {provider['label']} "
            f"({auth}){aliases}{marker}"
        )
    lines.append("")
    lines.append(
        "Note: provider switching via `/model` is not available on "
        "the stateless `/v1/runs` and `/v1/chat/completions` endpoints."
    )
    return "\n".join(lines)


_STATELESS_HANDLERS: Dict[str, Callable[[str], str]] = {
    "help": _handle_help,
    "commands": _handle_commands,
    "profile": _handle_profile,
    "provider": _handle_provider,
}


# ---------------------------------------------------------------------------
# Stateful-command decline notice
# ---------------------------------------------------------------------------


def _stateful_decline_notice(user_token: str, canonical_name: str) -> str:
    """Build the decline notice for a stateful command.

    *user_token* is whatever the caller actually typed (minus the leading
    slash) so aliases like ``/reset`` echo back correctly.  *canonical_name*
    is the resolved command name — useful for the footer pointer to
    ``/help``.
    """
    displayed = user_token or canonical_name
    return (
        f"The `/{displayed}` command requires a persistent session and "
        f"isn't available on the stateless `/v1/runs` and "
        f"`/v1/chat/completions` endpoints.  Use a channel with session "
        f"state (CLI, Discord, Telegram, Slack) for commands that mutate "
        f"per-session configuration.\n\n"
        f"For the commands that *are* supported here, type `/help`."
    )


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def maybe_handle_gateway_command(user_message: str) -> Optional[SlashCommandResult]:
    """Intercept a gateway slash command if the message starts with one.

    Returns a :class:`SlashCommandResult` if the first whitespace-delimited
    token of *user_message* starts with ``/`` and resolves to a
    gateway-available command in :data:`hermes_cli.commands.COMMAND_REGISTRY`.
    Returns ``None`` otherwise — the caller then runs the normal LLM path.

    ``None`` is also returned when:

    * *user_message* is empty or does not start with ``/``.
    * The first token is an unknown command (pass it to the LLM; it may
      be a skill name the LLM recognizes).
    * The command is strictly ``cli_only`` (``/clear``, ``/history``,
      ``/save``, …) and has no ``gateway_config_gate`` — these are never
      exposed to gateway adapters.
    * Anything inside the preprocessor raises.  The exception is logged
      at WARNING and the caller falls through to the LLM path.  A bug in
      the preprocessor must never take down a normal chat request.
    """
    try:
        if not user_message:
            return None
        stripped = user_message.lstrip()
        if not stripped.startswith("/"):
            return None

        # Split into "/name" + remainder.  The remainder is passed to
        # stateless handlers as-is so they can do their own arg parsing.
        parts = stripped.split(None, 1)
        first_token = parts[0][1:]  # drop the leading slash
        args = parts[1] if len(parts) > 1 else ""

        if not first_token:
            return None

        cmd = resolve_command(first_token)
        if cmd is None:
            # Unknown command — could be a skill or plugin command the
            # LLM can handle.  Fall through silently.
            return None

        # Strictly CLI-only commands are never reachable via the gateway.
        # Config-gated commands (``cli_only=True`` plus a
        # ``gateway_config_gate``) are still gateway-reachable when the
        # gate is set, so let them through to the stateful-decline path.
        if cmd.cli_only and not cmd.gateway_config_gate:
            return None

        handler = _STATELESS_HANDLERS.get(cmd.name)
        if handler is not None:
            return SlashCommandResult(text=handler(args), command=cmd.name)

        return SlashCommandResult(
            text=_stateful_decline_notice(first_token, cmd.name),
            command=cmd.name,
        )
    except Exception as exc:
        logger.warning(
            "api_server slash preprocessor failed for %r — falling "
            "through to LLM path: %s",
            user_message[:80],
            exc,
        )
        return None
