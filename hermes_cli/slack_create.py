"""``hermes slack create`` — create the Hermes Slack app from one App Configuration token, and keep
its manifest in sync on later runs.

First run: the user pastes the *refresh* token from https://api.slack.com/apps ("Your App
Configuration Tokens"). Hermes rotates it (``tooling.tokens.rotate``), creates the app with
``apps.manifest.create`` from :func:`hermes_cli.slack_cli._build_full_manifest`, stores the app id
in ``config.yaml`` (``slack.app_id``) and the rotated refresh token in the profile ``.env``. Slack
cannot mint the bot (``xoxb-``) or Socket Mode (``xapp-``) tokens over this API, so the command
prints the two pages to visit and prompts for each, masked.

Later runs rotate the stored refresh token, export the live manifest and ``apps.manifest.update``
it with the current slash list (``COMMAND_REGISTRY``) plus any bot scopes/events Hermes now needs,
leaving the user's own edits (name, description, extra scopes) intact.
"""
from __future__ import annotations

import copy
import json
from typing import Any

import httpx

from hermes_cli.cli_output import (
    print_error, print_info, print_success, print_warning, prompt, prompt_yes_no)
from hermes_cli.config import (
    get_env_value, read_raw_config, require_env_writable, save_config, save_env_value_secure)
from hermes_cli.slack_cli import _build_full_manifest

SLACK_API_BASE = "https://slack.com/api/"
REFRESH_TOKEN_ENV = "SLACK_CONFIG_REFRESH_TOKEN"
_CONFIG_TOKENS_URL = "https://api.slack.com/apps"
_TIMEOUT_S = 30.0
# Manifest feature key -> messaging experience accepted by ``_build_full_manifest``.
_EXPERIENCE_BY_FEATURE = (("agent_view", "agent"), ("assistant_view", "assistant"))


class SlackApiError(RuntimeError):
    """A Slack Web API call returned ``ok: false`` (message carries Slack's error + pointers)."""


def _call(method: str, **fields: str) -> dict[str, Any]:
    """POST form fields to a Slack Web API method; raise :class:`SlackApiError` unless ``ok``."""
    response = httpx.post(f"{SLACK_API_BASE}{method}", data=fields, timeout=_TIMEOUT_S)
    response.raise_for_status()
    body = response.json()
    if body.get("ok"):
        return body
    details = "; ".join(
        f"{e.get('pointer', '?')}: {e.get('message', '')}" for e in body.get("errors") or [])
    raise SlackApiError(f"{method}: {body.get('error', 'unknown_error')}"
                        + (f" ({details})" if details else ""))


def _ask_refresh_token() -> str:
    print_info("Hermes needs a Slack App Configuration token to create the app for you.")
    print_info(f"  1. Open {_CONFIG_TOKENS_URL} and scroll to \"Your App Configuration Tokens\".")
    print_info("  2. Generate Token → pick your workspace → copy the *Refresh Token* (xoxe-...).")
    return prompt("Slack App Configuration refresh token (xoxe-...)", password=True)


def _rotate(refresh_token: str) -> str:
    """Exchange the refresh token for a 12-hour config token; persist the new refresh token.

    Slack invalidates the old refresh token on rotation, so the new one is saved before anything
    else can fail. The short-lived access token is used in memory only."""
    body = _call("tooling.tokens.rotate", refresh_token=refresh_token)
    save_env_value_secure(REFRESH_TOKEN_ENV, body["refresh_token"])
    return body["token"]


def _stored_app_id() -> str:
    return str((read_raw_config().get("slack") or {}).get("app_id") or "").strip()


def _save_app_id(app_id: str) -> None:
    config = read_raw_config()
    slack = config.get("slack")
    config["slack"] = {**(slack if isinstance(slack, dict) else {}), "app_id": app_id}
    save_config(config)


def _experience_of(manifest: dict[str, Any]) -> str:
    features = manifest.get("features") or {}
    return next((exp for key, exp in _EXPERIENCE_BY_FEATURE if key in features), "none")


def _union(existing: list[str] | None, wanted: list[str]) -> tuple[list[str], list[str]]:
    have = list(existing or [])
    added = sorted(set(wanted) - set(have))
    return sorted(set(have) | set(wanted)), added


def synced_manifest(current: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """``current`` with Hermes' slash list replaced and missing bot scopes/events added.

    Returns ``(manifest, added)`` where ``added`` lists new scopes/events (non-empty means Slack
    will ask for a reinstall). Everything else the user configured is preserved."""
    fresh = _build_full_manifest(
        "Hermes", "", messaging_experience=_experience_of(current))
    merged = copy.deepcopy(current)
    merged.setdefault("features", {})["slash_commands"] = fresh["features"]["slash_commands"]
    scopes = merged.setdefault("oauth_config", {}).setdefault("scopes", {})
    scopes["bot"], added_scopes = _union(scopes.get("bot"), fresh["oauth_config"]["scopes"]["bot"])
    events = merged.setdefault("settings", {}).setdefault("event_subscriptions", {})
    events["bot_events"], added_events = _union(
        events.get("bot_events"), fresh["settings"]["event_subscriptions"]["bot_events"])
    return merged, [*(f"scope {s}" for s in added_scopes), *(f"event {e}" for e in added_events)]


def _sync_existing(token: str, app_id: str) -> None:
    current = _call("apps.manifest.export", token=token, app_id=app_id)["manifest"]
    manifest, added = synced_manifest(current)
    _call("apps.manifest.update", token=token, app_id=app_id, manifest=json.dumps(manifest))
    before = len((current.get("features") or {}).get("slash_commands") or [])
    after = len(manifest["features"]["slash_commands"])
    print_success(f"Slack app {app_id} manifest updated: {before} → {after} slash commands.")
    if added:
        print_warning("Added " + ", ".join(added) + ". Slack will ask you to reinstall the app: "
                      f"https://api.slack.com/apps/{app_id}/install-on-team")


def _create_new(token: str, args) -> str:
    manifest = _build_full_manifest(
        getattr(args, "name", None) or "Hermes",
        getattr(args, "description", None) or "Your Hermes agent on Slack",
        messaging_experience=getattr(args, "messaging", None) or "agent")
    app_id = _call("apps.manifest.create", token=token, manifest=json.dumps(manifest))["app_id"]
    _save_app_id(app_id)
    print_success(f"Created Slack app {app_id} (slack.app_id saved to config.yaml).")
    return app_id


def _prompt_secret_into_env(env_key: str, prefix: str, label: str, steps: tuple[str, ...]) -> None:
    if get_env_value(env_key):
        print_info(f"{env_key} is already set; keeping it.")
        return
    for line in steps:
        print_info(line)
    value = prompt(f"{label} ({prefix}...; Enter to skip)", password=True)
    if not value:
        print_warning(f"Skipped {env_key}; set it later with `hermes setup gateway`.")
        return
    if not value.startswith(prefix):
        print_warning(f"That does not look like a {prefix} token; not saved.")
        return
    save_env_value_secure(env_key, value)
    print_success(f"{env_key} saved to .env.")


def _collect_install_tokens(app_id: str) -> None:
    page = f"https://api.slack.com/apps/{app_id}"
    _prompt_secret_into_env("SLACK_BOT_TOKEN", "xoxb-", "Slack Bot Token", (
        "Install the app (Slack requires this click):",
        f"  {page}/install-on-team → Install to Workspace → Allow,",
        "  then copy the Bot User OAuth Token."))
    _prompt_secret_into_env("SLACK_APP_TOKEN", "xapp-", "Slack App Token", (
        "One manual step left for Socket Mode (Slack has no API for app-level tokens):",
        f"  {page}/general → App-Level Tokens → Generate Token and Scopes,",
        "  add the connections:write scope, Generate, then copy the token."))


def _confirm_second_app() -> bool:
    if not get_env_value("SLACK_BOT_TOKEN"):
        return True
    print_warning("SLACK_BOT_TOKEN is already set, so you probably have a Slack app. To keep that "
                  "app in sync instead, re-run with `--app-id A...` (Basic Information → App ID).")
    return prompt_yes_no("Create a NEW Slack app anyway?", False)


def slack_create_command(args) -> int:
    """Create or re-sync the Hermes Slack app (flags documented in ``hermes_cli/subcommands/slack.py``)."""
    try:
        require_env_writable(REFRESH_TOKEN_ENV, "set")
    except ValueError as exc:
        print_error(str(exc))
        return 1
    app_id = (getattr(args, "app_id", None) or "").strip() or _stored_app_id()
    if not app_id and not _confirm_second_app():
        return 1
    refresh_token = get_env_value(REFRESH_TOKEN_ENV) or _ask_refresh_token()
    if not refresh_token:
        print_error("No App Configuration token given; nothing done.")
        return 1
    try:
        token = _rotate(refresh_token)
        if app_id:
            _sync_existing(token, app_id)
            if app_id != _stored_app_id():
                _save_app_id(app_id)
        else:
            app_id = _create_new(token, args)
    except (SlackApiError, httpx.HTTPError) as exc:
        print_error(f"Slack API call failed: {exc}")
        return 1
    _collect_install_tokens(app_id)
    print_info("Next: run `hermes setup gateway` → Slack to allow your Slack member ID and set a "
               "home channel, then `/invite` the bot to channels it should read.")
    print_info("Re-run `hermes slack create` after `hermes update` to sync new slash commands.")
    return 0
