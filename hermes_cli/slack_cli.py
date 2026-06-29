"""``hermes slack ...`` CLI subcommands.

Subcommands:

* ``hermes slack manifest`` — generate the Slack app manifest (JSON or YAML)
  registering every gateway command as a native Slack slash (``/btw``,
  ``/stop``, ``/model``, …) so users get the same first-class slash UX
  Discord and Telegram already have.
* ``hermes slack channels`` — list every channel the bot token can see and
  report which ones the bot is/isn't a member of (the "all channels" gap).
* ``hermes slack invite`` — add the bot to channels so it can read & post.
  Public channels are joined directly with the bot token
  (``conversations.join``); private channels need a human ``/invite`` or a
  user token (``conversations.invite``).

Typical workflow::

    $ hermes slack manifest --yaml > slack-manifest.yaml   # paste at api.slack.com
    $ hermes slack channels                                 # see membership gaps
    $ hermes slack invite --all                             # join all public channels

Then paste the manifest into the Slack app config (Features → App
Manifest → Edit) and click Save. Slack diffs the manifest and prompts
for reinstall when scopes/commands change.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _build_full_manifest(bot_name: str, bot_description: str) -> dict:
    """Build a full Slack manifest merging display info + our slash list.

    The slash-command list is always generated from ``COMMAND_REGISTRY`` so
    it stays in sync with the rest of Hermes. Other manifest sections
    (display info, OAuth scopes, socket mode) are set to sensible defaults
    for a Hermes deployment — users can tweak them in the Slack UI after
    pasting.
    """
    from hermes_cli.commands import slack_app_manifest

    partial = slack_app_manifest()
    slashes = partial["features"]["slash_commands"]

    return {
        "_metadata": {
            "major_version": 1,
            "minor_version": 1,
        },
        "display_information": {
            "name": bot_name[:35],
            "description": (bot_description or "Your Hermes agent on Slack")[:140],
            "background_color": "#1a1a2e",
        },
        "features": {
            "app_home": {
                "home_tab_enabled": False,
                "messages_tab_enabled": True,
                "messages_tab_read_only_enabled": False,
            },
            "bot_user": {
                "display_name": bot_name[:80],
                "always_online": True,
            },
            "slash_commands": slashes,
            "assistant_view": {
                "assistant_description": "Chat with Hermes in threads and DMs.",
            },
        },
        "oauth_config": {
            "scopes": {
                "bot": [
                    "app_mentions:read",
                    "assistant:write",
                    "channels:history",
                    "channels:join",
                    "channels:read",
                    "chat:write",
                    "commands",
                    "files:read",
                    "files:write",
                    "groups:history",
                    "groups:read",
                    "im:history",
                    "im:read",
                    "im:write",
                    "metadata.message:read",
                    "users:read",
                ],
            },
        },
        "settings": {
            "event_subscriptions": {
                "bot_events": [
                    "app_mention",
                    "assistant_thread_context_changed",
                    "assistant_thread_started",
                    "message.channels",
                    "message.groups",
                    "message.im",
                    "message_metadata_posted",
                    "message_metadata_updated",
                ],
                "metadata_subscriptions": [
                    {"app_id": "*", "event_type": "messages:hermes"},
                ],
            },
            "interactivity": {
                "is_enabled": True,
            },
            "org_deploy_enabled": False,
            "socket_mode_enabled": True,
            "token_rotation_enabled": False,
        },
    }


def _render_manifest(manifest, as_yaml: bool) -> str:
    """Serialize a manifest dict/list to JSON (default) or YAML text.

    Slack's "Create an app → From a manifest" flow accepts both JSON and
    YAML; YAML is friendlier to paste/diff, so we offer it via ``--yaml``.
    """
    if as_yaml:
        import yaml  # pyyaml is a hard dependency (see pyproject.toml)

        return yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True, default_flow_style=False)
    return json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"


def slack_manifest_command(args) -> int:
    """Print or write a Slack app manifest (JSON or YAML).

    Flags (all parsed in ``hermes_cli/main.py``):
      --write [PATH]  Write to file instead of stdout (default path:
                      ``$HERMES_HOME/slack-manifest.{json,yaml}``)
      --yaml          Emit YAML instead of JSON (Slack accepts both)
      --name NAME     Override the bot display name (default: "Hermes")
      --description DESC  Override the bot description
      --slashes-only  Emit only the ``features.slash_commands`` array (for
                      merging into an existing manifest manually)
    """
    name = getattr(args, "name", None) or "Hermes"
    description = getattr(args, "description", None) or "Your Hermes agent on Slack"
    as_yaml = bool(getattr(args, "yaml", False))

    if getattr(args, "slashes_only", False):
        from hermes_cli.commands import slack_app_manifest

        manifest = slack_app_manifest()["features"]["slash_commands"]
    else:
        manifest = _build_full_manifest(name, description)

    payload = _render_manifest(manifest, as_yaml)
    ext = "yaml" if as_yaml else "json"

    write_target = getattr(args, "write", None)
    if write_target is not None:
        if isinstance(write_target, bool) and write_target:
            # --write with no value → default location
            try:
                from hermes_constants import get_hermes_home

                target = Path(get_hermes_home()) / f"slack-manifest.{ext}"
            except Exception:
                target = Path(os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")) / f"slack-manifest.{ext}"
        else:
            target = Path(write_target).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload, encoding="utf-8")
        print(f"Slack manifest written to: {target}", file=sys.stderr)
        print(
            "\nNext steps:\n"
            "  1. Open https://api.slack.com/apps and pick your Hermes app\n"
            "     (or create a new one: Create New App → From an app manifest).\n"
            f"  2. Features → App Manifest → paste the contents of\n"
            f"     {target}\n"
            "  3. Save; Slack will prompt to reinstall the app if scopes or\n"
            "     slash commands changed.\n"
            "  4. Make sure Socket Mode is enabled and you have a bot token\n"
            "     (xoxb-...) and app token (xapp-...) configured via\n"
            "     `hermes setup`.\n"
            "  5. Add the bot to your channels:  hermes slack invite --all\n",
            file=sys.stderr,
        )
    else:
        sys.stdout.write(payload)
    return 0


# ---------------------------------------------------------------------------
# Channel membership: list + invite ("all channels" reproducibly)
# ---------------------------------------------------------------------------
#
# A Slack *bot* only sees and posts in channels it is a *member* of. There is
# no API to bulk-add a bot to every channel in one call, so "wire to all
# channels" decomposes into two scriptable steps:
#
#   1. list every channel the token can see + whether the bot is a member;
#   2. add the bot to the ones it's missing.
#
# Public channels: the bot can add *itself* with ``conversations.join`` (needs
# the ``channels:join`` scope, included in the generated manifest).
# Private channels: a bot CANNOT self-join. Either a human runs ``/invite
# @<bot>`` from inside the channel, or you pass a *user* token (xoxp-, from
# someone already in the channel) and we call ``conversations.invite``.


def _load_bot_token() -> Optional[str]:
    """Return the Slack bot token from the environment / ``~/.hermes/.env``.

    Mirrors how the gateway resolves it: ``SLACK_BOT_TOKEN`` may hold a
    single token or a comma-separated list for multi-workspace setups; we
    use the first one for these single-workspace helper commands.
    """
    token = os.getenv("SLACK_BOT_TOKEN")
    if not token:
        try:
            from hermes_cli.config import load_env

            token = load_env().get("SLACK_BOT_TOKEN")
        except Exception:
            token = None
    if not token:
        return None
    # Multi-workspace: first token is the primary (matches SlackAdapter).
    first = token.split(",")[0].strip()
    return first or None


def _make_web_client(token: str):
    """Build a synchronous ``slack_sdk.WebClient``, lazy-installing the SDK.

    Reuses the same proxy resolution the gateway adapter uses so these
    helpers work behind a corporate proxy too.
    """
    try:
        from slack_sdk import WebClient
    except ImportError:
        from tools.lazy_deps import ensure

        ensure("platform.slack", prompt=False)
        from slack_sdk import WebClient

    proxy = None
    try:
        from gateway.platforms.slack import _resolve_slack_proxy_url

        proxy = _resolve_slack_proxy_url()
    except Exception:
        proxy = None

    return WebClient(token=token, proxy=proxy)


def _list_channels(client, *, include_private: bool = True) -> List[Dict[str, Any]]:
    """Return all conversations the token can enumerate, paginated.

    Each entry is normalized to ``{id, name, is_member, is_private,
    is_archived}``. Uses ``conversations.list`` with cursor pagination.
    """
    types = "public_channel"
    if include_private:
        types += ",private_channel"

    channels: List[Dict[str, Any]] = []
    cursor: Optional[str] = None
    while True:
        kwargs: Dict[str, Any] = {"types": types, "limit": 1000, "exclude_archived": False}
        if cursor:
            kwargs["cursor"] = cursor
        resp = client.conversations_list(**kwargs)
        for ch in resp.get("channels", []) or []:
            channels.append(
                {
                    "id": ch.get("id", ""),
                    "name": ch.get("name", ch.get("id", "")),
                    "is_member": bool(ch.get("is_member", False)),
                    "is_private": bool(ch.get("is_private", False)),
                    "is_archived": bool(ch.get("is_archived", False)),
                }
            )
        cursor = (resp.get("response_metadata") or {}).get("next_cursor") or ""
        if not cursor:
            break
    channels.sort(key=lambda c: (c["is_private"], c["name"].lower()))
    return channels


def _resolve_bot_user_id(client) -> Optional[str]:
    """Return the bot's own user id via ``auth.test`` (needed for invites)."""
    try:
        resp = client.auth_test()
        return resp.get("user_id")
    except Exception:
        return None


def slack_channels_command(args) -> int:
    """List channels and report which ones the bot is / isn't a member of."""
    token = _load_bot_token()
    if not token:
        print(
            "SLACK_BOT_TOKEN not set. Add it to ~/.hermes/.env or run `hermes setup`.",
            file=sys.stderr,
        )
        return 1

    include_private = not getattr(args, "no_private", False)
    as_json = bool(getattr(args, "json", False))

    try:
        client = _make_web_client(token)
        channels = _list_channels(client, include_private=include_private)
    except Exception as e:
        print(f"Failed to list Slack channels: {e}", file=sys.stderr)
        return 1

    live = [c for c in channels if not c["is_archived"]]
    member = [c for c in live if c["is_member"]]
    missing = [c for c in live if not c["is_member"]]

    if as_json:
        json.dump(
            {
                "member": member,
                "missing": missing,
                "archived": [c for c in channels if c["is_archived"]],
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
        return 0

    def _fmt(c: Dict[str, Any]) -> str:
        kind = "private" if c["is_private"] else "public"
        return f"  #{c['name']}  ({c['id']}, {kind})"

    print(f"Bot is a MEMBER of {len(member)} channel(s):")
    for c in member:
        print(_fmt(c))
    print(f"\nBot is NOT a member of {len(missing)} channel(s):")
    for c in missing:
        print(_fmt(c))

    pub_missing = [c for c in missing if not c["is_private"]]
    priv_missing = [c for c in missing if c["is_private"]]
    if pub_missing:
        print(
            f"\n{len(pub_missing)} public channel(s) can be joined automatically:\n"
            "  hermes slack invite --all"
        )
    if priv_missing:
        print(
            f"\n{len(priv_missing)} private channel(s) need a manual invite "
            "(/invite @<bot> from inside each), or pass a user token:\n"
            "  hermes slack invite --all --user-token xoxp-..."
        )
    return 0


def _invite_bot_to_channel(
    client, channel: Dict[str, Any], bot_user_id: Optional[str], *, user_client=None
) -> Tuple[bool, str]:
    """Add the bot to a single channel. Returns ``(ok, message)``.

    Strategy:
      * Public channel → ``conversations.join`` with the bot token (self-join).
      * Private channel → ``conversations.invite`` with a *user* token if one
        was provided; otherwise it cannot be automated (report as skipped).

    Error handling is SDK-agnostic: any exception carrying a ``response``
    mapping with an ``error`` key (as ``slack_sdk.errors.SlackApiError``
    does) is inspected so an ``already_in_channel`` result still counts as
    success.
    """
    cid = channel["id"]
    name = channel["name"]
    try:
        if not channel["is_private"]:
            client.conversations_join(channel=cid)
            return True, f"joined #{name}"
        # Private channel.
        if user_client is not None and bot_user_id:
            user_client.conversations_invite(channel=cid, users=bot_user_id)
            return True, f"invited bot to private #{name} (user token)"
        return False, f"skipped private #{name} — needs /invite @<bot> or --user-token"
    except Exception as e:
        response = getattr(e, "response", None)
        err = ""
        if isinstance(response, dict):
            err = response.get("error", "")
        err = err or str(e)
        if err == "already_in_channel":
            return True, f"already in #{name}"
        return False, f"failed #{name}: {err}"


def slack_invite_command(args) -> int:
    """Add the bot to channels so it can read & post (the "all channels" step).

    Flags:
      --all                 Target every channel the bot isn't already in.
      --channel NAME/ID     Target a specific channel (repeatable).
      --no-private          Only consider public channels.
      --user-token TOKEN    User token (xoxp-) used to invite the bot to
                            private channels (falls back to SLACK_USER_TOKEN).
      --dry-run             Show what would happen without calling Slack.
    """
    token = _load_bot_token()
    if not token:
        print(
            "SLACK_BOT_TOKEN not set. Add it to ~/.hermes/.env or run `hermes setup`.",
            file=sys.stderr,
        )
        return 1

    target_all = bool(getattr(args, "all", False))
    requested = [c.lstrip("#") for c in (getattr(args, "channel", None) or [])]
    if not target_all and not requested:
        print(
            "Nothing to do: pass --all or one or more --channel NAME/ID.",
            file=sys.stderr,
        )
        return 1

    include_private = not getattr(args, "no_private", False)
    dry_run = bool(getattr(args, "dry_run", False))
    user_token = getattr(args, "user_token", None) or os.getenv("SLACK_USER_TOKEN")

    try:
        client = _make_web_client(token)
        channels = _list_channels(client, include_private=include_private)
    except Exception as e:
        print(f"Failed to list Slack channels: {e}", file=sys.stderr)
        return 1

    by_id = {c["id"]: c for c in channels}
    by_name = {c["name"].lower(): c for c in channels}

    if target_all:
        targets = [c for c in channels if not c["is_archived"] and not c["is_member"]]
    else:
        targets = []
        for token_str in requested:
            ch = by_id.get(token_str) or by_name.get(token_str.lower())
            if ch is None:
                print(f"  ⚠ channel not found: {token_str}", file=sys.stderr)
                continue
            targets.append(ch)

    if not targets:
        print("All targeted channels already have the bot. Nothing to do.")
        return 0

    bot_user_id = _resolve_bot_user_id(client)
    user_client = None
    if user_token:
        try:
            user_client = _make_web_client(user_token)
        except Exception as e:
            print(f"  ⚠ could not build user-token client: {e}", file=sys.stderr)

    succeeded = 0
    failed = 0
    for ch in targets:
        if dry_run:
            action = "join" if not ch["is_private"] else (
                "invite (user token)" if user_client else "skip (needs invite)"
            )
            print(f"  [dry-run] #{ch['name']} → {action}")
            continue
        ok, msg = _invite_bot_to_channel(
            client, ch, bot_user_id, user_client=user_client
        )
        print(("  ✓ " if ok else "  ✗ ") + msg)
        if ok:
            succeeded += 1
        else:
            failed += 1

    if not dry_run:
        print(f"\nDone: {succeeded} added/confirmed, {failed} need attention.")
    return 0 if failed == 0 else 1
