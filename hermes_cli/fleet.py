"""``hermes fleet`` — setup + status + roster management for Telegram Fleet.

Closes the setup gap: the swarm tools and the orchestration skill ship with
the agent, but using the Telegram variant requires a manager bot configured
in BotFather and the token written into ~/.hermes/.env.  This module turns
that into a one-command experience.

Subcommands:

* ``hermes fleet setup`` — interactive wizard.  Validates the manager bot
  token (calling ``getMe``), confirms ``can_manage_bots`` is enabled, saves
  the manager username to the roster.  Bails out with concrete instructions
  when something's wrong (token invalid, Bot Manager Mode disabled, etc.).
* ``hermes fleet status`` — show what's configured: token presence, roster
  state, active worker count.  Doctor-style output.
* ``hermes fleet add <username> [--persona]`` — mint a single deep link the
  user taps to confirm a new child bot.
* ``hermes fleet list`` — pretty-print the roster (tokens redacted).

The interactive UI is intentionally simple — it uses ``input()`` and
``getpass()`` so it works in plain TTY, SSH, and CI alike, matching how
``hermes setup`` already works in this repo.
"""

from __future__ import annotations

import argparse
import getpass
import logging
import os
import sys
from typing import Optional

logger = logging.getLogger(__name__)


def _isatty() -> bool:
    return sys.stdout.isatty()


_RST = "\033[0m" if _isatty() else ""
_BOLD = "\033[1m" if _isatty() else ""
_DIM = "\033[2m" if _isatty() else ""
_OK = "\033[32m" if _isatty() else ""
_WARN = "\033[33m" if _isatty() else ""
_ERR = "\033[31m" if _isatty() else ""
_INFO = "\033[36m" if _isatty() else ""


def _ok(msg: str) -> None:
    print(f"{_OK}\u2713{_RST} {msg}")


def _warn(msg: str) -> None:
    print(f"{_WARN}\u26a0{_RST} {msg}")


def _err(msg: str) -> None:
    print(f"{_ERR}\u2717{_RST} {msg}")


def _info(msg: str) -> None:
    print(f"{_INFO}\u2139{_RST} {msg}")


def _heading(msg: str) -> None:
    print(f"\n{_BOLD}{msg}{_RST}")


def cmd_fleet(args: argparse.Namespace) -> int:
    """Dispatch ``hermes fleet`` subcommands."""
    sub = getattr(args, "fleet_command", None)
    if sub == "setup":
        return _cmd_setup(args)
    if sub == "status":
        return _cmd_status(args)
    if sub == "list":
        return _cmd_list(args)
    if sub == "add":
        return _cmd_add(args)
    if sub == "adopt":
        return _cmd_adopt(args)
    if sub == "connect":
        return _cmd_connect(args)
    return _cmd_status(args)


def add_fleet_parser(subparsers) -> None:
    """Register the ``hermes fleet`` subparser tree."""
    fleet_parser = subparsers.add_parser(
        "fleet",
        help="Telegram fleet management (setup, status, roster)",
        description=(
            "Manage the Telegram bot fleet used by `telegram_orchestrate_swarm`.  "
            "The in-process `hermes_swarm` tool needs no setup; this is only "
            "for the visible Telegram variant where each worker posts as a "
            "named bot."
        ),
    )
    fleet_subparsers = fleet_parser.add_subparsers(dest="fleet_command")

    fleet_subparsers.add_parser(
        "setup",
        help="Interactive setup \u2014 validate manager token, confirm Bot Manager Mode",
    )
    fleet_subparsers.add_parser(
        "status",
        help="Show what's configured (token, roster, active workers)",
    )
    fleet_subparsers.add_parser(
        "list",
        help="List the roster (tokens redacted)",
    )

    add_parser = fleet_subparsers.add_parser(
        "add",
        help="Mint a deep link for one new child bot \u2014 user taps to confirm",
    )
    add_parser.add_argument(
        "username",
        help=(
            "Suggested username for the new bot (must end in 'bot' per "
            "Telegram rules; suffix is auto-added if missing)."
        ),
    )
    add_parser.add_argument(
        "--persona",
        default="",
        help="Free-text persona/role description for this worker.",
    )
    add_parser.add_argument(
        "--name",
        default=None,
        help="Display name shown in Telegram (defaults to persona/username).",
    )
    add_parser.add_argument(
        "--wait",
        type=int,
        default=180,
        help=(
            "After printing the deep link, poll for up to N seconds "
            "waiting for the user to tap (default 180 = 3 minutes).  Pass "
            "0 to skip polling and exit immediately."
        ),
    )

    adopt_parser = fleet_subparsers.add_parser(
        "adopt",
        help=(
            "Adopt an EXISTING bot into the fleet by token.  Use this when "
            "you already created the bot manually in @BotFather and "
            "`hermes fleet add` says the username is taken."
        ),
        description=(
            "Adds an already-existing Telegram bot to the fleet roster by "
            "validating its token (calls getMe) and writing it as active.  "
            "Bypasses the Managed Bots deep-link flow.  Trade-off: bots "
            "adopted this way cannot be rotated via replaceManagedBotToken; "
            "rotation means generating a fresh token in BotFather and "
            "re-running `hermes fleet adopt`."
        ),
    )
    adopt_parser.add_argument(
        "--token",
        default=None,
        help=(
            "Bot API token.  If omitted, you'll be prompted (input hidden)."
        ),
    )
    adopt_parser.add_argument(
        "--persona",
        default="",
        help="Free-text persona/role description for this worker.",
    )
    adopt_parser.add_argument(
        "--model",
        default=None,
        help="Optional model override for tasks routed through this bot.",
    )
    adopt_parser.add_argument(
        "--toolset",
        default=None,
        help=(
            "Optional toolset whitelist, comma-separated "
            "(e.g. 'web,file,terminal')."
        ),
    )

    connect_parser = fleet_subparsers.add_parser(
        "connect",
        help=(
            "Drain pending managed_bot updates from Telegram \u2014 promotes "
            "any pending entries whose deep link the user has tapped to "
            "active.  Run after `hermes fleet add` if you skipped --wait."
        ),
    )
    connect_parser.add_argument(
        "--wait",
        type=int,
        default=0,
        help=(
            "Long-poll for up to N seconds waiting for new updates.  "
            "Default 0 = non-blocking drain of already-queued updates."
        ),
    )

    fleet_parser.set_defaults(func=cmd_fleet)


def _resolve_token(*, prompt_if_missing: bool) -> Optional[str]:
    """Find the manager token from env, prompting interactively if asked."""
    token = (os.environ.get("TELEGRAM_FLEET_MANAGER_TOKEN") or "").strip()
    if token:
        return token
    if not prompt_if_missing:
        return None
    if not sys.stdin.isatty():
        _err("TELEGRAM_FLEET_MANAGER_TOKEN is not set and stdin is not a TTY.")
        return None
    print()
    _info(
        "We'll need a manager bot token.  Create one in @BotFather "
        "(/newbot), then enable Bot Manager Mode in BotFather's MiniApp.  "
        "The token looks like '12345:ABCdefGHIjklMNO...'."
    )
    try:
        token = getpass.getpass("Manager bot token (input hidden): ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    return token or None


def _save_token_to_env(token: str) -> bool:
    """Append TELEGRAM_FLEET_MANAGER_TOKEN to ~/.hermes/.env if not already set."""
    from hermes_constants import get_hermes_home

    env_path = get_hermes_home() / ".env"
    env_path.parent.mkdir(parents=True, exist_ok=True)
    existing = ""
    if env_path.exists():
        try:
            existing = env_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            existing = env_path.read_text(encoding="latin-1")
        if "TELEGRAM_FLEET_MANAGER_TOKEN" in existing:
            return False
    line = f"TELEGRAM_FLEET_MANAGER_TOKEN={token}\n"
    if existing and not existing.endswith("\n"):
        line = "\n" + line
    with env_path.open("a", encoding="utf-8") as f:
        f.write(line)
    try:
        os.chmod(env_path, 0o600)
    except OSError:
        pass
    return True


def _cmd_setup(args: argparse.Namespace) -> int:
    from gateway.telegram_fleet.api import BotApiError, FleetApiClient
    from gateway.telegram_fleet.roster import load_roster, save_roster

    _heading("Telegram Fleet \u2014 setup")

    token = _resolve_token(prompt_if_missing=True)
    if not token:
        _err("No manager bot token provided.  Aborting.")
        _info(
            "You can re-run this later: `hermes fleet setup`.  Or skip the "
            "Telegram fleet entirely \u2014 the in-process `hermes_swarm` tool "
            "needs no setup."
        )
        return 1
    if ":" not in token or len(token) < 30:
        _err(f"Token doesn't look right (got {len(token)} chars; expected ~46).")
        return 1

    try:
        client = FleetApiClient(token)
        me = client.get_me()
    except BotApiError as e:
        _err(f"Telegram rejected the token: {e}")
        return 1
    except Exception as e:
        _err(f"Couldn't reach Telegram ({type(e).__name__}: {e}).")
        return 1

    username = (me.get("username") or "").lstrip("@")
    if not username:
        _err("Manager bot has no username.  Set one in @BotFather first.")
        return 1
    _ok(f"Manager bot identified: @{username}")

    if not me.get("can_manage_bots"):
        _err(
            "This bot doesn't have Bot Manager Mode enabled.  "
            "Open @BotFather \u2192 /mybots \u2192 select this bot \u2192 BotFather MiniApp \u2192 "
            "enable 'Bot Management Mode'.  Then re-run `hermes fleet setup`."
        )
        return 1
    _ok("Bot Manager Mode is enabled \u2014 token can mint child bots.")

    try:
        roster = load_roster()
    except Exception as e:
        _warn(f"Existing roster unreadable ({e}); starting fresh.")
        from gateway.telegram_fleet.roster import FleetRoster

        roster = FleetRoster()
    roster.manager_bot_username = username
    save_roster(roster)
    _ok("Saved roster to ~/.hermes/telegram_fleet.yaml (mode 0600).")

    if os.environ.get("TELEGRAM_FLEET_MANAGER_TOKEN") != token:
        os.environ["TELEGRAM_FLEET_MANAGER_TOKEN"] = token
    saved = _save_token_to_env(token)
    if saved:
        _ok("Saved TELEGRAM_FLEET_MANAGER_TOKEN to ~/.hermes/.env.")
    else:
        _info(
            "TELEGRAM_FLEET_MANAGER_TOKEN already in ~/.hermes/.env "
            "(left unchanged)."
        )

    _heading("Next steps")
    print(
        "  1. Add worker bots:  hermes fleet add hermes_research_bot --persona 'research lead'\n"
        "  2. Watch the roster: hermes fleet list\n"
        "  3. In a Hermes session: 'spin up a swarm to research X' \u2014 the\n"
        "     agent proposes the plan, you confirm via clarify, the swarm runs.\n"
    )
    print(
        f"{_DIM}Tip: the in-process `hermes_swarm` tool runs without any of "
        f"this \u2014 it's the default for swarm-shaped requests.{_RST}"
    )
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    from gateway.telegram_fleet.roster import load_roster

    _heading("Telegram Fleet \u2014 status")

    token = (os.environ.get("TELEGRAM_FLEET_MANAGER_TOKEN") or "").strip()
    if token:
        _ok(
            f"Manager token configured (TELEGRAM_FLEET_MANAGER_TOKEN, "
            f"{len(token)} chars)."
        )
    else:
        _warn("Manager token NOT set.  Run `hermes fleet setup` to configure.")

    try:
        roster = load_roster()
    except Exception as e:
        _err(f"Roster unreadable: {e}")
        return 1
    if roster.manager_bot_username:
        _ok(f"Manager bot: @{roster.manager_bot_username}")
    else:
        _info("No manager bot recorded in roster yet.")
    active = roster.active_children()
    pending = roster.pending_children()
    decommissioned = [c for c in roster.children if c.status == "decommissioned"]
    _info(
        f"Roster: {len(active)} active, {len(pending)} pending, "
        f"{len(decommissioned)} decommissioned (cap: {roster.max_size})."
    )
    if not token:
        return 1
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    from gateway.telegram_fleet.roster import load_roster

    try:
        roster = load_roster()
    except Exception as e:
        _err(f"Roster unreadable: {e}")
        return 1

    if not roster.children:
        _info("Roster is empty.  Add a worker with `hermes fleet add <username>`.")
        return 0

    _heading(f"Telegram Fleet \u2014 roster ({len(roster.children)} entries)")
    width = max((len(c.username) for c in roster.children), default=8) + 2
    for c in roster.children:
        status = c.status
        marker = {
            "active": _OK + "\u25cf",
            "pending": _WARN + "\u25cb",
            "decommissioned": _DIM + "\u00b7",
        }.get(status, "\u00b7")
        persona = c.persona or "\u2014"
        if len(persona) > 60:
            persona = persona[:57] + "\u2026"
        print(
            f"  {marker}{_RST} @{c.username:<{width}} {_DIM}{status:<14}{_RST} {persona}"
        )
    return 0


def _cmd_add(args: argparse.Namespace) -> int:
    from gateway.telegram_fleet import (
        FleetGuardrailError,
        SpawnApprovalRequired,
        get_coordinator,
    )
    from gateway.telegram_fleet.api import BotApiError

    suggested = (args.username or "").strip().lstrip("@")
    if not suggested:
        _err("username is required.")
        return 1
    if not suggested.lower().endswith("bot"):
        _info(f"Adding 'bot' suffix \u2192 {suggested}_bot")
        suggested = f"{suggested}_bot"

    if not (os.environ.get("TELEGRAM_FLEET_MANAGER_TOKEN") or "").strip():
        _err(
            "TELEGRAM_FLEET_MANAGER_TOKEN is not set.  Run "
            "`hermes fleet setup` first."
        )
        return 1

    coord = get_coordinator(refresh=True)
    try:
        result = coord.spawn_bot(
            suggested_username=suggested,
            persona=args.persona,
            display_name=args.name,
        )
    except SpawnApprovalRequired as e:
        _err(str(e))
        return 1
    except FleetGuardrailError as e:
        _err(str(e))
        return 1
    except BotApiError as e:
        _err(f"Telegram API error: {e}")
        return 1

    _ok(f"Pending entry minted for @{result.suggested_username}.")
    print()
    print(f"  {_BOLD}Tap this link in Telegram to confirm:{_RST}")
    print(f"  {_INFO}{result.deep_link}{_RST}")
    print()
    print(
        f"  {_DIM}This link CREATES a new bot under your manager's "
        f"control.  If Telegram says 'username already taken' when you "
        f"tap, the bot already exists \u2014 use "
        f"`hermes fleet adopt --token <its_token> --persona \"...\"` "
        f"instead.{_RST}"
    )
    print()

    wait_seconds = int(getattr(args, "wait", 0) or 0)
    if wait_seconds <= 0:
        _info(
            "Skipping wait.  After tapping the link, run `hermes fleet "
            "connect` (or restart the gateway) to absorb the bot."
        )
        return 0

    _info(
        f"Watching Telegram for up to {wait_seconds}s for your tap "
        f"(Ctrl-C to stop and continue manually with `hermes fleet connect`)."
    )
    import time

    target = result.suggested_username.lower()
    deadline = time.monotonic() + wait_seconds
    poll_chunk = 30  # Telegram caps long-poll at ~50; 30s is a safe default.
    try:
        while time.monotonic() < deadline:
            remaining = max(1, int(deadline - time.monotonic()))
            this_timeout = min(poll_chunk, remaining)
            absorbed = coord.absorb_pending_updates(
                max_polls=1, poll_timeout=this_timeout
            )
            for child in absorbed:
                if child.username.lower() == target:
                    _ok(
                        f"@{child.username} confirmed and promoted to "
                        f"active (bot_id={child.bot_id})."
                    )
                    return 0
            # If we got OTHER bots but not ours, keep polling.
    except KeyboardInterrupt:
        print()
        _warn(
            "Stopped waiting.  Run `hermes fleet connect` later to absorb "
            "the bot once you tap the link."
        )
        return 0

    _warn(
        f"Didn't see @{target} within {wait_seconds}s.  Tap the link in "
        f"Telegram, then run `hermes fleet connect`."
    )
    return 0


def _explain_bot_api_error(e) -> None:
    """Translate well-known Telegram errors into actionable hints."""
    desc = str(getattr(e, "description", e)).lower()
    code = getattr(e, "code", None)
    if code == 409 or "webhook" in desc or "conflict" in desc:
        _info(
            "Telegram returned a conflict (409).  Either a webhook is set on "
            "the manager bot (run `curl https://api.telegram.org/bot<TOKEN>/"
            "deleteWebhook` to clear it) or another process is already "
            "polling getUpdates with this token (the gateway, perhaps).  "
            "Stop the other consumer and retry."
        )
    elif code == 401 or "unauthorized" in desc:
        _info(
            "The manager token was rejected (401).  Run `hermes fleet setup` "
            "again to validate / replace it."
        )
    elif code == 403 or "forbidden" in desc or "manager" in desc:
        _info(
            "Looks like Bot Manager Mode isn't enabled on this bot.  "
            "Open @BotFather \u2192 /mybots \u2192 select the bot \u2192 BotFather "
            "MiniApp \u2192 enable 'Bot Management Mode', then retry."
        )


def _cmd_adopt(args: argparse.Namespace) -> int:
    """Adopt an existing Telegram bot (created in BotFather) into the fleet."""
    from gateway.telegram_fleet import (
        FleetGuardrailError,
        SpawnApprovalRequired,
        get_coordinator,
    )
    from gateway.telegram_fleet.api import BotApiError

    token = (args.token or "").strip()
    if not token:
        if not sys.stdin.isatty():
            _err(
                "No --token provided and stdin is not a TTY.  Pass "
                "`--token <bot_token>` explicitly."
            )
            return 1
        _info(
            "Paste the bot's API token (from @BotFather \u2192 /mybots \u2192 "
            "select bot \u2192 API Token).  Looks like '12345:ABC...'."
        )
        try:
            token = getpass.getpass("Token (input hidden): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            _err("Aborted.")
            return 1
    if ":" not in token or len(token) < 30:
        _err(f"Token doesn't look right (got {len(token)} chars; expected ~46).")
        return 1

    toolset_list: Optional[list] = None
    if args.toolset:
        toolset_list = [t.strip() for t in str(args.toolset).split(",") if t.strip()]

    coord = get_coordinator(refresh=True)
    try:
        child = coord.adopt_existing_bot(
            token=token,
            persona=args.persona,
            model=args.model,
            toolset=toolset_list,
        )
    except SpawnApprovalRequired as e:
        _err(str(e))
        return 1
    except FleetGuardrailError as e:
        _err(str(e))
        return 1
    except BotApiError as e:
        _err(f"Telegram API error: {e}")
        _explain_bot_api_error(e)
        return 1
    _ok(
        f"Adopted @{child.username} into the fleet "
        f"(bot_id={child.bot_id}, status=active)."
    )
    if args.persona:
        _info(f"Persona: {args.persona}")
    print()
    _info(
        "The bot is ready for `telegram_orchestrate_swarm`.  "
        "Run `hermes fleet list` to verify."
    )
    return 0


def _cmd_connect(args: argparse.Namespace) -> int:
    """Drain managed_bot updates from Telegram and promote pending entries."""
    from gateway.telegram_fleet import get_coordinator
    from gateway.telegram_fleet.api import BotApiError

    if not (os.environ.get("TELEGRAM_FLEET_MANAGER_TOKEN") or "").strip():
        _err(
            "TELEGRAM_FLEET_MANAGER_TOKEN is not set.  Run "
            "`hermes fleet setup` first."
        )
        return 1

    coord = get_coordinator(refresh=True)
    pending_before = len(coord.list_children(status="pending"))
    if pending_before == 0:
        _info(
            "No pending entries in the roster.  Nothing to wait for.  "
            "Add a worker with `hermes fleet add <username>` first."
        )
        return 0

    wait_seconds = int(getattr(args, "wait", 0) or 0)
    _heading(
        f"Telegram Fleet \u2014 connect "
        f"({pending_before} pending entr{'y' if pending_before == 1 else 'ies'})"
    )

    if wait_seconds <= 0:
        try:
            absorbed = coord.absorb_pending_updates(
                max_polls=1, poll_timeout=0
            )
        except BotApiError as e:
            _err(f"Telegram API error: {e}")
            _explain_bot_api_error(e)
            return 1
        if not absorbed:
            _info(
                "No managed_bot updates queued.  Either the user hasn't "
                "tapped the link yet, or the gateway already absorbed them.  "
                "Pass `--wait 60` to long-poll."
            )
            return 0
        for child in absorbed:
            _ok(f"@{child.username} promoted to active (bot_id={child.bot_id}).")
        return 0

    _info(
        f"Long-polling for up to {wait_seconds}s "
        f"(Ctrl-C to stop early)."
    )
    import time

    deadline = time.monotonic() + wait_seconds
    seen: list = []
    try:
        while time.monotonic() < deadline:
            remaining = max(1, int(deadline - time.monotonic()))
            this_timeout = min(30, remaining)
            try:
                absorbed = coord.absorb_pending_updates(
                    max_polls=1, poll_timeout=this_timeout
                )
            except BotApiError as e:
                _err(f"Telegram API error: {e}")
                return 1
            for child in absorbed:
                seen.append(child)
                _ok(
                    f"@{child.username} promoted to active "
                    f"(bot_id={child.bot_id})."
                )
            # Stop early when no pending entries remain.
            if not coord.list_children(status="pending"):
                _ok("All pending entries absorbed.")
                return 0
    except KeyboardInterrupt:
        print()
    if not seen:
        _warn(
            f"Didn't see any new managed_bot events within {wait_seconds}s."
        )
    return 0
