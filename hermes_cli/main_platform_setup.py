"""Interactive messaging-platform setup wizards: WhatsApp (bridge + Cloud API), Slack manifest,
Skill Sync.

Split out of ``hermes_cli/main.py``. Names that still live in main are imported lazily at call time.
"""

import contextlib
import shutil
import subprocess
import sys

from hermes_cli.cli_output import line_input
from hermes_cli.model_setup_flows_common import _say


def _err(msg: str) -> None:
    print(msg, file=sys.stderr)


def _yes_no(prompt: str) -> bool:
    """``[y/N]`` prompt; Ctrl-C/EOF counts as "no"."""
    try:
        response = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        response = "n"
    return response.lower() in {"y", "yes"}


def _whatsapp_choose_mode(get_env_value, save_env_value):
    """Step 1 of ``hermes whatsapp``: ``"bot"`` / ``"self-chat"``, or None when cancelled."""
    current_mode = get_env_value("WHATSAPP_MODE") or ""
    if current_mode:
        mode_label = "separate bot number" if current_mode == "bot" else "personal number (self-chat)"
        print(f"\n✓ Mode: {mode_label}")
        return current_mode
    _say("", "How will you use WhatsApp with Hermes?", "",
         "  1. Separate bot number (recommended)",
         "     People message the bot's number directly — cleanest experience.",
         "     Requires a second phone number with WhatsApp installed on a device.", "",
         "  2. Personal number (self-chat)",
         "     You message yourself to talk to the agent.",
         "     Quick to set up, but the UX is less intuitive.", "")
    try:
        choice = input("  Choose [1/2]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nSetup cancelled.")
        return None
    if choice == "1":
        save_env_value("WHATSAPP_MODE", "bot")
        _say("  ✓ Mode: separate bot number", "",
             "  ┌─────────────────────────────────────────────────┐",
             "  │  Getting a second number for the bot:           │",
             "  │                                                 │",
             "  │  Easiest: Install WhatsApp Business (free app)  │",
             "  │  on your phone with a second number:            │",
             "  │    • Dual-SIM: use your 2nd SIM slot            │",
             "  │    • Google Voice: free US number (voice.google) │",
             "  │    • Prepaid SIM: $3-10, verify once            │",
             "  │                                                 │",
             "  │  WhatsApp Business runs alongside your personal │",
             "  │  WhatsApp — no second phone needed.             │",
             "  └─────────────────────────────────────────────────┘")
        return "bot"
    save_env_value("WHATSAPP_MODE", "self-chat")
    print("  ✓ Mode: personal number (self-chat)")
    return "self-chat"


def _whatsapp_allowed_users(wa_mode: str, get_env_value, save_env_value) -> None:
    """Step 3 of ``hermes whatsapp``: show / set WHATSAPP_ALLOWED_USERS."""
    current_users = get_env_value("WHATSAPP_ALLOWED_USERS") or ""
    if current_users:
        print(f"✓ Allowed users: {current_users}")
        if not _yes_no("\n  Update allowed users? [y/N] "):
            return
        if wa_mode == "bot":
            phone = line_input("  Phone numbers that can message the bot (comma-separated): ").strip()
        else:
            phone = line_input("  Your phone number (e.g. 15551234567): ").strip()
        if phone:
            save_env_value("WHATSAPP_ALLOWED_USERS", phone.replace(" ", ""))
            print(f"  ✓ Updated to: {phone}")
        return
    print()
    if wa_mode == "bot":
        print("  Who should be allowed to message the bot?")
        phone = line_input("  Phone numbers (comma-separated, or * for anyone): ").strip()
    else:
        phone = line_input("  Your phone number (e.g. 15551234567): ").strip()
    if phone:
        save_env_value("WHATSAPP_ALLOWED_USERS", phone.replace(" ", ""))
        print(f"  ✓ Allowed users set: {phone}")
    else:
        print("  ⚠ No allowlist — the agent will respond to ALL incoming messages")


def _whatsapp_install_bridge(bridge_dir) -> bool:
    """Step 4 of ``hermes whatsapp``: ``npm install`` the bridge when needed. False = stop."""
    from hermes_constants import find_node_executable, with_hermes_node_path
    if (bridge_dir / "node_modules").exists():
        print("✓ Bridge dependencies already installed")
        return True
    print("\n→ Installing WhatsApp bridge dependencies (this can take a few minutes)...")
    npm = find_node_executable("npm")
    if not npm:
        print("  ✗ npm not found on PATH — install Node.js first")
        return False
    try:
        result = subprocess.run(
            [npm, "install", "--no-fund", "--no-audit", "--progress=false"],
            cwd=str(bridge_dir), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
            encoding="utf-8", errors="replace", env=with_hermes_node_path())
    except KeyboardInterrupt:
        print("\n  ✗ Install cancelled")
        return False
    if result.returncode != 0:
        err = (result.stderr or "").strip()
        preview = "\n".join(err.splitlines()[-30:]) if err else "(no output)"
        _say("  ✗ npm install failed:", preview)
        return False
    print("  ✓ Dependencies installed")
    return True


def cmd_whatsapp(args):
    """Set up WhatsApp: choose mode, configure, install bridge, pair via QR."""
    from hermes_cli.main import _require_tty, get_hermes_home
    _require_tty("whatsapp")
    from hermes_cli.config import get_env_value, save_env_value
    from hermes_constants import find_node_executable, with_hermes_node_path
    _say("", "⚕ WhatsApp Setup", "=" * 50)

    wa_mode = _whatsapp_choose_mode(get_env_value, save_env_value)
    if wa_mode is None:
        return

    # WHATSAPP_ENABLED=true is deliberately NOT written here: an aborted wizard (Ctrl+C, failed npm
    # install, missed QR scan) would leave .env claiming WhatsApp is ready with no creds.json, and
    # every `hermes gateway` would pay a 30s bridge timeout + indefinite retries. Set only after
    # pairing succeeds; prior successful pairings stay enabled.
    print()
    if (get_env_value("WHATSAPP_ENABLED") or "").lower() == "true":
        print("✓ WhatsApp is already enabled")

    _whatsapp_allowed_users(wa_mode, get_env_value, save_env_value)

    from gateway.platforms.whatsapp_common import resolve_whatsapp_bridge_dir
    bridge_dir = resolve_whatsapp_bridge_dir()
    bridge_script = bridge_dir / "bridge.js"
    if not bridge_script.exists():
        print(f"\n✗ Bridge script not found at {bridge_script}")
        return
    if not _whatsapp_install_bridge(bridge_dir):
        return

    # Existing session: re-pair or keep.
    session_dir = get_hermes_home() / "whatsapp" / "session"
    session_dir.mkdir(parents=True, exist_ok=True)
    if (session_dir / "creds.json").exists():
        print("✓ Existing WhatsApp session found")
        if _yes_no("\n  Re-pair? This will clear the existing session. [y/N] "):
            shutil.rmtree(session_dir, ignore_errors=True)
            session_dir.mkdir(parents=True, exist_ok=True)
            print("  ✓ Session cleared")
        else:
            # Older installs may have lost WHATSAPP_ENABLED; a kept pairing re-asserts it.
            if (get_env_value("WHATSAPP_ENABLED") or "").lower() != "true":
                save_env_value("WHATSAPP_ENABLED", "true")
            _say("\n✓ WhatsApp is configured and paired!", "  Start the gateway with: hermes gateway")
            return

    # QR code pairing
    _say("", "─" * 50)
    if wa_mode == "bot":
        _say("📱 Open WhatsApp (or WhatsApp Business) on the", "   phone with the BOT's number, then scan:")
    else:
        print("📱 Open WhatsApp on your phone, then scan:")
    _say("", "   Settings → Linked Devices → Link a Device", "─" * 50, "")
    with contextlib.suppress(KeyboardInterrupt):
        subprocess.run(
            [find_node_executable("node") or "node", str(bridge_script), "--pair-only", "--session", str(session_dir)],
            cwd=str(bridge_dir), env=with_hermes_node_path())

    print()
    if not (session_dir / "creds.json").exists():
        print("⚠ Pairing may not have completed. Run 'hermes whatsapp' to try again.")
        return
    # Only enable WhatsApp now that pairing actually succeeded (see above).
    save_env_value("WHATSAPP_ENABLED", "true")
    _say("✓ WhatsApp paired successfully!", "")
    if wa_mode == "bot":
        _say("  Next steps:", "    1. Start the gateway:  hermes gateway",
             "    2. Send a message to the bot's WhatsApp number",
             "    3. The agent will reply automatically", "",
             "  Tip: Agent responses are prefixed with '⚕ Hermes Agent'")
    else:
        _say("  Next steps:", "    1. Start the gateway:  hermes gateway",
             "    2. Open WhatsApp → Message Yourself",
             "    3. Type a message — the agent will reply", "",
             "  Tip: Agent responses are prefixed with '⚕ Hermes Agent'",
             "  so you can tell them apart from your own messages.")
    _say("", "  Or install as a service: hermes gateway install")


def cmd_whatsapp_cloud(args):
    """Set up WhatsApp Business Cloud API (official Meta integration) — complementary to the
    ``hermes whatsapp`` Baileys bridge wizard. See ``hermes_cli/setup_whatsapp_cloud.py``."""
    from hermes_cli.main import _require_tty
    _require_tty("whatsapp-cloud")
    from hermes_cli.setup_whatsapp_cloud import run_whatsapp_cloud_setup
    return run_whatsapp_cloud_setup()


_SYNC_USAGE = (
    "usage: hermes sync "
    "<status|pull|push|now|enable|disable|device|propose>\n"
    "\n"
    "Your skills, across your devices:\n"
    "  status            Show what is synced, and from where\n"
    "  pull              Pull your synced skills\n"
    "  push              Push your opted-in skills\n"
    "  now               Reconcile now: pull then push\n"
    "  enable <skill>    Include a skill in your sync\n"
    "  disable <skill>   Exclude a skill from your sync\n"
    "  device [--name N] Show or set this device's label\n"
    "\n"
    "Shared with your team:\n"
    "  propose <skill>   Share a skill with your organisation")


def _sync_device(args, ssc) -> int:
    name = getattr(args, "device_name", None)
    if name is not None:
        try:
            stored = ssc.set_device_name(name)
        except ValueError as e:
            _err(f"error: {e}")
            return 1
        print(f"device label set to '{stored}'.")
        _err("New commits from this device will use this label; existing commits keep their previous one.")
        return 0
    # No --name: print the current (creating a default on first use).
    print(ssc.stable_device_id())
    return 0


def _sync_propose(args, ssc) -> int:
    from tools.skills_sync_client_org import propose_skill
    name = args.name
    try:
        result = propose_skill(name, message=args.message)
    except ssc.SyncInertError as e:
        _err(f"cannot share this skill: {e}")
        return 1
    except ssc.SyncError as e:
        _err(f"could not share '{name}': {e}")
        return 1
    if result.get("proposal_pending"):
        print(f"Shared '{name}' with your organisation — an admin needs to "
              f"approve it (proposal #{result.get('proposal_id')}). It is "
              f"not live for the team until then.")
    else:
        print(f"Added '{name}' to your organisation's shared skills.")
    return 0


def _sync_toggle(args, sub: str) -> int:
    from tools.skill_usage import set_sync, is_curation_eligible
    skill = args.skill
    if not is_curation_eligible(skill):
        _err(f"'{skill}' is not sync-eligible (bundled, hub-installed, "
             f"external, or not found). Only agent-created / user-authored "
             f"skills under ~/.hermes/skills/ can sync.")
        return 1
    set_sync(skill, sub == "enable")
    print(f"sync {'enabled' if sub == 'enable' else 'disabled'} for '{skill}'.")
    return 0


def _sync_status(ssc) -> int:
    import json as _json
    status = ssc.sync_status()
    print(_json.dumps(status, indent=2, ensure_ascii=False))
    if status.get("org_available"):
        n = len(status.get("org_skills") or [])
        modified = status.get("org_skills_modified") or []
        _err(f"\nOrg skills: {n} shared skill(s) from your organisation "
             f"(your role: {status.get('org_role')}). They load alongside "
             f"your own, labeled by origin, and you can edit them.")
        if modified:
            _err(f"  {len(modified)} with local edits not yet shared: "
                 f"{', '.join(modified)}\n"
                 f"  Share them back with `hermes sync propose <skill>`. "
                 f"Org updates will not overwrite them.")
    elif status.get("logged_in"):
        _err("\nOrg skills: not applicable — this account isn't a member of a shared organisation.")
    if not status.get("logged_in"):
        _err("\nNot logged into Nous Portal — sync is inert.")
    elif not status.get("nous_admin"):
        _err("\nSync is not enabled for your account yet.")
    elif not status.get("feature_enabled"):
        _err("\nSync feature is off for this instance (set HERMES_SYNC_ENABLED=1 "
             "or config.yaml sync.enabled: true). Sync is inert.")
    elif not status.get("base_url"):
        _err("\nNo sync base URL configured (config.yaml sync.base_url or HERMES_SYNC_BASE_URL). Sync is inert.")
    return 0


def _sync_pull(ssc, identity):
    result = ssc.pull_skills(identity=identity)
    # Refresh the org mirror too when this account belongs to an organisation (no-op
    # otherwise), so one pull covers both.
    from tools.skills_sync_client_org import maybe_pull_org_skills
    org_result = maybe_pull_org_skills()
    if org_result:
        n = len(org_result.get("updated") or [])
        _err(f"org: refreshed {n} shared skill(s) from your organisation.")
        clashes = org_result.get("conflicted") or []
        if clashes:
            _err(f"org: {len(clashes)} skill(s) have BOTH local edits "
                 f"and org updates, so they were left as-is: "
                 f"{', '.join(clashes)}\n"
                 f"     Your local version is intact. Review it, then "
                 f"either propose it or delete the local copy and pull "
                 f"again to take the org version.")
    return result


# gated (identity-checked) sync subcommands: name -> (ssc, identity) -> result
_SYNC_GATED = {
    "pull": _sync_pull,
    "push": lambda ssc, identity: ssc.push_skills(identity=identity, message="hermes sync push"),
    "now": lambda ssc, identity: {"pull": ssc.pull_skills(identity=identity),
                                  "push": ssc.push_skills(identity=identity, message="hermes sync now")}}


def cmd_sync(args):
    """Skill Sync — personal sync across devices, plus sharing with your org."""
    import json as _json
    sub = getattr(args, "sync_command", None)
    if sub in {None, ""}:
        _err(_SYNC_USAGE)
        return 1
    if sub in {"enable", "disable"}:
        return _sync_toggle(args, sub)

    from tools import skills_sync_client as ssc
    if sub == "device":
        return _sync_device(args, ssc)
    if sub == "propose":
        return _sync_propose(args, ssc)
    if sub == "status":
        return _sync_status(ssc)

    # pull / push / now — enforce the gate up front with a clear message.
    try:
        identity = ssc.resolve_identity()
    except ssc.SyncInertError as e:
        _err(f"sync inert: {e}")
        return 1
    if not identity.get("nous_admin"):
        _err("sync unavailable: not enabled for your account yet.")
        return 1
    if not ssc.resolve_sync_base_url():
        _err("sync inert: no sync base URL configured (config.yaml sync.base_url or HERMES_SYNC_BASE_URL).")
        return 1

    action = _SYNC_GATED.get(sub)
    if action is None:
        _err(f"Unknown sync subcommand: {sub}")
        return 1
    try:
        result = action(ssc, identity)
    except ssc.SyncError as e:
        _err(f"sync failed: {e}")
        return 1
    print(_json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_slack(args):
    """``hermes slack <subcommand>``; ``manifest`` prints or writes a Slack app manifest with
    every gateway command registered as a first-class slash."""
    sub = getattr(args, "slack_command", None)
    if sub in {None, ""}:
        _err("usage: hermes slack <subcommand>\n"
             "\n"
             "subcommands:\n"
             "  manifest   Generate a Slack app manifest with every gateway\n"
             "             command registered as a native slash\n"
             "\n"
             "Run `hermes slack manifest -h` for details.")
        return 1

    if sub == "manifest":
        from hermes_cli.slack_cli import slack_manifest_command
        status = slack_manifest_command(args)
        if status:
            raise SystemExit(status)
        return status

    _err(f"Unknown slack subcommand: {sub}")
    return 1
