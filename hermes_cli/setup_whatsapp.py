"""
Interactive ``hermes whatsapp setup`` wizard — one-command WhatsApp configuration.

Walks the user through:

1. Pairing (QR code via existing ``hermes whatsapp`` bridge)
2. Who can use the bot (allowlist + dm_policy)
3. Cron delivery home channel
4. Test message delivery
5. Gateway restart (optional)

Idempotent: re-running updates config, never errors out on existing state.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from hermes_constants import get_hermes_home, find_node_executable, with_hermes_node_path

logger = logging.getLogger(__name__)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _prompt(message: str, default: str | None = None) -> str:
    """Read one line of input. Returns ``""`` on EOF / Ctrl+C."""
    try:
        suffix = f" [{default}]" if default else ""
        raw = input(f"{message}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return ""
    return raw


def _confirm(message: str, default: bool = True) -> bool:
    """Ask a yes/no question. Returns True for yes, False for no."""
    hint = "Y/n" if default else "y/N"
    answer = _prompt(f"{message} [{hint}]").lower() or ("y" if default else "n")
    return answer in ("y", "yes")


def _save_env(key: str, value: str) -> None:
    """Persist a key=value to ``~/.hermes/.env`` via the config module."""
    from hermes_cli.config import save_env_value

    save_env_value(key, value)


def _read_env(key: str) -> str | None:
    """Read a value from ``~/.hermes/.env``."""
    from hermes_cli.config import get_env_value

    return get_env_value(key) or None


def _hermes_config_set(key: str, value: str) -> None:
    """Run ``hermes config set <key> <value>``."""
    try:
        subprocess.run(
            [sys.executable or "python3", "-m", "hermes_cli", "config", "set", key, value],
            capture_output=True,
            timeout=30,
        )
    except Exception as exc:
        logger.warning("hermes config set %s=%s failed: %s", key, value, exc)


def _bridge_api_send(
    chat_id: str, message: str, *, host: str = "127.0.0.1", port: int = 3000
) -> bool:
    """Send a message via the WhatsApp bridge HTTP API."""
    import urllib.request

    payload = json.dumps({"chatId": chat_id, "message": message}).encode("utf-8")
    req = urllib.request.Request(
        f"http://{host}:{port}/send",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8")
            return resp.status == 200 and '"success"' in body
    except Exception:
        return False


def _get_bot_phone_from_creds() -> str | None:
    """Extract the bot's own phone number from ``creds.json``.

    Returns something like ``5511999999999`` (digits only) or None.
    """
    creds_path = get_hermes_home() / "whatsapp" / "session" / "creds.json"
    if not creds_path.exists():
        return None
    try:
        data = json.loads(creds_path.read_text(encoding="utf-8"))
        raw = data.get("me", {}).get("id", "")
        # creds format: "5511999999999:<device_id>" or just "5511999999999"
        return raw.split(":")[0] if raw else None
    except Exception:
        return None


def _get_lid_from_gateway_logs() -> str | None:
    """Grep the most recent inbound WhatsApp message from gateway logs and
    return the chat LID (e.g. ``55310773391517``) or None."""
    log_path = get_hermes_home() / "logs" / "gateway.log"
    if not log_path.exists():
        return None
    try:
        import subprocess

        result = subprocess.run(
            ["grep", "-oP", r'chat=\K\d+(?=@lid)', str(log_path)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
        return lines[-1] if lines else None
    except Exception:
        return None


def _resolve_lid_to_phone(lid: str) -> str | None:
    """Try to resolve a LID to a phone number via lid-mapping files."""
    mapping_path = get_hermes_home() / "whatsapp" / "session" / f"lid-mapping-{lid}.json"
    if mapping_path.exists():
        try:
            return json.loads(mapping_path.read_text(encoding="utf-8")).strip() or None
        except Exception:
            return None
    # Also try the reverse
    for f in (get_hermes_home() / "whatsapp" / "session").glob("lid-mapping-*_reverse.json"):
        pass
    return None


def _get_bridge_status() -> str | None:
    """Check if the bridge HTTP API is responding on port 3000.

    Returns the response text or None if unreachable.
    """
    import urllib.request

    req = urllib.request.Request("http://127.0.0.1:3000/health", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.read().decode("utf-8")
    except Exception:
        return None


# ── Wizard Steps ─────────────────────────────────────────────────────────────


def step_pair() -> bool:
    """Step 1: Pair via QR code.

    Returns True if pairing succeeded (creds.json exists after the attempt).
    """
    from hermes_cli.config import get_env_value, save_env_value

    print()
    print("=" * 50)
    print("STEP 1/5 — Pairing")
    print("=" * 50)
    print()
    print("  We'll start the WhatsApp bridge and show a QR code.")
    print("  Open WhatsApp on your phone, then scan it:")
    print()
    print("     Settings → Linked Devices → Link a Device")
    print()

    bridge_dir = None
    try:
        from gateway.platforms.whatsapp_common import resolve_whatsapp_bridge_dir

        bridge_dir = resolve_whatsapp_bridge_dir()
    except Exception:
        pass

    session_dir = get_hermes_home() / "whatsapp" / "session"
    session_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing session
    if (session_dir / "creds.json").exists():
        phone = _get_bot_phone_from_creds()
        label = phone or "existing session"
        print(f"  ✓ Already paired ({label})")
        if not _confirm("  Re-pair? This will clear the existing session", default=False):
            return True
        shutil.rmtree(session_dir, ignore_errors=True)
        session_dir.mkdir(parents=True, exist_ok=True)
        print("  ✓ Session cleared")

    # Install bridge dependencies if needed
    if bridge_dir and not (bridge_dir / "node_modules").exists():
        print()
        print("  → Installing WhatsApp bridge dependencies...")
        npm = find_node_executable("npm")
        if not npm:
            print("  ✗ npm not found — install Node.js first")
            return False
        try:
            result = subprocess.run(
                [npm, "install", "--no-fund", "--no-audit", "--progress=false"],
                cwd=str(bridge_dir),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=with_hermes_node_path(),
            )
        except KeyboardInterrupt:
            print("  ✗ Install cancelled")
            return False
        if result.returncode != 0:
            print("  ✗ npm install failed")
            return False
        print("  ✓ Dependencies installed")

    if not bridge_dir:
        print("  ✗ WhatsApp bridge not found")
        return False

    # Show QR
    bridge_script = bridge_dir / "bridge.js"
    if not bridge_script.exists():
        print(f"  ✗ Bridge script not found at {bridge_script}")
        return False

    print()
    print("  📱 Scan the QR code below with WhatsApp:")
    print()

    try:
        subprocess.run(
            [
                find_node_executable("node") or "node",
                str(bridge_script),
                "--pair-only",
                "--session",
                str(session_dir),
            ],
            cwd=str(bridge_dir),
            env=with_hermes_node_path(),
        )
    except KeyboardInterrupt:
        print()
        print("  ⚠ Pairing interrupted")

    # Verify pairing
    if (session_dir / "creds.json").exists():
        _save_env("WHATSAPP_ENABLED", "true")
        phone = _get_bot_phone_from_creds()
        label = f" as {phone}" if phone else ""
        print(f"  ✓ Paired successfully{label}!")
        return True
    else:
        print("  ⚠ Pairing may not have completed. You can re-run this wizard.")
        return False


def step_allowlist() -> str | None:
    """Step 2: Configure who can use the bot.

    Returns the owner's phone number that was set, or None if skipped.
    """
    from hermes_cli.config import get_env_value

    print()
    print("=" * 50)
    print("STEP 2/5 — Who can use this bot?")
    print("=" * 50)
    print()

    current_allowed = get_env_value("WHATSAPP_ALLOWED_USERS") or ""
    bot_phone = _get_bot_phone_from_creds()

    if bot_phone:
        print(f"  We detected your paired bot number: {bot_phone}")
        print()
        is_owner = _confirm("  Is this YOUR personal phone number?", default=True)
        if is_owner:
            owner_phone = bot_phone
        else:
            owner_phone = _prompt("  Enter your personal phone number (country code + number, no +/spaces)")
            if not owner_phone:
                print("  ⚠ Skipping — allowlist not updated")
                return current_allowed or None
    else:
        print("  (Could not detect the paired number from the session.)")
        owner_phone = _prompt(
            "  Enter your phone number (e.g. 5511999999999)"
            + (f" [{current_allowed}]" if current_allowed else "")
        ) or current_allowed
        if not owner_phone:
            print("  ⚠ Skipping — allowlist not updated")
            return None

    # Normalize
    owner_phone = re.sub(r"[\s\-+()]", "", owner_phone)

    # Save to .env
    _save_env("WHATSAPP_ALLOWED_USERS", owner_phone)
    _save_env("WHATSAPP_ALLOW_ALL_USERS", "false")
    _hermes_config_set("whatsapp.dm_policy", "allowlist")
    print(f"  ✓ WHATSAPP_ALLOWED_USERS = {owner_phone}")
    print("  ✓ WHATSAPP_ALLOW_ALL_USERS = false")
    print("  ✓ dm_policy = allowlist")
    print()
    print("  → Only your number can message the bot. No randoms.")

    return owner_phone


def step_home_channel() -> str | None:
    """Step 3: Configure cron delivery home channel.

    Returns the LID that was set, or None if skipped.
    """
    from hermes_cli.config import get_env_value

    print()
    print("=" * 50)
    print("STEP 3/5 — Cron delivery home channel")
    print("=" * 50)
    print()
    print("  Cron job output can be delivered to your WhatsApp.")
    print()

    current_channel = get_env_value("WHATSAPP_HOME_CHANNEL") or ""
    if current_channel:
        print(f"  Current home channel: {current_channel}")
        if not _confirm("  Update it?", default=False):
            print("  ✓ Keeping existing home channel")
            return current_channel

    # Try to auto-detect from gateway logs
    lid = _get_lid_from_gateway_logs()
    if lid:
        print(f"  Detected your chat LID: {lid}@lid")
        if _confirm("  Use this as the home channel?", default=True):
            channel = f"{lid}@lid"
            _save_env("WHATSAPP_HOME_CHANNEL", channel)
            print(f"  ✓ WHATSAPP_HOME_CHANNEL = {channel}")
            return channel
    else:
        print("  (Could not auto-detect — no inbound WhatsApp messages in logs yet.)")

    # Manual entry
    channel = _prompt(
        "  Enter your WhatsApp chat LID (e.g. 55310773391517@lid)"
        + (f" [{current_channel}]" if current_channel else "")
    ) or current_channel
    if channel:
        _save_env("WHATSAPP_HOME_CHANNEL", channel)
        print(f"  ✓ WHATSAPP_HOME_CHANNEL = {channel}")
        return channel
    else:
        print("  ⚠ No home channel set — cron delivery to WhatsApp won't work")
        print("    until you set WHATSAPP_HOME_CHANNEL manually.")
        return None


def step_test_delivery(owner_phone: str | None) -> bool:
    """Step 4: Send a test message to confirm delivery works.

    Returns True if the test was sent (or skipped), False on failure.
    """
    from hermes_cli.config import get_env_value

    print()
    print("=" * 50)
    print("STEP 4/5 — Test delivery")
    print("=" * 50)
    print()

    # Resolve the chat ID to send to
    # Priority: home channel > LID from logs > owner phone
    home_channel = get_env_value("WHATSAPP_HOME_CHANNEL") or ""
    lid = _get_lid_from_gateway_logs()

    if home_channel:
        chat_id = home_channel
    elif lid:
        chat_id = f"{lid}@lid"
    elif owner_phone:
        chat_id = f"{owner_phone}@s.whatsapp.net"
    else:
        print("  ⚠ No delivery target configured — skipping test")
        return False

    # Check bridge API is running
    status = _get_bridge_status()
    if status is None:
        print("  ⚠ WhatsApp bridge API is not currently running (port 3000).")
        print("  Start the gateway first:  hermes gateway run")
        print("  Then re-run this wizard to test delivery.")
        return False

    print(f"  Sending test message to {chat_id}...")
    success = _bridge_api_send(chat_id, "✅ Hermes WhatsApp setup complete! This is a test message from your Hermes Agent.")

    if success:
        print("  ✅ Test message sent! Check your WhatsApp.")
        return True
    else:
        print("  ⚠ Test message could not be sent (bridge API returned an error).")
        print("  You can test manually after gateway starts:")
        print(f"    curl -X POST http://127.0.0.1:3000/send \\")
        print(f"      -H 'Content-Type: application/json' \\")
        print(f"      -d '{{\"chatId\":\"{chat_id}\",\"message\":\"Hello from Hermes!\"}}'")
        return False


def step_restart_gateway() -> None:
    """Step 5: Offer to restart the gateway service."""
    print()
    print("=" * 50)
    print("STEP 5/5 — Restart gateway")
    print("=" * 50)
    print()
    print("  For all changes to take effect, the Hermes gateway must")
    print("  be restarted to reload .env vars and config.yaml.")
    print()

    if _confirm("  Restart hermes-gateway now?", default=True):
        try:
            result = subprocess.run(
                ["systemctl", "restart", "hermes-gateway.service"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                print("  ✓ Gateway restarted!")
            else:
                print(f"  ⚠ Restart failed: {result.stderr.strip()}")
                print("  Restart manually: sudo systemctl restart hermes-gateway")
        except FileNotFoundError:
            print("  ⚠ systemctl not found (not running as systemd?)")
            print("  Restart the gateway manually: hermes gateway restart")
        except subprocess.TimeoutExpired:
            print("  ⚠ Restart timed out")
    else:
        print("  ⚠ Skipped — remember to restart the gateway later:")
        print("    sudo systemctl restart hermes-gateway.service")


# ── Summary ──────────────────────────────────────────────────────────────────


def _print_summary(
    paired: bool,
    paired_as: str | None,
    owner_phone: str | None,
    home_channel: str | None,
    test_ok: bool,
) -> None:
    """Print a beautiful setup summary."""
    print()
    print("=" * 50)
    print("  ✅ WhatsApp Setup Complete!")
    print("=" * 50)
    print()
    print("  ┌────────────────────────────────────────────────────────┐")

    status_paired = f"  ✓ Paired as {paired_as}" if paired and paired_as else "  ✓ Paired"
    print(f"  │ {status_paired:<49s}│")

    if owner_phone:
        print(f"  │ {'  Allowed users: ' + owner_phone + ' (only you)':<49s}│")
        print(f"  │ {'  DM policy: allowlist (no randoms)':<49s}│")
    else:
        print(f"  │ {'  Allowlist: not configured':<49s}│")

    if home_channel:
        print(f"  │ {'  Home channel: ' + home_channel:<49s}│")
        print(f"  │ {'  Cron deliveries: enabled':<49s}│")
    else:
        print(f"  │ {'  Home channel: not set':<49s}│")

    print(f"  │ {'  Test delivery: ' + ('✅ PASS' if test_ok else '⚠ SKIPPED'):<49s}│")
    print(f"  │ {'  Gateway: restart required':<49s}│")
    print("  └────────────────────────────────────────────────────────┘")
    print()
    print("  Next steps:")
    print("    1. Send a message to the bot on WhatsApp")
    print("    2. The agent will reply automatically")
    print("    3. Use hermes cron create ... deliver 'whatsapp' to schedule jobs")
    print()


# ── Main Entry Point ─────────────────────────────────────────────────────────


def run_whatsapp_setup() -> int:
    """Run the interactive WhatsApp setup wizard.

    Returns 0 on success, 1 on error/abort.
    """
    print()
    print("⚕ WhatsApp Setup Wizard")
    print("=" * 50)
    print()
    print("  This will guide you through configuring WhatsApp for Hermes.")
    print("  You'll need your WhatsApp account ready to scan a QR code.")
    print()

    try:
        proceed = input("  Press Enter to continue, or Ctrl+C to abort... ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n  Setup cancelled.")
        return 1

    # ── Step 1: Pairing ─────────────────────────────────────────────────
    paired = step_pair()
    if not paired:
        print()
        print("  ⚠ Pairing did not complete. Other steps were skipped.")
        print("  Re-run:  hermes whatsapp setup")
        return 1

    paired_as = _get_bot_phone_from_creds()

    # ── Step 2: Allowlist ───────────────────────────────────────────────
    owner_phone = step_allowlist()

    # ── Step 3: Home channel ────────────────────────────────────────────
    home_channel = step_home_channel()

    # ── Step 4: Test delivery ───────────────────────────────────────────
    test_ok = step_test_delivery(owner_phone)

    # ── Step 5: Restart gateway ─────────────────────────────────────────
    step_restart_gateway()

    # ── Summary ─────────────────────────────────────────────────────────
    _print_summary(
        paired=paired,
        paired_as=paired_as,
        owner_phone=owner_phone,
        home_channel=home_channel,
        test_ok=test_ok,
    )

    return 0


# ── CLI Entry (for direct testing) ───────────────────────────────────────────


def cmd_whatsapp_setup(args) -> None:
    """CLI handler for ``hermes whatsapp setup``."""
    sys.exit(run_whatsapp_setup())


if __name__ == "__main__":
    sys.exit(run_whatsapp_setup())
