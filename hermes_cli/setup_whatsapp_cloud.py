"""
Interactive setup wizard for the WhatsApp Cloud API adapter.

Entry point: ``hermes whatsapp-cloud`` (dispatched from
``cmd_whatsapp_cloud`` in ``hermes_cli/main.py``).

Walks the user through the 6 credentials Meta requires + recipient
allowlist, auto-generates the verify token, and prints exact follow-up
instructions for the parts that can't happen inside the wizard process
(starting cloudflared, starting the gateway, configuring Meta's
webhook dashboard, adding their phone to the recipient list).

Heavy emphasis on field-shape validation to catch the most common
configuration mistakes:

- Putting the actual phone number in ``WHATSAPP_CLOUD_PHONE_NUMBER_ID``
  (the field expects Meta's 15-17 digit internal ID, not a phone number).
  This is the #1 trap — caught us during Phase 3 live testing.
- Pasting tokens with trailing whitespace.
- Pasting an OpenAI / Slack / GitHub key by mistake.
- Confusing App ID with WABA ID with Phone Number ID.

Each prompt has contextual help showing exactly where to find the value
in Meta's App Dashboard, with a one-line description and the field's
expected shape ("starts with EAA", "15-17 digits", "32 hex chars", etc.).

The wizard intentionally does NOT smoke-test the webhook itself — the
Hermes gateway and the cloudflared tunnel both run in separate
processes the user starts AFTER this wizard exits, so any in-wizard
probe would fail by design. Instead the final SETUP COMPLETE block
prints the exact curl command the user can run from a third terminal
to verify the loop end-to-end once everything's running.
"""

from __future__ import annotations

import re
import secrets
import sys
from typing import Optional


# ---------------------------------------------------------------------------
# Field-shape validators
# ---------------------------------------------------------------------------
#
# Each validator returns (ok, reason_if_not_ok). The wizard uses them to
# reject obviously-malformed input before saving — saves users a round
# trip with Meta's 401 / 400 errors.


def _validate_phone_number_id(value: str) -> tuple[bool, Optional[str]]:
    """Phone Number ID is a 15-17 digit numeric ID assigned by Meta.

    It's NOT a phone number. The #1 setup mistake is pasting the actual
    phone number (e.g. ``15556422442``) into this field — that's only
    10-11 digits and gets rejected by Graph as "Object with ID does
    not exist."
    """
    if not value:
        return False, "Phone Number ID is required"
    s = value.strip()
    if not s.isdigit():
        return False, "Phone Number ID must be numeric (no '+', spaces, or dashes)"
    # Real phone numbers are 10-11 digits (US/CA country code + area code
    # + 7 digits). Meta's internal IDs are 15-17 digits. If we see a
    # phone-number-sized value, the user almost certainly pasted the
    # phone number by mistake.
    if 10 <= len(s) <= 12:
        return False, (
            "That looks like a phone number — but this field needs the "
            "Phone Number ID (Meta's internal ID, 15-17 digits, e.g. "
            "'7794189252778687'). Look just BELOW the 'From' dropdown in "
            "API Setup → it's labelled 'Phone number ID'."
        )
    if len(s) < 13:
        return False, "Phone Number ID looks too short (expected 13-18 digits)"
    if len(s) > 20:
        return False, "Phone Number ID looks too long (expected 13-18 digits)"
    return True, None


def _validate_waba_id(value: str) -> tuple[bool, Optional[str]]:
    """WABA ID is numeric, similar length range as Phone Number ID."""
    if not value:
        return False, "WABA ID is required"
    s = value.strip()
    if not s.isdigit():
        return False, "WABA ID must be numeric"
    if len(s) < 10 or len(s) > 25:
        return False, "WABA ID looks wrong (expected 10-25 digits)"
    return True, None


def _validate_app_id(value: str) -> tuple[bool, Optional[str]]:
    """Meta App ID is numeric, typically 15-16 digits."""
    if not value:
        return False, "App ID is required"
    s = value.strip()
    if not s.isdigit():
        return False, "App ID must be numeric"
    if len(s) < 13 or len(s) > 20:
        return False, "App ID looks wrong (expected 15-16 digits)"
    return True, None


def _validate_app_secret(value: str) -> tuple[bool, Optional[str]]:
    """App Secret is a 32-character lowercase hex string."""
    if not value:
        return False, "App Secret is required"
    s = value.strip()
    if not re.fullmatch(r"[0-9a-f]+", s.lower()):
        return False, (
            "App Secret should be a hex string (only digits 0-9 and "
            "letters a-f). Make sure you copied the 'App secret' from "
            "Settings → Basic, not some other token."
        )
    if len(s) != 32:
        return False, f"App Secret should be exactly 32 hex characters (got {len(s)})"
    return True, None


def _validate_access_token(value: str) -> tuple[bool, Optional[str]]:
    """Meta access tokens start with ``EAA`` and are 100-300+ characters.

    Both temp tokens (24h) and System User permanent tokens share this
    prefix. We don't try to distinguish them.
    """
    if not value:
        return False, "Access token is required"
    s = value.strip()
    if not s.startswith("EAA"):
        # Diagnose common paste mistakes
        if s.startswith("sk-"):
            return False, (
                "That's an OpenAI key (starts with 'sk-'), not a Meta "
                "WhatsApp access token. Meta tokens start with 'EAA'."
            )
        if s.startswith("xoxb-") or s.startswith("xoxp-"):
            return False, (
                "That's a Slack token, not a Meta WhatsApp access token. "
                "Meta tokens start with 'EAA'."
            )
        if s.startswith("ghp_") or s.startswith("gho_"):
            return False, (
                "That's a GitHub token, not a Meta WhatsApp access "
                "token. Meta tokens start with 'EAA'."
            )
        return False, (
            "Meta WhatsApp access tokens start with 'EAA'. Check that "
            "you're copying from the right place (API Setup → 'Generate "
            "access token', or Business Settings → System Users → "
            "'Generate token' for a permanent one)."
        )
    if len(s) < 100:
        return False, f"Access token looks too short ({len(s)} chars, expected 100+)"
    return True, None


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def _prompt(message: str, default: Optional[str] = None) -> str:
    """Read one line of input. Returns "" on EOF / Ctrl+C / empty input.

    The ``default`` parameter is shown to the user but NOT auto-applied
    on empty input — callers handle the "user kept existing" case
    explicitly so they can distinguish between a real value and a
    display preview (e.g. ``"abc12345..."`` for masked secrets).
    """
    try:
        suffix = f" [{default}]" if default else ""
        raw = input(f"{message}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return ""
    return raw


def _prompt_validated(
    message: str,
    validator,
    *,
    current: Optional[str] = None,
    help_text: Optional[str] = None,
) -> Optional[str]:
    """Repeat the prompt until the user enters a valid value or aborts.

    Returns the validated value, or None if the user gave up (empty
    response after an error, or Ctrl+C). ``current`` is shown as a
    default for re-runs of the wizard with existing config.
    """
    if help_text:
        for line in help_text.strip().splitlines():
            print(f"  {line}")
    attempts = 0
    while True:
        attempts += 1
        value = _prompt(f"  → {message}", default=current)
        if not value:
            return None
        ok, reason = validator(value)
        if ok:
            return value.strip()
        print(f"    ✗ {reason}")
        if attempts >= 3:
            try:
                cont = input("    Try again, or press Enter to skip: ").strip()
            except (EOFError, KeyboardInterrupt):
                return None
            if not cont:
                return None
            attempts = 0


# ---------------------------------------------------------------------------
# Wizard
# ---------------------------------------------------------------------------


def run_whatsapp_cloud_setup() -> int:
    """Interactive wizard for the WhatsApp Cloud API adapter.

    Returns 0 on full success, 1 on user abort, 2 on partial completion
    (some fields written but the user bailed before finishing).
    """
    from hermes_cli.config import get_env_value, save_env_value

    print()
    print("⚕ WhatsApp Business Cloud API Setup")
    print("=" * 50)
    print()
    print("This wizard configures Hermes to talk to WhatsApp via Meta's")
    print("official Cloud API. It's the production-grade path:")
    print()
    print("  • No QR codes, no Node.js bridge subprocess")
    print("  • Stable connection — no account-ban risk")
    print("  • Business account required (not personal WhatsApp)")
    print("  • Public webhook URL required (Cloudflare Tunnel, ngrok,")
    print("    or your own reverse proxy with TLS)")
    print()
    print("If you don't have a Meta app set up yet, follow these steps")
    print("FIRST, then come back and re-run this wizard:")
    print()
    print("  1. https://developers.facebook.com/apps → Create App")
    print("     → 'Connect with customers through WhatsApp'")
    print("  2. App Dashboard → WhatsApp → API Setup")
    print("  3. Click 'Generate access token' (temp 24h token is fine to")
    print("     start; switch to a System User permanent token later)")
    print()
    try:
        proceed = input("Press Enter to continue, or Ctrl+C to abort... ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nSetup cancelled.")
        return 1

    print()
    print("─" * 50)
    print("STEP 1 — Phone Number ID")
    print("─" * 50)
    current_phone_id = get_env_value("WHATSAPP_CLOUD_PHONE_NUMBER_ID") or None
    phone_id = _prompt_validated(
        "Phone Number ID",
        _validate_phone_number_id,
        current=current_phone_id,
        help_text=(
            "Found in: App Dashboard → WhatsApp → API Setup, in the\n"
            "'Send and receive messages' section.\n"
            "Look BELOW the 'From' dropdown — there's a 'Phone number ID'\n"
            "line with the value (15-17 digits, e.g. '7794189252778687').\n"
            "It is NOT the phone number itself (+1 555-...). That's the\n"
            "single most common setup mistake."
        ),
    )
    if not phone_id:
        if current_phone_id:
            phone_id = current_phone_id
            print(f"  ✓ Keeping existing: {phone_id}")
        else:
            print("\n✗ Phone Number ID is required. Aborting.")
            return 1
    else:
        save_env_value("WHATSAPP_CLOUD_PHONE_NUMBER_ID", phone_id)
        print(f"  ✓ Saved: {phone_id}")
    print()

    print("─" * 50)
    print("STEP 2 — Access Token")
    print("─" * 50)
    current_token = get_env_value("WHATSAPP_CLOUD_ACCESS_TOKEN") or None
    current_display = (current_token[:15] + "...") if current_token else None
    token = _prompt_validated(
        "Access Token",
        _validate_access_token,
        current=current_display,
        help_text=(
            "Two options for getting one:\n\n"
            "  (a) TEMP — App Dashboard → WhatsApp → API Setup →\n"
            "      'Generate access token' button. Lasts 24 hours.\n"
            "      Fine for testing today; you'll have to regenerate\n"
            "      tomorrow.\n\n"
            "  (b) PERMANENT (production) — System User token. One-time\n"
            "      setup, never expires:\n"
            "      • business.facebook.com → Settings → System users →\n"
            "        Add → Admin role\n"
            "      • Assign Assets → your app (Manage app), your\n"
            "        WhatsApp account (Manage WABAs)\n"
            "      • Generate token → expiration: Never → permissions:\n"
            "        business_management, whatsapp_business_messaging,\n"
            "        whatsapp_business_management\n\n"
            "Tokens start with 'EAA'."
        ),
    )
    # If they had a current token and just hit Enter, keep it.
    if not token:
        if current_token:
            token = current_token
            print("  ✓ Keeping existing token")
        else:
            print("\n✗ Access Token is required. Aborting.")
            return 1
    else:
        save_env_value("WHATSAPP_CLOUD_ACCESS_TOKEN", token)
        print("  ✓ Saved (token hidden)")
    print()

    print("─" * 50)
    print("STEP 3 — App Secret (required for webhook signature verification)")
    print("─" * 50)
    current_secret = get_env_value("WHATSAPP_CLOUD_APP_SECRET") or None
    current_secret_display = (current_secret[:8] + "...") if current_secret else None
    app_secret = _prompt_validated(
        "App Secret",
        _validate_app_secret,
        current=current_secret_display,
        help_text=(
            "Found in: App Dashboard → Settings → Basic →\n"
            "'App secret' field (click 'Show', enter your Facebook password).\n\n"
            "If 'Show' doesn't appear, you may need Admin role on the app.\n"
            "It's a 32-character lowercase hex string.\n\n"
            "Without the App Secret, inbound webhook POSTs are refused\n"
            "with HTTP 503 (we can't verify they actually came from Meta)."
        ),
    )
    if not app_secret:
        if current_secret:
            app_secret = current_secret
            print("  ✓ Keeping existing App Secret")
        else:
            print("\n⚠ Skipping App Secret — inbound webhooks will be refused")
            print("   until you set WHATSAPP_CLOUD_APP_SECRET manually.")
    else:
        save_env_value("WHATSAPP_CLOUD_APP_SECRET", app_secret)
        print("  ✓ Saved (secret hidden)")
    print()

    print("─" * 50)
    print("STEP 4 — App ID & WABA ID (optional, for analytics)")
    print("─" * 50)
    current_app_id = get_env_value("WHATSAPP_CLOUD_APP_ID") or None
    app_id = _prompt_validated(
        "App ID (optional, press Enter to skip)",
        lambda v: (True, None) if not v else _validate_app_id(v),
        current=current_app_id,
        help_text=(
            "Found in: App Dashboard → Settings → Basic → 'App ID' at the\n"
            "top of the page. Numeric, ~15-16 digits.\n"
            "Not required for messaging — useful only for analytics later."
        ),
    )
    if app_id:
        save_env_value("WHATSAPP_CLOUD_APP_ID", app_id)
        print(f"  ✓ Saved: {app_id}")
    elif current_app_id:
        print(f"  ✓ Keeping existing: {current_app_id}")

    current_waba_id = get_env_value("WHATSAPP_CLOUD_WABA_ID") or None
    waba_id = _prompt_validated(
        "WABA ID (optional, press Enter to skip)",
        lambda v: (True, None) if not v else _validate_waba_id(v),
        current=current_waba_id,
        help_text=(
            "WhatsApp Business Account ID. Found in: App Dashboard →\n"
            "WhatsApp → API Setup, near the top — 'WhatsApp Business\n"
            "Account ID'. Numeric, ~15+ digits.\n"
            "Not required for messaging — useful for analytics."
        ),
    )
    if waba_id:
        save_env_value("WHATSAPP_CLOUD_WABA_ID", waba_id)
        print(f"  ✓ Saved: {waba_id}")
    elif current_waba_id:
        print(f"  ✓ Keeping existing: {current_waba_id}")
    print()

    print("─" * 50)
    print("STEP 5 — Verify Token (auto-generated)")
    print("─" * 50)
    current_verify = get_env_value("WHATSAPP_CLOUD_VERIFY_TOKEN") or None
    if current_verify:
        print(f"  An existing verify token is already set ({current_verify[:8]}...).")
        try:
            regen = input("  Generate a new one? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            regen = "n"
        if regen in {"y", "yes"}:
            verify_token = secrets.token_urlsafe(32)
            save_env_value("WHATSAPP_CLOUD_VERIFY_TOKEN", verify_token)
            print(f"  ✓ New verify token: {verify_token}")
        else:
            verify_token = current_verify
            print("  ✓ Keeping existing verify token")
    else:
        verify_token = secrets.token_urlsafe(32)
        save_env_value("WHATSAPP_CLOUD_VERIFY_TOKEN", verify_token)
        print(f"  ✓ Generated: {verify_token}")
    print()
    print("  → COPY THIS TOKEN NOW. You'll paste it into Meta's webhook")
    print("    configuration dialog (next step).")
    print()

    print("─" * 50)
    print("STEP 6 — Recipient Allowlist")
    print("─" * 50)
    print()
    print("  Who is allowed to message the bot? (Comma-separated phone")
    print("  numbers with country code, no '+' / spaces / dashes. Use '*'")
    print("  to allow anyone — only safe if you've also configured Meta's")
    print("  recipient whitelist for app-development mode.)")
    print()
    current_allow = get_env_value("WHATSAPP_CLOUD_ALLOWED_USERS") or None
    allow_default = current_allow if current_allow else None
    try:
        allowed = input(
            f"  → Allowed users{' [' + allow_default + ']' if allow_default else ''}: "
        ).strip() or (allow_default or "")
    except (EOFError, KeyboardInterrupt):
        allowed = ""
    if allowed:
        # Light normalization — strip spaces and dashes from each entry.
        allowed = ",".join(
            re.sub(r"[\s\-+]", "", part) for part in allowed.split(",") if part.strip()
        )
        save_env_value("WHATSAPP_CLOUD_ALLOWED_USERS", allowed)
        print(f"  ✓ Saved: {allowed}")
    else:
        print("  ⚠ No allowlist — every inbound message will be denied.")
        print("    Re-run this wizard or set WHATSAPP_CLOUD_ALLOWED_USERS manually.")
    print()

    print("─" * 50)
    print("SETUP COMPLETE — Next steps")
    print("─" * 50)
    print()
    print("  Hermes needs a public HTTPS URL to receive WhatsApp messages.")
    print("  The recommended path is Cloudflare Tunnel (free, no port")
    print("  forwarding, no DNS setup).")
    print()
    print("    1. Install cloudflared (one-time, if you don't have it):")
    print("         Windows:  winget install Cloudflare.cloudflared")
    print("         macOS:    brew install cloudflared")
    print("         Linux:    https://github.com/cloudflare/cloudflared/releases")
    print()
    print("       Alternatives: ngrok, or your own domain + reverse proxy")
    print("       with TLS.")
    print()
    print("    2. Start the tunnel in a separate terminal:")
    print("         cloudflared tunnel --url http://localhost:8090")
    print("       Note the printed https://<random>.trycloudflare.com URL.")
    print()
    print("    3. Start the Hermes gateway in another terminal:")
    print("         hermes gateway")
    print()
    print("    4. Verify your local config is reachable. From a third")
    print("       terminal, with the tunnel URL substituted:")
    print()
    print("         curl 'https://YOUR-TUNNEL.trycloudflare.com/whatsapp/webhook?\\")
    print(f"               hub.mode=subscribe&hub.verify_token={verify_token}&\\")
    print("               hub.challenge=hello'")
    print()
    print("       Expected: HTTP 200 with body 'hello'.")
    print("       Also try: curl https://YOUR-TUNNEL.trycloudflare.com/health")
    print("       (should return JSON with verify_token_configured: true).")
    print()
    print("    5. Configure Meta to point at your tunnel:")
    print("         App Dashboard → WhatsApp → Configuration → Edit webhook")
    print("         Callback URL: <tunnel-url>/whatsapp/webhook")
    print(f"         Verify Token: {verify_token}")
    print("         → Click 'Verify and save'")
    print("         → Then 'Manage' webhook fields → subscribe to 'messages'")
    print()
    print("    6. Add your phone to Meta's recipient list:")
    print("         App Dashboard → WhatsApp → API Setup → 'To' →")
    print("         'Manage phone number list'")
    print()
    print("    7. DM the bot's test number from your phone.")
    print()
    print("─" * 50)
    print("Optional: polish your bot's WhatsApp profile")
    print("─" * 50)
    print()
    print("  WhatsApp shows a display name and profile picture for your bot")
    print("  in every chat header and contact list. These are set in Meta's")
    print("  Business Manager, not via this wizard — but here's where to do")
    print("  it once you're up and running:")
    print()
    effective_waba = waba_id or current_waba_id
    if effective_waba:
        print("    • Display name + profile picture:")
        print("        https://business.facebook.com/wa/manage/phone-numbers/"
              f"?waba_id={effective_waba}")
    else:
        print("    • Display name + profile picture:")
        print("        https://business.facebook.com/wa/manage/phone-numbers/")
        print("        (select your WhatsApp Business Account on that page)")
    print("        Display-name changes go through a ~24-48h Meta review.")
    print()
    print("    • About, description, website, hours, business category:")
    print("        Same page → click your phone number → 'Edit profile'.")
    print()
    print("    • Verified badge (the green check):")
    print("        Requires Meta's business verification process —")
    print("        Business Manager → Security Center → Start Verification.")
    print()
    print("  Docs: https://hermes-agent.nousresearch.com/docs/user-guide/")
    print("        messaging/whatsapp-cloud")
    print()
    return 0
