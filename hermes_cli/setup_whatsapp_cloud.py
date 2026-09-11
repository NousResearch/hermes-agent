"""Interactive setup wizard for the WhatsApp Cloud API adapter: the 6 Meta credentials + recipient
allowlist, an auto-generated verify token, and exact follow-up instructions for what can't happen
inside the wizard process (cloudflared, gateway, Meta's webhook dashboard, recipient list).

Intentionally does NOT smoke-test the webhook: the gateway and the tunnel both run in separate
processes the user starts AFTER this wizard exits, so any in-wizard probe would fail."""

from __future__ import annotations
from hermes_cli.cli_output import line_input

import re
import secrets
import sys
from typing import Optional


# --- Field-shape validators: each returns (ok, reason_if_not_ok) so obviously-malformed input is
# rejected before saving, sparing a round trip with Meta's 401 / 400 errors.

def _rules_validator(label: str, rules):
    """Validator from ``(fails(stripped), reason)`` rules, checked in order after the required check.
    ``reason`` may be a str or a ``fn(stripped) -> str``."""
    def validate(value: str) -> tuple[bool, Optional[str]]:
        if not value:
            return False, f"{label} is required"
        s = value.strip()
        for fails, reason in rules:
            if fails(s):
                return False, reason(s) if callable(reason) else reason
        return True, None
    return validate


def _numeric_id_validator(label: str, lo: int, hi: int, expected: str):
    """Validator for a numeric Meta ID whose digit count must fall in [lo, hi]."""
    return _rules_validator(label, (
        (lambda s: not s.isdigit(), f"{label} must be numeric"),
        (lambda s: len(s) < lo or len(s) > hi, f"{label} looks wrong (expected {expected})")))


# Phone Number ID is a 15-17 digit numeric ID assigned by Meta — NOT a phone number. The #1 setup
# mistake is pasting the actual phone number (10-11 digits), which Graph rejects with "Object with
# ID does not exist."
_validate_phone_number_id = _rules_validator("Phone Number ID", (
    (lambda s: not s.isdigit(), "Phone Number ID must be numeric (no '+', spaces, or dashes)"),
    (lambda s: 10 <= len(s) <= 12,  # phone-number-sized: almost certainly the number itself
     "That looks like a phone number — but this field needs the "
     "Phone Number ID (Meta's internal ID, 15-17 digits, e.g. "
     "'7794189252778687'). Look just BELOW the 'From' dropdown in "
     "API Setup → it's labelled 'Phone number ID'."),
    (lambda s: len(s) < 13, "Phone Number ID looks too short (expected 13-18 digits)"),
    (lambda s: len(s) > 20, "Phone Number ID looks too long (expected 13-18 digits)")))
# WABA ID: similar length range as Phone Number ID. App ID: typically 15-16 digits.
_validate_waba_id = _numeric_id_validator("WABA ID", 10, 25, "10-25 digits")
_validate_app_id = _numeric_id_validator("App ID", 13, 20, "15-16 digits")
# App Secret is a 32-character lowercase hex string.
_validate_app_secret = _rules_validator("App Secret", (
    (lambda s: not re.fullmatch(r"[0-9a-f]+", s.lower()),
     "App Secret should be a hex string (only digits 0-9 and "
     "letters a-f). Make sure you copied the 'App secret' from "
     "Settings → Basic, not some other token."),
    (lambda s: len(s) != 32, lambda s: f"App Secret should be exactly 32 hex characters (got {len(s)})")))

# Common paste mistakes for the access-token field: (prefixes, what it actually is).
_FOREIGN_TOKEN_PREFIXES = (
    (("sk-",), "That's an OpenAI key (starts with 'sk-'), not a Meta "
               "WhatsApp access token. Meta tokens start with 'EAA'."),
    (("xoxb-", "xoxp-"), "That's a Slack token, not a Meta WhatsApp access token. "
                         "Meta tokens start with 'EAA'."),
    (("ghp_", "gho_"), "That's a GitHub token, not a Meta WhatsApp access "
                       "token. Meta tokens start with 'EAA'."))


def _not_meta_token_reason(s: str) -> str:
    for prefixes, reason in _FOREIGN_TOKEN_PREFIXES:
        if s.startswith(prefixes):
            return reason
    return ("Meta WhatsApp access tokens start with 'EAA'. Check that "
            "you're copying from the right place (API Setup → 'Generate "
            "access token', or Business Settings → System Users → "
            "'Generate token' for a permanent one).")


# Meta access tokens (temp and System User alike) start with ``EAA``, 100-300+ chars.
_validate_access_token = _rules_validator("Access token", (
    (lambda s: not s.startswith("EAA"), _not_meta_token_reason),
    (lambda s: len(s) < 100, lambda s: f"Access token looks too short ({len(s)} chars, expected 100+)")))

# --- Prompt helpers

def _prompt(message: str, default: Optional[str] = None, secret: bool = False) -> str:
    """Read one line; "" on EOF / Ctrl+C / empty. ``default`` is shown but NOT auto-applied so a
    real value stays distinguishable from a masked preview; ``secret`` reads via ``getpass``."""
    try:
        suffix = f" [{default}]" if default else ""
        if secret and sys.stdin.isatty():
            import getpass
            return getpass.getpass(f"{message}{suffix} (input hidden): ").strip()
        return line_input(f"{message}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return ""


def _prompt_validated(
    message: str, validator, *, current: Optional[str] = None, help_text: Optional[str] = None,
    secret: bool = False) -> Optional[str]:
    """Repeat the prompt until a valid value or the user gives up (None: empty answer, Ctrl+C).
    ``current`` is shown as the default on wizard re-runs."""
    if help_text:
        for line in help_text.strip().splitlines():
            print(f"  {line}")
    attempts = 0
    while True:
        attempts += 1
        value = _prompt(f"  → {message}", default=current, secret=secret)
        if not value:
            return None
        ok, reason = validator(value)
        if ok:
            return value.strip()
        print(f"    ✗ {reason}")
        if attempts >= 3:
            try:
                if not input("    Try again, or press Enter to skip: ").strip():
                    return None
            except (EOFError, KeyboardInterrupt):
                return None
            attempts = 0


# --- Wizard

def _header(title: str) -> None:
    _lines("─" * 50, title, "─" * 50)


def _lines(*lines: str) -> None:
    """print() each line; ``""`` yields a blank line."""
    for line in lines:
        print(line)


def _persist(env_var: str, value: Optional[str], current: Optional[str],
             saved: str = "  ✓ Saved: {v}", kept: str = "  ✓ Keeping existing: {v}") -> Optional[str]:
    """Save ``value`` to .env, else keep ``current``; prints the matching line. Returns the effective
    value (None when neither exists). ``{v}`` in ``saved``/``kept`` is the value."""
    from hermes_cli.config import save_env_value
    if value:
        save_env_value(env_var, value)
        print(saved.format(v=value))
        return value
    if current:
        print(kept.format(v=current))
    return current or None


# Credential steps 1-3: (step title, env var, prompt label, validator, secret, preview chars of
# the existing value shown as default (0 = full), "saved" line, "kept" line, lines printed when
# nothing is configured, abort-when-missing, help text).
_CREDENTIAL_STEPS = (
    ("STEP 1 — Phone Number ID", "WHATSAPP_CLOUD_PHONE_NUMBER_ID", "Phone Number ID",
     _validate_phone_number_id, False, 0, "  ✓ Saved: {v}", "  ✓ Keeping existing: {v}",
     ("\n✗ Phone Number ID is required. Aborting.",), True,
     "Found in: App Dashboard → WhatsApp → API Setup, in the\n"
     "'Send and receive messages' section.\n"
     "Look BELOW the 'From' dropdown — there's a 'Phone number ID'\n"
     "line with the value (15-17 digits, e.g. '7794189252778687').\n"
     "It is NOT the phone number itself (+1 555-...). That's the\n"
     "single most common setup mistake."),
    ("STEP 2 — Access Token", "WHATSAPP_CLOUD_ACCESS_TOKEN", "Access Token",
     _validate_access_token, True, 15, "  ✓ Saved (token hidden)", "  ✓ Keeping existing token",
     ("\n✗ Access Token is required. Aborting.",), True,
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
     "Tokens start with 'EAA'."),
    ("STEP 3 — App Secret (required for webhook signature verification)", "WHATSAPP_CLOUD_APP_SECRET",
     "App Secret", _validate_app_secret, True, 8, "  ✓ Saved (secret hidden)",
     "  ✓ Keeping existing App Secret",
     ("\n⚠ Skipping App Secret — inbound webhooks will be refused",
      "   until you set WHATSAPP_CLOUD_APP_SECRET manually."), False,
     "Found in: App Dashboard → Settings → Basic →\n"
     "'App secret' field (click 'Show', enter your Facebook password).\n\n"
     "If 'Show' doesn't appear, you may need Admin role on the app.\n"
     "It's a 32-character lowercase hex string.\n\n"
     "Without the App Secret, inbound webhook POSTs are refused\n"
     "with HTTP 503 (we can't verify they actually came from Meta)."))

# Optional step-4 IDs: (prompt label, env var, validator, help text).
_OPTIONAL_ID_STEPS = (
    ("App ID (optional, press Enter to skip)", "WHATSAPP_CLOUD_APP_ID", _validate_app_id,
     "Found in: App Dashboard → Settings → Basic → 'App ID' at the\n"
     "top of the page. Numeric, ~15-16 digits.\n"
     "Not required for messaging — useful only for analytics later."),
    ("WABA ID (optional, press Enter to skip)", "WHATSAPP_CLOUD_WABA_ID", _validate_waba_id,
     "WhatsApp Business Account ID. Found in: App Dashboard →\n"
     "WhatsApp → API Setup, near the top — 'WhatsApp Business\n"
     "Account ID'. Numeric, ~15+ digits.\n"
     "Not required for messaging — useful for analytics."))


def _credential_step(step) -> bool:
    """Run one _CREDENTIAL_STEPS entry. Returns True when the wizard must abort."""
    from hermes_cli.config import get_env_value
    title, env_var, label, validator, secret, preview, saved, kept, missing, required, help_text = step
    _header(title)
    current = get_env_value(env_var) or None
    shown = (current[:preview] + "...") if (preview and current) else current
    value = _prompt_validated(label, validator, current=shown, secret=secret, help_text=help_text)
    if _persist(env_var, value, current, saved, kept) is None:
        _lines(*missing)
        if required:
            return True
    print()
    return False


def _step_optional_ids() -> dict:
    """STEP 4: optional App ID / WABA ID. Returns {env var: effective value or None}."""
    from hermes_cli.config import get_env_value
    _header("STEP 4 — App ID & WABA ID (optional, for analytics)")
    ids = {}
    for label, env_var, validator, help_text in _OPTIONAL_ID_STEPS:
        current = get_env_value(env_var) or None
        value = _prompt_validated(label, validator, current=current, help_text=help_text)
        ids[env_var] = _persist(env_var, value, current)
    print()
    return ids


def _step_verify_token() -> str:
    """STEP 5: generate (or keep) the webhook verify token; returns the effective token."""
    from hermes_cli.config import get_env_value, save_env_value
    _header("STEP 5 — Verify Token (auto-generated)")
    verify_token = get_env_value("WHATSAPP_CLOUD_VERIFY_TOKEN") or None
    regen = "y"
    if verify_token:
        print(f"  An existing verify token is already set ({verify_token[:8]}...).")
        try:
            regen = input("  Generate a new one? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            regen = "n"
    if regen in {"y", "yes"}:
        label = "New verify token" if verify_token else "Generated"
        verify_token = secrets.token_urlsafe(32)
        save_env_value("WHATSAPP_CLOUD_VERIFY_TOKEN", verify_token)
        print(f"  ✓ {label}: {verify_token}")
    else:
        print("  ✓ Keeping existing verify token")
    _lines("", "  → COPY THIS TOKEN NOW. You'll paste it into Meta's webhook",
           "    configuration dialog (next step).", "")
    return verify_token


def _step_allowlist() -> None:
    """STEP 6: recipient allowlist (spaces/dashes/'+' stripped from each entry)."""
    from hermes_cli.config import get_env_value, save_env_value
    _header("STEP 6 — Recipient Allowlist")
    _lines(
        "", "  Who is allowed to message the bot? (Comma-separated phone",
        "  numbers with country code, no '+' / spaces / dashes. Use '*'",
        "  to allow anyone — only safe if you've also configured Meta's",
        "  recipient whitelist for app-development mode.)", "")
    allow_default = get_env_value("WHATSAPP_CLOUD_ALLOWED_USERS") or None
    try:
        allowed = line_input(
            f"  → Allowed users{' [' + allow_default + ']' if allow_default else ''}: "
        ).strip() or (allow_default or "")
    except (EOFError, KeyboardInterrupt):
        allowed = ""
    if allowed:
        allowed = ",".join(re.sub(r"[\s\-+]", "", part) for part in allowed.split(",") if part.strip())
        save_env_value("WHATSAPP_CLOUD_ALLOWED_USERS", allowed)
        print(f"  ✓ Saved: {allowed}")
    else:
        _lines("  ⚠ No allowlist — every inbound message will be denied.",
               "    Re-run this wizard or set WHATSAPP_CLOUD_ALLOWED_USERS manually.")
    print()


def run_whatsapp_cloud_setup() -> int:
    """Interactive wizard for the WhatsApp Cloud API adapter. Returns 0 on success, 1 on abort."""
    _lines(
        "", "⚕ WhatsApp Business Cloud API Setup", "=" * 50, "",
        "This wizard configures Hermes to talk to WhatsApp via Meta's",
        "official Cloud API. It's the production-grade path:", "",
        "  • No QR codes, no Node.js bridge subprocess",
        "  • Stable connection — no account-ban risk",
        "  • Business account required (not personal WhatsApp)",
        "  • Public webhook URL required (Cloudflare Tunnel, ngrok,",
        "    or your own reverse proxy with TLS)", "",
        "If you don't have a Meta app set up yet, follow these steps",
        "FIRST, then come back and re-run this wizard:", "",
        "  1. https://developers.facebook.com/apps → Create App",
        "     → 'Connect with customers through WhatsApp'",
        "  2. App Dashboard → WhatsApp → API Setup",
        "  3. Click 'Generate access token' (temp 24h token is fine to",
        "     start; switch to a System User permanent token later)", "")
    try:
        input("Press Enter to continue, or Ctrl+C to abort... ")
    except (EOFError, KeyboardInterrupt):
        print("\nSetup cancelled.")
        return 1

    print()
    if any(_credential_step(step) for step in _CREDENTIAL_STEPS):
        return 1

    effective_waba = _step_optional_ids()["WHATSAPP_CLOUD_WABA_ID"]
    verify_token = _step_verify_token()
    _step_allowlist()

    _header("SETUP COMPLETE — Next steps")
    _lines(
        "", "  Hermes needs a public HTTPS URL to receive WhatsApp messages.",
        "  The recommended path is Cloudflare Tunnel (free, no port",
        "  forwarding, no DNS setup).", "",
        "    1. Install cloudflared (one-time, if you don't have it):",
        "         Windows:  winget install Cloudflare.cloudflared",
        "         macOS:    brew install cloudflared",
        "         Linux:    https://github.com/cloudflare/cloudflared/releases", "",
        "       Alternatives: ngrok, or your own domain + reverse proxy",
        "       with TLS.", "",
        "    2. Start the tunnel in a separate terminal:",
        "         cloudflared tunnel --url http://localhost:8090",
        "       Note the printed https://<random>.trycloudflare.com URL.", "",
        "    3. Start the Hermes gateway in another terminal:",
        "         hermes gateway", "",
        "    4. Verify your local config is reachable. From a third",
        "       terminal, with the tunnel URL substituted:", "",
        "         curl 'https://YOUR-TUNNEL.trycloudflare.com/whatsapp/webhook?\\",
        f"               hub.mode=subscribe&hub.verify_token={verify_token}&\\",
        "               hub.challenge=hello'", "",
        "       Expected: HTTP 200 with body 'hello'.",
        "       Also try: curl https://YOUR-TUNNEL.trycloudflare.com/health",
        "       (should return JSON with verify_token_configured: true).", "",
        "    5. Configure Meta to point at your tunnel:",
        "         App Dashboard → WhatsApp → Configuration → Edit webhook",
        "         Callback URL: <tunnel-url>/whatsapp/webhook",
        f"         Verify Token: {verify_token}",
        "         → Click 'Verify and save'",
        "         → Then 'Manage' webhook fields → subscribe to 'messages'", "",
        "    6. Add your phone to Meta's recipient list:",
        "         App Dashboard → WhatsApp → API Setup → 'To' →",
        "         'Manage phone number list'", "",
        "    7. DM the bot's test number from your phone.", "")
    _header("Optional: polish your bot's WhatsApp profile")
    _lines(
        "", "  WhatsApp shows a display name and profile picture for your bot",
        "  in every chat header and contact list. These are set in Meta's",
        "  Business Manager, not via this wizard — but here's where to do",
        "  it once you're up and running:", "",
        "    • Display name + profile picture:",
        "        https://business.facebook.com/wa/manage/phone-numbers/"
        + (f"?waba_id={effective_waba}" if effective_waba else ""))
    if not effective_waba:
        print("        (select your WhatsApp Business Account on that page)")
    _lines(
        "        Display-name changes go through a ~24-48h Meta review.", "",
        "    • About, description, website, hours, business category:",
        "        Same page → click your phone number → 'Edit profile'.", "",
        "    • Verified badge (the green check):",
        "        Requires Meta's business verification process —",
        "        Business Manager → Security Center → Start Verification.", "",
        "  Docs: https://hermes-agent.nousresearch.com/docs/user-guide/",
        "        messaging/whatsapp-cloud", "")
    return 0
