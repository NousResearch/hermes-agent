"""
``hermes photon ...`` CLI subcommands — registered by the plugin via
``ctx.register_cli_command()``.

Subcommands:

    setup              full first-time setup (device login + project + user + sidecar)
    status             show login + project + sidecar dep state
    install-sidecar    npm install inside plugins/platforms/photon/sidecar/
    telemetry          show or toggle Spectrum SDK telemetry (on/off)

The device-code login runs automatically as the first step of ``setup``;
there is no standalone ``login`` verb (matching how every other Hermes
gateway channel onboards through a single setup surface).

Photon uses the spectrum-ts gRPC stream for inbound — there is no webhook
to register, so there are no webhook subcommands.
"""
from __future__ import annotations

import argparse
import getpass
import os
import shutil
import subprocess
import sys
from pathlib import Path

from hermes_cli.colors import Colors, color

from . import auth as photon_auth

_SIDECAR_DIR = Path(__file__).parent / "sidecar"


# ---------------------------------------------------------------------------
# argparse wiring

def register_cli(parser: argparse.ArgumentParser) -> None:
    """Wire up `hermes photon ...` subcommands."""
    subs = parser.add_subparsers(dest="photon_command", required=False)

    p_setup = subs.add_parser(
        "setup",
        help="First-time setup (device login + project + user + sidecar)",
    )
    p_setup.add_argument("--project-name", default=None,
                         help="Project name (default: 'Hermes Agent')")
    p_setup.add_argument("--phone", default=None,
                         help="Your E.164 phone number (e.g. +15551234567)")
    p_setup.add_argument("--first-name", default=None)
    p_setup.add_argument("--last-name", default=None)
    p_setup.add_argument("--email", default=None)
    p_setup.add_argument("--no-browser", action="store_true",
                         help="Don't try to open a browser for device login; print the URL only")
    p_setup.add_argument("--skip-sidecar-install", action="store_true",
                         help="Skip `npm install` inside the sidecar directory")

    subs.add_parser("status", help="Show login + project + sidecar dep state")
    subs.add_parser("install-sidecar", help="Run npm install inside the sidecar directory")

    p_telemetry = subs.add_parser(
        "telemetry",
        help="Show or toggle Spectrum SDK telemetry (on/off)",
    )
    p_telemetry.add_argument(
        "state", nargs="?", choices=("on", "off"),
        help="Turn telemetry on or off (omit to show the current state)",
    )

    parser.set_defaults(func=dispatch)


# ---------------------------------------------------------------------------
# Dispatch

def dispatch(args: argparse.Namespace) -> int:
    sub = getattr(args, "photon_command", None)
    if sub is None:
        # No subcommand given — show status by default.
        return _cmd_status(args)
    if sub == "setup":
        return _cmd_setup(args)
    if sub == "status":
        return _cmd_status(args)
    if sub == "install-sidecar":
        return _cmd_install_sidecar(args)
    if sub == "telemetry":
        return _cmd_telemetry(args)
    print(f"unknown subcommand: {sub}", file=sys.stderr)
    return 2


# ---------------------------------------------------------------------------
# Subcommand handlers

def _run_device_login(args: argparse.Namespace) -> int:
    """Run the RFC 8628 device-code login flow and persist the token.

    Internal helper — invoked as the first step of ``setup``. There is
    no standalone ``hermes photon login`` command; Photon onboards
    through the single ``setup`` surface like every other channel.
    """
    def _print_code(code):
        target = code.verification_uri_complete or code.verification_uri
        print()
        print("┌─ Photon device login ────────────────────────────────────────")
        print(f"│  Open this URL:  {target}")
        print(f"│  Enter the code: {code.user_code}")
        print("│  (waiting for approval — Ctrl-C to cancel)")
        print("└──────────────────────────────────────────────────────────────")
        print()

    try:
        token = photon_auth.login_device_flow(
            open_browser=not args.no_browser,
            on_user_code=_print_code,
        )
    except Exception as e:
        print(f"login failed: {e}", file=sys.stderr)
        return 1
    # Don't print any portion of the token — even a prefix can help a
    # shoulder-surfer or accidentally leak into a screen recording.
    _ = token
    print(f"✓ logged in — token saved to {photon_auth._auth_json_path()}")
    return 0


def _cmd_setup(args: argparse.Namespace) -> int:
    # 1. Login (skip if we already have a token).
    token = photon_auth.load_photon_token()
    if not token:
        print("[1/5] No Photon token found — running device login...")
        rc = _run_device_login(args)
        if rc != 0:
            return rc
        token = photon_auth.load_photon_token()
        if not token:
            print("login completed but token was not stored", file=sys.stderr)
            return 1
    else:
        print("[1/5] Reusing existing Photon token")

    # 2. Find or create the "Hermes Agent" project.
    name = args.project_name or photon_auth.DEFAULT_PROJECT_NAME
    dashboard_id = photon_auth.load_dashboard_project_id()
    try:
        if dashboard_id:
            print("[2/5] Reusing configured Photon project")
        else:
            existing = photon_auth.find_project_by_name(token, name)
            if existing and existing.get("id"):
                dashboard_id = existing["id"]
                print(f"[2/5] Found existing project '{name}'")
            else:
                print(f"[2/5] Creating Photon project '{name}'...")
                created = photon_auth.create_project(token, name=name)
                dashboard_id = created.get("id")
                print("  ✓ project created")
    except Exception as e:
        print(f"project setup failed: {e}", file=sys.stderr)
        return 1
    if not dashboard_id:
        print("could not resolve a Photon project id", file=sys.stderr)
        return 1

    # 3. Rotate the project secret and persist creds (runtime -> ~/.hermes/.env,
    #    ids -> auth.json). Spectrum is always enabled and provisioned at
    #    create-time, and the dashboard project id *is* the Spectrum project id
    #    (ids unified), so there's nothing to enable — the id we already have is
    #    the Spectrum id.
    try:
        print("[3/5] Provisioning Spectrum credentials...")
        spectrum_id = dashboard_id
        secret = photon_auth.regenerate_project_secret(token, dashboard_id)
        photon_auth.store_project_credentials(
            spectrum_project_id=spectrum_id,
            project_secret=secret,
            dashboard_project_id=dashboard_id,
            name=name,
        )
        # spectrum_id is an opaque non-secret id; safe to show.
        print(f"  ✓ Spectrum ready (project id {spectrum_id}) — secret saved")
    except Exception as e:
        print(f"spectrum provisioning failed: {e}", file=sys.stderr)
        return 1

    # 4. Register the operator's phone number as a Spectrum user (idempotent).
    phone = args.phone or _prompt(
        color(
            "[4/5] Your iMessage phone number (E.164, e.g. +15551234567): ",
            Colors.CYAN,
        )
    )
    agent_number = None
    registered_phone = None
    registered_user_id = None
    if not phone:
        print("      Skipped user registration (no phone given). Re-run with --phone later.")
    else:
        # Name/email are optional and never prompted for — pass --first-name /
        # --email if you want them sent to the dashboard.
        first_name = args.first_name
        email = args.email
        try:
            user, created = photon_auth.register_user_if_absent(
                spectrum_id, secret,
                phone_number=phone,
                first_name=first_name,
                last_name=args.last_name,
                email=email,
            )
        except ValueError as e:
            print(f"      invalid phone number: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"      user registration failed: {e}", file=sys.stderr)
            return 1
        print("  ✓ phone registered" if created else "  ✓ phone already registered")
        registered_phone = phone
        registered_user_id = user.get("id")
        # The number to text the agent is the user's assigned iMessage line
        # (the dashboard's "TEXTS ON" column). On shared-number plans there is
        # no dedicated entry in /lines, so this per-user field is the source of
        # truth — and we already have it from the (reused) user object.
        agent_number = photon_auth.user_assigned_line(user)
        # Allowlist the operator and make their DM the cron home channel —
        # otherwise the gateway denies their own inbound messages
        # ("Unauthorized user") and has no default space for cron delivery.
        _autoconfigure_access(phone)

    # 5. Surface the agent's iMessage number (the number to text the agent).
    if not agent_number:
        # No per-user assignment — fall back to a dedicated line if the project
        # has one provisioned in its line inventory.
        try:
            line = photon_auth.get_imessage_line(token, dashboard_id)
            if line:
                agent_number = line.get("phoneNumber")
        except Exception as e:
            print(f"      (could not fetch the assigned line: {e})", file=sys.stderr)
    if agent_number:
        print()
        print(color("┌─ Your agent's iMessage number ───────────────────────────────", Colors.GREEN))
        print(
            color("│  📱 ", Colors.GREEN)
            + color(str(agent_number), Colors.GREEN, Colors.BOLD)
        )
        print(color("│  Text this number from your phone to talk to your agent.", Colors.GREEN))
        print(color("└──────────────────────────────────────────────────────────────", Colors.GREEN))
    else:
        print("      No iMessage line assigned yet — check the Photon dashboard.")
    if registered_phone:
        try:
            photon_auth.store_user_numbers(
                phone_number=registered_phone,
                assigned_phone_number=agent_number,
                user_id=str(registered_user_id) if registered_user_id else None,
                dashboard_project_id=dashboard_id,
            )
        except Exception as e:
            print(f"      (could not save Photon status metadata: {e})", file=sys.stderr)

    # 6. Sidecar deps (spectrum-ts).
    if args.skip_sidecar_install:
        print("[5/5] Skipping sidecar npm install (--skip-sidecar-install)")
    else:
        print("[5/5] Installing Node sidecar deps (spectrum-ts)...")
        rc = _install_sidecar()
        if rc != 0:
            return rc

    print()
    print("✓ Photon setup complete.")
    print("  Start the gateway:  hermes gateway start")
    return 0


def _autoconfigure_access(phone: str) -> None:
    """Allowlist the operator and set their DM as the cron home channel.

    Writes ``PHOTON_ALLOWED_USERS`` (so the gateway authorizes the operator's
    own inbound messages instead of denying them) and ``PHOTON_HOME_CHANNEL``
    (the default space for cron delivery) to the operator's E.164 number. Each
    is only filled when unset, so a hand-tuned allowlist / home channel is
    never clobbered on a re-run.
    """
    try:
        from hermes_cli.config import get_env_value, save_env_value
    except ImportError:
        return
    for key, label in (
        ("PHOTON_ALLOWED_USERS", "allowlisted your number"),
        ("PHOTON_HOME_CHANNEL", "set your DM as the cron home channel"),
    ):
        try:
            if get_env_value(key):
                print(f"      {key} already set — leaving it as-is.")
                continue
            save_env_value(key, phone)
            print(f"  ✓ {label} ({key})")
        except Exception as e:
            print(f"      could not set {key}: {e}", file=sys.stderr)


def _cmd_status(_args: argparse.Namespace) -> int:
    _refresh_status_numbers()
    # Defer the credential rows to auth.print_credential_summary — its emit
    # callback is the only sink that sees credential-derived strings, so
    # cli.py keeps zero taint flow according to CodeQL.
    photon_auth.print_credential_summary(print)
    node_bin = os.getenv("PHOTON_NODE_BIN") or shutil.which("node")
    sidecar_installed = (_SIDECAR_DIR / "node_modules").exists()
    print(f"  node binary         : {node_bin or '✗ missing (install Node 18+)'}")
    print(f"  sidecar deps        : {'✓ installed' if sidecar_installed else '✗ run `hermes photon install-sidecar`'}")
    print(f"  telemetry           : {'on' if _telemetry_enabled() else 'off'} (`hermes photon telemetry on|off`)")
    return 0


def _refresh_status_numbers() -> None:
    phone, assigned = photon_auth.load_user_numbers()
    if phone and assigned:
        return
    spectrum_id, project_secret = photon_auth.load_project_credentials()
    if not spectrum_id or not project_secret:
        return
    try:
        photon_auth.refresh_user_numbers(spectrum_id, project_secret)
    except Exception as e:
        print(f"      (could not refresh Photon user numbers: {e})", file=sys.stderr)


def _cmd_install_sidecar(_args: argparse.Namespace) -> int:
    return _install_sidecar()


def _telemetry_enabled() -> bool:
    """Read PHOTON_TELEMETRY from the env / ~/.hermes/.env.

    Mirrors the sidecar's truthy set (index.mjs) so the state shown here
    always matches what the sidecar will actually do.
    """
    try:
        from hermes_cli.config import get_env_value
        raw = get_env_value("PHOTON_TELEMETRY")
    except ImportError:
        raw = os.getenv("PHOTON_TELEMETRY")
    return (raw or "").strip().lower() in ("1", "true", "yes", "on")


def _cmd_telemetry(args: argparse.Namespace) -> int:
    state = getattr(args, "state", None)
    if state is None:
        print(f"Photon telemetry: {'on' if _telemetry_enabled() else 'off'}")
        print("  Toggle with `hermes photon telemetry on` / `hermes photon telemetry off`.")
        return 0
    try:
        from hermes_cli.config import save_env_value
        save_env_value("PHOTON_TELEMETRY", "true" if state == "on" else "false")
    except Exception as e:
        print(f"could not save PHOTON_TELEMETRY: {e}", file=sys.stderr)
        return 1
    print(f"✓ Spectrum telemetry turned {state} (PHOTON_TELEMETRY in ~/.hermes/.env)")
    print("  Restart the gateway for the sidecar to pick it up:  hermes gateway restart")
    return 0


def _install_sidecar() -> int:
    npm = shutil.which("npm") or "npm"
    if not shutil.which(npm):
        print(
            "npm is not on PATH. Install Node.js 18+ (https://nodejs.org/) "
            "and re-run.",
            file=sys.stderr,
        )
        return 1
    # spectrum-ts is pinned exactly in package.json/package-lock.json because
    # the SDK ships breaking majors (v2 removed defineFusorPlatform; v3
    # reworked space construction). Upgrades are deliberate: bump the pin,
    # migrate sidecar/index.mjs, re-run the photon tests — never `@latest`
    # (see README "Upgrading spectrum-ts"). `npm ci` installs the committed
    # lockfile verbatim; fall back to `npm install` when the lockfile is
    # missing or drifted (e.g. a dev checkout mid-upgrade).
    print(f"  $ cd {_SIDECAR_DIR} && {npm} ci")
    proc = subprocess.run(  # noqa: S603
        [npm, "ci"],
        cwd=str(_SIDECAR_DIR),
        check=False,
    )
    if proc.returncode != 0:
        print(f"  npm ci failed — falling back to:  {npm} install")
        proc = subprocess.run(  # noqa: S603
            [npm, "install"],
            cwd=str(_SIDECAR_DIR),
            check=False,
        )
    if proc.returncode != 0:
        print("npm install failed", file=sys.stderr)
    return proc.returncode


# ---------------------------------------------------------------------------
# Gateway-setup entry point
#
# `hermes gateway setup` discovers platforms via the registry and calls each
# entry's zero-arg ``setup_fn``. Photon registers this function so it appears
# in the unified setup wizard alongside every other channel — same onboarding
# surface, no Photon-specific detour. It runs the identical device-login +
# project + user + sidecar flow as ``hermes photon setup`` with interactive
# defaults (phone is prompted when stdin is a TTY).

def gateway_setup() -> None:
    """Run Photon first-time setup from the `hermes gateway setup` wizard."""
    args = argparse.Namespace(
        photon_command="setup",
        project_name=None,
        phone=None,
        first_name=None,
        last_name=None,
        email=None,
        no_browser=False,
        skip_sidecar_install=False,
    )
    _cmd_setup(args)


# ---------------------------------------------------------------------------
# Small interactive helpers

def _prompt(prompt: str, *, secret: bool = False) -> str:
    if not sys.stdin.isatty():
        return ""
    try:
        if secret:
            return getpass.getpass(prompt).strip()
        return input(prompt).strip()
    except (KeyboardInterrupt, EOFError):
        print()
        return ""
