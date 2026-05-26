"""
``hermes photon ...`` CLI subcommands — registered by the plugin via
``ctx.register_cli_command()``.

Subcommands:

    login              run the device-code OAuth flow
    setup              full first-time setup (login + project + user + sidecar)
    status             show login + project + sidecar dep state
    install-sidecar    npm install inside plugins/platforms/photon/sidecar/
    webhook register   register the local webhook URL with Photon
    webhook list       list registered webhooks
    webhook delete     delete a webhook by id
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import auth as photon_auth

_SIDECAR_DIR = Path(__file__).parent / "sidecar"


# ---------------------------------------------------------------------------
# argparse wiring

def register_cli(parser: argparse.ArgumentParser) -> None:
    """Wire up `hermes photon ...` subcommands."""
    subs = parser.add_subparsers(dest="photon_command", required=False)

    p_login = subs.add_parser("login", help="Authenticate with Photon (device flow)")
    p_login.add_argument("--no-browser", action="store_true",
                         help="Don't try to open a browser; print the URL only")

    p_setup = subs.add_parser("setup", help="First-time setup (login + project + user + sidecar)")
    p_setup.add_argument("--project-name", default=None, help="Project name (default: 'Hermes Agent')")
    p_setup.add_argument("--phone", default=None, help="Your E.164 phone number (e.g. +15551234567)")
    p_setup.add_argument("--first-name", default=None)
    p_setup.add_argument("--last-name", default=None)
    p_setup.add_argument("--email", default=None)
    p_setup.add_argument("--no-browser", action="store_true")
    p_setup.add_argument("--skip-sidecar-install", action="store_true",
                         help="Skip `npm install` inside the sidecar directory")

    subs.add_parser("status", help="Show login + project + sidecar dep state")
    subs.add_parser("install-sidecar", help="Run npm install inside the sidecar directory")

    p_hook = subs.add_parser("webhook", help="Manage Photon webhook registrations")
    hook_subs = p_hook.add_subparsers(dest="photon_webhook_command", required=True)
    p_hook_reg = hook_subs.add_parser("register", help="Register a webhook URL")
    p_hook_reg.add_argument("url", help="Publicly reachable URL Photon should POST to")
    hook_subs.add_parser("list", help="List registered webhooks for the current project")
    p_hook_del = hook_subs.add_parser("delete", help="Delete a webhook by id")
    p_hook_del.add_argument("webhook_id")

    parser.set_defaults(func=dispatch)


# ---------------------------------------------------------------------------
# Dispatch

def dispatch(args: argparse.Namespace) -> int:
    sub = getattr(args, "photon_command", None)
    if sub is None:
        # No subcommand given — show status by default.
        return _cmd_status(args)
    if sub == "login":
        return _cmd_login(args)
    if sub == "setup":
        return _cmd_setup(args)
    if sub == "status":
        return _cmd_status(args)
    if sub == "install-sidecar":
        return _cmd_install_sidecar(args)
    if sub == "webhook":
        return _cmd_webhook(args)
    print(f"unknown subcommand: {sub}", file=sys.stderr)
    return 2


# ---------------------------------------------------------------------------
# Subcommand handlers

def _cmd_login(args: argparse.Namespace) -> int:
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
        print("[1/4] No Photon token found — running device login...")
        rc = _cmd_login(args)
        if rc != 0:
            return rc
        token = photon_auth.load_photon_token()
        if not token:
            print("login completed but token was not stored", file=sys.stderr)
            return 1
    else:
        print("[1/4] Reusing existing Photon token")

    # 2. Create (or surface existing) project.
    existing_id, existing_secret = photon_auth.load_project_credentials()
    has_existing_project = bool(existing_id and existing_secret)
    if has_existing_project:
        project_id, project_secret = existing_id, existing_secret
        # `project_id` is a Photon-assigned UUID, not a secret — but we
        # keep the print terse to avoid CodeQL flow noise.
        print("[2/4] Reusing existing Photon project")
    else:
        name = args.project_name or "Hermes Agent"
        print(f"[2/4] Creating Photon project '{name}' (spectrum=true, imessage)...")
        try:
            data = photon_auth.create_project(token, name=name)
        except Exception as e:
            print(f"create-project failed: {e}", file=sys.stderr)
            return 1
        project_id = data.get("spectrumProjectId") or data.get("id") or ""
        project_secret = data.get("projectSecret") or ""
        if not project_id or not project_secret:
            print(
                "create-project did not return spectrumProjectId + "
                "projectSecret. Re-run after enabling Spectrum on the "
                "project, or open https://app.photon.codes/ to fetch the "
                "secret manually.",
                file=sys.stderr,
            )
            return 1
        photon_auth.store_project_credentials(project_id, project_secret, name=name)
        print("  ✓ project provisioned (run `hermes photon status` to see the id)")

    # 3. Create a Spectrum user for the operator.
    phone = args.phone or _prompt(
        "Your iMessage phone number (E.164, e.g. +15551234567): "
    )
    if not phone:
        print("[3/4] Skipped user creation (no phone given). Re-run with --phone later.")
    else:
        print("[3/4] Creating shared Spectrum user...")
        try:
            photon_auth.create_user(
                project_id, project_secret,
                phone_number=phone,
                first_name=args.first_name,
                last_name=args.last_name,
                email=args.email,
            )
        except Exception as e:
            print(f"create-user failed: {e}", file=sys.stderr)
            return 1
        print("  ✓ user created — check `hermes photon status` or the dashboard for the assigned iMessage line")

    # 4. Sidecar deps.
    if args.skip_sidecar_install:
        print("[4/4] Skipping sidecar npm install (--skip-sidecar-install)")
    else:
        print("[4/4] Installing Node sidecar deps (spectrum-ts)...")
        rc = _install_sidecar()
        if rc != 0:
            return rc

    print()
    print("✓ Photon setup complete.")
    print("  Next: register a webhook URL Photon can reach:")
    print("        hermes photon webhook register https://YOUR-PUBLIC-URL/photon/webhook")
    print("  Then start the gateway:")
    print("        hermes gateway start --platform photon")
    return 0


def _cmd_status(_args: argparse.Namespace) -> int:
    # Defer the whole table to auth.print_credential_summary — its emit
    # callback is the only sink that sees credential-derived strings, so
    # cli.py keeps zero taint flow according to CodeQL.
    photon_auth.print_credential_summary(print)
    # The two non-credential rows live here so the helper stays purely
    # about credentials.
    node_bin = os.getenv("PHOTON_NODE_BIN") or shutil.which("node")
    sidecar_installed = (_SIDECAR_DIR / "node_modules").exists()
    print(f"  node binary         : {node_bin or '✗ missing (install Node 18+)'}")
    print(f"  sidecar deps        : {'✓ installed' if sidecar_installed else '✗ run `hermes photon install-sidecar`'}")
    return 0


def _cmd_install_sidecar(_args: argparse.Namespace) -> int:
    rc = _install_sidecar()
    return rc


def _install_sidecar() -> int:
    npm = shutil.which("npm") or "npm"
    if not shutil.which(npm):
        print(
            "npm is not on PATH. Install Node.js 18+ (https://nodejs.org/) "
            "and re-run.",
            file=sys.stderr,
        )
        return 1
    print(f"  $ cd {_SIDECAR_DIR} && {npm} install")
    proc = subprocess.run(  # noqa: S603
        [npm, "install"],
        cwd=str(_SIDECAR_DIR),
        check=False,
    )
    if proc.returncode != 0:
        print("npm install failed", file=sys.stderr)
    return proc.returncode


def _cmd_webhook(args: argparse.Namespace) -> int:
    sub = getattr(args, "photon_webhook_command", None)
    project_id, project_secret = photon_auth.load_project_credentials()
    if not (project_id and project_secret):
        print(
            "no Photon project configured — run `hermes photon setup` first",
            file=sys.stderr,
        )
        return 1

    if sub == "register":
        try:
            data = photon_auth.register_webhook(
                project_id, project_secret, webhook_url=args.url
            )
        except Exception as e:
            print(f"register failed: {e}", file=sys.stderr)
            return 1
        # The helper does all the formatting + writing; cli.py never
        # touches the signing-secret value, the path it was written
        # to, or even the redacted-response dict. on_summary is a
        # plain printer callback.
        ok = photon_auth.persist_webhook_signing_secret(data, on_summary=print)
        if not ok:
            print(
                "‼  Photon returned no signing secret in the response, "
                "or the file write failed. Inspect your home directory "
                "permissions and re-run; do not retry without first "
                "deleting the orphaned webhook from the Photon dashboard.",
                file=sys.stderr,
            )
            return 1
        return 0

    if sub == "list":
        try:
            data = photon_auth.list_webhooks(project_id, project_secret)
        except Exception as e:
            print(f"list failed: {e}", file=sys.stderr)
            return 1
        print(json.dumps(data, indent=2))
        return 0

    if sub == "delete":
        try:
            photon_auth.delete_webhook(
                project_id, project_secret, webhook_id=args.webhook_id
            )
        except Exception as e:
            print(f"delete failed: {e}", file=sys.stderr)
            return 1
        print(f"deleted webhook {args.webhook_id}")
        return 0

    print(f"unknown webhook subcommand: {sub}", file=sys.stderr)
    return 2


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
