"""
``hermes photon ...`` CLI subcommands — registered by the plugin via
``ctx.register_cli_command()``.

Subcommands:

    login              run the device-code OAuth flow
    quick-setup        guided setup + managed webhook tunnel registration
    setup              full first-time setup (login + project + user + sidecar)
    allow-phone        authorize another E.164 sender for Photon gateway use
    status             show login + project + sidecar dep state
    install-sidecar    npm install inside plugins/platforms/photon/sidecar/
    webhook register   register the local webhook URL with Photon
    webhook list       list registered webhooks
    webhook delete     delete a webhook by id
    webhook tunnel     manage the local Cloudflare Quick Tunnel
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

from . import auth as photon_auth
from . import tunnel as photon_tunnel

_SIDECAR_DIR = Path(__file__).parent / "sidecar"
_MIN_SPECTRUM_TS_VERSION = (1, 7, 2)
_PHONE_FORMAT = "+<country-code><number>"
_PHONE_ARG_PLACEHOLDER = f"'{_PHONE_FORMAT}'"


# ---------------------------------------------------------------------------
# argparse wiring

def register_cli(parser: argparse.ArgumentParser) -> None:
    """Wire up `hermes photon ...` subcommands."""
    subs = parser.add_subparsers(dest="photon_command", required=False)

    p_login = subs.add_parser("login", help="Authenticate with Photon (device flow)")
    p_login.add_argument("--no-browser", action="store_true",
                         help="Don't try to open a browser; print the URL only")
    p_login.add_argument("--debug-auth", action="store_true",
                         help="Print sanitized Photon auth exchange diagnostics")

    p_quick = subs.add_parser(
        "quick-setup",
        help="Guided setup with managed Cloudflare webhook tunnel",
    )
    p_quick.add_argument("--project-name", default=None, help="Project name (default: 'Hermes Agent')")
    p_quick.add_argument("--phone", default=None, help=f"Your E.164 phone number (format: {_PHONE_FORMAT})")
    p_quick.add_argument("--first-name", default=None)
    p_quick.add_argument("--last-name", default=None)
    p_quick.add_argument("--email", default=None)
    p_quick.add_argument("--no-browser", action="store_true")
    p_quick.add_argument("--new-project", action="store_true",
                         help="Create a new Photon dashboard project instead of adopting an existing one")
    p_quick.add_argument("--skip-sidecar-install", action="store_true",
                         help="Skip `npm install` inside the sidecar directory")

    p_setup = subs.add_parser("setup", help="First-time setup (login + project + user + sidecar)")
    p_setup.add_argument("--project-name", default=None, help="Project name (default: 'Hermes Agent')")
    p_setup.add_argument("--phone", default=None, help=f"Your E.164 phone number (format: {_PHONE_FORMAT})")
    p_setup.add_argument("--first-name", default=None)
    p_setup.add_argument("--last-name", default=None)
    p_setup.add_argument("--email", default=None)
    p_setup.add_argument("--no-browser", action="store_true")
    p_setup.add_argument("--new-project", action="store_true",
                         help="Create a new Photon dashboard project instead of adopting an existing one")
    p_setup.add_argument("--skip-sidecar-install", action="store_true",
                         help="Skip `npm install` inside the sidecar directory")

    p_allow = subs.add_parser(
        "allow-phone",
        help="Allow a phone number to control Hermes over Photon",
    )
    p_allow.add_argument("phone", help=f"E.164 phone number (format: {_PHONE_FORMAT})")

    subs.add_parser("status", help="Show login + project + sidecar dep state")
    subs.add_parser("diagnose-auth", help="Print sanitized Photon auth diagnostics")
    subs.add_parser("install-sidecar", help="Run npm install inside the sidecar directory")

    p_projects = subs.add_parser("projects", help="List or select Photon projects")
    project_subs = p_projects.add_subparsers(dest="photon_projects_command", required=True)
    project_subs.add_parser("list", help="List Photon dashboard projects")
    p_project_select = project_subs.add_parser("select", help="Bind Hermes to an existing Photon project")
    p_project_select.add_argument("project_id", help="Dashboard or Spectrum project id")

    p_hook = subs.add_parser("webhook", help="Manage Photon webhook registrations")
    hook_subs = p_hook.add_subparsers(dest="photon_webhook_command", required=True)
    p_hook_reg = hook_subs.add_parser("register", help="Register a webhook URL")
    p_hook_reg.add_argument("url", help="Publicly reachable URL Photon should POST to")
    hook_subs.add_parser("list", help="List registered webhooks for the current project")
    p_hook_del = hook_subs.add_parser("delete", help="Delete a webhook by id")
    p_hook_del.add_argument("webhook_id")
    p_tunnel = hook_subs.add_parser("tunnel", help="Manage a local Cloudflare Quick Tunnel")
    tunnel_subs = p_tunnel.add_subparsers(dest="photon_tunnel_command", required=True)
    tunnel_subs.add_parser("start", help="Start tunnel and register its Photon webhook")
    tunnel_subs.add_parser("status", help="Show managed tunnel status")
    tunnel_subs.add_parser("stop", help="Stop the managed tunnel")
    tunnel_subs.add_parser("logs", help="Show recent cloudflared tunnel logs")

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
    if sub == "quick-setup":
        return _cmd_quick_setup(args)
    if sub == "setup":
        return _cmd_setup(args)
    if sub == "allow-phone":
        return _cmd_allow_phone(args)
    if sub == "status":
        return _cmd_status(args)
    if sub == "diagnose-auth":
        return _cmd_diagnose_auth(args)
    if sub == "install-sidecar":
        return _cmd_install_sidecar(args)
    if sub == "projects":
        return _cmd_projects(args)
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
            on_debug=(
                _print_login_auth_debug
                if getattr(args, "debug_auth", False)
                else None
            ),
        )
    except photon_auth.PhotonDashboardAuthError as e:
        if not getattr(args, "debug_auth", False):
            print(
                "For sanitized endpoint diagnostics, retry with "
                "`hermes photon login --debug-auth`.",
                file=sys.stderr,
            )
        print(f"login failed: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"login failed: {e}", file=sys.stderr)
        return 1
    # Don't print any portion of the token — even a prefix can help a
    # shoulder-surfer or accidentally leak into a screen recording.
    _ = token
    print(f"✓ logged in — token saved to {photon_auth._env_path()}")
    return 0


def _cmd_quick_setup(args: argparse.Namespace) -> int:
    project_id, project_secret = photon_auth.load_project_credentials()
    if (
        not photon_auth.load_photon_token()
        and not (project_id and project_secret)
    ):
        print_login_first_guidance()
        return 1

    setattr(args, "auto_create_project", True)
    print("Photon quick setup")
    print("──────────────────")
    rc = _run_base_setup(args, total_steps=5)
    if rc != 0:
        return rc

    print("[5/5] Starting Cloudflare Quick Tunnel and registering webhook...")
    rc = _start_managed_tunnel_and_register()
    if rc != 0:
        return rc

    print()
    print("Photon quick setup complete.")
    print("  Next: verify everything is ready:")
    print("        hermes photon status")
    print("  Then start the gateway in foreground QA mode:")
    print("        hermes gateway run -v")
    print("  If the gateway is already running, restart it so it loads the new webhook secret:")
    print("        hermes gateway restart")
    print("  More details:")
    print(f"        {_docs_paths()}")
    return 0


def interactive_setup() -> None:
    """Entry point used by `hermes setup gateway` when Photon is selected."""
    project_id, project_secret = photon_auth.load_project_credentials()
    if (
        not photon_auth.load_photon_token()
        and not (project_id and project_secret)
    ):
        print_incomplete_setup_guidance()
        return

    args = argparse.Namespace(
        project_name=None,
        phone=None,
        first_name=None,
        last_name=None,
        email=None,
        no_browser=False,
        new_project=False,
        skip_sidecar_install=False,
    )
    rc = _cmd_quick_setup(args)
    if rc != 0:
        print_incomplete_setup_guidance()


def print_login_first_guidance() -> None:
    """Explain the required login-before-quick-setup order."""
    print()
    print("Photon quick setup needs a Photon login first.")
    print("  First:")
    print("        hermes photon login")
    print("  Then:")
    print(f"        hermes photon quick-setup --phone {_PHONE_ARG_PLACEHOLDER}")


def print_incomplete_setup_guidance() -> None:
    """Print explicit next steps when Photon setup did not finish."""
    print()
    print("Photon iMessage setup is not complete yet.")
    project_id, project_secret = photon_auth.load_project_credentials()
    if (
        not photon_auth.load_photon_token()
        and not (project_id and project_secret)
    ):
        print("  First:")
        print("        hermes photon login")
        print("  Then:")
        print(f"        hermes photon quick-setup --phone {_PHONE_ARG_PLACEHOLDER}")
    else:
        print("  Guided setup:")
        print(f"        hermes photon quick-setup --phone {_PHONE_ARG_PLACEHOLDER}")
    print("  Check exact status and next step:")
    print("        hermes photon status")
    print("  Docs:")
    print(f"        {_docs_paths()}")


def _cmd_setup(args: argparse.Namespace) -> int:
    rc = _run_base_setup(args, total_steps=4)
    if rc != 0:
        return rc
    print()
    print("✓ Photon setup complete.")
    print("  Next: create a managed webhook tunnel and register it with Photon:")
    print("        hermes photon webhook tunnel start")
    print("  Or register a production/user-owned URL manually:")
    print("        hermes photon webhook register https://YOUR-PUBLIC-URL/photon/webhook")
    print("  Then start the gateway in foreground QA mode:")
    print("        hermes gateway run -v")
    print("  For always-on local use:")
    print("        hermes gateway install --force")
    print("        hermes gateway start")
    return 0


def _run_base_setup(args: argparse.Namespace, *, total_steps: int) -> int:
    # 1. Login (skip if we already have a token).
    token = photon_auth.load_photon_token()
    existing_id, existing_secret = photon_auth.load_project_credentials()
    if not token and not (existing_id and existing_secret):
        print(f"[1/{total_steps}] No Photon token found — running device login...")
        rc = _cmd_login(args)
        if rc != 0:
            return rc
        token = photon_auth.load_photon_token()
        if not token:
            print("login completed but token was not stored", file=sys.stderr)
            return 1
        print("  Next: Hermes will reuse/adopt/create the Photon project.")
    elif existing_id and existing_secret:
        print(f"[1/{total_steps}] Reusing existing Photon project credentials")
        print("  Next: Hermes will bind your phone number to the Photon project.")
    else:
        print(f"[1/{total_steps}] Reusing existing Photon token")
        print("  Next: Hermes will verify the Photon project.")

    # 2. Resolve a project without silently duplicating dashboard resources.
    try:
        with photon_auth.setup_lock():
            project_id, project_secret = _resolve_setup_project(
                args, token, total_steps=total_steps,
            )
    except TimeoutError as e:
        print(f"setup is already running: {e}", file=sys.stderr)
        return 1
    if not (project_id and project_secret):
        return 1
    print("  Next: Hermes will bind your phone number to a shared Photon iMessage line.")

    # 3. Create a Spectrum user for the operator.
    phone = args.phone or _prompt(
        f"Your iMessage phone number (E.164, format {_PHONE_FORMAT}): "
    )
    if not phone:
        print(f"[3/{total_steps}] Skipped user creation (no phone given). Re-run with --phone later.")
    else:
        print(f"[3/{total_steps}] Creating shared Spectrum user...")
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
        if not _ensure_operator_phone_allowed(phone):
            return 1
    print("  Next: Hermes will verify/install the Node sidecar dependencies.")

    # 4. Sidecar deps.
    if args.skip_sidecar_install:
        print(f"[4/{total_steps}] Skipping sidecar npm install (--skip-sidecar-install)")
    else:
        print(f"[4/{total_steps}] Installing Node sidecar deps (spectrum-ts)...")
        rc = _install_sidecar()
        if rc != 0:
            return rc
    if total_steps > 4:
        print("  Next: Hermes will start a Cloudflare Quick Tunnel and register the webhook.")
    return 0


def _cmd_allow_phone(args: argparse.Namespace) -> int:
    return 0 if _ensure_operator_phone_allowed(args.phone, explicit=True) else 1


def _ensure_operator_phone_allowed(phone: str, *, explicit: bool = False) -> bool:
    try:
        status = photon_auth.ensure_phone_allowed(phone)
    except Exception as e:
        print(f"allow-phone failed: {e}", file=sys.stderr)
        return False

    if status == "added":
        print("  ✓ phone authorized for Photon gateway access")
    elif status == "allow_all":
        print("  ✓ Photon gateway access is already open via PHOTON_ALLOW_ALL_USERS")
    else:
        print("  ✓ phone already authorized for Photon gateway access")

    if explicit:
        print("  Next: restart the gateway if it is already running:")
        print("        hermes gateway restart")
    return True


def _resolve_setup_project(
    args: argparse.Namespace,
    token: str,
    *,
    total_steps: int = 4,
) -> tuple[str, str]:
    name = args.project_name or "Hermes Agent"
    if getattr(args, "new_project", False):
        print(f"[2/{total_steps}] Creating new Photon project '{name}' (spectrum=true, imessage)...")
        return _create_and_store_project(token, name=name, source="explicit-new")

    existing_id, existing_secret = photon_auth.load_project_credentials()
    if existing_id and existing_secret:
        print(f"[2/{total_steps}] Reusing existing Photon project")
        return existing_id, existing_secret

    print(f"[2/{total_steps}] Looking for an existing Photon project...")
    try:
        projects = photon_auth.list_projects(token)
    except photon_auth.PhotonDashboardAuthError as e:
        _handle_dashboard_auth_error(e)
        return "", ""
    except Exception as e:
        print(
            "could not list Photon projects, so no new project was created. "
            f"Re-run with --new-project to create one explicitly. Details: {e}",
            file=sys.stderr,
        )
        return "", ""

    candidates = photon_auth.reusable_projects(projects, preferred_name=name)
    if len(candidates) == 1:
        candidate = _refresh_project_details(token, candidates[0])
        project_id = str(candidate.get("spectrum_project_id") or "")
        project_secret = str(candidate.get("project_secret") or "")
        if project_id and project_secret:
            _store_selected_project(candidate, source="remote-adopted")
            print("  ✓ adopted existing Photon project")
            return project_id, project_secret
        _print_project_choices(
            candidates,
            "Found an existing compatible Photon project, but Photon did not "
            "return a project secret for it.",
        )
        print(
            "No new project was created. Select a project whose credentials are "
            "available, or re-run with --new-project to create a replacement.",
            file=sys.stderr,
        )
        return "", ""

    if len(candidates) > 1:
        _print_project_choices(
            candidates,
            "Multiple compatible Photon projects were found.",
        )
        print(
            "No new project was created. Run `hermes photon projects select <id>` "
            "or re-run with --new-project.",
            file=sys.stderr,
        )
        return "", ""

    auto_created = bool(getattr(args, "auto_create_project", False))
    if auto_created:
        print("  No matching Photon project found; creating one for Hermes.")
    elif not _confirm_new_project(name):
        print(
            "No Photon project configured. Re-run with --new-project to create one.",
            file=sys.stderr,
        )
        return "", ""

    print(f"[2/{total_steps}] Creating Photon project '{name}' (spectrum=true, imessage)...")
    return _create_and_store_project(
        token,
        name=name,
        source="auto-new" if auto_created else "confirmed-new",
    )


def _create_and_store_project(token: str, *, name: str, source: str) -> tuple[str, str]:
    try:
        data = photon_auth.create_project(token, name=name)
    except photon_auth.PhotonDashboardAuthError as e:
        _handle_dashboard_auth_error(e)
        return "", ""
    except Exception as e:
        print(f"create-project failed: {e}", file=sys.stderr)
        return "", ""

    data = _complete_created_project_credentials(token, data)
    normalized = photon_auth.normalize_project(data)
    project_id = str(normalized.get("spectrum_project_id") or data.get("id") or "")
    project_secret = str(normalized.get("project_secret") or "")
    if not project_id or not project_secret:
        print(
            "create-project did not return spectrumProjectId + "
            "projectSecret. Re-run after enabling Spectrum on the "
            "project, or open https://app.photon.codes/ to fetch the "
            "secret manually.",
            file=sys.stderr,
        )
        return "", ""

    extra = {
        "name": name,
        "source": source,
        "created_by": "hermes-agent",
    }
    dashboard_project_id = normalized.get("dashboard_project_id")
    if dashboard_project_id and dashboard_project_id != project_id:
        extra["dashboard_project_id"] = dashboard_project_id
    platforms = normalized.get("platforms") or ["imessage"]
    extra["platforms"] = platforms
    photon_auth.store_project_credentials(project_id, project_secret, **extra)
    print("  ✓ project provisioned (run `hermes photon status` to see the id)")
    return project_id, project_secret


def _complete_created_project_credentials(token: str, data: dict[str, Any]) -> dict[str, Any]:
    normalized = photon_auth.normalize_project(data)
    dashboard_project_id = str(
        normalized.get("dashboard_project_id") or data.get("id") or ""
    )
    if not dashboard_project_id:
        return data
    if normalized.get("spectrum_project_id") and normalized.get("project_secret"):
        return data

    try:
        details = photon_auth.get_project(token, dashboard_project_id)
    except photon_auth.PhotonDashboardAuthError as e:
        _handle_dashboard_auth_error(e)
        return data
    except Exception as e:
        print(
            "created Photon project, but could not fetch its Spectrum "
            f"credentials: {e}",
            file=sys.stderr,
        )
        details = {}
    if details:
        data = _merge_project_payloads(data, details)
        normalized = photon_auth.normalize_project(data)
        if normalized.get("spectrum_project_id") and normalized.get("project_secret"):
            print("  ✓ fetched Spectrum credentials for the new project")
            return data

    try:
        secret_data = photon_auth.regenerate_project_secret(
            token,
            dashboard_project_id,
        )
    except photon_auth.PhotonDashboardAuthError as e:
        _handle_dashboard_auth_error(e)
        return data
    except Exception as e:
        print(
            "created Photon project, but could not retrieve its Spectrum "
            f"secret: {e}",
            file=sys.stderr,
        )
        return data

    merged = _merge_project_payloads(data, secret_data)
    normalized = photon_auth.normalize_project(merged)
    if normalized.get("project_secret"):
        print("  ✓ retrieved Spectrum secret for the new project")
    return merged


def _refresh_project_details(token: str, project: dict[str, Any]) -> dict[str, Any]:
    normalized = photon_auth.normalize_project(project)
    dashboard_project_id = str(
        normalized.get("dashboard_project_id")
        or project.get("dashboard_project_id")
        or project.get("id")
        or ""
    )
    if not dashboard_project_id:
        return normalized
    if normalized.get("spectrum_project_id") and normalized.get("project_secret"):
        return normalized

    try:
        details = photon_auth.get_project(token, dashboard_project_id)
    except photon_auth.PhotonDashboardAuthError as e:
        _handle_dashboard_auth_error(e)
        return normalized
    except Exception as e:
        print(
            f"could not fetch Photon project details for {dashboard_project_id}: {e}",
            file=sys.stderr,
        )
        return normalized
    return photon_auth.normalize_project(
        _merge_project_payloads(project, details)
    )


def _merge_project_payloads(*payloads: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for payload in payloads:
        if isinstance(payload, dict):
            merged.update(payload)
    return merged


def _store_selected_project(project: dict[str, Any], *, source: str) -> None:
    project_id = str(project.get("spectrum_project_id") or "")
    project_secret = str(project.get("project_secret") or "")
    extra = {
        "name": project.get("name") or "Photon Project",
        "platforms": project.get("platforms") or [],
        "source": source,
        "selected_at": int(time.time()),
        "created_by": "hermes-agent",
    }
    dashboard_project_id = project.get("dashboard_project_id")
    if dashboard_project_id and dashboard_project_id != project_id:
        extra["dashboard_project_id"] = dashboard_project_id
    photon_auth.store_project_credentials(project_id, project_secret, **extra)


def _confirm_new_project(name: str) -> bool:
    if not sys.stdin.isatty():
        return False
    print()
    print("No existing Photon project was found for this Hermes setup.")
    print(f"Creating a new dashboard project named '{name}' may add another project to Photon.")
    answer = _prompt("Type CREATE NEW to continue: ")
    return answer == "CREATE NEW"


def _print_project_choices(projects: list[dict[str, Any]], heading: str) -> None:
    print()
    print(heading)
    for index, project in enumerate(projects, start=1):
        print(f"  {index}. {_project_summary(project)}")


def _project_summary(project: dict[str, Any]) -> str:
    name = project.get("name") or "(unnamed)"
    dashboard_id = project.get("dashboard_project_id") or "-"
    spectrum_id = project.get("spectrum_project_id") or "-"
    platforms = ",".join(project.get("platforms") or []) or "-"
    credentials = "yes" if project.get("project_secret") else "no"
    return (
        f"{name}  dashboard={dashboard_id}  spectrum={spectrum_id}  "
        f"platforms={platforms}  credentials={credentials}"
    )


def _cmd_status(_args: argparse.Namespace) -> int:
    # Defer the whole table to auth.print_credential_summary — its emit
    # callback is the only sink that sees credential-derived strings, so
    # cli.py keeps zero taint flow according to CodeQL.
    photon_auth.print_credential_summary(print)
    # The two non-credential rows live here so the helper stays purely
    # about credentials.
    node_bin = os.getenv("PHOTON_NODE_BIN") or shutil.which("node")
    sidecar_status = _sidecar_dependency_status()
    public_url = _get_env_value("PHOTON_WEBHOOK_PUBLIC_URL") or "✗ missing"
    tunnel_state = photon_tunnel.status()
    tunnel_label = _format_tunnel_status(tunnel_state)
    print(f"  node binary         : {node_bin or '✗ missing (install Node 20.18.1+)'}")
    print(f"  sidecar deps        : {sidecar_status}")
    print(f"  authorized phones   : {_photon_sender_access_status()}")
    print(f"  webhook public URL  : {public_url}")
    print(f"  managed tunnel      : {tunnel_label}")
    print(f"  next step           : {_next_status_step(sidecar_status, tunnel_state)}")
    print(f"  docs                : {_docs_paths()}")
    return 0


def _cmd_diagnose_auth(_args: argparse.Namespace) -> int:
    _print_auth_diagnostics(photon_auth.dashboard_auth_diagnostics())
    return 0


def _cmd_install_sidecar(_args: argparse.Namespace) -> int:
    rc = _install_sidecar()
    return rc


def _install_sidecar() -> int:
    npm = shutil.which("npm") or "npm"
    if not shutil.which(npm):
        print(
            "npm is not on PATH. Install Node.js 20.18.1+ (https://nodejs.org/) "
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


def _sidecar_dependency_status() -> str:
    if not (_SIDECAR_DIR / "node_modules").exists():
        return "✗ run `hermes photon install-sidecar`"

    version, problems = _installed_spectrum_ts()
    if not version:
        return "✗ spectrum-ts missing; run `hermes photon install-sidecar`"

    parsed = _parse_semver(version)
    if parsed is None:
        return f"⚠ spectrum-ts {version} installed; unable to verify version"
    if parsed < _MIN_SPECTRUM_TS_VERSION:
        return (
            f"✗ spectrum-ts {version} is too old; "
            "run `hermes photon install-sidecar`"
        )
    if problems:
        detail = str(problems[0]).splitlines()[0][:120]
        return (
            f"⚠ spectrum-ts {version} installed but npm reports {detail}; "
            "run `hermes photon install-sidecar`"
        )
    return f"✓ installed (spectrum-ts {version})"


def _installed_spectrum_ts() -> tuple[Optional[str], list[Any]]:
    npm = shutil.which("npm") or "npm"
    if not shutil.which(npm):
        return None, ["npm missing"]
    try:
        proc = subprocess.run(  # noqa: S603
            [npm, "ls", "spectrum-ts", "--depth=0", "--json"],
            cwd=str(_SIDECAR_DIR),
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return None, [str(e)]

    data: dict[str, Any] = {}
    if proc.stdout:
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError:
            data = {}
    deps = data.get("dependencies") or {}
    spectrum = deps.get("spectrum-ts") or {}
    version = spectrum.get("version")
    problems = data.get("problems") or []
    if proc.returncode != 0 and not problems:
        msg = (proc.stderr or "npm ls failed").strip()
        if msg:
            problems = [msg]
    return str(version) if version else None, list(problems)


def _parse_semver(version: str) -> Optional[tuple[int, int, int]]:
    match = re.match(r"^\s*(\d+)\.(\d+)\.(\d+)", version)
    if not match:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def _cmd_projects(args: argparse.Namespace) -> int:
    token = photon_auth.load_photon_token()
    if not token:
        print("not logged in — run `hermes photon login` first", file=sys.stderr)
        return 1

    sub = getattr(args, "photon_projects_command", None)
    try:
        projects = photon_auth.list_projects(token)
    except photon_auth.PhotonDashboardAuthError as e:
        _handle_dashboard_auth_error(e)
        return 1
    except Exception as e:
        print(f"project list failed: {e}", file=sys.stderr)
        return 1
    normalized = [photon_auth.normalize_project(project) for project in projects]

    if sub == "list":
        if not normalized:
            print("No Photon projects found.")
            return 0
        print("Photon projects")
        for project in normalized:
            print("  " + _project_summary(project))
        return 0

    if sub == "select":
        requested = str(args.project_id)
        matches = [
            project for project in normalized
            if requested in {
                str(project.get("dashboard_project_id") or ""),
                str(project.get("spectrum_project_id") or ""),
            }
        ]
        if not matches:
            print(f"project not found: {requested}", file=sys.stderr)
            return 1
        if len(matches) > 1:
            _print_project_choices(matches, "Multiple projects matched that id.")
            return 1
        project = _refresh_project_details(token, matches[0])
        if not (project.get("spectrum_enabled") and project.get("imessage_enabled")):
            print(
                "selected project is not a Spectrum iMessage project",
                file=sys.stderr,
            )
            return 1
        if not (project.get("spectrum_project_id") and project.get("project_secret")):
            print(
                "selected project cannot be adopted because Photon did not "
                "return spectrumProjectId + projectSecret for it",
                file=sys.stderr,
            )
            return 1
        _store_selected_project(project, source="manual-select")
        print("✓ selected Photon project")
        return 0

    print(f"unknown projects subcommand: {sub}", file=sys.stderr)
    return 2


def _handle_dashboard_auth_error(exc: photon_auth.PhotonDashboardAuthError) -> None:
    diagnostics = photon_auth.dashboard_auth_diagnostics()
    if diagnostics.get("token", {}).get("present"):
        _print_auth_diagnostics(diagnostics, stream=sys.stderr)
    photon_auth.clear_photon_token()
    print(str(exc), file=sys.stderr)
    print(
        "Cleared the saved Photon login token. Run `hermes photon login`, "
        "then retry the Photon command.",
        file=sys.stderr,
    )


def _print_auth_diagnostics(
    diagnostics: dict[str, Any],
    *,
    stream: Any = sys.stdout,
) -> None:
    token = diagnostics.get("token") or {}
    print("Photon auth diagnostics", file=stream)
    print("───────────────────────", file=stream)
    print(f"  env path        : {diagnostics.get('env_path')}", file=stream)
    print(f"  dashboard host  : {diagnostics.get('dashboard_host')}", file=stream)
    if diagnostics.get("candidate_source"):
        print(
            f"  token source    : {diagnostics.get('candidate_source')}",
            file=stream,
        )
    if token.get("present"):
        print(
            "  token           : present "
            f"(len={token.get('length')}, dots={token.get('dot_count')}, "
            f"jwt={_yes_no(bool(token.get('looks_jwt')))})",
            file=stream,
        )
    else:
        print("  token           : missing", file=stream)
    checks = diagnostics.get("checks") or []
    if checks:
        print("  endpoint checks :", file=stream)
        for check in checks:
            status = check.get("status")
            state = "ok" if check.get("ok") else "fail"
            detail = check.get("detail") or ""
            print(
                f"    - {check.get('name')} {check.get('path')} -> "
                f"{status} {state}; {detail}",
                file=stream,
            )


def _print_login_auth_debug(event: dict[str, Any]) -> None:
    kind = event.get("event")
    if kind == "device-token-response":
        token = event.get("token") or {}
        print("Photon login debug", file=sys.stderr)
        print("──────────────────", file=sys.stderr)
        print(
            f"  device token POST : {event.get('status')} "
            f"json={_yes_no(bool(event.get('body_is_json')))}",
            file=sys.stderr,
        )
        print(
            "  body keys         : "
            f"{_format_key_list(event.get('body_keys') or [])}",
            file=sys.stderr,
        )
        print(
            "  data keys         : "
            f"{_format_key_list(event.get('data_keys') or [])}",
            file=sys.stderr,
        )
        print(
            "  session keys      : "
            f"{_format_key_list(event.get('session_keys') or [])}",
            file=sys.stderr,
        )
        print(
            "  user object       : "
            f"{_yes_no(bool(event.get('user_present')))}",
            file=sys.stderr,
        )
        print(
            "  set-auth-token    : "
            f"{_yes_no(bool(event.get('has_set_auth_token_header')))}",
            file=sys.stderr,
        )
        print(
            "  selected token    : "
            f"{event.get('access_token_source')} "
            f"(len={token.get('length')}, dots={token.get('dot_count')}, "
            f"jwt={_yes_no(bool(token.get('looks_jwt')))})",
            file=sys.stderr,
        )
        candidates = event.get("candidates") or []
        if candidates:
            print(
                "  token candidates : "
                + _format_token_candidates(candidates),
                file=sys.stderr,
            )
        return

    if kind == "dashboard-validation":
        _print_auth_diagnostics(event, stream=sys.stderr)
        return

    print(f"Photon auth debug: {kind or 'unknown event'}", file=sys.stderr)


def _format_key_list(keys: list[Any]) -> str:
    return ", ".join(str(key) for key in keys) if keys else "-"


def _format_token_candidates(candidates: list[Any]) -> str:
    parts = []
    for candidate in candidates[:8]:
        if not isinstance(candidate, dict):
            continue
        shape = candidate.get("token") or {}
        parts.append(
            f"{candidate.get('source')}("
            f"len={shape.get('length')},"
            f"dots={shape.get('dot_count')},"
            f"jwt={_yes_no(bool(shape.get('looks_jwt')))})"
        )
    return ", ".join(parts) if parts else "-"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _cmd_webhook(args: argparse.Namespace) -> int:
    sub = getattr(args, "photon_webhook_command", None)
    if sub == "tunnel" and getattr(args, "photon_tunnel_command", None) != "start":
        return _cmd_webhook_tunnel(args)

    project_id, project_secret = photon_auth.load_project_credentials()
    if not (project_id and project_secret):
        print(
            f"no Photon project configured — run `hermes photon quick-setup --phone {_PHONE_ARG_PLACEHOLDER}` first",
            file=sys.stderr,
        )
        return 1

    if sub == "register":
        return _register_webhook_url(project_id, project_secret, args.url)

    if sub == "tunnel":
        return _cmd_webhook_tunnel(args)

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


def _cmd_webhook_tunnel(args: argparse.Namespace) -> int:
    sub = getattr(args, "photon_tunnel_command", None)
    if sub == "start":
        return _start_managed_tunnel_and_register()

    if sub == "status":
        state = photon_tunnel.status()
        print("Photon managed webhook tunnel")
        print("─────────────────────────────")
        print(f"  status              : {_format_tunnel_status(state)}")
        print(f"  public URL          : {state.get('public_url') or '✗ missing'}")
        print(f"  webhook URL         : {state.get('webhook_url') or '✗ missing'}")
        print(f"  state file          : {state.get('state_path')}")
        print(f"  log file            : {state.get('log_path')}")
        print("  Next: " + (
            "hermes photon webhook tunnel start"
            if not state.get("running")
            else "hermes photon status"
        ))
        return 0

    if sub == "stop":
        result = photon_tunnel.stop()
        print(result.get("message") or "managed tunnel stopped")
        print("  Next: hermes photon webhook tunnel start")
        return 0

    if sub == "logs":
        logs = photon_tunnel.tail_logs()
        if not logs:
            print(f"No cloudflared logs yet ({photon_tunnel.log_path()})")
            return 0
        print(logs)
        return 0

    print(f"unknown webhook tunnel subcommand: {sub}", file=sys.stderr)
    return 2


def _start_managed_tunnel_and_register() -> int:
    old_state = photon_tunnel.status()
    old_managed_url = (
        str(old_state.get("webhook_url") or "")
        if old_state.get("managed")
        else ""
    )

    result = photon_tunnel.start(on_install=print)
    if not result.success:
        print(f"cloudflared tunnel failed: {result.error}", file=sys.stderr)
        _print_cloudflared_install_help()
        if result.log_path:
            print(f"  Logs: {result.log_path}", file=sys.stderr)
        return 1

    action = "Reusing" if result.reused else "Started"
    print(f"  ✓ {action.lower()} Cloudflare Quick Tunnel")
    print(f"  public URL: {result.public_url}")
    print(f"  webhook URL: {result.webhook_url}")

    project_id, project_secret = photon_auth.load_project_credentials()
    if not (project_id and project_secret):
        print(
            f"no Photon project configured — run `hermes photon quick-setup --phone {_PHONE_ARG_PLACEHOLDER}` first",
            file=sys.stderr,
        )
        return 1

    try:
        existing_hooks = photon_auth.list_webhooks(project_id, project_secret)
    except Exception as e:
        print(
            "could not check existing Photon webhooks, so no webhook was registered. "
            f"Details: {e}",
            file=sys.stderr,
        )
        return 1

    if (
        old_managed_url
        and old_managed_url != result.webhook_url
        and photon_tunnel.is_trycloudflare_url(old_managed_url)
    ):
        _delete_matching_webhook(
            project_id,
            project_secret,
            existing_hooks,
            old_managed_url,
            reason="old managed trycloudflare.com webhook",
        )
        try:
            existing_hooks = photon_auth.list_webhooks(project_id, project_secret)
        except Exception:
            existing_hooks = [
                hook for hook in existing_hooks
                if _webhook_url(hook) != old_managed_url
            ]

    return _register_webhook_url(
        project_id,
        project_secret,
        result.webhook_url,
        existing_hooks=existing_hooks,
        recreate_managed_without_secret=True,
    )


def _register_webhook_url(
    project_id: str,
    project_secret: str,
    url: str,
    *,
    existing_hooks: Optional[list] = None,
    recreate_managed_without_secret: bool = False,
) -> int:
    if existing_hooks is None:
        try:
            existing_hooks = photon_auth.list_webhooks(project_id, project_secret)
        except Exception as e:
            print(
                "could not check existing Photon webhooks, so no new webhook "
                f"was registered. Details: {e}",
                file=sys.stderr,
            )
            return 1

    matching_hooks = [
        hook for hook in existing_hooks
        if _webhook_url(hook) == url
    ]
    if matching_hooks:
        if _webhook_secret_present():
            if not _save_public_webhook_url(url):
                return 1
            print("✓ webhook URL already registered; keeping existing local signing secret")
            print("  Next: restart the gateway if it was already running:")
            print("        hermes gateway restart")
            return 0
        if recreate_managed_without_secret and photon_tunnel.is_trycloudflare_url(url):
            _delete_matching_webhook(
                project_id,
                project_secret,
                matching_hooks,
                url,
                reason="managed webhook with missing local signing secret",
            )
        else:
            print(
                "webhook URL is already registered, but PHOTON_WEBHOOK_SECRET "
                "is not set locally. Photon only returns the signing secret at "
                "registration time. Delete or recreate the webhook in the "
                "Photon dashboard, then save the new signing secret locally.",
                file=sys.stderr,
            )
            return 1

    try:
        data = photon_auth.register_webhook(
            project_id, project_secret, webhook_url=url
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
    if not _save_public_webhook_url(url):
        return 1
    print("  ✓ webhook public URL saved")
    print("  Next: restart the gateway if it was already running:")
    print("        hermes gateway restart")
    return 0


def _delete_matching_webhook(
    project_id: str,
    project_secret: str,
    hooks: list,
    url: str,
    *,
    reason: str,
) -> None:
    for hook in hooks:
        if _webhook_url(hook) != url:
            continue
        webhook_id = _webhook_id(hook)
        if not webhook_id:
            continue
        try:
            photon_auth.delete_webhook(
                project_id, project_secret, webhook_id=webhook_id
            )
        except Exception as e:
            print(f"could not delete {reason} {webhook_id}: {e}", file=sys.stderr)
            continue
        print(f"  ✓ deleted {reason}: {webhook_id}")


def _webhook_id(webhook: Any) -> str:
    if not isinstance(webhook, dict):
        return ""
    for key in ("id", "webhookId", "webhook_id", "uuid"):
        value = webhook.get(key)
        if value:
            return str(value)
    return ""


def _webhook_url(webhook: Any) -> str:
    if not isinstance(webhook, dict):
        return ""
    return str(webhook.get("webhookUrl") or webhook.get("url") or "")


def _get_env_value(key: str) -> Optional[str]:
    try:
        from hermes_cli.config import get_env_value  # type: ignore
        return get_env_value(key)
    except Exception:
        return os.getenv(key)


def _truthy_env(key: str) -> bool:
    return (_get_env_value(key) or "").strip().lower() in {"true", "1", "yes"}


def _photon_sender_access_configured() -> bool:
    if _truthy_env("PHOTON_ALLOW_ALL_USERS") or _truthy_env("GATEWAY_ALLOW_ALL_USERS"):
        return True
    allowed = photon_auth.load_allowed_phone_numbers()
    return bool(allowed)


def _photon_sender_access_status() -> str:
    if _truthy_env("PHOTON_ALLOW_ALL_USERS"):
        return "open (PHOTON_ALLOW_ALL_USERS=true)"
    if _truthy_env("GATEWAY_ALLOW_ALL_USERS"):
        return "open (GATEWAY_ALLOW_ALL_USERS=true)"
    allowed = photon_auth.load_allowed_phone_numbers()
    if "*" in allowed:
        return "open (PHOTON_ALLOWED_USERS=*)"
    if allowed:
        return f"{len(allowed)} configured"
    return "✗ none (unknown senders will request pairing)"


def _save_public_webhook_url(url: str) -> bool:
    try:
        from hermes_cli.config import save_env_value  # type: ignore
        save_env_value("PHOTON_WEBHOOK_PUBLIC_URL", url)
        return True
    except Exception as e:
        print(f"could not save PHOTON_WEBHOOK_PUBLIC_URL: {e}", file=sys.stderr)
        return False


def _webhook_secret_present() -> bool:
    return bool(_get_env_value("PHOTON_WEBHOOK_SECRET"))


def _format_tunnel_status(state: dict[str, Any]) -> str:
    if state.get("running"):
        return f"✓ running (pid {state.get('pid')})"
    if state.get("public_url") or state.get("webhook_url"):
        return "✗ stopped (run `hermes photon webhook tunnel start`)"
    return "✗ not started"


def _next_status_step(sidecar_status: str, tunnel_state: dict[str, Any]) -> str:
    project_id, project_secret = photon_auth.load_project_credentials()
    if not photon_auth.load_photon_token() and not (project_id and project_secret):
        return "hermes photon login"
    if not (project_id and project_secret):
        return f"hermes photon quick-setup --phone {_PHONE_ARG_PLACEHOLDER}"
    if sidecar_status.startswith("✗"):
        return "hermes photon install-sidecar"
    public_url = _get_env_value("PHOTON_WEBHOOK_PUBLIC_URL") or ""
    if not (_webhook_secret_present() and public_url):
        return "hermes photon webhook tunnel start"
    if photon_tunnel.is_trycloudflare_url(public_url) and not tunnel_state.get("running"):
        return "hermes photon webhook tunnel start"
    if not _photon_sender_access_configured():
        return f"hermes photon allow-phone {_PHONE_ARG_PLACEHOLDER}"
    return "hermes gateway run -v  (or `hermes gateway restart` if already running)"


def _docs_paths() -> str:
    return "plugins/platforms/photon/README.md; website/docs/user-guide/messaging/photon.md"


def _print_cloudflared_install_help() -> None:
    print("Install cloudflared manually, then re-run:", file=sys.stderr)
    print("  macOS (Homebrew): brew install cloudflared", file=sys.stderr)
    print("  Other platforms: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/", file=sys.stderr)
    print("Manual webhook path:", file=sys.stderr)
    print(f"  1. Expose {photon_tunnel.local_url()} with your reverse proxy.", file=sys.stderr)
    print(f"  2. hermes photon webhook register https://YOUR-PUBLIC-URL{photon_tunnel.webhook_path()}", file=sys.stderr)


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
