"""
``hermes photon ...`` CLI subcommands — registered by the plugin via
``ctx.register_cli_command()``.

Subcommands:

    login              run the device-code OAuth flow
    setup              reconcile Photon setup for the fixed hermes-agent project
    phones             list/add/remove Photon project phones
    allow-phone        low-level local sender authorization only
    status             show Photon setup/runtime invariant state
    reset              clear local Photon state, or --all to clear auth too
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home

from . import auth as photon_auth
from . import phone_management

_SIDECAR_DIR = Path(__file__).parent / "sidecar"
_MIN_SPECTRUM_TS_VERSION = (1, 17, 1)
_FIXED_PROJECT_NAME = "hermes-agent"
_PHONE_FORMAT = "+<country-code><number>"
_PHONE_ARG_PLACEHOLDER = f"'{_PHONE_FORMAT}'"
_PHOTON_ASSIGNED_PHONE_ENV = "PHOTON_ASSIGNED_PHONE_NUMBER"
_PHOTON_RUNTIME_RESET_ENV_KEYS = (
    "PHOTON_PROJECT_ID",
    "PHOTON_PROJECT_SECRET",
    "PHOTON_PROJECT_NAME",
    "PHOTON_DASHBOARD_PROJECT_ID",
    "PHOTON_OPERATOR_PHONE",
    _PHOTON_ASSIGNED_PHONE_ENV,
)
_PHOTON_ALL_RESET_ENV_KEYS = (
    *_PHOTON_RUNTIME_RESET_ENV_KEYS,
    "PHOTON_DASHBOARD_TOKEN",
    "PHOTON_ALLOWED_USERS",
    "PHOTON_ALLOW_ALL_USERS",
    "PHOTON_HOME_CHANNEL",
    "PHOTON_HOME_CHANNEL_NAME",
)


@dataclass
class _PhotonSetupContext:
    args: argparse.Namespace
    hermes_home: Path
    env_path: Path
    project_name: str
    project_id: str = ""
    project_secret: str = ""
    dashboard_token: str = ""
    dashboard_project_id: str = ""
    dashboard_project_name: str = ""
    operator_phone: Optional[str] = None
    assigned_phone_number: Optional[str] = None
    runtime_secrets_changed: bool = False
    verbose: bool = False
    log_offsets: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "_PhotonSetupContext":
        hermes_home = Path(get_hermes_home())
        return cls(
            args=args,
            hermes_home=hermes_home,
            env_path=photon_auth._env_path(),
            project_name=_FIXED_PROJECT_NAME,
            verbose=bool(getattr(args, "verbose", False)),
        )


class _FailedInvariant(RuntimeError):
    def __init__(
        self,
        *,
        step: str,
        summary: str,
        expected: str,
        observed: Any,
        evidence: dict[str, Any],
        repair: str,
        logs: Optional[dict[str, list[str]]] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(summary)
        self.step = step
        self.summary = summary
        self.expected = expected
        self.observed = observed
        self.evidence = evidence
        self.repair = repair
        self.logs = logs or {}
        self.verbose = verbose


# ---------------------------------------------------------------------------
# argparse wiring

def register_cli(parser: argparse.ArgumentParser) -> None:
    """Wire up `hermes photon ...` subcommands."""
    subs = parser.add_subparsers(
        dest="photon_command",
        required=False,
        metavar="command",
    )

    p_login = subs.add_parser("login", help="Authenticate with Photon (device flow)")
    p_login.add_argument("--no-browser", action="store_true",
                         help="Don't try to open a browser; print the URL only")
    p_login.add_argument("--debug-auth", action="store_true",
                         help="Print sanitized Photon auth exchange diagnostics")

    p_setup = subs.add_parser(
        "setup",
        help="Set up Photon for the fixed hermes-agent project",
    )
    p_setup.add_argument("phone", help=f"Your E.164 phone number (format: {_PHONE_FORMAT})")
    p_setup.add_argument("--first-name", default=None)
    p_setup.add_argument("--last-name", default=None)
    p_setup.add_argument("--email", default=None)
    p_setup.add_argument("--no-browser", action="store_true")
    p_setup.add_argument("--skip-adapter-install", action="store_true",
                         help="Skip `npm install` inside the sidecar directory")
    p_setup.add_argument("-v", "--verbose", action="store_true",
                         help="Stream existing gateway/Photon logs while setup waits")

    p_allow = subs.add_parser(
        "allow-phone",
        help="Authorize a phone locally only; does not create a Photon user",
    )
    p_allow.add_argument("phone", help=f"E.164 phone number (format: {_PHONE_FORMAT})")

    p_phones = subs.add_parser("phones", help="Manage phones on the Photon project")
    phone_subs = p_phones.add_subparsers(
        dest="photon_phones_command",
        required=True,
        metavar="command",
    )
    phone_subs.add_parser("list", help="List Photon project phones")
    p_phones_add = phone_subs.add_parser("add", help="Add a phone to Photon and Hermes access")
    p_phones_add.add_argument("phone", type=_phone_arg, help=f"E.164 phone number (format: {_PHONE_FORMAT})")
    p_phones_add.add_argument("-v", "--verbose", action="store_true",
                              help="Show underlying Photon error details")
    p_phones_remove = phone_subs.add_parser(
        "remove",
        help="Remove a phone from Photon and Hermes access",
    )
    p_phones_remove.add_argument("phone", type=_phone_arg, help=f"E.164 phone number (format: {_PHONE_FORMAT})")
    p_phones_remove.add_argument("-v", "--verbose", action="store_true",
                                 help="Show underlying Photon error details")

    subs.add_parser("status", help="Show Photon setup/runtime invariant state")
    p_reset = subs.add_parser("reset", help="Reset Photon setup state")
    p_reset.add_argument(
        "--all",
        action="store_true",
        help="After confirmation, clear Photon auth state too",
    )

    p_projects = subs.add_parser("projects", help="List Photon projects")
    project_subs = p_projects.add_subparsers(dest="photon_projects_command", required=True)
    project_subs.add_parser("list", help="List Photon dashboard projects")

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
    if sub == "allow-phone":
        return _cmd_allow_phone(args)
    if sub == "phones":
        return _cmd_phones(args)
    if sub == "status":
        return _cmd_status(args)
    if sub == "reset":
        return _cmd_reset(args)
    if sub == "projects":
        return _cmd_projects(args)
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


def _cmd_setup(args: argparse.Namespace) -> int:
    title = "Photon setup"
    print(title)
    print("─" * len(title))
    ctx = _PhotonSetupContext.from_args(args)
    if ctx.verbose:
        _print_setup_log_paths(ctx)
    _init_log_offsets(ctx)

    try:
        with photon_auth.setup_lock():
            _run_setup_reconciler(ctx)
    except TimeoutError as e:
        failure = _failed_invariant(
            ctx,
            step="setup lock",
            summary="another Photon setup process is already running",
            expected="exclusive access to Photon setup state",
            observed=str(e),
            repair="wait for the other setup to finish, then rerun setup",
        )
        _finalize_failed_invariant_logs(failure, ctx)
        _print_failed_invariant(failure)
        return 1
    except _FailedInvariant as e:
        _finalize_failed_invariant_logs(e, ctx)
        _print_failed_invariant(e)
        return 1
    except Exception as e:
        failure = _failed_invariant(
            ctx,
            step="setup",
            summary="unexpected Photon setup failure",
            expected="all Photon runtime invariants reconciled",
            observed=f"{type(e).__name__}: {e}",
            repair="rerun with `hermes photon status`; if it repeats, inspect IMPLEMENTATION_ERRORS.md and gateway logs",
        )
        _finalize_failed_invariant_logs(failure, ctx)
        _print_failed_invariant(failure)
        return 1

    print()
    _print_setup_reconciled(ctx)
    return 0


def _cmd_reset(args: argparse.Namespace) -> int:
    reset_all = bool(getattr(args, "all", False))
    hermes_home = Path(get_hermes_home())
    env_path = photon_auth._env_path()
    project_id, _project_secret = photon_auth.load_project_credentials()

    if reset_all and not _confirm_reset_all(hermes_home, project_id or ""):
        print("Photon reset aborted.")
        return 1

    print("Photon reset")
    print("────────────")
    print(f"  Hermes home : {hermes_home}")
    print(f"  env path    : {env_path}")
    if project_id:
        print(f"  project     : {project_id}")
    else:
        print("  project     : ✗ none configured")

    if reset_all:
        _clear_photon_env_keys(_PHOTON_ALL_RESET_ENV_KEYS)
        print("  auth        : dashboard token removed if it was present")
        print("Photon reset complete.")
        return 0

    _clear_photon_env_keys(_PHOTON_RUNTIME_RESET_ENV_KEYS)
    print("  auth        : dashboard token kept")
    print("Photon local reset complete.")
    print("  Remote Photon projects and users were not deleted.")
    return 0


def _confirm_reset_all(hermes_home: Path, project_id: str) -> bool:
    print(_reset_all_confirmation_text(hermes_home, project_id))
    try:
        answer = input().strip()
    except EOFError:
        return False
    return answer == "PHOTON"


def _reset_all_confirmation_text(hermes_home: Path, project_id: str) -> str:
    project_label = project_id or "(none configured)"
    return "\n".join([
        "This will reset Photon for Hermes home:",
        f"  {hermes_home}",
        "",
        "Project:",
        f"  {project_label}",
        "",
        (
            "It clears local Photon credentials, gateway settings, and the "
            "dashboard token."
        ),
        "It will not delete Photon dashboard projects, users, or gateway state.",
        "",
        "Type PHOTON to continue:",
    ])


def _clear_photon_env_keys(keys: tuple[str, ...]) -> None:
    removed = [key for key in keys if _remove_env_value(key)]
    if removed:
        print(f"  env         : removed {', '.join(removed)}")
    else:
        print("  env         : no Photon env keys needed removal")


def _run_setup_reconciler(ctx: _PhotonSetupContext) -> None:
    token = _ensure_dashboard_auth(ctx)
    ctx.dashboard_token = token
    _ensure_fixed_spectrum_project(ctx, token)
    _ensure_operator_phone(ctx)
    _ensure_sidecar_ready(ctx)
    _ensure_photon_gateway_platform_enabled(ctx)
    ctx.runtime_secrets_changed = False
    _report_gateway_handoff(ctx)


def _ensure_dashboard_auth(ctx: _PhotonSetupContext) -> str:
    print("[auth] Validating Photon dashboard login...")
    token = photon_auth.load_photon_token()
    if token:
        try:
            photon_auth.validate_photon_token(token)
            print("  ✓ dashboard token is valid for Photon project APIs")
            return token
        except photon_auth.PhotonDashboardAuthError:
            photon_auth.clear_photon_token()
            print("  saved dashboard token is invalid; running device login")
            token = None
        except Exception as e:
            if _http_status(e) in {401, 403}:
                photon_auth.clear_photon_token()
                print("  saved dashboard token was rejected; running device login")
                token = None
            else:
                raise _failed_invariant(
                    ctx,
                    step="dashboard auth",
                    summary="could not validate Photon dashboard token",
                    expected="saved token can access Photon project APIs",
                    observed=f"{type(e).__name__}: {e}",
                    evidence={"dashboard_host": _dashboard_url().rstrip("/")},
                    repair="check network access to Photon, then rerun setup",
                ) from e
        if token:
            return token
    else:
        print("  no dashboard token found; running device login")

    rc = _cmd_login(ctx.args)
    if rc != 0:
        raise _failed_invariant(
            ctx,
            step="dashboard auth",
            summary="Photon device login did not complete",
            expected="device login stores a dashboard API token",
            observed=f"login command exited with {rc}",
            evidence={"dashboard_host": _dashboard_url().rstrip("/")},
            repair="complete `hermes photon login`, then rerun setup",
        )
    token = photon_auth.load_photon_token()
    if not token:
        raise _failed_invariant(
            ctx,
            step="dashboard auth",
            summary="Photon login completed but no token was saved",
            expected="PHOTON_DASHBOARD_TOKEN stored in Hermes env",
            observed="missing PHOTON_DASHBOARD_TOKEN",
            repair=f"inspect env file permissions at {ctx.env_path}",
        )
    try:
        photon_auth.validate_photon_token(token)
    except Exception as e:
        raise _failed_invariant(
            ctx,
            step="dashboard auth",
            summary="new Photon dashboard token is not valid for project APIs",
            expected="device login returns a project-valid dashboard token",
            observed=f"{type(e).__name__}: {e}",
            evidence={"dashboard_host": _dashboard_url().rstrip("/")},
            repair="retry login; if it repeats, Photon must return or accept a project API bearer token",
        ) from e
    print("  ✓ dashboard token is valid for Photon project APIs")
    return token


def _ensure_fixed_spectrum_project(
    ctx: _PhotonSetupContext,
    token: str,
) -> None:
    print(f"[project] Resolving Photon project {_FIXED_PROJECT_NAME!r}...")
    project = _resolve_fixed_dashboard_project(ctx, token)
    project_id = str(project.get("spectrum_project_id") or "").strip()
    project_secret = str(project.get("project_secret") or "").strip()
    if not (project_id and project_secret):
        raise _failed_invariant(
            ctx,
            step="fixed project",
            summary="Photon project did not expose Spectrum credentials",
            expected=(
                f"dashboard project {_FIXED_PROJECT_NAME!r} has "
                "spectrumProjectId and projectSecret"
            ),
            observed={
                "project_name": project.get("name") or _FIXED_PROJECT_NAME,
                "dashboard_project_id": project.get("dashboard_project_id") or "",
                "has_project_id": bool(project_id),
                "has_project_secret": bool(project_secret),
            },
            repair=(
                "open the Photon dashboard and repair the hermes-agent "
                "Spectrum/iMessage project, or reset local Photon state before retrying"
            ),
        )

    existing_id, _existing_secret = photon_auth.load_project_credentials()
    if existing_id and existing_id != project_id:
        raise _failed_invariant(
            ctx,
            step="fixed project",
            summary="local Photon project state is stale",
            expected=(
                f"local PHOTON_PROJECT_ID is empty or matches the "
                f"{_FIXED_PROJECT_NAME!r} dashboard project"
            ),
            observed={
                "local_project_id": existing_id,
                "dashboard_project_id": project.get("dashboard_project_id") or "",
                "resolved_project_id": project_id,
                "project_name": _FIXED_PROJECT_NAME,
            },
            repair="run `hermes photon reset` before setup overwrites local project state",
        )

    before = _photon_env_snapshot(
        "PHOTON_PROJECT_ID",
        "PHOTON_PROJECT_SECRET",
        "PHOTON_PROJECT_NAME",
    )
    photon_auth.store_project_credentials(
        project_id,
        project_secret,
        name=_FIXED_PROJECT_NAME,
        dashboard_project_id=project.get("dashboard_project_id") or "",
        source=project.get("source") or "fixed-project",
        created_by="hermes-agent",
    )
    after = _photon_env_snapshot(
        "PHOTON_PROJECT_ID",
        "PHOTON_PROJECT_SECRET",
        "PHOTON_PROJECT_NAME",
    )
    ctx.runtime_secrets_changed = ctx.runtime_secrets_changed or before != after
    ctx.project_id = project_id
    ctx.project_secret = project_secret
    ctx.dashboard_project_id = str(project.get("dashboard_project_id") or "")
    ctx.dashboard_project_name = _FIXED_PROJECT_NAME

    try:
        photon_auth.list_project_users(project_id, project_secret)
    except Exception as e:
        raise _failed_invariant(
            ctx,
            step="Spectrum credentials",
            summary="resolved Spectrum credentials failed validation",
            expected="selected project credentials can list Spectrum users",
            observed=f"{type(e).__name__}: {e}",
            evidence={"project_id": project_id, "http_status": _http_status(e)},
            repair="check Photon Spectrum API access, then rerun setup",
        ) from e

    print(f"  ✓ project ready: {_FIXED_PROJECT_NAME} ({project_id})")


def _resolve_fixed_dashboard_project(
    ctx: _PhotonSetupContext,
    token: str,
) -> dict[str, Any]:
    try:
        projects = photon_auth.list_projects(token)
    except photon_auth.PhotonDashboardAuthError as e:
        _handle_dashboard_auth_error(e)
        raise
    except Exception as e:
        raise _failed_invariant(
            ctx,
            step="fixed project",
            summary="could not list Photon dashboard projects",
            expected="Photon dashboard projects are readable",
            observed=f"{type(e).__name__}: {e}",
            evidence={"dashboard_host": _dashboard_url().rstrip("/")},
            repair="check Photon dashboard access, then rerun setup",
        ) from e

    normalized = [photon_auth.normalize_project(project) for project in projects]
    matches = [
        project for project in normalized
        if str(project.get("name") or "") == _FIXED_PROJECT_NAME
    ]
    if len(matches) > 1:
        raise _failed_invariant(
            ctx,
            step="fixed project",
            summary="duplicate Photon dashboard projects named hermes-agent",
            expected=f"exactly one dashboard project named {_FIXED_PROJECT_NAME!r}",
            observed=[
                {
                    "dashboard_project_id": project.get("dashboard_project_id") or "",
                    "spectrum_project_id": project.get("spectrum_project_id") or "",
                    "name": project.get("name") or "",
                }
                for project in matches
            ],
            repair="delete or rename the duplicate Photon dashboard project, then rerun setup",
        )

    existing_id, _existing_secret = photon_auth.load_project_credentials()
    if not matches:
        if existing_id:
            raise _failed_invariant(
                ctx,
                step="fixed project",
                summary="local Photon project state is stale",
                expected=(
                    "no local PHOTON_PROJECT_ID before creating the fixed "
                    f"{_FIXED_PROJECT_NAME!r} project"
                ),
                observed={
                    "local_project_id": existing_id,
                    "project_name": _FIXED_PROJECT_NAME,
                },
                repair="run `hermes photon reset` before creating the fixed Photon project",
            )
        print(f"  no {_FIXED_PROJECT_NAME!r} project found; creating it")
        project_id, project_secret = _create_and_store_project(
            token,
            name=_FIXED_PROJECT_NAME,
            source="fixed-project-created",
        )
        if not (project_id and project_secret):
            raise _failed_invariant(
                ctx,
                step="fixed project",
                summary="could not create the fixed Photon dashboard project",
                expected=f"Photon creates {_FIXED_PROJECT_NAME!r} with Spectrum credentials",
                observed="create_project returned no usable project id or secret",
                repair="check Photon dashboard access, then rerun setup",
            )
        return {
            "name": _FIXED_PROJECT_NAME,
            "spectrum_project_id": project_id,
            "project_secret": project_secret,
            "source": "fixed-project-created",
        }

    project = _refresh_project_details(token, matches[0])
    project["source"] = "fixed-project-existing"
    if not (project.get("spectrum_enabled") and project.get("imessage_enabled")):
        raise _failed_invariant(
            ctx,
            step="fixed project",
            summary="the hermes-agent dashboard project is not a Spectrum iMessage project",
            expected=f"{_FIXED_PROJECT_NAME!r} has Spectrum and iMessage enabled",
            observed={
                "dashboard_project_id": project.get("dashboard_project_id") or "",
                "spectrum_enabled": bool(project.get("spectrum_enabled")),
                "imessage_enabled": bool(project.get("imessage_enabled")),
                "platforms": project.get("platforms") or [],
            },
            repair="repair the Photon dashboard project or rename/delete it before rerunning setup",
        )

    project_id = str(project.get("spectrum_project_id") or "").strip()
    _existing_id, existing_secret = photon_auth.load_project_credentials()
    if project_id and existing_id == project_id and existing_secret and not project.get("project_secret"):
        project["project_secret"] = existing_secret
    print(f"  ✓ found existing {_FIXED_PROJECT_NAME!r} project")
    return project


def _ensure_operator_phone(ctx: _PhotonSetupContext) -> None:
    print("[phone] Reconciling Spectrum shared iMessage user...")
    phone = ctx.args.phone or _prompt(
        f"Your iMessage phone number (E.164, format {_PHONE_FORMAT}): "
    )
    ctx.operator_phone = phone or None
    if not phone:
        raise _failed_invariant(
            ctx,
            step="operator phone",
            summary="operator phone number is required",
            expected=f"an E.164 phone number like {_PHONE_FORMAT}",
            observed="missing --phone and no interactive phone was provided",
            repair=f"rerun `hermes photon setup {_PHONE_ARG_PLACEHOLDER}`",
        )
    if not photon_auth.E164_RE.match(phone):
        raise _failed_invariant(
            ctx,
            step="operator phone",
            summary="operator phone number is not E.164",
            expected=f"format {_PHONE_FORMAT}",
            observed=phone,
            repair=f"rerun with a phone number like {_PHONE_ARG_PLACEHOLDER}",
        )

    try:
        user = photon_auth.find_project_user_by_phone(
            ctx.project_id,
            ctx.project_secret,
            phone,
        )
    except Exception as e:
        raise _failed_invariant(
            ctx,
            step="Spectrum user",
            summary="could not list Spectrum users before phone reconciliation",
            expected="project users are listed before creating a new user",
            observed=f"{type(e).__name__}: {e}",
            evidence={"project_id": ctx.project_id, "http_status": _http_status(e)},
            repair="check Photon Spectrum API access, then rerun setup",
        ) from e
    if user:
        print("  ✓ existing Spectrum user reused for this phone")
    else:
        _ensure_shared_imessage_available(ctx, phone)
        try:
            created_user = photon_auth.create_user(
                ctx.project_id,
                ctx.project_secret,
                phone_number=phone,
                first_name=ctx.args.first_name,
                last_name=ctx.args.last_name,
                email=ctx.args.email,
            )
        except Exception as e:
            if _error_looks_like_existing_user(e):
                print("  phone already exists; verifying it belongs to the current Photon project")
                user = _require_project_user_by_phone(ctx, phone, cause=e)
                print("  ✓ phone verified in the current Photon project")
            else:
                raise _failed_invariant(
                    ctx,
                    step="Spectrum user",
                    summary="could not create or verify the Spectrum user",
                    expected="operator phone exists as a shared Spectrum iMessage user",
                    observed=_format_exception_with_http_detail(e),
                    evidence={"project_id": ctx.project_id, "http_status": _http_status(e)},
                    repair="verify the phone number in Photon, then rerun setup",
                ) from e
        else:
            # Creation is scoped to PHOTON_PROJECT_ID, so a 2xx response proves the
            # user was created under the canonical project. A follow-up lookup is
            # best-effort so we can print the assigned iMessage number when Photon
            # exposes it outside the create response.
            user = photon_auth.normalize_user(created_user)
            looked_up = _lookup_project_user_by_phone(ctx, phone)
            if looked_up:
                user = looked_up

    ctx.assigned_phone_number = _extract_assigned_phone_number(user)
    _persist_operator_phone(ctx)
    _ensure_home_channel_default(ctx, phone)
    if ctx.assigned_phone_number:
        print(f"  ✓ assigned Photon iMessage number: {ctx.assigned_phone_number}")
    else:
        print(
            "  ✓ Spectrum user is verified in the current project; "
            "Photon did not return the assigned iMessage number"
        )
    if not _ensure_operator_phone_allowed(phone):
        raise _failed_invariant(
            ctx,
            step="sender access",
            summary="operator phone was not authorized in Hermes sender access",
            expected="operator phone is present in PHOTON_ALLOWED_USERS or access is open",
            observed=_photon_sender_access_status(),
            repair=f"run `hermes photon allow-phone {phone}`",
        )


def _persist_operator_phone(ctx: _PhotonSetupContext) -> None:
    before = _photon_env_snapshot("PHOTON_OPERATOR_PHONE", _PHOTON_ASSIGNED_PHONE_ENV)
    if ctx.operator_phone:
        _save_photon_env_value_checked(ctx, "PHOTON_OPERATOR_PHONE", ctx.operator_phone)
    if ctx.assigned_phone_number:
        _save_photon_env_value_checked(
            ctx,
            _PHOTON_ASSIGNED_PHONE_ENV,
            ctx.assigned_phone_number,
        )
    after = _photon_env_snapshot("PHOTON_OPERATOR_PHONE", _PHOTON_ASSIGNED_PHONE_ENV)
    ctx.runtime_secrets_changed = ctx.runtime_secrets_changed or before != after


def _ensure_home_channel_default(ctx: _PhotonSetupContext, phone: str) -> None:
    before = _photon_env_snapshot("PHOTON_HOME_CHANNEL", "PHOTON_HOME_CHANNEL_NAME")
    existing = before.get("PHOTON_HOME_CHANNEL", "").strip()
    home_channel = f"any;-;{phone}"
    if not existing:
        _save_photon_env_value_checked(ctx, "PHOTON_HOME_CHANNEL", home_channel)
    if not before.get("PHOTON_HOME_CHANNEL_NAME", "").strip():
        _save_photon_env_value_checked(
            ctx,
            "PHOTON_HOME_CHANNEL_NAME",
            "You (iMessage)",
        )
    after = _photon_env_snapshot("PHOTON_HOME_CHANNEL", "PHOTON_HOME_CHANNEL_NAME")
    ctx.runtime_secrets_changed = ctx.runtime_secrets_changed or before != after
    if existing:
        print("  ✓ Photon home channel already configured")
    else:
        print("  ✓ default Photon home channel set to operator DM")


def _ensure_shared_imessage_available(
    ctx: _PhotonSetupContext,
    phone: str,
) -> None:
    if not ctx.dashboard_token:
        return
    try:
        available = photon_auth.check_phone_availability(ctx.dashboard_token, phone)
    except photon_auth.PhotonDashboardAuthError as e:
        _handle_dashboard_auth_error(e)
        raise _failed_invariant(
            ctx,
            step="Spectrum user",
            summary="could not check Photon shared iMessage availability",
            expected="dashboard token can call phone availability preflight",
            observed=str(e),
            evidence={"dashboard_host": _dashboard_url().rstrip("/"), "phone": phone},
            repair="rerun Photon login, then rerun setup",
        ) from e
    except Exception as e:
        print(
            "  shared iMessage availability precheck skipped: "
            f"{_short_error(str(e))}"
        )
        return

    if available is False:
        raise _failed_invariant(
            ctx,
            step="Spectrum user",
            summary="Photon has no shared iMessage number available for this phone",
            expected="Photon reports shared iMessage availability before user creation",
            observed={"phone": phone, "available": False},
            evidence={"dashboard_host": _dashboard_url().rstrip("/")},
            repair="try a different phone number or configure a dedicated Photon iMessage line",
        )
    if available is True:
        print("  ✓ shared iMessage number available")


def _lookup_project_user_by_phone(
    ctx: _PhotonSetupContext,
    phone: str,
) -> Optional[dict[str, Any]]:
    try:
        return photon_auth.find_project_user_by_phone(
            ctx.project_id,
            ctx.project_secret,
            phone,
        )
    except Exception:
        return None


def _require_project_user_by_phone(
    ctx: _PhotonSetupContext,
    phone: str,
    *,
    cause: BaseException,
) -> dict[str, Any]:
    try:
        user = photon_auth.find_project_user_by_phone(
            ctx.project_id,
            ctx.project_secret,
            phone,
        )
    except Exception as e:
        raise _failed_invariant(
            ctx,
            step="Spectrum user",
            summary="phone exists, but Hermes could not verify it in the current Photon project",
            expected="current PHOTON_PROJECT_ID user list contains the requested phone",
            observed=f"{type(e).__name__}: {e}",
            evidence={
                "project_id": ctx.project_id,
                "phone": phone,
                "create_user_error": _format_exception_with_http_detail(cause),
                "http_status": _http_status(cause),
            },
            repair="verify the phone under the Photon project that matches this project id, then rerun setup",
        ) from e

    if user:
        return user
    raise _failed_invariant(
        ctx,
        step="Spectrum user",
        summary="phone exists, but not in the current Photon project",
        expected="current PHOTON_PROJECT_ID user list contains the requested phone",
        observed={
            "project_id": ctx.project_id,
            "phone": phone,
            "create_user_error": _format_exception_with_http_detail(cause),
            "http_status": _http_status(cause),
        },
        evidence={"dashboard_project": ctx.dashboard_project_name or ctx.project_name},
        repair="attach this phone to the hermes-agent Photon project or run `hermes photon reset` before retrying",
    )


def _ensure_sidecar_ready(ctx: _PhotonSetupContext) -> None:
    print("[adapter] Verifying Spectrum sidecar dependencies...")
    node_bin = os.getenv("PHOTON_NODE_BIN") or "node"
    if not shutil.which(node_bin):
        raise _failed_invariant(
            ctx,
            step="sidecar dependencies",
            summary="Node.js is not available for the Photon sidecar",
            expected="Node.js 20.18.1+ is on PATH or PHOTON_NODE_BIN points to it",
            observed=f"missing node binary: {node_bin}",
            evidence={"sidecar_dir": str(_SIDECAR_DIR)},
            repair="install Node.js 20.18.1+, then rerun setup",
        )

    status = _sidecar_dependency_status()
    if status.startswith("✓"):
        print(f"  {status}")
        return
    if getattr(ctx.args, "skip_adapter_install", False):
        raise _failed_invariant(
            ctx,
            step="sidecar dependencies",
            summary="Photon sidecar dependencies are not installed",
            expected="spectrum-ts dependency is installed and current",
            observed=status,
            evidence={"sidecar_dir": str(_SIDECAR_DIR)},
            repair="rerun setup without `--skip-adapter-install`",
        )

    rc = _install_sidecar()
    if rc != 0:
        raise _failed_invariant(
            ctx,
            step="sidecar dependencies",
            summary="npm install failed for the Photon sidecar",
            expected="npm install completes successfully",
            observed=f"npm exited with {rc}",
            evidence={"sidecar_dir": str(_SIDECAR_DIR)},
            repair="fix npm/Node errors shown above, then rerun setup",
        )
    status = _sidecar_dependency_status()
    if not status.startswith("✓"):
        raise _failed_invariant(
            ctx,
            step="sidecar dependencies",
            summary="Photon sidecar dependencies still are not runnable after install",
            expected="spectrum-ts dependency is installed and current",
            observed=status,
            evidence={"sidecar_dir": str(_SIDECAR_DIR)},
            repair="inspect npm output in the sidecar directory, then rerun setup",
        )
    print(f"  {status}")


def _report_gateway_handoff(ctx: _PhotonSetupContext) -> None:
    print("[gateway] Photon config saved; gateway lifecycle stays with Hermes core.")
    runtime = _read_adapter_runtime_state(ctx.hermes_home)
    health = runtime.get("health") if isinstance(runtime, dict) else {}
    if isinstance(health, dict) and health.get("healthy"):
        pid = health.get("pid") or runtime.get("pid") or "-"
        print(f"  Photon adapter is already connected (pid {pid})")
    else:
        print(
            "  start or restart the Hermes gateway to launch the Photon "
            "adapter and connect the Spectrum session"
        )


def _ensure_photon_gateway_platform_enabled(ctx: _PhotonSetupContext) -> None:
    """Persist the explicit gateway platform bit Photon setup relies on."""
    config_path = ctx.hermes_home / "config.yaml"
    try:
        from utils import atomic_roundtrip_yaml_update  # type: ignore

        atomic_roundtrip_yaml_update(config_path, "platforms.photon.enabled", True)
    except Exception as e:
        raise _failed_invariant(
            ctx,
            step="gateway platform config",
            summary="could not enable Photon in gateway config",
            expected="config.yaml contains platforms.photon.enabled=true",
            observed=f"{type(e).__name__}: {e}",
            evidence={"config_path": str(config_path)},
            repair="fix config.yaml permissions or syntax, then rerun setup",
        ) from e

    print(f"  ✓ enabled platforms.photon in {config_path}")


def _setup_log_paths(ctx: _PhotonSetupContext) -> dict[str, Path]:
    log_dir = ctx.hermes_home / "logs"
    return {
        "gateway": log_dir / "gateway.log",
        "errors": log_dir / "errors.log",
        "gateway-error": log_dir / "gateway.error.log",
    }


def _print_setup_log_paths(ctx: _PhotonSetupContext) -> None:
    print("[logs] Existing logs for this setup:")
    for label, path in _setup_log_paths(ctx).items():
        print(f"  {label:<13}: {path}")
    if ctx.verbose:
        print("  verbose       : streaming new log lines while setup waits")


def _init_log_offsets(ctx: _PhotonSetupContext) -> None:
    ctx.log_offsets = {}
    for label, path in _setup_log_paths(ctx).items():
        try:
            ctx.log_offsets[label] = path.stat().st_size
        except OSError:
            ctx.log_offsets[label] = 0


def _stream_setup_logs(ctx: _PhotonSetupContext) -> None:
    if not ctx.verbose:
        return
    for label, path in _setup_log_paths(ctx).items():
        offset = ctx.log_offsets.get(label, 0)
        try:
            with path.open("rb") as fh:
                fh.seek(max(0, offset))
                data = fh.read()
                ctx.log_offsets[label] = fh.tell()
        except OSError:
            continue
        if not data:
            continue
        for line in data.decode("utf-8", errors="replace").splitlines():
            rendered = _redact_log_line(line).strip()
            if rendered:
                print(f"[{label}] {rendered[:500]}")


def _collect_relevant_log_tail(
    ctx: _PhotonSetupContext,
    *,
    max_lines: int = 40,
) -> dict[str, list[str]]:
    logs: dict[str, list[str]] = {}
    for label, path in _setup_log_paths(ctx).items():
        lines = _tail_text_file(path, max_lines=max_lines * 4)
        relevant = [
            _redact_log_line(line)
            for line in lines
            if _log_line_is_relevant(line)
        ]
        if not relevant:
            relevant = [_redact_log_line(line) for line in lines[-8:]]
        relevant = [line for line in relevant if line.strip()]
        if relevant:
            logs[label] = relevant[-max_lines:]
    return logs


def _tail_text_file(path: Path, *, max_lines: int) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return text.splitlines()[-max_lines:]


def _log_line_is_relevant(line: str) -> bool:
    lowered = line.lower()
    markers = (
        "photon",
        "spectrum",
        "gateway",
        "health",
        "error",
        "failed",
        "fatal",
        "paused",
        "connected",
        "traceback",
        "exception",
        "unauthorized",
        "401",
    )
    return any(marker in lowered for marker in markers)


def _redact_log_line(line: str) -> str:
    redacted = str(line)
    redacted = re.sub(
        r"(?i)(authorization:\s*bearer\s+)[^\s]+",
        r"\1<redacted>",
        redacted,
    )
    redacted = re.sub(
        r"(?i)(PHOTON_PROJECT_SECRET=)[^\s]+",
        r"\1<redacted>",
        redacted,
    )
    redacted = re.sub(
        r"(?i)(PHOTON_DASHBOARD_TOKEN=)[^\s]+",
        r"\1<redacted>",
        redacted,
    )
    redacted = re.sub(
        r'(?i)("?(?:projectSecret|signingSecret|secret|token)"?\s*[:=]\s*")([^"]+)(")',
        r"\1<redacted>\3",
        redacted,
    )
    return redacted


def _finalize_failed_invariant_logs(
    error: _FailedInvariant,
    ctx: _PhotonSetupContext,
) -> None:
    if not ctx.verbose:
        error.logs = {}
        return
    _stream_setup_logs(ctx)
    if not error.logs:
        error.logs = _collect_relevant_log_tail(ctx)


def _save_photon_env_value_checked(
    ctx: _PhotonSetupContext,
    key: str,
    value: str,
) -> bool:
    current = (_get_env_value(key) or "").strip()
    if current == value:
        return False
    try:
        from hermes_cli.config import save_env_value  # type: ignore

        save_env_value(key, value)
    except Exception as e:
        raise _failed_invariant(
            ctx,
            step="Photon env state",
            summary=f"could not save {key}",
            expected=f"{key} is saved in Hermes env",
            observed=f"{type(e).__name__}: {e}",
            evidence={"env_path": str(ctx.env_path), key: value},
            repair="fix Hermes env file permissions, then rerun setup",
        ) from e
    return True


def _photon_env_snapshot(*keys: str) -> dict[str, str]:
    return {key: _get_env_value(key) or "" for key in keys}


def _remove_env_value(key: str) -> bool:
    try:
        from hermes_cli.config import remove_env_value  # type: ignore

        return bool(remove_env_value(key))
    except Exception:
        return os.environ.pop(key, None) is not None


def _error_looks_like_existing_user(exc: BaseException) -> bool:
    status = _http_status(exc)
    detail = _http_error_detail(exc).lower()
    if status == 409:
        return (
            ("already" in detail or "exist" in detail)
            and ("user" in detail or "phone" in detail)
        )
    fallback = str(exc).lower()
    return "already" in fallback and "user" in fallback


def _http_error_detail(exc: BaseException) -> str:
    response = getattr(exc, "response", None)
    if response is None:
        return ""
    try:
        return photon_auth._response_error_detail(response)  # type: ignore[attr-defined]
    except Exception:
        return str(getattr(response, "text", "") or "").strip()[:200]


def _format_exception_with_http_detail(exc: BaseException) -> str:
    formatted = f"{type(exc).__name__}: {exc}"
    detail = _http_error_detail(exc)
    if detail and detail not in formatted:
        return f"{formatted} ({detail})"
    return formatted


def _http_status(exc: BaseException) -> Optional[int]:
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def _failed_invariant(
    ctx: _PhotonSetupContext,
    *,
    step: str,
    summary: str,
    expected: str,
    observed: Any,
    repair: str,
    evidence: Optional[dict[str, Any]] = None,
) -> _FailedInvariant:
    merged: dict[str, Any] = {
        "hermes_home": str(ctx.hermes_home),
        "env_path": str(ctx.env_path),
    }
    if ctx.project_id:
        merged["project_id"] = ctx.project_id
    if evidence:
        merged.update(evidence)
    return _FailedInvariant(
        step=step,
        summary=summary,
        expected=expected,
        observed=observed,
        evidence=merged,
        repair=repair,
        verbose=ctx.verbose,
    )


def _print_failed_invariant(error: _FailedInvariant) -> None:
    print("", file=sys.stderr)
    print(f"Photon setup stopped: {error.summary}", file=sys.stderr)
    if not error.verbose:
        print(f"  repair   : {error.repair}", file=sys.stderr)
        return

    print(f"  step     : {error.step}", file=sys.stderr)
    print(f"  expected : {error.expected}", file=sys.stderr)
    print("  observed :", file=sys.stderr)
    _print_evidence_value(error.observed, indent="    ", stream=sys.stderr)
    print("  evidence :", file=sys.stderr)
    _print_evidence_value(error.evidence, indent="    ", stream=sys.stderr)
    print(f"  repair   : {error.repair}", file=sys.stderr)
    if error.logs:
        print("  relevant logs :", file=sys.stderr)
        for label, lines in error.logs.items():
            print(f"    [{label}]", file=sys.stderr)
            for line in lines[-40:]:
                print(f"      {line}", file=sys.stderr)


def _print_evidence_value(value: Any, *, indent: str, stream: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, (dict, list, tuple)):
                print(f"{indent}{key}:", file=stream)
                _print_evidence_value(item, indent=indent + "  ", stream=stream)
            else:
                print(f"{indent}{key}: {item}", file=stream)
        return
    if isinstance(value, (list, tuple)):
        if not value:
            print(f"{indent}-", file=stream)
            return
        for item in value:
            if isinstance(item, (dict, list, tuple)):
                print(f"{indent}-", file=stream)
                _print_evidence_value(item, indent=indent + "  ", stream=stream)
            else:
                print(f"{indent}- {item}", file=stream)
        return
    print(f"{indent}{value}", file=stream)


def _print_setup_reconciled(ctx: _PhotonSetupContext) -> None:
    print("Photon setup complete.")
    print(f"  Hermes home       : {ctx.hermes_home}")
    print(f"  env path          : {ctx.env_path}")
    print(f"  project name      : {_FIXED_PROJECT_NAME}")
    print(f"  project id        : {ctx.project_id}")
    if ctx.operator_phone:
        print(f"  operator phone    : {ctx.operator_phone}")
    if ctx.assigned_phone_number:
        print(f"  assigned number   : {ctx.assigned_phone_number}")
    home_channel, home_channel_name = _home_channel_status()
    print(f"  home channel      : {home_channel}")
    print(f"  home channel name : {home_channel_name}")
    print("  gateway lifecycle : managed by Hermes core")
    _print_text_photon_number_step(ctx)


def interactive_setup() -> None:
    """Entry point used by `hermes setup gateway` when Photon is selected."""
    from hermes_cli.cli_output import print_info, prompt_yes_no

    setup_phone = None
    if _interactive_setup_already_configured():
        binding = _interactive_setup_binding_summary()
        print_info(f"Photon iMessage is already configured: {binding}.")
        if not prompt_yes_no(
            f"Reconfigure Photon iMessage bound to {binding}?",
            False,
        ):
            print_info(
                "Leaving existing Photon iMessage configuration unchanged. "
                "Run `hermes photon status` to inspect it."
            )
            return
        setup_phone = (_get_env_value("PHOTON_OPERATOR_PHONE") or "").strip() or None
        if setup_phone:
            print_info(f"Reusing existing operator phone {setup_phone}.")

    args = argparse.Namespace(
        phone=setup_phone,
        first_name=None,
        last_name=None,
        email=None,
        no_browser=False,
        skip_adapter_install=False,
        verbose=False,
    )
    _cmd_setup(args)


def _interactive_setup_already_configured() -> bool:
    """Return True when this Hermes profile has enough Photon state to run."""
    project_id, project_secret = photon_auth.load_project_credentials()
    if not (project_id and project_secret):
        return False
    node_bin = os.getenv("PHOTON_NODE_BIN") or "node"
    if not shutil.which(node_bin):
        return False
    if not (_SIDECAR_DIR / "node_modules").exists():
        return False
    operator_phone = _get_env_value("PHOTON_OPERATOR_PHONE") or ""
    if not operator_phone:
        return False
    return _photon_sender_access_configured()


def _interactive_setup_binding_summary() -> str:
    operator_phone = (_get_env_value("PHOTON_OPERATOR_PHONE") or "").strip()
    assigned_phone = (_get_env_value(_PHOTON_ASSIGNED_PHONE_ENV) or "").strip()
    if operator_phone and assigned_phone:
        return f"operator {operator_phone}; assigned Photon number {assigned_phone}"
    if operator_phone:
        return f"operator {operator_phone}"
    if assigned_phone:
        return f"assigned Photon number {assigned_phone}"
    return "this Hermes profile"


def _dashboard_url() -> str:
    return (
        os.getenv("PHOTON_DASHBOARD_HOST")
        or photon_auth.DEFAULT_DASHBOARD_HOST
    ).rstrip("/") + "/"


def _extract_assigned_phone_number(user: Any) -> Optional[str]:
    candidates: list[Any] = []
    for container in _candidate_user_payloads(user):
        candidates.extend([
            container.get("assignedPhoneNumber"),
            container.get("assigned_phone_number"),
            container.get("assignedNumber"),
            container.get("assigned_number"),
            container.get("imessageNumber"),
            container.get("iMessageNumber"),
            container.get("imessage_number"),
            container.get("photonNumber"),
            container.get("photon_number"),
        ])
    for value in candidates:
        if isinstance(value, str) and photon_auth.E164_RE.match(value.strip()):
            return value.strip()
    return None


def _candidate_user_payloads(user: Any) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    if isinstance(user, dict):
        payloads.append(user)
        for key in ("data", "user", "profile", "raw"):
            nested = user.get(key)
            if isinstance(nested, dict):
                payloads.append(nested)
    return payloads


def _print_text_photon_number_step(ctx: _PhotonSetupContext) -> None:
    if ctx.assigned_phone_number:
        print("  Text Hermes:")
        print(f"        Send \"hi Hermes\" to {ctx.assigned_phone_number}")
        return

    project_name = ctx.dashboard_project_name or ctx.project_name
    phone_label = ctx.operator_phone or "your phone number"
    print("  Find the assigned Photon number:")
    print(f"        Open {_dashboard_url()}")
    print(f"        Open project \"{project_name}\"")
    print(f"        Users -> {phone_label} -> assigned iMessage number")
    print("        Send \"hi Hermes\" to that number")


def _cmd_allow_phone(args: argparse.Namespace) -> int:
    return 0 if _ensure_operator_phone_allowed(args.phone, explicit=True) else 1


def _cmd_phones(args: argparse.Namespace) -> int:
    sub = getattr(args, "photon_phones_command", None)
    project_id, project_secret = photon_auth.load_project_credentials()
    if not (project_id and project_secret):
        print(
            "Photon project credentials are missing — run "
            f"`hermes photon setup {_PHONE_ARG_PLACEHOLDER}` first.",
            file=sys.stderr,
        )
        return 1

    if sub == "list":
        try:
            data = phone_management.list_phones(project_id, project_secret)
        except phone_management.PhotonPhoneManagementError as e:
            _print_phone_management_error("phones list", e, verbose=False)
            return 1
        users = _normalized_phone_users(data)
        _print_phones_list(project_id, users)
        return 0

    if sub == "add":
        phone = args.phone
        try:
            data = phone_management.add_phone(project_id, project_secret, phone)
        except phone_management.PhotonPhoneManagementError as e:
            _print_phone_management_error(
                "phones add",
                e,
                verbose=bool(getattr(args, "verbose", False)),
                phone=phone,
            )
            return 1
        try:
            _append_photon_allowed_user(phone)
        except Exception as e:
            print(
                f"phones add partially completed: Photon user was created, "
                f"but PHOTON_ALLOWED_USERS could not be updated: {e}",
                file=sys.stderr,
            )
            return 1
        user = _normalized_phone_user(data.get("user") if isinstance(data, dict) else {})
        print(f"✓ added {phone} to Photon project {_FIXED_PROJECT_NAME}")
        if user.get("assigned_phone_number"):
            print(f"  assigned number : {user['assigned_phone_number']}")
        print("  Hermes access   : authorized in PHOTON_ALLOWED_USERS")
        if _truthy_env("PHOTON_ALLOW_ALL_USERS"):
            print("  access note     : PHOTON_ALLOW_ALL_USERS=true is also open")
        print("  Next: restart the gateway if it is already running:")
        print("        hermes gateway restart")
        return 0

    if sub == "remove":
        phone = args.phone
        try:
            data = phone_management.remove_phone(project_id, project_secret, phone)
        except phone_management.PhotonPhoneManagementError as e:
            _print_phone_management_error(
                "phones remove",
                e,
                verbose=bool(getattr(args, "verbose", False)),
                phone=phone,
            )
            return 1
        try:
            _remove_photon_allowed_user(phone)
        except Exception as e:
            print(
                f"phones remove partially completed: Photon user was removed, "
                f"but PHOTON_ALLOWED_USERS could not be updated: {e}",
                file=sys.stderr,
            )
            return 1
        user = _normalized_phone_user(data.get("user") if isinstance(data, dict) else {})
        print(f"✓ removed {phone} from Photon project {_FIXED_PROJECT_NAME}")
        if user.get("assigned_phone_number"):
            print(f"  removed number : {user['assigned_phone_number']}")
        print("  Hermes access  : removed from PHOTON_ALLOWED_USERS")
        if _truthy_env("PHOTON_ALLOW_ALL_USERS"):
            print("  access note    : PHOTON_ALLOW_ALL_USERS=true still allows all senders")
        print("  Next: restart the gateway if it is already running:")
        print("        hermes gateway restart")
        return 0

    print(f"unknown phones subcommand: {sub}", file=sys.stderr)
    return 2


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
        print("  Note: allow-phone only updates local Hermes authorization.")
        print("        To create a Photon project user, use:")
        print(f"        hermes photon phones add {phone}")
        print("  Next: restart the gateway if it is already running:")
        print("        hermes gateway restart")
    return True


def _phone_arg(value: str) -> str:
    phone = str(value or "").strip()
    if not photon_auth.E164_RE.match(phone):
        raise argparse.ArgumentTypeError(
            f"phone must be E.164 (format {_PHONE_FORMAT})"
        )
    return phone


def _append_photon_allowed_user(phone: str) -> str:
    phones = photon_auth.load_allowed_phone_numbers()
    if phone in phones:
        return "already_allowed"
    phones.append(phone)
    _save_allowed_users(phones)
    return "added"


def _remove_photon_allowed_user(phone: str) -> str:
    phones = photon_auth.load_allowed_phone_numbers()
    if phone not in phones:
        return "not_present"
    remaining = [item for item in phones if item != phone]
    _save_allowed_users(remaining)
    return "removed"


def _save_allowed_users(phones: list[str]) -> None:
    value = ",".join(dict.fromkeys(item.strip() for item in phones if item.strip()))
    if not value:
        _remove_env_value("PHOTON_ALLOWED_USERS")
        return
    try:
        from hermes_cli.config import save_env_value  # type: ignore

        save_env_value("PHOTON_ALLOWED_USERS", value)
    except ImportError as exc:
        raise RuntimeError(
            "hermes_cli.config is required to save Photon gateway access"
        ) from exc


def _normalized_phone_users(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, dict):
        return []
    users = data.get("users")
    if not isinstance(users, list):
        return []
    return [_normalized_phone_user(user) for user in users if isinstance(user, dict)]


def _normalized_phone_user(user: Any) -> dict[str, Any]:
    if not isinstance(user, dict):
        return {
            "phone": "",
            "assigned_phone_number": "",
            "user_id": "",
            "raw": {},
        }
    raw = user.get("raw") if isinstance(user.get("raw"), dict) else {}
    phone = _first_user_string(
        user.get("phone"),
        user.get("phoneNumber"),
        user.get("phone_number"),
        user.get("submittedPhoneNumber"),
        user.get("submitted_phone_number"),
        raw.get("phoneNumber"),
        raw.get("phone_number"),
        raw.get("submittedPhoneNumber"),
        raw.get("submitted_phone_number"),
    )
    assigned = _first_user_string(
        user.get("assigned_phone_number"),
        user.get("assignedPhoneNumber"),
        user.get("assigned_number"),
        user.get("assignedNumber"),
        user.get("imessageNumber"),
        user.get("iMessageNumber"),
        user.get("photonNumber"),
        raw.get("assignedPhoneNumber"),
        raw.get("assigned_phone_number"),
        raw.get("imessageNumber"),
        raw.get("iMessageNumber"),
        raw.get("photonNumber"),
    )
    user_id = _first_user_string(
        user.get("user_id"),
        user.get("id"),
        user.get("userId"),
        raw.get("id"),
        raw.get("userId"),
    )
    return {
        "phone": phone,
        "assigned_phone_number": assigned,
        "user_id": user_id,
        "raw": raw or user,
    }


def _first_user_string(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _print_phones_list(project_id: str, users: list[dict[str, Any]]) -> None:
    print("Photon phones")
    print(f"  project name : {_FIXED_PROJECT_NAME}")
    print(f"  project id   : {project_id}")
    print(f"  access       : {_photon_sender_access_status()}")
    if not users:
        print("  users        : none")
        return
    print("  users:")
    for user in users:
        phone = user.get("phone") or "(unknown phone)"
        assigned = user.get("assigned_phone_number") or "-"
        user_id = user.get("user_id") or "-"
        authorized = _phone_user_authorization_label(user)
        print(
            f"    - phone={phone}  assigned={assigned}  "
            f"user={user_id}  authorized={authorized}"
        )


def _phone_user_authorization_label(user: dict[str, Any]) -> str:
    if _truthy_env("PHOTON_ALLOW_ALL_USERS"):
        return "yes (allow-all)"
    if _truthy_env("GATEWAY_ALLOW_ALL_USERS"):
        return "yes (gateway allow-all)"
    allowed = set(photon_auth.load_allowed_phone_numbers())
    if "*" in allowed:
        return "yes (wildcard)"
    candidates = {
        str(user.get("phone") or "").strip(),
        str(user.get("assigned_phone_number") or "").strip(),
        str(user.get("user_id") or "").strip(),
    }
    return "yes" if candidates & allowed else "no"


def _format_phones_summary(users: list[dict[str, Any]]) -> str:
    normalized = [
        _normalized_phone_user(photon_auth.normalize_user(user))
        for user in users
        if isinstance(user, dict)
    ]
    total = len(normalized)
    if total == 0:
        return "0 project users"
    authorized = sum(
        1 for user in normalized
        if _phone_user_authorization_label(user).startswith("yes")
    )
    if _truthy_env("PHOTON_ALLOW_ALL_USERS"):
        return f"{total} project user(s); access open via PHOTON_ALLOW_ALL_USERS"
    if _truthy_env("GATEWAY_ALLOW_ALL_USERS"):
        return f"{total} project user(s); access open via GATEWAY_ALLOW_ALL_USERS"
    return f"{total} project user(s); {authorized} authorized in Hermes"


def _print_phone_management_error(
    command: str,
    error: phone_management.PhotonPhoneManagementError,
    *,
    verbose: bool,
    phone: str = "",
) -> None:
    if error.code == "PHONE_EXISTS":
        print(
            f"{command} failed: phone {phone or ''} already exists on "
            f"the Photon project {_FIXED_PROJECT_NAME}.",
            file=sys.stderr,
        )
    elif error.code == "PHONE_NOT_FOUND":
        print(
            f"{command} failed: phone {phone or ''} does not exist on "
            f"the Photon project {_FIXED_PROJECT_NAME}.",
            file=sys.stderr,
        )
    elif command == "phones add" and _phone_error_is_capacity(error):
        print(
            "phones add failed: Photon free plan shared-user limit reached.",
            file=sys.stderr,
        )
        print(
            "  Upgrade path: open https://app.photon.codes/dashboard, "
            f"select project {_FIXED_PROJECT_NAME!r}, then go to "
            "Billing -> Upgrade plan.",
            file=sys.stderr,
        )
    else:
        print(f"{command} failed: {error.message}", file=sys.stderr)

    if verbose and (error.detail or error.status or error.code):
        print("  Photon detail:", file=sys.stderr)
        print(f"    code   : {error.code}", file=sys.stderr)
        if error.status:
            print(f"    status : {error.status}", file=sys.stderr)
        if error.detail:
            print(f"    detail : {_short_error(error.detail)}", file=sys.stderr)


def _phone_error_is_capacity(error: phone_management.PhotonPhoneManagementError) -> bool:
    text = f"{error.code} {error.message} {error.detail}".lower()
    markers = (
        "limit",
        "quota",
        "capacity",
        "maxsharedusers",
        "max shared users",
        "maximum number of shared users",
        "shared-line",
        "shared line",
        "too many users",
        "over plan",
    )
    return any(marker in text for marker in markers)


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


def _adapter_runtime_state_path(hermes_home: Optional[Path] = None) -> Path:
    return (hermes_home or Path(get_hermes_home())) / "photon" / "adapter-runtime.json"


def _read_adapter_runtime_state(hermes_home: Optional[Path] = None) -> dict[str, Any]:
    try:
        data = json.loads(_adapter_runtime_state_path(hermes_home).read_text())
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _format_adapter_runtime_status(state: dict[str, Any]) -> str:
    health = state.get("health") if isinstance(state, dict) else {}
    if not isinstance(health, dict):
        health = {}
    if health.get("healthy"):
        pid = health.get("pid") or state.get("pid") or "-"
        return f"✓ connected (pid {pid})"
    if health:
        label = health.get("state") or "not connected"
        last_error = health.get("last_error")
        if isinstance(last_error, dict) and last_error.get("message"):
            return f"✗ {label} ({_short_error(str(last_error.get('message')))})"
        return f"✗ {label}"
    return "✗ not running"


def _home_channel_status() -> tuple[str, str]:
    channel = (_get_env_value("PHOTON_HOME_CHANNEL") or "").strip()
    name = (_get_env_value("PHOTON_HOME_CHANNEL_NAME") or "").strip()
    return channel or "✗ missing", name or "-"


def _cmd_status(_args: argparse.Namespace) -> int:
    ctx = _PhotonSetupContext.from_args(_args)
    # Defer the whole table to auth.print_credential_summary — its emit
    # callback is the only sink that sees credential-derived strings, so
    # cli.py keeps zero taint flow according to CodeQL.
    photon_auth.print_credential_summary(print)
    # The two non-credential rows live here so the helper stays purely
    # about credentials.
    node_bin = os.getenv("PHOTON_NODE_BIN") or shutil.which("node")
    sidecar_deps_status = _sidecar_dependency_status()
    adapter_runtime = _read_adapter_runtime_state(ctx.hermes_home)
    project_name = _get_env_value("PHOTON_PROJECT_NAME") or _FIXED_PROJECT_NAME
    operator_phone = _get_env_value("PHOTON_OPERATOR_PHONE") or "✗ missing"
    assigned_phone = _get_env_value(_PHOTON_ASSIGNED_PHONE_ENV) or "not recorded"
    home_channel, home_channel_name = _home_channel_status()
    project_id, project_secret = photon_auth.load_project_credentials()
    spectrum_status = "✗ missing Photon project credentials"
    phones_summary = "✗ missing Photon project credentials"
    if project_id and project_secret:
        try:
            users = photon_auth.list_project_users(project_id, project_secret)
            spectrum_status = "✓ valid"
            phones_summary = _format_phones_summary(users)
        except Exception as e:
            status = _http_status(e)
            if status:
                spectrum_status = f"✗ invalid or unreachable (HTTP {status})"
            else:
                spectrum_status = f"✗ invalid or unreachable ({_short_error(str(e))})"
            phones_summary = "unavailable while Spectrum credentials are invalid"
    print(f"  Hermes home         : {ctx.hermes_home}")
    print(f"  env path            : {ctx.env_path}")
    print(f"  dashboard auth      : {_dashboard_token_status()}")
    print(f"  Spectrum creds      : {spectrum_status}")
    print(f"  project name        : {project_name}")
    print(f"  project id          : {project_id or '✗ missing'}")
    print(f"  operator phone      : {operator_phone}")
    print(f"  assigned number     : {assigned_phone}")
    print(f"  home channel        : {home_channel}")
    print(f"  home channel name   : {home_channel_name}")
    print(f"  phones summary      : {phones_summary}")
    print("  gateway lifecycle   : managed by Hermes core")
    print(f"  node binary         : {node_bin or '✗ missing (install Node 20.18.1+)'}")
    print(f"  sidecar deps        : {sidecar_deps_status}")
    print(f"  authorized phones   : {_photon_sender_access_status()}")
    print(
        "  next step           : "
        + _next_status_step(
            sidecar_deps_status,
            adapter_runtime=adapter_runtime,
        )
    )
    print(f"  docs                : {_docs_paths()}")
    return 0


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
        return f"✗ run `hermes photon setup {_PHONE_ARG_PLACEHOLDER}`"

    version, problems = _installed_spectrum_ts()
    if not version:
        return (
            "✗ spectrum-ts missing; rerun "
            f"`hermes photon setup {_PHONE_ARG_PLACEHOLDER}`"
        )

    parsed = _parse_semver(version)
    if parsed is None:
        return f"⚠ spectrum-ts {version} installed; unable to verify version"
    if parsed < _MIN_SPECTRUM_TS_VERSION:
        return (
            f"✗ spectrum-ts {version} is too old; "
            f"rerun `hermes photon setup {_PHONE_ARG_PLACEHOLDER}`"
        )
    if problems:
        detail = str(problems[0]).splitlines()[0][:120]
        return (
            f"⚠ spectrum-ts {version} installed but npm reports {detail}; "
            f"rerun `hermes photon setup {_PHONE_ARG_PLACEHOLDER}`"
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


def _short_error(error: str) -> str:
    return (error or "").replace("\n", " ")[:120] or "unknown"


def _dashboard_token_status() -> str:
    token = photon_auth.load_photon_token()
    if not token:
        return "✗ missing"
    try:
        photon_auth.validate_photon_token(token)
    except Exception as e:
        return f"✗ invalid ({_short_error(str(e))})"
    return "✓ valid"


def _next_status_step(
    sidecar_deps_status: str,
    *,
    adapter_runtime: Optional[dict[str, Any]] = None,
) -> str:
    project_id, project_secret = photon_auth.load_project_credentials()
    if not photon_auth.load_photon_token() and not (project_id and project_secret):
        return f"hermes photon setup {_PHONE_ARG_PLACEHOLDER}"
    if not (project_id and project_secret):
        return f"hermes photon setup {_PHONE_ARG_PLACEHOLDER}"
    if sidecar_deps_status.startswith("✗"):
        return f"hermes photon setup {_PHONE_ARG_PLACEHOLDER}"
    if not (_get_env_value("PHOTON_HOME_CHANNEL") or "").strip():
        return (
            f"rerun `hermes photon setup {_PHONE_ARG_PLACEHOLDER}` "
            "or set PHOTON_HOME_CHANNEL"
        )
    runtime_health = {}
    if isinstance(adapter_runtime, dict):
        value = adapter_runtime.get("health")
        runtime_health = value if isinstance(value, dict) else {}
    if not runtime_health.get("healthy"):
        return "start or restart the Hermes gateway"
    if not _photon_sender_access_configured():
        return f"hermes photon allow-phone {_PHONE_ARG_PLACEHOLDER}"
    return "send an iMessage to the Photon number"


def _docs_paths() -> str:
    return "plugins/platforms/photon/README.md; website/docs/user-guide/messaging/photon.md"


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
