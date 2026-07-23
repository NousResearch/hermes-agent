"""CLI handlers for machine-wide Anthropic shared OAuth pool."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from hermes_cli.auth import AuthError


def _cap():
    from agent.anthropic_shared_pool import get_shared_mutation_capability

    return get_shared_mutation_capability()


def _exit_auth_error(exc: AuthError) -> None:
    print(f"error: {exc}", file=sys.stderr)
    code = getattr(exc, "code", "") or ""
    if code in {
        "shared_mutation_forbidden",
        "shared_add_while_active",
        "shared_remove_while_active",
        "shared_pool_full",
        "shared_repair_needs_yes",
        "shared_restore_needs_yes",
        "shared_gateways_live",
    }:
        raise SystemExit(2)
    raise SystemExit(1)


def _shared_flag(args: Any) -> bool:
    return bool(getattr(args, "shared", False))


def guard_unscoped_anthropic_mutation(provider: str, *, shared: bool, verb: str) -> None:
    """Refuse unscoped Anthropic mutations while shared scope is active."""
    if (provider or "").strip().lower() != "anthropic":
        return
    from agent.anthropic_shared_pool import is_shared_scope_active

    try:
        active = is_shared_scope_active()
    except AuthError as exc:
        _exit_auth_error(exc)
        return
    if active and not shared:
        print(
            f"error: Anthropic shared scope is active. "
            f"Use: hermes auth {verb} anthropic ... --shared",
            file=sys.stderr,
        )
        raise SystemExit(2)


def auth_scope_command(args: Any) -> None:
    from agent import anthropic_shared_pool as sp

    provider = (getattr(args, "provider", "") or "").strip().lower()
    if provider != "anthropic":
        print("error: only provider 'anthropic' is supported for auth scope v1", file=sys.stderr)
        raise SystemExit(2)
    mode = (getattr(args, "scope_mode", None) or "").strip().lower()
    try:
        if not mode:
            state = sp.read_scope_state()
            print(f"provider: anthropic")
            print(f"scope: {state.mode}")
            if state.epoch:
                print(f"epoch: {state.epoch}")
            print(f"marker: {sp.scope_marker_path()}")
            print(f"root_auth: {sp.root_auth_path()}")
            return
        if mode == "shared":
            if not getattr(args, "attest_distinct_accounts", False):
                print(
                    "error: enabling shared scope requires --attest-distinct-accounts\n"
                    "(operator attestation that the three browser sessions used different "
                    "Anthropic accounts; Hermes cannot machine-verify account identity)",
                    file=sys.stderr,
                )
                raise SystemExit(2)
            # Already-valid shared revalidates without requiring gateways stopped.
            already = False
            try:
                st = sp.read_scope_state()
                already = st.mode == "shared"
            except AuthError:
                already = False
            if not already:
                sp.require_no_live_gateways_for_scope_change()
            epoch = sp.enable_shared_scope(
                attest_distinct_accounts=True,
                capability=_cap(),
            )
            print(f"scope: shared")
            print(f"epoch: {epoch}")
            print("Restart all Hermes gateways/workers to pick up the new scope.")
            return
        if mode == "profile":
            sp.require_no_live_gateways_for_scope_change()
            sp.disable_shared_scope(capability=_cap())
            print("scope: profile")
            print("Shared marker removed; root grants remain dormant for rollback.")
            print("Restart all Hermes gateways/workers.")
            return
        if mode == "repair":
            path = sp.repair_malformed_marker(
                yes=bool(getattr(args, "yes", False)),
                capability=_cap(),
            )
            print(f"marker repaired; forensic backup: {path}")
            return
        print(f"error: unknown scope mode {mode!r}", file=sys.stderr)
        raise SystemExit(2)
    except AuthError as exc:
        _exit_auth_error(exc)


def auth_backup_shared_command(args: Any) -> None:
    from agent import anthropic_shared_pool as sp

    provider = (getattr(args, "provider", "") or "").strip().lower()
    if provider != "anthropic" or not _shared_flag(args):
        print("error: use: hermes auth backup anthropic --shared --output <abs-path>", file=sys.stderr)
        raise SystemExit(2)
    out = getattr(args, "output", None)
    if not out:
        print("error: --output is required", file=sys.stderr)
        raise SystemExit(2)
    try:
        path = sp.create_shared_backup(Path(out), capability=_cap())
        print(f"shared backup written: {path}")
    except AuthError as exc:
        _exit_auth_error(exc)


def auth_restore_shared_command(args: Any) -> None:
    from agent import anthropic_shared_pool as sp

    provider = (getattr(args, "provider", "") or "").strip().lower()
    if provider != "anthropic" or not _shared_flag(args):
        print("error: use: hermes auth restore anthropic --shared --input <abs-path> --yes", file=sys.stderr)
        raise SystemExit(2)
    inp = getattr(args, "input", None)
    if not inp:
        print("error: --input is required", file=sys.stderr)
        raise SystemExit(2)
    try:
        sp.require_no_live_gateways_for_scope_change()
        sp.restore_shared_backup(
            Path(inp),
            yes=bool(getattr(args, "yes", False)),
            capability=_cap(),
        )
        print("shared backup restored")
    except AuthError as exc:
        _exit_auth_error(exc)


def add_shared_oauth_grant(args: Any) -> None:
    """Stage a dormant shared OAuth grant (profile scope only)."""
    from agent import anthropic_shared_pool as sp
    from agent import anthropic_adapter as anthropic_mod

    try:
        if sp.is_shared_scope_active():
            print(
                "error: cannot add while shared scope is active "
                "(switch to profile scope first)",
                file=sys.stderr,
            )
            raise SystemExit(2)
        pool = sp.load_shared_pool_for_management()
        if len(pool["entries"]) >= 3:
            print("error: shared pool already has 3 grants", file=sys.stderr)
            raise SystemExit(2)
        creds = anthropic_mod.run_hermes_oauth_login_pure()
        if not creds:
            raise SystemExit("Anthropic OAuth login did not return credentials.")
        # Endpoint recorded by login — prefer platform.
        endpoint = sp.ENDPOINT_PLATFORM
        endpoint_url = creds.get("token_endpoint") or creds.get("oauth_token_endpoint")
        if isinstance(endpoint_url, str):
            mapped = sp.URL_TO_ENDPOINT.get(endpoint_url.strip())
            if mapped:
                endpoint = mapped
            elif endpoint_url in sp.ENDPOINT_URLS:
                endpoint = endpoint_url
        label = (getattr(args, "label", None) or "").strip()
        if not label:
            label = f"Anthropic account {len(pool['entries']) + 1}"
        row = sp.new_shared_row(
            access_token=creds["access_token"],
            refresh_token=creds["refresh_token"],
            expires_at_ms=int(creds["expires_at_ms"]),
            oauth_token_endpoint=endpoint,
            label=label,
            priority=len(pool["entries"]),
            initial_refresh_token=creds["refresh_token"],
        )
        stored = sp.append_row(row, capability=_cap())
        print(f'Added shared Anthropic OAuth grant: "{stored["label"]}" id={stored["id"]}')
        print("Grant is dormant until: hermes auth scope anthropic shared --attest-distinct-accounts")
    except AuthError as exc:
        _exit_auth_error(exc)


def remove_shared_grant(args: Any) -> None:
    from agent import anthropic_shared_pool as sp

    try:
        sp.remove_row(str(getattr(args, "target", "")), capability=_cap())
        print("Removed shared staged grant")
    except AuthError as exc:
        _exit_auth_error(exc)


def reset_shared_statuses(args: Any) -> None:
    from agent import anthropic_shared_pool as sp

    try:
        n = sp.reset_statuses(capability=_cap())
        print(f"Reset recoverable exhaustion on {n} shared Anthropic credentials")
    except AuthError as exc:
        _exit_auth_error(exc)


def logout_shared(args: Any) -> None:
    from agent import anthropic_shared_pool as sp

    try:
        sp.clear_shared_namespace(capability=_cap())
        print("Shared Anthropic scope disabled and root shared namespace cleared")
    except AuthError as exc:
        _exit_auth_error(exc)


def print_shared_list_status() -> bool:
    """If shared scope active, print redacted shared status and return True."""
    from agent import anthropic_shared_pool as sp

    try:
        if not sp.is_shared_scope_active():
            # Still show dormant staging info when listing anthropic?
            return False
        info = sp.list_redacted(require_active=True)
    except AuthError as exc:
        _exit_auth_error(exc)
        return True
    print(f"anthropic shared scope (revision={info['revision']}, strategy={info['strategy']}):")
    print(f"  root: {info['root_auth']}")
    print(f"  epoch: {info.get('epoch')}")
    print(f"  attested: {info['account_distinctness_attested']}")
    for i, e in enumerate(info["entries"], start=1):
        status = e.get("last_status") or "ok"
        print(
            f"  #{i}  {e['label']:<20} gen={e['token_generation']} "
            f"status={status} id={e['id']}"
        )
    print()
    return True
