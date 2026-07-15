"""``hermes oauth-broker`` command handlers.

All privileged boundaries (device flow, Keychain, launchctl, HTTP, migration)
sit behind module-level seams so tests inject fakes and production resolves
real implementations lazily. Nothing here prints or logs secret values —
confirmations use aliases and one-way fingerprints only.
"""

from __future__ import annotations

import json
import platform
import secrets
import time
from pathlib import Path
from typing import Optional

from agent.oauth_broker.models import (
    ACCOUNT_ALIASES,
    CLIENT_KEY_KEYCHAIN_ACCOUNT,
    CLIENT_KEY_KEYCHAIN_SERVICE,
    GrantStoreError,
    OAuthGrant,
)

_EXPIRING_WINDOW_SECONDS = 120  # matches AccountSlot's refresh skew


# ── injectable boundaries ────────────────────────────────────────────────────


def _is_darwin() -> bool:
    return platform.system() == "Darwin"


def _grant_store():
    from agent.oauth_broker.grant_store import KeychainGrantStore

    return KeychainGrantStore()


def _client_key_ref():
    from agent.keychain_secret import KeychainRef

    return KeychainRef(
        service=CLIENT_KEY_KEYCHAIN_SERVICE, account=CLIENT_KEY_KEYCHAIN_ACCOUNT
    )


def _read_client_key() -> str:
    from agent.keychain_secret import read_keychain_secret

    return read_keychain_secret(_client_key_ref())


def _write_client_key(value: str) -> None:
    from agent.keychain_secret import write_keychain_secret

    write_keychain_secret(_client_key_ref(), value)


def _device_code_login() -> dict:
    from hermes_cli.auth import _codex_device_code_login

    return _codex_device_code_login()


def _run_broker(**kwargs) -> None:
    from agent.oauth_broker.server import run_broker

    run_broker(**kwargs)


def _broker_state_dir() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "oauth-broker"


def _account_process_lock(alias: str):
    from agent.oauth_broker.account_slot import account_process_lock

    return account_process_lock(_broker_state_dir(), alias)


def _plist_path() -> Path:
    from agent.oauth_broker.service import broker_launchd_plist_path

    return broker_launchd_plist_path()


def _render_plist(port: int) -> bytes:
    from agent.oauth_broker.service import render_broker_launchd_plist

    return render_broker_launchd_plist(port=port)


def _runner():
    import subprocess

    return subprocess.run


def _confirm(prompt: str) -> bool:
    try:
        return input(prompt).strip().lower() in {"y", "yes"}
    except (EOFError, KeyboardInterrupt):
        return False


def _broker_health(port: int) -> Optional[dict]:
    import httpx

    try:
        with httpx.Client(timeout=5.0, trust_env=False) as client:
            response = client.get(f"http://127.0.0.1:{port}/health")
        if response.status_code != 200:
            return None
        return response.json()
    except Exception:
        return None


def _broker_health_detailed(port: int) -> Optional[dict]:
    import httpx

    try:
        key = _read_client_key()
        with httpx.Client(timeout=5.0, trust_env=False) as client:
            response = client.get(
                f"http://127.0.0.1:{port}/health/detailed",
                headers={"Authorization": f"Bearer {key}"},
            )
        if response.status_code not in (200, 503):
            return None
        payload = response.json()
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _plan_migration(profiles_root: Path, groups: dict, port: int) -> dict:
    from agent.oauth_broker.migration import plan_migration

    return plan_migration(profiles_root, groups, port=port)


def _apply_migration(profiles_root: Path, snapshot: dict, journal_path: Path) -> dict:
    from agent.oauth_broker.migration import apply_migration

    return apply_migration(profiles_root, snapshot, journal_path=journal_path)


def _rollback_migration(
    profiles_root: Path,
    snapshot: dict,
    journal_path: Path,
) -> dict:
    from agent.oauth_broker.migration import rollback_migration

    return rollback_migration(
        profiles_root,
        snapshot,
        journal_path=journal_path,
    )


# ── helpers ──────────────────────────────────────────────────────────────────


def _require_darwin(action: str) -> bool:
    if _is_darwin():
        return True
    print(
        f"oauth-broker {action} requires macOS (Keychain-backed); "
        "refusing on this platform."
    )
    return False


def _fingerprint(value: str) -> str:
    from agent.credential_persistence import _fingerprint_value

    return _fingerprint_value(value) or "sha256:unavailable"


def _account_id_from_tokens(tokens: dict) -> str:
    account_id = str(tokens.get("account_id") or "").strip()
    if account_id:
        return account_id
    import base64

    try:
        parts = str(tokens.get("access_token") or "").split(".")
        payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload_b64))
        claim = claims.get("https://api.openai.com/auth", {}).get(
            "chatgpt_account_id"
        )
        return str(claim or "").strip()
    except Exception:
        return ""


# ── handlers ─────────────────────────────────────────────────────────────────


def _handle_run(args) -> int:
    if not _require_darwin("run"):
        return 1
    try:
        local_key = _read_client_key()
    except Exception:
        print(
            "oauth-broker: no local client key in the Keychain (fail closed). "
            "Run `hermes oauth-broker install` first."
        )
        return 1
    from agent.oauth_broker.account_slot import AccountSlot

    store = _grant_store()
    # Preload and validate every grant before binding: a broker that cannot
    # serve all three accounts fails closed instead of half-starting.
    grants = {}
    missing = []
    for alias in ACCOUNT_ALIASES:
        try:
            grants[alias] = store.load(alias)
        except Exception:
            missing.append(alias)
    if missing:
        print(
            f"oauth-broker: missing Keychain grants for {missing}; failing "
            "closed. Run `hermes oauth-broker auth login <alias>` first."
        )
        return 1
    state_dir = _broker_state_dir()
    slots = {
        alias: AccountSlot(
            alias,
            grant_store=store,
            state_dir=state_dir,
            initial_grant=grants[alias],
        )
        for alias in ACCOUNT_ALIASES
    }
    blocked = []
    for alias, slot in slots.items():
        status = slot.status()
        if not status.present or not status.healthy or status.persistence_degraded:
            blocked.append(
                f"{alias}:{status.terminal_category or 'persistence_degraded'}"
            )
    if blocked:
        print(
            "oauth-broker: account slots are not startup-ready "
            f"({', '.join(blocked)}); refusing to bind. Complete a fresh "
            "`hermes oauth-broker auth login <alias>` for blocked accounts."
        )
        return 1
    _run_broker(host=args.host, port=args.port, slots=slots, local_key=local_key)
    return 0


def _handle_status(args) -> int:
    # Authenticated /health/detailed only: the unauthenticated liveness
    # route could be spoofed by any local process squatting the port.
    detailed = _broker_health_detailed(args.port)
    if detailed is None:
        print(
            f"oauth-broker: no authenticated broker on 127.0.0.1:{args.port} "
            "(fail closed — profiles do not fall back to legacy OAuth)."
        )
        return 1
    print(f"oauth-broker: {json.dumps(detailed)}")
    return 0 if detailed.get("status") == "ok" else 1


def _handle_doctor(args) -> int:
    checks = []
    checks.append(("macOS", _is_darwin()))
    try:
        _read_client_key()
        checks.append(("client key in Keychain", True))
    except Exception:
        checks.append(("client key in Keychain", False))
    store = _grant_store()
    for alias in ACCOUNT_ALIASES:
        try:
            store.load(alias)
            checks.append((f"grant {alias}", True))
        except Exception:
            checks.append((f"grant {alias}", False))
    checks.append(("launchd plist installed", _plist_path().exists()))
    checks.append((f"broker /health on :{args.port}", _broker_health(args.port) is not None))
    detailed = _broker_health_detailed(args.port)
    checks.append(
        (
            f"broker readiness on :{args.port}",
            detailed is not None and detailed.get("status") == "ok",
        )
    )
    ok = True
    for name, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'} {name}")
        ok = ok and passed
    return 0 if ok else 1


def _handle_install(args) -> int:
    if not _require_darwin("install"):
        return 1
    try:
        _read_client_key()
    except Exception:
        _write_client_key(secrets.token_urlsafe(32))
        print(
            "oauth-broker: generated local client key "
            "(stored in the Keychain; never displayed)."
        )
    from agent.oauth_broker.service import install_broker_service

    apply_now = False
    if getattr(args, "apply", False):
        apply_now = _confirm(
            "Load ai.hermes.oauth-broker into launchd now? [y/N] "
        )
        if not apply_now:
            print("oauth-broker: install left in render-only mode.")
    result = install_broker_service(
        plist_path=_plist_path(),
        content=_render_plist(args.port),
        apply=apply_now,
        runner=_runner() if apply_now else None,
    )
    print(f"oauth-broker: plist written to {result['plist_path']}")
    print(f"oauth-broker: bootstrap argv: {' '.join(result['bootstrap'])}")
    if result["executed"]:
        print("oauth-broker: service loaded via launchctl.")
    return 0


def _handle_uninstall(args) -> int:
    if not _require_darwin("uninstall"):
        return 1
    from agent.oauth_broker.service import uninstall_broker_service

    apply_now = False
    if getattr(args, "apply", False):
        apply_now = _confirm(
            "Boot ai.hermes.oauth-broker out of launchd now? [y/N] "
        )
    result = uninstall_broker_service(
        plist_path=_plist_path(),
        apply=apply_now,
        runner=_runner() if apply_now else None,
    )
    print(f"oauth-broker: bootout argv: {' '.join(result['bootout'])}")
    print(
        "oauth-broker: Keychain grants are untouched — use "
        "`hermes oauth-broker auth logout <alias> --yes` separately."
    )
    return 0


def _handle_auth_login(args) -> int:
    if not _require_darwin("auth login"):
        return 1
    from hermes_cli.auth import AuthError

    try:
        creds = _device_code_login()
    except AuthError as exc:
        print(f"oauth-broker: login for account {args.alias} failed: {exc}")
        return 1
    tokens = (creds or {}).get("tokens") or {}
    access = str(tokens.get("access_token") or "").strip()
    refresh = str(tokens.get("refresh_token") or "").strip()
    account_id = _account_id_from_tokens(tokens)
    if not access or not refresh or not account_id:
        print("oauth-broker: login did not return a complete grant; nothing stored.")
        return 1
    from agent.oauth_broker.account_slot import _jwt_expiry

    account_fp = _fingerprint(account_id)
    if not _confirm(
        f"Store the completed Codex login under account {args.alias} "
        f"(account-id fingerprint {account_fp})? [y/N] "
    ):
        print(
            f"oauth-broker: login for account {args.alias} discarded; "
            "nothing stored."
        )
        return 1
    grant = OAuthGrant(
        access_token=access,
        refresh_token=refresh,
        expires_at=_jwt_expiry(access) or (time.time() + 3600.0),
        account_id=account_id,
    )
    with _account_process_lock(args.alias):
        _grant_store().replace(args.alias, grant)
    print(
        f"oauth-broker: grant stored under account {args.alias} "
        f"(account-id fingerprint {account_fp})."
    )
    return 0


def _handle_auth_status(args) -> int:
    store = _grant_store()
    detailed = _broker_health_detailed(args.port)
    healthy_by_alias = {}
    if detailed:
        for entry in detailed.get("accounts", []):
            healthy_by_alias[entry.get("alias")] = entry.get("healthy")
    aliases = [args.alias] if getattr(args, "alias", None) else list(ACCOUNT_ALIASES)
    now = time.time()
    for alias in aliases:
        try:
            grant = store.load(alias)
            present = True
            expiring = grant.expires_at <= now + _EXPIRING_WINDOW_SECONDS
        except GrantStoreError:
            present, expiring = False, False
        healthy = healthy_by_alias.get(alias)
        healthy_text = "unknown" if healthy is None else str(bool(healthy))
        print(
            f"{alias} present={present} expiring={expiring} healthy={healthy_text}"
        )
    return 0


def _handle_auth_logout(args) -> int:
    if not getattr(args, "yes", False):
        print(
            f"oauth-broker: refusing to delete the account {args.alias} grant "
            "without --yes."
        )
        return 1
    with _account_process_lock(args.alias):
        _grant_store().delete(args.alias)
    print(f"oauth-broker: grant for account {args.alias} deleted from the Keychain.")
    return 0


def _handle_migrate(args) -> int:
    groups = json.loads(Path(args.groups).read_text(encoding="utf-8"))
    snapshot = _plan_migration(Path(args.profiles_root), groups, args.port)
    from agent.oauth_broker.migration import save_snapshot

    snapshot_path = save_snapshot(snapshot, Path(args.snapshot))
    counts = snapshot.get("group_counts", {})
    print(f"oauth-broker: dry-run snapshot written to {snapshot_path}")
    print(f"oauth-broker: group counts {json.dumps(counts, sort_keys=True)}")
    if not getattr(args, "apply", False):
        print("oauth-broker: dry-run only; re-run with --apply to migrate.")
        return 0
    profile_count = len(snapshot.get("profiles", {}))
    if not _confirm(
        f"Apply the broker migration to {profile_count} profiles now? [y/N] "
    ):
        print(
            "oauth-broker: apply declined; dry-run snapshot kept, "
            "nothing migrated."
        )
        return 0
    report = _apply_migration(
        Path(args.profiles_root),
        snapshot,
        Path(str(snapshot_path) + ".journal"),
    )
    print(f"oauth-broker: migrated profiles {report['written']}")
    return 0


def _handle_rollback(args) -> int:
    if not getattr(args, "yes", False):
        print("oauth-broker: refusing rollback without --yes.")
        return 1
    snapshot_path = Path(args.snapshot)
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    report = _rollback_migration(
        Path(args.profiles_root),
        snapshot,
        Path(str(snapshot_path) + ".rollback.journal"),
    )
    print(f"oauth-broker: restored profiles {report['restored']}")
    return 0


def cmd_oauth_broker(args) -> int:
    command = str(getattr(args, "oauth_broker_command", "") or "")
    if command == "auth":
        auth_command = str(getattr(args, "oauth_broker_auth_command", "") or "")
        auth_handlers = {
            "login": _handle_auth_login,
            "status": _handle_auth_status,
            "logout": _handle_auth_logout,
        }
        handler = auth_handlers.get(auth_command)
        if handler is None:
            print("usage: hermes oauth-broker auth {login,status,logout} ...")
            return 2
        return handler(args)
    handlers = {
        "run": _handle_run,
        "status": _handle_status,
        "doctor": _handle_doctor,
        "install": _handle_install,
        "uninstall": _handle_uninstall,
        "migrate": _handle_migrate,
        "rollback": _handle_rollback,
    }
    handler = handlers.get(command)
    if handler is None:
        print(
            "usage: hermes oauth-broker "
            "{run,status,doctor,install,uninstall,auth,migrate,rollback} ..."
        )
        return 2
    return handler(args)


__all__ = ["cmd_oauth_broker"]
