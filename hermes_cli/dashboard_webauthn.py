"""``hermes dashboard webauthn`` — enrol/forget passkeys for the dashboard gate.

``enrol`` gives a headless box a no-browser first credential: it writes a
single-use, 10-minute enrolment token into the credential store and prints the
``<origin>/auth/webauthn/enrol?token=…`` URL to open in a browser that presents
a configured origin (TLS hostname or the localhost SSH tunnel). ``forget``
removes enrolled credentials by label/id or wholesale — the operator's admin
surface for lost or suspicious credentials (single-credential lockout is
otherwise already covered by the password provider fallback).
"""

from __future__ import annotations

import sys
from urllib.parse import quote


def _resolve_provider_settings() -> dict:
    """Resolve ``dashboard.webauthn`` config exactly as the server plugin does;
    print the actionable skip reason and exit on any misconfiguration."""
    from plugins.dashboard_auth._shared import SkipRegistration
    from plugins.dashboard_auth.webauthn import _settings

    try:
        return _settings()
    except SkipRegistration as skip:
        print(f"✗ {skip.reason}")
        sys.exit(1)


def cmd_webauthn_enrol(args) -> None:
    """Issue a one-shot enrolment URL: ``hermes dashboard webauthn enrol``."""
    from plugins.dashboard_auth.webauthn import WebauthnDashboardAuthProvider

    kwargs = _resolve_provider_settings()
    provider = WebauthnDashboardAuthProvider(**kwargs)
    origins = tuple(provider._origins)  # noqa: SLF001
    origin_arg = (getattr(args, "origin", "") or "").strip().rstrip("/")
    chosen = ""
    if origin_arg:
        for entry in origins:
            if entry == origin_arg or entry.split("://", 1)[-1] == origin_arg:
                chosen = entry
                break
        if not chosen:
            print(f"✗ --origin {origin_arg!r} is not in dashboard.webauthn.origins "
                  f"({', '.join(origins)}).")
            sys.exit(1)
    elif len(origins) > 1:
        print("Multiple configured origins; pick one with --origin:")
        for entry in origins:
            print(f"  - {entry}")
        sys.exit(1)
    else:
        chosen = origins[0]
    token = provider.issue_enrolment_token()
    url = f"{chosen}/auth/webauthn/enrol?token={token}"
    label = (getattr(args, "label", "") or "").strip()
    if label:
        url += f"&label={quote(label, safe='')}"

    print("Open this URL in a browser on the configured origin (secure context):")
    print(f"  {url}")
    print("")
    print("The token is single-use and expires in 10 minutes. After enrolling,")
    print("enrol a SECOND passkey (another device or a hardware key) so a lost")
    print("authenticator cannot lock you out.")


def cmd_webauthn_forget(args) -> None:
    """Remove enrolled credentials: ``hermes dashboard webauthn forget``."""
    from plugins.dashboard_auth._shared import SkipRegistration
    from plugins.dashboard_auth.webauthn import _TAG, logger, store
    from plugins.dashboard_auth.webauthn import store as webauthn_store

    del webauthn_store
    forget_all = bool(getattr(args, "all_credentials", False))
    label = (getattr(args, "label", "") or "").strip()
    credential_id = (getattr(args, "credential_id", "") or "").strip()
    if not (forget_all or label or credential_id):
        print("Nothing to forget: pass --all, --label <label> or --credential-id <b64url rawId>.")
        print("(The password provider stays your recovery path regardless.)")
        sys.exit(1)
    from plugins.dashboard_auth._shared import load_config_section

    section = load_config_section(logger, _TAG, "dashboard", "webauthn")
    if not section:
        print("✗ dashboard.webauthn is not configured; there is no credential store "
              "to read. Add the dashboard.webauthn block to config.yaml first.")
        sys.exit(1)
    store_path = store.resolve_store_path(str(section.get("credential_store", "")).strip())
    try:
        doc = store.load(store_path)
    except Exception as exc:  # noqa: BLE001 - fail closed, never erase on a bad read
        print(f"✗ Refusing to edit the credential store: {exc}")
        sys.exit(1)
    credentials = doc.get("credentials") or []
    keep, removed = [], []
    for record in credentials:
        matches = (
            (forget_all and not label and not credential_id)
            or (bool(label) and record.get("label", "") == label)
            or (bool(credential_id) and record.get("credential_id", "") == credential_id))
        (removed if matches else keep).append(record)
    if not removed:
        print("No matching credential found. Enrolled:")
        for record in credentials:
            print(f"  - {record.get('label', '?')}  {record.get('credential_id', '')[:24]}…  "
                  f"(enrolled {record.get('created_at', '?')})")
        sys.exit(1)
    doc["credentials"] = keep
    if not keep:
        doc.pop("rp_id", None)  # a store with zero credentials is origin-agnostic again
    store.prune_pending_enrolments(doc)
    store.save(store_path, doc)
    for record in removed:
        print(f"forgot: {record.get('label', '?')}  {record.get('credential_id', '')[:24]}…")
    if not keep:
        print("All credentials removed. New enrolments append to the same file.")
    else:
        print(f"{len(keep)} credential(s) remain enrolled.")