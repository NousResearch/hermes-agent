"""``hermes dashboard webauthn`` — enrol/forget passkeys for the dashboard gate.

``setup`` writes the ``dashboard.webauthn`` config block (rp_id / rp_name /
origins): the values that MUST be decided before a first credential exists,
because passkeys bind to ``rp_id`` forever. ``enrol`` gives a headless box a
no-browser first credential: it writes a single-use, 10-minute enrolment token
into the credential store and prints the ``<origin>/auth/webauthn/enrol?token=…``
URL to open in a browser that presents a configured origin (TLS hostname or the
localhost SSH tunnel). ``forget`` removes enrolled credentials by label/id or
wholesale — the operator's admin surface for lost or suspicious credentials
(single-credential lockout is otherwise already covered by the password
provider fallback).
"""

from __future__ import annotations

import re
import shutil
import sys
from time import strftime
from urllib.parse import quote, urlsplit


def _resolve_provider_settings() -> dict:
    """Resolve ``dashboard.webauthn`` config exactly as the server plugin does;
    print the actionable skip reason and exit on any misconfiguration."""
    from plugins.dashboard_auth._shared import SkipRegistration
    from plugins.dashboard_auth.webauthn import _settings

    try:
        return _settings()
    except SkipRegistration as skip:
        print(f"✗ {skip.reason}")
        if skip.reason.startswith("dashboard.webauthn"):
            print("→ Configure it first: hermes dashboard webauthn setup")
        sys.exit(1)


def cmd_webauthn_enrol(args) -> None:
    """Issue a one-shot enrolment URL: ``hermes dashboard webauthn enrol``."""
    import time

    from plugins.dashboard_auth.webauthn import store as webauthn_store
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
    expires_at = int(time.time()) + webauthn_store.ENROLMENT_TOKEN_TTL_SECONDS
    store_path = getattr(provider, "_store_path", "")
    print(f"Token {token[:8]}… pending in {store_path or 'the dashboard_webauthn.json store'};")
    print(f"single-use, expires {time.strftime('%H:%M:%S', time.gmtime(expires_at))}Z "
          f"({int(webauthn_store.ENROLMENT_TOKEN_TTL_SECONDS / 60)} minutes) — open it before then.")
    print("")
    print("After enrolling, enrol a SECOND passkey (another device or a hardware")
    print("key) so a lost authenticator cannot lock you out. Re-running enrol does")
    print("NOT invalidate earlier tokens, but a link whose token was already")
    print("consumed (successful enrolment) shows 'unknown, already used' forever.")


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


# ---------------------------------------------------------------------------
# setup — write the dashboard.webauthn block BEFORE any enrolment
# ---------------------------------------------------------------------------

# A bare hostname: "localhost", or ≥2 dot-separated labels (a registrable
# domain can never be a single word). No scheme, port, path or trailing dot.
_HOST_RE = re.compile(
    r"(?:localhost|[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
    r"(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9]))+)")


def _preflight(args) -> None:
    """Fail fast on prerequisites BEFORE the operator types anything.

    Two checks, both blocking (a config the box can't serve is noise):
    1. config.yaml must be readable — save_config fails closed later, but
       that is after twenty keystrokes; say it here instead.
    2. fido2 (the `webauthn` extra) must be importable — the pairing check
       and the provider both need it. Interactive runs get a one-prompt
       install (PM unions the extra into the recorded graph); non-interactive
       runs print the remedy and stop. --skip-preflight stages the config
       without either check (for provisioning a config for ANOTHER box).
    """
    from hermes_cli.config import load_config_readonly

    try:
        load_config_readonly()
    except Exception as exc:  # noqa: BLE001 — same failure save_config would hit later
        print(f"✗ config.yaml is unreadable — fix it before setup; nothing was written: {exc}")
        sys.exit(1)
    if getattr(args, "skip_preflight", False):
        return
    import importlib.util

    if importlib.util.find_spec("fido2") is not None:
        return
    print("✗ fido2 (the 'webauthn' extra) is not installed — passkeys cannot be "
          "provisioned or served on this box without it.")
    remedy = "pip install 'hermes-agent[webauthn]' (or: hermes pm install --extra webauthn)"
    installed = False
    if sys.stdin.isatty():
        try:
            import pm
        except Exception:  # noqa: BLE001 — not a PM-managed install here
            pm = None
        if pm is not None and _confirm("Install the 'webauthn' extra now?", True):
            try:
                pm.sync_venv(["webauthn"], explicit=True)
                print("  ✓ webauthn extra installed")
                installed = True
            except Exception as exc:  # noqa: BLE001 — fall through to the manual remedy
                print(f"  ✗ {exc}")
    if not installed:
        print(f"→ {remedy}")
        print("  Run setup again after installing; --skip-preflight stages the "
              "config on THIS box without the check (use only when the "
              "dashboard runs elsewhere).")
        sys.exit(1)


class _OriginRejected(Exception):
    """A candidate origin fails a validation rule; message = operator hint."""


def _clean_rp_id(raw: str) -> tuple:
    """A paste-tolerant rp_id cleaner: accepts a bare domain or a full URL and
    returns ``(candidate, problem_text)``."""
    raw = str(raw or "").strip().strip('"')
    if not raw:
        return None, ""
    url_host = urlsplit(raw).hostname if "://" in raw else None
    candidate = (url_host or raw).strip(".").lower()
    if _HOST_RE.fullmatch(candidate) is None:
        return None, (f"{candidate!r} is not a bare hostname — no scheme, port, "
                      "path or trailing dot; only 'localhost' may be a single word.")
    return candidate, ""


def _pairing_text(canonical: str, rp_id: str, host: str) -> str:
    if host == "localhost" or host.endswith(".localhost") or host == "127.0.0.1":
        return (f"{canonical} cannot pair with rp_id {rp_id!r}: a localhost origin "
                "needs rp_id 'localhost' — and then no https:// origin can be "
                "enrolled against the same store.")
    return (f"{canonical} does not pair with rp_id {rp_id!r}: the RP ID must equal "
            "the origin's host or be a registrable suffix of it. Fix rp_id or the "
            "origin; a mismatched pairing cannot work after enrolment.")


def _structural_pairing(rp_id: str, host: str) -> bool:
    return host == rp_id or host.endswith("." + rp_id)


def _validated_origin(raw: str, rp_id: str) -> str:
    """Canonicalize + validate one origin exactly as the server plugin would
    (https-only off localhost; fido2's ``verify_rp_id`` when the extra is
    installed here, a structural proxy of the W3C rule when not — the server
    re-validates with the real library at boot either way)."""
    from plugins.dashboard_auth.webauthn import _normalize_origin

    canonical = _normalize_origin(raw)
    if canonical is None:
        raise _OriginRejected(f"{raw!r} is not a valid origin (use scheme://host[:port]).")
    host = urlsplit(canonical).hostname or ""
    if canonical.startswith("http://") and host != "localhost" and not host.endswith(".localhost"):
        raise _OriginRejected(
            f"{canonical} is plain http on a non-localhost host. WebAuthn needs a "
            "secure context: use https:// (reverse proxy) or the SSH-tunnel path "
            "'http://localhost:<port>'.")
    verify = None
    try:
        from plugins.dashboard_auth.webauthn import crypto as _wa_crypto
        verify = _wa_crypto.verify_rp_id
    except Exception:  # noqa: BLE001 — any import failure here means "no fido2 on this box"
        verify = None
    if verify is not None:
        if not verify(rp_id, canonical):
            raise _OriginRejected(_pairing_text(canonical, rp_id, host))
    elif not _structural_pairing(rp_id, host):
        raise _OriginRejected(_pairing_text(canonical, rp_id, host))
    return canonical


def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        return input(f"{prompt}{suffix}: ").strip() or default
    except EOFError:
        print("\n✗ no interactive input available; pass --rp-id/--rp-name/--origin flags.")
        sys.exit(1)


def _confirm(prompt: str, default: bool = True) -> bool:
    try:
        answer = input(f"{prompt} {'[Y/n]' if default else '[y/N]'}: ").strip().lower()
    except EOFError:
        print("\n✗ no interactive input available; pass --rp-id/--rp-name/--origin flags.")
        sys.exit(1)
    if not answer:
        return default
    return answer in ("y", "yes")


def _acquire_rp_id(args, section: dict) -> str:
    """--rp-id wins; otherwise prompt (loop on invalid input; default = current)."""
    candidate, problem = _clean_rp_id(str(getattr(args, "rp_id", "") or ""))
    if candidate:
        return candidate
    if problem:
        print(f"✗ --rp-id: {problem}")
        sys.exit(1)
    if not sys.stdin.isatty():
        print("✗ no interactive input available and --rp-id was not given. Flags:")
        print("    hermes dashboard webauthn setup --rp-id <domain> --rp-name <label> --origin <url>")
        sys.exit(1)
    current = str(section.get("rp_id", "")).strip().strip(".")
    prompt = "RP ID — registrable domain, e.g. dash.example.com (passkeys bind to it FOREVER)"
    if current:
        prompt += f"; press Enter to keep {current!r}"
    for _ in range(5):
        candidate, problem = _clean_rp_id(_ask(prompt, current))
        if candidate:
            return candidate
        print(f"  ✗ {problem}")
    print("✗ rp_id stayed invalid after 5 tries; nothing was written.")
    sys.exit(1)  # type: ignore[unreachable]


def _acquire_rp_name(args, rp_id: str, section: dict) -> str:
    name = str(getattr(args, "rp_name", "") or "").strip()
    if name:
        return name
    if sys.stdin.isatty():
        current = str(section.get("rp_name", "")).strip()
        name = _ask("RP name — the label the browser shows in its passkey UI", current or rp_id)
    return name or rp_id


def _acquire_origins(args, rp_id: str, section: dict) -> list:
    """--origin (repeatable) wins; otherwise prompt with sane proposals."""
    flag_origins = [str(o).strip() for o in (getattr(args, "origin", None) or []) if str(o).strip()]
    try:
        if flag_origins:
            canonical = [_validated_origin(o, rp_id) for o in flag_origins]
        else:
            if not sys.stdin.isatty():
                print("✗ no interactive input available and --origin was not given. Flags:")
                print("    hermes dashboard webauthn setup --rp-id <domain> --rp-name <label> --origin <url>")
                sys.exit(1)
            canonical = []
            if rp_id == "localhost":
                port = _ask("Local port your SSH tunnel forwards to the dashboard", "9119")
                canonical.append(_validated_origin(f"http://localhost:{port}", rp_id))
            else:
                proposed = f"https://{rp_id}"
                print(f"  The origin is what the browser's address bar shows, e.g. {proposed}.")
                if _confirm(f"Include {proposed}?", True):
                    canonical.append(_validated_origin(proposed, rp_id))
                if canonical and _confirm("Do users ALSO reach it via www.<domain>?", False):
                    canonical.append(_validated_origin(f"https://www.{rp_id}", rp_id))
                while True:
                    entry = _ask("Another origin (scheme://host[:port]; blank = done)")
                    if not entry:
                        break
                    try:
                        extra = _validated_origin(entry, rp_id)
                    except _OriginRejected as rejected:
                        print(f"  ✗ {rejected}")
                        continue
                    if extra in canonical:
                        print("  already included.")
                        continue
                    canonical.append(extra)
    except _OriginRejected as rejected:
        print(f"✗ --origin: {rejected}")
        sys.exit(1)
    if not canonical:
        print("✗ setup needs at least one origin; nothing was written.")
        sys.exit(1)
    return canonical


def _store_has_credentials(section: dict) -> bool:
    """True when the credential store holds/pends anything (or cannot be read —
    conservative, since changing rp_id would strand it)."""
    from plugins.dashboard_auth.webauthn import store as webauthn_store

    try:
        doc = webauthn_store.load(webauthn_store.resolve_store_path(
            str(section.get("credential_store", "")).strip()))
    except Exception:  # noqa: BLE001 — a unreadable store is treated as populated
        return True
    return bool((doc.get("credentials") or []) or (doc.get("pending_enrolments") or {}))


def _write_webauthn_block(rp_id: str, rp_name: str, origins: list) -> tuple:
    """Timestamped-backup + partial save of the three decided keys; returns
    ``(config_path, backup_path_or_None)``."""
    from hermes_cli import config as config_mod

    config_path = config_mod.get_config_path()
    backup = None
    if config_path.exists():
        backup = config_path.with_name(f"config.yaml.bak-{strftime('%Y%m%d-%H%M%S')}")
        shutil.copy2(config_path, backup)
    partial = {"dashboard": {"webauthn": {
        "rp_id": rp_id, "rp_name": rp_name, "origins": origins}}}
    preserve = {("dashboard", "webauthn", key) for key in ("rp_id", "rp_name", "origins")}
    try:
        config_mod.save_config(partial, preserve_keys=preserve, merge_existing=True)
    except Exception as exc:  # noqa: BLE001 — the fail-closed reader refuses damaged files
        print(f"✗ config.yaml was not written (nothing changed): {exc}")
        sys.exit(1)
    return str(config_path), str(backup) if backup else None


def cmd_webauthn_setup(args) -> None:
    """Write the ``dashboard.webauthn`` block: ``hermes dashboard webauthn setup``."""
    import logging

    from plugins.dashboard_auth._shared import SkipRegistration, load_config_section

    log = logging.getLogger("hermes_cli.dashboard_webauthn")

    _preflight(args)
    section = load_config_section(log, "dashboard-webauthn-setup", "dashboard", "webauthn")
    existing_rp_id = str(section.get("rp_id", "")).strip().strip(".")

    rp_id = _acquire_rp_id(args, section)
    if existing_rp_id and rp_id != existing_rp_id and _store_has_credentials(section):
        warn = (
            f"⚠ {existing_rp_id!r} already has enrolled credentials in the store; every "
            f"passkey is bound to its rp_id — switching to {rp_id!r} makes ALL of them "
            "unusable (remove them with 'hermes dashboard webauthn forget --all' or "
            "delete the store).")
        if getattr(args, "force", False):
            print(warn)
            print("  proceeding (--force given).")
        elif sys.stdin.isatty():
            print(warn)
            if input("Type YES to switch rp_id anyway: ").strip() != "YES":
                print("Cancelled — nothing was written.")
                sys.exit(1)
        else:
            print(warn)
            print("✗ re-run with --force to confirm the switch.")
            sys.exit(1)
    rp_name = _acquire_rp_name(args, rp_id, section)
    origins = _acquire_origins(args, rp_id, section)

    if sys.stdin.isatty() and not (
            str(getattr(args, "rp_id", "") or "").strip()
            and str(getattr(args, "rp_name", "") or "").strip()
            and getattr(args, "origin", None)):
        print()
        print("Writing this dashboard.webauthn block:")
        print(f"  rp_id:   {rp_id}")
        print(f"  rp_name: {rp_name}")
        for entry in origins:
            print(f"  origin:  {entry}")
        if not _confirm("Write to config.yaml?", True):
            print("Cancelled — nothing written.")
            return

    config_path, backup = _write_webauthn_block(rp_id, rp_name, origins)

    resolution = "the server re-validates it at boot"
    try:
        from plugins.dashboard_auth.webauthn import _settings
        _settings()
        resolution = "resolves cleanly"
    except SkipRegistration as skip:
        resolution = f"the server will refuse it at boot: {skip.reason}"
    except Exception as exc:  # noqa: BLE001 — informational only; the server decides
        resolution = f"could not re-check on this box ({type(exc).__name__}: {exc}); the server re-validates at boot"

    print()
    print(f"✓ dashboard.webauthn written to {config_path}" + (f"; backup: {backup}" if backup else ""))
    print(f"  rp_id:   {rp_id}   ← frozen per credential; changing it later bricks every passkey")
    print(f"  rp_name: {rp_name}")
    print(f"  origins: {', '.join(origins)}")
    print(f"  config {resolution}.")
    print()
    print("Next:")
    print("  1. Restart the dashboard process — the passkey provider registers at boot.")
    print("  2. hermes dashboard webauthn enrol --label 'laptop touchid'")
    print("     Open the printed URL in a browser on a device that has the passkey")
    print("     manager you want (Touch ID, Windows Hello, iCloud/Google password")
    print("     manager, or a plugged-in security key). The BROWSER creates the key")
    print("     and sends only the public half to the dashboard — the private key")
    print("     never leaves the authenticator. Authenticator on another device?")
    print("     The browser shows a QR code for cross-device setup.")
    print("  3. Enrol a SECOND key from a different device so a lost authenticator")
    print("     cannot lock you out.")