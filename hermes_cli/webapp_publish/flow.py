"""Interactive setup flow for publishing the browser-hosted Desktop.

Reached from the platform menu of ``hermes gateway setup`` and ``hermes setup``
(the 🖥️ Hermes Desktop (browser) row; headless publishing, see issue #147984).
Two access modes:

1. LOCAL  — the gateway's webapp binds all interfaces (sign-in gate engages),
   avahi publishes ``<name>.local`` -> this machine's LAN address, and Nous
   sign-in is registered for the ``.local`` callback (the OAuth redirect follows
   the request Host on a wildcard bind).
2. DOMAIN — the user creates a DNS A record to this machine's public IP, we
   verify it resolves, Caddy fronts the loopback webapp with automatic HTTPS,
   and Nous sign-in is registered for the fixed public callback.

Sign-in is Nous-account only: the flow never enables the password provider and
offers to disable it when it finds one configured.

Prompt/print helpers are imported from ``hermes_cli.setup`` inside each function
(the plugin convention) so tests monkeypatch that facade.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_LOCAL_NAME = "hermes-desktop.local"
_DNS_WAIT_SECONDS = 30
_DNS_MAX_ROUNDS = 4


# ── preflight ────────────────────────────────────────────────────────────────

def _supported_host() -> str:
    """'' when this host can run the publishing flow, else what's missing.
    The flow configures Caddy + avahi through systemd and apt — Linux only."""
    if sys.platform != "linux":
        return "Linux"
    if shutil.which("systemctl") is None or not Path("/run/systemd/system").exists():
        return "systemd"
    return ""


def _webapp_supported() -> bool:
    try:
        import importlib.util
        from hermes_cli.config import get_project_root
        return (importlib.util.find_spec("hermes_cli.webapp") is not None
                and (get_project_root() / "apps" / "desktop").is_dir())
    except Exception:  # health: allow BLE001 -- support probe must never raise into the wizard
        return False


def _preflight() -> bool:
    from hermes_cli.setup import print_error, print_info
    from hermes_cli.webapp_access import nous_signed_in
    missing = _supported_host()
    if missing:
        print_error(f"The Hermes Desktop publishing flow needs {missing} (it configures Caddy and avahi).")
        print_info("On this host use 'hermes webapp' for local browser access, or the")
        print_info("Desktop app's Settings → Web Access for remote access.")
        return False
    if not _webapp_supported():
        print_error("This Hermes build cannot serve the Desktop webapp (needs 'hermes webapp' + apps/desktop).")
        print_info("Update Hermes, then run setup again.")
        return False
    if not nous_signed_in():
        print_error("Hermes Desktop sign-in uses your Nous account, and none is signed in here.")
        print_info("Sign in first:  hermes portal login   (a free-tier guest account cannot register sign-in)")
        return False
    return True


def _already_configured() -> bool:
    """True when the webapp is on and the user declines reconfiguring."""
    from hermes_cli.setup import print_info, print_success, prompt_yes_no
    from hermes_cli.webapp_access import access_status
    try:
        status = access_status()
    except Exception:  # health: allow BLE001 -- best-effort status read; unreadable config means not configured
        return False
    if not status.get("enabled"):
        return False
    where = status.get("public_url") or (
        f"http://{status.get('lan_address')}:{status.get('port')}" if status.get("mode") == "lan"
        else f"http://127.0.0.1:{status.get('port')}")
    print_success(f"Hermes Desktop is already enabled ({where}).")
    return not prompt_yes_no("  Reconfigure Hermes Desktop?", False)


# ── shared steps ─────────────────────────────────────────────────────────────

def _ensure_nous_only() -> None:
    """The vision: sign-in via Nous Research only. The flow never enables the
    password ('basic') or self-hosted OIDC providers; when one is already
    configured, offer to disable it via plugins.disabled so the login page
    offers Nous alone. (Non-interactive token providers never appear on the
    login page; the deployment session token is left untouched.)"""
    from hermes_cli.setup import print_info, print_success, print_warning, prompt_yes_no
    from hermes_cli.config import load_config, save_config
    try:
        cfg = load_config()
    except Exception:  # health: allow BLE001 -- best-effort config read; skip enforcement rather than crash setup
        return
    dashboard = cfg.get("dashboard") or {}
    basic = dashboard.get("basic_auth") or {}
    env_basic = any(os.environ.get(k) for k in (
        "HERMES_DASHBOARD_BASIC_AUTH_USERNAME", "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD",
        "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH"))
    oidc = (dashboard.get("oauth") or {}).get("self_hosted") or {}
    env_oidc = bool(os.environ.get("HERMES_DASHBOARD_OIDC_ISSUER") and os.environ.get("HERMES_DASHBOARD_OIDC_CLIENT_ID"))
    found = []
    if basic.get("username") or env_basic:
        found.append(("basic", "username/password"))
    if oidc.get("issuer") or env_oidc:
        found.append(("self-hosted", "self-hosted OIDC"))
    if not found:
        return
    labels = " and ".join(label for _, label in found)
    print_warning(f"Another sign-in provider is configured on this host: {labels}.")
    print_info("Hermes Desktop setup uses Nous-account sign-in only.")
    if not prompt_yes_no("  Disable the other provider(s) (Nous-only)?", True):
        print_warning("Keeping them enabled — the login page will offer more than Nous.")
        return
    plugins = cfg.setdefault("plugins", {})
    disabled = plugins.get("disabled")
    disabled = list(disabled) if isinstance(disabled, list) else []
    for key, _label in found:
        if key not in disabled:
            disabled.append(key)
    plugins["disabled"] = disabled
    save_config(cfg)
    print_success("Other sign-in provider(s) disabled. Nous-only from the next gateway start.")


def _build_bundle() -> bool:
    from hermes_cli.setup import print_error, print_info, print_success
    from hermes_cli.webapp_access import WebappAccessError, build_bundle
    print_info("Building the Desktop renderer bundle (first build takes a few minutes)…")
    try:
        build_bundle()
    except WebappAccessError as exc:
        print_error(str(exc))
        return False
    print_success("Renderer bundle ready.")
    return True


def _offer_restart() -> bool:
    """webapp.* is startup-only. Offer a service restart when one is installed;
    otherwise print the command. Returns True when a restart was performed."""
    from hermes_cli.setup import print_error, print_info, prompt_yes_no
    systemd_restart = None
    try:
        from hermes_cli.gateway import _is_service_running, systemd_restart as _restart
        running = _is_service_running()
        systemd_restart = _restart
    except Exception:  # health: allow BLE001 -- service probe failure becomes instructions, never a crash
        running = False
    if running and systemd_restart is not None and prompt_yes_no("  Restart the gateway now to apply?", True):
        try:
            systemd_restart()
            return True
        except Exception as exc:  # health: allow BLE001 -- restart failure is reported with remediation, not raised
            print_error(f"Restart failed: {exc}")
    print_info("Apply with:  hermes gateway restart   (or 'hermes gateway run' in a terminal)")
    print_info("Note: run that from a shell outside an agent chat — a restart ends active sessions.")
    return False


def _print_instructions(lines: list[str]) -> None:
    from hermes_cli.setup import print_info
    for line in lines:
        print_info(f"  {line}")


def _effective_port(port) -> int:
    """plan() returns None to keep the configured port; Caddy/callbacks need the number."""
    if port is not None:
        return port
    from gateway.webapp_listener import webapp_settings
    from hermes_cli.webapp_access import gateway_home
    return webapp_settings(gateway_home()).port


def _installed_blocks(blocks: list[str]) -> list[str]:
    """Which of the rendered blocks actually landed in the Caddyfile (for the
    mode-switch record; only exact matches are ever cleaned up later)."""
    from . import caddy
    try:
        text = caddy.DEFAULT_CADDYFILE.read_text(encoding="utf-8-sig")
    except OSError:
        return []
    return [b for b in blocks if b in text]


def _switch_mode_cleanup(new_mode: str) -> None:
    """Remove the previous mode's recorded artifacts when the user picks the
    other mode (old Caddy blocks verbatim; offer the mDNS identity restore)."""
    from hermes_cli.setup import print_info, prompt_yes_no
    from hermes_cli.webapp_access import gateway_home
    from . import state
    try:
        actions = state.cleanup_previous_mode(gateway_home(), new_mode, confirm=prompt_yes_no)
    except Exception as exc:  # health: allow BLE001 -- cleanup is best-effort; leftovers are reported, not fatal
        logger.debug("mode-switch cleanup failed: %s", exc)
        return
    for line in actions:
        print_info(f"  {line}")


# ── mode 1: local network (mDNS) ─────────────────────────────────────────────

def _configure_local_redirect(name: str, ip: str, port: int) -> bool:
    """Caddy :80 site redirecting the friendly .local name to the LAN origin.
    Best-effort: without it the IP URL still works and the flow says so."""
    from hermes_cli.setup import print_info, print_success, print_warning
    from . import caddy
    if not caddy.caddy_installed():
        print_warning("Caddy not installed — the .local name will resolve, but open the IP URL below.")
        print_info("Optional:  sudo apt-get install -y caddy   (adds the hermes-desktop.local front door)")
        return False
    caddyfile = caddy.DEFAULT_CADDYFILE
    block = caddy.render_redirect_block(name, ip, port)
    if not caddyfile.exists():
        print_warning(f"No Caddyfile at {caddyfile} — add this block yourself for the .local front door:")
        _print_instructions(block.splitlines())
        return False
    try:
        changed, backup = caddy.upsert_block(caddyfile, name, block)
    except OSError:
        print_warning(f"Cannot write {caddyfile} (need root) — add this block yourself:")
        _print_instructions(block.splitlines())
        return False
    if not caddy.validate(caddyfile):
        if changed and backup:
            caddy.restore_backup(caddyfile, backup)
        print_warning("caddy validate failed — previous config restored; use the IP URL.")
        return False
    if caddy.reload_caddy():
        print_success(f"http://{name} now redirects to the gateway (Caddy).")
        return True
    print_warning("Could not reload Caddy. Run:  sudo systemctl reload caddy")
    return False


def _setup_local() -> None:
    from hermes_cli.setup import (print_error, print_header, print_info, print_success,
                                  print_warning, prompt)
    from hermes_cli.webapp_access import WebappAccessError, lan_address, plan, register, serve
    from . import caddy, mdns, netops

    print_header("Hermes Desktop — local network")
    address = lan_address()
    if address is None:
        print_error("This machine has no network address other devices can reach.")
        return

    name = ""
    while not name:
        candidate = prompt(f"  Local hostname [{DEFAULT_LOCAL_NAME}]", default=DEFAULT_LOCAL_NAME).strip().lower()
        if not candidate:
            return
        if netops.is_local_name(candidate):
            name = candidate
        else:
            print_error("Use a single label ending in .local (e.g. hermes-desktop.local).")

    try:
        host, port, _pinned, callback = plan("lan")
    except WebappAccessError as exc:
        print_error(str(exc))
        return
    port = _effective_port(port)

    # Nous Portal OAuth accepts only IP / real-domain callbacks (never .local),
    # so sign-in is registered for — and completes on — the LAN IP origin.
    try:
        register(callback)
        serve(host, "", port)
    except WebappAccessError as exc:
        print_error(str(exc))
        return
    print_success(f"Nous sign-in registered for {callback}")

    # Avahi publishes one name per local address: giving this machine the
    # desired .local identity means setting the daemon's own host-name — a
    # machine-wide change, so it is always confirmed (skippable).
    previous = mdns.current_hostname(mdns.DEFAULT_AVAHI_CONF)
    desired_label = name.removesuffix(".local")
    if previous == desired_label:
        result = mdns.publish_local_name(name, address)  # idempotent refresh
    else:
        import socket
        from hermes_cli.setup import prompt_yes_no
        current_identity = previous or socket.gethostname()
        if prompt_yes_no(
                f"  Publish {name} by setting this machine's mDNS identity "
                f"(currently {current_identity}.local)?", True):
            result = mdns.publish_local_name(name, address)
        else:
            print_info("Keeping the current mDNS identity — use the direct IP URL below.")
            result = {"published": False, "reason": "kept existing mDNS identity",
                      "instructions": [], "backup": None}
    if result["published"]:
        if mdns.resolves(name, address):
            print_success(f"{name} is live on the local network (resolution verified).")
        else:
            print_success(f"{name} published; resolution from this host unconfirmed — try from a LAN device.")
    else:
        print_warning(f"mDNS publishing incomplete: {result['reason']}")
        if result["instructions"]:
            print_info("Run these, then continue:")
            _print_instructions(result["instructions"])

    redirected = _configure_local_redirect(name, address, port)

    if not _build_bundle():
        return
    _ensure_nous_only()
    restarted = _offer_restart()

    from hermes_cli.webapp_access import gateway_home
    from . import state
    state.save_state(gateway_home(), {
        "mode": "local",
        "mdns_name": name if result["published"] else None,
        "avahi_backup": result.get("backup"),
        "caddy_blocks": _installed_blocks([caddy.render_redirect_block(name, address, port)]) if redirected else [],
        "public_url": "",
    })

    print()
    print_success("🖥️  Hermes Desktop — local access")
    if redirected:
        print_info(f"  Open:  http://{name}   (redirects to the gateway)")
    print_info(f"  Direct:  http://{address}:{port}")
    if not restarted:
        print_info("  (available after the gateway restarts)")
    print()
    print_info("Notes:")
    print_info("  • Devices must be on the same LAN; mDNS does not cross routers.")
    print_info("  • Sign-in completes on the IP address: Nous Portal OAuth requires")
    print_info("    an IP or real domain, so .local names cannot carry the callback.")
    print_info("  • macOS/iOS resolve .local natively; Windows 10 1703+ and Linux")
    print_info("    (avahi/nss-mdns) usually do. Android 12+ (Nov 2021 patch) can too —")
    print_info("    type http://hermes-desktop.local/ with the scheme, or use the IP.")
    print_info("    Older Android and VPN-connected devices: use the direct IP URL.")
    print_info("  • HTTP on LAN: clipboard/mic/notifications need a secure context;")
    print_info("    for full features use the custom-domain (HTTPS) mode instead.")
    if shutil.which("tailscale"):
        print_info(f"  • Tailscale detected: 'tailscale serve --https=443 {port}' gives trusted")
        print_info("    HTTPS on your tailnet without a domain (see the Hermes Desktop docs).")


# ── mode 2: custom domain (Caddy + Let's Encrypt) ────────────────────────────

def _prompt_hostname() -> str:
    from hermes_cli.setup import print_error, prompt
    from . import netops
    while True:
        name = prompt("  Hostname (e.g. desk.example.com)", default="").strip().lower()
        if not name:
            return ""
        if netops.valid_hostname(name) and not netops.is_local_name(name) and not name.replace(".", "").isdigit():
            return name
        print_error("Enter a real domain name (not .local, not an IP, no http://).")


def _verify_dns_loop(name: str, ip: str) -> bool:
    """Poll until the A record matches, the user skips, or cancels."""
    from hermes_cli.setup import print_error, print_info, print_success, print_warning, prompt_choice
    from . import netops
    for round_no in range(1, _DNS_MAX_ROUNDS + 1):
        ok, seen = netops.verify_a_record(name, ip)
        if ok:
            print_success(f"DNS verified: {name} -> {ip}")
            return True
        print_warning(f"{name} does not resolve to {ip} yet (round {round_no}/{_DNS_MAX_ROUNDS}).")
        for source, answers in seen.items():
            print_info(f"    {source}: {', '.join(answers)}")
        if not seen:
            print_info("    no A answers from any resolver yet")
        choice = prompt_choice("  What now?", [
            "Retry now", f"Wait {_DNS_WAIT_SECONDS}s and retry", "Skip verification (advanced)", "Cancel",
        ], 1)
        if choice == 0:
            continue
        if choice == 1:
            time.sleep(_DNS_WAIT_SECONDS)
            continue
        if choice == 2:
            print_warning("Skipping DNS verification — certificate issuance will fail until the record exists.")
            return True
        print_error("Cancelled.")
        return False
    print_error("DNS still not verified — create the A record, then run setup again.")
    return False


def _ensure_caddy() -> bool:
    from hermes_cli.setup import print_error, print_info, print_success, prompt_yes_no
    from . import caddy
    if caddy.caddy_installed():
        return True
    print_error("Caddy is not installed (it terminates HTTPS for your domain).")
    if os.geteuid() == 0 and shutil.which("apt-get"):
        if prompt_yes_no("  Install Caddy now (apt-get)?", True):
            result = subprocess.run(["apt-get", "install", "-y", "caddy"],
                                    capture_output=True, text=True, timeout=300, check=False)
            if result.returncode == 0 and caddy.caddy_installed():
                print_success("Caddy installed.")
                return True
            print_error("Caddy installation failed.")
    print_info("Install it, then run setup again:  sudo apt-get install -y caddy")
    return False


def _configure_caddy(name: str, port: int) -> bool:
    """Upsert the site block, validate, reload. Permission problems become
    instructions (the webapp config still lands; Caddy is the user's step)."""
    from hermes_cli.setup import print_error, print_info, print_success, print_warning
    from . import caddy
    caddyfile = caddy.DEFAULT_CADDYFILE
    if not caddyfile.exists():
        print_warning(f"No Caddyfile at {caddyfile} — create one containing:")
        _print_instructions(caddy.render_site_block(name, port).splitlines())
        return False
    try:
        changed, backup = caddy.upsert_site_block(caddyfile, name, port)
    except OSError:
        print_warning(f"Cannot write {caddyfile} (need root). Add this site block yourself:")
        _print_instructions(caddy.render_site_block(name, port).splitlines())
        print_info("Then:  sudo caddy validate --config /etc/caddy/Caddyfile && sudo systemctl reload caddy")
        return False
    if changed:
        print_success(f"Site block added for {name} (backup: {backup.name if backup else 'none'}).")
    else:
        print_info(f"Site block for {name} already present.")
    if not caddy.validate(caddyfile):
        if changed and backup:
            caddy.restore_backup(caddyfile, backup)
        print_error("caddy validate failed — the previous config was restored.")
        return False
    if caddy.reload_caddy():
        print_success("Caddy reloaded — certificate issuance starts now (ports 80/443 must be open).")
        return True
    print_warning("Could not reload Caddy. Run:  sudo systemctl reload caddy")
    return False


def _verify_https_gate(name: str) -> None:
    """Best-effort post-restart check: the gate should redirect / to /login."""
    from hermes_cli.setup import print_info, print_success, print_warning
    import urllib.error
    import urllib.request

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    url = f"https://{name}/"
    try:
        opener = urllib.request.build_opener(_NoRedirect)
        try:
            response = opener.open(url, timeout=15)
            code, location = response.status, ""
        except urllib.error.HTTPError as exc:
            code, location = exc.code, exc.headers.get("Location", "")
        if code in (301, 302, 303, 307, 308) and "login" in (location or ""):
            print_success(f"Auth gate verified: {url} -> {code} {location}")
        else:
            print_warning(f"{url} answered {code} — certificate may still be issuing; retry in a minute.")
    except Exception as exc:  # health: allow BLE001 -- post-setup check is advisory; any failure becomes a retry hint
        print_warning(f"Could not verify {url} yet ({exc.__class__.__name__}).")
        print_info("A VPS often cannot reach its own public IP; test from your own device.")


def _setup_domain() -> None:
    from hermes_cli.setup import (print_error, print_header, print_info, print_success,
                                  print_warning, prompt)
    from hermes_cli.webapp_access import WebappAccessError, plan, register, serve
    from . import caddy, netops

    print_header("Hermes Desktop — custom domain (HTTPS)")
    ip = netops.public_ip()
    if ip:
        print_info(f"This machine's public IP:  {ip}")
    else:
        print_warning("Could not detect the public IP automatically.")
        ip = prompt("  Enter this machine's public IP", default="").strip()
        if not ip:
            return
    print_info("Create a DNS A record at your domain provider:")
    print_info(f"    type: A    name: <your hostname>    value: {ip}")
    print_info("Ports 80 and 443 must reach this machine (cloud firewall + host firewall).")

    name = _prompt_hostname()
    if not name:
        return
    if not _verify_dns_loop(name, ip):
        return
    if not _ensure_caddy():
        return

    try:
        host, port, pinned, callback = plan("public", f"https://{name}")
    except WebappAccessError as exc:
        print_error(str(exc))
        return
    port = _effective_port(port)
    try:
        register(callback)
    except WebappAccessError as exc:
        print_error(str(exc))
        return
    print_success(f"Nous sign-in registered for {callback}")

    _configure_caddy(name, port)
    try:
        serve(host, pinned, port)
    except WebappAccessError as exc:
        print_error(str(exc))
        return

    if not _build_bundle():
        return
    _ensure_nous_only()
    restarted = _offer_restart()

    from hermes_cli.webapp_access import gateway_home
    from . import state
    state.save_state(gateway_home(), {
        "mode": "domain",
        "public_url": pinned,
        "caddy_blocks": _installed_blocks([caddy.render_site_block(name, port)]),
        "mdns_name": None,
        "avahi_backup": None,
    })

    if restarted:
        _verify_https_gate(name)

    print()
    print_success("🖥️  Hermes Desktop — domain access")
    print_info(f"  Open:  https://{name}")
    print_info("  Sign in with your Nous account.")
    if not restarted:
        print_info("  (available after the gateway restarts; the certificate issues on first request)")


# ── entry point ──────────────────────────────────────────────────────────────

def interactive_setup() -> None:
    from hermes_cli.setup import print_header, print_info, prompt_choice

    print_header("Hermes Desktop (browser)")
    print_info("The full Desktop workspace — sessions, files, terminal, cron — in a")
    print_info("browser, served by this gateway. Sign-in is your Nous account.")
    print_info("This is a browser surface, not a chat bot: nothing is enabled on")
    print_info("the messaging channels.")
    if not _preflight():
        return
    if _already_configured():
        return
    print()
    choice = prompt_choice("  How do you want to access Hermes Desktop?", [
        "Local network — open it as hermes-desktop.local (mDNS, same LAN)",
        "Custom domain — HTTPS at your own domain (Caddy + Let's Encrypt)",
        "Cancel",
    ], 0)
    if choice == 0:
        _switch_mode_cleanup("local")
        _setup_local()
    elif choice == 1:
        _switch_mode_cleanup("domain")
        _setup_domain()