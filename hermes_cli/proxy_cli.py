"""CLI handlers for ``hermes egress ...``."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agent.proxy_sources import iron_proxy as ip
from hermes_cli.config import load_config, load_env, save_config


def register_cli(parent_parser: argparse.ArgumentParser) -> None:
    """Attach the egress subcommand tree to a parent parser."""
    # dest='egress_command' keeps this tree disjoint from the inbound OAuth ``hermes proxy``
    # subparser (dest='proxy_command') so a grep-and-refactor on one never hits the other.
    sub = parent_parser.add_subparsers(dest="egress_command")
    # (name, help, handler, [(flag, add_argument kwargs), ...]) — declaration order is the
    # ``--help`` order, so keep it stable.
    commands = [
        ("install", f"Download iron-proxy binary (v{ip._IRON_PROXY_VERSION})", cmd_install, [
            ("--force", dict(action="store_true", help="Re-download even if a managed copy already exists")),
        ]),
        ("setup", "Interactive wizard: install + CA + mint tokens + write config", cmd_setup, [
            ("--tunnel-port", dict(type=int, default=None,
                                   help=f"Override the tunnel port (default {ip._DEFAULT_TUNNEL_PORT})")),
            ("--from-bitwarden", dict(action="store_true", help=(
                "Treat secrets as managed by Bitwarden — discover provider keys "
                "from secrets.bitwarden config instead of the current env.  Fails "
                "loudly if BW is unreachable rather than silently falling back."))),
            ("--no-bitwarden", dict(action="store_true", help=(
                "Explicitly switch credential_source back to env on re-setup "
                "(only meaningful when the previous setup used --from-bitwarden)."))),
            ("--rotate-tokens", dict(action="store_true", help=(
                "Mint fresh proxy tokens for every provider (default is to "
                "preserve tokens for providers that already had one — avoids "
                "401-ing already-running sandboxes on re-setup)."))),
            ("--restart", dict(dest="restart", action="store_true", default=None, help=(
                "If a daemon is already running, restart it automatically after "
                "writing the new config/tokens (non-interactive default on a tty is to ask)."))),
            ("--no-restart", dict(dest="restart", action="store_false", help=(
                "Do not restart a running daemon after setup; you'll need to run "
                "`hermes egress restart` yourself for changes to take effect."))),
        ]),
        ("start", "Start the managed iron-proxy", cmd_start, []),
        ("stop", "Stop the managed iron-proxy", cmd_stop, []),
        ("restart", "Restart the managed iron-proxy (stop if running, then start)", cmd_restart, []),
        ("reload", "Hot-reload the running daemon's ruleset from proxy.yaml "
                   "(management API — no restart, no dropped connections)", cmd_reload, []),
        ("status", "Show proxy state and mappings", cmd_status, [
            ("--show-tokens", dict(action="store_true", help=(
                "Print the proxy tokens (default: redacted prefix only). "
                "Beware: tokens may persist in your shell history."))),
        ]),
        ("disable", "Turn off the proxy integration", cmd_disable, []),
        ("config", "Print the generated proxy.yaml path", cmd_config, []),
    ]
    for name, help_text, func, arguments in commands:
        parser = sub.add_parser(name, help=help_text)
        for flag, kwargs in arguments:
            parser.add_argument(flag, **kwargs)
        parser.set_defaults(func=func)


def cmd_install(args: argparse.Namespace) -> int:
    console = Console()
    try:
        binary = ip.install_iron_proxy(force=bool(args.force))
    except Exception as exc:  # noqa: BLE001 — top-level user-facing error funnel
        console.print(f"[red]✗ install failed:[/red] {exc}")
        console.print("  Manual install: https://github.com/ironsh/iron-proxy/releases")
        return 1
    version = ip.iron_proxy_version(binary) or "(version unknown)"
    console.print(f"[green]✓[/green] installed {binary}  {version}")
    return 0


def cmd_setup(args: argparse.Namespace) -> int:
    """Four-step wizard; each ``_setup_*`` phase prints its own step and returns ``None`` to abort."""
    console = Console()
    console.print(Panel.fit(
        "[bold]iron-proxy setup[/bold]\n\n"
        "Routes outbound sandbox traffic through a local TLS-intercepting\n"
        "proxy so prompt-injected agents never see real provider API keys.\n\n"
        "[dim]Project: https://github.com/ironsh/iron-proxy  (Apache-2.0)[/dim]",
        border_style="cyan",
    ))
    if not _setup_install_binary(console):
        return 1
    ca = _setup_ca_cert(console)
    if ca is None:
        return 1
    mappings = _setup_mint_tokens(console, args)
    if mappings is None:
        return 1
    proxy_cfg = _setup_write_config(console, args, mappings, *ca)
    if proxy_cfg is None:
        return 1
    _setup_restart_daemon(console, args, proxy_cfg)
    console.print()
    console.print("[green]✓ iron-proxy is configured.[/green]  Sandboxes will route outbound traffic through it.")
    console.print(
        "  Start:   [cyan]hermes egress start[/cyan]\n"
        "  Restart: [cyan]hermes egress restart[/cyan]  (after any re-setup)\n"
        "  Reload:  [cyan]hermes egress reload[/cyan]   (apply ruleset edits "
        "in-place, no restart)\n"
        "  Status:  [cyan]hermes egress status[/cyan]\n"
        "  Stop:    [cyan]hermes egress stop[/cyan]\n"
        "  Disable: [cyan]hermes egress disable[/cyan]"
    )
    return 0


def _setup_install_binary(console: Console) -> bool:
    _step(console, 1, "Install the iron-proxy binary")
    try:
        binary = ip.find_iron_proxy(install_if_missing=False)
        if binary is None:
            console.print("  No iron-proxy on PATH — downloading…")
            binary = ip.install_iron_proxy()
        version = ip.iron_proxy_version(binary) or "(version unknown)"
        console.print(f"  [green]✓[/green] {binary}  {version}")
    except Exception as exc:  # noqa: BLE001
        console.print(f"  [red]✗ install failed: {exc}[/red]")
        return False
    return True


def _setup_ca_cert(console: Console):
    _step(console, 2, "Generate a CA cert")
    try:
        ca_crt, ca_key = ip.ensure_ca_cert()
    except Exception as exc:  # noqa: BLE001
        console.print(f"  [red]✗ CA generation failed: {exc}[/red]")
        return None
    console.print(f"  [green]✓[/green] {ca_crt}")
    return ca_crt, ca_key


def _setup_mint_tokens(console: Console, args: argparse.Namespace):
    """Discover providers, merge with existing tokens (rotating on request), print the table."""
    _step(console, 3, "Mint proxy tokens for known providers")
    available_env_names: List[str] = []
    if args.from_bitwarden:
        available_env_names = _bitwarden_env_names(console)
        if available_env_names is None:
            return None
    else:
        # Operators commonly keep provider keys only in ~/.hermes/.env (loaded when the agent
        # runs, NOT exported into an interactive shell); backfill so discovery sees them.
        loaded = _load_env_file_into_environ()
        if loaded:
            console.print(f"  [dim]Loaded {loaded} provider key name(s) from ~/.hermes/.env for discovery.[/dim]")
    discovered = ip.discover_provider_mappings(available_env_names=available_env_names or None)
    # Preserve existing tokens unless rotation was requested — re-running setup must not
    # invalidate tokens baked into already-running sandboxes.
    existing = ip.load_mappings()
    rotate = bool(getattr(args, "rotate_tokens", False))
    if rotate and existing:
        # Rotation is unrecoverable for running sandboxes, so gate it on an explicit confirmation
        # when stdin is a tty (non-interactive callers passed the flag deliberately).
        if sys.stdin.isatty():
            console.print(
                "[yellow]⚠[/yellow]  --rotate-tokens will invalidate proxy "
                "tokens in every running Hermes sandbox.  They will start 401-ing against upstreams until restarted."
            )
            if _prompt("Type 'rotate' to confirm: ") != "rotate":
                console.print("[yellow]Cancelled.[/yellow]")
                return None
        # Plain-JSON ``.rotated-<ts>`` backup lets the operator recover tokens by hand.
        try:
            state_dir = ip._proxy_state_dir()
            mappings_src = state_dir / "mappings.json"
            if mappings_src.exists():
                ts = datetime.now().strftime("%Y%m%dT%H%M%S")
                backup = state_dir / f"mappings.json.rotated-{ts}"
                shutil.copy2(str(mappings_src), str(backup))
                console.print(f"  [dim]backup: {backup}[/dim]")
        except OSError as exc:
            console.print(f"  [yellow]Could not back up mappings before rotation: {exc}[/yellow]")
    elif rotate and not existing:
        console.print(
            "[dim]Note: --rotate-tokens is a no-op on first-time setup (no existing tokens to rotate).[/dim]"
        )
    mappings = ip.merge_mappings(existing=existing, discovered=discovered, rotate=rotate)
    if not mappings:
        console.print("  [yellow]No known provider API keys found in env/Bitwarden.[/yellow]")
        console.print("  Set at least one of these and rerun setup:")
        for env_name in sorted(ip._BEARER_PROVIDERS):
            console.print(f"    - {env_name}")
        return None
    # Providers we recognise but can't proxy (SigV4, service-account OAuth) still work — they
    # just bypass the egress isolation, so say so.
    uncovered = ip.discover_uncovered_providers(available_env_names=available_env_names or None)
    if uncovered:
        console.print()
        console.print("  [yellow]⚠[/yellow]  Detected provider env vars that the proxy does not yet cover:")
        for name in uncovered:
            console.print(f"    - {name}")
        console.print(
            "  [dim]These providers use request signing or SDK-minted "
            "OAuth (SigV4, service-account files) and will hold real "
            "credentials inside the sandbox.  Egress isolation is INCOMPLETE for these.[/dim]"
        )
    console.print(_mappings_table(mappings, "Provider env", "Upstream hosts", show_tokens=False))
    return mappings


def _setup_write_config(console: Console, args: argparse.Namespace, mappings, ca_crt, ca_key):
    """Write proxy.yaml + mappings, then enable the integration in config; returns ``proxy_cfg``."""
    _step(console, 4, "Write config and persist mappings")
    cfg = load_config()
    proxy_cfg = cfg.setdefault("proxy", {})
    # None = flag not given. ``0`` is not a valid TCP listener, so it is a hard error rather
    # than a silent fallback to the default.
    if args.tunnel_port is not None:
        if args.tunnel_port < 1 or args.tunnel_port > 65534:
            console.print(
                "  [red]✗ --tunnel-port must be between 1 and 65534 (the plain-HTTP listener uses port+1).[/red]"
            )
            return None
        tunnel_port = int(args.tunnel_port)
    else:
        tunnel_port = int(proxy_cfg.get("tunnel_port", ip._DEFAULT_TUNNEL_PORT))
    proxy_cfg["tunnel_port"] = tunnel_port
    extra_hosts = list(proxy_cfg.get("extra_allowed_hosts") or [])
    allowed = list(ip._DEFAULT_ALLOWED_HOSTS) + [h for h in extra_hosts if h not in ip._DEFAULT_ALLOWED_HOSTS]
    # Pre-create the audit log 0o600. The pinned v0.39 daemon never writes it (reserved for
    # v0.40+ per-request records), so a pre-create failure is a WARNING, not a setup abort.
    audit_log_path = ip._proxy_state_dir() / "audit.log"
    audit_log_ok = True
    try:
        ip.ensure_audit_log(audit_log_path)
    except RuntimeError as exc:
        audit_log_ok = False
        console.print(f"  [yellow]⚠ {exc}[/yellow]")
    # ``proxy.upstream_deny_cidrs`` overrides the deny list; None yields the documented safe
    # default-deny set (loopback, IMDS, RFC1918).
    iron_cfg = ip.build_proxy_config(
        mappings=mappings,
        ca_cert=ca_crt,
        ca_key=ca_key,
        tunnel_port=tunnel_port,
        audit_log=audit_log_path,
        allowed_hosts=allowed,
        upstream_deny_cidrs=proxy_cfg.get("upstream_deny_cidrs"),
    )
    cfg_path = ip.write_proxy_config(iron_cfg)
    mappings_path = ip.write_mappings(mappings)
    # The generated config enables a loopback management listener (used by ``egress reload``);
    # the daemon requires its bearer key env var to be non-empty at startup, so mint it now.
    ip.ensure_management_token()
    console.print(f"  [green]✓[/green] config:   {cfg_path}")
    console.print(f"  [green]✓[/green] mappings: {mappings_path}")
    if audit_log_ok:
        console.print(
            f"  [green]✓[/green] audit log: {audit_log_path} "
            f"[dim](reserved — not written by iron-proxy v0.39; "
            f"per-request records land in iron-proxy.log)[/dim]"
        )
    proxy_cfg["enabled"] = True
    proxy_cfg.setdefault("auto_install", True)
    proxy_cfg.setdefault("enforce_on_docker", True)
    # CRITICAL: never silently downgrade credential_source on re-run. A previous
    # ``--from-bitwarden`` setup keeps bitwarden mode (the documented rotation guarantee)
    # unless the operator passes an explicit --no-bitwarden.
    existing_source = proxy_cfg.get("credential_source")
    if args.from_bitwarden:
        proxy_cfg["credential_source"] = "bitwarden"
    elif getattr(args, "no_bitwarden", False):
        proxy_cfg["credential_source"] = "env"
        if existing_source == "bitwarden":
            console.print("[yellow]Switched credential_source from bitwarden to env.[/yellow]")
    elif existing_source == "bitwarden":
        console.print(
            "[dim]Keeping credential_source=bitwarden from existing config. "
            "Pass --no-bitwarden to switch to env-based credentials.[/dim]"
        )
    else:
        proxy_cfg["credential_source"] = "env"
    save_config(cfg)
    return proxy_cfg


def _setup_restart_daemon(console: Console, args: argparse.Namespace, proxy_cfg: dict) -> None:
    """Stop a running daemon and decide whether to (re)start it with the new config.

    --restart → always (re)start; --no-restart → never (print the manual hint); neither + tty →
    ask only when a daemon was running; neither + !tty → restart iff one was running (first-time
    setup never auto-starts — matches the "configured, now run start" flow).
    """
    was_running = ip.get_status().pid is not None
    if was_running:
        ip.stop_proxy()
    restart_pref = getattr(args, "restart", None)
    if restart_pref is True or restart_pref is False:
        do_restart = restart_pref
    elif was_running:
        do_restart = (
            _prompt("  Restart the running proxy now with the new config? [Y/n] ")
            in ("", "y", "yes")
            if sys.stdin.isatty()
            else True
        )
    else:
        do_restart = False
    if do_restart:
        try:
            new_status = ip.start_proxy(install_if_missing=bool(proxy_cfg.get("auto_install", True)))
        except Exception as exc:  # noqa: BLE001 — user-facing funnel
            console.print(f"  [yellow]⚠ could not start iron-proxy with the new config: {exc}[/yellow]")
            console.print("  Run [cyan]hermes egress start[/cyan] manually before launching new Docker sandboxes.")
        else:
            listening = "listening" if new_status.listening else "not yet listening"
            verb = "restarted" if was_running else "started"
            console.print(
                f"  [green]✓[/green] {verb} iron-proxy with the new config "
                f"(pid={new_status.pid}, port={new_status.tunnel_port}, {listening})"
            )
    elif was_running:
        console.print(
            "  [yellow]⚠ stopped the running iron-proxy; config or tokens "
            "changed.  Run [cyan]hermes egress restart[/cyan] (or "
            "[cyan]start[/cyan]) before launching new Docker sandboxes.[/yellow]"
        )


def cmd_start(args: argparse.Namespace) -> int:
    console = Console()
    cfg = load_config()
    proxy_cfg = cfg.get("proxy") or {}
    if not proxy_cfg.get("enabled"):
        console.print("[yellow]proxy.enabled is false — run `hermes egress setup` first.[/yellow]")
        return 1
    # ``credential_source: bitwarden`` refreshes upstream secrets from BSM at startup — that is
    # the rotation guarantee distinguishing it from ``env``.
    credential_source = proxy_cfg.get("credential_source", "env")
    bw_cfg = (cfg.get("secrets") or {}).get("bitwarden")
    refresh_bw = (credential_source == "bitwarden" and bw_cfg is not None and bool(bw_cfg.get("enabled")))
    # Silent-degrade guard: bitwarden mode chosen but secrets.bitwarden disabled/removed. Refuse
    # (quietly starting on host env is the bug class BW mode exists to defeat) unless the
    # documented escape hatch is set.
    if credential_source == "bitwarden" and not refresh_bw:
        if bool(proxy_cfg.get("allow_env_fallback", False)):
            console.print(
                "[yellow]⚠ credential_source=bitwarden but secrets.bitwarden is disabled or missing — falling back "
                "to host-env secrets (allow_env_fallback=true).  Rotated Bitwarden keys will NOT propagate.[/yellow]"
            )
        else:
            return _refuse(
                console,
                "proxy.credential_source is 'bitwarden' but secrets.bitwarden is disabled or missing.",
                "Re-enable it (`secrets.bitwarden.enabled: true`), switch "
                "back to env credentials with `hermes egress setup "
                "--no-bitwarden`, or set `proxy.allow_env_fallback: true` to opt into the host-env fallback.",
            )
    # Pass the allow_env_fallback opt-in through to start_proxy: when set the daemon falls back
    # to host env if BWS is unreachable instead of raising. Default is strict (raise).
    if refresh_bw and bw_cfg is not None:
        bw_cfg = dict(bw_cfg)
        bw_cfg["allow_env_fallback"] = bool(proxy_cfg.get("allow_env_fallback", False))
    # (fail_on_uncovered_providers was removed: the fail-closed provider tier is now empty.)
    # Pre-check the BWS token + project here so bitwarden mode fails loud with actionable
    # messages BEFORE start_proxy could silently degrade to a stale/mismatched host env.
    if refresh_bw:
        bw_access_env = (bw_cfg or {}).get("access_token_env", "BWS_ACCESS_TOKEN")
        if not os.environ.get(bw_access_env, "").strip():
            return _refuse(
                console,
                f"credential_source=bitwarden but {bw_access_env} is not set in the environment.",
                "Either export the access token, or run "
                "`hermes egress setup --no-bitwarden` to switch back to env-based credentials.",
            )
        if not (bw_cfg or {}).get("project_id"):
            return _refuse(
                console,
                "credential_source=bitwarden but secrets.bitwarden.project_id is empty.",
                "Run `hermes secrets bitwarden setup` to configure the "
                "project, or switch back via `hermes egress setup --no-bitwarden`.",
            )
    try:
        status = ip.start_proxy(
            install_if_missing=bool(proxy_cfg.get("auto_install", True)),
            refresh_secrets_from_bitwarden=refresh_bw,
            bitwarden_config=bw_cfg,
        )
    except Exception as exc:  # noqa: BLE001 — top-level user-facing funnel
        console.print(f"[red]✗ failed to start iron-proxy:[/red] {exc}")
        return 1
    if not status.pid:
        console.print("[red]✗ iron-proxy did not come up cleanly[/red]")
        return 1
    listening = ("[green]listening[/green]" if status.listening else "[yellow]not yet listening[/yellow]")
    console.print(
        f"[green]✓[/green] iron-proxy running  pid={status.pid}  port={status.tunnel_port}  {listening}"
    )
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    console = Console()
    if ip.stop_proxy():
        console.print("[green]✓[/green] iron-proxy stopped")
    else:
        console.print("[dim]iron-proxy was not running[/dim]")
    return 0


def cmd_restart(args: argparse.Namespace) -> int:
    """Stop the daemon (if any) then delegate to ``cmd_start`` so every credential-source guard runs."""
    console = Console()
    if ip.stop_proxy():
        console.print("[dim]stopped the running iron-proxy[/dim]")
    return cmd_start(args)


def cmd_reload(args: argparse.Namespace) -> int:
    """Hot-reload the ruleset via the management API (no restart, no dropped connections).

    New upstream SECRETS still need ``hermes egress restart``: the daemon reads credentials from
    its own environment at spawn time and a reload does not re-populate that env.
    """
    console = Console()
    try:
        ip.reload_proxy()
    except Exception as exc:  # noqa: BLE001 — top-level user-facing funnel
        console.print(f"[red]✗ reload failed:[/red] {exc}")
        return 1
    console.print("[green]✓[/green] iron-proxy ruleset reloaded in-place (no restart, connections preserved)")
    console.print(
        "[dim]Note: new upstream secrets (rotated keys, new providers) "
        "still need `hermes egress restart` — the daemon reads real "
        "credentials from its environment at spawn time.[/dim]"
    )
    return 0


def format_status_text(*, show_tokens: bool = False) -> str:
    """Plain-text egress status for slash commands, Dashboard, and Desktop."""
    cfg = load_config()
    proxy_cfg = cfg.get("proxy") or {}
    status = ip.get_status()
    lines = ["Egress proxy status", ""]
    lines.extend(
        f"{label}: {value}"
        for label, value in _status_rows(proxy_cfg, status, yn=lambda v: "yes" if v else "no", dim=lambda t: t)
    )
    lines.append("Scope: Docker backend only in this release")
    mappings = ip.load_mappings()
    if mappings:
        lines.extend(["", "Token mappings:"])
        for m in mappings:
            tok = m.proxy_token if show_tokens else _redact_token(m.proxy_token)
            lines.append(f"  - {m.real_env_name}: {tok} ({', '.join(m.upstream_hosts)})")
    uncovered = ip.discover_uncovered_providers()
    if uncovered:
        lines.extend(["", "Uncovered providers (real credentials still visible inside the sandbox):"])
        for name in uncovered:
            lines.append(f"  - {name}")
    if bool(proxy_cfg.get("enabled")) and not status.configured:
        lines.extend(["", "Next: run `hermes egress setup` to mint tokens and write proxy.yaml."])
    elif bool(proxy_cfg.get("enabled")) and not (status.pid and status.listening):
        lines.extend(["", "Next: run `hermes egress start` before launching Docker sandboxes."])
    return "\n".join(lines)


def cmd_status(args: argparse.Namespace) -> int:
    console = Console()
    cfg = load_config()
    proxy_cfg = cfg.get("proxy") or {}
    status = ip.get_status()
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("", style="bold")
    table.add_column("")
    for label, value in _status_rows(proxy_cfg, status, yn=_yn, dim=lambda t: f"[dim]{t}[/dim]"):
        table.add_row(label, value)
    console.print(table)
    mappings = ip.load_mappings()
    if mappings:
        console.print()
        console.print("[bold]Token mappings[/bold]")
        console.print(_mappings_table(mappings, "Real env", "Upstream", show_tokens=args.show_tokens))
        if args.show_tokens:
            console.print(
                "[yellow]⚠[/yellow]  proxy tokens just printed in full — "
                "they may persist in your shell history.  Consider clearing it after this command."
            )
    # Uncovered providers = the isolation boundary is incomplete for those upstreams.
    uncovered = ip.discover_uncovered_providers()
    if uncovered:
        console.print()
        console.print("[yellow]Uncovered providers[/yellow] (real credentials still visible inside the sandbox):")
        for name in uncovered:
            console.print(f"  - {name}")
    return 0


def cmd_disable(args: argparse.Namespace) -> int:
    console = Console()
    cfg = load_config()
    proxy_cfg = cfg.setdefault("proxy", {})
    if not proxy_cfg.get("enabled"):
        console.print("[dim]proxy.enabled was already false.[/dim]")
        return 0
    proxy_cfg["enabled"] = False
    save_config(cfg)
    console.print("[green]✓[/green] proxy.enabled set to false")
    # get_status().pid already applies the liveness check; ip._read_pid() alone would fire
    # spuriously on a stale pidfile from a crashed run.
    if ip.get_status().pid is not None:
        console.print(
            "  iron-proxy is still running — stop it with [cyan]hermes egress stop[/cyan] if you want it down too."
        )
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    console = Console()
    status = ip.get_status()
    if status.config_path is None:
        console.print("[yellow](no config generated — run `hermes egress setup`)[/yellow]")
        return 1
    console.print(str(status.config_path))
    return 0


def _bitwarden_env_names(console: Console) -> Optional[List[str]]:
    """Secret names from Bitwarden for ``setup --from-bitwarden``; prints the error and returns
    ``None`` on any failure so the wizard aborts loudly instead of falling back to the host env.
    """
    cfg = load_config()
    bw_cfg = (cfg.get("secrets") or {}).get("bitwarden") or {}
    if not bw_cfg.get("enabled"):
        console.print("  [red]✗ --from-bitwarden requested but secrets.bitwarden.enabled is false.[/red]")
        console.print("  Run `hermes secrets bitwarden setup` first, or omit --from-bitwarden.")
        return None
    try:
        from agent.secret_sources import bitwarden as bw
        token_env = bw_cfg.get("access_token_env", "BWS_ACCESS_TOKEN")
        access_token = os.environ.get(token_env, "").strip()
        if not access_token:
            console.print(f"  [red]✗ --from-bitwarden requested but {token_env} is not set in the environment.[/red]")
            return None
        secrets, _ = bw.fetch_bitwarden_secrets(
            access_token=access_token, project_id=bw_cfg.get("project_id", ""), cache_ttl_seconds=0, use_cache=False
        )
        names = list(secrets.keys())
        if not names:
            console.print(
                "  [red]✗ Bitwarden returned an empty secrets list.[/red]\n"
                "  Check the project_id in secrets.bitwarden and the BWS access-token's project scope."
            )
            return None
        console.print(f"  Pulled {len(names)} env names from Bitwarden.")
        return names
    except Exception as exc:  # noqa: BLE001 — explicit user-facing error
        console.print(f"  [red]✗ Could not enumerate Bitwarden secrets: {exc}[/red]")
        console.print(
            "  Either fix the Bitwarden config and retry, or rerun setup "
            "without --from-bitwarden (the proxy will read secrets from the host process env at start time)."
        )
        return None


def _load_env_file_into_environ() -> int:
    """Backfill known provider keys from ``~/.hermes/.env`` into ``os.environ``; returns the count.
    Never overrides an exported value; only known provider names, so unrelated secrets stay out."""
    try:
        file_env = load_env()
    except Exception:  # noqa: BLE001 — best-effort convenience, never fatal
        return 0
    added = 0
    known = set(ip._BEARER_PROVIDERS) | set(ip._NON_BEARER_PROVIDERS)
    for name in known:
        if name in os.environ and os.environ[name].strip():
            continue
        val = (file_env.get(name) or "").strip()
        if val:
            os.environ[name] = val
            added += 1
    return added


def _mappings_table(mappings, env_header: str, hosts_header: str, *, show_tokens: bool) -> Table:
    table = Table(show_header=True, header_style="bold")
    table.add_column(env_header, style="cyan")
    table.add_column(hosts_header, style="dim")
    table.add_column("Proxy token", style="green")
    for m in mappings:
        tok = m.proxy_token if show_tokens else _redact_token(m.proxy_token)
        table.add_row(m.real_env_name, ", ".join(m.upstream_hosts), tok)
    return table


def _step(console: Console, n: int, title: str) -> None:
    console.print()
    console.print(f"[bold]Step {n}[/bold]  {title}")


def _refuse(console: Console, reason: str, hint: str) -> int:
    """Print a ``✗ Refusing to start`` headline plus an indented remedy; returns exit code 1."""
    console.print(f"[red]✗ Refusing to start: {reason}[/red]")
    console.print(f"  {hint}")
    return 1


def _yn(value: bool) -> str:
    return "[green]yes[/green]" if value else "[dim]no[/dim]"


def _prompt(text: str) -> str:
    """``input()`` lowered+stripped; EOF (closed stdin) reads as an empty answer."""
    try:
        return input(text).strip().lower()
    except EOFError:
        return ""


def _status_rows(proxy_cfg: dict, status, *, yn, dim) -> list[tuple[str, str]]:
    """``(label, value)`` rows shared by the rich ``status`` table and the plain-text variant;
    ``yn`` renders booleans, ``dim`` wraps placeholder text for missing values."""
    return [
        ("Enabled", yn(bool(proxy_cfg.get("enabled")))),
        ("Binary", str(status.binary_path or dim("(missing)"))),
        ("Binary version", status.binary_version or dim("(unknown)")),
        ("Config", str(status.config_path or dim("(not generated)"))),
        ("CA cert", str(status.ca_cert_path or dim("(not generated)"))),
        ("Tunnel port", str(status.tunnel_port)),
        ("Process", f"pid {status.pid}" if status.pid else dim("(stopped)")),
        ("Listening", yn(status.listening)),
        ("Credential src", str(proxy_cfg.get("credential_source", "env"))),
        ("Docker enforce", yn(bool(proxy_cfg.get("enforce_on_docker", True)))),
    ]


def _redact_token(token: str) -> str:
    if len(token) < 16:
        return token
    return f"{token[:12]}…{token[-4:]}"
