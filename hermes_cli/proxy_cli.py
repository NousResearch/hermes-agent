"""CLI handlers for ``hermes egress ...``.

Subcommands:
    install  — download the pinned iron-proxy binary
    setup    — interactive wizard: install binary, generate CA, mint tokens, write config
    start    — launch the proxy as a managed subprocess
    stop     — terminate the managed proxy
    status   — show binary version + config presence + listen state + mappings
    disable  — flip ``proxy.enabled`` to False (does not stop a running proxy)
    config   — print the generated proxy.yaml path (for debugging / external review)

The top-level command is ``hermes egress``.  Note that the inbound OAuth
reverse-proxy command (``hermes proxy``) lives elsewhere in
``hermes_cli/main.py`` — different direction, different purpose.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agent.proxy_sources import iron_proxy as ip
from hermes_cli.config import load_config, save_config


# ---------------------------------------------------------------------------
# Argparse wiring — called from hermes_cli.main
# ---------------------------------------------------------------------------


def register_cli(parent_parser: argparse.ArgumentParser) -> None:
    """Attach the egress subcommand tree to a parent parser.

    Called from ``hermes_cli.main`` as part of building the top-level
    ``hermes egress`` parser.
    """

    # dest='egress_command' — keeps this subparser tree disjoint from the
    # inbound OAuth ``hermes proxy`` subparser (which uses dest='proxy_command').
    # No runtime collision today since they live in separate parser trees,
    # but a future grep-and-refactor on ``proxy_command`` would otherwise
    # hit both handlers.
    sub = parent_parser.add_subparsers(dest="egress_command")

    install = sub.add_parser(
        "install",
        help=f"Download iron-proxy binary (v{ip._IRON_PROXY_VERSION})",
    )
    install.add_argument(
        "--force", action="store_true",
        help="Re-download even if a managed copy already exists",
    )
    install.set_defaults(func=cmd_install)

    setup = sub.add_parser(
        "setup",
        help="Interactive wizard: install + CA + mint tokens + write config",
    )
    setup.add_argument(
        "--tunnel-port", type=int, default=None,
        help=f"Override the tunnel port (default {ip._DEFAULT_TUNNEL_PORT})",
    )
    setup.add_argument(
        "--from-bitwarden", action="store_true",
        help="Treat secrets as managed by Bitwarden — discover provider keys "
             "from secrets.bitwarden config instead of the current env.  Fails "
             "loudly if BW is unreachable rather than silently falling back.",
    )
    setup.add_argument(
        "--no-bitwarden", action="store_true",
        help="Explicitly switch credential_source back to env on re-setup "
             "(only meaningful when the previous setup used --from-bitwarden).",
    )
    setup.add_argument(
        "--rotate-tokens", action="store_true",
        help="Mint fresh proxy tokens for every provider (default is to "
             "preserve tokens for providers that already had one — avoids "
             "401-ing already-running sandboxes on re-setup).",
    )
    setup.add_argument(
        "--restart", dest="restart", action="store_true", default=None,
        help="If a daemon is already running, restart it automatically after "
             "writing the new config/tokens (non-interactive default on a tty "
             "is to ask).",
    )
    setup.add_argument(
        "--no-restart", dest="restart", action="store_false",
        help="Do not restart a running daemon after setup; you'll need to run "
             "`hermes egress restart` yourself for changes to take effect.",
    )
    setup.set_defaults(func=cmd_setup)

    start = sub.add_parser("start", help="Start the managed iron-proxy")
    start.set_defaults(func=cmd_start)

    stop = sub.add_parser("stop", help="Stop the managed iron-proxy")
    stop.set_defaults(func=cmd_stop)

    restart = sub.add_parser(
        "restart",
        help="Restart the managed iron-proxy (stop if running, then start)",
    )
    restart.set_defaults(func=cmd_restart)

    status = sub.add_parser("status", help="Show proxy state and mappings")
    status.add_argument(
        "--show-tokens", action="store_true",
        help="Print the proxy tokens (default: redacted prefix only). "
             "Beware: tokens may persist in your shell history.",
    )
    status.set_defaults(func=cmd_status)

    disable = sub.add_parser("disable", help="Turn off the proxy integration")
    disable.set_defaults(func=cmd_disable)

    cfg = sub.add_parser("config", help="Print the generated proxy.yaml path")
    cfg.set_defaults(func=cmd_config)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def cmd_install(args: argparse.Namespace) -> int:
    console = Console()
    try:
        binary = ip.install_iron_proxy(force=bool(args.force))
    except Exception as exc:  # noqa: BLE001 — top-level user-facing error funnel
        console.print(f"[red]✗ install failed:[/red] {exc}")
        console.print(
            "  Manual install: https://github.com/ironsh/iron-proxy/releases"
        )
        return 1
    version = ip.iron_proxy_version(binary) or "(version unknown)"
    console.print(f"[green]✓[/green] installed {binary}  {version}")
    return 0


def cmd_setup(args: argparse.Namespace) -> int:
    console = Console()
    console.print(Panel.fit(
        "[bold]iron-proxy setup[/bold]\n\n"
        "Routes outbound sandbox traffic through a local TLS-intercepting\n"
        "proxy so prompt-injected agents never see real provider API keys.\n\n"
        "[dim]Project: https://github.com/ironsh/iron-proxy  (Apache-2.0)[/dim]",
        border_style="cyan",
    ))

    # ------------------------------------------------------------------ binary
    console.print()
    console.print("[bold]Step 1[/bold]  Install the iron-proxy binary")
    try:
        binary = ip.find_iron_proxy(install_if_missing=False)
        if binary is None:
            console.print("  No iron-proxy on PATH — downloading…")
            binary = ip.install_iron_proxy()
        version = ip.iron_proxy_version(binary) or "(version unknown)"
        console.print(f"  [green]✓[/green] {binary}  {version}")
    except Exception as exc:  # noqa: BLE001
        console.print(f"  [red]✗ install failed: {exc}[/red]")
        return 1

    # ------------------------------------------------------------------ CA
    console.print()
    console.print("[bold]Step 2[/bold]  Generate a CA cert")
    try:
        ca_crt, ca_key = ip.ensure_ca_cert()
    except Exception as exc:  # noqa: BLE001
        console.print(f"  [red]✗ CA generation failed: {exc}[/red]")
        return 1
    console.print(f"  [green]✓[/green] {ca_crt}")

    # ------------------------------------------------------------------ mint
    console.print()
    console.print("[bold]Step 3[/bold]  Mint proxy tokens for known providers")

    available_env_names: List[str] = []
    if args.from_bitwarden:
        cfg = load_config()
        bw_cfg = (cfg.get("secrets") or {}).get("bitwarden") or {}
        if not bw_cfg.get("enabled"):
            console.print(
                "  [red]✗ --from-bitwarden requested but "
                "secrets.bitwarden.enabled is false.[/red]"
            )
            console.print(
                "  Run `hermes secrets bitwarden setup` first, or omit "
                "--from-bitwarden."
            )
            return 1
        try:
            from agent.secret_sources import bitwarden as bw
            access_token = os.environ.get(
                bw_cfg.get("access_token_env", "BWS_ACCESS_TOKEN"), ""
            ).strip()
            if not access_token:
                console.print(
                    f"  [red]✗ --from-bitwarden requested but "
                    f"{bw_cfg.get('access_token_env', 'BWS_ACCESS_TOKEN')} "
                    "is not set in the environment.[/red]"
                )
                return 1
            secrets, _ = bw.fetch_bitwarden_secrets(
                access_token=access_token,
                project_id=bw_cfg.get("project_id", ""),
                cache_ttl_seconds=0,
                use_cache=False,
            )
            available_env_names = list(secrets.keys())
            if not available_env_names:
                console.print(
                    "  [red]✗ Bitwarden returned an empty secrets list.[/red]\n"
                    "  Check the project_id in secrets.bitwarden and the "
                    "BWS access-token's project scope."
                )
                return 1
            console.print(
                f"  Pulled {len(available_env_names)} env names from Bitwarden."
            )
        except Exception as exc:  # noqa: BLE001 — explicit user-facing error
            console.print(
                f"  [red]✗ Could not enumerate Bitwarden secrets: {exc}[/red]"
            )
            console.print(
                "  Either fix the Bitwarden config and retry, or rerun setup "
                "without --from-bitwarden (the proxy will read secrets from "
                "the host process env at start time)."
            )
            return 1
    else:
        # Env-based discovery reads os.environ.  Operators commonly keep their
        # provider keys only in ~/.hermes/.env (loaded automatically when the
        # agent runs, but NOT exported into an interactive shell).  Fall back
        # to loading that file so `hermes egress setup` finds the same keys the
        # agent would — otherwise a user with keys solely in .env sees a
        # confusing "no provider keys found" when the keys clearly "exist".
        loaded = _load_env_file_into_environ()
        if loaded:
            console.print(
                f"  [dim]Loaded {loaded} provider key name(s) from "
                f"~/.hermes/.env for discovery.[/dim]"
            )

    discovered = ip.discover_provider_mappings(
        available_env_names=available_env_names or None,
    )

    # Preserve tokens for providers we already had unless the operator
    # explicitly requested rotation.  This prevents re-running `hermes
    # egress setup` from invalidating tokens baked into already-running
    # sandboxes.
    existing = ip.load_mappings()
    rotate = bool(getattr(args, "rotate_tokens", False))

    # P3 confirmation gate: --rotate-tokens invalidates every running
    # sandbox's proxy tokens immediately.  An accidental re-run (history
    # scroll-back, tmux paste) is unrecoverable, so require explicit
    # confirmation when there's something to actually rotate.  Skipped
    # when stdin isn't a tty (CI / non-interactive use), in which case
    # the operator passed the flag deliberately.
    if rotate and existing:
        import sys as _sys
        from datetime import datetime as _dt
        if _sys.stdin.isatty():
            console.print(
                "[yellow]⚠[/yellow]  --rotate-tokens will invalidate proxy "
                "tokens in every running Hermes sandbox.  They will start "
                "401-ing against upstreams until restarted."
            )
            try:
                ans = input("Type 'rotate' to confirm: ").strip().lower()
            except EOFError:
                ans = ""
            if ans != "rotate":
                console.print("[yellow]Cancelled.[/yellow]")
                return 1
        # Backup the existing mappings before we overwrite.  The
        # resulting ``.rotated-<unix>`` sibling is plain JSON and lets
        # the operator manually recover tokens if they realise the
        # rotation was a mistake.
        try:
            import shutil as _shutil
            state_dir = ip._proxy_state_dir()
            mappings_src = state_dir / "mappings.json"
            if mappings_src.exists():
                ts = _dt.now().strftime("%Y%m%dT%H%M%S")
                backup = state_dir / f"mappings.json.rotated-{ts}"
                _shutil.copy2(str(mappings_src), str(backup))
                console.print(f"  [dim]backup: {backup}[/dim]")
        except OSError as exc:
            console.print(
                f"  [yellow]Could not back up mappings before rotation: "
                f"{exc}[/yellow]"
            )
    elif rotate and not existing:
        console.print(
            "[dim]Note: --rotate-tokens is a no-op on first-time setup "
            "(no existing tokens to rotate).[/dim]"
        )

    mappings = ip.merge_mappings(
        existing=existing,
        discovered=discovered,
        rotate=rotate,
    )

    if not mappings:
        console.print(
            "  [yellow]No known provider API keys found in env/Bitwarden.[/yellow]"
        )
        console.print(
            "  Set at least one of these and rerun setup:"
        )
        for env_name in sorted(ip._BEARER_PROVIDERS):
            console.print(f"    - {env_name}")
        return 1

    # Warn the operator about providers we recognize but can't proxy
    # (Anthropic native, AWS Bedrock, Azure OpenAI, etc).  These still
    # work — they just bypass the egress isolation.
    uncovered = ip.discover_uncovered_providers(
        available_env_names=available_env_names or None,
    )
    if uncovered:
        console.print()
        console.print(
            "  [yellow]⚠[/yellow]  Detected provider env vars that the "
            "proxy does not yet cover:"
        )
        for name in uncovered:
            console.print(f"    - {name}")
        console.print(
            "  [dim]These providers use non-bearer auth (x-api-key, "
            "SigV4, etc.) and will hold real credentials inside the "
            "sandbox.  Egress isolation is INCOMPLETE for these.[/dim]"
        )

    table = Table(show_header=True, header_style="bold")
    table.add_column("Provider env", style="cyan")
    table.add_column("Upstream hosts", style="dim")
    table.add_column("Proxy token", style="green")
    for m in mappings:
        table.add_row(
            m.real_env_name,
            ", ".join(m.upstream_hosts),
            _redact_token(m.proxy_token),
        )
    console.print(table)

    # ------------------------------------------------------------------ write
    console.print()
    console.print("[bold]Step 4[/bold]  Write config and persist mappings")

    cfg = load_config()
    proxy_cfg = cfg.setdefault("proxy", {})
    # ``args.tunnel_port`` is None when the flag was not given; ``0`` is
    # invalid for a TCP listener so we treat it as an explicit refusal
    # and surface a clear error rather than silently substituting the
    # default.
    if args.tunnel_port is not None:
        if args.tunnel_port < 1 or args.tunnel_port > 65534:
            console.print(
                "  [red]✗ --tunnel-port must be between 1 and 65534 "
                "(the plain-HTTP listener uses port+1).[/red]"
            )
            return 1
        tunnel_port = int(args.tunnel_port)
    else:
        tunnel_port = int(proxy_cfg.get("tunnel_port", ip._DEFAULT_TUNNEL_PORT))
    proxy_cfg["tunnel_port"] = tunnel_port

    extra_hosts = list(proxy_cfg.get("extra_allowed_hosts") or [])
    allowed = list(ip._DEFAULT_ALLOWED_HOSTS) + [
        h for h in extra_hosts if h not in ip._DEFAULT_ALLOWED_HOSTS
    ]

    audit_log_path = ip._proxy_state_dir() / "audit.log"
    # Pre-create the audit log with 0o600.  On the pinned v0.39 the
    # daemon does NOT write to this file (no ``log.audit_path`` field in
    # its config schema) — it's reserved for the v0.40+ upgrade where
    # per-request records start flowing.  Because the file is
    # non-load-bearing today, a pre-create failure (immutable parent,
    # pre-existing foreign-owned file, full disk) is a WARNING, not a
    # setup abort.
    audit_log_ok = True
    try:
        ip.ensure_audit_log(audit_log_path)
    except RuntimeError as exc:
        audit_log_ok = False
        console.print(f"  [yellow]⚠ {exc}[/yellow]")

    # Allow operator override of the deny list via
    # ``proxy.upstream_deny_cidrs`` — but the default (None) gives a safe
    # default-deny list (loopback, IMDS, RFC1918) that matches the docs
    # promise.
    deny_cidrs = proxy_cfg.get("upstream_deny_cidrs")
    iron_cfg = ip.build_proxy_config(
        mappings=mappings,
        ca_cert=ca_crt,
        ca_key=ca_key,
        tunnel_port=tunnel_port,
        audit_log=audit_log_path,
        allowed_hosts=allowed,
        upstream_deny_cidrs=deny_cidrs,
    )
    cfg_path = ip.write_proxy_config(iron_cfg)
    mappings_path = ip.write_mappings(mappings)
    console.print(f"  [green]✓[/green] config:   {cfg_path}")
    console.print(f"  [green]✓[/green] mappings: {mappings_path}")
    if audit_log_ok:
        console.print(
            f"  [green]✓[/green] audit log: {audit_log_path} "
            f"[dim](reserved — not written by iron-proxy v0.39; "
            f"per-request records land in iron-proxy.log)[/dim]"
        )

    # ------------------------------------------------------------------ enable
    proxy_cfg["enabled"] = True
    proxy_cfg.setdefault("auto_install", True)
    proxy_cfg.setdefault("enforce_on_docker", True)
    # CRITICAL: do NOT silently downgrade credential_source on re-run.
    # If the operator previously configured `bitwarden` mode (e.g. for
    # rotation), running `hermes egress setup` again WITHOUT
    # --from-bitwarden must not rewrite credential_source to "env" —
    # that silently breaks the Bitwarden rotation guarantee the docs
    # make.  Require an explicit --no-bitwarden to switch back.
    existing_source = proxy_cfg.get("credential_source")
    if args.from_bitwarden:
        proxy_cfg["credential_source"] = "bitwarden"
    elif getattr(args, "no_bitwarden", False):
        proxy_cfg["credential_source"] = "env"
        if existing_source == "bitwarden":
            console.print(
                "[yellow]Switched credential_source from bitwarden to env.[/yellow]"
            )
    elif existing_source == "bitwarden":
        # Preserve the existing bitwarden mode.  Surface the decision so
        # the operator knows we kept it.
        console.print(
            "[dim]Keeping credential_source=bitwarden from existing config. "
            "Pass --no-bitwarden to switch to env-based credentials.[/dim]"
        )
    else:
        proxy_cfg["credential_source"] = "env"
    proxy_cfg.setdefault("fail_on_uncovered_providers", False)
    save_config(cfg)

    live_status = ip.get_status()
    was_running = live_status.pid is not None
    if was_running:
        ip.stop_proxy()

    # Decide whether to (re)start the daemon so the new config/tokens take
    # effect, rather than leaving the operator to remember a manual restart
    # (the #1 UX papercut for this feature).
    #   --restart      → always (re)start, even if nothing was running
    #   --no-restart   → never; leave it as-is and print the manual hint
    #   neither + tty  → ask (only when a daemon was running)
    #   neither + !tty → restart when a daemon was running; otherwise no-op
    #                    (first-time setup never auto-starts — matches the
    #                    "configured, now run start" flow)
    import sys as _sys
    restart_pref = getattr(args, "restart", None)
    if restart_pref is True:
        do_restart = True
    elif restart_pref is False:
        do_restart = False
    elif was_running:
        if _sys.stdin.isatty():
            try:
                ans = input(
                    "  Restart the running proxy now with the new config? [Y/n] "
                ).strip().lower()
            except EOFError:
                ans = ""
            do_restart = ans in ("", "y", "yes")
        else:
            do_restart = True
    else:
        do_restart = False

    if do_restart:
        try:
            new_status = ip.start_proxy(
                install_if_missing=bool(proxy_cfg.get("auto_install", True)),
            )
        except Exception as exc:  # noqa: BLE001 — user-facing funnel
            console.print(
                f"  [yellow]⚠ could not start iron-proxy with the new "
                f"config: {exc}[/yellow]"
            )
            console.print(
                "  Run [cyan]hermes egress start[/cyan] manually before "
                "launching new Docker sandboxes."
            )
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

    console.print()
    console.print(
        "[green]✓ iron-proxy is configured.[/green]  "
        "Sandboxes will route outbound traffic through it."
    )
    console.print(
        "  Start:   [cyan]hermes egress start[/cyan]\n"
        "  Restart: [cyan]hermes egress restart[/cyan]  (after any re-setup)\n"
        "  Status:  [cyan]hermes egress status[/cyan]\n"
        "  Stop:    [cyan]hermes egress stop[/cyan]\n"
        "  Disable: [cyan]hermes egress disable[/cyan]"
    )
    return 0


def cmd_start(args: argparse.Namespace) -> int:
    console = Console()
    cfg = load_config()
    proxy_cfg = cfg.get("proxy") or {}
    if not proxy_cfg.get("enabled"):
        console.print(
            "[yellow]proxy.enabled is false — run `hermes egress setup` "
            "first.[/yellow]"
        )
        return 1

    # If the operator opted in to Bitwarden-rotation semantics, refresh
    # upstream secrets from BSM at startup.  This is what delivers the
    # rotation guarantee that distinguishes ``credential_source:
    # bitwarden`` from ``credential_source: env``.  Without it, rotating
    # a key in the Bitwarden web app doesn't reach the proxy.
    credential_source = proxy_cfg.get("credential_source", "env")
    bw_cfg = (cfg.get("secrets") or {}).get("bitwarden")
    refresh_bw = (
        credential_source == "bitwarden"
        and bw_cfg is not None
        and bool(bw_cfg.get("enabled"))
    )
    # Silent-degrade guard: the operator explicitly chose
    # ``credential_source: bitwarden``, but secrets.bitwarden has since
    # been disabled or removed.  Proceeding would quietly start on host
    # env — exactly the bug class the BW mode is meant to defeat.  Refuse
    # unless the documented escape hatch is set.
    if credential_source == "bitwarden" and not refresh_bw:
        if bool(proxy_cfg.get("allow_env_fallback", False)):
            console.print(
                "[yellow]⚠ credential_source=bitwarden but "
                "secrets.bitwarden is disabled or missing — falling back "
                "to host-env secrets (allow_env_fallback=true).  Rotated "
                "Bitwarden keys will NOT propagate.[/yellow]"
            )
        else:
            console.print(
                "[red]✗ Refusing to start: proxy.credential_source is "
                "'bitwarden' but secrets.bitwarden is disabled or "
                "missing.[/red]"
            )
            console.print(
                "  Re-enable it (`secrets.bitwarden.enabled: true`), switch "
                "back to env credentials with `hermes egress setup "
                "--no-bitwarden`, or set `proxy.allow_env_fallback: true` "
                "to opt into the host-env fallback."
            )
            return 1
    # Pass the proxy-side allow_env_fallback opt-in through to
    # start_proxy.  This is a deliberate, documented escape hatch: when
    # set, the daemon silently falls back to host env if BWS is
    # unreachable, instead of raising.  Default is strict (raise).
    if refresh_bw and bw_cfg is not None:
        bw_cfg = dict(bw_cfg)
        bw_cfg["allow_env_fallback"] = bool(
            proxy_cfg.get("allow_env_fallback", False)
        )

    # fail_on_uncovered_providers: when true, refuse to start if any
    # LLM-specific non-bearer providers (Anthropic native, Azure OpenAI,
    # Gemini) have env vars set in the host process — those would
    # otherwise leak real credentials into the sandbox while bypassing
    # the proxy.  Only the strict LLM-specific subset blocks; generic
    # cloud creds (AWS_*, GOOGLE_APPLICATION_CREDENTIALS) still surface
    # as warnings via `discover_uncovered_providers` but don't block, to
    # avoid tripping every operator with terraform / gcloud set up.
    if bool(proxy_cfg.get("fail_on_uncovered_providers", False)):
        blocked = ip.discover_blocked_providers()
        if blocked:
            console.print(
                "[red]✗ Refusing to start: provider env vars present "
                "that bypass the proxy:[/red]"
            )
            for name in blocked:
                console.print(f"    - {name}")
            console.print(
                "  Set `proxy.fail_on_uncovered_providers: false` in "
                "config.yaml to start anyway (sandbox will hold real "
                "credentials for those providers)."
            )
            return 1

    # stephenschoettler #1: when `credential_source: bitwarden`, the
    # operator picked BWS specifically to get the rotation guarantee —
    # silently falling back to parent-env at start_proxy time reintroduces
    # exactly the bug class the BW mode is supposed to defeat (host env
    # is stale / mismatched).  Pre-check at the wizard layer so we fail
    # loud with actionable error messages BEFORE start_proxy degrades.
    if refresh_bw:
        bw_access_env = (bw_cfg or {}).get("access_token_env", "BWS_ACCESS_TOKEN")
        if not os.environ.get(bw_access_env, "").strip():
            console.print(
                f"[red]✗ Refusing to start: credential_source=bitwarden but "
                f"{bw_access_env} is not set in the environment.[/red]"
            )
            console.print(
                "  Either export the access token, or run "
                "`hermes egress setup --no-bitwarden` to switch back to "
                "env-based credentials."
            )
            return 1
        if not (bw_cfg or {}).get("project_id"):
            console.print(
                "[red]✗ Refusing to start: credential_source=bitwarden but "
                "secrets.bitwarden.project_id is empty.[/red]"
            )
            console.print(
                "  Run `hermes secrets bitwarden setup` to configure the "
                "project, or switch back via `hermes egress setup "
                "--no-bitwarden`."
            )
            return 1

    try:
        status = ip.start_proxy(
            install_if_missing=bool(proxy_cfg.get("auto_install", True)),
            refresh_secrets_from_bitwarden=refresh_bw,
            bitwarden_config=bw_cfg,
        )
    except Exception as exc:  # noqa: BLE001 — top-level user-facing funnel
        console.print(f"[red]✗ failed to start iron-proxy:[/red] {exc}")
        return 1
    if status.pid:
        listening = (
            "[green]listening[/green]"
            if status.listening
            else "[yellow]not yet listening[/yellow]"
        )
        console.print(
            f"[green]✓[/green] iron-proxy running  pid={status.pid}  "
            f"port={status.tunnel_port}  {listening}"
        )
    else:
        console.print("[red]✗ iron-proxy did not come up cleanly[/red]")
        return 1
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    console = Console()
    if ip.stop_proxy():
        console.print("[green]✓[/green] iron-proxy stopped")
    else:
        console.print("[dim]iron-proxy was not running[/dim]")
    return 0


def cmd_restart(args: argparse.Namespace) -> int:
    """Stop the running daemon (if any) and start it with the current config.

    The one-command way to apply config changes (new allowlist hosts, rotated
    tokens, a Bitwarden key rotation) without making the operator remember the
    stop/start dance.  Delegates to ``cmd_start`` so all the credential-source
    and fail-on-uncovered-provider guards run exactly as they do for ``start``.
    """
    console = Console()
    was_running = ip.stop_proxy()
    if was_running:
        console.print("[dim]stopped the running iron-proxy[/dim]")
    return cmd_start(args)


def format_status_text(*, show_tokens: bool = False) -> str:
    """Plain-text egress status for slash commands, Dashboard, and Desktop."""
    cfg = load_config()
    proxy_cfg = cfg.get("proxy") or {}
    status = ip.get_status()

    def yn(value: bool) -> str:
        return "yes" if value else "no"

    lines = [
        "Egress proxy status",
        "",
        f"Enabled: {yn(bool(proxy_cfg.get('enabled')))}",
        f"Binary: {status.binary_path or '(missing)'}",
        f"Binary version: {status.binary_version or '(unknown)'}",
        f"Config: {status.config_path or '(not generated)'}",
        f"CA cert: {status.ca_cert_path or '(not generated)'}",
        f"Tunnel port: {status.tunnel_port}",
        f"Process: pid {status.pid}" if status.pid else "Process: (stopped)",
        f"Listening: {yn(status.listening)}",
        f"Credential src: {proxy_cfg.get('credential_source', 'env')}",
        f"Docker enforce: {yn(bool(proxy_cfg.get('enforce_on_docker', True)))}",
        "Scope: Docker backend only in this release",
    ]

    mappings = ip.load_mappings()
    if mappings:
        lines.extend(["", "Token mappings:"])
        for m in mappings:
            tok = m.proxy_token if show_tokens else _redact_token(m.proxy_token)
            lines.append(f"  - {m.real_env_name}: {tok} ({', '.join(m.upstream_hosts)})")

    uncovered = ip.discover_uncovered_providers()
    if uncovered:
        lines.extend([
            "",
            "Uncovered providers (real credentials still visible inside the sandbox):",
        ])
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
    table.add_row("Enabled",        _yn(bool(proxy_cfg.get("enabled"))))
    table.add_row("Binary",         str(status.binary_path or "[dim](missing)[/dim]"))
    table.add_row("Binary version", status.binary_version or "[dim](unknown)[/dim]")
    table.add_row("Config",         str(status.config_path or "[dim](not generated)[/dim]"))
    table.add_row("CA cert",        str(status.ca_cert_path or "[dim](not generated)[/dim]"))
    table.add_row("Tunnel port",    str(status.tunnel_port))
    table.add_row("Process",        f"pid {status.pid}" if status.pid else "[dim](stopped)[/dim]")
    table.add_row("Listening",      _yn(status.listening))
    table.add_row("Credential src", str(proxy_cfg.get("credential_source", "env")))
    table.add_row("Docker enforce", _yn(bool(proxy_cfg.get("enforce_on_docker", True))))
    console.print(table)

    mappings = ip.load_mappings()
    if mappings:
        console.print()
        console.print("[bold]Token mappings[/bold]")
        m_table = Table(show_header=True, header_style="bold")
        m_table.add_column("Real env", style="cyan")
        m_table.add_column("Upstream", style="dim")
        m_table.add_column("Proxy token", style="green")
        for m in mappings:
            tok = m.proxy_token if args.show_tokens else _redact_token(m.proxy_token)
            m_table.add_row(m.real_env_name, ", ".join(m.upstream_hosts), tok)
        console.print(m_table)
        if args.show_tokens:
            console.print(
                "[yellow]⚠[/yellow]  proxy tokens just printed in full — "
                "they may persist in your shell history.  Consider clearing "
                "it after this command."
            )

    # Surface uncovered providers so the operator knows the isolation
    # boundary is incomplete for those upstreams.
    uncovered = ip.discover_uncovered_providers()
    if uncovered:
        console.print()
        console.print(
            "[yellow]Uncovered providers[/yellow] "
            "(real credentials still visible inside the sandbox):"
        )
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
    # Use the public get_status() pid (which already incorporates the
    # _pid_alive check) instead of reaching into ip._read_pid().  That
    # private accessor only proves the pidfile is non-empty — a stale
    # pidfile from a crashed previous run would fire the warning
    # spuriously.
    if ip.get_status().pid is not None:
        console.print(
            "  iron-proxy is still running — stop it with "
            "[cyan]hermes egress stop[/cyan] if you want it down too."
        )
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    console = Console()
    status = ip.get_status()
    if status.config_path is None:
        console.print(
            "[yellow](no config generated — run `hermes egress setup`)[/yellow]"
        )
        return 1
    console.print(str(status.config_path))
    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_env_file_into_environ() -> int:
    """Backfill provider keys from ``~/.hermes/.env`` into ``os.environ``.

    ``hermes egress setup`` discovers providers by reading ``os.environ``, but
    many operators keep their keys ONLY in ``~/.hermes/.env`` (which the agent
    loads at runtime but which is NOT exported into an interactive shell).
    Without this, ``setup`` reports "no provider keys found" even though the
    keys plainly exist — a confusing first-run papercut.

    Only fills names that aren't already set in the process env (an exported
    value always wins), and only for known bearer-provider names so we don't
    slurp unrelated secrets into the process. Returns the count of names added.
    """
    try:
        from hermes_cli.config import load_env
    except ImportError:
        return 0
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


def _yn(value: bool) -> str:
    return "[green]yes[/green]" if value else "[dim]no[/dim]"


def _redact_token(token: str) -> str:
    if len(token) < 16:
        return token
    return f"{token[:12]}…{token[-4:]}"
