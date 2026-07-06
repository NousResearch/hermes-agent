"""CLI handlers for ``hermes secrets protonpass ...``.

Subcommands:
    setup    — interactive wizard: install pass-cli, store token, pick a vault
    status   — show current config + binary version + token state
    sync     — run a fetch right now and show what would be applied (dry-run friendly)
    disable  — flip ``secrets.protonpass.enabled`` to False
    install  — just download the pass-cli binary (no token / vault required)

Parallel to :mod:`hermes_cli.secrets_cli` (Bitwarden).  Config parsing and the
apply/skip rules are NOT re-derived here: ``ProtonPassConfig.from_mapping`` owns
config invariants and ``plan_application`` owns the applied/skipped decision, so
the wizard, the sync dry-run, and startup all agree.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agent.secret_sources import protonpass as pp
from agent.secret_sources.protonpass import install as _install
from agent.secret_sources.protonpass import (
    SKIP_ALREADY_SET,
    SKIP_BOOTSTRAP_TOKEN,
    ProtonPassConfig,
    plan_application,
)
from agent.secret_sources.protonpass.config import is_valid_env_name, strip_bootstrap_ref
from agent.secret_sources.protonpass.session import _redact_token
from hermes_cli.config import (
    get_env_path,
    load_config,
    save_config,
    save_env_value,
)
from hermes_cli.secret_prompt import masked_secret_prompt


# ---------------------------------------------------------------------------
# config-section access — the single normalization for `secrets.protonpass`
# ---------------------------------------------------------------------------
#
# A hand-edited config.yaml may carry a non-dict at ``secrets`` or
# ``secrets.protonpass`` (e.g. ``secrets: true`` or ``protonpass: true``).
# Every command reads that section through one of these two helpers so a scalar
# can never crash a command with ``'bool' object has no attribute 'get'``.


def _read_protonpass_section(cfg: dict) -> object:
    """Read the raw ``secrets.protonpass`` value for read-only commands.

    Returns whatever is stored there (handed straight to
    ``ProtonPassConfig.from_mapping``, which tolerates ``None``/bool/junk), or
    ``None`` when ``secrets`` itself is a non-dict scalar — so status/sync never
    call ``.get`` on a bool.  Does NOT mutate ``cfg``.
    """
    secrets = cfg.get("secrets")
    if not isinstance(secrets, dict):
        return None
    return secrets.get("protonpass")


def _ensure_protonpass_section(cfg: dict) -> dict:
    """Return a mutable ``secrets.protonpass`` dict for write commands.

    Normalizes a non-dict scalar at either ``secrets`` or ``secrets.protonpass``
    to ``{}`` IN PLACE so setup/disable can mutate it without crashing.
    """
    if not isinstance(cfg.get("secrets"), dict):
        cfg["secrets"] = {}
    secrets = cfg["secrets"]
    if not isinstance(secrets.get("protonpass"), dict):
        secrets["protonpass"] = {}
    return secrets["protonpass"]


# ---------------------------------------------------------------------------
# argparse wiring — called from hermes_cli.main
# ---------------------------------------------------------------------------


def register_protonpass_cli(parent_parser: argparse.ArgumentParser) -> None:
    """Attach the ``protonpass`` subcommand tree to a parent parser.

    Called from ``hermes_cli.main`` as part of building the top-level
    ``hermes secrets`` parser.  Mirrors the Bitwarden ``register_cli``.
    """
    sub = parent_parser.add_subparsers(dest="secrets_pp_command")

    setup = sub.add_parser(
        "setup",
        help="Interactive wizard: install pass-cli, store token, pick a vault",
    )
    setup.add_argument(
        "--vault",
        help="Pre-select a vault name (MODE A) instead of prompting",
    )
    setup.add_argument(
        "--token-env",
        metavar="VAR",
        help=(
            "Read the service token from this environment variable instead of "
            "prompting (non-interactive use). Never pass the token itself on "
            "the command line."
        ),
    )
    setup.set_defaults(func=cmd_pp_setup)

    status = sub.add_parser("status", help="Show config + binary + token state")
    status.set_defaults(func=cmd_pp_status)

    sync = sub.add_parser("sync", help="Fetch secrets now and report what changed")
    sync.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Set the secrets in this hermes process now (does NOT export to "
            "your parent shell; default: dry-run)"
        ),
    )
    sync.set_defaults(func=cmd_pp_sync)

    disable = sub.add_parser("disable", help="Turn off the Proton Pass integration")
    disable.set_defaults(func=cmd_pp_disable)

    install = sub.add_parser(
        "install",
        help=f"Download and verify the pinned pass-cli binary (v{pp._PASS_CLI_VERSION})",
    )
    install.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if a managed copy already exists",
    )
    install.set_defaults(func=cmd_pp_install)


# ---------------------------------------------------------------------------
# handlers
# ---------------------------------------------------------------------------


def cmd_pp_setup(args: argparse.Namespace) -> int:
    console = Console()
    console.print(
        Panel.fit(
            "[bold]Proton Pass setup[/bold]\n\n"
            "Hermes pulls secrets from Proton Pass via the official "
            "[cyan]pass-cli[/cyan].\n"
            "You need a service token. Two kinds exist:\n\n"
            "  [bold]Agent token[/bold]  (RECOMMENDED) - scoped to ONE vault, "
            "viewer role, expiring:\n"
            "    [cyan]pass-cli agent create --vault <VAULT> "
            "--expiration <e> <name>[/cyan]\n"
            "    [cyan]pass-cli agent access grant --role viewer ...[/cyan]\n\n"
            "  [bold]Personal access token (PAT)[/bold] - grants FULL account "
            "access.\n\n"
            "[bold red]SECURITY:[/bold red] a PERSONAL access token can read "
            "and write your ENTIRE\n"
            "Proton account. An AGENT token is limited to a single vault with a "
            "viewer\n"
            "role and an expiration. Prefer the agent token - it is the "
            "least-privilege\n"
            "choice for an unattended process like Hermes.",
            border_style="cyan",
        )
    )

    # ------------------------------------------------------------------ binary
    console.print()
    console.print("[bold]Step 1[/bold]  Install the pass-cli CLI")
    try:
        binary = pp.find_pass_cli(install_if_missing=False)
        if binary is None:
            console.print("  No pass-cli on PATH - downloading...")
            binary = pp.install_pass_cli()
        version = _pp_version(binary)
        console.print(f"  [green]OK[/green] {binary}  ({version})")
    except Exception as exc:  # noqa: BLE001
        console.print(f"  [red]Could not install pass-cli: {exc}[/red]")
        console.print(
            "  Manual install: https://proton.me/download/pass-cli"
        )
        return 1

    # ------------------------------------------------------------------- token
    console.print()
    console.print("[bold]Step 2[/bold]  Provide your service token")
    console.print(
        "  [yellow]Reminder:[/yellow] prefer a scoped [bold]agent token[/bold] "
        "(one vault, viewer role,\n"
        "  expiring) over a full personal access token. The PAT can access your "
        "whole account."
    )
    cfg = load_config()
    # A hand-edited config.yaml may carry a non-dict at `secrets` or
    # `secrets.protonpass` (e.g. `protonpass: true`); normalize BOTH levels to a
    # dict BEFORE mutating so the writes below can't crash on a scalar.
    secrets_cfg = _ensure_protonpass_section(cfg)
    # Parse the existing config through the single coercion home so a malformed
    # value (a bare bool, a non-numeric TTL, wrong types) can't break the setup
    # wizard before we re-write a clean config.  We still mutate `secrets_cfg`
    # (the raw dict) below so `save_config` persists the assembled result.
    parsed_cfg = ProtonPassConfig.from_mapping(secrets_cfg)
    token_env = parsed_cfg.service_token_env

    if args.token_env and not is_valid_env_name(args.token_env):
        # NEVER echo the value: a fumbled `--token-env <actual-token>` (Proton
        # tokens contain '-'/'/'/'=', which fail validation) would otherwise
        # copy the secret into terminal/CI logs.  Name only the constraint.
        console.print(
            "  [red]--token-env must be a valid environment variable name "
            "(letters, digits, underscores; not starting with a digit).[/red]"
        )
        return 1

    token = _read_pp_service_token(args, token_env, console)
    if not token:
        console.print("  [red]Empty token, aborting.[/red]")
        return 1
    # Early, clear shape check so an obviously-wrong paste fails here rather
    # than as an opaque auth error during the test fetch below.  Proton Pass
    # service tokens are prefixed `pst_`; warn (not fatal) if it doesn't match.
    if not _looks_like_pp_token(token):
        console.print(
            "  [yellow]Warning: token doesn't start with 'pst_' or is shorter "
            "than expected - usually that means you pasted something other "
            "than a Proton Pass service token. Continuing anyway.[/yellow]"
        )

    # Set the token in-memory so the Step-4 test fetch below can authenticate,
    # but DEFER persisting it to .env until AFTER the test fetch succeeds (or the
    # no-target path).  Otherwise a failed test fetch (return 1 below) would
    # leave the token saved in .env with no matching config entry.
    os.environ[token_env] = token  # so the test fetch below sees it
    console.print(
        "  [dim]The token will be saved to ~/.hermes/.env (never config.yaml) "
        "once the test fetch succeeds.[/dim]"
    )

    # ------------------------------------------------------------------- vault
    console.print()
    console.print("[bold]Step 3[/bold]  Pick a fetch mode")
    console.print(
        "  [bold]MODE B[/bold] (preferred): map env vars to single fields via "
        "pass:// refs in\n"
        "  config.yaml under [cyan]secrets.protonpass.env[/cyan]. Works under "
        "agent and PAT\n"
        "  sessions; least-privilege. Example:\n"
        "    [cyan]OPENROUTER_API_KEY: \"pass://SHARE_ID/ITEM_ID/api_key\"[/cyan]\n"
        "  [bold]MODE A[/bold]: list one vault by name. Maps every item field to "
        "an env var.\n"
        "  PAT-session-only - bulk listing with secret values is rejected under "
        "agent tokens."
    )

    vault = (args.vault or "").strip()
    if not vault:
        try:
            vault = console.input(
                "  Vault name for MODE A (Enter to skip and use MODE B refs): "
            ).strip()
        except EOFError:
            # A piped-token setup can be non-interactive after stdin is
            # consumed. Treat EOF the same as pressing Enter so the normal
            # no-target path leaves the integration disabled cleanly.
            vault = ""

    env_refs = parsed_cfg.env_refs

    # Single no-vault-no-refs invariant (same check startup uses): a config
    # with neither a vault nor any refs has nothing to fetch.  Reuse the config
    # model's invariant rather than re-deriving it.
    has_target = ProtonPassConfig(vault=vault, env_refs=env_refs).has_fetch_target()
    if not has_target:
        console.print(
            "  [yellow]No vault chosen and no env refs in config yet.[/yellow]\n"
            "  Add MODE B refs under [cyan]secrets.protonpass.env[/cyan] in "
            "config.yaml, then\n"
            "  re-run [cyan]hermes secrets protonpass setup[/cyan]. Leaving the "
            "integration DISABLED for now (an enabled-but-empty config would be "
            "a guaranteed no-op that warns on every startup)."
        )

    # ------------------------------------------------------------------- test
    console.print()
    console.print("[bold]Step 4[/bold]  Test fetch")
    if not has_target:
        console.print(
            "  [dim]Skipping fetch test - no vault and no env refs configured.[/dim]"
        )
    else:
        try:
            secrets, warnings = pp.fetch_protonpass_secrets(
                service_token=token,
                vault=vault,
                env_refs=env_refs,
                binary=binary,
                use_cache=False,
                auto_install=parsed_cfg.auto_install,
                bootstrap_env=token_env,
            )
        except Exception as exc:  # noqa: BLE001
            # Defense-in-depth: fetch/session errors are already scrubbed, but
            # redact the token out of any error string before display in case a
            # message ever interpolates it.  ``_redact_token`` is a no-op on an
            # empty token.
            console.print(
                f"  [red]Fetch failed: {_redact_token(str(exc), token)}[/red]"
            )
            console.print(
                "  [dim]The token was NOT saved to .env (the test fetch failed); "
                "re-run setup once the token / config is correct.[/dim]"
            )
            return 1

        if not secrets:
            console.print(
                "  [yellow]Fetch succeeded but returned no secrets.[/yellow]"
            )
        else:
            # Render the SAME plan startup would apply (default override=False).
            plan = plan_application(
                secrets, os.environ, override_existing=False, token_env=token_env
            )
            table = Table(show_header=True, header_style="bold")
            table.add_column("Name", style="cyan")
            table.add_column("Status")
            for item in plan:
                table.add_row(item.name, _plan_status_label(item))
            console.print(table)
        for w in warnings:
            console.print(f"  [yellow]warning:[/yellow] {w}")

    # ------------------------------------------------------------------- save
    # Persist the token to .env now, AFTER the test fetch succeeded (or the
    # no-target path, where the token is intentionally kept for next time).  A
    # failed test fetch returned 1 above WITHOUT reaching here, so we never leave
    # a saved token with no matching config entry.
    save_env_value(token_env, token)
    console.print(f"  [green]OK[/green] token stored in {get_env_path()} as {token_env}")

    # Setup invariant: never enable a config that has nothing to fetch — that
    # would be an enabled-but-guaranteed-no-op state that warns on every
    # startup.  Match how startup treats that same state (disabled).
    secrets_cfg["enabled"] = has_target
    secrets_cfg["vault"] = vault
    secrets_cfg.setdefault("service_token_env", token_env)
    secrets_cfg.setdefault("env", env_refs)
    secrets_cfg.setdefault("cache_ttl_seconds", 300)
    secrets_cfg.setdefault("override_existing", False)
    secrets_cfg.setdefault("auto_install", True)
    save_config(cfg)

    console.print()
    if has_target:
        console.print(
            "[green]Proton Pass is enabled.[/green]  "
            "Secrets will be pulled at the start of every Hermes process."
        )
    else:
        console.print(
            "[yellow]Proton Pass is left disabled[/yellow] until you add a vault "
            "or MODE B refs. The token is saved so setup will be quick next time."
        )
    console.print(
        "  Status:  [cyan]hermes secrets protonpass status[/cyan]\n"
        "  Refresh: [cyan]hermes secrets protonpass sync[/cyan]\n"
        "  Disable: [cyan]hermes secrets protonpass disable[/cyan]"
    )
    return 0


def cmd_pp_status(args: argparse.Namespace) -> int:
    console = Console()
    cfg = load_config()
    pp_cfg = ProtonPassConfig.from_mapping(_read_protonpass_section(cfg))

    token_set = bool(os.environ.get(pp_cfg.service_token_env))

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("", style="bold")
    table.add_column("")
    table.add_row("Enabled",         _yn(pp_cfg.enabled))
    table.add_row("Token env var",   pp_cfg.service_token_env)
    table.add_row("Token in env",    _yn(token_set))
    table.add_row("Vault (MODE A)",  pp_cfg.vault or "[dim](unset)[/dim]")
    table.add_row(
        "Env refs (MODE B)",
        str(len(pp_cfg.env_refs)) if pp_cfg.env_refs else "[dim](none)[/dim]",
    )
    table.add_row("Override existing", _yn(pp_cfg.override_existing))
    table.add_row("Cache TTL (s)",   str(pp_cfg.cache_ttl_seconds))
    table.add_row("Auto-install",    _yn(pp_cfg.auto_install))

    binary = pp.find_pass_cli(install_if_missing=False)
    if binary:
        table.add_row("pass-cli binary", f"{binary} ({_pp_version(binary)})")
    else:
        table.add_row("pass-cli binary", "[yellow]not installed[/yellow]")

    console.print(Panel(table, title="Proton Pass", border_style="cyan"))

    if not pp_cfg.enabled:
        console.print("\n  Run [cyan]hermes secrets protonpass setup[/cyan] to enable.")
        return 0
    if not token_set:
        console.print(
            f"\n  [yellow]Enabled but {pp_cfg.service_token_env} is not set - "
            "Hermes will skip Proton Pass and warn on next startup.[/yellow]"
        )
    if not pp_cfg.has_fetch_target():
        console.print(
            "\n  [yellow]Enabled but no vault (MODE A) and no env refs (MODE B) "
            "- nothing to fetch.[/yellow]"
        )
    return 0


def cmd_pp_sync(args: argparse.Namespace) -> int:
    console = Console()
    cfg = load_config()
    pp_cfg = ProtonPassConfig.from_mapping(_read_protonpass_section(cfg))
    if not pp_cfg.enabled:
        console.print(
            "[yellow]Proton Pass integration is disabled.  Run "
            "`hermes secrets protonpass setup` first.[/yellow]"
        )
        return 1

    token_env = pp_cfg.service_token_env
    token = os.environ.get(token_env, "").strip()
    if not token:
        console.print(f"[red]{token_env} is not set.[/red]")
        return 1

    if not pp_cfg.has_fetch_target():
        console.print(
            "[red]No vault (MODE A) and no env refs (MODE B) configured.[/red]"
        )
        return 1

    # Strip — before fetching — a MODE B ref whose env var equals the bootstrap
    # service-token env var, via the single centralized helper.
    # ``apply_protonpass_secrets`` does the same; the planner would refuse to
    # apply it anyway (it must never clobber the token we authenticated with),
    # so fetching it is wasted work (a wasted ``item view``).  Mirror that here
    # so sync and apply agree.
    env_refs, _skipped, _strip_warnings = strip_bootstrap_ref(
        dict(pp_cfg.env_refs), token_env
    )

    # Re-check after stripping: an env-refs-only config whose single ref was the
    # bootstrap token is now empty, so there is genuinely nothing to fetch.
    if not pp_cfg.vault and not env_refs:
        console.print(
            "[yellow]The only configured MODE B ref targets the bootstrap "
            "service-token env var (never overridden) - nothing to fetch.[/yellow]"
        )
        return 0

    try:
        secrets, warnings = pp.fetch_protonpass_secrets(
            service_token=token,
            vault=pp_cfg.vault,
            env_refs=env_refs,
            use_cache=False,
            auto_install=pp_cfg.auto_install,
            bootstrap_env=token_env,
        )
    except Exception as exc:  # noqa: BLE001
        # Defense-in-depth: redact the token out of any error string before
        # display (fetch/session errors are already scrubbed).  No-op on empty.
        console.print(f"[red]Fetch failed: {_redact_token(str(exc), token)}[/red]")
        return 1

    if not secrets:
        console.print("[yellow]No secrets returned.[/yellow]")
        # B8: still surface fetch warnings even when nothing was applied — they
        # are usually the reason every ref / vault item was skipped (timeouts,
        # rejected --show-secrets, malformed refs). Returning early here hid them.
        for w in warnings:
            console.print(f"[yellow]warning:[/yellow] {w}")
        return 0

    # Render the SAME plan apply uses.  --apply forces override_existing so the
    # exported set matches what the user asked to apply now.
    override = pp_cfg.override_existing or args.apply
    plan = plan_application(
        secrets, os.environ, override_existing=override, token_env=token_env
    )
    table = Table(show_header=True, header_style="bold")
    table.add_column("Name", style="cyan")
    table.add_column("Action")
    applied = 0
    for item in plan:
        if not item.applied:
            if item.reason == SKIP_BOOTSTRAP_TOKEN:
                table.add_row(item.name, "[dim]skip (bootstrap token)[/dim]")
            else:
                table.add_row(item.name, "[dim]skip (already set)[/dim]")
            continue
        suffix = (" (overrode)" if item.overrides else "") if args.apply else (
            " (overrides)" if item.overrides else ""
        )
        if args.apply:
            os.environ[item.name] = item.value
            applied += 1
            table.add_row(item.name, "[green]exported[/green]" + suffix)
        else:
            table.add_row(item.name, "[green]would export[/green]" + suffix)

    console.print(table)
    for w in warnings:
        console.print(f"[yellow]warning:[/yellow] {w}")

    if not args.apply:
        console.print(
            "\n  This was a dry-run - secrets are picked up automatically on the "
            "next [cyan]hermes[/cyan] invocation.  Re-run with [cyan]--apply[/cyan] "
            "to set them in THIS hermes process now (it does NOT export into "
            "your parent shell)."
        )
    else:
        console.print(
            f"\n  [green]Set {applied} secret(s) in this hermes process.[/green] "
            "These are not exported to your parent shell; they are picked up "
            "automatically on the next [cyan]hermes[/cyan] invocation."
        )
    return 0


def cmd_pp_disable(args: argparse.Namespace) -> int:
    console = Console()
    cfg = load_config()
    # Normalize a non-dict `secrets` / `secrets.protonpass` (e.g.
    # `protonpass: true`) before mutating so disable can't crash on a scalar.
    pp_cfg = _ensure_protonpass_section(cfg)
    pp_cfg["enabled"] = False
    save_config(cfg)
    console.print(
        "[green]Disabled.[/green]  Proton Pass secrets will NOT be pulled on the "
        "next Hermes invocation.\n"
        "  Your service token is left in .env - remove it manually (and revoke it "
        "in Proton Pass) if you also want to revoke the credential."
    )
    return 0


def cmd_pp_install(args: argparse.Namespace) -> int:
    console = Console()
    try:
        path = pp.install_pass_cli(force=bool(args.force))
        console.print(f"[green]OK[/green] {path}  ({_pp_version(path)})")
        return 0
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Install failed: {exc}[/red]")
        return 1


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _yn(b: bool) -> str:
    return "[green]yes[/green]" if b else "[dim]no[/dim]"


def _plan_status_label(item) -> str:
    """Render a setup-test status cell for one plan item (dry-run, override=False)."""
    if not item.applied:
        if item.reason == SKIP_BOOTSTRAP_TOKEN:
            return "[dim]bootstrap token - never overrides itself[/dim]"
        if item.reason == SKIP_ALREADY_SET:
            return "[yellow]already set in env (kept by default)[/yellow]"
        return "[dim]skip[/dim]"
    return "[green]new[/green]"


def _looks_like_pp_token(token: str) -> bool:
    """Heuristic shape check for a Proton Pass service token.

    Service tokens are prefixed ``pst_`` and are not tiny.  This is a cheap
    early guard, not authentication — a real token is still validated by the
    test fetch.
    """
    return token.startswith("pst_") and len(token) >= 12


def _read_pp_service_token(
    args: argparse.Namespace, token_env: str, console: Console
) -> str:
    """Obtain the service token WITHOUT ever accepting it on argv.

    Resolution order:
      1. ``--token-env VAR``: read the token from the named environment
         variable (non-interactive use, e.g. CI). The value never appears in
         the process's own argv.
      2. Non-interactive stdin (piped): read one line.
      3. Interactive: masked ``getpass``-style prompt.

    Returning the empty string means "no token provided" and the caller aborts.
    """
    # 1. Env-var indirection (explicit) - never the token on the command line.
    src_var = (getattr(args, "token_env", None) or "").strip()
    if src_var:
        value = os.environ.get(src_var, "").strip()
        if not value:
            console.print(
                f"  [red]--token-env {src_var} is set but {src_var} is empty "
                "in the environment.[/red]"
            )
            return ""
        console.print(f"  [dim]Read token from ${src_var}.[/dim]")
        return value

    # 2/3. stdin: a pipe (non-interactive) reads one line; a TTY uses the
    # masked prompt (which itself falls back to getpass when not a TTY).
    if not sys.stdin.isatty():
        line = sys.stdin.readline()
        return line.strip()
    return masked_secret_prompt(f"  Paste service token ({token_env}): ").strip()


def _pp_version(binary: Path) -> str:
    """Return a human-readable pass-cli version string, or "version unknown".

    V4: this delegates to ``install.get_pass_cli_version`` (the public, scrubbed-
    env probe), which probes the binary with a MINIMAL, scrubbed environment (no
    token, none of the loaded secrets).  The previous bespoke ``subprocess.run``
    here inherited the FULL process env, leaking every loaded secret to whatever
    ``pass-cli`` happened to be first on ``PATH`` (which may be unverified at this
    point).  Reusing the scrubbed probe closes that leak.
    """
    reported = _install.get_pass_cli_version(Path(binary))
    if reported:
        # The probe already trims whitespace; take the first line for display
        # (a multi-line --version banner stays a single status cell).
        lines = reported.splitlines()
        if lines:
            return lines[0]
    return "version unknown"
