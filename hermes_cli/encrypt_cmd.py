"""CLI handlers for ``hermes encrypt ...`` — opt-in encryption-at-rest.

Subcommands:
    status         — show what is encrypted, the key source, and keystore health
    enable         — create the keystore and encrypt the sensitive files
    disable        — decrypt everything back to plaintext and remove the keystore
    rotate-key     — re-wrap the data key under a fresh key, or fully re-key with --full
    add-recovery   — add a one-time recovery code as a second key slot
    unlock         — prime the data key for a headless run (passphrase mode)
    clean-backups  — remove plaintext migration backups under .encryption/backup/
    read-log       — decrypt a rotated log segment or session transcript
    sweep-sessions — encrypt plaintext sessions left behind by a crashed writer

The heavy crypto dependencies (cryptography, argon2-cffi, keyring) are
lazy-installed on first ``enable`` via ``tools.lazy_deps``.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


# ---------------------------------------------------------------------------
# Argparse wiring — called from hermes_cli.main
# ---------------------------------------------------------------------------


def register_cli(parent_parser: argparse.ArgumentParser) -> None:
    """Attach the ``encrypt`` subcommand tree to a parent parser."""
    sub = parent_parser.add_subparsers(dest="encrypt_command")

    status = sub.add_parser("status", help="Show encryption state and keystore health")
    status.set_defaults(func=cmd_status)

    enable = sub.add_parser("enable", help="Encrypt the sensitive files under HERMES_HOME")
    enable.add_argument(
        "--key-source",
        choices=["keyring", "passphrase", "keyfile"],
        help="Where the encryption key lives (default: config / keyring)",
    )
    enable.add_argument(
        "--databases",
        action="store_true",
        help="Also encrypt state.db / kanban.db with SQLCipher",
    )
    enable.add_argument(
        "--encrypt-logs",
        action="store_true",
        help=(
            "Also encrypt rotated log segments and existing session transcripts. "
            "Live logs/sessions stay plaintext; rotated/closed files get HRMSENC."
        ),
    )
    enable.add_argument(
        "--no-recovery",
        action="store_true",
        help="Skip generating a recovery code (passphrase mode)",
    )
    enable.add_argument("--yes", "-y", action="store_true", help="Skip the confirmation prompt")
    enable.add_argument(
        "--force",
        action="store_true",
        help="Proceed even if another Hermes gateway/agent process is running",
    )
    enable.set_defaults(func=cmd_enable)

    disable = sub.add_parser("disable", help="Decrypt everything back to plaintext")
    disable.add_argument("--yes", "-y", action="store_true", help="Skip the confirmation prompt")
    disable.add_argument(
        "--force",
        action="store_true",
        help="Proceed even if another Hermes gateway/agent process is running",
    )
    disable.set_defaults(func=cmd_disable)

    rotate = sub.add_parser(
        "rotate-key",
        help="Re-wrap the data key under a fresh key, or fully re-key encrypted data",
    )
    rotate.add_argument(
        "--full",
        action="store_true",
        help="Generate a new data key and re-encrypt every encrypted artifact "
        "(use after suspected DEK compromise)",
    )
    rotate.add_argument(
        "--key-source",
        choices=["keyring", "passphrase", "keyfile"],
        help="Switch to a different key source while rotating",
    )
    rotate.add_argument(
        "--force",
        action="store_true",
        help=(
            "Proceed even if another Hermes process is running. With a live "
            "writer, SQLCipher -wal/-shm sidecars are snapshotted best-effort "
            "but may be locked; stop the gateway first for guaranteed rollback "
            "safety on encrypted databases."
        ),
    )
    rotate.add_argument(
        "--keep-backups",
        action="store_true",
        help="With --full, retain old-DEK ciphertext backups under .encryption/backup/",
    )
    rotate.add_argument("--yes", "-y", action="store_true", help="Skip the confirmation prompt")
    rotate.set_defaults(func=cmd_rotate_key)

    recovery = sub.add_parser("add-recovery", help="Generate a one-time recovery code")
    recovery.set_defaults(func=cmd_add_recovery)

    unlock = sub.add_parser(
        "unlock",
        help=(
            "Verify the passphrase / key can unlock the keystore "
            "(headless: set HERMES_ENCRYPTION_PASSPHRASE)"
        ),
    )
    unlock.set_defaults(func=cmd_unlock)

    clean = sub.add_parser(
        "clean-backups",
        help="Remove plaintext migration backups under .encryption/backup/",
    )
    clean.add_argument(
        "--older-than",
        type=int,
        metavar="N",
        help="Only delete backups at least N days old",
    )
    clean.add_argument("--yes", "-y", action="store_true", help="Skip the confirmation prompt")
    clean.add_argument(
        "--include-ciphertext",
        action="store_true",
        help=(
            "Also delete rekey-run ciphertext backups (sealed under the old DEK) "
            "in addition to plaintext credential backups. Off by default — these "
            "are typically kept for forensic recovery after `rotate-key --full "
            "--keep-backups`."
        ),
    )
    clean.set_defaults(func=cmd_clean_backups)

    read_log = sub.add_parser(
        "read-log",
        help=(
            "Decrypt and print an encrypted rotated log segment "
            "(agent.log.1, …) or a session transcript (sessions/*.jsonl)"
        ),
    )
    read_log.add_argument(
        "file",
        help=(
            "Path to a live or rotated log under ~/.hermes/logs/, or a session "
            "transcript under ~/.hermes/sessions/"
        ),
    )
    read_log.set_defaults(func=cmd_read_log)

    sweep = sub.add_parser(
        "sweep-sessions",
        help=(
            "Encrypt plaintext session transcripts left behind by a crashed "
            "writer (skips files whose lockfile is held by a live writer)"
        ),
    )
    sweep.set_defaults(func=cmd_sweep_sessions)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_deps(console: "Console", *, databases: bool = False) -> bool:
    """Lazy-install the encryption dependencies. Returns False on failure."""
    try:
        from tools.lazy_deps import FeatureUnavailable, ensure
    except ImportError:
        console.print("[red]Lazy-install support unavailable.[/red]")
        return False
    try:
        ensure("tool.encryption")
        if databases:
            ensure("tool.encryption_db")
    except FeatureUnavailable as exc:
        console.print(f"[red]Could not install encryption dependencies:[/red] {exc}")
        return False
    return True


def _argon2_params() -> dict:
    """Return the configured Argon2id cost parameters."""
    try:
        from hermes_cli.config import cfg_get, load_config

        block = cfg_get(load_config(), "security", "encryption", "argon2", default=None)
        if isinstance(block, dict):
            return {
                "time_cost": int(block.get("time_cost", 3)),
                "memory_cost_kib": int(block.get("memory_cost_kib", 131072)),
                "parallelism": int(block.get("parallelism", 4)),
            }
    except Exception:
        pass
    return {}


def _no_passphrase_available(console: "Console", what: str = "the passphrase") -> None:
    """Explain why we cannot collect a passphrase in this environment."""
    env = _passphrase_env()
    console.print(
        f"[red]stdin is not a TTY and {env} is not set — cannot prompt for "
        f"{what}.[/red]\n"
        f"  Set [cyan]{env}[/cyan] or run this command in an interactive "
        "terminal."
    )


def _read_current_passphrase(
    console: "Console", prompt: str = "  Encryption passphrase: "
) -> "str | None":
    """Collect the CURRENT keystore passphrase without ever blocking.

    Resolution order mirrors ``hermes_crypto.get_data_key``: the
    ``HERMES_ENCRYPTION_PASSPHRASE`` environment variable (headless / CI)
    first, then an interactive ``getpass`` prompt when stdin is a TTY.
    Returns ``None`` — after printing an actionable error — when neither is
    available, so callers exit cleanly instead of ``getpass`` hanging
    forever on a piped or closed stdin.
    """
    env_pw = os.environ.get(_passphrase_env())
    if env_pw:
        return env_pw
    if sys.stdin is not None and sys.stdin.isatty():
        return getpass.getpass(prompt) or None
    _no_passphrase_available(console)
    return None


def _prompt_new_passphrase(console: "Console") -> "str | None":
    """Collect a NEW passphrase: prompt twice on a TTY, env var when headless."""
    if sys.stdin is None or not sys.stdin.isatty():
        # Headless enable/rotate (CI, Docker build): accept the new passphrase
        # from the environment — there is nobody to type the confirmation.
        env_pw = os.environ.get(_passphrase_env())
        if env_pw:
            return env_pw
        _no_passphrase_available(console, what="a new passphrase")
        return None
    first = getpass.getpass("  New encryption passphrase: ")
    if not first:
        console.print("  [red]Empty passphrase — aborting.[/red]")
        return None
    second = getpass.getpass("  Confirm passphrase: ")
    if first != second:
        console.print("  [red]Passphrases do not match — aborting.[/red]")
        return None
    return first


def _prepare_new_key_source(
    console: "Console",
    args: argparse.Namespace,
    current: "str | None",
) -> "tuple[bool, str | None]":
    """Validate the new key source upfront. Returns (ok, new_passphrase).

    Called BEFORE full_rekey() / rotate_primary() so a typoed passphrase or
    a missing keyring backend aborts the rekey before any disk write commits.
    Returns (True, passphrase) when a new passphrase was collected, (True, None)
    when no rotation is needed or the new source is non-passphrase, and
    (False, None) when collection failed (caller should return 1).

    when the operator asks to rotate to ``passphrase`` while
    already in passphrase mode, we *must* still prompt for a new passphrase
    — that's the whole point of ``rotate-key --key-source passphrase``. The
    old short-circuit (``new_source == current → return True, None``) passed
    ``new_passphrase=None`` through to ``keystore.rotate_primary``, which
    crashed with a confusing ``ValueError`` from the keystore layer. The
    same-source shortcut now only applies for keyring/keyfile modes, where
    no fresh secret is needed.
    """
    from hermes_crypto import keystore

    new_source = args.key_source
    if not new_source:
        return True, None  # no key-source change requested at all
    # Same-source case: only short-circuit when no new secret is needed.
    # passphrase mode falls through to the prompt branch below — see note.
    if new_source == current and new_source != "passphrase":
        return True, None
    if new_source == "passphrase":
        console.print("  Set the new passphrase:")
        pw = _prompt_new_passphrase(console)
        if pw is None:
            return False, None
        return True, pw
    if new_source == "keyring" and not keystore.keyring_is_secure():
        console.print("[red]No secure OS keyring backend available.[/red]")
        return False, None
    return True, None


def _state_style(state: str) -> str:
    return {
        "encrypted": "[green]encrypted[/green]",
        "plaintext": "[yellow]plaintext[/yellow]",
        "missing": "[dim]—[/dim]",
    }.get(state, state)


def _format_bytes(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KiB"
    return f"{size / (1024 * 1024):.1f} MiB"


def _format_sessions(sessions: dict) -> str:
    count = int(sessions.get("count", 0))
    if count == 0:
        return "[dim]none[/dim]"
    encrypted = int(sessions.get("encrypted", 0))
    plaintext = int(sessions.get("plaintext", 0))
    locked = int(sessions.get("locked", 0))
    parts = [f"{count} total"]
    if encrypted:
        parts.append(f"[green]{encrypted} encrypted[/green]")
    if plaintext:
        parts.append(f"[yellow]{plaintext} plaintext[/yellow]")
    if locked:
        parts.append(f"[dim]{locked} live[/dim]")
    return ", ".join(parts)


def _format_backups(backups: dict) -> str:
    count = int(backups.get("count", 0))
    if count == 0:
        return "[dim]none[/dim]"
    parts = [f"[yellow]{count} plaintext file(s)[/yellow]"]
    total_bytes = backups.get("total_bytes")
    if isinstance(total_bytes, int) and total_bytes > 0:
        parts.append(_format_bytes(total_bytes))
    oldest_days = backups.get("oldest_days")
    if isinstance(oldest_days, int):
        parts.append(f"oldest {oldest_days} day(s)")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def cmd_status(args: argparse.Namespace) -> int:
    console = Console()
    from hermes_crypto import migrate

    snapshot = migrate.status()

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("", style="bold")
    table.add_column("")
    enabled = snapshot["enabled"]
    table.add_row("Encryption", "[green]enabled[/green]" if enabled else "[dim]disabled[/dim]")
    table.add_row("Key source", str(snapshot["primary_slot"] or snapshot["config_key_source"] or "—"))
    table.add_row("Keystore", "present" if snapshot["keystore_exists"] else "[dim]none[/dim]")
    table.add_row("Recovery code", "set" if snapshot["has_recovery"] else "[yellow]none[/yellow]")
    table.add_row(
        "OS keyring",
        "[green]secure backend[/green]" if snapshot["keyring_secure"]
        else "[yellow]insecure/unavailable[/yellow]",
    )
    table.add_row(
        "SQLCipher",
        "[green]available[/green]" if snapshot["sqlcipher_available"]
        else "[dim]not installed[/dim]",
    )
    encrypt_logs = snapshot.get("encrypt_logs", False)
    table.add_row(
        "Log rotation encryption",
        "[green]enabled[/green]" if encrypt_logs else "[dim]disabled[/dim]",
    )
    from hermes_crypto import running_under_openshell

    table.add_row(
        "Runtime sandbox",
        "[green]inside an OpenShell sandbox[/green]" if running_under_openshell()
        else "[dim]not sandboxed (see SECURITY.md)[/dim]",
    )
    table.add_row("Session transcripts", _format_sessions(snapshot.get("sessions", {})))
    backups = snapshot.get("backups", {})
    table.add_row("Migration backups", _format_backups(backups))
    console.print(Panel(table, title="Encryption-at-rest", border_style="cyan"))

    files = Table(show_header=True, header_style="bold")
    files.add_column("File")
    files.add_column("State")
    for entry in snapshot["files"]:
        files.add_row(entry["name"], _state_style(entry["state"]))
    for entry in snapshot["databases"]:
        files.add_row(entry["name"], _state_style(entry["state"]))
    console.print(files)

    # Recent security-audit events — the failed-unlock line is the one to watch.
    from hermes_crypto import audit

    events = audit.read_recent(5)
    if events:
        console.print("\n[bold]Recent security events[/bold]")
        for event in events:
            colour = {"critical": "red", "warning": "yellow"}.get(event.get("severity"), "dim")
            console.print(
                f"  [{colour}]{event.get('timestamp', '?')}  "
                f"{event.get('activity', '?')}  ({event.get('outcome', '?')})[/{colour}]"
            )

    if not running_under_openshell():
        console.print(
            "\n  [dim]Tip: encryption protects data on a cold disk. To also confine "
            "the running\n  agent, run Hermes under a sandbox such as NVIDIA OpenShell.[/dim]"
        )
    backup_count = int(backups.get("count", 0)) if isinstance(backups, dict) else 0
    if backup_count:
        console.print(
            "\n  [yellow]Plaintext migration backups defeat encryption-at-rest on a "
            "stolen disk.[/yellow]\n"
            "  Remove them with [cyan]hermes encrypt clean-backups[/cyan] once you "
            "have verified encryption."
        )
    if not enabled:
        console.print("\n  Run [cyan]hermes encrypt enable[/cyan] to turn on encryption.")
    return 0


def cmd_enable(args: argparse.Namespace) -> int:
    console = Console()
    from hermes_crypto import keystore, migrate

    if keystore.keystore_exists():
        from hermes_crypto import is_encryption_enabled

        if not is_encryption_enabled():
            # The enabled flag is flipped last by migrate.enable(), so a
            # keystore without it means a previous enable was interrupted.
            console.print(
                "[yellow]A keystore exists but encryption is not enabled in "
                "config — a previous enable was likely interrupted.[/yellow]\n"
                "  Run [cyan]hermes encrypt disable[/cyan] to roll back to "
                "plaintext, then re-run [cyan]hermes encrypt enable[/cyan]."
            )
            return 1
        console.print(
            "[yellow]A keystore already exists — encryption appears enabled.[/yellow]\n"
            "  Use [cyan]hermes encrypt status[/cyan] or [cyan]rotate-key[/cyan]."
        )
        return 1

    # Resolve the key source: explicit flag → config default → keyring.
    key_source = args.key_source
    if not key_source:
        try:
            from hermes_cli.config import cfg_get, load_config

            key_source = cfg_get(
                load_config(), "security", "encryption", "key_source", default="keyring"
            )
        except Exception:
            key_source = "keyring"

    console.print(
        Panel.fit(
            "[bold]Enable encryption-at-rest[/bold]\n\n"
            "The sensitive files under HERMES_HOME (.env, auth.json, OAuth\n"
            "tokens" + (", state.db, kanban.db" if args.databases else "") + ") will be "
            "AES-256-GCM encrypted on disk.\n\n"
            "[bold red]If you lose the key or passphrase, this data is\n"
            "PERMANENTLY UNRECOVERABLE.[/bold red] Plaintext backups are kept\n"
            "under .encryption/backup/ until you remove them.\n\n"
            f"Key source: [cyan]{key_source}[/cyan]",
            border_style="yellow",
        )
    )
    console.print(
        "  [dim]Stop any running gateway/agent before continuing — migration "
        "aborts if another Hermes process is detected (use --force to override).[/dim]"
    )
    if not args.yes:
        if not sys.stdin.isatty():
            console.print("[red]Refusing to enable non-interactively without --yes.[/red]")
            return 1
        if input("\n  Continue? [y/N] ").strip().lower() not in {"y", "yes"}:
            console.print("  Aborted.")
            return 0

    if not _ensure_deps(console, databases=args.databases):
        return 1

    # keyring mode needs a real OS keyring backend.
    if key_source == "keyring" and not keystore.keyring_is_secure():
        console.print(
            "[red]No secure OS keyring backend is available on this host.[/red]\n"
            "  Use passphrase mode instead:  "
            "[cyan]hermes encrypt enable --key-source passphrase[/cyan]"
        )
        return 1

    passphrase = None
    if key_source == "passphrase":
        passphrase = _prompt_new_passphrase(console)
        if passphrase is None:
            return 1

    try:
        result = migrate.enable(
            key_source,
            passphrase=passphrase,
            encrypt_databases=args.databases,
            encrypt_logs=args.encrypt_logs,
            argon2_params=_argon2_params() or None,
            progress=lambda msg: console.print(f"  [dim]{msg}[/dim]"),
            force=args.force,
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"\n[red]Encryption failed:[/red] {exc}")
        console.print("  No config flag was set — your install is unchanged.")
        return 1

    # Offer a recovery code (passphrase mode, unless opted out).
    if key_source == "passphrase" and not args.no_recovery:
        try:
            code = keystore.add_recovery_slot()
            console.print(
                Panel.fit(
                    "[bold]Recovery code[/bold] — store this somewhere safe and "
                    "OFFLINE.\nIt is the only way back in if you forget the "
                    f"passphrase.\n\n  [bold cyan]{code}[/bold cyan]",
                    border_style="green",
                )
            )
        except Exception as exc:  # noqa: BLE001
            console.print(f"  [yellow]Could not create a recovery code: {exc}[/yellow]")

    n = (
        len(result.encrypted_files)
        + len(result.encrypted_databases)
        + len(result.encrypted_sessions)
        + len(getattr(result, "encrypted_logs", []))
    )
    console.print(f"\n[green]✓ Encryption enabled — {n} file(s) encrypted.[/green]")
    if result.encrypted_sessions:
        console.print(
            f"  [dim]{len(result.encrypted_sessions)} session transcript(s) "
            "encrypted at migration time.[/dim]"
        )
    encrypted_logs = getattr(result, "encrypted_logs", [])
    if encrypted_logs:
        console.print(
            f"  [dim]{len(encrypted_logs)} rotated log segment(s) "
            "encrypted at migration time.[/dim]"
        )
    console.print("  Check it any time with [cyan]hermes encrypt status[/cyan].")
    return 0


def cmd_disable(args: argparse.Namespace) -> int:
    console = Console()
    from hermes_crypto import keystore, migrate

    if not keystore.keystore_exists():
        console.print("[yellow]Encryption is not enabled — nothing to do.[/yellow]")
        return 0

    if not args.yes:
        if not sys.stdin.isatty():
            console.print("[red]Refusing to disable non-interactively without --yes.[/red]")
            return 1
        console.print(
            "[yellow]This decrypts every encrypted file back to plaintext on disk.[/yellow]"
        )
        if input("  Continue? [y/N] ").strip().lower() not in {"y", "yes"}:
            console.print("  Aborted.")
            return 0

    if not _ensure_deps(console):
        return 1

    # Unlock first (passphrase / recovery mode needs the secret). In keyring
    # or keyfile mode, keystore.unlock() reads from the configured slot with
    # no input — and on failure we fall through to a recovery-code prompt
    # when one is available, so a broken keyring / missing keyfile is not a
    # lockout if the operator kept their recovery code.
    passphrase: str | None = None
    recovery: str | None = None
    source = keystore.primary_slot_type()
    has_recovery = keystore.has_recovery_slot()

    if source == "passphrase":
        # Env var first (headless), then the interactive prompt; never let
        # getpass block on a non-TTY stdin.
        passphrase = os.environ.get(_passphrase_env()) or None
        if passphrase is None and sys.stdin is not None and sys.stdin.isatty():
            if has_recovery:
                console.print(
                    "  Enter the encryption passphrase, "
                    "or leave it empty to use a recovery code."
                )
            passphrase = getpass.getpass("  Encryption passphrase: ") or None
            if passphrase is None and has_recovery:
                recovery = getpass.getpass("  Recovery code: ").strip() or None
        if passphrase is None and recovery is None:
            if sys.stdin is None or not sys.stdin.isatty():
                _no_passphrase_available(console)
            console.print(
                "[red]No passphrase or recovery code provided — cannot disable.[/red]"
            )
            return 1
    elif source in {"keyring", "keyfile"} and has_recovery:
        # Try the configured slot first; on failure, fall back to recovery.
        # keystore.unlock() populates the process-lifetime DEK cache
        # (AGENTS.md §3.10), so migrate.disable's own unlock is a cache hit.
        try:
            keystore.unlock()
        except Exception as exc:  # noqa: BLE001
            console.print(
                f"  [yellow]Could not unlock via {source} "
                f"({type(exc).__name__}). Falling back to recovery code.[/yellow]"
            )
            if sys.stdin is not None and sys.stdin.isatty():
                recovery = getpass.getpass("  Recovery code: ").strip() or None
            else:
                console.print(
                    "[red]stdin is not a TTY — cannot prompt for the recovery "
                    "code. Run this command in an interactive terminal.[/red]"
                )
            if recovery is None:
                console.print(
                    "[red]No recovery code provided — cannot disable.[/red]"
                )
                return 1

    try:
        migrate.disable(
            passphrase=passphrase,
            recovery_code=recovery,
            progress=lambda msg: console.print(f"  [dim]{msg}[/dim]"),
            force=args.force,
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"\n[red]Disable failed:[/red] {exc}")
        return 1
    console.print("\n[green]✓ Encryption disabled — files are plaintext again.[/green]")
    return 0


def cmd_read_log(args: argparse.Namespace) -> int:
    console = Console()
    from hermes_crypto.log_handler import read_log_text

    if not _ensure_deps(console):
        return 1
    try:
        text = read_log_text(args.file)
    except FileNotFoundError:
        console.print(f"[red]File not found:[/red] {args.file}")
        return 1
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Could not read log:[/red] {exc}")
        return 1
    sys.stdout.write(text)
    if text and not text.endswith("\n"):
        sys.stdout.write("\n")
    return 0


def cmd_sweep_sessions(args: argparse.Namespace) -> int:
    console = Console()
    from hermes_crypto import keystore, logs_encryption_active, migrate

    if not keystore.keystore_exists():
        console.print("[yellow]Encryption is not enabled — nothing to sweep.[/yellow]")
        return 1
    if not logs_encryption_active():
        console.print(
            "[yellow]Session encryption is not active "
            "(security.encryption.encrypt_logs is off).[/yellow]"
        )
        return 1
    if not _ensure_deps(console):
        return 1

    try:
        result = migrate.sweep_sessions(
            progress=lambda msg: console.print(f"  [dim]{msg}[/dim]"),
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Sweep failed:[/red] {exc}")
        return 1

    swept = result.get("swept") or []
    skipped = result.get("skipped_locked") or []
    errors = result.get("errors") or []

    if not (swept or skipped or errors):
        console.print("[dim]No abandoned session transcripts found.[/dim]")
        return 0

    if swept:
        console.print(
            f"[green]✓ Encrypted {len(swept)} abandoned session "
            f"transcript(s).[/green]"
        )
    if skipped:
        console.print(
            f"  [dim]Skipped {len(skipped)} session(s) currently held by a "
            "live writer.[/dim]"
        )
    if errors:
        console.print(
            f"  [yellow]{len(errors)} session(s) could not be encrypted — "
            "see security-audit.jsonl for details.[/yellow]"
        )
        return 1
    return 0


def cmd_clean_backups(args: argparse.Namespace) -> int:
    console = Console()
    from hermes_crypto import migrate

    summary = migrate.backup_summary()
    count = int(summary.get("count", 0))
    if count == 0:
        console.print("[dim]No migration backups found under .encryption/backup/.[/dim]")
        return 0

    plaintext_count = int(summary.get("plaintext_count", count))
    ciphertext_count = int(summary.get("ciphertext_count", 0))

    if args.older_than is not None:
        if args.older_than < 0:
            console.print("[red]--older-than must be zero or positive.[/red]")
            return 1
        age_clause = f" older than {args.older_than} day(s)"
    else:
        age_clause = ""

    # the panel now distinguishes plaintext credential
    # backups (which the original copy described as "cleartext credentials")
    # from rekey-run ciphertext backups (sealed under the OLD DEK by
    # ``full_rekey --keep-backups``) so an operator running ``clean-backups``
    # doesn't unexpectedly delete forensic ciphertext.
    if args.include_ciphertext:
        target_desc = (
            f"all {count} migration backup(s){age_clause} — "
            f"{plaintext_count} plaintext + {ciphertext_count} ciphertext "
            "(ciphertext sealed under the OLD DEK; included via --include-ciphertext)"
        )
        body = (
            "[bold]Remove ALL migration backups (plaintext + ciphertext)[/bold]\n\n"
            f"This will delete {target_desc} under .encryption/backup/.\n"
            "Plaintext backups contain the original cleartext credentials.\n"
            "Ciphertext backups are old-DEK rekey-run snapshots; they are "
            "[yellow]not[/yellow] cleartext but cannot be opened with the "
            "current DEK either."
        )
    else:
        target_desc = f"{plaintext_count} plaintext migration backup(s){age_clause}"
        body = (
            "[bold]Remove plaintext migration backups[/bold]\n\n"
            f"This will delete {target_desc} under .encryption/backup/.\n"
            "These files contain the original cleartext credentials — remove "
            "them only after you have verified encryption works."
        )
        if ciphertext_count:
            body += (
                f"\n\n[dim]Skipping {ciphertext_count} ciphertext rekey-run "
                "backup(s) — pass [cyan]--include-ciphertext[/cyan] to remove "
                "those too.[/dim]"
            )

    console.print(Panel.fit(body, border_style="yellow"))
    if not args.yes:
        if not sys.stdin.isatty():
            console.print("[red]Refusing to clean backups non-interactively without --yes.[/red]")
            return 1
        if input("\n  Continue? [y/N] ").strip().lower() not in {"y", "yes"}:
            console.print("  Aborted.")
            return 0

    result = migrate.clean_backups(
        older_than_days=args.older_than,
        include_ciphertext=bool(args.include_ciphertext),
    )
    deleted = int(result.get("deleted_count", 0))
    kept = int(result.get("kept_count", 0))
    skipped_ct = int(result.get("skipped_ciphertext", 0))
    errors = result.get("errors") or []

    if errors:
        console.print(
            f"\n[yellow]Removed {deleted} backup(s), kept {kept}, "
            f"but {len(errors)} file(s) could not be deleted.[/yellow]"
        )
        return 1

    if deleted == 0:
        console.print(
            f"\n[dim]No backups matched"
            + (f" --older-than {args.older_than}" if args.older_than is not None else "")
            + f"; {kept} kept.[/dim]"
        )
    else:
        kind = "backup(s)" if args.include_ciphertext else "plaintext backup(s)"
        console.print(f"\n[green]✓ Removed {deleted} {kind}.[/green]")
        if kept:
            console.print(f"  Kept {kept} newer backup(s).")
    if skipped_ct and not args.include_ciphertext:
        console.print(
            f"  [dim]Skipped {skipped_ct} ciphertext rekey-run backup(s). "
            "Re-run with [cyan]--include-ciphertext[/cyan] to remove them.[/dim]"
        )
    return 0


def cmd_rotate_key(args: argparse.Namespace) -> int:
    console = Console()
    from hermes_crypto import keystore
    from hermes_crypto.migrate import full_rekey

    if not keystore.keystore_exists():
        console.print("[yellow]Encryption is not enabled — nothing to rotate.[/yellow]")
        return 1
    if not _ensure_deps(console):
        return 1

    if args.full:
        has_recovery = keystore.has_recovery_slot()
        if args.keep_backups:
            backup_note = (
                "Old-DEK ciphertext backups from this run will be kept "
                "under .encryption/backup/ ([cyan]--keep-backups[/cyan])."
            )
        else:
            backup_note = (
                "Old-DEK ciphertext backups from this run will be removed "
                "after success unless you pass [cyan]--keep-backups[/cyan]."
            )
        recovery_note = (
            "\n\n[bold yellow]Recovery code slot(s) will be invalidated.[/bold yellow]"
            if has_recovery
            else "\n\n[dim]No recovery slot is configured.[/dim]"
        )
        # always print the panel and the gateway-stop hint, even when
        # --yes was passed. A scripted operator running this from cron still
        # needs the warnings to land in the job log; only the interactive
        # confirm-prompt is gated on --yes.
        console.print(
            Panel.fit(
                "[bold]Full data-key re-key[/bold]\n\n"
                "This generates a [bold]new data encryption key (DEK)[/bold] and "
                "re-encrypts every encrypted artifact — credentials, databases, "
                "sessions, and log segments.\n\n"
                "[yellow]This is not a simple key-source rotation — the DEK "
                "itself changes.[/yellow]"
                + recovery_note
                + f"\n\n{backup_note}",
                border_style="yellow",
            )
        )
        console.print(
            "  [dim]Stop any running gateway/agent before continuing — re-key "
            "aborts if another Hermes process is detected (use --force to override).[/dim]"
        )
        if not args.yes:
            if not sys.stdin.isatty():
                console.print(
                    "[red]Refusing to full re-key non-interactively without --yes.[/red]"
                )
                return 1
            if input("\n  Continue? [y/N] ").strip().lower() not in {"y", "yes"}:
                console.print("  Aborted.")
                return 0

    # Unlock with the current credentials.
    current = keystore.primary_slot_type()
    current_passphrase: str | None = None
    try:
        if current == "passphrase":
            current_passphrase = _read_current_passphrase(
                console, prompt="  Current passphrase: "
            )
            if current_passphrase is None:
                return 1
            keystore.unlock(passphrase=current_passphrase)
        else:
            keystore.unlock()
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Could not unlock the keystore:[/red] {exc}")
        return 1

    # Validate (and collect) the new key source BEFORE any disk-committing
    # rekey runs — a typoed new passphrase or missing keyring backend must
    # abort before full_rekey()/rotate_primary() touches the keystore.
    ok, new_passphrase = _prepare_new_key_source(console, args, current)
    if not ok:
        return 1

    if args.full:
        try:
            result = full_rekey(
                passphrase=current_passphrase,
                progress=lambda msg: console.print(f"  {msg}"),
                force=args.force,
                keep_backups=args.keep_backups,
            )
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Full re-key failed:[/red] {exc}")
            return 1
        console.print(
            "[green]✓ Data key rotated.[/green] "
            f"Re-encrypted {len(result.rekeyed_files)} credential file(s), "
            f"{len(result.rekeyed_databases)} database(s), "
            f"{len(result.rekeyed_sessions)} session(s), "
            f"{len(result.rekeyed_logs)} log segment(s)."
        )
        if result.backups_removed:
            console.print(
                f"  [dim]Cleaned {result.backups_removed} old-DEK backup(s).[/dim]"
            )
        if result.recovery_slots_dropped:
            console.print(
                "[yellow]Recovery code(s) were invalidated — run "
                "[cyan]hermes encrypt add-recovery[/cyan] to create a new one.[/yellow]"
            )
        if result.skipped:
            console.print(f"  Skipped {len(result.skipped)} plaintext artifact(s).")
        if args.key_source and args.key_source != current:
            return _finish_key_source_rotation(
                console, args, current, current_passphrase, new_passphrase
            )
        return 0

    new_source = args.key_source or current
    # new_passphrase was collected upfront by _prepare_new_key_source. The
    # non-rotate case (new_source == current) leaves new_passphrase as None.

    try:
        keystore.rotate_primary(
            new_source, new_passphrase=new_passphrase, argon2_params=_argon2_params() or None
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Rotation failed:[/red] {exc}")
        return 1
    console.print(
        f"[green]✓ Key rotated.[/green] New key source: [cyan]{new_source}[/cyan]. "
        "Encrypted files were not rewritten — only the key wrapping changed."
    )
    if args.key_source:
        from hermes_crypto.migrate import _set_config

        _set_config("security.encryption.key_source", new_source)
    return 0


def _finish_key_source_rotation(
    console: Console,
    args: argparse.Namespace,
    current: str | None,
    current_passphrase: str | None,
    new_passphrase: str | None,
) -> int:
    """After a full re-key, optionally re-wrap under a new key source."""
    from hermes_crypto import keystore
    from hermes_crypto.migrate import _set_config

    new_source = args.key_source
    # Prompt and keyring availability were already validated in
    # _prepare_new_key_source(); proceed straight to rotation.
    try:
        keystore.rotate_primary(
            new_source,
            new_passphrase=new_passphrase,
            argon2_params=_argon2_params() or None,
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Key-source rotation failed:[/red] {exc}")
        return 1
    _set_config("security.encryption.key_source", new_source)
    console.print(
        f"[green]✓ Key source updated to [cyan]{new_source}[/cyan].[/green]"
    )
    return 0


def cmd_add_recovery(args: argparse.Namespace) -> int:
    console = Console()
    from hermes_crypto import keystore

    if not keystore.keystore_exists():
        console.print("[yellow]Encryption is not enabled.[/yellow]")
        return 1
    if not _ensure_deps(console):
        return 1

    source = keystore.primary_slot_type()
    try:
        if source == "passphrase":
            passphrase = _read_current_passphrase(console)
            if passphrase is None:
                return 1
            keystore.unlock(passphrase=passphrase)
        else:
            keystore.unlock()
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Could not unlock the keystore:[/red] {exc}")
        return 1

    try:
        code = keystore.add_recovery_slot()
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Could not add a recovery code:[/red] {exc}")
        return 1
    console.print(
        Panel.fit(
            "[bold]Recovery code[/bold] — store it OFFLINE. Any previous "
            f"recovery code is now invalid.\n\n  [bold cyan]{code}[/bold cyan]",
            border_style="green",
        )
    )
    return 0


def cmd_unlock(args: argparse.Namespace) -> int:
    console = Console()
    from hermes_crypto import keystore

    if not keystore.keystore_exists():
        console.print("[yellow]Encryption is not enabled.[/yellow]")
        return 1
    if not _ensure_deps(console):
        return 1

    source = keystore.primary_slot_type()
    try:
        if source == "passphrase":
            passphrase = _read_current_passphrase(console)
            if passphrase is None:
                return 1
            keystore.unlock(passphrase=passphrase)
        else:
            keystore.unlock()
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Unlock failed:[/red] {exc}")
        return 1
    console.print("[green]✓ Keystore unlocked — the key is valid.[/green]")
    if source == "passphrase":
        console.print(
            "  For a headless gateway, set the "
            f"[cyan]{_passphrase_env()}[/cyan] environment variable so the "
            "agent can unlock at startup."
        )
    return 0


def _passphrase_env() -> str:
    try:
        from hermes_crypto import PASSPHRASE_ENV_VAR

        return PASSPHRASE_ENV_VAR
    except Exception:
        return "HERMES_ENCRYPTION_PASSPHRASE"
