"""``hermes vault`` — manage model-blind autofill credentials.

Subcommands:
- ``hermes vault add``   interactive wizard; logins go through the configured
  credential broker while cards and addresses remain in the local vault.
- ``hermes vault list``  metadata — labels, kinds, identifiers, origins,
  handles. Passwords are never shown.
- ``hermes vault rm``    remove an item by handle/id.

The vault backs the password-blind browser autofill tools
(``browser_vault_list`` / ``browser_vault_fill``): the agent sees handles
and login identifiers, types the identifier itself, and fills the password
server-side without ever seeing it.
"""

from __future__ import annotations

import getpass


def _console():
    from rich.console import Console

    return Console()


def _cmd_add(args) -> None:
    from agent.vault_store import (
        LOGIN_IDENTIFIER_TYPES,
        VAULT_KINDS,
        get_vault_store,
    )

    c = _console()
    c.print(
        "[bold]Add a vault item[/] (the password is encrypted at rest and the "
        "agent never sees it; the identifier is visible metadata the agent "
        "can type itself)"
    )

    kind = (args.kind or "").strip().lower()
    while kind not in VAULT_KINDS:
        kind = input(f"Kind ({'/'.join(VAULT_KINDS)}) [login]: ").strip().lower() or "login"
        if kind not in VAULT_KINDS:
            c.print(f"[red]Unknown kind {kind!r}[/]")
            kind = ""

    label = ""
    while not label:
        label = input("Label (e.g. 'GitHub work account'): ").strip()

    sensitive = {}
    try:
        if kind == "login":
            from agent.credential_broker import get_credential_broker

            origin = ""
            while not origin:
                origin = input("Site origin (e.g. https://github.com): ").strip()
            id_type = ""
            while id_type not in LOGIN_IDENTIFIER_TYPES:
                id_type = (
                    input(f"Identifier type ({'/'.join(LOGIN_IDENTIFIER_TYPES)}) [email]: ")
                    .strip()
                    .lower()
                    or "email"
                )
            identifier = ""
            while not identifier:
                identifier = input(f"{id_type.capitalize()}: ").strip()
            password = ""
            while not password:
                password = getpass.getpass("Password (hidden): ")
            sensitive["password"] = password
            broker = get_credential_broker()
            backend = broker.write_backend()
            if backend.needs_unlock and not backend.is_unlocked():
                master = getpass.getpass(f"Unlock {backend.display_name} (hidden): ")
                try:
                    backend.unlock(master)  # type: ignore[attr-defined]
                finally:
                    del master
            saved = broker.save_login(label=label, origin=origin, identifier_type=id_type,
                                      identifier=identifier, password=password)
            meta = saved.meta
            del password
        else:
            from agent.vault_store import ADDRESS_FIELDS, PAYMENT_FIELDS, REQUIRED_FIELDS

            fields = PAYMENT_FIELDS if kind == "payment" else ADDRESS_FIELDS
            origin = ""
            while not origin:
                origin = input("Site origin the item may be filled on (e.g. https://shop.example.com): ").strip()
            c.print(f"[dim]{kind} fields are filled only on that origin; card values are read hidden.[/]")
            secret = {}
            sensitive = secret
            for field in fields:
                required = field in REQUIRED_FIELDS[kind]
                prompt = f"{field.replace('_', ' ')}{'' if required else ' (optional)'}: "
                read = getpass.getpass if kind == "payment" else input
                value = read(prompt).strip()
                while required and not value:
                    value = read(prompt).strip()
                if value:
                    secret[field] = value
            meta = get_vault_store().add_item(kind=kind, label=label, secret=secret, origin=origin)
    except Exception as exc:
        from agent.vault_store import scrub_secret_from_text
        c.print(f"[red]Error:[/] {scrub_secret_from_text(str(exc), sensitive)}")
        return
    finally:
        sensitive.clear()
        if "password" in locals():
            password = ""

    c.print(f"[green]Stored.[/] handle=[bold]{meta.id}[/] kind={meta.kind} origin={meta.origin or '-'}")


def _cmd_list(args) -> None:
    """Local items always; external managers only for the lifetime of this CLI process (a
    `hermes vault list` unlock does not carry into a chat session — unlock there when asked)."""
    from agent.vault_backends import enabled_backends

    c = _console()
    rows, locked = [], []
    for backend in enabled_backends():
        if backend.needs_unlock and not backend.is_unlocked():
            locked.append(backend.display_name)
            continue
        rows.extend((backend.display_name, meta) for meta in backend.list_items())
    if not rows and not locked:
        c.print("[dim]Vault is empty. Add an item with `hermes vault add`.[/]")
        return
    if rows:
        from rich.table import Table

        table = Table(title=f"Vault items ({len(rows)})")
        for col in ("Handle", "Source", "Kind", "Label", "Identifier", "Origin"):
            table.add_column(col, style="bold" if col == "Handle" else None)
        for source, meta in rows:
            table.add_row(meta.id, source, meta.kind, meta.label, meta.identifier or "-", meta.origin or "-")
        c.print(table)
        c.print("[dim]Passwords are never shown; the agent fills them server-side from the handle.[/]")
    for name in locked:
        c.print(f"[yellow]{name}[/] is enabled but locked — the agent will ask you to unlock it when it needs a login.")


def _cmd_sources(args) -> None:
    """Show password managers and configure read/write routing."""
    from agent.vault_backends import enabled_backends
    from agent.vault_backends.base import external_backend_classes, is_installed
    from hermes_cli.config import _ensure_dict, load_config, save_config

    c = _console()
    classes = {cls.name: cls for cls in external_backend_classes()}
    if args.enable or args.disable or args.write:
        name = args.enable or args.disable or args.write
        if name == "local" and not args.write:
            c.print("[red]local is always enabled and cannot be toggled[/]")
            return
        if name == "local" and args.write:
            cfg = load_config()
            cfg.setdefault("vault", {})["write_backend"] = "local"
            save_config(cfg)
            c.print("[green]Hermes vault[/] is now the credential write backend.")
            return
        if args.write and name not in {"local", "bitwarden"}:
            c.print("[red]Writable credential backends are local and bitwarden.[/]")
            return
        if name not in classes:
            c.print(f"[red]Unknown password manager {name!r}[/] (expected one of {', '.join(classes)})")
            return
        if (args.enable or args.write) and not is_installed(name):
            c.print(f"[red]{classes[name].display_name} CLI is not installed.[/]")
            return
        cfg = load_config()
        section = _ensure_dict(_ensure_dict(cfg, "vault"), name)
        if args.enable:
            section["enabled"] = True
        elif args.disable:
            section["enabled"] = False
            if cfg["vault"].get("write_backend") == name:
                cfg["vault"]["write_backend"] = "local"
        else:
            section["enabled"] = True
            cfg["vault"]["write_backend"] = name
        save_config(cfg)
        if args.write:
            c.print(f"[green]{classes[name].display_name}[/] is now the credential write backend.")
        else:
            c.print(f"[green]{classes[name].display_name} {'on' if args.enable else 'off'}[/] for browser logins.")
        return
    enabled = {b.name for b in enabled_backends()}
    from agent.vault_backends.base import vault_config
    write_backend = str(vault_config().get("write_backend") or "local")
    for name, cls in classes.items():
        if name in enabled:
            status = "[green]detected[/] · the agent asks you to unlock it when it needs a login"
        elif is_installed(name):
            status = "[dim]turned off[/] (`hermes vault sources --enable {name}` to use it)".format(name=name)
        else:
            status = "[dim]not installed[/]"
        marker = " · [bold]writes here[/]" if write_backend == name else ""
        c.print(f"  {cls.display_name:<10} {status}{marker}")
    if write_backend == "local":
        c.print("  Hermes vault [green]enabled[/] · [bold]writes here[/]")
    c.print("[dim]Managers are picked up automatically when their CLI is installed and signed in.[/]")


def _cmd_rm(args) -> None:
    from agent.credential_broker import get_credential_broker

    c = _console()
    broker = get_credential_broker()
    sensitive = {}
    try:
        backend = broker.backend_for_handle(args.handle)
        if backend is not None and backend.needs_unlock and not backend.is_unlocked():
            master = getpass.getpass(f"Unlock {backend.display_name} (hidden): ")
            sensitive["master_password"] = master
            backend.unlock(master)  # type: ignore[attr-defined]
            del master
        if broker.remove_item(args.handle):
            c.print(f"[green]Removed[/] {args.handle}")
        else:
            c.print(f"[red]No vault item with handle {args.handle!r}[/]")
    except Exception as exc:
        from agent.vault_store import scrub_secret_from_text
        c.print(f"[red]Error:[/] {scrub_secret_from_text(str(exc), sensitive)}")
    finally:
        sensitive.clear()
        if "master" in locals():
            master = ""


def _cmd_migrate_local(args) -> None:
    from agent.credential_broker import get_credential_broker
    from agent.vault_backends.base import UnlockRequired

    c = _console()
    broker = get_credential_broker()
    while True:
        try:
            result = broker.migrate_local_logins(execute=bool(args.execute))
            break
        except UnlockRequired as exc:
            master = getpass.getpass(f"Unlock {exc.backend.display_name} (hidden): ")
            try:
                exc.backend.unlock(master)  # type: ignore[attr-defined]
            except Exception as unlock_exc:
                from agent.vault_store import scrub_secret_from_text
                c.print(f"[red]Error:[/] {scrub_secret_from_text(str(unlock_exc), {'master': master})}")
                return
            finally:
                master = ""
        except Exception as exc:
            c.print(f"[red]Error:[/] {exc}")
            return

    planned = sum(item.status == "would_import" for item in result.items)
    c.print(
        f"Local login migration: scanned={len(result.items)} target={result.target} "
        f"imported={result.imported} skipped={result.skipped} failed={result.failed} planned={planned}"
    )
    for item in result.items:
        style = "red" if item.status == "failed" else "dim"
        suffix = f" — {item.error}" if item.error else ""
        c.print(f"[{style}]{item.status}[/] {item.label} · {item.identifier} · {item.origin}{suffix}")
    c.print("[dim]Local source items were not deleted.[/]")


def register_cli(subparser) -> None:
    """Build the ``hermes vault`` argparse tree (called from main.py)."""
    subs = subparser.add_subparsers(dest="vault_action")

    p_add = subs.add_parser(
        "add",
        help="Save a login, card or address ahead of time (optional: the agent asks you on the page when it needs one)",
    )
    p_add.add_argument(
        "--kind", choices=["login", "payment", "address"], default=None,
        help="Item kind (interactive prompt when omitted)",
    )
    p_add.set_defaults(_vault_handler=_cmd_add)

    p_list = subs.add_parser("list", help="List vault items (metadata only, never values)")
    p_list.set_defaults(_vault_handler=_cmd_list)

    p_rm = subs.add_parser("rm", help="Remove a vault item by handle")
    p_rm.add_argument("handle", help="Item handle (see `hermes vault list`)")
    p_rm.set_defaults(_vault_handler=_cmd_rm)

    p_migrate = subs.add_parser(
        "migrate-local",
        help="Copy local login items into the configured external write backend",
    )
    p_migrate.add_argument(
        "--execute",
        action="store_true",
        help="Perform the copy (without this flag, show a dry-run plan)",
    )
    p_migrate.set_defaults(_vault_handler=_cmd_migrate_local)

    p_src = subs.add_parser("sources", help="Show detected password managers (1Password, Bitwarden); they are on automatically")
    group = p_src.add_mutually_exclusive_group()
    group.add_argument("--disable", metavar="NAME", help="Stop using a detected manager: onepassword | bitwarden")
    group.add_argument("--enable", metavar="NAME", help="Undo --disable")
    group.add_argument("--write", metavar="NAME", help="Save new/changed logins here: local | bitwarden")
    p_src.set_defaults(_vault_handler=_cmd_sources)


def vault_command(args) -> None:
    from agent.vault_store import VaultError

    handler = getattr(args, "_vault_handler", None)
    try:
        if handler is None:
            _cmd_list(args)
            return
        handler(args)
    except VaultError as exc:
        _console().print(f"[red]Error:[/] {exc}")
