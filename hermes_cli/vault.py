"""``hermes vault`` — manage the local encrypted autofill vault.

Subcommands:
- ``hermes vault add``   interactive wizard; the password is read via
  getpass (never echoed, never accepted as argv). The login identifier is
  visible metadata and prompted normally.
- ``hermes vault list``  metadata — labels, kinds, identifiers, origins,
  handles. Passwords are never shown.
- ``hermes vault rm``    remove an item by handle/id.
- ``hermes vault key``   where the vault key lives; ``key migrate --to keychain|file`` moves it.

The vault backs the password-blind browser autofill tools
(``browser_vault`` list / fill): the agent sees handles
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
        VaultError,
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

    try:
        if kind == "login":
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
            otp_secret = getpass.getpass(
                "Authenticator key (optional, hidden; the 2FA \"setup key\" or otpauth:// link — Enter to skip): ")
            # identifier_type/identifier are stored as metadata (not secret);
            # add_item moves them out of the encrypted payload.
            secret = {
                "identifier_type": id_type,
                "identifier": identifier,
                "password": password,
                **({"otp_secret": otp_secret} if otp_secret.strip() else {}),
            }
            meta = get_vault_store().add_item(
                kind="login", label=label, secret=secret, origin=origin
            )
        else:
            from agent.vault_store import ADDRESS_FIELDS, PAYMENT_FIELDS, REQUIRED_FIELDS

            fields = PAYMENT_FIELDS if kind == "payment" else ADDRESS_FIELDS
            origin = ""
            while not origin:
                origin = input("Site origin the item may be filled on (e.g. https://shop.example.com): ").strip()
            c.print(f"[dim]{kind} fields are filled only on that origin; card values are read hidden.[/]")
            secret = {}
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
    except VaultError as exc:
        c.print(f"[red]Error:[/] {exc}")
        return

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
    """Show the detected password managers; `--disable`/`--enable` flip the opt-out (`vault.<name>.enabled`)."""
    from agent.vault_backends import enabled_backends
    from agent.vault_backends.base import external_backend_classes, is_installed
    from hermes_cli.config import _ensure_dict, load_config, save_config

    c = _console()
    classes = {cls.name: cls for cls in external_backend_classes()}
    if args.enable or args.disable:
        name = args.enable or args.disable
        if name not in classes:
            c.print(f"[red]Unknown password manager {name!r}[/] (expected one of {', '.join(classes)})")
            return
        cfg = load_config()
        section = _ensure_dict(_ensure_dict(cfg, "vault"), name)
        if args.enable:
            section.pop("enabled", None)  # detected managers are on by default; drop the opt-out
        else:
            section["enabled"] = False
        save_config(cfg)
        c.print(f"[green]{classes[name].display_name} {'on' if args.enable else 'off'}[/] for browser logins.")
        return
    enabled = {b.name for b in enabled_backends()}
    for name, cls in classes.items():
        if name in enabled:
            status = "[green]detected[/] · the agent asks you to unlock it when it needs a login"
        elif is_installed(name):
            status = f"[dim]turned off[/] (`hermes vault sources --enable {name}` to use it)"
        else:
            status = "[dim]not installed[/]"
        c.print(f"  {cls.display_name:<10} {status}")
    c.print("[dim]Managers are picked up automatically when their CLI is installed and signed in.[/]")


def _cmd_rm(args) -> None:
    from agent.vault_store import get_vault_store

    c = _console()
    if get_vault_store().remove_item(args.handle):
        c.print(f"[green]Removed[/] {args.handle}")
    else:
        c.print(f"[red]No vault item with handle {args.handle!r}[/]")


def _cmd_key(args) -> None:
    """Show where the local vault's key lives, or move it between vault.key and the OS keychain."""
    from agent import vault_keychain
    from agent.vault_store import configured_key_storage, get_vault_store

    c = _console()
    store = get_vault_store()
    target = getattr(args, "to", None)
    if target:
        if target == "keychain" and not _ensure_keychain_extra(c):
            return
        store.migrate_key(target)
        _set_key_storage(target)
        if store.key_location()["storage"] == "none":  # nothing was moved or verified, only the setting
            c.print(f"No vault key exists yet. The first saved item creates one in "
                    f"{'the OS keychain' if target == 'keychain' else 'vault.key'} (vault.key_storage = {target}).")
        elif target == "keychain":
            c.print(f"[green]Vault key moved to the OS keychain[/] ({vault_keychain.backend_label()}); "
                    "vault.key was removed after the keychain copy decrypted the vault.")
            c.print("[dim]Backups of this profile now hold the encrypted vault without its key; "
                    "`hermes vault key migrate --to file` before moving the profile to another machine.[/]")
        else:
            where = store.key_location()
            c.print(f"[green]Vault key moved to[/] {where.get('path')} (0600).")
            if where.get("keychain_entry_left"):
                c.print(f"[yellow]The keychain copy ({where['keychain_entry_left']}) could not be removed yet;[/] "
                        "run this command again when the keychain is reachable.")
            else:
                c.print("Keychain entry removed.")
        return
    where = store.key_location()
    if where["storage"] == "file":
        c.print(f"Vault key: [bold]file[/] {where['path']} (0600, beside the encrypted vault)")
    elif where["storage"] == "keychain":
        c.print(f"Vault key: [bold]OS keychain[/] service {where.get('service')!r}, account {where['account']!r}")
    else:
        c.print("Vault key: [dim]none yet[/] (created with the first saved item)")
    c.print(f"New vaults use: vault.key_storage = {configured_key_storage()}")
    c.print(f"OS keychain here: {vault_keychain.backend_label()}")


def _ensure_keychain_extra(c) -> bool:
    from pm.extras import available, ensure_import
    from pm.package import InstallError

    if available("keychain"):
        return True
    try:
        ensure_import("keychain")
    except InstallError as exc:
        c.print(f"[red]The keychain feature is not installed:[/] {exc}")
        return False
    return True


def _set_key_storage(value: str) -> None:
    from hermes_cli.config import _ensure_dict, load_config, save_config

    cfg = load_config()
    _ensure_dict(cfg, "vault")["key_storage"] = value
    save_config(cfg)


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

    p_src = subs.add_parser("sources", help="Show detected password managers (1Password, Bitwarden); they are on automatically")
    group = p_src.add_mutually_exclusive_group()
    group.add_argument("--disable", metavar="NAME", help="Stop using a detected manager: onepassword | bitwarden")
    group.add_argument("--enable", metavar="NAME", help="Undo --disable")
    p_src.set_defaults(_vault_handler=_cmd_sources)

    p_key = subs.add_parser("key", help="Show where the local vault key is stored (vault.key file or OS keychain)")
    p_key.set_defaults(_vault_handler=_cmd_key, to=None)
    key_subs = p_key.add_subparsers(dest="vault_key_action")
    p_migrate = key_subs.add_parser(
        "migrate", help="Move the vault key to the OS keychain or back to vault.key (verified before the old copy is removed)")
    p_migrate.add_argument("--to", choices=["keychain", "file"], required=True, help="Destination for the key")
    p_migrate.set_defaults(_vault_handler=_cmd_key)


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
