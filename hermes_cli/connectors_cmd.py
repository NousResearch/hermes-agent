"""``hermes connectors``: list, connect, rename and remove hosted connector accounts."""

from __future__ import annotations

import re
import sys
import time
from typing import Any, Callable, NoReturn

_POLL_SECONDS = 2.0
_DONE_STATES = frozenset({"connected", "skipped", "failed", "expired", "not_connected"})
_SIGN_IN = "Sign in with `hermes login` to manage connectors."


class _CommandError(Exception):
    """A user-facing failure: the message is printed and the command exits 1."""


def _fail(message: str) -> NoReturn:
    raise _CommandError(message)


def _valid_alias(alias: str) -> str:
    from tools.connectors.targets import ALIAS_PATTERN

    if not re.match(ALIAS_PATTERN, alias):
        _fail(f"Invalid name {alias!r}. Use 1 to 32 lowercase letters, digits or hyphens, "
              "starting with a letter or digit.")
    return alias


def _account_name(row: dict[str, Any]) -> str:
    return str(row.get("alias") or row.get("label") or row.get("connectionId") or "")


def _display_name(row: dict[str, Any]) -> str:
    return row["alias"] if row.get("alias") else f"{row.get('label') or row.get('connectionId')} (unnamed)"


def _app_accounts(app: str) -> list[dict[str, Any]]:
    from tools.connectors.portal.client import PortalConnectorClient

    return [r for r in PortalConnectorClient().list_accounts() if str(r.get("connector", "")).lower() == app]


def _names_hint(app: str, rows: list[dict[str, Any]]) -> str:
    names = sorted({_account_name(r) for r in rows})
    return f"Its accounts are: {', '.join(names)}." if names else f"You have no {app} account."


def _match_named(app: str, rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    matches = [r for r in rows if r.get("alias") == name] or [
        r for r in rows if not r.get("alias") and r.get("label") == name]
    if len(matches) != 1:
        what = "No" if not matches else "More than one"
        _fail(f"{what} {app} account is called {name!r}. {_names_hint(app, rows)}")
    return matches[0]


def _list(args) -> int:
    from tools.connectors.managed import managed_client

    rows = managed_client().list_connectors()
    if not rows:
        print("No connectors are available.")
        return 0
    width = max(len(str(r.get("connector", ""))) for r in rows)
    for row in sorted(rows, key=lambda r: str(r.get("connector", ""))):
        state = "connected" if row.get("connected") else "not connected"
        print(f"{str(row.get('connector', '')).ljust(width)}  {state}")
    return 0


def _accounts(args) -> int:
    from tools.connectors.portal.client import PortalConnectorClient

    app = (args.app or "").strip().lower()
    rows = [r for r in PortalConnectorClient().list_accounts()
            if not app or str(r.get("connector", "")).lower() == app]
    retired = [r for r in rows if r.get("disabled")]
    shown = rows if args.all else [r for r in rows if not r.get("disabled")]
    if not shown and not retired:
        print(f"You have no {app} account." if app else "You have no connected accounts.")
        return 0
    table = [(str(r.get("connector", "")), _display_name(r), str(r.get("status", "")) + (" (retired)" if r.get("disabled") else ""))
             for r in sorted(shown, key=lambda r: (str(r.get("connector", "")), _account_name(r)))]
    if table:
        app_width = max(len(row[0]) for row in table)
        name_width = max(len(row[1]) for row in table)
        for connector, name, status in table:
            print(f"{connector.ljust(app_width)}  {name.ljust(name_width)}  {status}")
    if retired and not args.all:
        print(f"{len(retired)} retired (use --all to list)")
    return 0


def _wait_for_link(start, app: str) -> dict[str, Any]:
    """Poll the operation until its one target settles; returns the last target snapshot."""
    printed_url = False
    while True:
        snapshot = start.operation.snapshot()
        target = snapshot["targets"][0]
        url = target.get("connect_url")
        if url and not printed_url:
            print(url)
            print("Open this link in your browser to finish.")
            printed_url = True
        if target.get("state") in _DONE_STATES or snapshot.get("settled_at") is not None:
            return target
        if time.time() >= start.operation.deadline_at:
            return target
        time.sleep(_POLL_SECONDS)


def _report_connect(app: str, alias: str | None, target: dict[str, Any]) -> int:
    from tools.tool_labels import app_title

    state = target.get("state")
    if state == "connected":
        if alias:
            print(f"{app_title(app)} account {alias} is connected. "
                  f"Rename it anytime: hermes connectors rename {app} {alias} --to <new-name>")
        else:
            print(f"{app_title(app)} is connected.")
        return 0
    if state == "skipped":
        print(f"Skipped connecting {app_title(app)}.")
        return 0
    detail = target.get("detail") or f"{app_title(app)} did not finish connecting."
    print(detail, file=sys.stderr)
    return 1


def _connect(args) -> int:
    from tools.connectors import account, live
    from tools.connectors.contract import SettleReason

    app = args.app.strip().lower()
    alias = _valid_alias(args.alias) if args.alias else None
    start = account.find_or_start_operation(
        [app], action="reconnect" if args.reconnect else "connect", profile_home=None, alias=alias)
    try:
        account.wait_for_prepare(start)
        if start.failed:
            _fail(f"Could not start connecting {app}. Try again in a moment.")
        target = _wait_for_link(start, app)
    except KeyboardInterrupt:
        # The operation would otherwise stay open and block the next connect of this app.
        start.operation.settle(SettleReason.continue_)
        live.close(start.operation)
        print("\nStopped waiting.")
        return 130
    return _report_connect(app, alias, target)


def _rename(args) -> int:
    from tools.connectors.gateway.errors import GatewayUnavailable, IdempotencyConflict, ToolGatewayError
    from tools.connectors.portal.client import PortalConnectorClient

    app = args.app.strip().lower()
    new = _valid_alias(args.to)
    row = _match_named(app, _app_accounts(app), args.name)
    try:
        PortalConnectorClient().rename_account(row["connectionId"], new)
    except IdempotencyConflict:
        _fail(f"The name {new} is already used by another {app} account.")
    except GatewayUnavailable as exc:
        if exc.code == "connection_not_found":
            _fail("That account no longer exists.")
        _fail("This Nous Portal cannot rename connector accounts yet.")
    except ToolGatewayError as exc:
        if exc.status == 405:
            _fail("This Nous Portal cannot rename connector accounts yet.")
        raise
    print(f"Renamed {app} account {args.name} to {new}.")
    return 0


def _pick_for_disconnect(app: str, alias: str | None) -> dict[str, Any]:
    rows = _app_accounts(app)
    if alias:
        return _match_named(app, rows, alias)
    live_rows = [r for r in rows if not r.get("disabled")]
    if len(live_rows) == 1:
        return live_rows[0]
    if not live_rows:
        _fail(f"You have no {app} account.")
    _fail(f"You have more than one {app} account. Pass --alias to pick one. {_names_hint(app, live_rows)}")


def _disconnect(args) -> int:
    from tools.connectors.portal.client import PortalConnectorClient

    app = args.app.strip().lower()
    row = _pick_for_disconnect(app, args.alias)
    name = _account_name(row)
    if not args.yes:
        if not sys.stdin.isatty():
            _fail("Refusing to remove an account without confirmation. Pass --yes to confirm.")
        answer = input(f"Remove {app} account {name}? This signs Hermes out of it. [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("Nothing removed.")
            return 0
    PortalConnectorClient().delete_account(row["connectionId"])
    print(f"Removed {app} account {name}.")
    return 0


_ACTIONS: dict[str, Callable[[Any], int]] = {
    "list": _list,
    "ls": _list,
    "accounts": _accounts,
    "connect": _connect,
    "rename": _rename,
    "disconnect": _disconnect,
}


def cmd_connectors(args) -> int:
    from tools.connectors import connectors_available
    from tools.connectors.gateway.errors import GatewayAuthError, ToolGatewayError

    handler = _ACTIONS.get(getattr(args, "connectors_action", None) or "list")
    if handler is None:
        print(f"Unknown connectors action: {args.connectors_action}", file=sys.stderr)
        return 2
    if not connectors_available():
        print("Connectors are not available. Sign in with `hermes login` first.", file=sys.stderr)
        return 1
    try:
        return handler(args)
    except _CommandError as exc:
        print(str(exc), file=sys.stderr)
    except GatewayAuthError:
        print(_SIGN_IN, file=sys.stderr)
    except ToolGatewayError as exc:
        print(str(exc), file=sys.stderr)
    return 1
