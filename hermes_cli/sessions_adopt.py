"""``hermes sessions adopt``: bring this profile's pre-gateway history under its gateway, as the host.

Since #106742 the profile's gateway owns its sessions. Chats written before that (main's CLI, TUI and
Desktop rows) carry no gateway receipt until one native, same-OS-user client resumes them, which
adopts them (``gateway/session_local_migration.py::adopt_legacy_session``). A Webapp browser reaches
the gateway through its dashboard mount as the login it authenticated, never natively, so it may not
make that first claim. Run this once on the host instead: it is what opening each old chat in the
native TUI or Desktop would do, and afterwards the browser continues every adopted chat through
shared operator access.
"""
from __future__ import annotations

from collections import Counter
import sqlite3
import sys

_PROGRESS_EVERY = 25


def adoptable_sessions(path) -> list[dict]:
    """This store's unadopted pre-gateway local chats: one compression tip each, oldest first.

    The same eligibility the owner applies on resume (``validate_legacy_row``), read-only, so the
    count a dry run reports is what an adoption run attempts. A row the owner already bound is not
    offered again, which makes a second run a no-op."""
    from hermes_state import SessionDB
    from hermes_state_local import POLICY_PREFIX
    from hermes_state_local_migration import LEGACY_LOCAL_SOURCES, LEGACY_PREFIX, validate_legacy_row
    from hermes_state_runtime import RuntimeStoreError

    db = SessionDB(db_path=path, read_only=True)
    try:
        sources = tuple(sorted(LEGACY_LOCAL_SOURCES))
        with db._read_ctx() as conn:
            ids = [row[0] for row in conn.execute(
                f"SELECT id FROM sessions WHERE source IN ({','.join('?' * len(sources))}) ORDER BY started_at, id",
                sources)]
            bound = {row[0] for row in conn.execute(
                'SELECT key FROM state_meta WHERE key LIKE ? OR key LIKE ?', (POLICY_PREFIX + '%', LEGACY_PREFIX + '%'))}
        found, seen = [], set()
        for sid in ids:
            tip_id = db.get_compression_tip(sid) or sid
            if tip_id in seen:
                continue
            seen.add(tip_id)
            tip = db.get_session(tip_id)
            key = tip and (tip.get('chat_id') or tip_id)
            if tip is None or POLICY_PREFIX + key in bound or LEGACY_PREFIX + key in bound:
                continue
            try:
                validate_legacy_row(db, tip)
            except RuntimeStoreError:
                continue
            found.append(tip)
        return found
    finally:
        db.close()


async def adopt_sessions(rows, *, report=print) -> tuple[int, Counter]:
    """Resume each row once over this host's native connection (the owner adopts it), then detach
    without ending it. ``(adopted, skipped-by-refusal)``; a row the owner refuses is skipped,
    never retried as another principal."""
    from hermes_cli.gateway_client import GatewayRPCError, connect_gateway
    from hermes_cli.web_server_boundary import _DESKTOP_ATTACHMENT_WS_MAX_BYTES

    adopted, skipped = 0, Counter()
    # The owner answers a resume with the session's whole history; a long chat outgrows the default frame.
    async with connect_gateway(max_size=_DESKTOP_ATTACHMENT_WS_MAX_BYTES) as client:
        for done, row in enumerate(rows, 1):
            try:
                snapshot = await client.rpc('session.resume', session_id=row['id'])
            except GatewayRPCError as exc:
                skipped[_refusal(exc)] += 1
                continue
            await client.rpc('session.detach', session_id=snapshot.get('session_id') or row['id'],
                             subscription_id=snapshot['subscription_id'])
            adopted += 1
            if done % _PROGRESS_EVERY == 0:
                report(f'  {done}/{len(rows)} …')
    return adopted, skipped


_SKIP_HINTS = {
    'launch_credentials_unavailable': 'its saved model route is gone; open it once and pick a model',
    'not_found': 'not adoptable here (another identity or profile owns it)',
}


def _refusal(exc) -> str:
    """One owner refusal as the report prints it: the reason code, then why. A field-level refusal
    (a launch directory that no longer exists) names the field and the owner's bounded detail."""
    reason, field, detail = str(exc), exc.data.get('field'), exc.data.get('detail')
    why = _SKIP_HINTS.get(reason) or (f'{field}: {detail}' if field and detail else 'refused by the gateway')
    return f'{reason} ({why})'


def _unreadable_store(exc: sqlite3.Error, store: str) -> str:
    """One line for a store adopt cannot read: what is wrong and what to run. Adopt opens the store
    read-only, so it neither upgrades a store from an older release nor waits out another writer."""
    from hermes_state_errors import is_sqlite_lock_error
    from hermes_state_repair import _schema_not_built

    if is_sqlite_lock_error(exc):
        return f'Error: {store} is locked by another program. Close that program, then run this command again.'
    if _schema_not_built(exc):
        return (f'Error: {store} is from an older Hermes and has not been upgraded ({exc}). Run '
                '`hermes sessions repair` to upgrade it (it makes a backup first), then run this command again.')
    return f'Error: could not read {store} ({exc}). Run `hermes sessions repair` to check it.'


def cmd_adopt(args) -> int:
    import asyncio
    from websockets.exceptions import WebSocketException
    from hermes_cli.gateway_client import GatewayClientError
    from hermes_constants import display_hermes_home, get_hermes_home

    path = get_hermes_home() / 'state.db'
    if not path.exists():
        print(f'No session database at {display_hermes_home()}/state.db (nothing to adopt).')
        return 0
    try:
        rows = adoptable_sessions(path)
    except sqlite3.Error as exc:
        print(_unreadable_store(exc, f'{display_hermes_home()}/state.db'), file=sys.stderr)
        return 1
    if not rows:
        print('Nothing to adopt: every local chat in this profile already belongs to its gateway.')
        return 0
    counts = Counter(row['source'] for row in rows)
    print(f"{len(rows)} pre-gateway chat(s) to adopt ({', '.join(f'{n} {s}' for s, n in sorted(counts.items()))}).")
    if args.dry_run:
        print('Dry run: nothing was adopted.')
        return 0
    try:
        adopted, skipped = asyncio.run(adopt_sessions(rows))
    except (GatewayClientError, OSError, TimeoutError, WebSocketException) as exc:
        message = str(exc) if isinstance(exc, GatewayClientError) else 'Gateway connection/read failed'
        print('Error: ' + message, file=sys.stderr)
        return 1
    print(f'Adopted {adopted} chat(s).')
    for refusal, count in sorted(skipped.items()):
        print(f'Skipped {count}: {refusal}.')
    return 0


def add_adopt_parser(sessions_subparsers) -> None:
    parser = sessions_subparsers.add_parser(
        'adopt', help="Bring this profile's pre-gateway chats under its gateway (run on the host)",
        description="Chats written before the gateway owned sessions become gateway sessions the first "
            "time a native client on this host opens them. This adopts them all at once, so remote "
            "clients such as the Webapp can continue them. Run it as the gateway's OS user; use "
            "`hermes -p <profile> sessions adopt` for another profile.")
    parser.add_argument('--dry-run', action='store_true', help='Only count what would be adopted')
