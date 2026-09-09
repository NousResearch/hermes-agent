"""Shared recipe CLI/slash boundary; validation precedes all writable storage."""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import tempfile

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_recipes import RecipeError, canonical, load_json, prepare_definition


_EXIT_CODES = {
    'RECIPE_INVALID': 2, 'BINDING_INVALID': 2, 'OUTPUT_EXISTS': 2,
    'INSTANCE_NOT_FOUND': 3, 'IDEMPOTENCY_CONFLICT': 3, 'INSTANCE_DAMAGED': 3,
    'IMPORTED_INSTANCE_REBIND_REQUIRED': 3,
    'STORAGE_UNAVAILABLE': 4, 'COMMIT_UNCERTAIN': 4,
    'BOARD_CONTEXT_CONFLICT': 5, 'AUTHORITY_DENIED': 5,
}


def _resolve_board(explicit):
    """Do not let a worker DB pin silently override metadata/project routing."""
    context = kb._CURRENT_BOARD_OVERRIDE.get()
    environment = os.environ.get('HERMES_KANBAN_BOARD')
    candidates = [value for value in (explicit, context, environment) if value is not None]
    normalized = []
    for value in candidates:
        try:
            slug = kb._normalize_board_slug(value)
        except ValueError:
            raise RecipeError('BINDING_INVALID', '/board', 'Invalid board identity') from None
        if not slug:
            raise RecipeError('BINDING_INVALID', '/board', 'Board identity is required')
        normalized.append(slug)
    if len(set(normalized)) > 1:
        raise RecipeError('BOARD_CONTEXT_CONFLICT', '/board', 'Board selections disagree')
    board = normalized[0] if normalized else kb.get_current_board()
    # Compute the canonical path without consulting the worker override again.
    expected = (kb.kanban_home() / 'kanban.db' if board == kb.DEFAULT_BOARD
                else kb.board_dir(board) / 'kanban.db').resolve()
    pin = os.environ.get('HERMES_KANBAN_DB', '').strip()
    if pin and Path(pin).expanduser().resolve() != expected:
        raise RecipeError('BOARD_CONTEXT_CONFLICT', '/board', 'Board and database selections disagree')
    if not kb.board_exists(board):
        raise RecipeError('BINDING_INVALID', '/board', 'Selected board does not exist')
    return board, expected


@contextlib.contextmanager
def _readonly(path):
    # Native connect performs WAL setup, integrity repair and migration; never use it here.
    conn = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _stored_key(path, key, prepared, bindings):
    # Reuse the store's replay validator in this read-only preflight. The write
    # transaction repeats it authoritatively; this only prevents invalid requests
    # from reaching migration, including damaged/imported historical invocations.
    from hermes_cli.kanban_recipes_store import _replay

    if not path.exists():
        return False
    with _readonly(path) as conn:
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='recipe_instances'").fetchone():
            return False
        row = conn.execute('SELECT * FROM recipe_instances WHERE idempotency_key=?', (key,)).fetchone()
        if row is None:
            return False
        _replay(conn, row, prepared, bindings)
        return True


def _prepare(args):
    from hermes_cli.kanban_recipes_bindings import normalize_bindings

    definition = load_json(args.definition)
    inputs = load_json(args.inputs) if args.inputs is not None else {}
    prepared = prepare_definition(definition, inputs)
    try:
        bindings = load_json(args.bindings) if args.bindings is not None else {}
    except RecipeError as exc:
        raise RecipeError('BINDING_INVALID', exc.path, exc.message) from None
    return prepared, normalize_bindings(prepared, bindings)


def _validate(args, board, path):
    from hermes_cli.kanban_recipes_bindings import resolve_plan

    prepared, bindings = _prepare(args)
    plan = resolve_plan(prepared, bindings, board)
    return {'valid': True, 'definition_digest': prepared['definition_digest'],
            'effective_plan': plan}


def _run(args, board, path):
    from hermes_cli.kanban_recipes_bindings import resolve_plan
    from hermes_cli.kanban_recipes_store import instantiate
    from hermes_cli.kanban import _profile_author

    if type(args.key) is not str or re.fullmatch(r'[!-~]{1,128}', args.key) is None:
        raise RecipeError('RECIPE_INVALID', '/key', 'Key must contain 1-128 printable non-whitespace ASCII characters')
    prepared, bindings = _prepare(args)
    # Replay resolves no current defaults or installed-profile prerequisites. The store
    # repeats the definitive lookup under its write lock, including conflict/damage checks.
    if not _stored_key(path, args.key, prepared, bindings):
        prepared = dict(prepared, effective_plan=resolve_plan(prepared, bindings, board))
    kb.init_db(path, board=board)
    try:
        with kbc.connect_closing(path, board=board) as conn:
            return instantiate(conn, prepared, bindings, args.key, board=board,
                               created_by=_profile_author(),
                               creator_task_id=os.environ.get('HERMES_KANBAN_TASK'))
    except sqlite3.Error:
        # A native post-COMMIT invariant can raise after durability. Reopen,
        # rather than treating a failed response as proof of rollback.
        try:
            committed = _stored_key(path, args.key, prepared, bindings)
        except (sqlite3.Error, OSError, RecipeError):
            committed = True  # Storage cannot establish the outcome.
        if committed:
            raise RecipeError('COMMIT_UNCERTAIN', '/key',
                              'Outcome uncertain; inspect or replay the same key', True) from None
        raise


def _show(args, board, path):
    from hermes_cli.kanban_recipes_store import show_instance

    if re.fullmatch(r'ri_[0-9a-f]{32}', args.instance_id) is None:
        raise RecipeError('RECIPE_INVALID', '/instance_id', 'Invalid instance identifier')
    if not path.exists():
        raise RecipeError('INSTANCE_NOT_FOUND', '/instance_id', 'Recipe instance does not exist')
    with _readonly(path) as conn:
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='recipe_instances'").fetchone():
            raise RecipeError('INSTANCE_NOT_FOUND', '/instance_id', 'Recipe instance does not exist')
        return show_instance(conn, args.instance_id)


def _export(args, board, path):
    shown = _show(args, board, path)
    output = Path(args.output).expanduser()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='',
                                         dir=output.parent, prefix='.' + output.name + '.',
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(canonical(shown['definition']))
            stream.flush()
            os.fsync(stream.fileno())
        if args.overwrite:
            os.replace(temporary, output)
        else:
            # Atomic publish-if-absent: exists()+replace would clobber a racing writer.
            os.link(temporary, output)
    except FileExistsError:
        raise RecipeError('OUTPUT_EXISTS', '/output', 'Output exists; explicit overwrite is required') from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {'exported': True, 'definition_digest': shown['definition_digest']}


_HANDLERS = {'validate': _validate, 'run': _run, 'show': _show, 'export': _export}


def _error(args, error):
    if getattr(args, 'json', False):
        print(canonical(error.as_dict()))
    else:
        print(f'{error.code} at {error.path or "/"}: {error.message}', file=sys.stderr)
    return _EXIT_CODES.get(error.code, 2)


def recipe_command(args):
    """Translate native failures without exposing input values or exception details."""
    action = getattr(args, 'recipe_action', None)
    try:
        if action not in _HANDLERS:
            raise RecipeError('RECIPE_INVALID', '', 'A recipe subcommand is required')
        if action in ('run', 'export'):
            from agent.delegation_context import is_delegated_child_process_context
            if is_delegated_child_process_context():
                raise RecipeError('AUTHORITY_DENIED', '', 'Delegated children cannot mutate recipes')
        board, path = _resolve_board(getattr(args, 'board', None))
        with kb.scoped_current_board(board):
            result = _HANDLERS[action](args, board, path)
        print(json.dumps(result, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
              if getattr(args, 'json', False)
              else json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except RecipeError as exc:
        return _error(args, exc)
    except PermissionError:
        return _error(args, RecipeError('AUTHORITY_DENIED', '', 'Operation is not authorized'))
    except (sqlite3.Error, OSError):
        return _error(args, RecipeError('STORAGE_UNAVAILABLE', '', 'Storage unavailable; inspect or replay the same key before retrying', True))
    except RuntimeError:
        return _error(args, RecipeError('COMMIT_UNCERTAIN', '', 'Outcome uncertain; inspect or replay the same key before retrying', True))
    except (ValueError, TypeError):
        return _error(args, RecipeError('BINDING_INVALID', '', 'Invalid native task or binding configuration'))
