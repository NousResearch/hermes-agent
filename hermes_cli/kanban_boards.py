"""``hermes kanban boards …`` — board directories, the ``current`` pointer and ``board.json``.
Filesystem-only, so every action works before ``kanban init`` and must ignore the shared
``--board`` task-routing override.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_generation import GenerationError, canonical_json, encode_reader_frame
from hermes_cli.kanban_output import _err, _fmt_counts, _json_out

_READER_CONSUMERS = frozenset({"hermes-office"})
_READER_PUBLIC_FIELDS = (
    "database_path", "storage_mode", "generation_id", "generation_nonce",
    "ready_sha256", "activated_sha256", "lease_id", "lease_keeper_pid",
    "lease_keeper_start_fingerprint", "consumer_pid", "consumer_start_fingerprint",
    "selected_at",
)
_READER_CAPABILITY_FIELDS = (
    "database_path", "storage_mode", "generation_id", "generation_nonce",
    "ready_sha256", "activated_sha256", "lease_id",
)
_READER_ERROR_CODES = frozenset({
    "UNSUPPORTED_PROTOCOL", "STATUS_ID_CONFLICT", "BOARD_NOT_FOUND",
    "POINTER_INVALID", "READY_INVALID", "ACTIVATION_INVALID",
    "GENERATION_META_INVALID", "GENERATION_NOT_ACTIVATED",
    "ACTIVATION_BUSY", "CATALOG_BUSY", "LEASE_CREATE_FAILED",
    "LEASE_DURABILITY_FAILED", "LEASE_REVALIDATION_FAILED", "LEASE_LOST",
    "LEASE_ID_MISMATCH", "LEASE_RELEASE_FAILED", "CHILD_START_FAILED",
    "CHILD_HANDSHAKE_FAILED", "STATUS_PUBLICATION_FAILED",
    "CHILD_CAPABILITY_FAILED", "CHILD_FAILED",
})
_READER_RETRYABLE_CODES = frozenset({
    "GENERATION_NOT_ACTIVATED", "ACTIVATION_BUSY", "CATALOG_BUSY",
    "LEASE_CREATE_FAILED", "LEASE_DURABILITY_FAILED", "LEASE_REVALIDATION_FAILED",
})
_READER_ERROR_ALIASES = {
    "ACTIVATION_POISONED": "ACTIVATION_INVALID",
    "PUBLICATION_POISONED": "POINTER_INVALID",
    "BOARD_NOT_GENERATED": "BOARD_NOT_FOUND",
}
_READER_HANDSHAKE_TIMEOUT_SECONDS = 10.0
_READER_MAX_FRAME_BYTES = 16 * 1024 * 1024


def _reader_public_result(result: dict) -> dict:
    return {key: result.get(key) for key in _READER_PUBLIC_FIELDS}


def _reader_error_code(exc: BaseException, default: str) -> str:
    code = _READER_ERROR_ALIASES.get(str(exc), str(exc))
    return code if code in _READER_ERROR_CODES else default


def _reader_is_retryable(code: str) -> bool:
    return code in _READER_RETRYABLE_CODES


def _reader_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _dispatch_boards(args: argparse.Namespace) -> int:
    """``hermes kanban boards <action>`` — filesystem-only, so it works before ``kanban init``."""
    sub = getattr(args, "boards_action", None) or "list"
    handler = _BOARD_HANDLERS.get(sub)
    if handler is None:
        return _err(f"kanban boards: unknown action {sub!r}", 2)
    from agent.delegation_context import is_delegated_child_process_context
    if is_delegated_child_process_context() and sub in {
        "reconcile", "publish-prepared-generation", "rollback-generation", "gc-generations",
        "create", "new", "rm", "remove", "delete", "switch", "use", "rename",
        "set-default-workdir", "import",
    }:
        return _err("delegated child contexts cannot mutate Kanban boards", 1)
    return handler(args)


def _cmd_reconcile(args: argparse.Namespace) -> int:
    from hermes_cli.kanban_reconcile import reconcile
    from hermes_cli.kanban_generation import GenerationError
    try:
        result = reconcile(Path(args.source_db), Path(args.target_db), Path(args.seed_file),
                           Path(args.output_generation), apply=args.apply)
    except (OSError, sqlite3.Error, GenerationError, ValueError) as exc:
        return _err(f"kanban boards reconcile: {exc}")
    if _json_out(args, result):
        return 0
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_verify_prepared(args: argparse.Namespace) -> int:
    from hermes_cli.kanban_generation import validate_generation
    try:
        ready = validate_generation(Path(args.path))
    except (OSError, ValueError, sqlite3.Error, GenerationError) as exc:
        return _err(f"kanban boards verify-prepared-generation: {exc}")
    result = {"valid": True, "generation_id": ready["generation_id"], "manifest_digest": ready["manifest_digest"]}
    if _json_out(args, result):
        return 0
    print(f"Prepared generation {result['generation_id']} is valid.")
    return 0


def _cmd_publish_prepared(args: argparse.Namespace) -> int:
    from hermes_cli.kanban_generation import publish_prepared
    try:
        result = publish_prepared(Path(args.path), expected_target_manifest=args.expected_target_manifest,
                                  fence_token=args.fence_token)
    except (OSError, ValueError, sqlite3.Error, GenerationError) as exc:
        return _err(f"kanban boards publish-prepared-generation: {exc}")
    if _json_out(args, result):
        return 0
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_gc_generations(args: argparse.Namespace) -> int:
    from hermes_cli import kanban_db as db
    from hermes_cli.kanban_generation import gc_generations
    try:
        result = gc_generations(db.board_dir(db.get_current_board()), dry_run=args.dry_run)
    except (OSError, ValueError, sqlite3.Error, GenerationError) as exc:
        return _err(f"kanban boards gc-generations: {exc}")
    if _json_out(args, result):
        return 0
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_rollback_generation(args: argparse.Namespace) -> int:
    from hermes_cli.kanban_generation import rollback_pre_activation
    try:
        rollback_pre_activation(kb.board_dir(kb.get_current_board()), expected_active_manifest=args.expected_active_manifest, fence_token=args.fence_token)
    except (OSError, ValueError, sqlite3.Error, GenerationError) as exc:
        return _err(f"kanban boards rollback-generation: {exc}")
    return 0


def _reader_error(operation_id: str, code: str, board: str, message: str | None = None,
                  *, generation_id: str | None = None, retryable: bool = False) -> int:
    import sys
    value = {"protocol_version": 1, "type": "error", "operation_id": operation_id,
             "board_slug": board, "generation_id": generation_id, "code": code,
             "retryable": retryable, "message": message or code}
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()
    return 1


def _reader_write_json(value: dict) -> None:
    import sys
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


def _read_pipe_line_deadline(stream, timeout: float) -> bytes:
    """Read one child line without allowing a silent child to pin the parent."""
    import selectors
    fd = stream.fileno()
    deadline = time.monotonic() + timeout
    line = bytearray()
    with selectors.DefaultSelector() as selector:
        selector.register(fd, selectors.EVENT_READ)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not selector.select(remaining):
                raise TimeoutError("reader handshake timed out")
            chunk = os.read(fd, 1)
            if not chunk:
                return bytes(line)
            line.extend(chunk)
            if chunk == b"\n":
                return bytes(line)


def _publish_status(directory: Path, name: str, value: dict) -> None:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    os.chmod(directory, 0o700)
    payload = canonical_json(value) + b"\n"
    destination = directory / name
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _cmd_reader_session(args: argparse.Namespace) -> int:
    if args.protocol_version != 1:
        return _reader_error(args.operation_id, "UNSUPPORTED_PROTOCOL", args.board)
    if args.consumer not in _READER_CONSUMERS:
        return _reader_error(args.operation_id, "LEASE_CREATE_FAILED", args.board,
                             message="unsupported reader consumer")
    from hermes_cli.kanban_generation import acquire_reader_lease
    board_dir = kb.board_dir(args.board)
    try:
        if not (board_dir / "generation.current").exists():
            legacy_path = kb.kanban_db_path(board=args.board)
            if not legacy_path.is_file():
                return _reader_error(args.operation_id, "BOARD_NOT_FOUND", args.board)
            result = {"database_path": str(legacy_path), "storage_mode": "legacy", "generation_id": None, "generation_nonce": None, "ready_sha256": None, "activated_sha256": None, "lease_id": None, "lease_keeper_pid": None, "lease_keeper_start_fingerprint": None, "consumer_pid": None, "consumer_start_fingerprint": None, "selected_at": _reader_timestamp()}
        else:
            result = acquire_reader_lease(board_dir, consumer=args.consumer, operation_id=args.operation_id, consumer_pid=os.getppid())
    except (OSError, ValueError, sqlite3.Error, GenerationError) as exc:
        code = _reader_error_code(exc, "LEASE_CREATE_FAILED")
        return _reader_error(
            args.operation_id, code, args.board, retryable=_reader_is_retryable(code)
        )
    ready = {"protocol_version": 1, "type": "ready", "operation_id": args.operation_id, "board_slug": args.board, **_reader_public_result(result)}
    _reader_write_json(ready)
    from hermes_cli.kanban_generation import release_reader_lease
    lease_generation = Path(result["database_path"]).parent if result["lease_id"] else None
    try:
        def release() -> None:
            if lease_generation is not None:
                release_reader_lease(lease_generation, result["lease_id"])

        def release_before_error(code: str, *, retryable: bool = False) -> int:
            try:
                release()
            except (OSError, GenerationError):
                return _reader_error(args.operation_id, "LEASE_RELEASE_FAILED", args.board,
                                     generation_id=result["generation_id"])
            return _reader_error(args.operation_id, code, args.board,
                                 generation_id=result["generation_id"], retryable=retryable)
        import sys
        raw = sys.stdin.buffer.readline(_READER_MAX_FRAME_BYTES + 1)
        if len(raw) > _READER_MAX_FRAME_BYTES:
            raw = b""
        try:
            message = json.loads(raw) if raw else None
        except (TypeError, ValueError):
            message = object()
        if raw and (not isinstance(message, dict) or
                    set(message) != {"protocol_version", "op", "lease_id"} or
                    message.get("protocol_version") != 1 or message.get("op") != "release"):
            return release_before_error("LEASE_RELEASE_FAILED")
        if message is None:
            release()
            _reader_write_json({"protocol_version": 1, "type": "released",
                                "lease_id": result["lease_id"]})
            return 0
        if message.get("lease_id") != result["lease_id"]:
            return _reader_error(
                args.operation_id, "LEASE_ID_MISMATCH", args.board,
                generation_id=result["generation_id"],
            )
        release()
        _reader_write_json({"protocol_version": 1, "type": "released",
                            "lease_id": result["lease_id"]})
        return 0
    except (OSError, ValueError, sqlite3.Error, GenerationError) as exc:
        return _reader_error(
            args.operation_id, _reader_error_code(exc, "LEASE_RELEASE_FAILED"), args.board,
            generation_id=result.get("generation_id"),
        )


def _cmd_with_reader_lease(args: argparse.Namespace) -> int:  # health: allow CC -- frozen reader protocol phases must remain linear for custody/order proof
    if args.protocol_version != 1:
        return _reader_error(args.operation_id, "UNSUPPORTED_PROTOCOL", args.board)
    if args.consumer not in _READER_CONSUMERS:
        return _reader_error(args.operation_id, "CHILD_START_FAILED", args.board,
                             message="unsupported reader consumer")
    command = list(args.command or [])
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        return _reader_error(args.operation_id, "CHILD_START_FAILED", args.board)
    from hermes_cli.kanban_generation import acquire_reader_lease, bind_reader_consumer, release_reader_lease
    status_dir = Path(args.json_status_dir)
    try:
        status_dir.mkdir(mode=0o700)
    except FileExistsError:
        return _reader_error(args.operation_id, "STATUS_ID_CONFLICT", args.board)
    except OSError:
        return _reader_error(args.operation_id, "STATUS_PUBLICATION_FAILED", args.board)

    child = None
    lease = None
    legacy_result: dict = {}
    bootstrap_r = bootstrap_w = None
    outcome = None
    exit_code = 1
    cleanup_error = None
    terminal_lease_id = None
    generated_board = kb.board_dir(args.board)
    if (generated_board / "generation.current").exists():
        try:
            lease = acquire_reader_lease(
                generated_board, consumer=args.consumer,
                operation_id=args.operation_id, consumer_pid=os.getpid(),
            )
        except (OSError, ValueError, sqlite3.Error, GenerationError) as exc:
            code = _reader_error_code(exc, "LEASE_CREATE_FAILED")
            return _reader_error(
                args.operation_id, code, args.board, retryable=_reader_is_retryable(code)
            )
    else:
        legacy_path = kb.kanban_db_path(board=args.board)
        if not legacy_path.is_file():
            return _reader_error(args.operation_id, "BOARD_NOT_FOUND", args.board)
        legacy_result = {
            "database_path": str(legacy_path), "storage_mode": "legacy",
            "generation_id": None, "generation_nonce": None,
            "ready_sha256": None, "activated_sha256": None, "lease_id": None,
            "lease_keeper_pid": None, "lease_keeper_start_fingerprint": None,
            "consumer_pid": None, "consumer_start_fingerprint": None,
            "selected_at": _reader_timestamp(),
        }
    class _ReaderAbort(Exception):
        pass
    try:
        from hermes_cli.process_identity import spawn_owned, stop_owned_group_survivors
        bootstrap_r, bootstrap_w = os.pipe()
        spawn_options = {
            "stdin": subprocess.PIPE, "stdout": subprocess.PIPE, "text": False,
            "env": {"HERMES_READER_PROTOCOL_VERSION": "1",
                    "HERMES_READER_OPERATION_ID": args.operation_id,
                    "HERMES_READER_BOOTSTRAP_FD": str(bootstrap_r)},
        }
        if os.name == "posix":
            spawn_options["pass_fds"] = (bootstrap_r,)
        child = spawn_owned(command, purpose="kanban-reader", **spawn_options)
        os.close(bootstrap_r)
        bootstrap_r = None
        try:
            line = _read_pipe_line_deadline(child.stdout, _READER_HANDSHAKE_TIMEOUT_SECONDS)
        except TimeoutError:
            outcome = "CHILD_HANDSHAKE_FAILED"
            raise _ReaderAbort
        if line.strip() != b"started":
            outcome = "CHILD_HANDSHAKE_FAILED"
            raise _ReaderAbort
        if lease is not None:
            generation_dir = Path(lease["database_path"]).parent
            start_token = os.urandom(16).hex()
            bind_reader_consumer(generation_dir, lease["lease_id"], pid=child.pid, start_token=start_token)
            result = {**lease, "consumer_pid": child.pid,
                      "consumer_start_fingerprint": __import__("hermes_cli.kanban_generation", fromlist=["_start_fingerprint"])._start_fingerprint(child.pid),
                      "start_token": start_token}
        else:
            result = legacy_result
        capability = {"protocol_version": 1, "operation_id": args.operation_id,
                      "board_slug": args.board,
                      **{key: result.get(key) for key in _READER_CAPABILITY_FIELDS},
                      "start_token": result.get("start_token") or os.urandom(16).hex()}
        terminal_lease_id = result.get("lease_id")
        # The durable ready record is the publication boundary.  The child
        # receives exactly one capability, only after this record is fsynced;
        # the old stdin resend let consumers observe a second, differently
        # framed capability.
        _publish_status(status_dir, "ready.json", {"protocol_version": 1, "type": "ready", "operation_id": args.operation_id, "board_slug": args.board, **_reader_public_result(result), "start_token": capability["start_token"]})
        os.write(bootstrap_w, encode_reader_frame(capability))
        os.close(bootstrap_w)
        bootstrap_w = None

        # Capability delivery is complete on the private bootstrap FD.  Close
        # stdin so a non-daemonizing child can enter its normal read loop and
        # observe EOF without a second protocol message.
        child.stdin.close()
        try:
            rc = child.wait(timeout=86400)
        except subprocess.TimeoutExpired:
            outcome = "CHILD_FAILED"
            exit_code = 124
        else:
            exit_code = rc
            outcome = None if rc == 0 else "CHILD_FAILED"
    except _ReaderAbort:
        pass
    except (OSError, ValueError, sqlite3.Error, GenerationError, subprocess.SubprocessError):
        outcome = outcome or (
            "CHILD_CAPABILITY_FAILED" if terminal_lease_id is not None
            else "CHILD_START_FAILED"
        )
    finally:
        # All terminal paths use this one order: reap the whole owned group,
        # close protocol descriptors, release custody, then publish one status.
        if child is not None:
            try:
                stop_owned_group_survivors(child.pid, proc=child, grace=1.0)
                if child.poll() is None:
                    child.wait(timeout=10)
            except (OSError, ValueError, sqlite3.Error, GenerationError, subprocess.SubprocessError) as exc:
                cleanup_error = cleanup_error or exc
        for fd in (bootstrap_r, bootstrap_w):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        for stream in ((child.stdin if child is not None else None), (child.stdout if child is not None else None)):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        if lease is not None:
            try:
                release_reader_lease(Path(lease["database_path"]).parent, lease["lease_id"])
                lease = None
            except (OSError, ValueError, sqlite3.Error, GenerationError, subprocess.SubprocessError) as exc:
                cleanup_error = cleanup_error or exc
        if cleanup_error is not None:
            outcome = "LEASE_RELEASE_FAILED"
            exit_code = 1
        if outcome is None:
            terminal = {"protocol_version": 1, "operation_id": args.operation_id,
                        "type": "released", "lease_id": terminal_lease_id}
        else:
            terminal = {"protocol_version": 1, "type": "child_failed",
                        "operation_id": args.operation_id, "code": outcome,
                        "lease_id": terminal_lease_id}
        try:
            if not (status_dir / "terminal.json").exists():
                _publish_status(status_dir, "terminal.json", terminal)
        except OSError:
            outcome = "STATUS_PUBLICATION_FAILED"
            exit_code = 1
    if outcome is not None:
        return _reader_error(args.operation_id, outcome, args.board)
    return exit_code


def _board_task_counts(slug: str) -> dict[str, int]:
    """``{status: count}`` for a board. Safe to call on an empty DB."""
    try:
        path = kb.kanban_db_path(board=slug)
        if not path.exists() and not (kb.board_dir(slug) / "generation.current").exists():
            return {}
        with kbc.connect_closing(board=slug) as conn:
            rows = conn.execute("SELECT status, COUNT(*) AS n FROM tasks GROUP BY status").fetchall()
        return {r["status"]: int(r["n"]) for r in rows}
    except Exception:
        return {}


def _board_slug_arg(args: argparse.Namespace, cmd: str, *, must_exist: bool) -> tuple[Optional[str], int]:
    """Normalize ``args.slug`` for a ``boards`` subcommand; ``(slug, 0)`` or ``(None, rc)``."""
    try:
        normed = kb._normalize_board_slug(args.slug)
    except ValueError as exc:
        return None, _err(f"kanban boards {cmd}: {exc}", 2)
    if must_exist:
        if not normed or not kb.board_exists(normed):
            return None, _err(f"kanban boards {cmd}: board {args.slug!r} does not exist")
    elif not normed:
        return None, _err(f"kanban boards {cmd}: slug is required", 2)
    return normed, 0


def _cmd_boards_list(args: argparse.Namespace) -> int:
    boards = kb.list_boards(include_archived=bool(getattr(args, "all", False)))
    current = kb.get_current_board()
    for b in boards:
        b["is_current"] = (b["slug"] == current)
        b["counts"] = _board_task_counts(b["slug"])
        b["total"] = sum(b["counts"].values())
    if _json_out(args, boards):
        return 0
    if not boards:
        print("(no boards — create one with `hermes kanban boards create <slug>`)")
        return 0
    print(f"{'':2s}  {'SLUG':24s}  {'NAME':28s}  COUNTS")
    for b in boards:
        marker = "●" if b["is_current"] else " "
        name = (b.get("name") or "") + (" [archived]" if b.get("archived") else "")
        print(f"{marker:2s}  {b['slug']:24s}  {name:28s}  {_fmt_counts(b['counts'] or {}, '(empty)')}")
    print(f"\nCurrent board: {current}")
    if len(boards) > 1:
        print("Switch boards with `hermes kanban boards switch <slug>`.")
    return 0


def _cmd_boards_create(args: argparse.Namespace) -> int:
    normed, rc = _board_slug_arg(args, "create", must_exist=False)
    if rc:
        return rc
    already = kb.board_exists(normed) and normed != kb.DEFAULT_BOARD
    meta = kb.create_board(
        normed, name=args.name, description=args.description, icon=args.icon, color=args.color,
        default_workdir=args.default_workdir,
    )
    print(f"Board {meta['slug']!r} {'already exists' if already else 'created'}.\n"
          f"  Display name: {meta.get('name', '')}\n"
          f"  DB path:      {meta['db_path']}")
    if getattr(args, "switch", False):
        kb.set_current_board(meta["slug"])
        print(f"  Switched to {meta['slug']!r}.")
    else:
        print(f"  Use `hermes kanban boards switch {meta['slug']}` to make it current.")
    return 0


def _cmd_boards_rm(args: argparse.Namespace) -> int:
    # `boards delete <slug>` (alias) never sets args.delete because --delete belongs to the 'rm'
    # subparser only; treat the alias as `rm --delete`.
    # See #23139.
    force_delete = getattr(args, "delete", False) or getattr(args, "boards_action", "") == "delete"
    try:
        res = kb.remove_board(args.slug, archive=not force_delete)
    except ValueError as exc:
        return _err(f"kanban boards rm: {exc}")
    if res["action"] == "archived":
        print(f"Board {res['slug']!r} archived → {res['new_path']}\n"
              "Recover by moving the directory back to <root>/kanban/boards/<slug>/ "
              "and deleting the archived tombstone board.json left in its place.")
    else:
        print(f"Board {res['slug']!r} deleted.")
    return 0


def _cmd_boards_switch(args: argparse.Namespace) -> int:
    normed, rc = _board_slug_arg(args, "switch", must_exist=False)
    if rc:
        return rc
    if not kb.board_exists(normed):
        return _err(
            f"kanban boards switch: board {normed!r} does not exist. "
            f"Create it with `hermes kanban boards create {normed}`."
        )
    # An archived board has a tombstone board.json; switching to it would pin
    # `current` at a dead slug (#43243).
    if kb.read_board_metadata(normed).get("archived"):
        return _err(f"kanban boards switch: board {normed!r} is archived.")
    kb.set_current_board(normed)
    print(f"Active board is now {normed!r}.")
    return 0


def _cmd_boards_show(args: argparse.Namespace) -> int:
    current = kb.get_current_board()
    meta = kb.read_board_metadata(current)
    counts = _board_task_counts(current)
    print(f"Current board: {current}\n  Display name: {meta.get('name', '')}")
    if meta.get("description"):
        print(f"  Description:  {meta['description']}")
    print(f"  DB path:      {meta['db_path']}\n"
          f"  Tasks:        {sum(counts.values())} total" + (f" ({_fmt_counts(counts)})" if counts else ""))
    return 0


def _cmd_boards_rename(args: argparse.Namespace) -> int:
    normed, rc = _board_slug_arg(args, "rename", must_exist=True)
    if rc:
        return rc
    meta = kb.write_board_metadata(normed, name=args.name)
    print(f"Board {normed!r} renamed to {meta['name']!r}.")
    return 0


def _cmd_boards_set_default_workdir(args: argparse.Namespace) -> int:
    normed, rc = _board_slug_arg(args, "set-default-workdir", must_exist=True)
    if rc:
        return rc
    new_val = kb.write_board_metadata(normed, default_workdir=args.path).get("default_workdir")
    if new_val:
        print(f"Board {normed!r} default workdir set to {new_val!r}.")
    else:
        print(f"Board {normed!r} default workdir cleared.")
    return 0


def _cmd_boards_export(args: argparse.Namespace) -> int:
    from hermes_cli import kanban_transfer
    from hermes_cli.sizefmt import format_bytes

    slug = args.slug or kb.get_current_board()
    output = args.output or f"{slug}.tar.gz"
    try:
        res = kanban_transfer.export_board(
            slug, output, include_attachments=not args.no_attachments, include_logs=args.include_logs,
        )
    except (OSError, ValueError) as exc:
        return _err(f"kanban boards export: {exc}")
    if _json_out(args, res):
        return 0
    counts = res["counts"]
    print(f"Exported board {res['board']!r} → {res['archive']}\n"
          f"  Size:        {format_bytes(res['size'])}\n"
          f"  Tasks:       {counts['tasks']}\n"
          f"  Comments:    {counts['task_comments']}\n"
          f"  Attachments: {counts['attachment_files']}\n"
          "Import it with `hermes kanban boards import <archive>`.")
    return 0


def _cmd_boards_import(args: argparse.Namespace) -> int:
    from hermes_cli import kanban_transfer

    try:
        res = kanban_transfer.import_board(args.archive, args.as_slug, activate=args.switch)
    except (OSError, ValueError) as exc:
        return _err(f"kanban boards import: {exc}")
    if _json_out(args, res):
        return 0
    print(f"Imported board {res['board']!r} ({res['name']}).")
    if res["renamed"]:
        print(f"  Renamed from {res['requested_board']!r} — that slug was taken.")
    print(f"  Path:  {res['path']}\n  Tasks: {res['counts']['tasks']}")
    for warning in res["warnings"]:
        print(f"  Note:  {warning}")
    if res["activated"]:
        print(f"  Active board is now {res['board']!r}.")
    else:
        print(f"  Switch to it with `hermes kanban boards switch {res['board']}`.")
    return 0


_BOARD_HANDLERS = {
    "reconcile": _cmd_reconcile,
    "verify-prepared-generation": _cmd_verify_prepared,
    "publish-prepared-generation": _cmd_publish_prepared,
    "gc-generations": _cmd_gc_generations,
    "rollback-generation": _cmd_rollback_generation,
    "reader-session": _cmd_reader_session,
    "with-reader-lease": _cmd_with_reader_lease,
    "list": _cmd_boards_list, "ls": _cmd_boards_list,
    "create": _cmd_boards_create, "new": _cmd_boards_create,
    "rm": _cmd_boards_rm, "remove": _cmd_boards_rm, "delete": _cmd_boards_rm,
    "switch": _cmd_boards_switch, "use": _cmd_boards_switch,
    "show": _cmd_boards_show, "current": _cmd_boards_show,
    "rename": _cmd_boards_rename,
    "set-default-workdir": _cmd_boards_set_default_workdir,
    "export": _cmd_boards_export,
    "import": _cmd_boards_import,
}
