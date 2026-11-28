"""External-oneshot probe: does a live ``hermes -z`` worker already own this card?

Split out of ``hermes_cli.kanban_db_dispatch``. A gateway session that hands a card to a
one-shot CLI worker spawns it WITHOUT claiming, so the DB-side guards
(``check_respawn_guard``, the ``claim_task`` CAS) cannot see that process and the next tick
spawns a twin on the same workspace. The origin namespace ``_kb`` and
``gateway.status_proc_table`` are reached late-bound INSIDE the functions (import-cycle
breaking), so monkeypatching those modules keeps working.
"""

from __future__ import annotations

import contextlib
import re
import shlex
from typing import Optional


# A task id as it appears on a worker's argv: inside the oneshot prompt
# (``-z "接单:部署卡 t_bb66a4a2…"``), a workspace path, a branch name. The shape
# ``create_task`` mints — lowercase hex, 6..32 — bounded by word breaks so a
# truncated prompt title still matches while ``xt_deadbeef`` and ``t_deadbeef1``
# do not (see :func:`_oneshot_argv_names_task` for why that must be equality).
_ONESHOT_TASK_ID_RE = re.compile(r"\b(t_[0-9a-f]{6,32})\b")

# The flag that makes an oneshot an oneshot. A bare task id on argv does not:
# ``hermes kanban show t_x`` and a gateway carrying the id in its env both name
# the card without working it.
_ONESHOT_FLAGS = ("-z", "--oneshot")

# Basenames of the Hermes CLI entrypoint (``hermes_cli.main`` / ``hermes_cli/main.py``
# are matched token-wise by :func:`_looks_like_hermes_cli`).
_HERMES_CLI_NAMES = ("hermes", "hermes.exe")


def _argv_tokens(command: str) -> list[str]:
    """Quote-aware tokens of a space-joined command line, surrounding quotes stripped.

    Same normalization as ``gateway.status._gateway_command_subcommand``: Windows paths with
    spaces need ``shlex`` (a ``str.split`` cuts them in half), and a command line with an
    unbalanced quote degrades to a whitespace split rather than dropping the candidate."""
    try:
        raw = shlex.split(command, posix=False)
    except ValueError:
        raw = command.split()
    return [token.strip("\"'") for token in raw if token.strip("\"'")]


def _looks_like_hermes_cli(argv0: str, joined: str) -> bool:
    """True when a command line is a Hermes CLI process.

    The same identity test as the preamble of ``gateway.status._gateway_command_subcommand``:
    the launcher basename (``hermes``, ``hermes.exe``) or the ``hermes_cli.main`` module /
    ``hermes_cli/main.py`` entrypoint. Every test is token/basename equality, never a bare
    substring of a path — ``/opt/hermes-notify/bin/notify``, ``hermes_cli.mainlib`` and a
    ``vendor/hermes_cli/main.pyo`` all name something that is not this CLI."""
    def _normalized(token: str) -> str:
        return (token or "").strip().strip("\"'").replace("\\", "/").lower()

    if _normalized(argv0).rsplit("/", 1)[-1] in _HERMES_CLI_NAMES:
        return True
    for token in _normalized(joined).split():
        if token == "hermes_cli.main":
            return True
        if token == "hermes_cli/main.py" or token.endswith("/hermes_cli/main.py"):
            return True
    return False


def _oneshot_argv_names_task(command: str, task_id: str) -> bool:
    """True when a live command line is a ``hermes … -z/--oneshot <PROMPT>`` naming *task_id*.

    Two token-wise conditions: the literal ``-z``/``--oneshot`` flag token is present, and
    *task_id* is one of the task ids the argv names. The flag is what separates an oneshot
    worker from every other Hermes process that merely mentions the card (``gateway run`` with
    the id in its env, an operator's ``kanban show t_x``).

    A bootstrap-wrapped spawn — the real shape is
    ``python -I -c <src> -p forge -z 接单:部署卡 t_bb66a4a2… --yolo`` — is peeled with
    ``gateway.status.inline_bootstrap_argv`` first, exactly as the gateway identity code does,
    so the inline source's own text is never read as argv. A ``-c`` source that is not a
    recognized Hermes bootstrap keeps its raw tokens: the trailing argv is still this process's
    own words, so the scan stays conservative rather than going blind.

    Id matching is EQUALITY, not substring: ids are collected with
    ``_ONESHOT_TASK_ID_RE.findall`` and *task_id* must equal one of them. A worker running
    ``t_deadbeef1`` therefore does NOT answer a lookup for ``t_deadbeef`` — the regex's greedy
    hex run consumes the extra digit, so only the longer id is ever found — and a substring
    test would have parked ``t_deadbeef`` behind an unrelated card for the whole of that run.
    """
    if not command or not task_id:
        return False
    tokens = _argv_tokens(command)
    if not tokens:
        return False
    lowered = [token.lower() for token in tokens]
    with contextlib.suppress(Exception):
        from gateway.status import command_line_runs_inline_source, inline_bootstrap_argv
        if command_line_runs_inline_source(tokens):
            peeled = inline_bootstrap_argv(tokens)
            if peeled is not None:
                lowered = [token.lower() for token in peeled]
    if not any(token in _ONESHOT_FLAGS for token in lowered):
        return False
    return task_id.lower() in set(_ONESHOT_TASK_ID_RE.findall(" ".join(lowered)))


def find_external_oneshot_worker(task_id: str) -> Optional[int]:
    """PID of a live external Hermes oneshot already working *task_id*, else None.

    The gateway session's "接单: <card>" path spawns
    ``hermes -p <profile> -z "接单:部署卡 t_x…" --yolo`` as a background terminal and never
    calls ``claim_task``, so the row sits in ``ready`` with no claim for the whole run.
    ``check_respawn_guard`` and the ``claim_task`` CAS are both DB-side and cannot see a
    process, so the next dispatcher tick claimed the same card and spawned a twin worker on the
    same task and the same workspace — same ports, one ``.env`` overwritten (t_c18fc591, then
    t_bb66a4a2). This probe reads the live process table instead of the board.

    Cost is one ``psutil`` sweep per spawnable row per tick
    (:func:`gateway.status_proc_table.iter_process_cmdlines`), cheap against the collision it
    prevents. Each candidate is re-confirmed with :func:`_pid_alive`, because a process table can
    be a moment stale. Everything fails open to None: an unreadable process table must never wedge
    the board, at the price of the twin this tick failed to catch."""
    if not task_id:
        return None
    try:
        from gateway.status_proc_table import iter_process_cmdlines
        entries = iter_process_cmdlines() or []
    except Exception:  # health: allow BLE001 -- an unreadable process table must never wedge the board; the probe fails open
        return None
    from hermes_cli import kanban_db as _kb
    for pid, command in entries:
        try:
            tokens = _argv_tokens(command or "")
            if not tokens or not _looks_like_hermes_cli(tokens[0], command):
                continue
            if not _oneshot_argv_names_task(command, task_id):
                continue
            if not _kb._pid_alive(int(pid)):
                continue
            return int(pid)
        except Exception:  # health: allow BLE001 -- one unparsable or vanished argv entry must not lose the rest of the sweep
            continue
    return None


def external_oneshot_detection_enabled() -> bool:
    """Whether the dispatcher probes for an external ``hermes -z`` oneshot already
    working a card before claiming it (``kanban.detect_external_oneshot``). Default
    true — twin workers collide on one workspace — and a config that cannot be read
    keeps the default instead of silently switching the guard off.
    """
    try:
        from hermes_cli.config import load_config
        return bool((load_config() or {}).get("kanban", {}).get("detect_external_oneshot", True))
    except Exception:  # health: allow BLE001 -- unreadable config keeps the default-on guard instead of silently switching it off
        return True


def defer_for_external_oneshot(conn, task_id: str, result, *, dry_run: bool) -> bool:
    """True when this card is left to a live external ``hermes -z`` worker instead of claimed.

    A gateway session can hand the card to a one-shot CLI worker (``hermes -z …``) that never
    claims it, leaving the row ``ready`` with no claim while the worker runs — no DB-side guard
    sees that process, and the dispatch tick would have spawned a twin onto the same workspace.
    The deferral is recorded as ``(task_id, pid)`` on ``result.external_oneshot_skipped``.

    Unless this is a dry run, the same ``respawn_guarded`` event the respawn guard writes is
    appended with ``reason="external_oneshot_worker"`` and the worker's ``pid``: ``hermes kanban
    tail`` already renders those rows, and the reason distinguishes "someone else owns it" from a
    cooldown.
    """
    if not external_oneshot_detection_enabled():
        return False
    oneshot_pid = find_external_oneshot_worker(task_id)
    if oneshot_pid is None:
        return False
    result.external_oneshot_skipped.append((task_id, oneshot_pid))
    if not dry_run:
        from hermes_cli import kanban_db as _kb
        with _kb.write_txn(conn):
            _kb._append_event(conn, task_id, "respawn_guarded",
                              {"reason": "external_oneshot_worker", "pid": oneshot_pid})
    return True


def _error_fingerprint(error_text: str) -> str:
    """Normalize an error message (strip PIDs, timestamps) so same-root-cause errors group."""
    fp = re.sub(r'\bpid \d+\b', 'pid N', error_text[:80])
    fp = re.sub(r'\b\d{10,}\b', '<TS>', fp)
    return fp.lower().strip()
