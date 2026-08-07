#!/usr/bin/env python3
"""kb_client — the ONE thin transport every kanban call site uses.

Workers, the `kb` CLI, kanban-sync, the bridge, board-steward, the disk guard,
the integrity guard, doctor — ALL board access goes through this module, which
speaks boardd's newline-delimited JSON over a Unix domain socket. No caller
ever opens kanban.db directly again (a CI guard enforces that; see
check_no_direct_openers.sh).

Idempotency contract (exactly-once for mutations):
  * Each *logical* mutation gets ONE op_id (uuid4), generated once.
  * On any transport failure (broker restart, timeout, reset) the SAME op_id is
    resent with jittered bounded backoff. boardd's applied_ops ledger returns
    the stored result for a replayed op_id, so a ~2s broker restart is
    invisible and a claim is never double-applied.
  * Reads carry no op_id and are simply retried (idempotent by nature).

Driver-swap seam (Sequence B->C): this file is the only thing callers import.
A Postgres future swaps boardd's connection; kb_client's wire protocol is
unchanged.
"""
from __future__ import annotations

import json
import os
import random
import socket
import time
import uuid

DEFAULT_SOCK = os.environ.get(
    # Dedicated-user design: socket in its own boardd-owned dir (boardd-run,
    # 0710 group boardd), reachable by boardd-group members; the 0700 DB dir is
    # not. Must match boardd.py DEFAULT_SOCK + boardd.service. See RUNBOOK §4.
    "BOARDD_SOCK",
    os.path.join(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")),
                 "kanban", "boardd-run", "boardd.sock"),
)

# jittered bounded backoff. MUST exceed boardd's HANDLER_TIMEOUT_S so that when
# the DB thread stalls, the server drops the socket (a retryable transport
# failure) and the client keeps resending the SAME op_id until it collects the
# committed result via applied_ops replay — exactly-once even through a stall or
# a ~2s restart (BLOCKER-2). 90s >> 30s handler timeout.
_RETRY_DEADLINE_S = float(os.environ.get("KB_CLIENT_RETRY_DEADLINE_S", "90"))
_BACKOFF_MIN_S = 0.02
_BACKOFF_MAX_S = 0.75
_CONNECT_TIMEOUT_S = float(os.environ.get("KB_CLIENT_CONNECT_TIMEOUT_S", "5"))
_READ_TIMEOUT_S = float(os.environ.get("KB_CLIENT_READ_TIMEOUT_S", "60"))


class BoarddError(RuntimeError):
    def __init__(self, error, etype=None):
        super().__init__(error)
        self.etype = etype


class BoarddUnavailable(RuntimeError):
    pass


class Client:
    """A connection to boardd. Cheap to create; safe to keep one per process.

    NOT thread-safe on a single instance's socket — use one Client per thread,
    or the module-level helpers which use a thread-local client.
    """

    def __init__(self, sock_path: str = DEFAULT_SOCK, worker_id: str | None = None):
        self.sock_path = sock_path
        self.worker_id = worker_id or f"{socket.gethostname()}:{os.getpid()}"
        self._seq = 0
        self._sock: socket.socket | None = None
        self._rfile = None

    # ---- transport -------------------------------------------------------- #
    def _connect(self) -> None:
        self._close()
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(_CONNECT_TIMEOUT_S)
        s.connect(self.sock_path)
        s.settimeout(_READ_TIMEOUT_S)
        self._sock = s
        self._rfile = s.makefile("rb")

    def _close(self) -> None:
        try:
            if self._rfile is not None:
                self._rfile.close()
        except Exception:
            pass
        try:
            if self._sock is not None:
                self._sock.close()
        except Exception:
            pass
        self._sock = self._rfile = None

    def _roundtrip(self, req: dict) -> dict:
        """One send+recv on the current connection; raises on transport error."""
        if self._sock is None:
            self._connect()
        line = (json.dumps(req) + "\n").encode("utf-8")
        self._sock.sendall(line)
        raw = self._rfile.readline()
        if not raw:
            raise BoarddUnavailable("connection closed by broker")
        return json.loads(raw)

    # ---- request with idempotent retry ------------------------------------ #
    def _request(self, op: str, args: dict | None = None,
                 *, mutation: bool, op_id: str | None = None) -> dict:
        req: dict = {"op": op}
        if args:
            req["args"] = args
        if mutation:
            self._seq += 1
            req["op_id"] = op_id or uuid.uuid4().hex
            req["worker_id"] = self.worker_id
            req["seq"] = self._seq
        deadline = time.monotonic() + _RETRY_DEADLINE_S
        attempt = 0
        while True:
            attempt += 1
            try:
                resp = self._roundtrip(req)
                if not resp.get("ok", False):
                    raise BoarddError(resp.get("error", "unknown error"),
                                      resp.get("etype"))
                return resp
            except (OSError, BoarddUnavailable, ValueError) as exc:
                # transport-level failure: reconnect + resend SAME op_id
                self._close()
                if time.monotonic() >= deadline:
                    raise BoarddUnavailable(
                        f"boardd unreachable after {attempt} attempts: {exc}")
                time.sleep(min(_BACKOFF_MAX_S,
                               _BACKOFF_MIN_S * (2 ** min(attempt, 6)))
                           * (0.5 + random.random()))

    # ---- typed API (mirrors hermes_cli.kanban_db) ------------------------- #
    # mutations
    def create_task(self, *, title, assignee=None, status="running", body=None,
                    created_by=None, priority=0, workspace_kind="scratch",
                    id=None, op_id=None):
        return self._request("create_task", {
            "title": title, "assignee": assignee, "status": status, "body": body,
            "created_by": created_by, "priority": priority,
            "workspace_kind": workspace_kind, "id": id}, mutation=True,
            op_id=op_id)["result"]

    def add_comment(self, task_id, author, body, *, op_id=None):
        return self._request("add_comment",
                             {"task_id": task_id, "author": author, "body": body},
                             mutation=True, op_id=op_id)["result"]

    def set_workspace_path(self, task_id, path, *, op_id=None):
        return self._request("set_workspace_path",
                             {"task_id": task_id, "path": str(path)},
                             mutation=True, op_id=op_id)["result"]

    def set_branch_name(self, task_id, branch_name, *, op_id=None):
        return self._request("set_branch_name",
                             {"task_id": task_id, "branch_name": str(branch_name)},
                             mutation=True, op_id=op_id)["result"]

    def set_body(self, task_id, body, *, op_id=None):
        return self._request("set_body", {"task_id": task_id, "body": body},
                             mutation=True, op_id=op_id)["result"]

    def set_status(self, task_id, status, *, op_id=None):
        return self._request("set_status", {"task_id": task_id, "status": status},
                             mutation=True, op_id=op_id)["result"]

    def heartbeat(self, task_id, note=None, *, op_id=None):
        return self._request("heartbeat", {"task_id": task_id, "note": note},
                             mutation=True, op_id=op_id)["result"]

    def claim(self, task_id, *, claimer=None, ttl_seconds=7200, op_id=None):
        return self._request("claim", {
            "task_id": task_id, "claimer": claimer, "ttl_seconds": ttl_seconds,
            "worker_id": self.worker_id}, mutation=True, op_id=op_id)["result"]

    def exec_write(self, sql, params=None, *, op_id=None):
        # returns {"rowcount":..., "lastrowid":...}
        return self._request("exec", {"sql": sql, "params": params or []},
                             mutation=True, op_id=op_id)["result"]

    # ---- interactive transactions (token-based; connection-independent) ----- #
    # Used by boardd_shim.BrokerConnection so the REAL kanban_db write_txn
    # functions compose their atomic transaction through the single writer.
    # mutation=False: no op_id (the txn TOKEN is the identity). The token is
    # connection-independent, so the standard reconnect-retry is safe — a resent
    # txn_exec targets the same open txn if the broker is alive, or gets a
    # TxnStale error (surfaced, not retried) if the broker restarted, which
    # aborts the write_txn so the caller re-runs (lifecycle ops are CAS-idempotent).
    def txn_begin(self):
        return self._request("txn_begin", mutation=False)["result"]["txn"]

    def txn_exec(self, txn, sql, params=None):
        return self._request("txn_exec", {"txn": txn, "sql": sql,
                                          "params": params or []},
                             mutation=False)["result"]

    def txn_commit(self, txn):
        return self._request("txn_commit", {"txn": txn}, mutation=False)["result"]

    def txn_rollback(self, txn):
        return self._request("txn_rollback", {"txn": txn}, mutation=False)["result"]

    # reads
    def query(self, sql, params=None, max_rows=None):
        # default cap ~20k rows; pass max_rows for a known bounded larger read,
        # or page with LIMIT/OFFSET for unbounded sets (dashboard).
        args = {"sql": sql, "params": params or []}
        if max_rows:
            args["max_rows"] = int(max_rows)
        return self._request("query", args, mutation=False)["result"]

    def integrity_check(self):
        return self._request("integrity_check", mutation=False)["result"]

    def quick_check(self):
        return self._request("quick_check", mutation=False)["result"]

    def reindex(self):
        """Broker-owned in-place recovery (REINDEX on the single owned
        connection). Use this from the integrity watchdog INSTEAD of an
        external REINDEX/inode-swap, which corrupts under a live stale handle."""
        return self._request("_reindex", mutation=False)["result"]

    def get_task(self, task_id):
        return self._request("get_task", {"task_id": task_id},
                             mutation=False)["result"]

    def list_tasks(self, assignee=None, limit=500):
        return self._request("list_tasks", {"assignee": assignee, "limit": limit},
                             mutation=False)["result"]

    def ping(self):
        return self._request("ping", mutation=False)["result"]

    def close(self):
        self._close()


# ---- module-level convenience (THREAD-LOCAL client) ----------------------- #
# A single Client owns one socket and is NOT safe to share across threads
# (interleaved send/recv corrupts the stream). The A2 gateway reroute is heavily
# threaded, so callers MUST use a per-thread client. get_client() returns a
# thread-local Client; every module-level helper and the boardd_shim use it, so
# threaded callers get isolation for free (M-9).
import threading as _threading  # noqa: E402
_tl = _threading.local()


def get_client() -> Client:
    c = getattr(_tl, "client", None)
    if c is None:
        c = Client()
        _tl.client = c
    return c


_client = get_client  # back-compat alias


def ping():
    return get_client().ping()


def query(sql, params=None):
    return get_client().query(sql, params)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="kb_client CLI probe")
    ap.add_argument("cmd", choices=["ping", "query"])
    ap.add_argument("sql", nargs="?")
    a = ap.parse_args()
    if a.cmd == "ping":
        print(json.dumps(_client().ping(), indent=2))
    else:
        print(json.dumps(_client().query(a.sql), indent=2))
