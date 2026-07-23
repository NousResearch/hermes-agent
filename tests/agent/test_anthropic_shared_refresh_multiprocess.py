"""Cross-process single-use Anthropic shared refresh tests."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread

import pytest

from tests.agent.anthropic_shared_test_helpers import (
    enable_marker,
    make_row,
    shared_root,
    write_root_auth,
)


class _SingleUseHandler(BaseHTTPRequestHandler):
    hits = 0
    seen_tokens = []

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode()
        type(self).hits += 1
        type(self).seen_tokens.append(body)
        if type(self).hits > 1:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error":"invalid_grant"}')
            return
        self.send_response(200)
        self.end_headers()
        self.wfile.write(
            json.dumps(
                {
                    "access_token": "fixture-oauth-access-rotated-zzzzzzzz",
                    "refresh_token": "fixture-oauth-refresh-rotated-zzzzzzzz",
                    "expires_in": 3600,
                    "token_type": "Bearer",
                }
            ).encode()
        )

    def log_message(self, fmt, *args):  # noqa: A003
        return


def _worker(root: str, home: str, row_id: str, gen: int, q):
    os.environ["HERMES_HOME"] = home
    os.environ["HOME"] = str(Path(root).parent)
    os.environ.pop("ANTHROPIC_API_KEY", None)
    os.environ.pop("ANTHROPIC_TOKEN", None)
    try:
        from agent import anthropic_shared_pool as sp

        sp.reset_startup_epoch_for_tests()

        def transport(*, refresh_token, endpoint_id):
            import urllib.request

            data = json.dumps(
                {
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": "test",
                }
            ).encode()
            req = urllib.request.Request(
                os.environ["TEST_TOKEN_URL"],
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                result = json.loads(resp.read().decode())
            return {
                "access_token": result["access_token"],
                "refresh_token": result["refresh_token"],
                "expires_at_ms": int(time.time() * 1000) + 3_600_000,
            }

        updated = sp.commit_refresh(
            row_id, expected_generation=gen, transport=transport
        )
        q.put(("ok", updated["token_generation"], updated["access_token"]))
    except Exception as exc:
        q.put(("err", type(exc).__name__, str(exc)[:200]))


@pytest.mark.skipif(os.name == "nt", reason="fcntl multiproc test")
def test_two_process_refresh_single_post(shared_root, monkeypatch):
    # Local single-use HTTP stub
    _SingleUseHandler.hits = 0
    _SingleUseHandler.seen_tokens = []
    server = HTTPServer(("127.0.0.1", 0), _SingleUseHandler)
    port = server.server_address[1]
    t = Thread(target=server.serve_forever, daemon=True)
    t.start()
    monkeypatch.setenv("TEST_TOKEN_URL", f"http://127.0.0.1:{port}/v1/oauth/token")

    # One expired-but-loadable row (skip enrollment future check)
    row = make_row(priority=0)
    row["expires_at_ms"] = int(time.time() * 1000) - 1000
    # Need 3 rows for active shared
    entries = [row] + [make_row(priority=i, refresh=f"fixture-oauth-refresh-extra-{i}") for i in range(1, 3)]
    for i, e in enumerate(entries):
        e["priority"] = i
    from agent.anthropic_shared_pool import validate_shared_pool

    pool = {
        "schema_version": 1,
        "revision": 1,
        "strategy": "fill_first",
        "account_distinctness_attested": True,
        "account_distinctness_attested_at": "2026-01-01T00:00:00Z",
        "entries": entries,
    }
    pool = validate_shared_pool(pool, require_three=True)
    write_root_auth(
        shared_root,
        {"version": 1, "providers": {}, "shared_credential_pools": {"anthropic": pool}},
    )
    enable_marker(shared_root)

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    home = str(shared_root)
    root = str(shared_root)
    procs = [
        ctx.Process(target=_worker, args=(root, home, row["id"], 1, q))
        for _ in range(2)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=30)
        assert p.exitcode == 0

    results = [q.get(timeout=5) for _ in range(2)]
    oks = [r for r in results if r[0] == "ok"]
    assert len(oks) >= 1
    # Exactly one successful HTTP POST
    assert _SingleUseHandler.hits == 1
    # Disk generation advanced once
    data = json.loads((shared_root / "auth.json").read_text())
    stored = data["shared_credential_pools"]["anthropic"]["entries"][0]
    assert stored["token_generation"] == 2
    assert stored["access_token"].startswith("fixture-oauth-access-rotated")
    server.shutdown()


def test_refresh_response_table_unknown_on_timeout(shared_root):
    from agent import anthropic_shared_pool as sp
    from hermes_cli.auth import AuthError
    from tests.agent.anthropic_shared_test_helpers import stage_three

    stage_three(shared_root, attest=True)
    enable_marker(shared_root)
    sp.reset_startup_epoch_for_tests()
    pool = sp.load_shared_pool_for_management(require_active_three=True)
    row_id = pool["entries"][0]["id"]

    def boom(**kwargs):
        err = AuthError("timeout", provider="anthropic", code="refresh_transport")
        err.ambiguous = True
        raise err

    with pytest.raises(AuthError):
        sp.commit_refresh(row_id, expected_generation=1, transport=boom)

    pool2 = sp.load_shared_pool_for_management(require_active_three=True)
    row = next(e for e in pool2["entries"] if e["id"] == row_id)
    assert row["last_status"] == "dead"
    assert row["last_error_reason"] == "refresh_outcome_unknown"
    assert row["refresh_attempt"]["outcome"] == "unknown"


def test_abandon_inflight_no_post(shared_root):
    from agent import anthropic_shared_pool as sp
    from tests.agent.anthropic_shared_test_helpers import stage_three

    stage_three(shared_root, attest=True)
    enable_marker(shared_root)
    sp.reset_startup_epoch_for_tests()
    with sp.root_auth_lock():
        store = sp.load_root_auth_strict()
        pool = sp.get_shared_namespace(store)
        row = pool["entries"][0]
        row["refresh_attempt"] = {
            "attempt_id": "00000000-0000-0000-0000-000000000001",
            "expected_generation": 1,
            "started_at": "2026-01-01T00:00:00Z",
            "outcome": "inflight",
        }
        pool["revision"] += 1
        sp.set_shared_namespace(store, pool)
        sp.save_root_auth_strict(store)

    # Successor: load path / commit_refresh should abandon inflight without POST.
    with sp.root_auth_lock():
        store = sp.load_root_auth_strict()
        pool = sp.get_shared_namespace(store)
        assert pool is not None
        changed = sp._abandon_inflight_attempts(pool)
        assert changed
        pool["revision"] = int(pool["revision"]) + 1
        sp.set_shared_namespace(store, pool)
        sp.save_root_auth_strict(store)

    posts = {"n": 0}

    def transport(**kwargs):
        posts["n"] += 1
        return {
            "access_token": "x",
            "refresh_token": "y",
            "expires_at_ms": int(time.time() * 1000) + 10000,
        }

    # Row is now dead/unknown — commit_refresh must refuse without POST
    from hermes_cli.auth import AuthError

    with pytest.raises(AuthError):
        sp.commit_refresh(
            pool["entries"][0]["id"],
            expected_generation=1,
            transport=transport,
        )
    assert posts["n"] == 0
    pool2 = sp.load_shared_pool_for_management(require_active_three=True)
    row = pool2["entries"][0]
    assert row["refresh_attempt"]["outcome"] == "unknown"
    assert row["last_status"] == "dead"
