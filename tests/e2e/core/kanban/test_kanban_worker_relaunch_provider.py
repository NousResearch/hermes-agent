"""A dispatcher retry must keep its model/provider pin and launch from the task workspace.

The first OpenAI Codex request returns 429; a fresh worker must use the same route,
complete the card, and persist ``billing_provider=openai-codex``. HERMES_BIN is
cleared and PATH contains no Hermes executable so dispatch uses the install-bound
launcher/bootstrap rather than an ambient command.
"""
from __future__ import annotations

import base64
import json
import sqlite3
import sys
import time

import pytest

from tests.e2e.core.kanban._helpers import Board
from tests.fakes.fake_llm_provider import Error, FakeLLMServer
from tests.fakes.providers.openai_responses import (
    FakeResponsesServer, FunctionCall, HttpError, Message, Turn,
)

pytestmark = [
    pytest.mark.skipif(not sys.platform.startswith("linux"), reason="worker liveness uses /proc"),
    pytest.mark.live_system_guard_bypass,
]

MODEL = "gpt-6-luna"
PROVIDER = "openai-codex"


def _codex_store() -> tuple[dict, str]:
    """One local fake OAuth grant with enough TTL to avoid contacting the token endpoint."""
    def segment(value: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(value).encode()).rstrip(b"=").decode()

    exp = int(time.time()) + 8 * 60 * 60
    access_token = (
        f"{segment({'alg': 'none'})}."
        f"{segment({'exp': exp, 'sub': 'e2e-codex-user', 'https://api.openai.com/auth': {'chatgpt_account_id': 'e2e-codex-account'}})}.sig"
    )
    tokens = {"access_token": access_token, "refresh_token": "e2e-refresh-not-used"}
    refreshed = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 60))
    store = {
        "version": 1,
        "active_provider": PROVIDER,
        "providers": {PROVIDER: {"tokens": tokens, "last_refresh": refreshed, "auth_mode": "chatgpt"}},
        "credential_pool": {PROVIDER: [{
            "id": "e2e-codex-login", "label": "device_code", "auth_type": "oauth",
            "priority": 0, "source": "device_code", **tokens, "last_refresh": refreshed,
        }]},
    }
    return store, access_token


def _session_routes(board: Board) -> list[tuple[str, str]]:
    path = board.hermes_home / "state.db"
    if not path.is_file():
        return []
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
    try:
        rows = conn.execute(
            """SELECT DISTINCT u.model, u.billing_provider
               FROM session_model_usage AS u
               JOIN sessions AS s ON s.id = u.session_id
               WHERE s.source = 'kanban'"""
        ).fetchall()
        return [(str(model), str(provider)) for model, provider in rows]
    finally:
        conn.close()


def test_worker_relaunch_and_rate_limit_retry_keep_pinned_provider(tmp_path) -> None:
    auth_store, bearer = _codex_store()
    codex_steps = [
        HttpError(status=429, message="scripted Codex rate limit", retry_after=0),
        Turn([FunctionCall("kanban_complete", {"summary": "Codex route survived retry"})]),
        Turn([Message("card completed through OpenAI Codex")]),
    ]

    def default_route_must_not_be_used(_record: dict):
        return Error(429, "default provider was used instead of the task override", retry_after=0)

    with FakeLLMServer(default_route_must_not_be_used) as default_server, \
            FakeResponsesServer(codex_steps, api_key=bearer) as codex_server:
        board = Board(
            tmp_path,
            default_server.base_url,
            env_extra={
                "HERMES_BIN": "",
                "HERMES_CODEX_BASE_URL": codex_server.base_url,
                "PATH": "/usr/bin:/bin",
            },
        )
        board.hermes_home.joinpath(".env").write_text(
            f"OPENAI_API_KEY=sk-fake-default\nHERMES_CODEX_BASE_URL={codex_server.base_url}\n",
            encoding="utf-8",
        )
        board.hermes_home.joinpath("auth.json").write_text(json.dumps(auth_store), encoding="utf-8")
        try:
            task_id = board.create(
                "pinned Codex retry probe",
                "--model", MODEL,
                "--provider", PROVIDER,
            )

            first_tick = board.dispatch()
            assert task_id in [row["task_id"] for row in first_tick.get("spawned", [])], board.diag(task_id)
            first_pid = int(board.task(task_id)["worker_pid"])
            board.wait_worker_exit(task_id, first_pid)

            second_tick = board.dispatch()
            assert task_id in [row["task_id"] for row in second_tick.get("spawned", [])], board.diag(task_id)
            second_pid = int(board.task(task_id)["worker_pid"])
            assert second_pid != first_pid
            board.wait_worker_exit(task_id, second_pid)

            assert board.task(task_id)["status"] == "done", board.diag(task_id)
            outcomes = [run["outcome"] for run in board.runs(task_id)]
            assert outcomes[:2] == ["rate_limited", "completed"], board.diag(task_id)
            assert default_server.main_requests() == [], "task override fell through to the profile provider"
            requests = codex_server.main_requests()
            assert len(requests) == 3, board.diag(task_id)
            assert codex_server.invalid_requests() == [], codex_server.invalid_requests()
            assert all(request.get("model") == MODEL for request in requests), requests
            assert all(
                request["headers"].get("authorization") == f"Bearer {bearer}"
                for request in codex_server.requests if request["kind"] == "main"
            ), codex_server.requests
            assert (MODEL, PROVIDER) in _session_routes(board), board.diag(task_id)
        finally:
            board.kill_workers()
