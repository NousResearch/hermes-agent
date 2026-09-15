"""Route-pin guard: a card whose model pin the route cannot serve must be refused
*before* a worker is spawned, with an actionable reason on the board.

Contract under test (from the ROUTE-GUARD card, 2026-09-12):

* a card pinned to a model the route answers ``HTTP 400 No provider available for
  model '<X>'`` for is never spawned, is blocked (not completed) and is surfaced
  by ``kanban_diagnostics``;
* a healthy pin still dispatches and still carries ``-m``/``--provider``;
* every *other* probe outcome (401/403, 5xx, unreachable route, 200 with empty
  content) is explicitly NOT a bad pin — a credential/route hiccup must never
  stall the board;
* the verdict is cached per distinct pin so a dispatch tick stays cheap.

The routes are real HTTP servers on loopback, so the dispatcher exercises the real
transport rather than a mocked one.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

BAD_MODEL = "OL : deepseek-v4-flash:0731"
GOOD_MODEL = "ds : deepseek-flash"

# Stand-ins for a credential/transport problem: never a bad pin.
UNAUTHORIZED_MODEL = "auth-wall-model"
SERVER_ERROR_MODEL = "flaky-route-model"


def _router_body(model: str) -> str:
    return json.dumps({
        "error": {"message": f"No provider available for model '{model}'", "provider": "router"}
    })


def _closed_port() -> int:
    """A loopback port nothing is listening on (bound, then released)."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class _StubRouter:
    """Minimal OpenAI-compatible route: serves GOOD_MODEL, 400s BAD_MODEL.

    ``serves_all=True`` makes the route claim every model id — used to prove
    *which* config entry decided a probe.
    """

    def __init__(self, *, serves_all: bool = False):
        self.serves_all = serves_all
        self.requests: list[dict] = []
        self._server = None
        self._thread = None
        self.port = 0

    def start(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
                length = int(self.headers.get("content-length") or 0)
                raw = self.rfile.read(length).decode("utf-8", "replace")
                payload = json.loads(raw or "{}")
                model = str(payload.get("model") or "")
                outer.requests.append({
                    "path": self.path,
                    "model": model,
                    "max_tokens": payload.get("max_tokens"),
                    "authorization": self.headers.get("authorization") or "",
                })
                if model == BAD_MODEL and not outer.serves_all:
                    self._reply(400, _router_body(model).encode())
                elif model == UNAUTHORIZED_MODEL:
                    self._reply(401, b'{"error":{"message":"Unauthorized"}}')
                elif model == SERVER_ERROR_MODEL:
                    self._reply(503, b'{"error":{"message":"upstream down"}}')
                else:
                    # 200 with EMPTY content: the reasoning-budget artefact the
                    # card explicitly says must not read as a bad pin.
                    self._reply(200, json.dumps({
                        "choices": [{"message": {"content": "", "reasoning_content": ""},
                                     "finish_reason": "length"}],
                    }).encode())

            def _reply(self, status: int, body: bytes):
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):  # keep pytest output clean
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def models_probed(self) -> list[str]:
        return [r["model"] for r in self.requests]


@pytest.fixture()
def stub_router():
    router = _StubRouter().start()
    try:
        yield router
    finally:
        router.stop()


@pytest.fixture()
def pin_home(stub_router, monkeypatch):
    """Fresh HERMES_HOME whose providers point at the stub / at a closed port.

    Also seeds the ``alpha`` profile the way a real fleet profile looks: it has
    its own ``config.yaml`` but does not redeclare ``providers``, so the pin
    resolves through the launching home (see
    ``test_profile_provider_entry_wins_over_the_launching_home`` for the other
    direction).
    """
    home = tempfile.mkdtemp(prefix="kanban_route_pin_home_")
    profile_dir = os.path.join(home, "profiles", "alpha")
    os.makedirs(profile_dir, exist_ok=True)
    with open(os.path.join(profile_dir, "config.yaml"), "w", encoding="utf-8") as fh:
        fh.write("model:\n  default: 'ds : deepseek-flash'\n  provider: inference-router\n")
    with open(os.path.join(home, "config.yaml"), "w", encoding="utf-8") as fh:
        fh.write(
            "providers:\n"
            "  inference-router:\n"
            f"    base_url: http://127.0.0.1:{stub_router.port}/v1\n"
            "    api_key: test-router-key\n"
            "  dead-router:\n"
            f"    base_url: http://127.0.0.1:{_closed_port()}/v1\n"
            "    api_key: test-router-key\n"
        )
    monkeypatch.setenv("HERMES_HOME", home)
    for mod in list(sys.modules):
        if mod.startswith(("hermes_cli", "hermes_state")) or mod == "hermes_constants":
            del sys.modules[mod]
    from hermes_cli import kanban_db, kanban_route_pin

    kanban_route_pin.reset_pin_cache()
    try:
        yield kanban_db
    finally:
        kanban_route_pin.reset_pin_cache()


def _dispatch(spawns, **kwargs):
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    def spawn_fn(task, workspace, board=None):
        spawns.append((task, workspace))
        return 4242

    with kbc.connect_closing() as conn:
        return kbd.dispatch_once(conn, spawn_fn=spawn_fn, dry_run=False, **kwargs)


def _events(kb, conn, task_id):
    return kb.list_events(conn, task_id)


def _kinds(kb, conn, task_id):
    return [e.kind for e in _events(kb, conn, task_id)]


def test_bad_pin_is_refused_before_spawn_and_healthy_pin_still_dispatches(pin_home, stub_router):
    """The whole point: bad pin refused + blocked + diagnosed; healthy pin untouched."""
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    from hermes_cli import kanban_diagnostics as kdiag

    kb = pin_home
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        bad_id = kb.create_task(
            conn, title="bad pin", assignee="alpha",
            model_override=BAD_MODEL, provider_override="inference-router")
        good_id = kb.create_task(
            conn, title="good pin", assignee="alpha",
            model_override=GOOD_MODEL, provider_override="inference-router")

    spawns: list = []
    res = _dispatch(spawns, max_spawn=2)

    # Refused, and NOT spawned.
    assert [tid for tid, _model, _msg in res.skipped_bad_pin] == [bad_id]
    assert [t.id for t, _ws in spawns] == [good_id]
    assert [tid for tid, _who, _ws in res.spawned] == [good_id]

    _bad_id, refused_model, message = res.skipped_bad_pin[0]
    assert refused_model == BAD_MODEL
    assert message.startswith(
        f"model '{BAD_MODEL}' is not served by route 'inference-router' (bad pin) — "
        "fix the card's model_override or clear it to fall back to the profile/contract binding")
    # The route's own 400 body rides along as evidence.
    assert "No provider available for model" in message

    with kbc.connect_closing() as conn:
        bad = kb.get_task(conn, bad_id)
        # Blocked, never done: the pin is unfixable by any worker.
        assert bad.status == "blocked"
        assert bad.block_kind == "capability"

        events = _events(kb, conn, bad_id)
        blocked = [e for e in events if e.kind == "blocked"]
        assert len(blocked) == 1
        blocked_payload = blocked[0].payload
        if isinstance(blocked_payload, str):
            blocked_payload = json.loads(blocked_payload)
        assert "bad pin" in blocked_payload["reason"]

        rejections = [e for e in events if e.kind == "model_pin_rejected"]
        assert len(rejections) == 1
        payload = rejections[0].payload
        if isinstance(payload, str):
            payload = json.loads(payload)
        assert payload["model"] == BAD_MODEL
        assert payload["provider"] == "inference-router"
        assert payload["http_status"] == 400
        assert "No provider available for model" in payload["router_error"]

        # The board's live diagnostic surface must tell the operator it was a bad
        # pin (not an outage) and how to fix it.
        diags = kdiag.compute_task_diagnostics(
            bad, events, kb.list_runs(conn, bad_id), config={"kanban": {}})
        bad_pin = [d for d in diags if d.kind == "bad_model_pin"]
        assert bad_pin, [d.kind for d in diags]
        assert bad_pin[0].severity == "error"
        assert BAD_MODEL in bad_pin[0].title
        commands = [a.payload.get("command", "") for a in bad_pin[0].actions]
        assert any(cmd.endswith(f"set-model {bad_id} none") for cmd in commands)

        # The healthy card went to running with its pin forwarded verbatim.
        good = kb.get_task(conn, good_id)
        assert good.status == "running"
        assert good.model_override == GOOD_MODEL
        argv = kbd._worker_argv(good, "alpha", None)
        assert argv[argv.index(GOOD_MODEL) - 1] == "-m"
        assert argv[argv.index("--provider") + 1] == "inference-router"
        assert "model_pin_rejected" not in _kinds(kb, conn, good_id)


def test_probe_failure_is_never_a_bad_pin(pin_home, stub_router):
    """401, 5xx and an unreachable route must all spawn normally.

    A guard that refused here would turn a rotated credential or a router restart
    into a board-wide stall — worse than the bug it fixes.
    """
    from hermes_cli import kanban_db_connect as kbc

    kb = pin_home
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        cases = [
            (kb.create_task(conn, title="401", assignee="alpha",
                            model_override=UNAUTHORIZED_MODEL,
                            provider_override="inference-router"), "inference-router"),
            (kb.create_task(conn, title="503", assignee="alpha",
                            model_override=SERVER_ERROR_MODEL,
                            provider_override="inference-router"), "inference-router"),
            (kb.create_task(conn, title="no listener", assignee="alpha",
                            model_override=GOOD_MODEL,
                            provider_override="dead-router"), "dead-router"),
            # No provider to check against at all: the profile resolves the model.
            (kb.create_task(conn, title="unpinned provider", assignee="alpha",
                            model_override=BAD_MODEL), None),
        ]

    spawns: list = []
    res = _dispatch(spawns, max_spawn=len(cases))

    assert res.skipped_bad_pin == []
    assert sorted(t.id for t, _ws in spawns) == sorted(tid for tid, _prov in cases)
    with kbc.connect_closing() as conn:
        for tid, _prov in cases:
            assert kb.get_task(conn, tid).status == "running"
            assert "model_pin_rejected" not in _kinds(kb, conn, tid)


def test_verdict_is_cached_per_pin(pin_home, stub_router):
    """A dispatch tick stays cheap: one probe per distinct pin, not per tick."""
    from hermes_cli import kanban_db_connect as kbc

    kb = pin_home
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        good_id = kb.create_task(
            conn, title="good pin", assignee="alpha",
            model_override=GOOD_MODEL, provider_override="inference-router")

    spawns: list = []
    _dispatch(spawns, max_spawn=1)
    with kbc.connect_closing() as conn:  # release the claim so it can be re-dispatched
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'ready', claim_lock = NULL WHERE id = ?", (good_id,))
    _dispatch(spawns, max_spawn=1)

    assert stub_router.models_probed().count(GOOD_MODEL) == 1
    assert len(spawns) == 2


def test_classification_policy_is_pure_and_explicit():
    """The policy that decides a refusal, as data — incl. the empty-200 carve-out."""
    from hermes_cli.kanban_route_pin import BAD_PIN, SERVED, UNKNOWN, classify_pin_response

    assert classify_pin_response(400, _router_body(BAD_MODEL)).status == BAD_PIN
    # 200 with EMPTY content is a reasoning-budget artefact, not a dead route.
    assert classify_pin_response(200, json.dumps(
        {"choices": [{"message": {"content": ""}}]})).status == SERVED
    assert classify_pin_response(401, '{"error":{"message":"Unauthorized"}}').status == UNKNOWN
    assert classify_pin_response(503, "upstream down").status == UNKNOWN
    assert classify_pin_response(None, "").status == UNKNOWN
    # A 400 that is NOT the marker (schema/context errors) is not a dead route.
    assert classify_pin_response(
        400, '{"error":{"message":"context length exceeded"}}').status == UNKNOWN


def test_profile_provider_entry_wins_over_the_launching_home(pin_home, stub_router):
    """A profile that redeclares the provider is probed there, not at the launch home.

    The worker resolves its model inside its own profile's config, so a profile
    entry that points elsewhere must decide the verdict.
    """
    from hermes_cli.kanban_route_pin import check_model_pin, reset_pin_cache

    home = os.environ["HERMES_HOME"]
    profile_dir = os.path.join(home, "profiles", "alpha")
    reset_pin_cache()
    # Launch home's route 400s BAD_MODEL → bad pin.
    assert check_model_pin(BAD_MODEL, "inference-router", hermes_home=profile_dir).bad_pin

    serving = _StubRouter(serves_all=True).start()
    try:
        with open(os.path.join(profile_dir, "config.yaml"), "a", encoding="utf-8") as fh:
            fh.write(
                "providers:\n  inference-router:\n"
                f"    base_url: http://127.0.0.1:{serving.port}/v1\n")
        reset_pin_cache()
        # The profile's own entry now decides: the model is served.
        assert not check_model_pin(BAD_MODEL, "inference-router", hermes_home=profile_dir).bad_pin
        assert BAD_MODEL in serving.models_probed()
    finally:
        serving.stop()


def test_fixing_the_pin_clears_the_diagnostic(pin_home, stub_router):
    """The refusal is history the moment the operator re-pins the card."""
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_diagnostics as kdiag

    kb = pin_home
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        bad_id = kb.create_task(
            conn, title="bad pin", assignee="alpha",
            model_override=BAD_MODEL, provider_override="inference-router")

    _dispatch([], max_spawn=1)

    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, bad_id)
        before = kdiag.compute_task_diagnostics(task, _events(kb, conn, bad_id), [], config={})
        assert [d.kind for d in before] == ["bad_model_pin"]

        kb.set_model_override(conn, bad_id, GOOD_MODEL, "inference-router")
        task = kb.get_task(conn, bad_id)
        after = kdiag.compute_task_diagnostics(task, _events(kb, conn, bad_id), [], config={})
        assert [d.kind for d in after] == []
