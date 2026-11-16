"""The classic CLI's ``/queue`` goes into the gateway's durable FIFO (refs #134889, N5).

On main ``/queue <prompt>`` typed while a turn runs only ``put`` the text on the CLI's own
``_pending_input``: no ``session_admissions`` row, so an owner restart (or the N2 disconnect) lost or
failed it. Now the CLI submits it as an ordinary queued admission (``prompt.submit``), the way Ink and
Desktop queue busy input, and ``/queue list`` / ``rm`` / ``clear`` read and change the gateway's queue.

Real owners (``initialize_session_authority`` again on the same state.db is the restart), real
``AuthorityConnection.dispatch``, real ``GatewayConnection`` and ``GatewaySessionStore``. The one stub:
the first owner's drain is held (``_schedule`` no-op), which is what a running turn does to the FIFO.
"""
import asyncio
import queue
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli.gateway_client import rpc_error
from hermes_cli.gateway_connection import GatewayConnection
from hermes_cli.gateway_session_store import GatewaySessionStore

TEXT = "N5 queued while the turn runs"


class _Life:
    def __init__(self, connection, loop):
        self.connection, self.loop = connection, loop
        self.events = asyncio.Queue()

    async def rpc(self, method, *, _timeout=None, **params):
        frame = {'id': 1, 'method': method, 'params': params}
        reply = await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(self.connection.dispatch(frame), self.loop))
        if 'error' in reply:
            raise rpc_error(reply['error'])
        return reply['result']


class _Owners:
    async def start(self, tmp_path, monkeypatch):
        from gateway import run
        from gateway.config import GatewayConfig
        from gateway.session import SessionStore
        from gateway.session_controls import AuthorityConnection
        from gateway.session_local import create_local_session
        monkeypatch.setattr(run, '_load_gateway_config', lambda *a, **k: {'platform_toolsets': {'cli': []}})
        self.profile, self.loop, self.lives = str(tmp_path), asyncio.get_running_loop(), []
        self.store = SessionStore(tmp_path / 'sessions', GatewayConfig())
        first = await self.boot('first')
        first._schedule = lambda ref: None  # a turn is running: the FIFO does not drain
        actor = AuthorityConnection(first, object(), {'user_id': 'owner'}).actor
        self.sid = create_local_session(first, actor, {'cwd': self.profile, 'model': 'm', 'toolsets': [],
                                                       'request_id': 'n5'}).session_id
        return self

    async def boot(self, instance_id):
        from gateway.session_authority import initialize_session_authority
        owner = SimpleNamespace(session_store=self.store, _session_db=self.store._db, adapters={}, _draining=False,
                                _cached_agent_for=lambda route: None, _evict_cached_agent=lambda route: None)
        owner._adapter_for_source = lambda source: owner.adapters.get(source.platform)
        self.current = await initialize_session_authority(owner, profile_id=self.profile, instance_id=instance_id)
        return self.current

    async def _open(self):
        from gateway.session_controls import AuthorityConnection
        self.lives.append(_Life(AuthorityConnection(self.current, object(), {'user_id': 'owner'}), self.loop))
        return self.lives[-1]

    @asynccontextmanager
    async def dial(self):
        yield await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(self._open(), self.loop))

    def rows(self):
        from hermes_state_runtime import list_session_admissions
        return [row for row in list_session_admissions(self.store._db, session_id=self.sid, pending_only=False)
                if row['payload'].get('text') == TEXT]

    async def close(self, conn):
        await asyncio.to_thread(conn.close)
        for life in self.lives:
            await life.connection.close()
        self.store._db.close()


def _busy_cli(store, sid):
    from cli import HermesCLI
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = sid
    cli._pending_input = queue.Queue()
    cli._pending_resume_sessions = None
    cli._session_db = store
    cli._agent_running = True
    return cli


def _command(cli, text):
    with patch("cli._cprint") as printed:
        assert cli.process_command(text) is True
    return " ".join(str(call.args[0]) for call in printed.call_args_list)


@pytest.mark.asyncio
async def test_queue_while_busy_is_a_durable_admission_that_survives_an_owner_restart(tmp_path, monkeypatch):
    owners = await _Owners().start(tmp_path, monkeypatch)
    conn = GatewayConnection(connect=owners.dial)
    try:
        store = GatewaySessionStore(conn)
        await asyncio.to_thread(store.attach, owners.sid)
        cli = _busy_cli(store, owners.sid)
        await asyncio.to_thread(_command, cli, f"/queue {TEXT}")
        # Not held client-side: one queued admission in state.db, nothing for the CLI to send again.
        assert cli._pending_input.empty()
        assert [row['status'] for row in owners.rows()] == ['queued']
        listed = await asyncio.to_thread(_command, cli, "/queue list")
        assert TEXT in listed

        await owners.boot('restarted')  # the owner restarts; state.db keeps the FIFO
        assert [row['status'] for row in owners.rows()] == ['queued']
        assert len({row['admission_id'] for row in owners.rows()}) == 1  # exactly once, never re-submitted
        # The restarted owner lists it to a viewer that attaches now: it is the next turn it runs.
        fresh = GatewayConnection(connect=owners.dial)
        try:
            snapshot = await asyncio.to_thread(GatewaySessionStore(fresh).attach, owners.sid)
            assert [(row['text'], row['status']) for row in snapshot['pending']] == [(TEXT, 'queued')]
        finally:
            await asyncio.to_thread(fresh.close)
    finally:
        await owners.close(conn)


@pytest.mark.asyncio
async def test_queue_rm_and_clear_retire_the_gateway_admissions(tmp_path, monkeypatch):
    owners = await _Owners().start(tmp_path, monkeypatch)
    conn = GatewayConnection(connect=owners.dial)
    try:
        store = GatewaySessionStore(conn)
        await asyncio.to_thread(store.attach, owners.sid)
        cli = _busy_cli(store, owners.sid)
        await asyncio.to_thread(_command, cli, f"/queue {TEXT}")
        await asyncio.to_thread(_command, cli, "/queue second item")
        await asyncio.to_thread(_command, cli, "/queue rm 1")
        assert [row['status'] for row in owners.rows()] == ['terminal']
        listed = await asyncio.to_thread(_command, cli, "/queue list")
        assert "second item" in listed and TEXT not in listed
        await asyncio.to_thread(_command, cli, "/queue clear")
        assert "empty" in (await asyncio.to_thread(_command, cli, "/queue list")).lower()
    finally:
        await owners.close(conn)


def test_the_owner_claim_of_a_queued_prompt_runs_as_a_typed_turn_without_a_second_submit():
    """When the owner starts the queued admission, the idle CLI paints it as the user's turn and
    ``run_conversation`` pumps that admission (no new ``prompt.submit``: it runs exactly once)."""
    from hermes_cli.gateway_agent import GatewayAgent
    from tests.hermes_cli.test_gateway_agent import SID, frame
    from tests.hermes_cli.test_gateway_tools_notify import _connected, _idle_cli, _play
    conn, client = _connected()
    try:
        cli = _idle_cli(conn)
        conn.state(SID).queued_inputs["adm-q"] = TEXT
        _play(conn, client, [frame("message.start", {}, admission="adm-q"),
                             frame("message.delta", {"text": "ran"}, admission="adm-q"),
                             frame("message.complete", {"text": "ran", "status": "complete"}, admission="adm-q")])
        cli._drain_process_notifications("cli-idle")
        item = cli._pending_input.get_nowait()
        assert str(item) == TEXT and type(item).__name__ == "_SeededQueryMessage"  # literal, never a command
        cli._tui_unwrap_input(item)  # the input loop takes it: its admission is armed by id
        agent = GatewayAgent(connection=conn, session_id=SID)
        agent._pending_cli_user_message = {"role": "user", "content": TEXT}
        result = agent.run_conversation(TEXT, conversation_history=[])
        assert "prompt.submit" not in [method for method, _ in client.calls]
        assert result["final_response"] == "ran"
        assert conn.state(SID).queued_inputs == {}  # cleared once its turn is the one painted
    finally:
        conn.close()


def test_an_adopted_queued_prompt_never_submits_a_second_admission():
    """Review N5-F3: the adopted item is matched to its admission by id, not by text. A queued text the
    input path's cleanup rewrites (a leaked CPR, a bracketed-paste marker) must still be pumped as the
    admission the owner is running, never sent again as a new ``prompt.submit``."""
    from cli import _strip_leaked_bracketed_paste_wrappers, _strip_leaked_terminal_responses_with_meta
    from hermes_cli.gateway_agent import GatewayAgent
    from tests.hermes_cli.test_gateway_agent import SID, frame
    from tests.hermes_cli.test_gateway_tools_notify import _connected, _idle_cli, _play
    text = "compare cursor log ^[[12;40R with the expected rows [200~"
    conn, client = _connected()
    try:
        cli = _idle_cli(conn)
        conn.state(SID).queued_inputs["adm-q"] = text
        _play(conn, client, [frame("message.start", {}, admission="adm-q"),
                             frame("message.complete", {"text": "ran", "status": "complete"}, admission="adm-q")])
        cli._drain_process_notifications("cli-idle")
        client.script = [frame("message.complete", {"text": "again", "status": "complete"}, admission="adm-1")]
        item = cli._pending_input.get_nowait()
        user_input, _, seeded = cli._tui_unwrap_input(item)
        assert seeded and user_input == text
        # Even if a sanitizer still rewrote it (chat() strips lone surrogates), identity decides.
        user_input = _strip_leaked_bracketed_paste_wrappers(user_input)
        user_input, _ = _strip_leaked_terminal_responses_with_meta(user_input)
        agent = GatewayAgent(connection=conn, session_id=SID)
        agent._pending_cli_user_message = {"role": "user", "content": user_input}
        result = agent.run_conversation(user_input, conversation_history=[])
        assert "prompt.submit" not in [m for m, _ in client.calls], "second admission submitted"
        assert result["final_response"] == "ran"
    finally:
        conn.close()


def test_a_typed_turn_with_the_same_text_is_never_taken_for_the_adopted_admission():
    """Identity cuts both ways: a typed input that happens to equal an adopted queued text, run before the
    adopted item was unwrapped, is its own new submit."""
    from hermes_cli.gateway_agent import GatewayAgent
    from tests.hermes_cli.test_gateway_agent import SID, frame
    from tests.hermes_cli.test_gateway_tools_notify import _connected, _idle_cli, _play
    conn, client = _connected()
    try:
        cli = _idle_cli(conn)
        conn.state(SID).queued_inputs["adm-q"] = TEXT
        _play(conn, client, [frame("message.start", {}, admission="adm-q")])
        cli._drain_process_notifications("cli-idle")
        client.script = [frame("message.complete", {"text": "typed", "status": "complete"}, admission="adm-1")]
        agent = GatewayAgent(connection=conn, session_id=SID)
        agent._pending_cli_user_message = {"role": "user", "content": TEXT}
        agent.run_conversation(TEXT, conversation_history=[])
        assert "prompt.submit" in [m for m, _ in client.calls]
    finally:
        conn.close()


def _run_adopted(cli, conn):
    """The input loop takes the adopted item and chat() runs it: ``run_conversation`` result."""
    from hermes_cli.gateway_agent import GatewayAgent
    from tests.hermes_cli.test_gateway_agent import SID
    item = cli._pending_input.get_nowait()
    text, _, seeded = cli._tui_unwrap_input(item)
    assert seeded and item.admission_id
    agent = GatewayAgent(connection=conn, session_id=SID)
    agent._pending_cli_user_message = {"role": "user", "content": text}
    return agent.run_conversation(text, conversation_history=[])


def test_a_local_turn_does_not_swallow_the_start_of_a_cli_queued_admission():
    """Review N5-F2: the owner started the CLI's queued admission, but a local input (busy-mode Enter,
    a /goal continuation) reached run_conversation before the idle tick adopted it. Its drain must not
    lose that turn: the idle tick after the local turn paints it from its receipt (no second submit),
    and only then is its ``queued_inputs`` entry cleared."""
    from hermes_cli.gateway_agent import GatewayAgent
    from tests.hermes_cli.test_gateway_agent import SID, frame
    from tests.hermes_cli.test_gateway_tools_notify import _connected, _idle_cli, _play
    conn, client = _connected()
    try:
        cli = _idle_cli(conn)
        state = conn.state(SID)
        state.queued_inputs["adm-q"] = TEXT
        _play(conn, client, [frame("message.start", {}, admission="adm-q"),
                             frame("message.delta", {"text": "queued reply"}, admission="adm-q")])
        client.script = [frame("message.complete", {"text": "local", "status": "complete"}, admission="adm-1")]
        agent = GatewayAgent(connection=conn, session_id=SID)
        agent._pending_cli_user_message = {"role": "user", "content": "local"}
        agent.run_conversation("local", conversation_history=[])
        assert "adm-q" in state.queued_inputs  # not painted yet: still owed
        submits = [m for m, _ in client.calls].count("prompt.submit")
        client.result = {"final_response": "queued reply"}
        cli._drain_process_notifications("cli-post-turn")
        result = _run_adopted(cli, conn)
        assert [m for m, _ in client.calls].count("prompt.submit") == submits  # never submitted again
        assert result["final_response"] == "queued reply"
        assert state.queued_inputs == {}
    finally:
        conn.close()


def test_a_queued_admission_another_client_cancelled_is_forgotten_not_painted():
    """A remembered admission that settled without running (cancelled by Ink/Desktop) is dropped from
    ``queued_inputs`` on the next published list; nothing is painted for it."""
    from tests.hermes_cli.test_gateway_agent import SID
    from tests.hermes_cli.test_gateway_tools_notify import _connected, _idle_cli
    conn, client = _connected()
    try:
        cli = _idle_cli(conn)
        state = conn.state(SID)
        state.queued_inputs["adm-q"] = TEXT
        real = client.rpc

        async def rpc(method, **params):
            if method == "prompt.receipt":
                client.calls.append((method, params))
                return {"admission_id": "adm-q", "status": "terminal", "outcome": "cancelled"}
            return await real(method, **params)
        client.rpc = rpc
        state.apply_event("session.info", {"pending": []})
        cli._drain_process_notifications("cli-idle")
        assert cli._pending_input.empty() and state.queued_inputs == {}
    finally:
        conn.close()


def test_queue_list_never_erases_a_finished_queued_turn_this_viewer_has_not_painted():
    """Greptile P1 (gateway_queue.py:62): a fast queued turn settles while this CLI is still busy.
    ``/queue list`` reads its receipt as terminal; that must not drop the tracking the idle tick needs
    to paint the turn whose frames are still buffered."""
    from hermes_cli import gateway_queue
    from tests.hermes_cli.test_gateway_agent import SID, frame
    from tests.hermes_cli.test_gateway_tools_notify import _connected, _idle_cli, _play
    conn, client = _connected()
    try:
        cli = _idle_cli(conn)
        conn.state(SID).queued_inputs["adm-q"] = TEXT
        _play(conn, client, [frame("message.start", {}, admission="adm-q"),
                             frame("message.delta", {"text": "fast"}, admission="adm-q"),
                             frame("message.complete", {"text": "fast", "status": "complete"}, admission="adm-q")])
        assert gateway_queue.queued_rows(cli, cli._session_db) == []  # receipt: terminal, not queued
        cli._drain_process_notifications("cli-idle")
        result = _run_adopted(cli, conn)
        assert "prompt.submit" not in [m for m, _ in client.calls]
        assert result["final_response"] == "fast"
        assert conn.state(SID).queued_inputs == {}
    finally:
        conn.close()



def test_a_queued_turn_the_owner_ran_before_this_viewer_reattached_is_still_painted():
    """Greptile P1 (gateway_tools_notify.py:76): after an owner restart the drain can run the queued
    admission before this CLI resubscribes; ``session.resume`` replays no frames. The resumed pending
    list no longer lists it, so the idle tick reads its receipt and paints the saved reply (no pump,
    no submit)."""
    from tests.hermes_cli.test_gateway_agent import SID
    from tests.hermes_cli.test_gateway_tools_notify import _connected, _idle_cli
    conn, client = _connected()
    try:
        cli = _idle_cli(conn)
        state = conn.state(SID)
        state.queued_inputs["adm-q"] = TEXT
        client.result = {"final_response": "ran while you were away"}
        state.adopt_snapshot({"stored_session_id": SID, "pending": []})  # the reconnect's resume
        cli._drain_process_notifications("cli-idle")
        result = _run_adopted(cli, conn)
        assert "prompt.submit" not in [m for m, _ in client.calls]
        assert result["final_response"] == "ran while you were away"
        assert state.queued_inputs == {}
    finally:
        conn.close()


def test_dock_drops_a_cli_queued_admission_another_client_cancelled():
    """Review N5-F1: another client retired the CLI's queued admission (prompt.cancel, replace, steer).
    The owner's next published pending list omits it; the dock must stop listing it as queued."""
    from hermes_cli.cli_session_dock import queued_prompts
    from tests.hermes_cli.test_gateway_agent import SID
    from tests.hermes_cli.test_gateway_tools_notify import _connected, _idle_cli
    conn, _client = _connected()
    try:
        cli = _idle_cli(conn)
        state = conn.state(SID)
        state.queued_inputs["adm-q"] = TEXT
        state.apply_event("session.info", {"pending": [{"admission_id": "adm-q", "status": "queued",
                                                         "sequence": 1, "text": TEXT}]})
        assert queued_prompts(cli) == [TEXT]
        state.apply_event("session.info", {"pending": []})  # another client cancelled it
        assert queued_prompts(cli) == [], "retired admission still painted as queued"
    finally:
        conn.close()


def test_dock_shows_a_fresh_submit_until_the_owner_publishes_its_list():
    """The gap between the submit's receipt and the owner's next pending list: the new admission is shown;
    the first list published after it decides."""
    from hermes_cli import gateway_queue
    from hermes_cli.cli_session_dock import queued_prompts
    from tests.hermes_cli.test_gateway_agent import SID
    from tests.hermes_cli.test_gateway_tools_notify import _connected, _idle_cli
    conn, _client = _connected()
    try:
        cli = _idle_cli(conn)
        state = conn.state(SID)
        state.apply_event("session.info", {"pending": []})
        gateway_queue.submit_queued(cli, cli._session_db, TEXT)  # FakeClient admits it as adm-1
        assert queued_prompts(cli) == [TEXT]
        state.apply_event("session.info", {"pending": [{"admission_id": "adm-1", "status": "queued",
                                                         "sequence": 1, "text": TEXT}]})
        assert queued_prompts(cli) == [TEXT]
        state.apply_event("session.info", {"pending": []})
        assert queued_prompts(cli) == []
    finally:
        conn.close()


def _queue_command(cli, fn, *args):
    with patch("cli._cprint") as printed:
        fn(*args)
    return " ".join(str(c.args[0]) for c in printed.call_args_list)


def _owned_cli():
    from tests.hermes_cli.test_gateway_agent import SID
    from tests.hermes_cli.test_gateway_tools_notify import _connected, _idle_cli
    conn, _client = _connected()
    cli = _idle_cli(conn)
    cli._agent_running = True
    conn.state(SID).queued_inputs["adm-q"] = TEXT
    cli._mutate_pending_input = lambda fn: ([], [])
    return conn, cli


@pytest.mark.parametrize("verb, args", [("_queue_remove", ("1",)), ("_queue_edit", ("1 new text",))])
def test_a_refused_queue_rm_or_edit_says_what_was_refused(verb, args):
    """Review N5-F4: an rm/edit the owner refuses (unsupported_control, ...) gets its own message, never
    the submit refusal's "Nothing was queued; send it again"."""
    from hermes_cli.gateway_client import GatewayRPCError
    conn, cli = _owned_cli()
    try:
        with patch("hermes_cli.gateway_queue.queued_rows", lambda c, s: [("adm-q", TEXT)]), \
             patch("hermes_cli.gateway_queue.cancel", side_effect=GatewayRPCError("unsupported_control")), \
             patch("hermes_cli.gateway_queue.replace", side_effect=GatewayRPCError("unsupported_control")):
            out = _queue_command(cli, getattr(cli, verb), *args)
        assert "Nothing was queued" not in out and "send it again" not in out, out
        assert "queue item 1" in out.lower() and "unsupported_control" in out, out
    finally:
        conn.close()


@pytest.mark.parametrize("verb, args", [("_queue_remove", ("1",)), ("_queue_edit", ("1 new text",))])
def test_an_rm_or_edit_of_a_row_the_drain_already_started_says_it_started(verb, args):
    """The drain claimed the row between the list and the verb (stale_generation): the user is told it
    already started; its turn is still owed a paint."""
    from hermes_cli.gateway_client import GatewayRPCError
    from tests.hermes_cli.test_gateway_agent import SID
    conn, cli = _owned_cli()
    try:
        with patch("hermes_cli.gateway_queue.queued_rows", lambda c, s: [("adm-q", TEXT)]), \
             patch("hermes_cli.gateway_queue.cancel", side_effect=GatewayRPCError("stale_generation")), \
             patch("hermes_cli.gateway_queue.replace", side_effect=GatewayRPCError("stale_generation")):
            out = _queue_command(cli, getattr(cli, verb), *args)
        assert "already started" in out, out
        assert "adm-q" in conn.state(SID).queued_inputs
    finally:
        conn.close()


def test_an_rm_racing_the_drain_claim_tells_the_user_the_gateway_stopped_it():
    """Owner ``prompt.cancel`` interrupts a row the drain claimed between its read and the cancel commit and
    answers a non-terminal receipt. That rm used to print nothing and forget the admission."""
    from tests.hermes_cli.test_gateway_agent import SID
    conn, cli = _owned_cli()
    try:
        real = cli._session_db.connection.call

        def call(method, **params):
            if method == "prompt.cancel":
                return {"admission_id": "adm-q", "status": "started"}
            return real(method, **params)
        cli._session_db.connection.call = call
        with patch("hermes_cli.gateway_queue.queued_rows", lambda c, s: [("adm-q", TEXT)]):
            out = _queue_command(cli, cli._queue_remove, "1")
        assert "already started" in out and "stopped" in out, out
        assert "adm-q" in conn.state(SID).queued_inputs  # its interrupted turn is still painted
    finally:
        conn.close()


def _flaky_submit(client, failures):
    """``prompt.submit`` loses its answer ``failures`` times (sent, no reply), then admits ``adm-q``."""
    from hermes_cli.gateway_client import GatewayClientError
    real, seen = client.rpc, []

    async def rpc(method, **params):
        if method != "prompt.submit":
            return await real(method, **params)
        client.calls.append((method, params))
        seen.append(params["submission_id"])
        if len(seen) <= failures:
            raise GatewayClientError("Gateway did not answer prompt.submit")
        return {"admission_id": "adm-q", "status": "queued"}
    client.rpc = rpc
    return seen


def test_a_queue_submit_whose_answer_was_lost_is_retried_under_the_same_identity():
    """Greptile P1 (cli_loops_mixin.py:383): a timeout or disconnect after the submit was sent is not a
    refusal; the owner may already hold the prompt. The CLI asks again with the SAME submission id (the
    owner answers the existing admission), so the prompt is queued exactly once."""
    from tests.hermes_cli.test_gateway_agent import SID
    from tests.hermes_cli.test_gateway_tools_notify import _connected, _idle_cli
    conn, client = _connected()
    try:
        cli = _idle_cli(conn)
        cli._agent_running = True
        cli._expand_paste_references = lambda text: text
        seen = _flaky_submit(client, 1)
        out = _queue_command(cli, cli._queue_enqueue, TEXT)
        assert len(seen) == 2 and seen[0] == seen[1]
        assert "Nothing was queued" not in out
        assert conn.state(SID).queued_inputs == {"adm-q": TEXT}
    finally:
        conn.close()


def test_a_queue_submit_with_no_answer_at_all_never_says_nothing_was_queued():
    from tests.hermes_cli.test_gateway_agent import SID
    from tests.hermes_cli.test_gateway_tools_notify import _connected, _idle_cli
    conn, client = _connected()
    try:
        cli = _idle_cli(conn)
        cli._agent_running = True
        cli._expand_paste_references = lambda text: text
        _flaky_submit(client, 2)
        out = _queue_command(cli, cli._queue_enqueue, TEXT)
        assert "Nothing was queued" not in out and "send it again" not in out, out
        assert "/queue list" in out
        assert conn.state(SID).queued_inputs == {}
    finally:
        conn.close()


def test_a_queue_submit_the_owner_refused_still_says_nothing_was_queued():
    from hermes_cli.gateway_client import GatewayRPCError
    from tests.hermes_cli.test_gateway_tools_notify import _connected, _idle_cli
    conn, client = _connected()
    try:
        cli = _idle_cli(conn)
        cli._agent_running = True
        cli._expand_paste_references = lambda text: text
        real = client.rpc

        async def rpc(method, **params):
            if method == "prompt.submit":
                client.calls.append((method, params))
                raise GatewayRPCError("queue_full")
            return await real(method, **params)
        client.rpc = rpc
        out = _queue_command(cli, cli._queue_enqueue, TEXT)
        assert "Nothing was queued" in out and "queue_full" in out
        assert [m for m, _ in client.calls].count("prompt.submit") == 1  # a verdict is never retried
    finally:
        conn.close()


def _busy_enter(cli, mode, text, images=()):
    """Enter typed while the turn runs, through main's ``_tui_enter_while_busy``."""
    cli.busy_input_mode = mode
    cli.agent = getattr(cli, "agent", None)
    with patch("agent.onboarding.is_seen", lambda *a: True), patch("cli._cprint"):
        cli._tui_enter_while_busy(text, list(images), (text, list(images)) if images else text)


@pytest.mark.asyncio
async def test_probe_queue_mode_enter_then_queue_keep_entry_order(tmp_path, monkeypatch):
    """Review round 2, major 1: a queue-mode Enter typed before ``/queue X`` used to stay in
    ``_pending_input`` while X went to the owner's FIFO, so X ran first (and the typed item had no
    ``session_admissions`` row). With a gateway store attached it is an admission too, in entry order."""
    owners = await _Owners().start(tmp_path, monkeypatch)
    conn = GatewayConnection(connect=owners.dial)
    try:
        store = GatewaySessionStore(conn)
        await asyncio.to_thread(store.attach, owners.sid)
        cli = _busy_cli(store, owners.sid)
        await asyncio.to_thread(_busy_enter, cli, "queue", "typed first")
        await asyncio.to_thread(_command, cli, f"/queue {TEXT}")
        assert cli._pending_input.empty()
        listed = await asyncio.to_thread(_command, cli, "/queue list")
        assert listed.index("typed first") < listed.index(TEXT), f"entry order lost: {listed}"
    finally:
        await owners.close(conn)


def test_a_steer_that_falls_back_to_queue_is_a_gateway_admission_and_images_stay_local():
    """The steer fallback inherits the queue-mode path; an image follow-up (no text-only admission) and
    a run with no gateway store keep main's ``_pending_input``."""
    from tests.hermes_cli.test_gateway_agent import SID
    from tests.hermes_cli.test_gateway_tools_notify import _connected, _idle_cli
    conn, client = _connected()
    try:
        cli = _idle_cli(conn)
        cli._agent_running = True
        cli.agent = SimpleNamespace(steer=lambda text: False)  # the running turn refused the steer
        _busy_enter(cli, "steer", "steer me")
        assert cli._pending_input.empty() and list(conn.state(SID).queued_inputs.values()) == ["steer me"]
        _busy_enter(cli, "queue", "look", images=["/img.png"])
        assert cli._pending_input.get_nowait() == ("look", ["/img.png"])
        submits = [m for m, _ in client.calls].count("prompt.submit")
        assert submits == 1
    finally:
        conn.close()
    local = _idle_cli(conn)
    local._session_db = object()  # not a gateway viewer
    local._agent_running = True
    _busy_enter(local, "queue", "plain")
    assert local._pending_input.get_nowait() == "plain"



def _run_loop(cli, until):
    """Run main's ``_tui_process_loop`` in a thread; returns ``(processed, stop)``."""
    import threading
    import time
    processed = []

    def one(item):
        processed.append((time.monotonic(), item))
        if len(processed) >= until:
            cli._should_exit = True
    cli._should_exit = False
    cli._tui_process_one_input = one
    cli._tui_idle_tick = lambda: cli._drain_process_notifications("cli-idle")
    thread = threading.Thread(target=cli._tui_process_loop, daemon=True)
    thread.start()

    def stop(timeout):
        thread.join(timeout)
        cli._should_exit = True
        thread.join(2)
    return processed, stop


def test_probe_slash_command_typed_after_queue_runs_after_it():
    """Review round 2, major 2: ``/queue X`` then ``/new`` typed while busy. On main both sat in
    ``_pending_input`` in that order; with X in the owner's FIFO the input loop must still run X first,
    not let ``/new`` detach the session before the owner claims X."""
    from tests.waiting import wait_until
    from tests.hermes_cli.test_gateway_agent import SID, frame
    from tests.hermes_cli.test_gateway_tools_notify import _connected, _idle_cli
    conn, client = _connected()
    try:
        cli = _idle_cli(conn)
        state = conn.state(SID)
        state.queued_inputs["adm-q"] = TEXT
        state.apply_event("session.info", {"running": True, "pending": [
            {"admission_id": "adm-1", "status": "started", "sequence": 1, "text": "turn"},
            {"admission_id": "adm-q", "status": "queued", "sequence": 2, "text": TEXT}]})
        cli._pending_input.put("/new")  # typed while the CLI's turn ran; that turn just returned
        processed, stop = _run_loop(cli, 2)
        wait_until(lambda: cli._pending_input.empty(), timeout=5)  # the loop took /new and holds it
        # The owner settles the turn, then claims X.
        frames = [frame("session.info", {"running": False, "pending": [
                      {"admission_id": "adm-q", "status": "queued", "sequence": 2, "text": TEXT}]}, admission=None),
                  frame("message.start", {}, admission="adm-q")]

        async def put():  # not _play: the held loop consumes the frames as they land
            for item in frames:
                await client.events.put(item)
        asyncio.run_coroutine_threadsafe(put(), conn.loop).result(5)
        stop(5)
        order = [str(item) for _, item in processed]
        assert order == [TEXT, "/new"], f"input loop runs {order[0]!r} before the /queue prompt"
        assert getattr(processed[0][1], "admission_id", None) == "adm-q"
    finally:
        conn.close()


@pytest.mark.parametrize("pending, running", [
    ([{"admission_id": "adm-0", "status": "unknown", "sequence": 1, "text": "lost"},
      {"admission_id": "adm-q", "status": "queued", "sequence": 2, "text": TEXT}], False),
    ([{"admission_id": "adm-q", "status": "queued", "sequence": 2, "text": TEXT}], False),
    ([{"admission_id": "adm-q", "status": "queued", "sequence": 2, "text": TEXT}], True),
])
def test_typing_stays_responsive_when_the_owner_queue_is_paused(pending, running):
    """The hold is bounded: a FIFO paused on an ``unknown`` row runs a typed command at once, and an owner
    that never claims this viewer's admission (idle, or busy with someone else's turn) delays it by at
    most ``OWN_QUEUE_HOLD_S``; the admission stays queued, never dropped."""
    import time
    from hermes_cli.gateway_queue import OWN_QUEUE_HOLD_S
    from tests.hermes_cli.test_gateway_agent import SID
    from tests.hermes_cli.test_gateway_tools_notify import _connected, _idle_cli
    assert OWN_QUEUE_HOLD_S >= 2.0
    conn, _client = _connected()
    try:
        cli = _idle_cli(conn)
        state = conn.state(SID)
        state.queued_inputs["adm-q"] = TEXT
        state.apply_event("session.info", {"running": running, "pending": pending})
        started = time.monotonic()
        cli._pending_input.put("/help")
        processed, stop = _run_loop(cli, 1)
        stop(OWN_QUEUE_HOLD_S + 2)
        assert [str(item) for _, item in processed] == ["/help"]
        waited = processed[0][0] - started
        assert waited < (0.5 if pending[0]["status"] == "unknown" else OWN_QUEUE_HOLD_S + 1.0), waited
        assert state.queued_inputs == {"adm-q": TEXT}
    finally:
        conn.close()
