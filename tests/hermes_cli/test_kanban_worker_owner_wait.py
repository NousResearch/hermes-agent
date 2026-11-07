"""A Kanban worker spawned while `hermes update` drains the owner waits for it instead of failing.

The dispatcher keeps spawning workers during an update window; the owner answers ``draining``
until the update releases the install. Exiting 1 there books a crash against the card, and two
of those trip the breaker and park a card that never ran (the in-flight handoff cell). The wait
must never submit the claim twice, and a non-transient refusal still fails at once.

The fence part (t_90b63d52): an owner that keeps ANSWERING with its fence — discovery
``draining``/``update_paused`` verdicts, in-band ``runtime_draining`` refusals — is alive, and
update pauses routinely run past ``OWNER_WAIT_S``. Such a stretch waits out the card's fence
budget (its ``max_runtime_seconds``), and exhausting it is a distinguishable infrastructure
exit, never a card failure.
"""

import asyncio
import sqlite3
from contextlib import asynccontextmanager, closing

import pytest

from hermes_cli import gateway_client, kanban_worker_client


def _owner(states, submitted):
    @asynccontextmanager
    async def connect():
        state = states.pop(0) if states else None
        if state:
            # The exact refusal connect_gateway raises for a draining/starting owner.
            unavailable = getattr(gateway_client, "GatewayUnavailableError", None)
            if unavailable is None:  # pre-fix tree: the same message as a plain client error
                raise gateway_client.GatewayClientError(f"Gateway {state}: update_paused")
            raise unavailable(f"Gateway {state}: update_paused", state)

        class _Client:
            events = asyncio.Queue()

            async def rpc(self, method, **params):
                submitted.append(method)
                return {"session_id": "s1", "receipt": {"status": "terminal", "admission_id": "a1"}}

        yield _Client()

    return connect


def _fenced_owner(states, submitted):
    """A connect_gateway that refuses with the owner's in-band runtime_draining refusal."""
    @asynccontextmanager
    async def connect():
        state = states.pop(0) if states else None
        if state == "runtime_draining":
            raise gateway_client.GatewayRPCError("runtime_draining")
        if state:
            unavailable = getattr(gateway_client, "GatewayUnavailableError", None)
            raise unavailable(f"Gateway {state}: update_paused", state)

        class _Client:
            events = asyncio.Queue()

            async def rpc(self, method, **params):
                submitted.append(method)
                return {"session_id": "s1", "receipt": {"status": "terminal", "admission_id": "a1"}}

        yield _Client()

    return connect


def _post_wave_owner(states, submitted):
    """A connect_gateway that walks *states*: fence verdicts, 'oserror' (a plain
    restartable cut, not a fence), then a live owner."""
    @asynccontextmanager
    async def connect():
        state = states.pop(0) if states else None
        if state == "oserror":
            raise OSError("connection reset post-wave")
        if state:
            unavailable = getattr(gateway_client, "GatewayUnavailableError", None)
            raise unavailable(f"Gateway {state}: update_paused", state)

        class _Client:
            events = asyncio.Queue()

            async def rpc(self, method, **params):
                submitted.append(method)
                return {"session_id": "s1", "receipt": {"status": "terminal", "admission_id": "a1"}}

        yield _Client()

    return connect


@pytest.mark.asyncio
async def test_worker_waits_out_a_draining_owner_and_submits_once(monkeypatch, tmp_path):
    submitted = []
    monkeypatch.setattr(gateway_client, "connect_gateway", _owner(["draining", "starting"], submitted))
    monkeypatch.setattr(kanban_worker_client.asyncio, "sleep", _no_sleep)
    from gateway import session_kanban
    monkeypatch.setattr(session_kanban, "worker_result", lambda *_: {"exit_code": 0, "last_output": "card closed"})
    monkeypatch.delenv("HERMES_TUI_GATEWAY_URL", raising=False)

    assert await kanban_worker_client.run({"task_id": "t1"}, str(tmp_path / "kanban.db")) == 0
    assert submitted == ["kanban.run"]


@pytest.mark.asyncio
async def test_worker_waits_past_the_silent_window_while_the_wave_answers(monkeypatch, tmp_path):
    """A fence verdict is an owner answer: a wave that outlasts OLD OWNER_WAIT_S minute-for-minute
    never fails the worker — it keeps waiting until the wave lifts, then submits exactly once."""
    submitted = []
    # 40 fence answers, with the silent window shrunk to nothing: under the old
    # silent-only rule every one of these cycles would have failed the worker.
    monkeypatch.setattr(
        gateway_client, "connect_gateway",
        _owner(["draining"] * 20 + ["starting"] * 20, submitted))
    monkeypatch.setattr(kanban_worker_client.asyncio, "sleep", _no_sleep)
    monkeypatch.setattr(kanban_worker_client, "OWNER_WAIT_S", 0.0)
    from gateway import session_kanban
    monkeypatch.setattr(session_kanban, "worker_result", lambda *_: {"exit_code": 0, "last_output": "card closed"})
    monkeypatch.delenv("HERMES_TUI_GATEWAY_URL", raising=False)

    assert await kanban_worker_client.run({"task_id": "t1"}, str(tmp_path / "kanban.db")) == 0
    assert submitted == ["kanban.run"]


@pytest.mark.asyncio
async def test_in_band_runtime_draining_refusal_is_also_a_fence_answer(monkeypatch, tmp_path):
    submitted = []
    monkeypatch.setattr(
        gateway_client, "connect_gateway", _fenced_owner(["runtime_draining"] * 30, submitted))
    monkeypatch.setattr(kanban_worker_client.asyncio, "sleep", _no_sleep)
    monkeypatch.setattr(kanban_worker_client, "OWNER_WAIT_S", 0.0)
    from gateway import session_kanban
    monkeypatch.setattr(session_kanban, "worker_result", lambda *_: {"exit_code": 0, "last_output": "card closed"})
    monkeypatch.delenv("HERMES_TUI_GATEWAY_URL", raising=False)

    assert await kanban_worker_client.run({"task_id": "t1"}, str(tmp_path / "kanban.db")) == 0
    assert submitted == ["kanban.run"]


@pytest.mark.asyncio
async def test_fenced_owner_that_never_lifts_gives_up_as_a_drain(monkeypatch, tmp_path):
    """An owner still advertising its fence past the card's fence budget fails the worker as
    a DRAIN (distinguishable from the silent-owner give-up), and nothing was submitted."""
    submitted = []
    monkeypatch.setattr(
        gateway_client, "connect_gateway", _owner(["draining"] * 200, submitted))
    monkeypatch.setattr(kanban_worker_client.asyncio, "sleep", _no_sleep)
    monkeypatch.setattr(kanban_worker_client, "_fence_budget_s", lambda *_: 0.001)
    monkeypatch.delenv("HERMES_TUI_GATEWAY_URL", raising=False)

    with pytest.raises(kanban_worker_client.OwnerFenceTimeout) as excinfo:
        await kanban_worker_client.run({"task_id": "t1"}, str(tmp_path / "kanban.db"))
    assert "draining" in str(excinfo.value)
    assert "silent for" not in str(excinfo.value)
    assert submitted == []


def test_fenced_giveup_exits_the_drain_sentinel(monkeypatch, tmp_path, capsys):
    """`_submit` surfaces the fenced give-up as ``KANBAN_OWNER_DRAIN_EXIT_CODE`` so the
    dead-worker sweep can classify it from the exit trailer without parsing prose."""
    from hermes_cli import kanban_db as kb

    async def _fenced(*_):
        raise kanban_worker_client.OwnerFenceTimeout(600.0, "Gateway draining: update_paused")

    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_1")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "1")
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "lock")
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "kanban.db"))
    monkeypatch.setattr(kanban_worker_client, "run", _fenced)

    assert kanban_worker_client._submit() == kb.KANBAN_OWNER_DRAIN_EXIT_CODE == 74
    assert "Kanban owner draining" in capsys.readouterr().err


def test_fence_budget_is_the_cards_max_runtime_never_below_the_silent_wait(tmp_path):
    db = tmp_path / "kanban.db"
    with closing(sqlite3.connect(str(db))) as conn:
        conn.execute("CREATE TABLE tasks (id TEXT PRIMARY KEY, max_runtime_seconds INTEGER)")
        conn.execute("INSERT INTO tasks VALUES ('t_limited', 600)")
        conn.execute("INSERT INTO tasks VALUES ('t_unlimited', NULL)")
        conn.commit()

    assert kanban_worker_client._card_max_runtime_seconds("t_limited", str(db)) == 600.0
    assert kanban_worker_client._card_max_runtime_seconds("t_unlimited", str(db)) is None
    assert kanban_worker_client._card_max_runtime_seconds("t_missing", str(db)) is None
    assert kanban_worker_client._card_max_runtime_seconds("t_x", str(tmp_path / "nope.db")) is None
    # A card with a SHORT budget still gets the silent window: the fence budget only
    # ever lengthens what the worker would have waited for a silent owner.
    assert kanban_worker_client._fence_budget_s("t_limited", str(db)) == 600.0
    assert kanban_worker_client._fence_budget_s("t_unlimited", str(db)) == (
        kanban_worker_client.OWNER_FENCE_DEFAULT_S)


@pytest.mark.asyncio
async def test_first_post_wave_nonfence_error_never_fails_instant_silence(monkeypatch, tmp_path):
    """Every fence answer refreshes the silent window: a plain post-wave error (the owner's
    own restart cutting the socket the moment it came back) is judged age < ``OWNER_WAIT_S``
    from the LAST fence answer, not from the anchor of the whole fence stretch."""
    clock = {'t': 1000.0}

    async def _clocked_sleep(_seconds):
        clock['t'] += 30.0  # each backoff cycle advances the fake wall clock 30s

    async def _no_sleep(_seconds):
        return None

    submitted = []
    monkeypatch.setattr(kanban_worker_client.time, 'monotonic', lambda: clock['t'])
    monkeypatch.setattr(kanban_worker_client.asyncio, 'sleep', _clocked_sleep)
    monkeypatch.setattr(kanban_worker_client, 'OWNER_WAIT_S', 180.0)
    monkeypatch.setattr(kanban_worker_client, '_fence_budget_s', lambda *_: 100000.0)
    # 7 fence answers (210s of wave on the fake by the 8th cycle), then the non-fence cut,
    # then the fence again and finally the owner back: the OLD anchor froze claim.answered at
    # the stretch start and failed the worker 'owner silent for 210s' on that exact cut.
    monkeypatch.setattr(
        gateway_client, 'connect_gateway',
        _post_wave_owner(['draining'] * 7 + ['oserror'] + ['draining'] * 3 + [None], submitted))
    from gateway import session_kanban
    monkeypatch.setattr(session_kanban, 'worker_result', lambda *_: {'exit_code': 0, 'last_output': 'ok'})
    monkeypatch.delenv('HERMES_TUI_GATEWAY_URL', raising=False)

    assert await kanban_worker_client.run({'task_id': 't1'}, str(tmp_path / 'kanban.db')) == 0
    assert submitted == ['kanban.run']


@pytest.mark.asyncio
async def test_worker_fails_at_once_when_the_owner_is_not_transiently_down(monkeypatch, tmp_path):
    submitted = []
    monkeypatch.setattr(gateway_client, "connect_gateway", _owner(["inaccessible"], submitted))
    monkeypatch.setattr(kanban_worker_client.asyncio, "sleep", _no_sleep)
    monkeypatch.delenv("HERMES_TUI_GATEWAY_URL", raising=False)

    with pytest.raises(gateway_client.GatewayClientError):
        await kanban_worker_client.run({"task_id": "t1"}, str(tmp_path / "kanban.db"))
    assert submitted == []


async def _no_sleep(_seconds):
    return None
