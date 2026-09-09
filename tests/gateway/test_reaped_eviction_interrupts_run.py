"""Regression test: evicting a session's turn slot must interrupt the in-flight run.

``_hm_evict_reaped_agent`` (``ws_orphan_reap`` / ``agent_close`` / a durable session
row ended while the gateway lived) and the stale-turn path both call
``_hm_evict_running_agent``. Bumping the run generation only makes the gateway
*discard* the eventual result — it does not stop the agent, which kept calling the
model and running tools until it finished on its own. Releasing the slot first also
left ``/stop`` and ``/new`` with nothing to interrupt (``_interrupt_and_clear_session``
reads ``state.turn.agent``), so the orphan could not be stopped by any means.

Reported with a 25-minute / 51-call / 7.9M-input-token orphan in #106963.

Behaviour tests: they pin the contract ("evicting a session interrupts the run it
evicts, before the slot — and the cache — stop pointing at it"), not the shape of
the source.
"""
from __future__ import annotations

from types import SimpleNamespace

from gateway.run import _AGENT_PENDING_SENTINEL
from gateway.run_inbound import GatewayInboundMixin

KEY = "agent:main:telegram:dm:0"


class _RecordingAgent:
    """Minimal agent exposing the ``hard_interrupt`` ABI the gateway uses."""

    def __init__(self, eventos: list[tuple], en_slot) -> None:
        self._eventos = eventos
        self._en_slot = en_slot

    def hard_interrupt(self, message: str | None = None, **kwargs) -> None:
        # ``en_slot`` lets a test assert the interrupt arrived while the turn slot
        # still held this agent — releasing it first is the bug being pinned.
        self._eventos.append(("interrupt", message, self._en_slot() is self))


class _ThrowingAgent:
    """Third-party/legacy ABI that raises instead of latching the interrupt."""

    def hard_interrupt(self, message: str | None = None, **kwargs) -> None:
        raise RuntimeError("legacy interrupt ABI blew up")


class _FakeStore:
    """Durable session store that reports the row as ended — what triggers the
    reaped-session eviction in production."""

    def peek_session_id(self, session_key: str) -> str:
        return "20260909_121453_ee99a4"

    def _is_session_ended_in_db(self, session_id: str) -> bool:
        return True


class _FakeGateway(GatewayInboundMixin):
    """Just enough gateway for the eviction path; the cache mixin's collaborators
    are recorded instead of executed."""

    def __init__(self, eventos: list[tuple]) -> None:
        self._sessions: dict = {}
        self._eventos = eventos
        self.session_store = _FakeStore()
        self.cache: set = set()

    def _peek_session_state(self, session_key: str):
        return self._sessions.get(session_key)

    def _invalidate_session_run_generation(self, session_key: str, reason: str | None = None) -> int:
        self._eventos.append(("invalidate", reason))
        return 2

    def _release_running_agent_state(self, session_key: str, **kwargs) -> bool:
        self._eventos.append(("release", session_key))
        self._sessions.pop(session_key, None)
        return True

    def _evict_cached_agent(self, session_key: str) -> None:
        self._eventos.append(("cache_evict", session_key))
        self.cache.discard(session_key)


def _montar(ocupante: object = "agente"):
    """Gateway con un ocupante en el turn slot (y en el cache); devuelve
    (gateway, agente, eventos)."""
    eventos: list[tuple] = []
    gateway = _FakeGateway(eventos)
    agente = _RecordingAgent(
        eventos,
        lambda: (gateway._sessions.get(KEY) or SimpleNamespace(turn=SimpleNamespace(agent=None))).turn.agent,
    )
    if ocupante == "agente":
        gateway._sessions = {KEY: SimpleNamespace(turn=SimpleNamespace(agent=agente))}
    elif ocupante is not None:
        gateway._sessions = {KEY: SimpleNamespace(turn=SimpleNamespace(agent=ocupante))}
    gateway.cache = {KEY}  # el agente cacheado que un reattach en frío reusaría
    return gateway, agente, eventos


def _orden(eventos: list[tuple], tipo: str) -> int:
    return next(i for i, e in enumerate(eventos) if e[0] == tipo)


def test_evicting_interrupts_while_the_slot_still_holds_the_run() -> None:
    gateway, _agente, eventos = _montar()

    gateway._hm_evict_running_agent(KEY, "reaped_session_eviction")

    interrupciones = [e for e in eventos if e[0] == "interrupt"]
    assert interrupciones, (
        "the evicted run was never interrupted: it keeps calling the model and running tools "
        "after the gateway discarded its result"
    )
    assert interrupciones[0][1], "the interrupt must carry a reason for the transcript"
    assert interrupciones[0][2] is True, (
        "the interrupt must be requested while the turn slot still holds the agent, "
        "otherwise a later /stop cannot reach it"
    )
    assert _orden(eventos, "interrupt") < _orden(eventos, "release"), (
        "interrupt MUST come before _release_running_agent_state empties the slot"
    )
    assert gateway._peek_session_state(KEY) is None
    assert _orden(eventos, "release") < _orden(eventos, "cache_evict"), (
        "the evicted agent must also leave the agent cache, or a cold reattach reuses it "
        "still latched and the next message dies before its first API call (#44212)"
    )
    assert gateway.cache == set()


def test_reaped_session_detection_reaches_the_interrupt() -> None:
    """Same contract through the caller that detects the reaped durable row."""
    gateway, _agente, eventos = _montar()

    gateway._hm_evict_reaped_agent(KEY)

    assert [e for e in eventos if e[0] == "interrupt"], (
        "the reaped-session path evicted the slot without interrupting the run"
    )
    assert gateway.cache == set()


def test_a_raising_interrupt_abi_still_evicts() -> None:
    """A third-party/legacy ``hard_interrupt`` that raises must not skip cleanup:
    invalidate, release and cache evict are the eviction's own contract."""
    eventos: list[tuple] = []
    gateway = _FakeGateway(eventos)
    gateway._sessions = {KEY: SimpleNamespace(turn=SimpleNamespace(agent=_ThrowingAgent()))}
    gateway.cache = {KEY}

    gateway._hm_evict_running_agent(KEY, "reaped_session_eviction")

    tipos = [e[0] for e in eventos]
    assert "invalidate" in tipos and "release" in tipos and "cache_evict" in tipos, (
        f"a raising interrupt left the dead runtime slot reachable: {tipos}"
    )
    assert gateway._peek_session_state(KEY) is None
    assert gateway.cache == set()


def test_pending_sentinel_and_empty_slot_are_not_interrupted() -> None:
    """The placeholder sentinel and an empty slot have nothing to interrupt, and
    the eviction still completes."""
    for ocupante in (_AGENT_PENDING_SENTINEL, None):
        gateway, _agente, eventos = _montar(ocupante)

        gateway._hm_evict_running_agent(KEY, "stale_running_agent_eviction")

        assert [e for e in eventos if e[0] == "interrupt"] == [], f"no debe interrumpir a {ocupante!r}"
        assert [e for e in eventos if e[0] == "release"], "la evicción debe completarse igual"
