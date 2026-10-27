
"""RED-Beleg: das Verhalten, das ADR-0081 verlangt, gegen den unveraenderten
Hermes-Stand.  Nutzt nur die oeffentliche Gateway-Oberflaeche -- keine der neuen
Module -- damit der Test auf BASELINE und PATCH identisch laeuft.

Erwartung:
  BASELINE (unveraendertes Hermes) -> FAIL (kein Task-Gate / spaetes Ergebnis
  PATCH                            -> PASS
"""
import pytest
from unittest.mock import AsyncMock, patch
from types import SimpleNamespace

SESSION = "agent:main:telegram:dm:12345"

def _runner():
    from gateway.run import GatewayRunner
    r = object.__new__(GatewayRunner)
    r.adapters = {}
    r.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    r._running_agents = {}
    r._pending_messages = {}
    return r

def _evt(**over):
    e = {"type": "completion", "session_id": "proc_x", "session_key": SESSION,
         "platform": "telegram", "chat_type": "dm", "chat_id": "1", "user_id": "1",
         "owner_task_id": "turn-1", "task_id": "turn-1",
         "started_at": 1000.0, "exit_code": 0, "completion_reason": "exited"}
    e.update(over)
    return e

@pytest.mark.asyncio
async def test_late_result_does_not_reactivate_a_finished_task():
    """Nach /stop darf ein spaetes Ergebnis den Auftrag nicht wiederbeleben."""
    from gateway.run import GatewayRunner
    r = _runner()
    gate = getattr(r, "_task_may_start_work", None)
    cancel = getattr(r, "_task_cancel", None)
    assert gate is not None, "ADR-0081: es gibt kein Task-Gate"
    assert cancel is not None, "ADR-0081: es gibt keinen Auftrags-Abbruch"
    # Der Auftrag laeuft und wird dann abgebrochen.
    begin = getattr(r, "_task_begin_turn", None)
    if begin is not None:
        begin(SESSION, is_internal=False, task_id="turn-1")
    with patch("tools.process_registry.process_registry", SimpleNamespace(
            list_sessions=lambda **kw: [], kill_process=lambda *a, **k: {"status": "killed"})):
        cancel(SESSION, reason="user_stop")
    # Das spaete Ergebnis darf den Auftrag nicht wiederbeleben.
    assert gate(_evt()) is False, "ADR-0081: ein spaetes Ergebnis wird zugelassen"
