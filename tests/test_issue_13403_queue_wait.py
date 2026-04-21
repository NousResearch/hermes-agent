import time
from unittest.mock import patch

from tools.process_registry import ProcessRegistry, ProcessSession


def test_wait_does_not_interrupt_in_queue_mode():
    registry = ProcessRegistry()
    session = ProcessSession(
        id="proc_test_13403",
        command="sleep 1",
        started_at=time.time(),
        exited=False,
        exit_code=None,
        output_buffer="still running",
    )

    refresh_state = {"calls": 0}

    def fake_refresh(s):
        refresh_state["calls"] += 1
        if refresh_state["calls"] >= 2:
            s.exited = True
            s.exit_code = 0
            s.output_buffer = "done"
        return s

    interrupt_state = {"calls": 0}

    def fake_interrupt():
        interrupt_state["calls"] += 1
        return interrupt_state["calls"] == 1

    with patch.object(registry, "get", return_value=session), \
         patch.object(registry, "_refresh_detached_session", side_effect=fake_refresh), \
         patch("tools.process_registry._gateway_busy_input_mode", return_value="queue"), \
         patch("tools.interrupt.is_interrupted", side_effect=fake_interrupt), \
         patch("tools.process_registry.time.sleep", return_value=None):
        result = registry.wait("proc_test_13403", timeout=2)

    assert result["status"] == "exited"
    assert result["exit_code"] == 0


def test_wait_interrupts_when_not_queue_mode():
    registry = ProcessRegistry()
    session = ProcessSession(
        id="proc_test_interrupt",
        command="sleep 1",
        started_at=time.time(),
        exited=False,
        exit_code=None,
        output_buffer="still running",
    )

    with patch.object(registry, "get", return_value=session), \
         patch.object(registry, "_refresh_detached_session", side_effect=lambda s: s), \
         patch("tools.process_registry._gateway_busy_input_mode", return_value="interrupt"), \
         patch("tools.interrupt.is_interrupted", return_value=True), \
         patch("tools.process_registry.time.sleep", return_value=None):
        result = registry.wait("proc_test_interrupt", timeout=2)

    assert result["status"] == "interrupted"
