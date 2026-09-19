import threading

import pytest

from subagent_handles.registry import SubagentHandle, SubagentRegistry, _ALLOWED_STATES


def test_register_and_resolve():
    registry = SubagentRegistry()
    handle = SubagentHandle(
        subagent_id="a1", session_id="s1", goal="g1", parent_subagent_id=None
    )
    registry.register(handle)
    assert registry.resolve("a1") is handle
    assert registry.resolve("missing") is None


def test_set_state():
    registry = SubagentRegistry()
    handle = SubagentHandle(subagent_id="a1", session_id="s1", goal="g1")
    registry.register(handle)
    assert registry.set_state("a1", "done") is True
    assert handle.state == "done"
    assert registry.set_state("missing", "failed") is False


def test_remove():
    registry = SubagentRegistry()
    handle = SubagentHandle(subagent_id="a1", session_id="s1", goal="g1")
    registry.register(handle)
    assert registry.remove("a1") is True
    assert registry.resolve("a1") is None
    assert registry.remove("a1") is False


def test_duplicate_register_raises():
    registry = SubagentRegistry()
    registry.register(SubagentHandle(subagent_id="a1", session_id="s1", goal="g1"))
    with pytest.raises(ValueError, match="Duplicate subagent_id"):
        registry.register(SubagentHandle(subagent_id="a1", session_id="s2", goal="g2"))


def test_empty_subagent_id_raises():
    registry = SubagentRegistry()
    with pytest.raises(ValueError, match="subagent_id must be a non-empty string"):
        registry.register(SubagentHandle(subagent_id="", session_id="s1", goal="g1"))
    with pytest.raises(ValueError, match="subagent_id must be a non-empty string"):
        registry.register(SubagentHandle(subagent_id=None, session_id="s1", goal="g1"))


def test_invalid_state_raises():
    registry = SubagentRegistry()
    registry.register(SubagentHandle(subagent_id="a1", session_id="s1", goal="g1"))
    with pytest.raises(ValueError, match="Invalid state"):
        registry.set_state("a1", "succeeded")


def test_allowed_states_are_exhaustive():
    expected = {"running", "done", "failed", "cancelled"}
    assert _ALLOWED_STATES == expected


def test_subagent_handle_repr_and_eq():
    h1 = SubagentHandle(subagent_id="a1", session_id="s1", goal="g1", state="running")
    h2 = SubagentHandle(subagent_id="a1", session_id="s1", goal="g1", state="running")
    h3 = SubagentHandle(subagent_id="a1", session_id="s2", goal="g1", state="running")
    assert h1 == h2
    assert h1 != h3
    assert repr(h1) == (
        "SubagentHandle(subagent_id='a1', session_id='s1', state='running', role='')"
    )


def test_registry_concurrent_access():
    registry = SubagentRegistry()

    errors: list[Exception] = []

    def worker(i: int) -> None:
        try:
            handle = SubagentHandle(subagent_id=f"a{i}", session_id=f"s{i}", goal=f"g{i}")
            registry.register(handle)
            assert registry.resolve(f"a{i}") is handle
            assert registry.set_state(f"a{i}", "done") is True
            assert handle.state == "done"
            assert registry.remove(f"a{i}") is True
            assert registry.resolve(f"a{i}") is None
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Concurrent errors: {errors}"
