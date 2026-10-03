import os
import sys
import tempfile

import pytest

from subagent_handles.persister import SessionPersister
from subagent_handles.registry import SubagentHandle, SubagentRegistry


def test_checkpoint_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        p = SessionPersister(os.path.join(tmp, "sessions"))
        h = SubagentHandle(subagent_id="a1", session_id="s1", goal="g1", state="running")
        p.checkpoint(h)
        loaded = p.load("a1")
        assert loaded is not None
        assert loaded.subagent_id == "a1"
        assert loaded.session_id == "s1"
        assert loaded.state == "running"


def test_remove():
    with tempfile.TemporaryDirectory() as tmp:
        p = SessionPersister(os.path.join(tmp, "sessions"))
        p.checkpoint(SubagentHandle(subagent_id="a1", session_id="s1", goal="g1"))
        assert p.remove("a1") is True
        assert p.load("a1") is None
        assert p.remove("a1") is False


def test_restore_into_registry():
    with tempfile.TemporaryDirectory() as tmp:
        p = SessionPersister(os.path.join(tmp, "sessions"))
        p.checkpoint(SubagentHandle(subagent_id="a1", session_id="s1", goal="g1"))
        p.checkpoint(SubagentHandle(subagent_id="a2", session_id="s2", goal="g2", state="done"))
        registry = SubagentRegistry()
        restored = p.restore(registry)
        assert set(restored.keys()) == {"a1", "a2"}
        # Stale "running" handles are reconciled to "failed" on restore
        # (the process that wrote them has since exited). "done" handles
        # are preserved as-is.
        assert registry.resolve("a1").state == "failed"
        assert registry.resolve("a2").state == "done"


def test_restore_skips_bad_file():
    import tempfile as _temp
    with _temp.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "sessions")
        os.makedirs(root, exist_ok=True)
        with open(os.path.join(root, "a1.json"), "w", encoding="utf-8") as f:
            f.write("{}")
        with open(os.path.join(root, "bad.json"), "w", encoding="utf-8") as f:
            f.write("not-json")
        p = SessionPersister(root)
        registry = SubagentRegistry()
        restored = p.restore(registry)
        assert restored == {}


def test_checkpoint_atomic():
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "sessions")
        p = SessionPersister(root)
        p.checkpoint(SubagentHandle(subagent_id="a1", session_id="s1", goal="g1"))
        assert os.listdir(root) == ["a1.json"]
        assert not os.path.exists(os.path.join(root, "a1.json.tmp"))


# --- Integration: hooks persist, plugin load restores (restart survival) ---

def test_hook_start_checkpoints_to_disk(monkeypatch, tmp_path):
    """subagent_start hook must write the handle to the persist store."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "subagent_handles_plugin", os.path.join(os.path.dirname(__file__), "..", "__init__.py")
    )
    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)

    store = tmp_path / "subagent-handles"
    monkeypatch.setattr(plugin, "default_persist_root", lambda: str(store))

    plugin._on_subagent_start(
        child_subagent_id="sa-restart-1",
        child_session_id="sess-restart-1",
        child_goal="persist me",
        parent_subagent_id="p1",
        child_role="coder",
    )

    assert (store / "sa-restart-1.json").exists()


def test_register_restores_persisted_handles(monkeypatch, tmp_path):
    """plugin.register() must reclaim handles persisted by a prior run."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "subagent_handles_plugin", os.path.join(os.path.dirname(__file__), "..", "__init__.py")
    )
    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)

    store = tmp_path / "subagent-handles"
    store.mkdir(parents=True, exist_ok=True)
    p = SessionPersister(str(store))
    p.checkpoint(SubagentHandle(subagent_id="sa-old", session_id="sess-old", goal="old run"))

    monkeypatch.setattr(plugin, "default_persist_root", lambda: str(store))

    class FakeCtx:
        def register_plugin(self, *a, **k): pass
        def register_hook(self, *a, **k): pass
        def register_tool(self, *a, **k): pass

    # Start from a clean in-memory registry to prove restore (the disk store
    # holds the only copy of sa-old), not in-memory carryover.
    plugin.registry = SubagentRegistry()

    plugin.register(FakeCtx())

    handle = plugin.registry.resolve("sa-old")
    assert handle is not None
    # Handle was checkpointed as "running" by a prior (now-dead) process;
    # restore reconciles stale "running" → "failed" so subagent_send doesn't
    # report queued to a dead child after restart.
    assert handle.state == "failed"


def test_stop_hook_persists_done_state(monkeypatch, tmp_path):
    """subagent_stop must persist the terminal 'done' state."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "subagent_handles_plugin", os.path.join(os.path.dirname(__file__), "..", "__init__.py")
    )
    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)

    store = tmp_path / "subagent-handles"
    monkeypatch.setattr(plugin, "default_persist_root", lambda: str(store))

    plugin._on_subagent_start(
        child_subagent_id="sa-x",
        child_session_id="sess-x",
        child_goal="g",
        parent_subagent_id=None,
    )
    plugin._on_subagent_stop(child_session_id="sess-x")

    p = SessionPersister(str(store))
    loaded = p.load("sa-x")
    assert loaded is not None
    assert loaded.state == "done"
