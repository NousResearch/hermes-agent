"""Real edit -> executor -> SQLite -> history projection, without model calls."""
import copy
import contextlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.agent.test_tool_call_incremental_persistence import (
    _attach_real_session_db,
    _make_agent,
    _mock_tool_call,
)


@pytest.mark.parametrize("executor_mode", ["sequential", "concurrent"])
@pytest.mark.parametrize("edit", ["create", "replace", "patch", "noop", "failed"])
def test_edit_preview_is_durable_before_emission_and_display_only(
    tmp_path, monkeypatch, executor_mode, edit,
):
    from agent.context_compressor import ContextCompressor
    from agent import secret_scope
    from agent.turn_context import build_api_messages
    from hermes_state import SessionDB
    from tools.environments.local import LocalEnvironment
    from tools.file_operations import ShellFileOperations
    from tools.terminal_tool import register_task_env_overrides, clear_task_env_overrides
    import tools.file_tools as file_tools
    import model_tools
    import tui_gateway.server as progress

    assert Path(model_tools.__file__).resolve().parent == Path(__file__).resolve().parents[2]
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *a, **kw: None)
    # Deliberately different from the execution cwd: relative previews must use
    # the same task target as the real file tool, not today's process directory.
    process_cwd = tmp_path / "process"
    process_cwd.mkdir()
    monkeypatch.chdir(process_cwd)
    emitted = []
    at_emission = []
    monkeypatch.setattr(progress, "_emit_tool_lifecycle", lambda *a: emitted.append(a))
    homes = {tag: tmp_path / tag for tag in ("alpha", "beta")}
    for home in homes.values():
        home.mkdir()
        (home / "config.yaml").write_text("terminal:\n  backend: local\n", encoding="utf-8")

    # Same tool identity reused in separate homes: no process-global diff cache.
    for visit, tag in enumerate(("alpha", "beta", "alpha")):
        home = homes[tag]
        scope = contextlib.ExitStack()
        # File-tool execution needs a current authoritative secret generation;
        # the no-hydration scope is reserved for persistence-only teardown.
        scope.enter_context(progress._session_profile_runtime_scope({"profile_home": str(home)}))
        work = home / f"work-{visit}"
        work.mkdir()
        env = LocalEnvironment(cwd=str(work), timeout=15)
        ops = ShellFileOperations(env, cwd=str(work))
        monkeypatch.setattr(file_tools, "_get_file_ops", lambda task_id="default": ops)
        path = work / "publish.py"
        before, after = "print('before')\n", f"print('{tag}-{visit}')\n"
        tool_name = "patch" if edit == "patch" else "write_file"
        if edit != "create":
            path.write_text(before, encoding="utf-8")
        args = {"path": "publish.py", "content": before if edit == "noop" else after}
        if edit == "patch":
            args = {"path": "publish.py", "old_string": before.strip(), "new_string": after.strip()}
        name = f"{executor_mode}-{edit}-{tag}-{visit}"
        call_id = "edit-call"
        agent = _make_agent()
        agent.valid_tool_names = {tool_name}
        db_path = home / "state.db"
        db = _attach_real_session_db(agent, db_path, name)
        register_task_env_overrides(name, {"cwd": str(work), "env_type": "local"})
        monkeypatch.setitem(progress._sessions, name, {"agent": agent, "session_key": name})
        # Wire the real gateway callbacks, including pre-commit preparation.
        for key, callback in progress._agent_cbs(name).items():
            setattr(agent, key, callback)
        if edit in {"replace", "noop"}:
            file_tools.read_file_tool("publish.py", task_id=name)
        messages = [
            {"role": "user", "content": "Update the file"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": call_id, "type": "function", "function": {
                    "name": tool_name, "arguments": json.dumps(args),
                },
            }]},
        ]
        agent._flush_messages_to_session_db(messages)
        emitted.clear()
        at_emission.clear()

        def emit(kind, sid, name, args, payload):
            emitted.append((kind, copy.deepcopy(payload)))
            if kind == "tool.complete":
                with SessionDB(db_path=db_path) as cold:
                    at_emission.extend(cold.get_messages(sid))

        monkeypatch.setattr(progress, "_emit_tool_lifecycle", emit)
        try:
            call = _mock_tool_call(tool_name, json.dumps(args), call_id)
            getattr(agent, f"_execute_tool_calls_{executor_mode}")(
                SimpleNamespace(tool_calls=[call]), messages, name,
            )
            completion = next(payload for kind, payload in emitted if kind == "tool.complete")
            row = next(row for row in at_emission if row["role"] == "tool")
            raw = json.loads(row["content"])
            assert raw == completion["result"]
            assert "inline_diff" not in raw
            changed = edit not in {"noop", "failed"}
            assert path.read_text(encoding="utf-8") == (after if changed else before)
            assert bool(raw.get("error")) == (edit == "failed")
            metadata = (row.get("display_metadata") or {}).get("tool_result_metadata") or {}
            # This assertion was red on base: completion saw no persisted sidecar.
            assert bool(metadata.get("inline_diff")) == changed
            assert metadata.get("inline_diff") == completion.get("inline_diff")
            if changed:
                assert after.strip() in metadata["inline_diff"]
            with SessionDB(db_path=db_path) as cold:
                conversation = cold.get_messages_as_conversation(name)
            projected = progress._history_to_messages(conversation)
            projected_tool = next(row for row in projected if row["role"] == "tool")
            assert projected_tool["tool_call_id"] == call_id
            assert projected_tool["content"] == row["content"]
            assert projected_tool.get("display_metadata") == row.get("display_metadata")
            assert projected_tool["timestamp"] == row["timestamp"]

            # Preparation never modifies provider bytes or the cached prefix.
            agent._current_turn_timestamp = 1.0
            frozen = copy.deepcopy(messages)
            def wire(history):
                return build_api_messages(
                    agent, history, current_turn_user_idx=0, ext_prefetch_cache="",
                    plugin_user_context="", moa_config=None, active_system_prompt="fixed prefix",
                )
            without_display = [{k: v for k, v in m.items() if k != "display_metadata"} for m in messages]
            assert wire(messages) == wire(without_display)
            before_next_turn = wire(messages)
            assert wire(messages + [{"role": "user", "content": "Continue"}])[0][:-1] == before_next_turn[0]
            assert messages == frozen

            # Cold display does not reread the now-changed target. Exercise real
            # compaction demotion + DB archival, preserving the original tool row.
            path.write_text("print('unrelated later edit')\n", encoding="utf-8")
            compacted = copy.deepcopy(conversation)
            tool_index = next(i for i, m in enumerate(compacted) if m["role"] == "tool")
            assert ContextCompressor._demote_tool_result_at(
                compacted, tool_index, {call_id: (tool_name, json.dumps(args))}, 0,
            )
            assert compacted[tool_index].get("display_metadata") == row.get("display_metadata")
            db.archive_and_compact(name, compacted)
            with SessionDB(db_path=db_path) as cold:
                archived = cold.get_messages(name, include_compacted=True)
                active = cold.get_messages_as_conversation(name)
            assert next(m for m in archived if m["id"] == row["id"])["content"] == row["content"]
            assert next(m for m in active if m["role"] == "tool").get("display_metadata") == row.get("display_metadata")
        finally:
            clear_task_env_overrides(name)
            db.close()
            env.cleanup()
            scope.close()


def test_orphaned_preview_never_attaches_to_a_reused_call_id(monkeypatch):
    import tui_gateway.server as progress

    emitted = []
    monkeypatch.setattr(progress, "_emit_tool_lifecycle", lambda kind, sid, name, args, payload: emitted.append(payload))
    session = {"tool_result_metadata": {"call_0": {"inline_diff": "STALE"}}}
    monkeypatch.setitem(progress._sessions, "reuse", session)
    # The prepared call's flush failed, so its completion never fired; a later call reuses the id.
    progress._on_tool_start("reuse", "call_0", "todo", {})
    progress._on_tool_complete("reuse", "call_0", "todo", {}, '{"todos": []}')
    assert emitted and not any("inline_diff" in payload for payload in emitted)
    assert session["tool_result_metadata"] == {}

    # A completion dropped as stale still consumes its prepared entry.
    session["tool_result_metadata"]["call_1"] = {}
    monkeypatch.setattr(progress, "_connector_lifecycle_is_stale", lambda *a: True)
    progress._on_tool_complete("reuse", "call_1", "todo", {}, "{}")
    assert session["tool_result_metadata"] == {}
