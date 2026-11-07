"""Config → real review fork contracts for #16761; ported from the #115943 tests."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent import background_review as bg
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tests.agent.test_background_review_cache_parity import _make_agent_stub, _make_recorder_class, _SyncThread


@pytest.mark.parametrize("kind,scope", [("memory", (True, False)), ("skill", (False, True)), ("combined", (True, True))])
@pytest.mark.parametrize("settings,data,expected", [
    ({"inline": " custom ", "file": "p"}, b"file", "custom"),
    ({"file": "p"}, b" file ", "file"), ({"inline": ""}, b"file", None),
    ({"inline": " "}, b"file", None), ({"inline": False}, b"file", None),
    ({"file": "missing"}, b"file", None), ({"file": False}, b"file", None),
    ({"file": "p"}, b"", None), ({"file": "p"}, b" \n", None),
    ({"file": "p"}, b"\xff", None), ({"file": "p"}, b"x" * 65537, None),
    ({"file": "."}, b"file", None),
])
def test_config_reaches_fork_or_skips(tmp_path, monkeypatch, caplog, kind, scope, settings, data, expected):
    import run_agent
    from hermes_cli.config import _validate_config_key

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = {kind if key == "inline" else f"{kind}_file": value for key, value in settings.items()}
    (tmp_path / "config.yaml").write_text(json.dumps({"agent": {"review_prompts": config}}))
    (tmp_path / "p").write_bytes(data)
    agent = _make_agent_stub(run_agent.AIAgent)
    for name in bg._PROMPT_NAME_BY_SCOPE.values():
        agent.__dict__.pop(name, None)  # Exercise inherited production class defaults.
    captured, whitelist = {}, []
    recorder = _make_recorder_class()

    def record(self, *, user_message, **kwargs):
        captured["message"] = user_message
        assert self._cached_system_prompt == agent._cached_system_prompt
        return {"final_response": "Nothing to save."}

    monkeypatch.setattr(recorder, "run_conversation", record)
    monkeypatch.setattr("hermes_cli.plugins.set_thread_tool_whitelist", lambda tools, **kw: whitelist.append(set(tools)))
    with patch.object(run_agent, "AIAgent", recorder), patch("threading.Thread", _SyncThread):
        agent._spawn_background_review_now([], review_memory=scope[0], review_skills=scope[1], task_cfg={})
    assert getattr(agent, "_background_review_run", None) is None
    if expected is None:
        assert captured == {} and whitelist == []
        assert config.get(kind) == "" or "Fix or remove" in caplog.text
    else:
        assert captured["message"].startswith(expected + "\n\nYou can only call ")
        assert "terminal" not in whitelist[0]
    _, prompt = bg.spawn_background_review_thread(agent, [], *scope, task_cfg={}, explicit=True)
    assert prompt == (expected or getattr(bg, bg._PROMPT_NAME_BY_SCOPE[scope]))
    setattr(agent, bg._PROMPT_NAME_BY_SCOPE[scope], "programmatic")
    assert bg.spawn_background_review_thread(agent, [], *scope, task_cfg={})[1] == "programmatic"
    assert all(_validate_config_key(f"agent.review_prompts.{key}")[0] for key in (kind, f"{kind}_file"))


def test_profiles_and_file_snapshots(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "launch"))
    for label in ("a", "b", "a"):
        home = tmp_path / label
        home.mkdir(exist_ok=True)
        (home / "config.yaml").write_text('{"agent":{"review_prompts":{"memory_file":"p"}}}')
        (home / "p").write_text(label)
        agent = SimpleNamespace(_session_db=SimpleNamespace(db_path=home / "state.db"))
        token = set_hermes_home_override(home)
        try:
            assert bg.spawn_background_review_thread(agent, [], True, task_cfg={})[1] == label
        finally:
            reset_hermes_home_override(token)
        target, prompt = bg.spawn_background_review_thread(agent, [], True, task_cfg={})
        (home / "p").write_text("edited")
        with patch.object(bg, "_run_review_in_thread") as worker:
            target()
            assert worker.call_args.args[2] == prompt == label
        assert bg.spawn_background_review_thread(agent, [], True, task_cfg={})[1] == "edited"


@pytest.mark.parametrize("config", [{"memory": ""}, {"memory_file": "missing"}])
def test_focused_refine_without_explicit_still_runs(tmp_path, monkeypatch, config):
    """A gateway ``/refine <focus>`` passes ``focus`` but not ``explicit``; it must not be skipped."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(json.dumps({"agent": {"review_prompts": config}}))
    agent = SimpleNamespace(_session_db=SimpleNamespace(db_path=tmp_path / "state.db"))
    target, prompt = bg.spawn_background_review_thread(agent, [], True, focus="tidy notes", task_cfg={})
    assert target is not None and prompt.startswith(bg._MEMORY_REVIEW_PROMPT) and "tidy notes" in prompt
    assert bg.spawn_background_review_thread(agent, [], True, task_cfg={}) == (None, None)
