from types import SimpleNamespace


def test_pre_verify_can_run_again_after_continuation(monkeypatch):
    from agent.turn_stop_gates import _pre_verify_nudge
    from agent import verify_hooks
    from hermes_cli import lifecycle, plugins

    attempts = []
    monkeypatch.setattr(verify_hooks, "max_verify_nudges", lambda: 2)
    monkeypatch.setattr(lifecycle, "has_hook", lambda name: True)
    monkeypatch.setattr(plugins, "get_pre_verify_continue_message", lambda **kw: attempts.append(kw["attempt"]) or "check")
    agent = SimpleNamespace(_turn_file_mutation_paths={"a.py"}, _resolved_is_coding=True)
    _pre_verify_nudge(agent, "draft", 0)
    _pre_verify_nudge(agent, "revised", 1)
    _pre_verify_nudge(agent, "last", 2)
    assert attempts == [0, 1]
