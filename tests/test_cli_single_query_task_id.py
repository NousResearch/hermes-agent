from __future__ import annotations


class _DummyAgent:
    def __init__(self):
        self.calls = []
        self.quiet_mode = False

    def run_conversation(self, query, task_id=None):
        self.calls.append((query, task_id))
        return {"final_response": "ok"}


class _DummyCLI:
    def __init__(self, **kwargs):
        del kwargs
        self.session_id = "session-123"
        self.system_prompt = ""
        self.preloaded_skills = []
        self.tool_progress_mode = "all"
        self._active_agent_route_signature = "sig"
        self.agent = _DummyAgent()

    def _ensure_runtime_credentials(self):
        return True

    def _resolve_turn_agent_config(self, query):
        del query
        return {
            "signature": "sig",
            "model": "gpt-5.4",
            "runtime": {},
            "label": "default",
        }

    def _init_agent(self, **kwargs):
        del kwargs
        return True

    def show_banner(self):
        return None

    def show_tools(self):
        return None

    def show_toolsets(self):
        return None

    def run(self):
        return None


def test_main_single_query_uses_cli_session_id_as_task_id(monkeypatch, capsys):
    import cli as cli_mod

    created = {}

    def fake_cli(**kwargs):
        del kwargs
        created["cli"] = _DummyCLI()
        return created["cli"]

    monkeypatch.setattr(cli_mod, "HermesCLI", fake_cli)

    cli_mod.main(query="hello", quiet=True)

    cli_obj = created["cli"]
    assert cli_obj.agent.calls == [("hello", "session-123")]

    out = capsys.readouterr().out
    assert "ok" in out
    assert "session_id: session-123" in out
