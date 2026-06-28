import inspect

import run_agent


class _RecordingAgent:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.prompts = []
        self.__class__.instances.append(self)

    def run_conversation(self, prompt):
        self.prompts.append(prompt)
        return {
            "completed": True,
            "api_calls": 0,
            "messages": [],
            "final_response": "ok",
        }


def _patch_agent(monkeypatch):
    _RecordingAgent.instances = []
    monkeypatch.setattr(run_agent, "AIAgent", _RecordingAgent)
    return _RecordingAgent.instances


def test_main_accepts_browser_test_flag():
    assert "browser_test" in inspect.signature(run_agent.main).parameters


def test_browser_test_defaults_to_browser_toolset_and_prompt(monkeypatch):
    agents = _patch_agent(monkeypatch)

    run_agent.main(browser_test=True)

    agent = agents[0]
    assert agent.kwargs["enabled_toolsets"] == ["browser"]
    assert "https://example.com" in agent.prompts[0]


def test_browser_test_respects_explicit_query_and_toolsets(monkeypatch):
    agents = _patch_agent(monkeypatch)

    run_agent.main(
        browser_test=True,
        query="open https://example.org",
        enabled_toolsets="browser,web",
    )

    agent = agents[0]
    assert agent.kwargs["enabled_toolsets"] == ["browser", "web"]
    assert agent.prompts == ["open https://example.org"]
