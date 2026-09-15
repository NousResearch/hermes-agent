"""ACP sessions must honour ``agent.disabled_toolsets`` from config.

``hermes config set agent.disabled_toolsets terminal,file`` is a hard suppression the
operator configured. Every other platform passes it to the agent; the ACP session builder
did not, so on ACP the setting silently did nothing and the tools stayed live.

Value shapes follow ``agent.skill_utils.parse_config_string_list`` — the same normalizer
``hermes_cli/prompt_size.py`` uses for this key — so ACP cannot diverge from the rest.
"""

from __future__ import annotations

import pytest

from acp_adapter.session import SessionManager


class _NoopDb:
    def get_session(self, *_a, **_k):
        return None

    def create_session(self, *_a, **_k):
        return None

    def update_session(self, *_a, **_k):
        return None


@pytest.mark.parametrize(
    "configured, expected",
    [
        (["terminal", "file"], ["terminal", "file"]),   # documented list form
        ("['terminal', 'file']", ["terminal", "file"]),  # literal-string form `hermes config set` writes
        ("terminal", ["terminal"]),                     # a scalar string is one name
        (None, None),                                   # unset stays unset
        ([], None),                                     # empty means "no suppression", not "[]"
    ],
)
def test_acp_make_agent_passes_configured_disabled_toolsets(monkeypatch, configured, expected):
    captured: dict = {}

    class _CaptureAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("run_agent.AIAgent", _CaptureAgent, raising=False)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda *_a, **_k: {"agent": ({} if configured is None else {"disabled_toolsets": configured})},
        raising=False,
    )

    manager = SessionManager(db=_NoopDb())  # no agent_factory -> the real build path
    manager._make_agent(session_id="acp-test", cwd=".")

    assert captured.get("disabled_toolsets") == expected
