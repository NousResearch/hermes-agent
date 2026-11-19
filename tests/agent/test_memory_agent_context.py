"""Regression for #80646: the ``agent_context`` handed to memory providers follows the platform
(``cron`` / ``subagent`` skip writes per the ``MemoryProvider.initialize`` contract) instead of a
hardcoded ``"primary"`` that let cron turns land in stores configured to skip them.
"""

from types import SimpleNamespace

import pytest

from agent.agent_init import _GATEWAY_IDENTITY_PARAMS, _memory_provider_init_kwargs


def _fake_agent():
    """The attribute surface ``_memory_provider_init_kwargs`` reads."""
    return SimpleNamespace(
        session_id="sess-80646", _session_db=None, _emit_warning=None, _emit_status=None,
        session_cwd=None, **{f"_{name}": None for name in _GATEWAY_IDENTITY_PARAMS},
    )


@pytest.mark.parametrize(
    ("platform", "expected"),
    [("cron", "cron"), ("subagent", "subagent"), ("telegram", "primary"), (None, "primary")],
)
def test_agent_context_follows_the_platform(platform, expected):
    assert _memory_provider_init_kwargs(_fake_agent(), platform)["agent_context"] == expected


class _Store:
    def __init__(self, rows, answered=()):
        self.rows, self.answered = rows, set(answered)

    def get_session_list_row(self, session_id):
        return self.rows.get(session_id)

    def latest_message_row_id(self, session_id, *, role="user", require_text=True):
        return 7 if role == "assistant" and session_id in self.answered else None

    def get_session_title(self, session_id):
        return None


@pytest.mark.parametrize(
    ("rows", "answered", "expected"),
    [
        ({"sess-80646": {"message_count": 12}}, {"sess-80646"}, (True, 12)),  # continuing a stored transcript
        ({"sess-80646": {"message_count": 1}}, (), (False, 1)),   # TUI stored the prompt being submitted
        ({"sess-80646": {"message_count": 0}}, (), (False, 0)),   # row created, nothing said yet
        ({}, (), (False, 0)),                                     # new session id (subagent, cron run)
    ],
)
def test_initialize_is_told_whether_it_continues_a_stored_session(rows, answered, expected):
    agent = _fake_agent()
    agent._session_db = _Store(rows, answered)
    kwargs = _memory_provider_init_kwargs(agent, "telegram")
    assert (kwargs["resumed"], kwargs["message_count"]) == expected


def test_no_session_store_reads_as_a_new_session():
    kwargs = _memory_provider_init_kwargs(_fake_agent(), "cli")
    assert (kwargs["resumed"], kwargs["message_count"]) == (False, 0)
