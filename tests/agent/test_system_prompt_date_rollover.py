"""System-prompt date rollover refresh (#86938).

Date-only prompts are byte-stable per day by design; a long-lived session
must refresh the 'Conversation started:' line exactly once when the UTC
date rolls over.
"""

from __future__ import annotations

from unittest.mock import patch


def test_build_records_prompt_date(monkeypatch):
    from datetime import datetime

    from agent import system_prompt as sp

    monkeypatch.setattr(
        sp,
        "build_system_prompt_parts",
        lambda agent, system_message=None: {
            "stable": "",
            "context": "",
            "volatile": "Conversation started: Saturday, August 15, 2026",
        },
    )
    monkeypatch.setattr(sp, "drain_truncation_warnings", lambda: [])

    class Agent:
        def _emit_status(self, *a, **k):
            pass

    a = Agent()
    with patch("hermes_time.now") as now_mock:
        now_mock.return_value = datetime(2026, 8, 15, 12, 0, 0)
        sp.build_system_prompt(a)

    assert a._system_prompt_date == "2026-08-15"


def test_date_stale_detects_rollover():
    from agent.system_prompt import system_prompt_date_stale

    class Agent:
        _system_prompt_date = "2026-08-15"

    a = Agent()
    with patch("hermes_time.now") as now_mock:
        from datetime import datetime

        now_mock.return_value = datetime(2026, 8, 15, 23, 59, 0)
        assert system_prompt_date_stale(a) is False

        now_mock.return_value = datetime(2026, 8, 16, 0, 1, 0)
        assert system_prompt_date_stale(a) is True


def test_stale_without_recorded_date_is_false():
    from agent.system_prompt import system_prompt_date_stale

    class Agent:
        pass

    assert system_prompt_date_stale(Agent()) is False
