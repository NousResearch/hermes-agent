"""GitHub/Copilot Responses reasoning fields on the Codex transport.

Regression for #46527: the GitHub Responses branch must request ``summary: "auto"`` so
Copilot returns reasoning text. Without it, reasoning items come back with no summary and
Hermes persists no reasoning/thinking content.
"""

import pytest

from agent.transports import get_transport


@pytest.fixture
def transport():
    import agent.transports.codex  # noqa: F401
    return get_transport("codex_responses")


def test_github_responses_requests_reasoning_summary(transport):
    kw = transport.build_kwargs(
        model="gpt-5.4", messages=[{"role": "user", "content": "Hi"}], tools=[],
        is_github_responses=True,
        github_reasoning_extra={"effort": "medium"},
    )
    assert kw.get("reasoning") == {"effort": "medium", "summary": "auto"}
