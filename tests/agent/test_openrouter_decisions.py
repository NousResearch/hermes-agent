"""Contracts for OpenRouter Decisions API auxiliary calls."""

import pytest


def test_decisions_request_uses_alpha_endpoint_and_returns_noul_answers():
    from agent.openrouter_decisions import call_decisions

    seen = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"answers": {"approve": {"type": "noul", "noul": 0.97}}}

    def post(url, *, headers, json, timeout):
        seen.update(url=url, headers=headers, json=json, timeout=timeout)
        return Response()

    answers = call_decisions(
        api_key="test-key", base_url="https://openrouter.ai/api/v1",
        model="typesafe/jev-1.13", state={"command": "git status"},
        questions={"approve": {"type": "noul", "instructions": "Is it safe?"}},
        timeout=12, post=post,
    )

    assert answers == {"approve": 0.97}
    assert seen["url"] == "https://openrouter.ai/api/alpha/decisions"
    assert seen["headers"]["Authorization"] == "Bearer test-key"
    assert seen["json"]["state"] == {"command": "git status"}


@pytest.mark.parametrize("answer", [-0.1, 1.1, "0.9"])
def test_decisions_rejects_invalid_noul_probabilities(answer):
    from agent.openrouter_decisions import call_decisions

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"answers": {"approve": {"type": "noul", "noul": answer}}}

    with pytest.raises(ValueError, match="approve"):
        call_decisions(
            api_key="test-key", base_url="https://openrouter.ai/api/v1",
            model="typesafe/jev-1.13", state={},
            questions={"approve": {"type": "noul", "instructions": "Is it safe?"}},
            post=lambda *args, **kwargs: Response(),
        )
