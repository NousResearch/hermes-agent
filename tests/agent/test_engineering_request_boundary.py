"""Stage reasoning reaches the existing auxiliary request builder without profile leakage."""

from agent.auxiliary_client import _build_call_kwargs
from agent.engineering_workflow import admit_stage_routes
from hermes_constants import parse_reasoning_effort


CATALOGUE = {
    "providers": [
        {
            "slug": "openai-codex",
            "models": ["gpt-6-sol"],
            "authenticated": True,
        }
    ]
}


def _routes(efforts):
    assignments = {
        stage: {
            "provider": "openai-codex",
            "model": "gpt-6-sol",
            "reasoning_effort": efforts[stage],
        }
        for stage in ("planner", "worker", "reviewer")
    }
    return admit_stage_routes(assignments, CATALOGUE)


def _request(route):
    return _build_call_kwargs(
        route.provider,
        route.model,
        [{"role": "user", "content": "bounded task"}],
        reasoning_config=parse_reasoning_effort(route.reasoning_effort),
        task="engineering_workflow",
    )


def test_codex_request_builder_preserves_each_stage_effort():
    routes = _routes({"planner": "medium", "worker": "medium", "reviewer": "high"})
    assert [
        _request(routes[stage])["extra_body"]["reasoning"]["effort"]
        for stage in ("planner", "worker", "reviewer")
    ] == ["medium", "medium", "high"]


def test_profile_switch_a_b_a_does_not_retain_previous_effort(monkeypatch, tmp_path):
    values = []
    for profile, effort in (("a", "medium"), ("b", "high"), ("a", "medium")):
        home = tmp_path / profile
        home.mkdir(exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(home))
        routes = _routes({"planner": effort, "worker": effort, "reviewer": effort})
        values.append(_request(routes["planner"])["extra_body"]["reasoning"]["effort"])
    assert values == ["medium", "high", "medium"]