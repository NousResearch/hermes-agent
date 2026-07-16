from conductor.config import resolve_conductor_config
from conductor.runtime import plan_from_config
from hermes_cli.config import DEFAULT_CONFIG


def test_config_defaults_are_bounded_and_roles_are_separate():
    resolved = resolve_conductor_config({})
    assert resolved["enabled"] is False
    assert resolved["writer"]["model"] == ""
    assert resolved["reviewer"]["model"] == ""
    assert resolved["writer"] is not resolved["reviewer"]
    assert resolved["budgets"]["max_processed_tokens_per_run"] > 0
    assert (
        resolved["budgets"]["max_processed_tokens_per_day"]
        >= resolved["budgets"]["max_processed_tokens_per_run"]
    )
    assert resolved["budgets"]["max_retries"] >= 0
    assert 1 <= resolved["tick_lease_seconds"] <= 3600
    assert DEFAULT_CONFIG["conductor"]["enabled"] is False


def test_config_overrides_do_not_mutate_defaults():
    first = resolve_conductor_config({
        "conductor": {"budgets": {"max_runs_per_day": 2}}
    })
    second = resolve_conductor_config({})
    assert first["budgets"]["max_runs_per_day"] == 2
    assert second["budgets"]["max_runs_per_day"] != 2


def test_tick_lease_config_is_strictly_bounded():
    for value in (0, 3601, True, "30"):
        try:
            resolve_conductor_config({"conductor": {"tick_lease_seconds": value}})
        except ValueError as exc:
            assert "tick_lease_seconds" in str(exc)
        else:
            raise AssertionError(f"accepted invalid tick lease: {value!r}")


def test_campaign_routing_comes_from_separate_config_roles(tmp_path):
    config = {
        "conductor": {
            "writer": {"command": ["writer"], "provider": "p1", "model": "m1"},
            "reviewer": {"command": ["reviewer"], "provider": "p2", "model": "m2"},
        }
    }
    plan = plan_from_config(
        {
            "campaign_id": "routed",
            "cwd": str(tmp_path),
            "mutable_manifest": ["x"],
            "steps": [{"step_id": "x", "kind": "implementation", "prompt": "x"}],
        },
        config,
    )
    assert plan.writer["model"] == "m1"
    assert plan.reviewer["model"] == "m2"
    assert plan.writer is not plan.reviewer
