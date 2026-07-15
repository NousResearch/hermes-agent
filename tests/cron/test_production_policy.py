from __future__ import annotations

import json
from unittest.mock import Mock, patch

import pytest

from cron import production_policy


def _agent_job(**overrides):
    value = {
        "id": "job-exact",
        "enabled": True,
        "no_agent": False,
        "prompt": "Perform the scheduled task using the full primary loop.",
        "script": None,
        "provider": "openai-codex",
        "model": "gpt-5.6-sol",
        "base_url": None,
        "deliver": "local",
        "enabled_toolsets": None,
    }
    value.update(overrides)
    return value


def _production_toolset_config():
    from gateway.production_capability_prerequisites import FIRST_WAVE_TOOLSETS

    return {
        "platform_toolsets": {
            "cron": list(FIRST_WAVE_TOOLSETS),
        }
    }


def _mechanical_job(**overrides):
    value = {
        "id": "job-mechanical",
        "enabled": True,
        "no_agent": True,
        "prompt": "",
        "script": "/opt/muncho/scripts/check.sh",
        "provider": None,
        "model": None,
        "base_url": None,
    }
    value.update(overrides)
    return value


def test_inactive_policy_preserves_normal_hermes_cron_routes(monkeypatch):
    monkeypatch.setattr(production_policy, "_active", False)
    production_policy.enforce_production_cron_job(
        _agent_job(provider="openrouter", model="another-model")
    )


def test_active_policy_allows_only_exact_primary_agent_jobs(monkeypatch):
    monkeypatch.setattr(production_policy, "_active", True)
    production_policy.enforce_production_cron_job(_agent_job())

    for invalid in (
        _agent_job(provider=None, model=None),
        _agent_job(provider="openrouter", model="gpt-5.6-sol"),
        _agent_job(base_url="https://example.invalid/v1"),
        _mechanical_job(),
        _mechanical_job(model="gpt-5.6-sol"),
    ):
        with pytest.raises(
            production_policy.ProductionCronPolicyError,
            match="production_cron_policy_blocked",
        ):
            production_policy.enforce_production_cron_job(invalid)


def test_active_policy_resolves_full_or_exact_unique_subset(monkeypatch):
    from gateway.production_capability_prerequisites import FIRST_WAVE_TOOLSETS

    monkeypatch.setattr(production_policy, "_active", True)
    assert production_policy.resolve_production_cron_toolsets(
        _agent_job(), _production_toolset_config()
    ) == list(FIRST_WAVE_TOOLSETS)
    assert production_policy.resolve_production_cron_toolsets(
        _agent_job(enabled_toolsets=["web", "file"]),
        _production_toolset_config(),
    ) == ["web", "file"]


@pytest.mark.parametrize(
    "job",
    [
        _agent_job(enabled_toolsets=["code_execution"]),
        _agent_job(enabled_toolsets=["web", "web"]),
        _agent_job(enabled_toolsets=[]),
    ],
)
def test_active_policy_rejects_widening_or_malformed_toolsets(monkeypatch, job):
    monkeypatch.setattr(production_policy, "_active", True)
    with pytest.raises(
        production_policy.ProductionCronPolicyError,
        match="production_cron_policy_blocked",
    ):
        production_policy.resolve_production_cron_toolsets(
            job, _production_toolset_config()
        )


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"platform_toolsets": {}},
        {"platform_toolsets": {"cron": None}},
        {"platform_toolsets": {"cron": ["web"]}},
    ],
)
def test_active_policy_fails_closed_on_missing_or_drifted_config(
    monkeypatch,
    config,
):
    monkeypatch.setattr(production_policy, "_active", True)
    with pytest.raises(
        production_policy.ProductionCronPolicyError,
        match="production_cron_policy_blocked",
    ):
        production_policy.resolve_production_cron_toolsets(_agent_job(), config)


def test_scheduler_resolver_never_falls_back_when_active_policy_errors(
    monkeypatch,
):
    from cron.scheduler import _resolve_cron_enabled_toolsets
    import gateway.production_model_sovereignty_runtime as runtime

    monkeypatch.setattr(production_policy, "_active", True)
    monkeypatch.setattr(
        runtime,
        "resolve_production_cron_enabled_toolsets",
        Mock(side_effect=RuntimeError("attestation failed")),
    )
    with pytest.raises(
        production_policy.ProductionCronPolicyError,
        match="attestation failed",
    ):
        _resolve_cron_enabled_toolsets(_agent_job(), _production_toolset_config())


def test_production_origin_delivery_projects_to_public_relay(monkeypatch):
    from cron.scheduler import _resolve_delivery_target

    monkeypatch.setattr(production_policy, "_active", True)
    target = _resolve_delivery_target(
        _agent_job(
            deliver="origin",
            origin={
                "platform": "discord",
                "chat_id": "1504852355588423801",
                "chat_name": None,
                "thread_id": "1504852355588423801",
                "user_id": "1279454038731264061",
            },
        )
    )
    assert target == {
        "platform": "relay",
        "chat_id": "1504852355588423801",
        "thread_id": "1504852355588423801",
    }


def test_scheduler_revalidates_store_job_at_execution_time(monkeypatch):
    monkeypatch.setattr(production_policy, "_active", True)
    from cron.scheduler import run_job

    with pytest.raises(production_policy.ProductionCronPolicyError):
        run_job(_agent_job(provider="openrouter", model="other"))


def test_scheduler_uses_attested_pinned_config_and_passes_exact_surface(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(production_policy, "_active", True)
    # This raw file deliberately carries a fallback route. Production must not
    # consume it; only the already-pinned in-memory projection below is valid.
    (tmp_path / "config.yaml").write_text(
        json.dumps(
            {
                **_production_toolset_config(),
                "fallback_providers": [
                    {"provider": "openrouter", "model": "other-model"}
                ],
            }
        ),
        encoding="utf-8",
    )
    pinned_config = {
        **_production_toolset_config(),
        "fallback_providers": [],
        "fallback_model": [],
    }
    monkeypatch.setenv("HERMES_PREFILL_MESSAGES_FILE", str(tmp_path / "untrusted.json"))
    (tmp_path / "untrusted.json").write_text(
        '[{"role":"user","content":"alternate prefill"}]',
        encoding="utf-8",
    )
    agent = Mock()
    agent.run_conversation.return_value = {"final_response": "ok"}
    runtime = {
        "api_key": "test-key",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "provider": "openai-codex",
        "api_mode": "codex_responses",
    }
    with (
        patch("cron.scheduler._hermes_home", tmp_path),
        patch("cron.scheduler._resolve_origin", return_value=None),
        patch("hermes_cli.env_loader.load_hermes_dotenv"),
        patch("hermes_cli.env_loader.reset_secret_source_cache"),
        patch("hermes_state.SessionDB"),
        patch(
            "hermes_cli.config.attest_pinned_effective_config_projection",
            return_value="0" * 64,
        ),
        patch("cron.scheduler.load_config", return_value=pinned_config),
        patch(
            "gateway.production_model_sovereignty_runtime.validate_production_gateway_config"
        ) as validate_config,
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value=runtime,
        ),
        patch("run_agent.AIAgent", return_value=agent) as agent_cls,
    ):
        from cron.scheduler import run_job

        result = run_job(_agent_job(enabled_toolsets=["web", "file"]))

    assert result.success is True
    validate_config.assert_called_once_with(pinned_config)
    assert agent_cls.call_args.kwargs["enabled_toolsets"] == ["web", "file"]
    assert agent_cls.call_args.kwargs["fallback_model"] is None
    assert agent_cls.call_args.kwargs["prefill_messages"] is None


def test_scheduler_missing_pinned_config_fails_before_agent_construction(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(production_policy, "_active", True)
    with (
        patch("cron.scheduler._hermes_home", tmp_path),
        patch("cron.scheduler._resolve_origin", return_value=None),
        patch("hermes_cli.env_loader.load_hermes_dotenv"),
        patch("hermes_cli.env_loader.reset_secret_source_cache"),
        patch("hermes_state.SessionDB"),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider"
        ) as resolve_provider,
        patch("run_agent.AIAgent") as agent_cls,
    ):
        from cron.scheduler import run_job

        result = run_job(_agent_job())

    assert result.success is False
    assert "production cron effective config pin is absent" in str(result.error)
    resolve_provider.assert_not_called()
    agent_cls.assert_not_called()


def test_scheduler_rejects_drifted_pinned_projection_before_provider_or_agent(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(production_policy, "_active", True)
    drifted = {
        **_production_toolset_config(),
        "fallback_providers": [
            {"provider": "openrouter", "model": "other-model"}
        ],
    }
    with (
        patch("cron.scheduler._hermes_home", tmp_path),
        patch("cron.scheduler._resolve_origin", return_value=None),
        patch("hermes_cli.env_loader.load_hermes_dotenv"),
        patch("hermes_cli.env_loader.reset_secret_source_cache"),
        patch("hermes_state.SessionDB"),
        patch(
            "hermes_cli.config.attest_pinned_effective_config_projection",
            return_value="0" * 64,
        ),
        patch("cron.scheduler.load_config", return_value=drifted),
        patch(
            "gateway.production_model_sovereignty_runtime.validate_production_gateway_config",
            side_effect=RuntimeError("production_fallback_providers_not_exact"),
        ),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider"
        ) as resolve_provider,
        patch("run_agent.AIAgent") as agent_cls,
    ):
        from cron.scheduler import run_job

        result = run_job(_agent_job())

    assert result.success is False
    assert "production_fallback_providers_not_exact" in str(result.error)
    resolve_provider.assert_not_called()
    agent_cls.assert_not_called()


def test_cron_tool_pins_omitted_create_route_before_store_write(monkeypatch):
    monkeypatch.setattr(production_policy, "_active", True)
    from tools import cronjob_tools

    stored = {
        "id": "job-created",
        "name": "Inspect state",
        "skills": [],
        "schedule_display": "every 1h",
        "repeat": {"times": None, "completed": 0},
        "deliver": "local",
        "next_run_at": "2026-07-14T12:00:00+00:00",
        "provider": "openai-codex",
        "model": "gpt-5.6-sol",
    }
    create = Mock(return_value=stored)
    monkeypatch.setattr(cronjob_tools, "create_job", create)
    monkeypatch.setattr(cronjob_tools, "_local_delivery_notice", lambda *_: None)
    result = json.loads(
        cronjob_tools.cronjob(
            action="create",
            schedule="every 1h",
            prompt="Inspect state",
        )
    )

    assert result["success"] is True
    assert create.call_args.kwargs["provider"] == "openai-codex"
    assert create.call_args.kwargs["model"] == "gpt-5.6-sol"


def test_cron_tool_rejects_explicit_alternate_create_route(monkeypatch):
    monkeypatch.setattr(production_policy, "_active", True)
    from tools import cronjob_tools

    create = Mock(side_effect=AssertionError("store write must not run"))
    monkeypatch.setattr(cronjob_tools, "create_job", create)
    result = json.loads(
        cronjob_tools.cronjob(
            action="create",
            schedule="every 1h",
            prompt="Inspect state",
            provider="openrouter",
            model="other-model",
        )
    )

    assert result["success"] is False
    assert "production_cron_policy_blocked" in result["error"]
    create.assert_not_called()


@pytest.mark.parametrize(
    "extra",
    [
        {"enabled_toolsets": ["code_execution"]},
        {"deliver": "all"},
        {"deliver": "discord:1504852355588423801"},
    ],
)
def test_cron_tool_rejects_widening_or_bypass_delivery_before_store_write(
    monkeypatch,
    extra,
):
    monkeypatch.setattr(production_policy, "_active", True)
    from tools import cronjob_tools

    create = Mock(side_effect=AssertionError("store write must not run"))
    monkeypatch.setattr(cronjob_tools, "create_job", create)
    result = json.loads(
        cronjob_tools.cronjob(
            action="create",
            schedule="every 1h",
            prompt="Inspect state",
            provider="openai-codex",
            model="gpt-5.6-sol",
            **extra,
        )
    )

    assert result["success"] is False
    assert "production_cron_policy_blocked" in result["error"]
    create.assert_not_called()


def test_cron_tool_empty_toolset_list_normalizes_to_exact_inheritance(
    monkeypatch,
):
    monkeypatch.setattr(production_policy, "_active", True)
    from tools import cronjob_tools

    stored = {
        "id": "job-created",
        "name": "Inspect state",
        "skills": [],
        "schedule_display": "every 1h",
        "repeat": {"times": None, "completed": 0},
        "deliver": "local",
        "next_run_at": "2026-07-14T12:00:00+00:00",
    }
    create = Mock(return_value=stored)
    monkeypatch.setattr(cronjob_tools, "create_job", create)
    monkeypatch.setattr(cronjob_tools, "_local_delivery_notice", lambda *_: None)

    result = json.loads(
        cronjob_tools.cronjob(
            action="create",
            schedule="every 1h",
            prompt="Inspect state",
            provider="openai-codex",
            model="gpt-5.6-sol",
            enabled_toolsets=[],
        )
    )

    assert result["success"] is True
    assert create.call_args.kwargs["enabled_toolsets"] is None


def test_cron_tool_default_delivery_keeps_authenticated_public_origin(
    monkeypatch,
):
    monkeypatch.setattr(production_policy, "_active", True)
    from tools import cronjob_tools

    origin = {
        "platform": "discord",
        "chat_id": "1504852355588423801",
        "chat_name": "operations",
        "thread_id": None,
        "user_id": "1279454038731264061",
    }
    stored = {
        "id": "job-created",
        "name": "Inspect state",
        "skills": [],
        "schedule_display": "every 1h",
        "repeat": {"times": None, "completed": 0},
        "deliver": "origin",
        "origin": origin,
        "next_run_at": "2026-07-14T12:00:00+00:00",
    }
    create = Mock(return_value=stored)
    monkeypatch.setattr(cronjob_tools, "create_job", create)
    monkeypatch.setattr(cronjob_tools, "_origin_from_env", lambda: origin)
    monkeypatch.setattr(cronjob_tools, "_local_delivery_notice", lambda *_: None)

    result = json.loads(
        cronjob_tools.cronjob(
            action="create",
            schedule="every 1h",
            prompt="Inspect state",
            provider="openai-codex",
            model="gpt-5.6-sol",
        )
    )

    assert result["success"] is True
    assert create.call_args.kwargs["deliver"] is None
    assert create.call_args.kwargs["origin"] == origin


def test_cron_tool_rejects_drifting_update_and_resume(monkeypatch):
    monkeypatch.setattr(production_policy, "_active", True)
    from tools import cronjob_tools

    invalid = _agent_job(provider="openrouter", model="other", state="paused")
    monkeypatch.setattr(cronjob_tools, "resolve_job_ref", lambda _ref: invalid)
    update = Mock(side_effect=AssertionError("store write must not run"))
    resume = Mock(side_effect=AssertionError("resume must not run"))
    monkeypatch.setattr(cronjob_tools, "update_job", update)
    monkeypatch.setattr(cronjob_tools, "resume_job", resume)

    updated = json.loads(
        cronjob_tools.cronjob(action="update", job_id="job-exact", name="new")
    )
    resumed = json.loads(cronjob_tools.cronjob(action="resume", job_id="job-exact"))

    assert updated["success"] is False
    assert resumed["success"] is False
    assert "production_cron_policy_blocked" in updated["error"]
    assert "production_cron_policy_blocked" in resumed["error"]
    update.assert_not_called()
    resume.assert_not_called()


def test_store_boundary_pins_omitted_create_and_rejects_direct_bypasses(
    monkeypatch,
    tmp_path,
):
    from cron.jobs import create_job, list_jobs, update_job, use_cron_store

    monkeypatch.setattr(production_policy, "_active", True)
    with use_cron_store(tmp_path):
        created = create_job(
            prompt="Inspect state",
            schedule="every 1h",
        )
        assert created["provider"] == "openai-codex"
        assert created["model"] == "gpt-5.6-sol"
        assert created["provider_snapshot"] is None
        assert created["model_snapshot"] is None

        with pytest.raises(
            production_policy.ProductionCronPolicyError,
            match="production_cron_delivery_not_exact",
        ):
            update_job(created["id"], {"deliver": "discord:123456789012345678"})

        persisted = list_jobs(include_disabled=True)
        assert len(persisted) == 1
        assert persisted[0]["deliver"] == "local"


def test_store_boundary_rejects_raw_full_list_and_leaves_no_file(
    monkeypatch,
    tmp_path,
):
    from cron.jobs import save_jobs, use_cron_store

    monkeypatch.setattr(production_policy, "_active", True)
    with use_cron_store(tmp_path):
        with pytest.raises(
            production_policy.ProductionCronPolicyError,
            match="production_cron_primary_route_not_exact",
        ):
            save_jobs([_agent_job(provider="openrouter", model="other")])

    assert not (tmp_path / "cron" / "jobs.json").exists()


@pytest.mark.parametrize("operation", ["resume", "trigger"])
def test_store_boundary_blocks_enabling_invalid_persisted_job(
    monkeypatch,
    tmp_path,
    operation,
):
    from cron.jobs import (
        create_job,
        get_job,
        pause_job,
        resume_job,
        trigger_job,
        use_cron_store,
    )

    monkeypatch.setattr(production_policy, "_active", False)
    with use_cron_store(tmp_path):
        job = create_job(
            prompt="Legacy route",
            schedule="every 1h",
            provider="openrouter",
            model="other",
        )
        pause_job(job["id"])
        monkeypatch.setattr(production_policy, "_active", True)

        with pytest.raises(production_policy.ProductionCronPolicyError):
            (resume_job if operation == "resume" else trigger_job)(job["id"])

        persisted = get_job(job["id"])
        assert persisted is not None
        assert persisted["enabled"] is False
        assert persisted["state"] == "paused"


def test_store_boundary_allows_disabling_and_removing_invalid_active_job(
    monkeypatch,
    tmp_path,
):
    from cron.jobs import create_job, get_job, pause_job, remove_job, use_cron_store

    monkeypatch.setattr(production_policy, "_active", False)
    with use_cron_store(tmp_path):
        first = create_job(
            prompt="Legacy route",
            schedule="every 1h",
            provider="openrouter",
            model="other",
        )
        monkeypatch.setattr(production_policy, "_active", True)
        paused = pause_job(first["id"])
        assert paused is not None and paused["enabled"] is False

        monkeypatch.setattr(production_policy, "_active", False)
        second = create_job(
            prompt="Another legacy route",
            schedule="every 1h",
            provider="openrouter",
            model="other",
        )
        monkeypatch.setattr(production_policy, "_active", True)
        assert remove_job(second["id"]) is True
        assert get_job(second["id"]) is None


def test_store_boundary_survives_bypassed_tool_precheck(
    monkeypatch,
    tmp_path,
):
    from cron.jobs import create_job, get_job, use_cron_store
    from tools import cronjob_tools

    monkeypatch.setattr(production_policy, "_active", True)
    with use_cron_store(tmp_path):
        job = create_job(prompt="Inspect state", schedule="every 1h")
        monkeypatch.setattr(
            cronjob_tools,
            "_enforce_production_job_candidate",
            lambda _candidate: None,
        )
        result = json.loads(
            cronjob_tools.cronjob(
                action="update",
                job_id=job["id"],
                deliver="discord:123456789012345678",
            )
        )

        assert result["success"] is False
        assert "production_cron_policy_blocked" in result["error"]
        assert get_job(job["id"])["deliver"] == "local"
