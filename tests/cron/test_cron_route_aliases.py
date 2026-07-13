"""Focused tests for scheduler route aliases.

These tests are deliberately provider-free: route resolution reads only a
temporary routing registry and deterministic execution must not load dotenv,
credentials, or an agent client.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import types
from unittest.mock import patch

import pytest


def _write_registry(home: Path, aliases: dict) -> None:
    (home / "model-routing.json").write_text(
        json.dumps({"schema_version": 1, "aliases": aliases}),
        encoding="utf-8",
    )


@pytest.fixture
def hermes_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "cron").mkdir(parents=True)
    (home / "scripts").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    import importlib
    import hermes_constants
    importlib.reload(hermes_constants)
    return home


def test_known_alias_resolves_without_provider_access(hermes_env):
    from cron.route_aliases import resolve_route_alias

    _write_registry(
        hermes_env,
        {
            "deterministic.none": {
                "provider": None,
                "model": None,
                "status": "active",
                "limits": {"provider_call_allowed": False},
            }
        },
    )
    route = resolve_route_alias("deterministic.none", hermes_home=hermes_env)
    assert route.alias == "deterministic.none"
    assert route.provider is None
    assert route.model is None
    assert route.allow_provider_call is False


def test_unknown_alias_fails_closed(hermes_env):
    from cron.route_aliases import RouteAliasError, resolve_route_alias

    _write_registry(hermes_env, {})
    with pytest.raises(RouteAliasError, match="unknown route alias"):
        resolve_route_alias("missing.route", hermes_home=hermes_env)


def test_deterministic_job_never_loads_credentials_or_agent(hermes_env):
    from cron.scheduler import run_job
    import hermes_cli.env_loader as env_loader

    _write_registry(
        hermes_env,
        {
            "deterministic.none": {
                "provider": None,
                "model": None,
                "status": "active",
                "limits": {"provider_call_allowed": False},
            }
        },
    )
    script = hermes_env / "scripts" / "check.sh"
    script.write_text("#!/bin/sh\nprintf 'ok\\n'\n", encoding="utf-8")
    script.chmod(0o755)
    job = {
        "id": "det-1",
        "name": "deterministic",
        "route_alias": "deterministic.none",
        "no_agent": True,
        "script": "check.sh",
    }
    fake_run_agent = types.SimpleNamespace(
        AIAgent=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError)
    )
    with patch.object(env_loader, "load_hermes_dotenv", side_effect=AssertionError), patch.dict(
        sys.modules, {"run_agent": fake_run_agent}
    ):
        ok, _doc, output, error = run_job(job)
    assert ok is True
    assert output.strip() == "ok"
    assert error is None


def test_prompt_only_deterministic_job_is_rejected(hermes_env):
    from cron.scheduler import run_job

    _write_registry(
        hermes_env,
        {
            "deterministic.none": {
                "provider": None,
                "model": None,
                "status": "active",
                "limits": {"provider_call_allowed": False},
            }
        },
    )
    job = {
        "id": "det-2",
        "name": "invalid deterministic",
        "route_alias": "deterministic.none",
        "prompt": "do work",
        "no_agent": False,
    }
    ok, _doc, _output, error = run_job(job)
    assert ok is False
    assert "deterministic.none" in (error or "")


def test_disabled_alias_fails_closed(hermes_env):
    from cron.route_aliases import RouteAliasError, resolve_route_alias

    _write_registry(
        hermes_env,
        {
            "reasoning.premium": {
                "provider": "anthropic",
                "model": "claude-premium",
                "status": "disabled-explicit-approval-required",
            }
        },
    )
    with pytest.raises(RouteAliasError, match="disabled"):
        resolve_route_alias("reasoning.premium", hermes_home=hermes_env)


def test_forbidden_provider_alias_is_rejected(hermes_env):
    from cron.route_aliases import RouteAliasError, resolve_route_alias

    _write_registry(
        hermes_env,
        {
            "bad.route": {
                "provider": "openrouter",
                "model": "bad-model",
                "status": "active",
            }
        },
    )
    with pytest.raises(RouteAliasError, match="forbidden provider"):
        resolve_route_alias("bad.route", hermes_home=hermes_env)


def test_active_executor_resolves_route_metadata(hermes_env):
    from cron.route_aliases import resolve_route_alias

    _write_registry(
        hermes_env,
        {
            "executor.infrastructure": {
                "provider": "openai-codex",
                "model": "codex-model",
                "effort": "high",
                "role": "executor",
                "status": "configuration-verified-no-provider-call",
                "fallback": [],
                "limits": {
                    "max_delegation_depth": 0,
                    "max_child_tasks": 0,
                    "max_retries": 1,
                    "timeout_seconds": 600,
                    "provider_call_allowed": True,
                    "premium_reasoning_allowed": False,
                },
            }
        },
    )
    route = resolve_route_alias("executor.infrastructure", hermes_home=hermes_env)
    assert (route.provider, route.model, route.effort, route.role) == (
        "openai-codex", "codex-model", "high", "executor"
    )
    assert route.max_retries == 1


def test_alias_fallback_is_explicit_and_preserved(hermes_env):
    from cron.route_aliases import resolve_route_alias

    fallback = [{"provider": "openai-codex", "model": "backup"}]
    _write_registry(
        hermes_env,
        {
            "planner.standard": {
                "provider": "openai-codex",
                "model": "planner",
                "status": "active",
                "fallback": fallback,
            }
        },
    )
    route = resolve_route_alias("planner.standard", hermes_home=hermes_env)
    assert list(route.fallback) == fallback


def test_registry_validation_reports_forbidden_routes_without_network(hermes_env):
    from cron.route_aliases import validate_route_registry

    _write_registry(
        hermes_env,
        {
            "bad": {"provider": "terra", "model": "x", "status": "active"}
        },
    )
    errors = validate_route_registry(hermes_home=hermes_env)
    assert errors and "forbidden" in errors[0]


def test_create_route_alias_omits_provider_snapshots(hermes_env):
    from cron.jobs import create_job

    _write_registry(
        hermes_env,
        {
            "deterministic.none": {
                "provider": None,
                "model": None,
                "status": "active",
            }
        },
    )
    script = hermes_env / "scripts" / "x.sh"
    script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    job = create_job(
        prompt=None,
        schedule="every 5m",
        script="x.sh",
        no_agent=True,
        route_alias="deterministic.none",
    )
    assert job["provider_snapshot"] is None
    assert job["model_snapshot"] is None


def test_update_route_alias_conflict_fails(hermes_env):
    from cron.jobs import create_job, update_job

    _write_registry(
        hermes_env,
        {
            "deterministic.none": {
                "provider": None,
                "model": None,
                "status": "active",
            }
        },
    )
    script = hermes_env / "scripts" / "x.sh"
    script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    job = create_job(
        prompt=None, schedule="every 5m", script="x.sh", no_agent=True
    )
    with pytest.raises(ValueError, match="conflicts"):
        update_job(job["id"], {"route_alias": "deterministic.none", "provider": "x"})


def test_route_limits_block_delegate_fanout(hermes_env):
    from cron.route_aliases import set_active_route_limits, reset_active_route_limits
    from tools.delegate_tool import _get_max_concurrent_children, _get_max_spawn_depth

    token = set_active_route_limits(
        {
            "max_child_tasks": 0,
            "max_delegation_depth": 0,
            "timeout_seconds": 600,
        }
    )
    try:
        assert _get_max_concurrent_children() == 0
        assert _get_max_spawn_depth() == 0
    finally:
        reset_active_route_limits(token)


def test_premium_route_requires_job_approval(hermes_env):
    from cron.scheduler import run_job

    _write_registry(
        hermes_env,
        {
            "reasoning.premium": {
                "provider": "anthropic",
                "model": "claude",
                "status": "active",
                "limits": {"provider_call_allowed": True, "premium_reasoning_allowed": True},
            }
        },
    )
    job = {
        "id": "premium-1",
        "name": "premium",
        "route_alias": "reasoning.premium",
        "prompt": "analyze",
        "no_agent": False,
    }
    ok, _doc, _out, error = run_job(job)
    assert ok is False
    assert "explicit" in (error or "")


def test_alias_never_silently_overrides_legacy_fields(hermes_env):
    from cron.jobs import create_job

    _write_registry(hermes_env, {"executor.infrastructure": {
        "provider": "openai-codex", "model": "codex", "status": "active"
    }})
    with pytest.raises(ValueError, match="conflicts"):
        create_job(
            prompt="work", schedule="every 5m", route_alias="executor.infrastructure",
            provider="legacy", model="legacy-model",
        )


def test_openrouter_fallback_is_rejected(hermes_env):
    from cron.route_aliases import RouteAliasError, resolve_route_alias

    _write_registry(hermes_env, {"safe": {
        "provider": "openai-codex", "model": "codex", "status": "active",
        "fallback": [{"provider": "openrouter", "model": "bad"}],
    }})
    with pytest.raises(RouteAliasError, match="forbidden"):
        resolve_route_alias("safe", hermes_home=hermes_env)


def test_terra_fallback_is_rejected(hermes_env):
    from cron.route_aliases import RouteAliasError, resolve_route_alias

    _write_registry(hermes_env, {"safe": {
        "provider": "openai-codex", "model": "codex", "status": "active",
        "fallback": [{"provider": "terra", "model": "bad"}],
    }})
    with pytest.raises(RouteAliasError, match="forbidden"):
        resolve_route_alias("safe", hermes_home=hermes_env)


def test_route_limits_are_read_without_credentials(hermes_env):
    from cron.route_aliases import resolve_route_alias

    _write_registry(hermes_env, {"executor.infrastructure": {
        "provider": "openai-codex", "model": "codex", "status": "active",
        "limits": {"max_retries": 1, "timeout_seconds": 45,
                   "max_delegation_depth": 0, "max_child_tasks": 0,
                   "provider_call_allowed": True,
                   "premium_reasoning_allowed": False},
    }})
    route = resolve_route_alias("executor.infrastructure", hermes_home=hermes_env)
    assert route.max_retries == 1 and route.timeout_seconds == 45


def test_active_route_with_missing_model_fails_closed(hermes_env):
    from cron.route_aliases import RouteAliasError, resolve_route_alias

    _write_registry(hermes_env, {"incomplete": {
        "provider": "openai-codex", "model": None, "status": "active"
    }})
    with pytest.raises(RouteAliasError, match="requires provider and model"):
        resolve_route_alias("incomplete", hermes_home=hermes_env)


def test_disabled_route_definition_can_be_inspected_but_not_resolved(hermes_env):
    from cron.route_aliases import resolve_route_alias, validate_route_alias_definition

    _write_registry(hermes_env, {"future": {
        "provider": "anthropic", "model": None,
        "status": "disabled-explicit-approval-required"
    }})
    assert validate_route_alias_definition("future", hermes_home=hermes_env).status.startswith("disabled")
    with pytest.raises(Exception, match="disabled"):
        resolve_route_alias("future", hermes_home=hermes_env)


def test_agent_route_cannot_be_marked_no_agent(hermes_env):
    from cron.jobs import create_job

    _write_registry(hermes_env, {"executor.infrastructure": {
        "provider": "openai-codex", "model": "codex", "status": "active"
    }})
    script = hermes_env / "scripts" / "x.sh"
    script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="agent route"):
        create_job(
            prompt=None, schedule="every 5m", script="x.sh", no_agent=True,
            route_alias="executor.infrastructure",
        )
