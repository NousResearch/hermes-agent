"""E2E subprocess tests for ``hermes models``/``hermes providers`` CLI."""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest


HERMES = shutil.which("hermes")

pytestmark = pytest.mark.skipif(
    HERMES is None,
    reason="`hermes` console script not on PATH; install with `pip install -e .`",
)


@pytest.fixture
def clean_env(tmp_path):
    h = tmp_path / "hermes_home"
    h.mkdir()
    env = os.environ.copy()
    env["HERMES_HOME"] = str(h)
    env.pop("PYTHONPATH", None)  # don't shadow the worktree install
    for var in [
        "ARCEEAI_API_KEY",
        "ARCEE_API_KEY",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "NOUS_API_KEY",
        "GH_TOKEN",
        "COPILOT_GITHUB_TOKEN",
        "DEEPSEEK_API_KEY",
        "GLM_API_KEY",
        "ZAI_API_KEY",
        "STEPFUN_API_KEY",
        "GMI_API_KEY",
        "TOGETHER_API_KEY",
        "FIREWORKS_API_KEY",
        "GROQ_API_KEY",
        "GOOGLE_API_KEY",
        "QWEN_API_KEY",
        "MOONSHOT_API_KEY",
        "MINIMAX_API_KEY",
        "BEDROCK_API_KEY",
        "CEREBRAS_API_KEY",
        "VERCEL_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_PROFILE",
    ]:
        env.pop(var, None)
    return env


def _run(args: list[str], env: dict, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        [HERMES, *args], env=env, capture_output=True, text=True, timeout=timeout
    )


def test_providers_list_offline_all_json(clean_env):
    r = _run(["providers", "list", "--offline", "--all", "--json"], clean_env)
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout)
    assert data["schema_version"] == 1
    assert len(data["providers"]) >= 35  # CANONICAL_PROVIDERS universe


def test_models_list_openrouter_offline_uses_constant(clean_env):
    """H3 regression via CLI: openrouter offline must match OPENROUTER_MODELS."""
    from hermes_cli.models import OPENROUTER_MODELS

    r = _run(
        ["models", "list", "--provider=openrouter", "--offline", "--json"],
        clean_env,
    )
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout)
    assert data["providers"][0]["total_models"] == len(OPENROUTER_MODELS)


def test_models_list_unknown_provider_exits_2(clean_env):
    r = _run(["models", "list", "--provider=does-not-exist"], clean_env)
    assert r.returncode == 2
    assert "unknown provider" in r.stderr.lower()


def test_models_missing_subcommand_exits_2(clean_env):
    r = _run(["models"], clean_env)
    assert r.returncode == 2


def test_models_unknown_subcommand_exits_2(clean_env):
    r = _run(["models", "foo"], clean_env)
    assert r.returncode == 2


def test_providers_missing_subcommand_exits_2(clean_env):
    r = _run(["providers"], clean_env)
    assert r.returncode == 2


def test_status_offline_returns_unconfigured_baseline(clean_env):
    """In a clean env, no providers should be configured."""
    r = _run(["models", "status", "--offline", "--json"], clean_env)
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout)
    # arcee is the canonical clean-baseline target (single env var, no
    # auth_store seeding) — guaranteed unconfigured here.
    arcee = next((r for r in data["providers"] if r["slug"] == "arcee"), None)
    assert arcee is not None
    assert arcee["auth_state"] == "unconfigured"


def test_arcee_env_var_flips_to_configured(clean_env):
    """H6 regression via CLI: ARCEEAI_API_KEY env var → auth_state=configured."""
    clean_env["ARCEEAI_API_KEY"] = "test-canary"
    r = _run(["providers", "list", "--all", "--offline", "--json"], clean_env)
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout)
    arcee = next((r for r in data["providers"] if r["slug"] == "arcee"), None)
    assert arcee is not None and arcee["auth_state"] == "configured"


def test_text_renderers_smoke(clean_env):
    r = _run(["providers", "list", "--offline", "--all"], clean_env)
    assert r.returncode == 0, r.stderr
    assert "SLUG" in r.stdout
    assert "providers configured" in r.stdout
