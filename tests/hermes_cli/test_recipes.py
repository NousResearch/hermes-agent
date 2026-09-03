"""Tests for hermes_cli/recipes.py — shareable setup bundles."""

import copy

import pytest

from hermes_cli.recipes import (
    RecipeError,
    _sanitize_job,
    _sanitize_mcp_entry,
    describe_recipe,
    dump_recipe,
    validate_recipe,
)


# ---------------------------------------------------------------------------
# validate_recipe
# ---------------------------------------------------------------------------

def _minimal_recipe(**overrides):
    recipe = {"recipe": 1, "name": "My Setup"}
    recipe.update(overrides)
    return recipe


def test_validate_minimal_recipe_passes():
    assert validate_recipe(_minimal_recipe())["name"] == "My Setup"


def test_validate_rejects_non_mapping():
    with pytest.raises(RecipeError, match="mapping"):
        validate_recipe(["not", "a", "dict"])


def test_validate_rejects_missing_version():
    with pytest.raises(RecipeError, match="version"):
        validate_recipe({"name": "x"})


def test_validate_rejects_future_version():
    with pytest.raises(RecipeError, match="newer"):
        validate_recipe(_minimal_recipe(recipe=99))


def test_validate_rejects_empty_name():
    with pytest.raises(RecipeError, match="name"):
        validate_recipe({"recipe": 1, "name": "  "})


def test_validate_rejects_script_jobs():
    recipe = _minimal_recipe(
        cron_jobs=[{"name": "bad", "prompt": "p", "schedule": "daily 9am",
                    "script": "evil.sh"}]
    )
    with pytest.raises(RecipeError, match="script"):
        validate_recipe(recipe)


def test_validate_rejects_no_agent_jobs():
    recipe = _minimal_recipe(
        cron_jobs=[{"name": "bad", "prompt": "p", "schedule": "daily", "no_agent": True}]
    )
    with pytest.raises(RecipeError, match="no_agent"):
        validate_recipe(recipe)


def test_validate_rejects_job_without_prompt():
    recipe = _minimal_recipe(cron_jobs=[{"name": "x", "schedule": "daily 9am"}])
    with pytest.raises(RecipeError, match="prompt"):
        validate_recipe(recipe)


def test_validate_rejects_stdio_mcp():
    recipe = _minimal_recipe(mcp_servers={"local": {"command": "npx something"}})
    with pytest.raises(RecipeError, match="stdio"):
        validate_recipe(recipe)


def test_validate_rejects_mcp_without_url():
    recipe = _minimal_recipe(mcp_servers={"s": {"transport": "http"}})
    with pytest.raises(RecipeError, match="url"):
        validate_recipe(recipe)


def test_validate_rejects_mcp_with_secret_field():
    recipe = _minimal_recipe(
        mcp_servers={"s": {"url": "https://mcp.example.com/mcp", "api_key": "sk-123"}}
    )
    with pytest.raises(RecipeError, match="secret"):
        validate_recipe(recipe)


def test_validate_rejects_mcp_with_headers():
    recipe = _minimal_recipe(
        mcp_servers={"s": {"url": "https://mcp.example.com/mcp",
                           "headers": {"Authorization": "Bearer x"}}}
    )
    with pytest.raises(RecipeError, match="secret-shaped"):
        validate_recipe(recipe)


def test_validate_rejects_unsafe_mcp_url():
    recipe = _minimal_recipe(
        mcp_servers={"s": {"url": "https://169.254.169.254/latest/meta-data"}}
    )
    with pytest.raises(RecipeError, match="SSRF|refused"):
        validate_recipe(recipe)


def test_validate_accepts_clean_remote_mcp(monkeypatch):
    import tools.url_safety as url_safety

    monkeypatch.setattr(url_safety, "is_safe_url", lambda url: True)
    recipe = _minimal_recipe(
        mcp_servers={"s": {"url": "https://mcp.example.com/mcp", "transport": "http"}}
    )
    assert validate_recipe(recipe)["mcp_servers"]["s"]["url"].startswith("https")


def test_validate_rejects_non_string_skills():
    with pytest.raises(RecipeError, match="skills"):
        validate_recipe(_minimal_recipe(skills=[{"nested": "dict"}]))


# ---------------------------------------------------------------------------
# export sanitizers
# ---------------------------------------------------------------------------

def test_sanitize_mcp_strips_headers_and_records_required():
    entry = {
        "transport": "http",
        "url": "https://mcp.example.com/sse",
        "headers": {"Authorization": "Bearer secret", "X-Custom": "v"},
        "api_key": "sk-live-123",
        "description": "example",
    }
    clean, required = _sanitize_mcp_entry("ex", entry)
    assert clean == {
        "transport": "http",
        "url": "https://mcp.example.com/sse",
        "description": "example",
    }
    assert "Authorization" in required and "api_key" in required
    dumped = dump_recipe({"recipe": 1, "name": "n", "mcp_servers": {"ex": clean}})
    assert "sk-live-123" not in dumped
    assert "Bearer" not in dumped


def test_sanitize_mcp_refuses_stdio():
    with pytest.raises(RecipeError, match="stdio"):
        _sanitize_mcp_entry("local", {"command": "npx foo"})


def test_sanitize_job_keeps_portable_fields_only():
    job = {
        "id": "abc123",
        "name": "daily digest",
        "prompt": "Summarize my day",
        "schedule": {"raw": "daily 8am", "kind": "cron"},
        "deliver": "telegram",
        "origin": {"chat_id": 12345},
        "last_run_at": "2026-08-01",
        "failure_streak": 3,
    }
    clean = _sanitize_job(copy.deepcopy(job))
    assert clean["schedule"] == "daily 8am"
    assert clean["deliver"] == "local"  # host-specific target rewritten
    assert "origin" not in clean and "last_run_at" not in clean
    assert "id" not in clean and "failure_streak" not in clean


def test_sanitize_job_refuses_script_jobs():
    with pytest.raises(RecipeError, match="script"):
        _sanitize_job({"name": "w", "prompt": "p", "schedule": "daily",
                       "script": "watch.sh"})


# ---------------------------------------------------------------------------
# describe
# ---------------------------------------------------------------------------

def test_describe_mentions_paused_jobs_and_secrets():
    recipe = {
        "recipe": 1,
        "name": "Research Kit",
        "cron_jobs": [{"name": "arxiv sweep", "prompt": "scan arxiv",
                       "schedule": "daily 7am"}],
        "mcp_servers": {"ex": {"url": "https://mcp.example.com/mcp"}},
        "required_secrets": {"ex": ["api_key"]},
        "skills": ["official/research/arxiv"],
        "starter_prompt": "Give me today's papers",
    }
    text = describe_recipe(recipe)
    assert "PAUSED" in text
    assert "api_key" in text
    assert "official/research/arxiv" in text
    assert "Give me today's papers" in text
