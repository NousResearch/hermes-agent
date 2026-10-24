"""Regression for #142839: a ``provider/model`` prefix repeated in ``model.default``.

``model.provider: china`` already selects the endpoint, so ``model.default: china/<model>`` names
the same provider twice and every reader forwards the whole string as the upstream ``model``
field. Exact-name routers (New API / one-api style) answer ``503 model_not_found`` and the turn
silently continues on the fallback model, which reads as an unrelated upstream outage.

The fix is diagnostic, not destructive: the value is never rewritten (a slash is part of the real
id for ``openrouter/auto``, ``openrouter/pareto-code``, ``nvidia/<catalog id>``, and for a LiteLLM
proxy serving ``openai/gpt-4o``), so what ships is a loud, actionable line on the load path plus
a ``hermes config check`` diagnostic that names the key and the command to fix it.
"""

import logging
import os
from unittest.mock import patch

import pytest

from hermes_cli import config as config_mod
from hermes_cli.config import load_config, redundant_provider_prefix_notice


def _cfg(default, provider, **extra):
    config = {"model": {"default": default, "provider": provider}}
    config.update(extra)
    return config


def _load(tmp_path, model_section: str, providers: str = ""):
    body = f"model:\n{model_section}"
    if providers:
        body += f"\nproviders:\n{providers}"
    (tmp_path / "config.yaml").write_text(body, encoding="utf-8")
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        return load_config()


# ── the reported shape: a declared custom endpoint, prefix repeats it ──────────

def test_redundant_prefix_on_a_custom_provider_is_reported():
    notice = redundant_provider_prefix_notice(
        _cfg("china/deepseek-v4-flash", "china", providers={"china": {"base_url": "http://x/v1"}})
    )
    assert notice and "hermes config set model.default deepseek-v4-flash" in notice


def test_bare_custom_provider_is_reported():
    notice = redundant_provider_prefix_notice(_cfg("custom/qwen3-32b", "custom"))
    assert notice and "hermes config set model.default qwen3-32b" in notice


def test_custom_slug_provider_matches_the_declared_name():
    # ``custom:<slug>`` is what a named entry canonicalizes to; users write the declared name.
    notice = redundant_provider_prefix_notice(_cfg("china/qwen3-32b", "custom:china"))
    assert notice is not None


def test_prefix_match_is_case_insensitive():
    notice = redundant_provider_prefix_notice(
        _cfg("China/qwen3-32b", "china", providers={"china": {}})
    )
    assert notice is not None


def test_message_names_both_sides():
    notice = redundant_provider_prefix_notice(
        _cfg("china/qwen3-32b", "china", providers={"china": {}})
    )
    assert "model.default 'china/qwen3-32b'" in notice
    assert "model.provider is already 'china'" in notice


# ── nothing is ever rewritten (the reviewer-proved corruptions) ───────────────

@pytest.mark.parametrize(
    "default,provider,providers",
    [
        # OpenRouter pseudo-ids whose prefix is the provider's own namespace.
        ("openrouter/auto", "openrouter", {"openrouter": {}}),
        ("openrouter/pareto-code", "openrouter", {"openrouter": {}}),
        # A curated catalogue whose ids carry the provider prefix (nvidia).
        ("nvidia/nemotron-3-super-120b-a12b", "nvidia", {"nvidia": {}}),
        # LiteLLM-style proxy under a *builtin* provider name: ``openai/gpt-4o`` is a real route.
        ("openai/gpt-4o", "openai", {"openai": {}}),
        # Aggregator vendor namespaces on a builtin route.
        ("anthropic/claude-sonnet-4.5", "openrouter", {"openrouter": {}}),
    ],
)
def test_namespaced_ids_are_left_alone(default, provider, providers):
    config = _cfg(default, provider, providers=providers)
    assert redundant_provider_prefix_notice(config) is None
    assert config["model"]["default"] == default


def test_loader_never_mutates_the_model_id(tmp_path):
    # The corruption the first draft caused: ``openrouter/auto`` -> ``auto`` (not a real id),
    # persisted on the next save.
    config = _load(
        tmp_path,
        "  default: openrouter/auto\n  provider: openrouter\n",
        providers="  openrouter:\n    base_url: https://openrouter.ai/api/v1\n",
    )
    assert config["model"]["default"] == "openrouter/auto"


def test_loader_never_mutates_a_reported_config_either(tmp_path):
    config = _load(
        tmp_path,
        "  default: china/qwen3-32b\n  provider: china\n",
        providers="  china:\n    base_url: http://feiniu:3001/v1\n",
    )
    assert config["model"]["default"] == "china/qwen3-32b"


# ── the guards that keep false positives out ─────────────────────────────────

def test_bare_id_without_prefix_is_silent():
    assert redundant_provider_prefix_notice(
        _cfg("qwen3-32b", "china", providers={"china": {}})) is None


def test_prefix_naming_a_different_provider_is_silent():
    assert redundant_provider_prefix_notice(
        _cfg("openai/gpt-5.4", "china", providers={"china": {}})) is None


def test_no_configured_provider_is_silent():
    assert redundant_provider_prefix_notice({"model": {"default": "china/qwen3-32b"}}) is None


def test_auto_provider_is_silent():
    assert redundant_provider_prefix_notice(_cfg("auto/qwen3-32b", "auto")) is None


def test_empty_remainder_is_silent():
    assert redundant_provider_prefix_notice(
        _cfg("china/", "china", providers={"china": {}})) is None


def test_undeclared_and_builtin_provider_is_silent():
    # No ``providers:`` entry and not a ``custom`` spelling: the route is builtin, so the prefix
    # may well be the upstream namespace. Reporting a guess is worse than silence.
    assert redundant_provider_prefix_notice(_cfg("openai/gpt-5.4", "openai")) is None


# ── the load path actually emits it, once ────────────────────────────────────

def test_load_emits_one_warning_and_no_mutation(tmp_path, caplog):
    config_mod._WARNED_PROVIDER_PREFIX_NOTICES.clear()
    with caplog.at_level(logging.WARNING, logger="hermes_cli.config"):
        first = _load(
            tmp_path,
            "  default: china/qwen3-32b\n  provider: china\n",
            providers="  china:\n    base_url: http://feiniu:3001/v1\n",
        )
        second = _load(
            tmp_path,
            "  default: china/qwen3-32b\n  provider: china\n",
            providers="  china:\n    base_url: http://feiniu:3001/v1\n",
        )
    hits = [r for r in caplog.records if "repeats the provider name" in r.getMessage()]
    assert len(hits) == 1, [r.getMessage() for r in caplog.records]
    assert first["model"]["default"] == second["model"]["default"] == "china/qwen3-32b"


def test_load_is_silent_for_a_namespaced_aggregator_id(tmp_path, caplog):
    config_mod._WARNED_PROVIDER_PREFIX_NOTICES.clear()
    with caplog.at_level(logging.WARNING, logger="hermes_cli.config"):
        _load(
            tmp_path,
            "  default: openrouter/auto\n  provider: openrouter\n",
            providers="  openrouter:\n    base_url: https://openrouter.ai/api/v1\n",
        )
    assert not [r for r in caplog.records if "repeats the provider name" in r.getMessage()]


def test_config_check_reports_the_same_notice():
    from hermes_cli.config_check_diagnostics import config_check_diagnostics

    config = _cfg("china/qwen3-32b", "china", providers={"china": {"base_url": "http://x/v1"}})
    diagnostics = config_check_diagnostics(config, lambda _name: None)
    assert any("hermes config set model.default qwen3-32b" in d for d in diagnostics)


def test_config_check_is_clean_for_an_aggregator_id():
    from hermes_cli.config_check_diagnostics import config_check_diagnostics

    config = _cfg("openrouter/auto", "openrouter", providers={"openrouter": {}})
    diagnostics = config_check_diagnostics(config, lambda _name: None)
    assert not any("repeats the provider name" in d for d in diagnostics)
