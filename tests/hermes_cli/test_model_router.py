"""Skeleton tests for #61371 model_router.

The actual routing dispatch lands in a follow-up PR. These tests pin
the parser/validator behavior so future schema drift is caught early:
default-off (no config or no `enabled: true`), the example block from
the issue body parses AND normalizes correctly, malformed buckets
fail-fast at load time, unknown task labels are rejected, and
comma-joined for_tasks shorthand parses.
"""

from __future__ import annotations

import logging

import pytest

from hermes_cli.model_router import (
    KNOWN_TASK_LABELS,
    ModelRouterConfigError,
    is_router_enabled,
    resolve_router_config,
)


# ---------------------------------------------------------------------------
# default-off contract — never silently route
# ---------------------------------------------------------------------------


class TestRouterIsOffByDefault:
    """#61371 acceptance criterion: existing single-provider behavior
    is preserved until the user explicitly opts in.
    """

    def test_none_config_yields_disabled(self):
        assert is_router_enabled(None) is False

    def test_empty_config_yields_disabled(self):
        assert is_router_enabled({}) is False

    def test_no_model_router_block_yields_disabled(self):
        assert is_router_enabled({"agent": {"max_turns": 90}}) is False

    def test_resolve_returns_disabled_shape(self):
        resolved = resolve_router_config({})
        assert resolved == {
            "enabled": False,
            "buckets": {},
            "default_task_label": None,
        }

    def test_explicit_enabled_false_stays_disabled(self):
        config = {"model_router": {"enabled": False, "local": {}}}
        assert is_router_enabled(config) is False


# ---------------------------------------------------------------------------
# the example block from issue #61371 parses + normalizes correctly
# ---------------------------------------------------------------------------


class TestRouterExampleBlockParses:
    def test_local_and_cloud_buckets_normalize(self):
        config = {
            "model_router": {
                "enabled": True,
                "local": {
                    "provider": "ollama",
                    "model": "llama3.2-3b",
                    "for_tasks": ["read_only", "shell_commands"],
                },
                "cloud": {
                    "provider": "nous",
                    "model": "anthropic/claude-sonnet-4.6",
                    "for_tasks": ["debugging", "code_review", "log_analysis"],
                },
            }
        }
        resolved = resolve_router_config(config)
        assert resolved["enabled"] is True
        assert set(resolved["buckets"].keys()) == {"local", "cloud"}
        assert resolved["buckets"]["local"]["provider"] == "ollama"
        assert resolved["buckets"]["local"]["model"] == "llama3.2-3b"
        assert resolved["buckets"]["local"]["for_tasks"] == [
            "read_only",
            "shell_commands",
        ]
        assert resolved["buckets"]["cloud"]["provider"] == "nous"
        assert resolved["buckets"]["cloud"]["model"] == "anthropic/claude-sonnet-4.6"


# ---------------------------------------------------------------------------
# fail-fast validation
# ---------------------------------------------------------------------------


class TestRouterFailFastValidation:
    def test_bucket_must_be_a_mapping(self):
        config = {"model_router": {"enabled": True, "local": "not-a-dict"}}
        with pytest.raises(ModelRouterConfigError, match="must be a mapping"):
            resolve_router_config(config)

    def test_provider_required_and_nonempty(self):
        config = {
            "model_router": {
                "enabled": True,
                "local": {"model": "llama3.2-3b", "for_tasks": ["read_only"]},
            }
        }
        with pytest.raises(ModelRouterConfigError, match="provider"):
            resolve_router_config(config)

    def test_provider_must_be_string_not_number(self):
        config = {
            "model_router": {
                "enabled": True,
                "local": {
                    "provider": 42,
                    "model": "llama3.2-3b",
                    "for_tasks": ["read_only"],
                },
            }
        }
        with pytest.raises(ModelRouterConfigError, match="provider"):
            resolve_router_config(config)

    def test_for_tasks_string_is_split_on_comma(self):
        # Comma-joined shorthand accepted; colon-joined typo still rejected
        # as an unknown label so users find out at config-load, not dispatch.
        config = {
            "model_router": {
                "enabled": True,
                "local": {
                    "provider": "ollama",
                    "model": "llama3.2-3b",
                    "for_tasks": "read_only, shell_commands",
                },
            }
        }
        resolved = resolve_router_config(config)
        assert resolved["buckets"]["local"]["for_tasks"] == [
            "read_only",
            "shell_commands",
        ]

    def test_non_list_non_string_for_tasks_rejected(self):
        config = {
            "model_router": {
                "enabled": True,
                "local": {
                    "provider": "ollama",
                    "model": "llama3.2-3b",
                    "for_tasks": 42,
                },
            }
        }
        with pytest.raises(ModelRouterConfigError, match="for_tasks"):
            resolve_router_config(config)

    def test_unknown_task_labels_rejected(self):
        config = {
            "model_router": {
                "enabled": True,
                "local": {
                    "provider": "ollama",
                    "model": "llama3.2-3b",
                    "for_tasks": ["read_only", "totally-unknown-label"],
                },
            }
        }
        with pytest.raises(ModelRouterConfigError, match="known vocabulary"):
            resolve_router_config(config)

    def test_enabled_but_no_buckets_is_rejected(self):
        config = {"model_router": {"enabled": True}}
        with pytest.raises(ModelRouterConfigError, match="at least one routing target"):
            resolve_router_config(config)


# ---------------------------------------------------------------------------
# known-label registry drift guards
# ---------------------------------------------------------------------------


class TestKnownTaskLabels:
    def test_vocabulary_includes_issue_example(self):
        """The issue body lists five example labels — they must all live
        in the known-vocabulary set, otherwise the example YAML fails
        the schema guard above.
        """
        for label in (
            "read_only",
            "shell_commands",
            "debugging",
            "code_review",
            "log_analysis",
        ):
            assert label in KNOWN_TASK_LABELS


# ---------------------------------------------------------------------------
# skeleton observability — the runtime logs a "not yet wired" notice so
# operators don't lose visibility when they enable the router skeleton.
# ---------------------------------------------------------------------------


class TestRouterSkeletonsLogs:
    def test_resolving_enabled_router_emits_info(self, caplog):
        config = {
            "model_router": {
                "enabled": True,
                "local": {
                    "provider": "ollama",
                    "model": "llama3.2-3b",
                    "for_tasks": ["read_only"],
                },
            }
        }
        with caplog.at_level(logging.INFO, logger="hermes_cli.model_router"):
            resolve_router_config(config)
        assert any(
            "model_router" in rec.message and "skeleton" in rec.message.lower()
            for rec in caplog.records
        ), "skeleton must surface a runtime-visible 'not yet wired' log"

    def test_disabled_router_does_not_emit_info(self, caplog):
        with caplog.at_level(logging.INFO, logger="hermes_cli.model_router"):
            resolve_router_config({})
        # Disabled router is silent — no spam for users who haven't opted in.
        assert not any(
            "model_router" in rec.message for rec in caplog.records
        )
