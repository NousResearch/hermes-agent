"""Tests for the bundled completion-auditor plugin."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("HERMES_COMPLETION_AUDITOR_LOG_DIR", raising=False)
    yield hermes_home


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_plugin_init():
    plugin_dir = _repo_root() / "plugins" / "completion-auditor"
    if "hermes_plugins" not in sys.modules:
        ns = types.ModuleType("hermes_plugins")
        ns.__path__ = []
        sys.modules["hermes_plugins"] = ns
    for key in list(sys.modules):
        if key.startswith("hermes_plugins.completion_auditor_under_test"):
            del sys.modules[key]
    spec = importlib.util.spec_from_file_location(
        "hermes_plugins.completion_auditor_under_test",
        plugin_dir / "__init__.py",
        submodule_search_locations=[str(plugin_dir)],
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "hermes_plugins.completion_auditor_under_test"
    mod.__path__ = [str(plugin_dir)]
    sys.modules["hermes_plugins.completion_auditor_under_test"] = mod
    spec.loader.exec_module(mod)
    mod._reset_for_tests()
    return mod


def _audit_records(hermes_home: Path) -> list[dict]:
    records = []
    for path in sorted((hermes_home / "logs" / "completion-auditor").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            records.append(json.loads(line))
    return records


class TestCompletionAuditorHooks:
    def test_post_llm_call_writes_metadata_only_jsonl_record(self, _isolate_env):
        mod = _load_plugin_init()

        assert mod._on_post_tool_call(
            session_id="s1",
            turn_id="turn-1",
            task_id="task-1",
            tool_name="terminal",
            tool_call_id="tc1",
            status="success",
            duration_ms=12,
            result='{"output": "SECRET RAW OUTPUT SHOULD NOT BE LOGGED"}',
        ) is None
        assert mod._ledger_size_for_tests() == 1

        original = "테스트 통과했어."
        assert mod._on_post_llm_call(
            session_id="s1",
            turn_id="turn-1",
            task_id="task-1",
            assistant_response=original,
        ) is None

        assert mod._ledger_size_for_tests() == 0
        records = _audit_records(_isolate_env)
        assert len(records) == 1
        record = records[0]
        assert record["schema"] == "hermes-completion-audit-v1"
        assert record["session_id"] == "s1"
        assert record["turn_id"] == "turn-1"
        assert record["task_id"] == "task-1"
        assert record["plugin_enabled"] is True
        assert record["audit_executed"] is True
        assert record["mode"] == "audit"
        assert record["verdict"] == "weak"
        assert record["claim_type"] == "tested"
        assert record["claim_text"] == original
        assert record["semantic_correctness_guaranteed"] is False
        assert record["final_response_mutated"] is False
        assert record["tool_result_excerpt_included"] is False
        assert record["assistant_response_chars"] == len(original)
        assert len(record["evidence_refs"]) == 1
        assert record["evidence_refs"][0]["tool_name"] == "terminal"
        assert "result_excerpt" not in record["evidence_refs"][0]
        assert "SECRET RAW OUTPUT" not in json.dumps(record)

        log_dir = _isolate_env / "logs" / "completion-auditor"
        log_file = next(log_dir.glob("*.jsonl"))
        if os.name == "posix":
            assert log_dir.stat().st_mode & 0o777 == 0o700
            assert log_file.stat().st_mode & 0o777 == 0o600

    def test_missing_turn_identity_records_audit_error(self, _isolate_env):
        mod = _load_plugin_init()
        mod._on_post_llm_call(session_id="s1", assistant_response="done")
        record = _audit_records(_isolate_env)[0]
        assert record["verdict"] == "audit_error"
        assert record["degraded"] is True
        assert record["degrade_reason"] == "missing session_id or turn_id"

    def test_unsupported_runtime_mode_is_noop(self, _isolate_env):
        import yaml

        (_isolate_env / "config.yaml").write_text(
            yaml.safe_dump({"plugins": {"completion_auditor": {"mode": "enforce"}}}),
            encoding="utf-8",
        )
        mod = _load_plugin_init()
        mod._on_post_llm_call(session_id="s1", turn_id="t1", assistant_response="done")
        assert _audit_records(_isolate_env) == []

    def test_result_excerpt_redacts_common_secret_shapes(self, _isolate_env):
        import yaml

        (_isolate_env / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "plugins": {
                        "completion_auditor": {
                            "include_tool_result_excerpt": True,
                            "redact_secrets": True,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        mod = _load_plugin_init()
        raw_secret = "sk-1234567890abcdefghijklmnop"
        bearer_secret = "abcdefghijklmnop1234567890"
        mod._on_post_tool_call(
            session_id="s1",
            turn_id="turn-1",
            tool_name="terminal",
            status="success",
            result=f"api_key={raw_secret} Authorization: Bearer {bearer_secret}",
        )
        mod._on_post_llm_call(
            session_id="s1", turn_id="turn-1", assistant_response="Updated config.yaml."
        )
        record = _audit_records(_isolate_env)[0]
        excerpt = record["evidence_refs"][0]["result_excerpt"]
        assert "[REDACTED]" in excerpt
        assert raw_secret not in excerpt
        assert bearer_secret not in excerpt

    def test_result_excerpt_redacts_quoted_dict_secret_values(self, _isolate_env):
        import yaml

        (_isolate_env / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "plugins": {
                        "completion_auditor": {
                            "include_tool_result_excerpt": True,
                            "redact_secrets": True,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        mod = _load_plugin_init()
        mod._on_post_tool_call(
            session_id="s1",
            turn_id="turn-1",
            tool_name="terminal",
            status="success",
            result={"password": "super-secret-token", "token": "another-secret-token"},
        )
        mod._on_post_llm_call(
            session_id="s1", turn_id="turn-1", assistant_response="Updated config.yaml."
        )
        excerpt = _audit_records(_isolate_env)[0]["evidence_refs"][0]["result_excerpt"]
        assert "super-secret-token" not in excerpt
        assert "another-secret-token" not in excerpt
        assert excerpt.count("[REDACTED]") == 2

    def test_hyphenated_runtime_config_alias_is_supported(self, _isolate_env):
        import yaml

        (_isolate_env / "config.yaml").write_text(
            yaml.safe_dump({"plugins": {"completion-auditor": {"log_verdicts": False}}}),
            encoding="utf-8",
        )
        mod = _load_plugin_init()
        mod._on_post_llm_call(session_id="s1", turn_id="t1", assistant_response="done")
        assert _audit_records(_isolate_env) == []

    def test_direct_config_default_log_dir_is_profile_safe(self, _isolate_env):
        _load_plugin_init()
        config_mod = sys.modules["hermes_plugins.completion_auditor_under_test.config"]
        cfg = config_mod.AuditorConfig()
        assert cfg.log_dir == _isolate_env / "logs" / "completion-auditor"

    def test_ledger_prunes_stale_turns(self):
        mod = _load_plugin_init()
        evidence_mod = sys.modules["hermes_plugins.completion_auditor_under_test.evidence"]
        assert mod._on_post_tool_call(
            session_id="s1", turn_id="old", tool_name="terminal", status="success"
        ) is None
        assert mod._on_post_tool_call(
            session_id="s1", turn_id="new", tool_name="terminal", status="success"
        ) is None
        evidence_mod._LEDGER[("s1", "old")].last_seen_at = 0
        evidence_mod._prune_locked(now=evidence_mod._DEFAULT_LEDGER_TTL_SECONDS + 1)
        assert ("s1", "old") not in evidence_mod._LEDGER
        assert ("s1", "new") in evidence_mod._LEDGER


class TestCompletionAuditorPluginRegistration:
    def test_register_adds_observer_hooks(self):
        mod = _load_plugin_init()
        hooks = []

        class Ctx:
            def register_hook(self, name, callback):
                hooks.append((name, callback))

        mod.register(Ctx())
        assert [name for name, _ in hooks] == ["post_tool_call", "post_llm_call"]
        assert all(callback(**{}) is None for _, callback in hooks)

    def test_bundled_plugin_is_disabled_by_default(self, _isolate_env):
        import yaml

        (_isolate_env / "config.yaml").write_text(
            yaml.safe_dump({"plugins": {"enabled": []}}), encoding="utf-8"
        )
        for key in list(sys.modules):
            if key.startswith("hermes_plugins"):
                del sys.modules[key]

        import hermes_cli.plugins as plugins_mod

        plugins_mod._plugin_manager = plugins_mod.PluginManager()
        mgr = plugins_mod._ensure_plugins_discovered(force=True)
        loaded = mgr._plugins["completion-auditor"]
        assert loaded.enabled is False
        assert loaded.error is not None
        assert loaded.error.startswith("not enabled in config")
        assert "post_tool_call" not in mgr._hooks
        assert "post_llm_call" not in mgr._hooks

    def test_bundled_plugin_loads_when_enabled(self, _isolate_env):
        import yaml

        (_isolate_env / "config.yaml").write_text(
            yaml.safe_dump({"plugins": {"enabled": ["completion-auditor"]}}),
            encoding="utf-8",
        )
        for key in list(sys.modules):
            if key.startswith("hermes_plugins"):
                del sys.modules[key]

        import hermes_cli.plugins as plugins_mod

        plugins_mod._plugin_manager = plugins_mod.PluginManager()
        mgr = plugins_mod._ensure_plugins_discovered(force=True)
        loaded = mgr._plugins["completion-auditor"]
        assert loaded.enabled is True
        assert "post_tool_call" in mgr._hooks
        assert "post_llm_call" in mgr._hooks
