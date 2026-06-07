"""Tests for hooks/pre_tool_use/entropy_guard.py — subprocess invocation pattern."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HOOK_SCRIPT = Path(__file__).parent.parent.parent / "hooks" / "pre_tool_use" / "entropy_guard.py"


def _run(payload, *, extra_env=None):
    stdin = json.dumps(payload).encode() if payload is not None else b""
    env = {**os.environ, **(extra_env or {})}
    return subprocess.run(
        [sys.executable, str(HOOK_SCRIPT)],
        input=stdin,
        capture_output=True,
        env=env,
    )


class TestNominalExit:
    def test_nominal_exit_zero(self):
        result = _run({"tool_name": "write_file", "tool_input": {}, "session_id": "t"})
        assert result.returncode == 0

    def test_medium_confidence_nominal(self):
        result = _run(
            {"tool_name": "write_file", "tool_input": {}, "session_id": "t"},
            extra_env={"HERMES_CONFIDENCE": "MEDIUM"},
        )
        assert result.returncode == 0


class TestSignalDetection:
    def test_unknown_confidence_exits_2(self):
        result = _run(
            {"tool_name": "write_file", "tool_input": {}, "session_id": "t"},
            extra_env={"HERMES_CONFIDENCE": "UNKNOWN"},
        )
        assert result.returncode == 2
        data = json.loads(result.stderr)
        assert data["signal"] == "unknown_confidence"

    def test_stale_marker_bracket_exits_2(self):
        result = _run(
            {"tool_name": "write_file", "tool_input": {"q": "[STALE] old data"}, "session_id": "t"},
        )
        assert result.returncode == 2
        data = json.loads(result.stderr)
        assert data["signal"] == "stale_memory"

    def test_stale_marker_colon_exits_2(self):
        result = _run(
            {"tool_name": "write_file", "tool_input": {"q": "STALE: old data"}, "session_id": "t"},
        )
        assert result.returncode == 2
        data = json.loads(result.stderr)
        assert data["signal"] == "stale_memory"

    def test_conflicting_sources_exits_2(self):
        result = _run(
            {"tool_name": "write_file", "tool_input": {}, "session_id": "t"},
            extra_env={"HERMES_CONFIDENCE": "HIGH", "HERMES_CONFLICTING_SOURCES": "1"},
        )
        assert result.returncode == 2
        data = json.loads(result.stderr)
        assert data["signal"] == "conflicting_sources"

    def test_unverified_claim_non_tier1_exits_2(self):
        result = _run(
            {"tool_name": "write_file", "tool_input": {}, "session_id": "t"},
            extra_env={"HERMES_CONFIDENCE": "LOW"},
        )
        assert result.returncode == 2
        data = json.loads(result.stderr)
        assert data["signal"] == "unverified_claim"


class TestTier1Suppression:
    def test_tier1_suppresses_high_signal(self):
        result = _run(
            {"tool_name": "read_file", "tool_input": {}, "session_id": "t"},
            extra_env={"HERMES_CONFIDENCE": "LOW"},
        )
        assert result.returncode == 0

    def test_critical_overrides_tier1(self):
        result = _run(
            {"tool_name": "read_file", "tool_input": {}, "session_id": "t"},
            extra_env={"HERMES_CONFIDENCE": "UNKNOWN"},
        )
        assert result.returncode == 2
        data = json.loads(result.stderr)
        assert data["signal"] == "unknown_confidence"


class TestToolLoop:
    def test_tool_loop_exits_2(self, tmp_path):
        tool_input = {}
        hash_key = hashlib.sha256(
            json.dumps(tool_input, sort_keys=True).encode()
        ).hexdigest()[:16]
        entry_key = f"write_file:{hash_key}"
        ledger = {"test-loop": [entry_key, entry_key, entry_key]}
        ledger_file = tmp_path / "call_ledger.json"
        ledger_file.write_text(json.dumps(ledger), encoding="utf-8")

        result = _run(
            {"tool_name": "write_file", "tool_input": tool_input, "session_id": "test-loop"},
            extra_env={"HERMES_LEDGER_PATH": str(ledger_file)},
        )
        assert result.returncode == 2
        data = json.loads(result.stderr)
        assert data["signal"] == "tool_loop"

    def test_missing_ledger_no_tool_loop(self, tmp_path):
        nonexistent = tmp_path / "no_ledger.json"
        result = _run(
            {"tool_name": "write_file", "tool_input": {}, "session_id": "t"},
            extra_env={
                "HERMES_CONFIDENCE": "HIGH",
                "HERMES_LEDGER_PATH": str(nonexistent),
            },
        )
        assert result.returncode == 0


class TestEdgeCases:
    def test_empty_stdin_exits_zero(self):
        env = {**os.environ}
        result = subprocess.run(
            [sys.executable, str(HOOK_SCRIPT)],
            input=b"",
            capture_output=True,
            env=env,
        )
        assert result.returncode == 0

    def test_invalid_json_exits_zero(self):
        env = {**os.environ}
        result = subprocess.run(
            [sys.executable, str(HOOK_SCRIPT)],
            input=b"not json",
            capture_output=True,
            env=env,
        )
        assert result.returncode == 0

    def test_stderr_is_json_on_block(self):
        result = _run(
            {"tool_name": "write_file", "tool_input": {}, "session_id": "t"},
            extra_env={"HERMES_CONFIDENCE": "UNKNOWN"},
        )
        assert result.returncode == 2
        data = json.loads(result.stderr)
        assert "signal" in data
        assert "level" in data
        assert "recommended_resolution" in data


class TestEnvVarOverride:
    def test_hermes_ledger_path_env_var(self, tmp_path):
        custom_path = tmp_path / "custom_ledger.json"
        result = _run(
            {"tool_name": "write_file", "tool_input": {}, "session_id": "t"},
            extra_env={"HERMES_LEDGER_PATH": str(custom_path)},
        )
        assert result.returncode == 0

    def test_hermes_memory_db_env_var(self, tmp_path):
        # [DEVIATION 005] DIP specified returncode == 0, but _get_fact_confidence returns
        # "UNKNOWN" for a missing DB, which fires unknown_confidence CRITICAL. This
        # matches RAFAEL source logic. See Field Discoveries.
        custom_db = tmp_path / "custom_memory.db"
        result = _run(
            {"tool_name": "recall_memory", "tool_input": {"fact_id": "x"}, "session_id": "t"},
            extra_env={"HERMES_MEMORY_DB": str(custom_db)},
        )
        assert result.returncode == 2
        data = json.loads(result.stderr)
        assert data["signal"] == "unknown_confidence"
