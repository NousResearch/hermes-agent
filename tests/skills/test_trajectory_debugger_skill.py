"""
Tests for the Trajectory Debugger skill (skills/software-development/trajectory-debugger).

Covers:
- Loading events from JSONL transcript files
- Calculating overall prompt cache hit rates
- Detecting cache-busting drops
- Aggregating tool invocation counts and error rates
- Step inspection via CLI
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEBUGGER_SCRIPT = (
    REPO_ROOT
    / "skills"
    / "software-development"
    / "trajectory-debugger"
    / "scripts"
    / "debug_trajectory.py"
)

# Import module directly
sys.path.insert(0, str(DEBUGGER_SCRIPT.parent))
import debug_trajectory


@pytest.fixture
def mock_transcript(tmp_path):
    transcript_file = tmp_path / "mock_transcript.jsonl"
    events = [
        {
            "step_index": 0,
            "type": "USER_INPUT",
            "content": "Analyze the codebase and fix bug in parser.",
        },
        {
            "step_index": 1,
            "type": "PLANNER_RESPONSE",
            "content": "Reading the parser file...",
            "tool_calls": [{"name": "read_file", "args": {"path": "parser.py"}}],
            "status": "DONE",
            "usage": {
                "prompt_tokens": 1000,
                "prompt_cache_hit_tokens": 900,
                "completion_tokens": 100,
            },
        },
        {
            "step_index": 2,
            "type": "PLANNER_RESPONSE",
            "content": "Running test suite...",
            "tool_calls": [{"name": "terminal", "args": {"command": "pytest"}}],
            "status": "ERROR",
            "usage": {
                "prompt_tokens": 1200,
                "prompt_cache_hit_tokens": 1100,
                "completion_tokens": 50,
            },
        },
        {
            "step_index": 3,
            "type": "PLANNER_RESPONSE",
            "content": "Cache was busted due to full system prompt change...",
            "tool_calls": [{"name": "replace_file_content", "args": {}}],
            "status": "DONE",
            "usage": {
                "prompt_tokens": 2000,
                "prompt_cache_hit_tokens": 200,  # 10% hit rate -> cache bust
                "completion_tokens": 150,
            },
        },
    ]
    with open(transcript_file, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")
    return transcript_file


class TestTrajectoryDebuggerCore:
    def test_load_transcript_events(self, mock_transcript):
        events = debug_trajectory.load_transcript_events(mock_transcript)
        assert len(events) == 4
        assert events[0]["type"] == "USER_INPUT"
        assert events[1]["tool_calls"][0]["name"] == "read_file"

    def test_analyze_cache_efficiency(self, mock_transcript):
        events = debug_trajectory.load_transcript_events(mock_transcript)
        res = debug_trajectory.analyze_cache_efficiency(events, threshold_pct=80.0)

        assert res["total_prompt_tokens"] == 4200
        assert res["total_cached_tokens"] == 2200
        assert round(res["overall_cache_hit_rate_pct"], 1) == 52.4
        assert res["cache_busts_detected"] == 1
        assert res["cache_busts"][0]["step_index"] == 3
        assert res["cache_busts"][0]["current_hit_rate"] == 10.0

    def test_analyze_tool_usage(self, mock_transcript):
        events = debug_trajectory.load_transcript_events(mock_transcript)
        res = debug_trajectory.analyze_tool_usage(events)

        assert res["total_tools_called"] == 3
        assert res["tools"]["read_file"]["calls"] == 1
        assert res["tools"]["read_file"]["errors"] == 0
        assert res["tools"]["terminal"]["calls"] == 1
        assert res["tools"]["terminal"]["errors"] == 1

    def test_analyze_trajectory_summary(self, mock_transcript):
        summary = debug_trajectory.analyze_trajectory_summary(mock_transcript)
        assert summary["total_steps"] == 4
        assert summary["user_messages"] == 1
        assert summary["model_responses"] == 3
        assert summary["cache_summary"]["cache_bust_drops"] == 1


class TestTrajectoryDebuggerCLI:
    def test_cli_analyze_json(self, mock_transcript):
        res = subprocess.run(
            [
                sys.executable,
                str(DEBUGGER_SCRIPT),
                "analyze",
                str(mock_transcript),
                "--json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(res.stdout)
        assert data["total_steps"] == 4
        assert data["cache_summary"]["overall_hit_rate_pct"] > 50

    def test_cli_cache_json(self, mock_transcript):
        res = subprocess.run(
            [
                sys.executable,
                str(DEBUGGER_SCRIPT),
                "cache",
                str(mock_transcript),
                "--threshold",
                "75",
                "--json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(res.stdout)
        assert data["cache_busts_detected"] == 1
        assert len(data["turns"]) == 3

    def test_cli_turn_inspection(self, mock_transcript):
        res = subprocess.run(
            [
                sys.executable,
                str(DEBUGGER_SCRIPT),
                "turn",
                str(mock_transcript),
                "--turn",
                "1",
                "--json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(res.stdout)
        assert data["step_index"] == 1
        assert data["tool_calls"][0]["name"] == "read_file"
