"""Tests for the tool control resolver (Phase A2)."""

import pytest
from agent.tool_control import (
    is_control_protocol,
    resolve_pre_tool_control,
    resolve_post_tool_control,
)


class TestIsControlProtocol:
    def test_valid_allow(self):
        assert is_control_protocol({"action": "allow"}) is True

    def test_valid_deny(self):
        assert is_control_protocol({"action": "deny", "reason": "blocked"}) is True

    def test_valid_modify_result(self):
        assert is_control_protocol({"action": "modify_result"}) is True

    def test_none_not_protocol(self):
        assert is_control_protocol(None) is False

    def test_string_not_protocol(self):
        assert is_control_protocol("allow") is False

    def test_dict_no_action(self):
        assert is_control_protocol({"reason": "x"}) is False

    def test_invalid_action(self):
        assert is_control_protocol({"action": "nuke"}) is False


class TestResolvePreToolControl:
    def test_empty_results(self):
        assert resolve_pre_tool_control([], {})["action"] == "allow"

    def test_none_results(self):
        assert resolve_pre_tool_control([None, "garbage"], {})["action"] == "allow"

    def test_single_deny(self):
        results = [{"action": "deny", "reason": "blocked"}]
        ctrl = resolve_pre_tool_control(results, {})
        assert ctrl["action"] == "deny"
        assert ctrl["reason"] == "blocked"

    def test_deny_beats_modify(self):
        results = [
            {"action": "modify", "args": {"x": 1}},
            {"action": "deny", "reason": "no"},
        ]
        assert resolve_pre_tool_control(results, {})["action"] == "deny"

    def test_short_circuit_beats_deny(self):
        results = [
            {"action": "deny", "reason": "no"},
            {"action": "short_circuit", "result": "cached"},
        ]
        ctrl = resolve_pre_tool_control(results, {})
        assert ctrl["action"] == "short_circuit"
        assert ctrl["result"] == "cached"

    def test_ask_beats_modify(self):
        results = [
            {"action": "modify", "args": {"x": 1}},
            {"action": "ask", "reason": "confirm?"},
        ]
        assert resolve_pre_tool_control(results, {})["action"] == "ask"

    def test_modify_merges_args(self):
        results = [
            {"action": "modify", "args": {"a": 1}},
            {"action": "modify", "args": {"b": 2}},
        ]
        ctrl = resolve_pre_tool_control(results, {"orig": True})
        assert ctrl["action"] == "modify"
        assert ctrl["args"]["orig"] is True
        assert ctrl["args"]["a"] == 1
        assert ctrl["args"]["b"] == 2

    def test_allow_passthrough(self):
        results = [{"action": "allow"}]
        assert resolve_pre_tool_control(results, {})["action"] == "allow"

    def test_non_protocol_ignored(self):
        results = [None, "log line", {"not": "protocol"}, {"action": "allow"}]
        assert resolve_pre_tool_control(results, {})["action"] == "allow"


class TestResolvePostToolControl:
    def test_empty_results(self):
        assert resolve_post_tool_control([], "original") == "original"

    def test_modify_result(self):
        results = [{"action": "modify_result", "result": "modified"}]
        assert resolve_post_tool_control(results, "original") == "modified"

    def test_chain_modify_result(self):
        results = [
            {"action": "modify_result", "result": "first"},
            {"action": "modify_result", "result": "second"},
        ]
        assert resolve_post_tool_control(results, "original") == "second"

    def test_non_protocol_ignored(self):
        results = [None, {"action": "allow"}]
        assert resolve_post_tool_control(results, "original") == "original"
