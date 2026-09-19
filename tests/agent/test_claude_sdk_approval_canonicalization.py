"""SDK approval canonicalization hardening — claude-agent-sdk runtime tests (#25267).

Split from ``tests/agent/test_claude_sdk_runtime.py``; the SDK message
stand-ins, fake clients and shared builders live in
``tests.agent.claude_sdk_fakes``.
"""

import asyncio
import logging
import tracemalloc
from types import SimpleNamespace

import pytest

from tools import approval_context as approval_ctx
from tools import approval_gateway_wait
from tools import approval_sdk_gateway as sdk_gateway
from tools import approval_session_notify as session_notify
from tests.agent.claude_sdk_fakes import (
    _make_session,
    _plant_claude_agent_sdk_stand_in,
    isolate_provider_config,
)


@pytest.fixture(autouse=True)
def _isolate_provider_config(monkeypatch):
    """Provider config, gateway contextvars and the CLI approval callback are
    reset around every test in this module — carried explicitly, never hoisted
    to a conftest (see ``isolate_provider_config``)."""
    yield from isolate_provider_config(monkeypatch)


class TestSdkApprovalCanonicalizationHardening:
    """Hostile SDK permission data is bounded before every approval sink."""

    @pytest.fixture(autouse=True)
    def _sdk_permission_results(self, monkeypatch):
        _plant_claude_agent_sdk_stand_in(monkeypatch)

    @pytest.mark.parametrize(
        "tool_input_factory",
        [
            lambda: {"path": object()},
            lambda: {"path": type("StrSubclass", (str,), {})("/tmp/x")},
            lambda: {"path": "bad\ud800path"},
            lambda: {"score": float("nan")},
            lambda: {"nested": [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]},
            lambda: {"nodes": [None] * 10_001},
            lambda: {"path": "x" * (64 * 1024)},
        ],
    )
    def test_malformed_autoallowed_request_denies_before_callback(
        self, tool_input_factory,
    ):
        calls = []
        session, _ = _make_session(
            approval_callback=lambda *a, **k: calls.append((a, k)) or "once",
            permission_mode="default",
        )
        tool_input = tool_input_factory()
        if "nodes" not in tool_input:
            tool_input["cycle"] = []
            if type(tool_input.get("path")) is object:
                tool_input.pop("cycle")
        result = asyncio.run(session._make_can_use_tool()(
            "mcp__hermes-tools__read_file", tool_input, None,
        ))
        assert type(result).__name__ == "PermissionResultDeny"
        assert result.message == "canonical request is unassessable"
        assert calls == []

    def test_cycle_denies_before_exact_mcp_autoallow(self):
        calls = []
        cyclic = {}
        cyclic["self"] = cyclic
        session, _ = _make_session(
            approval_callback=lambda *a, **k: calls.append((a, k)) or "once",
            permission_mode="default",
        )
        result = asyncio.run(session._make_can_use_tool()(
            "mcp__hermes-tools__search_files", cyclic, None,
        ))
        assert type(result).__name__ == "PermissionResultDeny"
        assert calls == []

    def test_canonical_serialization_is_deterministic_utf8_and_bounded(self):
        from agent.transports.claude_agent_sdk_session_sanitize import (
            _canonical_sdk_tool_request,
            validate_canonical_sdk_request_serialization,
        )

        canonical = _canonical_sdk_tool_request(
            "Pathé工具", {"z": "🙂", "a": {"路径": "/tmp/é"}},
        )
        assert canonical == (
            '{"tool_input":{"a":{"路径":"/tmp/é"},"z":"🙂"},'
            '"tool_name":"Pathé工具"}'
        )
        assert validate_canonical_sdk_request_serialization(canonical) == (
            canonical,
            {"tool_input": {"a": {"路径": "/tmp/é"}, "z": "🙂"}, "tool_name": "Pathé工具"},
        )
        assert validate_canonical_sdk_request_serialization(canonical + " ") is None
        assert validate_canonical_sdk_request_serialization(
            '{"tool_input":{"path":"\\ud800"},"tool_name":"Read"}'
        ) is None

    def test_marker_aware_callback_receives_canonical_and_bounded_meaningful_id(self):
        received = []

        def marked(command, description, *, allow_permanent=False, tool_use_id="",
                   canonical_tool_input=None):
            received.append((command, description, allow_permanent, tool_use_id,
                             canonical_tool_input))
            return "once"

        marked._accepts_tool_use_id = True
        marked._accepts_canonical_tool_input = True
        session, _ = _make_session(
            approval_callback=marked, permission_mode="default",
        )
        result = asyncio.run(session._make_can_use_tool()(
            "Bash", {"command": "printf safe"}, SimpleNamespace(tool_use_id="toolu_é"),
        ))
        assert type(result).__name__ == "PermissionResultAllow"
        assert received == [(
            "Bash(command=printf safe)", "Claude requests SDK tool Bash", False,
            "toolu_é", '{"tool_input":{"command":"printf safe"},"tool_name":"Bash"}',
        )]

    @pytest.mark.parametrize(
        ("tool_use_id", "expected"),
        [
            ("x" * 256, "x" * 256),
            ("x" * 257, ""),
            ("é" * 128, "é" * 128),
            (("é" * 128) + "x", ""),
        ],
    )
    def test_sdk_session_tool_use_id_enforces_exact_utf8_byte_cap(
        self, tool_use_id, expected,
    ):
        received = []

        def marked(command, description, *, allow_permanent=False, tool_use_id=""):
            received.append(tool_use_id)
            return "deny"

        marked._accepts_tool_use_id = True
        session, _ = _make_session(approval_callback=marked, permission_mode="default")
        result = asyncio.run(session._make_can_use_tool()(
            "Bash", {"command": "true"}, SimpleNamespace(tool_use_id=tool_use_id),
        ))

        assert type(result).__name__ == "PermissionResultDeny"
        assert received == [expected]
        if not expected:
            assert tool_use_id not in received

    def test_sdk_session_huge_multibyte_tool_use_id_has_bounded_peak_allocation(self):
        huge_id = "é" * (16 * 1024 * 1024)
        received = []

        def marked(command, description, *, allow_permanent=False, tool_use_id=""):
            received.append(tool_use_id)
            return "deny"

        marked._accepts_tool_use_id = True
        session, _ = _make_session(approval_callback=marked, permission_mode="default")
        asyncio.run(session._make_can_use_tool()(
            "Bash", {"command": "true"}, SimpleNamespace(tool_use_id="warmup"),
        ))
        received.clear()
        tracemalloc.start()
        try:
            result = asyncio.run(session._make_can_use_tool()(
                "Bash", {"command": "true"}, SimpleNamespace(tool_use_id=huge_id),
            ))
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert type(result).__name__ == "PermissionResultDeny"
        assert received == [""]
        assert huge_id not in received
        assert peak < 2 * 1024 * 1024

    @pytest.mark.parametrize(
        "bad_id",
        [None, 123, " \t\n", "\x00\x01", "bad\ud800", "x" * 257,
         type("IdSubclass", (str,), {})("toolu_x")],
    )
    def test_opted_in_callback_receives_empty_malformed_tool_use_id(self, bad_id):
        received = []

        def marked(command, description, *, allow_permanent=False, tool_use_id=""):
            received.append(tool_use_id)
            return "deny"

        marked._accepts_tool_use_id = True
        session, _ = _make_session(approval_callback=marked, permission_mode="default")
        asyncio.run(session._make_can_use_tool()(
            "Bash", {"command": "true"}, SimpleNamespace(tool_use_id=bad_id),
        ))
        assert received == [""]

    def test_markerless_callback_keeps_exact_abi_and_safe_actionable_presentations(
        self, monkeypatch,
    ):
        from agent.transports import claude_agent_sdk_session_sanitize as sanitize_mod

        monkeypatch.setattr(
            sanitize_mod, "redact_sensitive_text",
            lambda value, **_kwargs: value.replace("SECRET", "[REDACTED]"),
        )
        received = []

        def legacy(command, description, *, allow_permanent=False):
            received.append((command, description, allow_permanent))
            return "once"

        session, _ = _make_session(approval_callback=legacy, permission_mode="default")
        callback = session._make_can_use_tool()
        for name, payload in (
            ("Bash", {"command": "printf SECRET\nnext\x00line"}),
            ("Write", {"file_path": "/tmp/demo\n.txt", "content": "SECRET_CONTENT"}),
            ("Odd SECRET\x00Tool", {"payload": "SECRET_PAYLOAD"}),
        ):
            result = asyncio.run(callback(name, payload, SimpleNamespace(tool_use_id="hostile")))
            assert type(result).__name__ == "PermissionResultAllow"

        assert received[0] == (
            "Bash(command=printf [REDACTED] next line)",
            "Claude requests SDK tool Bash", False,
        )
        assert received[1] == (
            "Write(path=/tmp/demo .txt)", "Claude requests SDK tool Write", False,
        )
        assert "CONTENT" not in received[1][0]
        assert received[2] == (
            "SDK tool unknown", "Claude requests an SDK tool", False,
        )
        assert "PAYLOAD" not in received[2][0]
        assert all(len(command.encode("utf-8")) <= 512 for command, _, _ in received)

    def test_malformed_bash_denies_before_markerless_callback(self):
        calls = []
        session, _ = _make_session(
            approval_callback=lambda *a, **k: calls.append((a, k)) or "once",
            permission_mode="default",
        )
        result = asyncio.run(session._make_can_use_tool()(
            "Bash", {"command": ["rm", "-rf", "/"]}, None,
        ))
        assert type(result).__name__ == "PermissionResultDeny"
        assert result.message == "canonical request is unassessable"
        assert calls == []

    def test_hostile_callback_exception_and_sdk_metadata_never_reach_logs(
        self, caplog,
    ):
        marker = "SDK_SECRET_EXCEPTION_91f"

        def broken(*_args, **_kwargs):
            raise RuntimeError(marker)

        session, _ = _make_session(
            approval_callback=broken, permission_mode="default",
            hermes_session_id="safe-session",
        )
        with caplog.at_level(
            logging.INFO, logger="agent.transports.claude_agent_sdk_session",
        ):
            result = asyncio.run(session._make_can_use_tool()(
                f"Odd-{marker}", {"payload": marker}, None,
            ))
        assert type(result).__name__ == "PermissionResultDeny"
        assert result.message == "approval callback failed"
        assert marker not in caplog.text
        assert not any(record.exc_info for record in caplog.records)
        assert any(
            record.getMessage() == "SDK approval callback failed at protected boundary"
            for record in caplog.records
        )

    def test_gateway_notify_exception_text_is_contained_at_sdk_boundary(
        self, monkeypatch, caplog,
    ):
        import json


        marker = "SDK_NOTIFY_EXCEPTION_SECRET_4ab"
        session_key = "sdk-notify-containment"
        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        token = approval_ctx.set_current_session_key(session_key)

        def broken_notify(_approval_data):
            raise RuntimeError(marker)

        try:
            session_notify.register_session_notify(session_key, broken_notify)
            callback = sdk_gateway.build_sdk_gateway_approval_callback()
            canonical = json.dumps(
                {"tool_name": "Bash", "tool_input": {"command": "true"}},
                sort_keys=True,
                separators=(",", ":"),
            )
            with caplog.at_level(logging.WARNING, logger="tools.approval"):
                result = callback(
                    "hostile", "hostile", canonical_tool_input=canonical,
                )
            assert result == {
                "choice": "deny",
                "reason": (
                    "approval request could not be delivered to the operator "
                    "(notify failed)"
                ),
            }
            assert marker not in caplog.text
        finally:
            session_notify.unregister_session_notify(session_key)
            approval_ctx.reset_current_session_key(token)

    def test_wide_request_rejects_without_width_sized_validator_allocation(self):
        import tracemalloc

        from agent.transports.claude_agent_sdk_session_sanitize import _is_bounded_plain_sdk_json

        request = {"wide": [None] * 200_000}
        tracemalloc.start()
        try:
            assert not _is_bounded_plain_sdk_json(request)
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert peak < 2_000_000

    def test_wide_dictionary_rejection_time_is_node_budget_bounded(self):
        import time

        from agent.transports.claude_agent_sdk_session_sanitize import _is_bounded_plain_sdk_json

        smaller = {str(index): None for index in range(50_000)}
        larger = {str(index): None for index in range(500_000)}

        def fastest(value):
            samples = []
            for _ in range(3):
                started = time.perf_counter()
                assert not _is_bounded_plain_sdk_json(value)
                samples.append(time.perf_counter() - started)
            return min(samples)

        smaller_time = fastest(smaller)
        larger_time = fastest(larger)
        assert larger_time < (smaller_time * 3) + 0.01

    def test_canonicalization_runtime_error_denies_before_auto_allow(
        self, monkeypatch,
    ):
        from agent.transports import claude_agent_sdk_session_permissions as perm_mod

        def mutation_failure(*_args, **_kwargs):
            raise RuntimeError("dictionary changed size during iteration")

        monkeypatch.setattr(
            perm_mod, "_canonical_sdk_tool_request", mutation_failure,
        )
        calls = []
        session, _ = _make_session(
            approval_callback=lambda *a, **k: calls.append((a, k)) or "once",
            permission_mode="default",
        )
        result = asyncio.run(session._make_can_use_tool()(
            "mcp__hermes-tools__read_file", {"path": "/tmp/demo"}, None,
        ))
        assert type(result).__name__ == "PermissionResultDeny"
        assert result.message == "canonical request is unassessable"
        assert calls == []

    def test_wide_exact_callback_result_rejects_before_width_sized_allocation(
        self, caplog,
    ):
        import tracemalloc

        marker = "SDK_WIDE_RESULT_SECRET_91c"
        wide_result = {f"key-{index}": None for index in range(200_000)}
        wide_result[marker] = None
        session, _ = _make_session(
            approval_callback=lambda *_a, **_k: wide_result,
            permission_mode="default",
        )
        callback = session._make_can_use_tool()
        asyncio.run(callback("Bash", {"command": "true"}, None))

        caplog.clear()
        tracemalloc.start()
        try:
            with caplog.at_level(
                logging.INFO, logger="agent.transports.claude_agent_sdk_session",
            ):
                result = asyncio.run(callback("Bash", {"command": "true"}, None))
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert type(result).__name__ == "PermissionResultDeny"
        assert result.message == "approval callback failed"
        assert peak < 1_000_000
        assert marker not in caplog.text
        assert not any(record.exc_info for record in caplog.records)

    def test_overlong_multibyte_choice_rejects_without_full_copy_or_leak(
        self, caplog, monkeypatch,
    ):
        import tracemalloc

        from agent.transports import claude_agent_sdk_session_permissions as perm_mod
        from agent.transports import claude_agent_sdk_session_sanitize as sanitize_mod

        marker = "SDK_CHOICE_RESULT_SECRET_27e"
        huge_choice = marker + ("é" * (32 * 1024 * 1024))
        current_result = ["bogus"]
        session, _ = _make_session(
            approval_callback=lambda *_a, **_k: current_result[0],
            permission_mode="default",
        )
        callback = session._make_can_use_tool()
        asyncio.run(callback("Bash", {"command": "true"}, None))
        current_result[0] = huge_choice
        category_calls = []
        hostile_category_counts = []
        original_category = sanitize_mod.unicodedata.category
        original_validator = perm_mod._is_bounded_sdk_callback_string

        def counted_category(char):
            category_calls.append(char)
            return original_category(char)

        def counted_validator(value, max_utf8_bytes, *, allow_space):
            before = len(category_calls)
            valid = original_validator(
                value, max_utf8_bytes, allow_space=allow_space,
            )
            if value is huge_choice:
                hostile_category_counts.append(len(category_calls) - before)
            return valid

        monkeypatch.setattr(sanitize_mod.unicodedata, "category", counted_category)
        monkeypatch.setattr(
            perm_mod, "_is_bounded_sdk_callback_string", counted_validator,
        )

        caplog.clear()
        tracemalloc.start()
        try:
            with caplog.at_level(
                logging.INFO, logger="agent.transports.claude_agent_sdk_session",
            ):
                result = asyncio.run(callback("Bash", {"command": "true"}, None))
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert type(result).__name__ == "PermissionResultDeny"
        assert result.message == "approval callback failed"
        assert peak < 1_000_000
        assert hostile_category_counts == [17]
        assert marker not in caplog.text
        assert not any(record.exc_info for record in caplog.records)

    def test_overlong_multibyte_reason_rejects_without_full_copy_or_leak(
        self, caplog, monkeypatch,
    ):
        import tracemalloc

        from agent.transports import claude_agent_sdk_session_permissions as perm_mod
        from agent.transports import claude_agent_sdk_session_sanitize as sanitize_mod

        marker = "SDK_REASON_RESULT_SECRET_4af"
        huge_reason = marker + ("é" * (16 * 1024 * 1024))
        current_result = [{"choice": "deny", "reason": "ordinary denial"}]
        session, _ = _make_session(
            approval_callback=lambda *_a, **_k: current_result[0],
            permission_mode="default",
        )
        callback = session._make_can_use_tool()
        asyncio.run(callback("Bash", {"command": "true"}, None))
        current_result[0] = {"choice": "deny", "reason": huge_reason}
        category_calls = []
        hostile_category_counts = []
        original_category = sanitize_mod.unicodedata.category
        original_validator = perm_mod._is_bounded_sdk_callback_string

        def counted_category(char):
            category_calls.append(char)
            return original_category(char)

        def counted_validator(value, max_utf8_bytes, *, allow_space):
            before = len(category_calls)
            valid = original_validator(
                value, max_utf8_bytes, allow_space=allow_space,
            )
            if value is huge_reason:
                hostile_category_counts.append(len(category_calls) - before)
            return valid

        monkeypatch.setattr(sanitize_mod.unicodedata, "category", counted_category)
        monkeypatch.setattr(
            perm_mod, "_is_bounded_sdk_callback_string", counted_validator,
        )

        caplog.clear()
        tracemalloc.start()
        try:
            with caplog.at_level(
                logging.INFO, logger="agent.transports.claude_agent_sdk_session",
            ):
                result = asyncio.run(callback("Bash", {"command": "true"}, None))
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert type(result).__name__ == "PermissionResultDeny"
        assert result.message == "approval callback failed"
        assert peak < 1_000_000
        assert hostile_category_counts == [271]
        assert marker not in caplog.text
        assert not any(record.exc_info for record in caplog.records)

    def test_hostile_callback_result_mapping_fails_closed_without_leak(self, caplog):
        marker = "SDK_RESULT_DECODE_SECRET_7d2"
        override_calls = []

        class HostileResult(dict):
            def __len__(self):
                override_calls.append("len")
                raise RuntimeError(marker)

            def __iter__(self):
                override_calls.append("iter")
                raise RuntimeError(marker)

            def __contains__(self, _key):
                override_calls.append("contains")
                raise RuntimeError(marker)

            def __getitem__(self, _key):
                override_calls.append("getitem")
                raise RuntimeError(marker)

            def get(self, *_args, **_kwargs):
                override_calls.append("get")
                raise RuntimeError(marker)

        session, _ = _make_session(
            approval_callback=lambda *_a, **_k: HostileResult(choice="once"),
            permission_mode="default",
        )
        with caplog.at_level(
            logging.INFO, logger="agent.transports.claude_agent_sdk_session",
        ):
            result = asyncio.run(session._make_can_use_tool()(
                "Bash", {"command": "true"}, None,
            ))
        assert type(result).__name__ == "PermissionResultDeny"
        assert result.message == "approval callback failed"
        assert override_calls == []
        assert marker not in caplog.text
        assert not any(record.exc_info for record in caplog.records)

    @pytest.mark.parametrize("callback_result", [
        "bogus",
        {"choice": "bogus"},
    ])
    def test_unknown_exact_callback_choice_is_protocol_failure(
        self, callback_result, caplog,
    ):
        session, _ = _make_session(
            approval_callback=lambda *_a, **_k: callback_result,
            permission_mode="default",
        )
        with caplog.at_level(
            logging.INFO, logger="agent.transports.claude_agent_sdk_session",
        ):
            result = asyncio.run(session._make_can_use_tool()(
                "Bash", {"command": "true"}, None,
            ))
        assert type(result).__name__ == "PermissionResultDeny"
        assert result.message == "approval callback failed"
        assert "denied by user" not in caplog.text

    def test_callback_string_helper_enforces_exact_utf8_caps_and_text_contract(self):
        from agent.transports.claude_agent_sdk_session_sanitize import (
            _is_bounded_sdk_callback_string,
        )

        class StringSubclass(str):
            pass

        assert _is_bounded_sdk_callback_string("é" * 256, 512, allow_space=True)
        assert not _is_bounded_sdk_callback_string("é" * 257, 512, allow_space=True)
        assert _is_bounded_sdk_callback_string("ordinary reason", 512, allow_space=True)
        assert not _is_bounded_sdk_callback_string("line\nbreak", 512, allow_space=True)
        assert not _is_bounded_sdk_callback_string("tab\tbreak", 512, allow_space=True)
        assert not _is_bounded_sdk_callback_string("bad\ud800reason", 512, allow_space=True)
        assert not _is_bounded_sdk_callback_string(
            StringSubclass("once"), 16, allow_space=False,
        )

    @pytest.mark.parametrize("budget", [0, 1, 5, 20, 21])
    def test_head_tail_helper_never_exceeds_supplied_budget(self, budget):
        from agent.transports.claude_agent_sdk_session_sanitize import (
            _bounded_control_sanitized_head_tail,
        )

        rendered = _bounded_control_sanitized_head_tail("x" * 100, budget)
        assert len(rendered.encode("utf-8")) <= budget

    def test_head_tail_helper_preserves_multibyte_tail_within_budget(self):
        from agent.transports.claude_agent_sdk_session_sanitize import (
            _bounded_control_sanitized_head_tail,
        )

        rendered = _bounded_control_sanitized_head_tail(
            ("α" * 100) + "\nTAIL", 64,
        )
        assert "[truncated]" in rendered
        assert rendered.endswith("TAIL")
        assert "\n" not in rendered
        assert len(rendered.encode("utf-8")) <= 64

    def test_huge_integer_denies_before_encoder_entry_on_exact_mcp_path(
        self, monkeypatch,
    ):
        import json
        import sys


        old_limit = sys.get_int_max_str_digits()
        calls = []
        original = json.JSONEncoder.iterencode

        def observed_iterencode(encoder, value, *args, **kwargs):
            calls.append(value)
            return original(encoder, value, *args, **kwargs)

        monkeypatch.setattr(json.JSONEncoder, "iterencode", observed_iterencode)
        callback_calls = []
        session, _ = _make_session(
            approval_callback=lambda *a, **k: callback_calls.append((a, k)) or "once",
            permission_mode="default",
        )
        try:
            sys.set_int_max_str_digits(0)
            huge = 10 ** 99_999
            result = asyncio.run(session._make_can_use_tool()(
                "mcp__hermes-tools__read_file",
                {"path": "/tmp/x", "huge": huge},
                None,
            ))
        finally:
            sys.set_int_max_str_digits(old_limit)

        assert type(result).__name__ == "PermissionResultDeny"
        assert result.message == "canonical request is unassessable"
        assert callback_calls == []
        assert calls == []

    def test_post_validation_mutation_cannot_reach_encoder_or_allocate_width(
        self, monkeypatch,
    ):
        import tracemalloc

        from agent.transports import claude_agent_sdk_session_sanitize as sanitize_mod

        payload = {"path": "/tmp/x", "nested": {"safe": True}}
        wide = {f"k{i}": i for i in range(100_000)}
        encoded_values = []
        original = sanitize_mod._bounded_canonical_sdk_json

        def mutate_before_encoding(value):
            payload["nested"] = wide
            encoded_values.append(value)
            return original(value)

        monkeypatch.setattr(
            sanitize_mod, "_bounded_canonical_sdk_json", mutate_before_encoding,
        )
        callback_calls = []
        session, _ = _make_session(
            approval_callback=lambda *a, **k: callback_calls.append((a, k)) or "once",
            permission_mode="default",
        )
        tracemalloc.start()
        try:
            result = asyncio.run(session._make_can_use_tool()(
                "mcp__hermes-tools__read_file", payload, None,
            ))
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert type(result).__name__ == "PermissionResultAllow"
        assert callback_calls == []
        assert encoded_values
        assert all(
            value["tool_input"].get("nested") is not wide
            for value in encoded_values
            if type(value) is dict and type(value.get("tool_input")) is dict
        )
        assert peak < 2_000_000

    def test_alias_serialization_stops_near_canonical_byte_cap(self):
        import tracemalloc

        from agent.transports.claude_agent_sdk_session_sanitize import _canonical_sdk_tool_request

        big = 10 ** 3_999
        payload = {"path": "/tmp/x", "aliases": [big] * 9_000}
        tracemalloc.start()
        try:
            assert _canonical_sdk_tool_request(
                "mcp__hermes-tools__read_file", payload,
            ) is None
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert peak < 2_000_000

    @pytest.mark.parametrize(("callback_result", "expected_type", "expected_message"), [
        ("once", "PermissionResultAllow", None),
        ("session", "PermissionResultAllow", None),
        ("always", "PermissionResultAllow", None),
        ({"choice": "once"}, "PermissionResultAllow", None),
        ("deny", "PermissionResultDeny", "approval denied by callback"),
        ("timeout", "PermissionResultDeny", "approval timed out — no operator response"),
        (
            {"choice": "deny", "reason": "approval expired (turn ended)"},
            "PermissionResultDeny",
            "approval expired (turn ended)",
        ),
    ])
    def test_callback_result_valid_shape_matrix_preserves_decision_and_input(
        self, callback_result, expected_type, expected_message,
    ):
        original = {"command": "printf safe", "nested": {"value": 1}}
        session, _ = _make_session(
            approval_callback=lambda *_a, **_k: callback_result,
            permission_mode="default",
        )
        result = asyncio.run(session._make_can_use_tool()(
            "Bash", original, None,
        ))

        assert type(result).__name__ == expected_type
        if expected_type == "PermissionResultAllow":
            assert result.updated_input == original
            assert result.updated_input is not original
            assert result.updated_input["nested"] is not original["nested"]
        else:
            assert result.message == expected_message

    def test_trusted_structural_operator_denial_preserves_reason_and_provenance(self):

        def trusted_callback(*_args, **_kwargs):
            return {
                "choice": "deny",
                "operator_denial": True,
                "reason": "not now",
            }

        assert sdk_gateway._register_trusted_sdk_gateway_approval_callback(
            trusted_callback,
        )
        session, _ = _make_session(
            approval_callback=trusted_callback,
            permission_mode="default",
        )
        result = asyncio.run(session._make_can_use_tool()(
            "Bash", {"command": "true"}, None,
        ))

        assert type(result).__name__ == "PermissionResultDeny"
        assert result.message == "denied by user: not now"

    @pytest.mark.parametrize("callback_result", [
        {"choice": "once", "extra": "malformed"},
        {"choice": "once", "reason": "unused"},
        {"choice": "timeout", "reason": "raw"},
        {"choice": "deny", "extra": "malformed"},
        {"choice": True},
        {"choice": "deny", "reason": True},
        {"choice": "deny", "operator_denial": False, "reason": ""},
        {"choice": "deny", "operator_denial": 1, "reason": ""},
        {"choice": "deny", "operator_denial": True, "reason": ""},
        {"choice": "deny", "reason": "line\nbreak"},
        {"choice": "deny", "reason": "bad\ud800reason"},
    ])
    def test_malformed_structured_callback_result_is_protocol_failure(
        self, callback_result,
    ):
        session, _ = _make_session(
            approval_callback=lambda *_a, **_k: callback_result,
            permission_mode="default",
        )
        result = asyncio.run(session._make_can_use_tool()(
            "Bash", {"command": "true"}, None,
        ))
        assert type(result).__name__ == "PermissionResultDeny"
        assert result.message == "approval callback failed"


    def test_canonical_utf8_cap_and_cap_plus_one_are_exact(self):
        from agent.transports.claude_agent_sdk_session_sanitize import (
            _SDK_CANONICAL_MAX_UTF8_BYTES,
            validate_canonical_sdk_request_serialization,
        )

        prefix = '{"tool_input":{"path":"'
        suffix = '"},"tool_name":"Read"}'
        envelope_bytes = len((prefix + suffix).encode("utf-8"))
        fill = "é" * ((_SDK_CANONICAL_MAX_UTF8_BYTES - envelope_bytes) // 2)
        at_cap = prefix + fill + "a" + suffix
        cap_plus_one = prefix + fill + "aa" + suffix

        assert len(at_cap.encode("utf-8")) == _SDK_CANONICAL_MAX_UTF8_BYTES
        assert len(cap_plus_one.encode("utf-8")) == _SDK_CANONICAL_MAX_UTF8_BYTES + 1
        assert validate_canonical_sdk_request_serialization(at_cap) is not None
        assert validate_canonical_sdk_request_serialization(cap_plus_one) is None

    def test_oversized_multibyte_canonical_rejects_with_bounded_peak_allocation(
        self,
    ):
        import time
        import tracemalloc

        from agent.transports.claude_agent_sdk_session_sanitize import (
            validate_canonical_sdk_request_serialization,
        )

        huge = "é" * (16 * 1024 * 1024)
        tracemalloc.start()
        started = time.perf_counter()
        try:
            assert validate_canonical_sdk_request_serialization(huge) is None
            elapsed = time.perf_counter() - started
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert peak < 2_000_000
        assert elapsed < 1.0

    def test_gateway_validates_canonical_once_and_bounds_oversized_rejection(
        self, monkeypatch,
    ):
        import json
        import time
        import tracemalloc

        from agent.transports import claude_agent_sdk_session as session_mod
        from tools import approval as approval_mod

        sk = "sess-sdk-single-validation"
        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        token = approval_ctx.set_current_session_key(sk)
        calls = []
        original = session_mod.validate_canonical_sdk_request_serialization

        def counted(value):
            calls.append(value)
            return original(value)

        try:
            approval_mod.register_gateway_notify(
                sk,
                lambda _data: approval_mod.resolve_gateway_approval(sk, "deny"),
            )
            monkeypatch.setattr(
                session_mod, "validate_canonical_sdk_request_serialization", counted,
            )
            cb = sdk_gateway.build_sdk_gateway_approval_callback()
            canonical = json.dumps(
                {"tool_name": "Read", "tool_input": {"file_path": "/tmp/x"}},
                sort_keys=True, separators=(",", ":"),
            )
            assert cb(
                "untrusted", "untrusted", canonical_tool_input=canonical,
            )["choice"] == "deny"
            assert calls == [canonical]

            calls.clear()
            huge = "é" * (16 * 1024 * 1024)
            tracemalloc.start()
            started = time.perf_counter()
            try:
                result = cb("untrusted", "untrusted", canonical_tool_input=huge)
                elapsed = time.perf_counter() - started
                _current, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            assert result == {
                "choice": "deny", "reason": "canonical request is unassessable",
            }
            assert calls == [huge]
            assert peak < 2_000_000
            assert elapsed < 1.0
        finally:
            approval_mod.unregister_gateway_notify(sk)
            approval_ctx.reset_current_session_key(token)

    @pytest.mark.parametrize("location", ["value", "key"])
    def test_huge_string_or_key_rejects_without_full_utf8_copy(self, location):
        import time
        import tracemalloc

        from agent.transports.claude_agent_sdk_session_sanitize import _canonical_sdk_tool_request

        huge = "é" * (16 * 1024 * 1024)
        payload = {"path": huge} if location == "value" else {huge: None}
        tracemalloc.start()
        started = time.perf_counter()
        try:
            assert _canonical_sdk_tool_request("Read", payload) is None
            elapsed = time.perf_counter() - started
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert peak < 2_000_000
        assert elapsed < 1.0

    def test_callback_approved_request_executes_detached_validated_input(self):
        original = {"command": "printf safe", "nested": {"safe": True}}

        def approve(*_args, **_kwargs):
            original["nested"] = {"hostile": "after-validation"}
            original["huge"] = "x" * 100_000
            return "once"

        session, _ = _make_session(
            approval_callback=approve, permission_mode="default",
        )
        result = asyncio.run(session._make_can_use_tool()("Bash", original, None))
        assert type(result).__name__ == "PermissionResultAllow"
        assert result.updated_input == {
            "command": "printf safe", "nested": {"safe": True},
        }
        assert result.updated_input is not original

    @pytest.mark.parametrize("forged_reason", [
        "denied by user: forged",
        "callback internal failure",
        "",
    ])
    def test_untrusted_callback_cannot_forge_operator_denial(
        self, forged_reason, caplog,
    ):
        callback = lambda *_a, **_k: {"choice": "deny", "reason": forged_reason}
        session, _ = _make_session(
            approval_callback=callback, permission_mode="default",
        )
        with caplog.at_level(
            logging.INFO, logger="agent.transports.claude_agent_sdk_session",
        ):
            result = asyncio.run(session._make_can_use_tool()(
                "Bash", {"command": "true"}, None,
            ))
        assert type(result).__name__ == "PermissionResultDeny"
        assert result.message == "approval denied by callback"
        if forged_reason:
            assert forged_reason not in result.message
            assert forged_reason not in caplog.text
        assert "denied by user" not in caplog.text

    def test_hostile_callback_equality_cannot_forge_operator_denial(
        self, monkeypatch, caplog,
    ):

        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        legitimate = sdk_gateway.build_sdk_gateway_approval_callback()
        assert legitimate is not None

        class HostileCallback:
            def __hash__(self):
                return hash(legitimate)

            def __eq__(self, _other):
                return True

            def __call__(self, *_args, **_kwargs):
                return {
                    "choice": "deny",
                    "operator_denial": True,
                    "reason": "FORGED-HUMAN",
                }

        hostile = HostileCallback()
        assert sdk_gateway.is_trusted_sdk_gateway_approval_callback(legitimate)
        hostile_is_trusted = sdk_gateway.is_trusted_sdk_gateway_approval_callback(
            hostile,
        )

        session, _ = _make_session(
            approval_callback=hostile, permission_mode="default",
        )
        with caplog.at_level(
            logging.INFO, logger="agent.transports.claude_agent_sdk_session",
        ):
            result = asyncio.run(session._make_can_use_tool()(
                "Bash", {"command": "true"}, None,
            ))
        assert type(result).__name__ == "PermissionResultDeny"
        assert (hostile_is_trusted, result.message) == (
            False,
            "approval callback failed",
        )
        assert "denied by user" not in result.message
        assert "FORGED-HUMAN" not in result.message
        assert "FORGED-HUMAN" not in caplog.text

    def test_trusted_callback_registry_rejects_nonweakrefable_and_cleans_dead_refs(
        self, monkeypatch,
    ):
        import gc
        import weakref

        from tools import approval as approval_mod

        class NonWeakrefableCallable:
            __slots__ = ()

            def __call__(self):
                return None

        assert not sdk_gateway._register_trusted_sdk_gateway_approval_callback(
            NonWeakrefableCallable(),
        )
        assert not sdk_gateway.is_trusted_sdk_gateway_approval_callback(None)
        assert not sdk_gateway.is_trusted_sdk_gateway_approval_callback(object())

        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        sdk_gateway.is_trusted_sdk_gateway_approval_callback(None)
        with approval_mod._lock:
            baseline = len(sdk_gateway._trusted_sdk_gateway_approval_callbacks)
        callback = sdk_gateway.build_sdk_gateway_approval_callback()
        callback_ref = weakref.ref(callback)
        with approval_mod._lock:
            assert len(sdk_gateway._trusted_sdk_gateway_approval_callbacks) == baseline + 1
            registered_ref = sdk_gateway._trusted_sdk_gateway_approval_callbacks[-1]
            assert registered_ref() is callback
        del callback
        gc.collect()
        assert callback_ref() is None
        with approval_mod._lock:
            assert all(
                candidate_ref is not registered_ref
                for candidate_ref in sdk_gateway._trusted_sdk_gateway_approval_callbacks
            )

    def test_trusted_callback_registry_concurrent_lookup_is_identity_exact(
        self, monkeypatch,
    ):
        from concurrent.futures import ThreadPoolExecutor


        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        legitimate = sdk_gateway.build_sdk_gateway_approval_callback()

        class EqualProxy:
            def __hash__(self):
                return hash(legitimate)

            def __eq__(self, _other):
                return True

            def __call__(self):
                return None

        proxy = EqualProxy()
        callbacks = [legitimate, proxy, None, object()] * 64
        with ThreadPoolExecutor(max_workers=16) as executor:
            results = list(executor.map(
                sdk_gateway.is_trusted_sdk_gateway_approval_callback,
                callbacks,
            ))
        assert results == [True, False, False, False] * 64

    def test_trusted_gateway_operator_denial_is_structural(self, monkeypatch):
        from tools import approval as approval_mod

        sk = "sess-structural-operator-deny"
        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        token = approval_ctx.set_current_session_key(sk)
        try:
            approval_mod.register_gateway_notify(sk, lambda _data: None)
            monkeypatch.setattr(
                approval_gateway_wait,
                "_await_gateway_decision",
                lambda *_a, **_k: {
                    "resolved": True, "choice": "deny", "reason": "operator note",
                },
            )
            callback = sdk_gateway.build_sdk_gateway_approval_callback()
            canonical = (
                '{"tool_input":{"command":"true"},"tool_name":"Bash"}'
            )
            assert callback(
                "untrusted", "untrusted", canonical_tool_input=canonical,
            ) == {
                "choice": "deny",
                "operator_denial": True,
                "reason": "operator note",
            }
            session, _ = _make_session(
                approval_callback=callback, permission_mode="default",
            )
            result = asyncio.run(session._make_can_use_tool()(
                "Bash", {"command": "true"}, None,
            ))
            assert result.message == "denied by user: operator note"
        finally:
            approval_mod.unregister_gateway_notify(sk)
            approval_ctx.reset_current_session_key(token)

    def test_redactor_exception_is_fixed_fail_closed(self, monkeypatch, caplog):
        from agent.transports import claude_agent_sdk_session_sanitize as sanitize_mod

        marker = "REDACTOR_EXCEPTION_SECRET_a61"
        monkeypatch.setattr(
            sanitize_mod,
            "redact_sensitive_text",
            lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError(marker)),
        )
        callback_calls = []
        session, _ = _make_session(
            approval_callback=lambda *a, **k: callback_calls.append((a, k)) or "once",
            permission_mode="default",
        )
        with caplog.at_level(
            logging.WARNING, logger="agent.transports.claude_agent_sdk_session",
        ):
            result = asyncio.run(session._make_can_use_tool()(
                "Read", {"file_path": "/tmp/x"}, None,
            ))
        assert type(result).__name__ == "PermissionResultDeny"
        assert result.message == "canonical request is unassessable"
        assert callback_calls == []
        assert marker not in caplog.text
