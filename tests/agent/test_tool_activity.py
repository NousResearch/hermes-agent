"""Contracts for display-safe tool reasoning and deterministic summaries."""

from __future__ import annotations

import copy
import json


def _schema(name="terminal", properties=None, required=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {
                "type": "object",
                "properties": properties or {"command": {"type": "string"}},
                "required": required or ["command"],
            },
        },
    }


def test_augment_adds_required_activity_reason_first_without_mutating_source():
    from agent.tool_activity import augment_tool_schemas

    source = [_schema()]
    original = copy.deepcopy(source)

    augmented = augment_tool_schemas(source, enabled=True)

    params = augmented[0]["function"]["parameters"]
    assert list(params["properties"])[:1] == ["activity_reason"]
    assert params["properties"]["activity_reason"]["type"] == "string"
    assert params["required"][:1] == ["activity_reason"]
    assert source == original


def test_augment_skips_preexisting_activity_reason_and_preserves_contract(caplog):
    from agent.tool_activity import augment_tool_schemas

    string_reasoning = _schema(
        properties={"activity_reason": {"type": "string", "description": "Business rationale"}, "path": {"type": "string"}},
        required=["activity_reason"],
    )
    object_reasoning = _schema("other", properties={"activity_reason": {"type": "object"}})
    original = copy.deepcopy([string_reasoning, object_reasoning])
    activity_names: set[str] = set()

    augmented = augment_tool_schemas(
        [string_reasoning, object_reasoning],
        enabled=True,
        activity_tool_names=activity_names,
    )

    assert augmented == original
    assert activity_names == set()
    assert "pre-existing activity_reason property" in caplog.text


def test_augment_skips_tool_with_native_reasoning_argument():
    from agent.tool_activity import augment_tool_schemas

    source = [_schema(
        properties={
            "reasoning": {"type": "string", "description": "Business rationale"},
            "path": {"type": "string"},
        },
        required=["reasoning", "path"],
    )]
    original = copy.deepcopy(source)
    activity_names: set[str] = set()

    augmented = augment_tool_schemas(
        source,
        enabled=True,
        activity_tool_names=activity_names,
    )

    assert augmented == original
    assert activity_names == set()


def test_extract_reasoning_preserves_legitimate_tool_argument_when_not_augmented():
    from agent.tool_activity import extract_tool_reasoning

    args = {"reasoning": "Business rationale", "path": "/tmp/file"}

    assert extract_tool_reasoning(args, enabled=False) is None
    assert args == {"reasoning": "Business rationale", "path": "/tmp/file"}


def test_augment_leaves_malformed_required_schema_entirely_unchanged(caplog):
    from agent.tool_activity import augment_tool_schemas

    source = _schema()
    source["function"]["parameters"]["required"] = "command"

    augmented = augment_tool_schemas([source], enabled=True)

    assert augmented[0] == source
    assert "malformed required list" in caplog.text


def test_extract_reasoning_strips_and_sanitizes_to_one_short_line():
    from agent.tool_activity import extract_tool_reasoning

    args = {"reasoning": "  Check status\nthen report.  ", "command": "git status"}

    reason = extract_tool_reasoning(args)

    assert reason == "Check status then report"
    assert args == {"command": "git status"}


def test_extract_activity_reason_preserves_native_reasoning_argument():
    from agent.tool_activity import extract_tool_reasoning

    args = {
        "activity_reason": "  Inspect child metadata.  ",
        "reasoning": "Business rationale",
        "path": "/tmp/file",
    }

    reason = extract_tool_reasoning(args)

    assert reason == "Inspect child metadata"
    assert args == {"reasoning": "Business rationale", "path": "/tmp/file"}


def test_summarize_tool_result_is_deterministic_and_preserves_raw_result():
    from agent.tool_activity import summarize_tool_result

    raw = json.dumps({"content": "LINE_NUM|one\nLINE_NUM|two\n", "total_lines": 2})
    summary = summarize_tool_result("read_file", {"path": "private.txt"}, raw, duration_s=0.125)

    assert summary == "read_file: 2 lines"
    assert raw == json.dumps({"content": "LINE_NUM|one\nLINE_NUM|two\n", "total_lines": 2})


def test_summarize_read_uses_structured_range_total_and_next_offset():
    from agent.tool_activity import summarize_tool_result

    raw = json.dumps({
        "content": "158|one\n159|two\n160|three",
        "total_lines": 323,
        "truncated": True,
        "next_offset": 161,
    })

    assert summarize_tool_result(
        "read_file", {"offset": 158, "limit": 3}, raw
    ) == "read_file: 3 lines (158–160 of 323; next 161)"


def test_summarize_search_uses_match_and_file_counts_without_content():
    from agent.tool_activity import summarize_tool_result

    raw = json.dumps({
        "total_count": 3,
        "matches_format": "path-grouped",
        "matches_text": (
            "/private/alpha.py\n"
            "  10: SECRET_ALPHA\n"
            "  20: SECRET_BETA\n"
            "/private/beta.py\n"
            "  30: SECRET_GAMMA"
        ),
    })

    summary = summarize_tool_result("search_files", {"target": "content"}, raw)

    assert summary == "search_files: 3 matches in 2 files"
    assert "private" not in summary
    assert "SECRET" not in summary


def test_summarize_truncated_search_parses_json_before_safe_hint():
    from agent.tool_activity import summarize_tool_result

    raw = (
        json.dumps({
            "total_count": 11,
            "matches": [{"path": "/private/only.py", "line": 4, "content": "SECRET"}],
            "truncated": True,
        })
        + "\n\n[Hint: Results truncated. Use offset=5 to see more.]"
    )

    assert summarize_tool_result(
        "search_files", {"target": "content"}, raw
    ) == "search_files: 11 matches in 1 file (truncated)"


def test_summarize_file_name_search_uses_file_semantics():
    from agent.tool_activity import summarize_tool_result

    raw = json.dumps({"total_count": 2, "files": ["/secret/a.py", "/secret/b.py"]})

    assert summarize_tool_result(
        "search_files", {"target": "files"}, raw
    ) == "search_files: 2 files"


def test_summarize_patch_uses_structured_diff_and_changed_file_count():
    from agent.tool_activity import summarize_tool_result

    raw = json.dumps({
        "success": True,
        "diff": (
            "--- a/secret-a.py\n+++ b/secret-a.py\n@@\n-old\n+new\n+extra\n"
            "--- a/secret-b.py\n+++ b/secret-b.py\n@@\n-old2\n+new2\n"
        ),
        "files_modified": ["/private/secret-a.py", "/private/secret-b.py"],
    })

    summary = summarize_tool_result("patch", {}, raw)

    assert summary == "patch: +3/-2 across 2 files"
    assert "secret" not in summary


def test_summarize_apply_patch_counts_successful_request_without_echoing_it():
    from agent.tool_activity import summarize_tool_result

    patch_text = "*" * 3 + " Update File: private.py\n@@\n-old\n+new\n" + "*" * 3 + " End Patch"
    summary = summarize_tool_result(
        "apply_patch", {"patch": patch_text}, "Done!", status="completed"
    )

    assert summary == "apply_patch: +1/-1 in 1 file"
    assert "private.py" not in summary



def test_summarize_errors_uses_redacted_first_line_and_caps_output():
    from agent.tool_activity import summarize_tool_result

    summary = summarize_tool_result(
        "terminal",
        {"command": "secret"},
        '{"error": "failed TOKEN=abcdefghijklmnopqrstuvwxyz0123456789\\nsecond"}',
        duration_s=1.25,
    )

    assert summary.startswith("terminal: error: failed")
    assert "abcdefghijklmnopqrstuvwxyz0123456789" not in summary
    assert "second" not in summary


def test_activity_text_redacts_url_credentials_query_tokens_and_headers():
    from agent.tool_activity import extract_tool_reasoning, summarize_tool_result

    args = {
        "reasoning": (
            "Use https://alice:s3cr3t@example.com/x?token=abc123 "
            "Authorization: Bearer topsecret"
        )
    }
    reason = extract_tool_reasoning(args)
    summary = summarize_tool_result(
        "terminal",
        {},
        '{"error":"failed https://alice:s3cr3t@example.com/x?X-Amz-Signature=deadbeef&safe=yes Cookie: sid=secret"}',
        status="failed",
    )

    for rendered in (reason or "", summary):
        assert "alice" not in rendered
        assert "s3cr3t" not in rendered
        assert "abc123" not in rendered
        assert "topsecret" not in rendered
        assert "deadbeef" not in rendered
        assert "sid=secret" not in rendered
    assert "safe=yes" in summary


def test_activity_text_redacts_non_http_url_credentials_and_query_aliases():
    from agent.tool_activity import redact_activity_text

    rendered = redact_activity_text(
        "wss://alice:s3cr3t@example.test/socket?token=abc "
        "ftp://bob:password@example.test/file?accessKey=key123&sig=deadbeef&safe=yes "
        "https://example.test/x?access%4Bey=encoded-secret "
        "https://host.test/x?token%3Dsupersecret "
        "//carol:network-secret@example.test/x?token=network-query"
    )

    for secret in (
        "alice",
        "s3cr3t",
        "bob",
        "password",
        "abc",
        "key123",
        "deadbeef",
        "encoded-secret",
        "supersecret",
        "carol",
        "network-secret",
        "network-query",
    ):
        assert secret not in rendered
    assert "safe=yes" in rendered
    assert redact_activity_text(rendered) == rendered


def test_activity_text_redacts_standalone_authorization_credentials():
    from agent.tool_activity import redact_activity_text

    rendered = redact_activity_text(
        "Bearer abc.def.ghi Basic dXNlcjpwYXNz Token «redacted:ghp_…»"
    )

    assert rendered == "Bearer [REDACTED] Basic [REDACTED] Token [REDACTED]"


def test_activity_text_redacts_authorization_assignments_and_digest_idempotently():
    from agent.tool_activity import redact_activity_text

    raw = (
        "authorization=Bearer super-secret-token\n"
        "Authorization=Basic dXNlcjpwYXNz\n"
        "Proxy-Authorization=Bearer proxy-secret\n"
        'Digest username="alice", realm="private", nonce="nonce-secret", response="response-secret"'
    )
    rendered = redact_activity_text(raw)

    for secret in (
        "super-secret-token",
        "dXNlcjpwYXNz",
        "proxy-secret",
        "alice",
        "private",
        "nonce-secret",
        "response-secret",
    ):
        assert secret not in rendered
    assert redact_activity_text(rendered) == rendered


def test_activity_redaction_failure_omits_presentation_text(monkeypatch):
    import agent.tool_activity as tool_activity

    def fail_redaction(*_args, **_kwargs):
        raise RuntimeError("redactor unavailable")

    monkeypatch.setattr(tool_activity, "redact_sensitive_text", fail_redaction)

    assert tool_activity.redact_activity_text("PRIVATE_MARKER") == ""
    assert tool_activity.redact_activity_args(
        {"command": "PRIVATE_MARKER", "authorization": "Bearer CREDENTIAL_MARKER"}
    ) == {"command": "", "authorization": "[REDACTED]"}


def test_activity_args_recursively_redact_sensitive_keys_urls_and_auth_values():
    from agent.tool_activity import redact_activity_args

    raw = {
        "command": "curl -H 'Authorization=Bearer command-secret' https://u:p@example.test/x?sig=query-secret",
        "api_key": "key-secret",
        "nested": [{"password": "password-secret", "safe": "visible"}],
    }
    rendered = redact_activity_args(raw)

    assert raw["api_key"] == "key-secret"
    assert rendered["api_key"] == "[REDACTED]"
    assert rendered["nested"][0] == {"password": "[REDACTED]", "safe": "visible"}
    assert "command-secret" not in rendered["command"]
    assert "query-secret" not in rendered["command"]
    assert "u:p@" not in rendered["command"]


def test_high_content_activity_args_keep_targets_without_payloads():
    from types import SimpleNamespace

    from agent.tool_executor import _activity_display_args, _agent_activity_display_args

    content_sentinel = "RAW_WRITE_FILE_CONTENT_SENTINEL_7b1df8"
    patch_sentinel = "RAW_PATCH_BODY_SENTINEL_4d31c2"

    assert _activity_display_args(
        "write_file",
        {"path": "safe.txt", "content": content_sentinel},
    ) == {"path": "safe.txt"}
    assert _activity_display_args(
        "patch",
        {
            "mode": "replace",
            "path": "src/auth/session.py",
            "old_string": patch_sentinel,
            "new_string": "replacement",
            "replace_all": True,
        },
    ) == {
        "mode": "replace",
        "path": "src/auth/session.py",
        "replace_all": True,
    }

    bulk_args = {
        "mode": "patch",
        "patch": (
            "*** Begin Patch\n"
            "*** Update File: src/auth/session.py\n"
            f"-{patch_sentinel}\n"
            "+replacement\n"
            "*** End Patch"
        ),
    }
    bulk_display = _activity_display_args("patch", bulk_args)
    assert bulk_display == {"mode": "patch", "path": "src/auth/session.py"}
    assert content_sentinel not in repr(bulk_display)
    assert patch_sentinel not in repr(bulk_display)

    assert _activity_display_args(
        "browser_type",
        {"ref": "@e1", "text": "RAW_TYPED_SENTINEL"},
    ) == {"ref": "@e1"}
    assert _activity_display_args(
        "custom_plugin",
        {"path": "safe.txt", "query": "visible target", "content": "RAW_PLUGIN_SENTINEL"},
        supported_tool=True,
    ) == {"path": "safe.txt", "query": "visible target"}
    assert _activity_display_args(
        "unknown_tool",
        {"path": "safe.txt", "content": "RAW_UNKNOWN_SENTINEL"},
    ) == {}
    agent = SimpleNamespace(
        valid_tool_names={"read_file"},
        tool_reasons_enabled=False,
        _tool_reason_tool_names=set(),
    )
    assert _agent_activity_display_args(
        agent,
        "read_file",
        {"path": "safe.txt", "content": "RAW_SUPPORTED_SENTINEL"},
    ) == {"path": "safe.txt"}
    assert _agent_activity_display_args(
        agent,
        "unknown_tool",
        {"path": "safe.txt", "content": "RAW_UNKNOWN_SENTINEL"},
    ) == {}


def test_summarize_write_patch_and_search_semantics():
    from agent.tool_activity import summarize_tool_result

    assert summarize_tool_result(
        "write_file",
        {"content": "a\nb\n"},
        '{"bytes_written": 4}',
        duration_s=0,
    ) == "write_file: 2 lines, 4 bytes"
    assert summarize_tool_result(
        "write_file", {"content": "x"}, '{"bytes_written": 1}', duration_s=0
    ) == "write_file: 1 line, 1 byte"
    assert summarize_tool_result(
        "write_file", {}, '{"bytes_written": 4}', duration_s=0
    ) == "write_file: 4 bytes"
    assert summarize_tool_result("write_file", {}, '{"success": true}', duration_s=0) == ""
    assert summarize_tool_result(
        "patch", {}, "*** Begin Patch\n*** Update File: a.py\n-old\n+new\n*** End Patch"
    ) == "patch: +1/-1 in 1 file"
    assert summarize_tool_result(
        "search_files", {}, '{"total_count": 3}', duration_s=0
    ) == "search_files: 3 matches"
    assert summarize_tool_result(
        "search_files", {"target": "files"}, '{"total_count": 1}', duration_s=0
    ) == "search_files: 1 file"


def test_summarize_read_and_search_suppress_unstructured_acknowledgements():
    from agent.tool_activity import summarize_tool_result

    assert summarize_tool_result("read_file", {}, '{"success": true}') == ""
    assert summarize_tool_result("read_file", {}, "first\nsecond") == ""
    assert summarize_tool_result("search_files", {}, '{"success": true}') == ""
    assert summarize_tool_result("search_files", {}, "first\nsecond") == ""

def test_summarize_web_results_use_semantic_units_and_grammar():
    from agent.tool_activity import summarize_tool_result

    assert summarize_tool_result("web_search", {}, '{"count": 1}') == "web_search: 1 result"
    assert summarize_tool_result("web_search", {}, '{"count": 2}') == "web_search: 2 results"
    assert summarize_tool_result("web_search", {}, '{"data": {"web": [{}, {}, {}]}}') == "web_search: 3 results"
    assert summarize_tool_result("web_extract", {}, '{"results": [{}]}') == "web_extract: 1 page"
    assert summarize_tool_result("web_extract", {}, '{"results": [{}, {}]}') == "web_extract: 2 pages"


def test_summarize_skills_list_but_suppresses_skill_acknowledgements():
    from agent.tool_activity import summarize_tool_result

    assert summarize_tool_result(
        "skills_list", {}, '{"success": true, "count": 1, "skills": [{}]}'
    ) == "skills_list: 1 skill"
    assert summarize_tool_result(
        "skills_list", {}, '{"success": true, "count": 2, "skills": [{}, {}]}'
    ) == "skills_list: 2 skills"
    assert summarize_tool_result(
        "skill_view",
        {"name": "systematic-debugging"},
        '{"success": true, "content": "large"}',
    ) == ""
    assert summarize_tool_result(
        "skill_manage", {"action": "patch", "name": "example"}, '{"success": true}'
    ) == ""


def test_summarize_todo_uses_task_state_counts():
    from agent.tool_activity import summarize_tool_result

    raw = json.dumps({
        "todos": [{}, {}, {}, {}],
        "summary": {
            "total": 4,
            "pending": 1,
            "in_progress": 1,
            "completed": 2,
            "cancelled": 0,
        },
    })

    assert summarize_tool_result("todo", {}, raw) == "todo: 4 tasks · 1 active · 1 pending · 2 done"


def test_summarize_session_process_delegate_and_execute_semantics():
    from agent.tool_activity import summarize_tool_result

    assert summarize_tool_result(
        "session_search", {}, '{"mode": "browse", "count": 1}'
    ) == "session_search: 1 session"
    assert summarize_tool_result(
        "session_search", {}, '{"mode": "read", "message_count": 3}'
    ) == "session_search: 3 messages"
    assert summarize_tool_result(
        "process", {"action": "list"}, '{"processes": [{}]}'
    ) == "process: 1 process"
    assert summarize_tool_result(
        "process", {"action": "poll"}, '{"status": "running"}'
    ) == "process: running"
    assert summarize_tool_result(
        "process", {"action": "wait"}, '{"status": "exited", "exit_code": 0}'
    ) == "process: exit 0"
    assert summarize_tool_result(
        "delegate_task", {}, '{"status": "dispatched", "count": 1}'
    ) == "delegate_task: 1 task dispatched"
    assert summarize_tool_result(
        "delegate_task", {}, '{"status": "dispatched", "count": 3}'
    ) == "delegate_task: 3 tasks dispatched"
    assert summarize_tool_result(
        "cronjob", {"action": "list"}, '{"jobs": [{}]}'
    ) == "cronjob: 1 job"
    assert summarize_tool_result(
        "execute_code", {}, '{"output": "ok", "exit_code": 0}'
    ) == "execute_code: exit 0"
    assert summarize_tool_result(
        "execute_code", {}, '{"output": "bad", "exit_code": 2}'
    ) == "execute_code: error: exit 2"


def test_summarize_list_tools_use_domain_units():
    from agent.tool_activity import summarize_tool_result

    assert summarize_tool_result(
        "browser_get_images", {}, '{"images": [{}]}'
    ) == "browser_get_images: 1 image"
    assert summarize_tool_result(
        "ha_list_entities", {}, '[{}, {}]'
    ) == "ha_list_entities: 2 entities"
    assert summarize_tool_result(
        "ha_list_services", {}, '[{}]'
    ) == "ha_list_services: 1 service"
    assert summarize_tool_result(
        "mcp__qmd__list_resources", {}, '{"resources": [{}, {}]}'
    ) == "mcp__qmd__list_resources: 2 resources"


def test_summarize_acknowledgement_and_unknown_tools_emit_no_noise():
    from agent.tool_activity import summarize_tool_result

    cases = [
        ("clarify", {"question": "Which?"}, '{"answer": "A"}'),
        ("memory", {"action": "add"}, '{"success": true}'),
        ("browser_click", {"ref": "@e1"}, '{"ok": true}'),
        ("browser_navigate", {"url": "https://example.test"}, '{"url": "https://example.test"}'),
        ("vision_analyze", {}, "Image loaded into context"),
        ("image_generate", {}, '{"image": "/tmp/out.png"}'),
        ("ha_call_service", {"service": "turn_on"}, '{"success": true}'),
        ("unknown_plugin_tool", {}, "one\ntwo"),
    ]

    for name, args, raw in cases:
        assert summarize_tool_result(name, args, raw) == "", name


def test_legacy_cli_completion_uses_presentation_safe_arguments():
    from agent.tool_executor import _get_cute_tool_message_impl

    marker = "sk-proj-ABCDEF1234567890"
    rendered = _get_cute_tool_message_impl(
        "terminal",
        {"command": f"curl -H 'Authorization: Bearer {marker}' https://example.com"},
        0.25,
        result="ok",
    )

    assert marker not in rendered
    assert "Authorization" in rendered


def test_callback_metadata_is_filtered_without_retrying_callback_type_errors():
    from agent.tool_executor import _call_tool_callback

    calls = []

    def legacy(value):
        calls.append(value)

    _call_tool_callback(legacy, ("ok",), reason="inspect")
    assert calls == ["ok"]

    def positional_only(value, reason=None, /):
        calls.append((value, reason))

    _call_tool_callback(positional_only, ("positional",), reason="must not become a keyword")
    assert calls[-1] == ("positional", None)

    def broken(value, **metadata):
        calls.append(value)
        raise TypeError("callback body failed")

    try:
        _call_tool_callback(broken, ("once",), reason="inspect")
    except TypeError as exc:
        assert str(exc) == "callback body failed"
    else:
        raise AssertionError("callback TypeError should propagate to its caller")
    assert calls == ["ok", ("positional", None), "once"]


def test_callback_metadata_does_not_repeat_positionally_bound_names():
    from agent.tool_executor import _call_tool_callback

    received = []

    def callback(tool_call_id, **metadata):
        received.append((tool_call_id, metadata))

    _call_tool_callback(
        callback,
        ("call-1",),
        tool_call_id="call-1",
        call_id="call-1",
        status="completed",
    )

    assert received == [
        ("call-1", {"call_id": "call-1", "status": "completed"})
    ]


def test_persisted_activity_preserves_raw_result_and_is_omitted_when_disabled():
    from agent.tool_dispatch_helpers import make_tool_result_message
    from agent.tool_executor import _persisted_tool_activity

    metadata = {
        "reason": "Inspect the result",
        "summary": "terminal: exit 0",
        "status": "completed",
        "is_error": False,
        "duration_seconds": 0.2,
    }
    activity = _persisted_tool_activity(metadata)
    message = make_tool_result_message(
        "terminal",
        "raw output",
        "call-1",
        tool_activity=activity,
    )

    assert message["content"] == "raw output"
    assert message["_tool_activity"] == activity
    assert _persisted_tool_activity({"status": "success"}) is None


def test_start_metadata_uses_normalized_running_status():
    from agent.tool_executor import _start_metadata

    agent = type("Agent", (), {"_tool_call_reasons": {"call-1": "Inspect status"}})()

    assert _start_metadata(agent, "call-1") == {
        "reason": "Inspect status",
        "tool_call_id": "call-1",
        "call_id": "call-1",
        "status": "running",
    }


def test_completion_metadata_exposes_canonical_and_legacy_aliases():
    from agent.tool_executor import _completion_metadata

    agent = type(
        "Agent",
        (),
        {"_tool_call_reasons": {"call-1": "Inspect status"}, "tool_result_summaries_enabled": True},
    )()

    metadata = _completion_metadata(agent, "call-1", "terminal", {}, '{"exit_code": 0}', 1.25, False)

    assert metadata["tool_call_id"] == metadata["call_id"] == "call-1"
    assert metadata["duration"] == metadata["duration_seconds"] == metadata["duration_s"] == 1.25
    assert metadata["is_error"] is False
    assert metadata["status"] == "completed"
    assert metadata["reason"] == "Inspect status"
    assert metadata["summary"] == "terminal: exit 0 in 1.2s"
