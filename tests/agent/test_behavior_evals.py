"""Deterministic trajectory checks for repository behavior specs."""

import json

from agent.behavior_evals import evaluate_code_change_verification


def _assistant(message_id, tool, call_id, arguments):
    return {
        "id": message_id,
        "role": "assistant",
        "tool_calls": [{
            "id": call_id,
            "function": {"name": tool, "arguments": arguments},
        }],
    }


def _result(message_id, tool, call_id, content):
    return {
        "id": message_id,
        "role": "tool",
        "tool_name": tool,
        "tool_call_id": call_id,
        "content": content,
    }


def test_code_change_without_later_test_is_false():
    messages = [
        _assistant(1, "write_file", "write", '{"path":"src/app.py"}'),
        _result(2, "write_file", "write", '{"verified":true}'),
    ]
    result = evaluate_code_change_verification(messages)
    assert result["result"] == "false"
    assert result["evidence"]["last_mutation"]["path"] == "src/app.py"
    assert result["evidence"]["latest_verification"] is None


def test_successful_test_after_latest_change_is_true():
    messages = [
        _assistant(1, "patch", "patch", '{"mode":"replace","path":"src/app.py"}'),
        _result(2, "patch", "patch", '{"verified":true}'),
        _assistant(3, "terminal", "test", '{"command":"python -m pytest tests/test_app.py -q"}'),
        _result(4, "terminal", "test", '{"exit_code":0,"output":"1 passed"}'),
    ]
    result = evaluate_code_change_verification(messages)
    assert result["result"] == "true"
    assert "pytest" in result["evidence"]["latest_verification"]["command"]


def test_successful_unittest_after_latest_change_is_true():
    messages = [
        _assistant(1, "patch", "patch", '{"mode":"replace","path":"src/app.py"}'),
        _result(2, "patch", "patch", '{"verified":true}'),
        _assistant(3, "terminal", "test", '{"command":"python -m unittest discover -s tests -v"}'),
        _result(4, "terminal", "test", '{"exit_code":0,"output":"OK"}'),
    ]
    assert evaluate_code_change_verification(messages)["result"] == "true"


def test_successful_lint_and_typecheck_after_latest_change_are_true():
    commands = (
        "ruff check agent/behavior_evals.py",
        "python -m ruff check agent/behavior_evals.py",
        "uv run ruff check agent/behavior_evals.py",
        "mypy agent/behavior_evals.py",
        "python -m mypy agent/behavior_evals.py",
        "flake8 agent/behavior_evals.py",
        "black --check agent/behavior_evals.py",
        "npx eslint src/app.js",
    )
    for command in commands:
        messages = [
            _assistant(1, "patch", "patch", '{"mode":"replace","path":"src/app.py"}'),
            _result(2, "patch", "patch", '{"verified":true}'),
            _assistant(3, "terminal", "verify", json.dumps({"command": command})),
            _result(4, "terminal", "verify", '{"exit_code":0}'),
        ]
        assert evaluate_code_change_verification(messages)["result"] == "true", command


def test_any_mutating_eslint_segment_invalidates_verification_evidence():
    commands = (
        "npx eslint --fix src/app.js",
        "npx eslint src/app.js; npx eslint --fix src/app.js",
        "npx eslint --fix src/app.js; pytest -q",
        "eslint src/app.js && eslint --fix src/app.js",
        "pytest -q || npx eslint src/app.js --fix",
        "CI=1 npx eslint --fix src/app.js; pytest -q",
        "env FOO=1 eslint --fix src/app.js && pytest -q",
        "pnpm exec eslint --fix src/app.js; pytest -q",
        "yarn eslint --fix src/app.js; pytest -q",
        "npx --yes eslint --fix src/app.js; pytest -q",
        "./node_modules/.bin/eslint --fix src/app.js; pytest -q",
        'bash -c "eslint --fix src/app.js"; pytest -q',
        "eslint \\\n --fix src/app.js; pytest -q",
        "eslint.cmd --fix src/app.js; pytest -q",
        "eslint.exe src/app.js --fix; pytest -q",
    )
    for command in commands:
        messages = [
            _assistant(1, "write_file", "write", '{"path":"src/app.js"}'),
            _result(2, "write_file", "write", '{"verified":true}'),
            _assistant(3, "terminal", "lint", json.dumps({"command": command})),
            _result(4, "terminal", "lint", '{"exit_code":0}'),
        ]
        assert evaluate_code_change_verification(messages)["result"] == "false", command


def test_eslint_fix_dry_run_remains_non_mutating_verification():
    messages = [
        _assistant(1, "write_file", "write", '{"path":"src/app.js"}'),
        _result(2, "write_file", "write", '{"verified":true}'),
        _assistant(3, "terminal", "lint", '{"command":"npx eslint --fix-dry-run src/app.js"}'),
        _result(4, "terminal", "lint", '{"exit_code":0}'),
    ]
    assert evaluate_code_change_verification(messages)["result"] == "true"


def test_change_after_successful_test_requires_reverification():
    messages = [
        _assistant(1, "write_file", "first", '{"path":"src/app.py"}'),
        _result(2, "write_file", "first", '{"verified":true}'),
        _assistant(3, "terminal", "test", '{"command":"pytest -q"}'),
        _result(4, "terminal", "test", '{"exit_code":0}'),
        _assistant(5, "patch", "second", '{"mode":"replace","path":"src/app.py"}'),
        _result(6, "patch", "second", '{"verified":true}'),
    ]
    result = evaluate_code_change_verification(messages)
    assert result["result"] == "false"
    assert result["evidence"]["last_mutation"]["message_id"] == 6


def test_docs_only_change_is_not_applicable():
    messages = [
        _assistant(1, "write_file", "docs", '{"path":"README.md"}'),
        _result(2, "write_file", "docs", '{"verified":true}'),
    ]
    assert evaluate_code_change_verification(messages)["result"] == "n/a"


def test_unstructured_mutation_result_is_not_evidence():
    messages = [
        _assistant(1, "write_file", "write", '{"path":"src/app.py"}'),
        _result(2, "write_file", "write", "patched successfully"),
    ]
    assert evaluate_code_change_verification(messages)["result"] == "n/a"


def test_absolute_mutation_outside_repository_is_not_applicable():
    messages = [
        _assistant(1, "write_file", "temp", '{"path":"C:/Users/test/AppData/Local/Temp/verifier.py"}'),
        _result(2, "write_file", "temp", '{"verified":true}'),
    ]
    result = evaluate_code_change_verification(messages, repository_root="C:/work/project")
    assert result["result"] == "n/a"


def test_absolute_mutation_discovers_local_git_repository(tmp_path):
    (tmp_path / ".git").mkdir()
    source = tmp_path / "src" / "app.py"
    source.parent.mkdir()
    source.write_text("print('ok')", encoding="utf-8")
    messages = [
        _assistant(1, "write_file", "write", json.dumps({"path": str(source)})),
        _result(2, "write_file", "write", '{"verified":true}'),
    ]
    assert evaluate_code_change_verification(messages)["result"] == "false"


def test_failed_test_does_not_satisfy_behavior():
    messages = [
        _assistant(1, "write_file", "write", '{"path":"src/app.py"}'),
        _result(2, "write_file", "write", '{"verified":true}'),
        _assistant(3, "terminal", "test", '{"command":"pytest -q"}'),
        _result(4, "terminal", "test", '{"exit_code":1,"output":"failed"}'),
    ]
    result = evaluate_code_change_verification(messages)
    assert result["result"] == "false"
    assert result["evidence"]["latest_verification"]["success"] is False


def test_result_json_with_appended_context_keeps_exit_status():
    messages = [
        _assistant(1, "write_file", "write", '{"path":"src/app.py"}'),
        _result(2, "write_file", "write", '{"verified":true}'),
        _assistant(3, "terminal", "test", '{"command":"pytest -q"}'),
        _result(4, "terminal", "test", '{"output":"1 passed","exit_code":0}\n\n[Subdirectory context discovered]'),
    ]
    assert evaluate_code_change_verification(messages)["result"] == "true"


def test_summarized_terminal_result_keeps_exit_status():
    messages = [
        _assistant(1, "write_file", "write", '{"path":"src/app.py"}'),
        _result(2, "write_file", "write", '{"verified":true}'),
        _assistant(3, "terminal", "test", '{"command":"pytest -q"}'),
        _result(4, "terminal", "test", "[terminal] ran `pytest -q` -> exit 0, 1 lines output"),
    ]
    assert evaluate_code_change_verification(messages)["result"] == "true"


def test_msys_path_matches_windows_repository_root():
    messages = [
        _assistant(1, "write_file", "write", '{"path":"/c/work/project/src/app.py"}'),
        _result(2, "write_file", "write", '{"verified":true}'),
    ]
    result = evaluate_code_change_verification(messages, repository_root="C:/work/project")
    assert result["result"] == "false"


def test_relative_parent_traversal_is_not_applicable_without_root():
    messages = [
        _assistant(1, "write_file", "write", '{"path":"../outside.py"}'),
        _result(2, "write_file", "write", '{"verified":true}'),
    ]
    assert evaluate_code_change_verification(messages)["result"] == "n/a"


def test_string_message_ids_use_trace_order():
    messages = [
        _assistant("a", "write_file", "write", '{"path":"src/app.py"}'),
        _result("b", "write_file", "write", '{"verified":true}'),
        _assistant("c", "terminal", "test", '{"command":"pytest -q"}'),
        _result("d", "terminal", "test", '{"exit_code":0}'),
    ]
    assert evaluate_code_change_verification(messages)["result"] == "true"
