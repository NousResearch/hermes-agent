"""Focused contract tests for top-level ``-z --output-schema`` (#108397)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from hermes_cli._parser import build_top_level_parser
from hermes_cli.oneshot import _run_conversation_with_output_schema, run_oneshot


SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def _schema_file(tmp_path, schema=SCHEMA):
    path = tmp_path / "schema.json"
    path.write_text(json.dumps(schema), encoding="utf-8")
    return path


def test_top_level_parser_accepts_output_paths():
    parser = build_top_level_parser()[0]

    args = parser.parse_args([
        "-z", "summarize", "--output-schema", "schema.json",
        "--output-last-message", "result.json",
    ])

    assert args.output_schema == "schema.json"
    assert args.output_last_message == "result.json"


def test_main_threads_output_paths_to_oneshot(monkeypatch):
    import hermes_cli.main as main

    captured = {}
    args = SimpleNamespace(
        oneshot="summarize",
        model=None,
        provider=None,
        toolsets=None,
        skills=None,
        usage_file=None,
        resume=None,
        reasoning=None,
        output_schema="schema.json",
        output_last_message="result.json",
    )
    monkeypatch.setattr(main, "_confirm_startup_expensive_model_override", lambda _args: None)
    monkeypatch.setattr(main, "_resolve_chat_session_args", lambda _args, **_kwargs: None)
    monkeypatch.setattr(main, "_run_and_exit_oneshot", lambda prompt, **kwargs: captured.update(prompt=prompt, **kwargs))

    main._run_oneshot_from_args(args)

    assert captured["output_schema"] == "schema.json"
    assert captured["output_last_message"] == "result.json"


@pytest.mark.parametrize(
    ("filename", "contents", "error_fragment"),
    [
        ("missing.json", None, "cannot read --output-schema"),
        ("schema.json", "{", "not valid JSON"),
        ("schema.json", '{"type": 7}', "not a valid JSON Schema"),
    ],
)
def test_schema_file_errors_happen_before_agent(
    tmp_path, monkeypatch, capsys, filename, contents, error_fragment,
):
    schema_path = tmp_path / filename
    if contents is not None:
        schema_path.write_text(contents, encoding="utf-8")
    called = False

    def fail_if_called(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("agent must not run for an invalid schema")

    monkeypatch.setattr("hermes_cli.oneshot._run_agent", fail_if_called)

    rc = run_oneshot("summarize", output_schema=str(schema_path))

    captured = capsys.readouterr()
    assert rc != 0
    assert not called
    assert captured.out == ""
    assert error_fragment in captured.err


def test_output_last_message_requires_output_schema(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "hermes_cli.oneshot._run_agent",
        lambda *_args, **_kwargs: pytest.fail("agent must not run"),
    )

    rc = run_oneshot("summarize", output_last_message=str(tmp_path / "result.json"))

    captured = capsys.readouterr()
    assert rc != 0
    assert captured.out == ""
    assert "--output-last-message requires --output-schema" in captured.err


def test_valid_output_is_printed_and_contract_is_appended(tmp_path, monkeypatch, capsys):
    prompts = []

    def fake_run_agent(prompt, **_kwargs):
        prompts.append(prompt)
        return '{"answer":"ok"}', {"final_response": '{"answer":"ok"}'}

    monkeypatch.setattr("hermes_cli.oneshot._run_agent", fake_run_agent)

    rc = run_oneshot("summarize", output_schema=str(_schema_file(tmp_path)))

    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out == '{"answer":"ok"}\n'
    assert len(prompts) == 1
    assert prompts[0].startswith("summarize\n\nOUTPUT CONTRACT (machine-validated):")
    assert '"required": [\n    "answer"' in prompts[0]


def test_invalid_output_retries_exactly_once_then_prints_valid_payload(
    tmp_path, monkeypatch, capsys,
):
    prompts = []

    class FakeAgent:
        responses = iter([
            {
                "final_response": '{"wrong":1}',
                "messages": [
                    {"role": "user", "content": "contracted prompt"},
                    {"role": "assistant", "content": '{"wrong":1}'},
                ],
                "api_calls": 1,
                "total_tokens": 10,
            },
            {"final_response": '{"answer":"fixed"}', "api_calls": 1, "total_tokens": 20},
        ])

        def run_conversation(self, prompt, conversation_history=None):
            prompts.append((prompt, conversation_history))
            return next(self.responses)

    result = _run_conversation_with_output_schema(
        FakeAgent(),
        "contracted prompt",
        conversation_history=[{"role": "user", "content": "prior"}],
        output_schema=SCHEMA,
    )

    monkeypatch.setattr(
        "hermes_cli.oneshot._run_agent",
        lambda *_args, **_kwargs: (result["final_response"], result),
    )

    rc = run_oneshot("summarize", output_schema=str(_schema_file(tmp_path)))

    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out == '{"answer":"fixed"}\n'
    assert len(prompts) == 2
    assert prompts[0][1] == [{"role": "user", "content": "prior"}]
    assert "previous final response was rejected" in prompts[1][0]
    assert "$: 'answer' is a required property" in prompts[1][0]
    assert "OUTPUT CONTRACT (machine-validated):" not in prompts[1][0]
    assert prompts[1][1] == [
        {"role": "user", "content": "contracted prompt"},
        {"role": "assistant", "content": '{"wrong":1}'},
    ]
    assert result["api_calls"] == 2
    assert result["total_tokens"] == 30


def test_second_invalid_output_is_not_printed_or_written(tmp_path, monkeypatch, capsys):
    output_path = tmp_path / "result.json"
    prompts = []

    class FakeAgent:
        def run_conversation(self, prompt, conversation_history=None):
            prompts.append(prompt)
            return {"final_response": "not json", "api_calls": 1}

    result = _run_conversation_with_output_schema(
        FakeAgent(), "contracted prompt", conversation_history=None, output_schema=SCHEMA,
    )

    monkeypatch.setattr(
        "hermes_cli.oneshot._run_agent",
        lambda *_args, **_kwargs: (result["final_response"], result),
    )

    rc = run_oneshot(
        "summarize",
        output_schema=str(_schema_file(tmp_path)),
        output_last_message=str(output_path),
    )

    captured = capsys.readouterr()
    assert rc != 0
    assert len(prompts) == 2
    assert captured.out == ""
    assert "did not satisfy --output-schema after one retry" in captured.err
    assert not output_path.exists()


def test_output_last_message_contains_only_validated_payload(tmp_path, monkeypatch, capsys):
    output_path = tmp_path / "nested" / "result.json"
    monkeypatch.setattr(
        "hermes_cli.oneshot._run_agent",
        lambda *_args, **_kwargs: (
            '{"answer":"saved"}',
            {"final_response": '{"answer":"saved"}'},
        ),
    )

    rc = run_oneshot(
        "summarize",
        output_schema=str(_schema_file(tmp_path)),
        output_last_message=str(output_path),
    )

    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out == '{"answer":"saved"}\n'
    assert output_path.read_text(encoding="utf-8") == '{"answer":"saved"}\n'
