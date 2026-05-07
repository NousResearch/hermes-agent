import importlib.util
import sys
from pathlib import Path


def _load_eval_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "evals"
        / "codeact_research_eval.py"
    )
    spec = importlib.util.spec_from_file_location("codeact_research_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_code_tool_call(code: str, thoughts: str = "thinking"):
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "function": {
                    "name": "run_code",
                    "arguments": {
                        "thoughts": thoughts,
                        "code": code,
                    },
                }
            }
        ],
    }


def test_analyze_run_passes_when_first_call_uses_research_recipe():
    mod = _load_eval_module()
    messages = [
        _run_code_tool_call(
            "result = medical_pharma_research(question='latest GLP-1 drugs')"
        ),
        {
            "role": "tool",
            "name": "run_code",
            "content": (
                '{"success": true, "source_table": [{"id": "S1"}], '
                '"citation_metadata": {"count": 1}}'
            ),
        },
    ]

    analysis = mod.analyze_run(
        query={"id": "q", "topic": "medical_pharma", "prompt": "prompt"},
        result={
            "completed": True,
            "api_calls": 2,
            "messages": messages,
            "final_response": "Report with [S1].",
        },
        events=[],
    )

    assert analysis["verdict"] == "pass"
    assert analysis["summary"]["first_used_research_recipe"] is True
    assert analysis["summary"]["has_source_table"] is True
    assert analysis["summary"]["final_has_citations"] is True


def test_analyze_run_can_use_tool_callback_events_when_messages_are_sparse():
    mod = _load_eval_module()
    events = [
        mod.EvalEvent(
            kind="tool_start",
            timestamp="2026-05-06T00:00:00+00:00",
            tool_name="run_code",
            args={
                "thoughts": "researching",
                "code": "data = research_web(query='latest GLP-1 GIP drugs')",
            },
        ),
        mod.EvalEvent(
            kind="tool_complete",
            timestamp="2026-05-06T00:00:01+00:00",
            tool_name="run_code",
            result_preview='{"success": true, "source_table": [{"id": "S1"}]}',
        ),
    ]

    analysis = mod.analyze_run(
        query={"id": "q", "topic": "medical_pharma", "prompt": "prompt"},
        result={
            "completed": True,
            "api_calls": 1,
            "messages": [],
            "final_response": "Report with [S1].",
        },
        events=events,
    )

    assert analysis["verdict"] == "pass"
    assert analysis["summary"]["first_used_research_recipe"] is True
    assert analysis["summary"]["has_source_table"] is True


def test_analyze_run_accepts_research_web_redirect_from_web_search():
    mod = _load_eval_module()
    messages = [
        _run_code_tool_call("result = web_search(query='latest GLP-1 GIP drugs')"),
        {
            "role": "tool",
            "name": "run_code",
            "content": (
                '{"success": true, "redirected_from": "web_search", '
                '"redirected_to": "research_web", "source_table": [{"id": "S1"}]}'
            ),
        },
    ]

    analysis = mod.analyze_run(
        query={"id": "q", "topic": "medical_pharma", "prompt": "prompt"},
        result={
            "completed": True,
            "api_calls": 2,
            "messages": messages,
            "final_response": "Report with [S1].",
        },
        events=[],
    )

    assert analysis["verdict"] == "pass"
    assert analysis["summary"]["first_used_web_search"] is True
    assert analysis["summary"]["redirected_web_search"] is True


def test_analyze_run_fails_when_model_debugs_web_search_and_curls():
    mod = _load_eval_module()
    messages = [
        _run_code_tool_call("result = web_search(query='latest GLP-1 GIP drugs')"),
        {"role": "tool", "name": "run_code", "content": '{"error": "failed"}'},
        _run_code_tool_call("print(help('web_search'))"),
        {"role": "tool", "name": "run_code", "content": "docs"},
        _run_code_tool_call(
            "import subprocess\nsubprocess.run(['curl', 'https://html.duckduckgo.com/html/?q=x'])"
        ),
    ]

    analysis = mod.analyze_run(
        query={"id": "q", "topic": "medical_pharma", "prompt": "prompt"},
        result={
            "completed": True,
            "api_calls": 4,
            "messages": messages,
            "final_response": "uncited report",
        },
        events=[],
    )

    assert analysis["verdict"] == "fail"
    assert analysis["summary"]["debugged_web_search"] is True
    assert analysis["summary"]["used_raw_curl_or_search_scrape"] is True
