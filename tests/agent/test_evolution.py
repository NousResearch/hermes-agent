import json

from agent.evolution import (
    analyze_sessions,
    load_trajectory_entries,
    render_markdown_report,
)


def test_analyze_sessions_surfaces_repeated_tool_failures_and_user_corrections():
    sessions = [
        {
            "id": "s1",
            "messages": [
                {"role": "tool", "tool_name": "web_extract", "content": "Error: Invalid URL '/v2/scrape'"},
                {"role": "user", "content": "Use terminal/python requests instead of web_extract."},
            ],
        },
        {
            "id": "s2",
            "messages": [
                {"role": "tool", "tool_name": "web_extract", "content": "Failed: Invalid URL '/v2/scrape'"},
                {"role": "user", "content": "Don't use web_extract here; use requests."},
            ],
        },
    ]

    report = analyze_sessions(sessions, min_count=2)

    finding_keys = {finding["key"] for finding in report["findings"]}
    assert "tool_failure:web_extract" in finding_keys
    assert "user_correction:tool_choice" in finding_keys

    tool_failure = next(f for f in report["findings"] if f["key"] == "tool_failure:web_extract")
    assert tool_failure["count"] == 2
    assert any("Invalid URL" in example for example in tool_failure["examples"])


def test_analyze_sessions_ignores_successful_tool_payloads_that_only_mention_errors_in_embedded_docs():
    sessions = [
        {
            "id": "s1",
            "messages": [
                {
                    "role": "tool",
                    "content": json.dumps(
                        {
                            "success": True,
                            "name": "skill_view",
                            "content": "Docs mention error handling and invalid commands, but the tool call succeeded.",
                        }
                    ),
                }
            ],
        },
        {
            "id": "s2",
            "messages": [
                {
                    "role": "tool",
                    "content": json.dumps(
                        {
                            "success": True,
                            "name": "skill_view",
                            "content": "Another successful payload that mentions failure modes in documentation.",
                        }
                    ),
                }
            ],
        },
    ]

    report = analyze_sessions(sessions, min_count=2)

    finding_keys = {finding["key"] for finding in report["findings"]}
    assert "tool_failure:unknown_tool" not in finding_keys
    assert "tool_failure:skill_view" not in finding_keys


def test_analyze_sessions_ignores_scheduler_wrapper_prompts_as_user_corrections():
    scheduler_wrapper = (
        "[SYSTEM: You are running as a scheduled cron job. DELIVERY: Your final response will be automatically delivered to the user.]\n\n"
        "Produce a daily Hermes self-evolution report and use terminal instead of web_extract if needed."
    )
    sessions = [
        {"id": "s1", "messages": [{"role": "user", "content": scheduler_wrapper}]},
        {"id": "s2", "messages": [{"role": "user", "content": scheduler_wrapper}]},
    ]

    report = analyze_sessions(sessions, min_count=2)

    finding_keys = {finding["key"] for finding in report["findings"]}
    assert "user_correction:tool_choice" not in finding_keys


def test_analyze_sessions_ignores_successful_terminal_payloads_even_if_output_mentions_errors():
    successful_payload = json.dumps(
        {
            "output": "# Hermes Self-Evolution Report\nThis output mentions hermes: error: invalid choice and other failure examples.",
            "exit_code": 0,
            "error": None,
        }
    )
    sessions = [
        {"id": "s1", "messages": [{"role": "tool", "content": successful_payload}]},
        {"id": "s2", "messages": [{"role": "tool", "content": successful_payload}]},
    ]

    report = analyze_sessions(sessions, min_count=2)

    finding_keys = {finding["key"] for finding in report["findings"]}
    assert "tool_failure:unknown_tool" not in finding_keys


def test_analyze_sessions_labels_terminal_failures_when_terminal_payload_lacks_tool_name():
    terminal_failure = json.dumps(
        {
            "output": "/usr/bin/bash: line 3: python: command not found",
            "exit_code": 127,
            "error": None,
        }
    )
    sessions = [
        {"id": "s1", "messages": [{"role": "tool", "content": terminal_failure}]},
        {"id": "s2", "messages": [{"role": "tool", "content": terminal_failure}]},
    ]

    report = analyze_sessions(sessions, min_count=2)

    finding_keys = {finding["key"] for finding in report["findings"]}
    assert "tool_failure:terminal" in finding_keys
    assert "tool_failure:unknown_tool" not in finding_keys


def test_analyze_sessions_ignores_terminal_non_error_exit_code_meanings():
    non_error_payload = json.dumps(
        {
            "output": "",
            "exit_code": 1,
            "error": None,
            "exit_code_meaning": "No matches found (not an error)",
        }
    )
    sessions = [
        {"id": "s1", "messages": [{"role": "tool", "content": non_error_payload}]},
        {"id": "s2", "messages": [{"role": "tool", "content": non_error_payload}]},
    ]

    report = analyze_sessions(sessions, min_count=2)

    finding_keys = {finding["key"] for finding in report["findings"]}
    assert "tool_failure:terminal" not in finding_keys
    assert "tool_failure:unknown_tool" not in finding_keys


def test_analyze_sessions_labels_execute_code_failures_when_payload_matches_sandbox_shape():
    execute_code_failure = json.dumps(
        {
            "status": "error",
            "output": "Traceback... ModuleNotFoundError: No module named 'llama_cpp'",
            "tool_calls_made": 0,
            "duration_seconds": 0.21,
            "error": "Traceback... ModuleNotFoundError: No module named 'llama_cpp'",
        }
    )
    sessions = [
        {"id": "s1", "messages": [{"role": "tool", "content": execute_code_failure}]},
        {"id": "s2", "messages": [{"role": "tool", "content": execute_code_failure}]},
    ]

    report = analyze_sessions(sessions, min_count=2)

    finding_keys = {finding["key"] for finding in report["findings"]}
    assert "tool_failure:execute_code" in finding_keys
    assert "tool_failure:unknown_tool" not in finding_keys


def test_render_markdown_report_includes_recommendations_and_prompt_deltas():
    report = {
        "summary": {"sessions_analyzed": 2, "trajectory_files": 1, "findings": 2},
        "findings": [
            {
                "key": "tool_failure:web_extract",
                "category": "tool_failure",
                "label": "Repeated failures for tool `web_extract`",
                "count": 2,
                "examples": ["Error: Invalid URL '/v2/scrape'"],
            }
        ],
        "recommendations": [
            "Investigate `web_extract` reliability or strengthen fallback guidance.",
        ],
        "prompt_deltas": [
            "When `web_extract` fails with URL/schema errors, immediately fall back to terminal or browser retrieval.",
        ],
    }

    markdown = render_markdown_report(report)

    assert "# Hermes Self-Evolution Report" in markdown
    assert "Repeated failures for tool `web_extract`" in markdown
    assert "## Recommendations" in markdown
    assert "## Candidate Prompt Deltas" in markdown
    assert "fall back to terminal or browser retrieval" in markdown


def test_load_trajectory_entries_skips_blank_and_malformed_lines(tmp_path):
    path = tmp_path / "trajectory.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"role": "assistant", "content": "Sorry, I can't do that."}),
                "",
                "{not json}",
                json.dumps({"role": "tool", "tool_name": "browser", "content": "Timeout error"}),
            ]
        ),
        encoding="utf-8",
    )

    entries = load_trajectory_entries(path)

    assert len(entries) == 2
    assert entries[0]["role"] == "assistant"
    assert entries[1]["tool_name"] == "browser"
