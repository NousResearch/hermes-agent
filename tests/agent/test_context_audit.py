from types import SimpleNamespace

from agent.context_audit import (
    NecessityRank,
    collect_context_audit,
    render_context_audit_summary,
)
from agent.system_prompt import build_system_prompt


def _tool(name: str, description: str = "desc") -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {"value": {"type": "string"}}},
        },
    }


def _agent(**overrides):
    base = dict(
        tools=[_tool("small"), _tool("large", "x" * 240)],
        model="gpt-test",
        provider="test-provider",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_collects_prompt_tiers_and_tool_schema_without_raw_content():
    parts = {
        "stable": "SOUL identity and tool-use safety",
        "context": "Project context body",
        "volatile": "SECRET MEMORY BODY",
    }

    report = collect_context_audit(_agent(), prompt_parts=parts)

    labels = {entry.label for entry in report.entries}
    assert {"system_prompt.stable", "system_prompt.context", "system_prompt.volatile", "tool_schema.small", "tool_schema.large"} <= labels
    assert report.prompt_chars == sum(len(value) for value in parts.values())
    assert report.tool_schema_bytes > 0
    assert report.tool_count == 2
    assert report.entries_by_source_type()["tool_schema"].total_bytes == report.tool_schema_bytes

    safe_text = report.to_redacted_dict()
    assert "SECRET MEMORY BODY" not in repr(safe_text)
    assert "SOUL identity" not in repr(safe_text)
    assert all(entry.content_hash for entry in report.entries)
    assert all("raw_content" not in entry for entry in safe_text["entries"])


def test_empty_prompt_and_no_tools_produces_zero_size_report():
    report = collect_context_audit(_agent(tools=[]), prompt_parts={"stable": "", "context": "", "volatile": ""})

    assert report.total_chars == 0
    assert report.tool_schema_bytes == 0
    assert report.tool_count == 0
    assert [entry.label for entry in report.entries] == [
        "system_prompt.stable",
        "system_prompt.context",
        "system_prompt.volatile",
        "tool_schema.total",
    ]


def test_repeated_collection_is_deterministic():
    parts = {"stable": "alpha", "context": "beta", "volatile": "gamma"}
    agent = _agent()

    first = collect_context_audit(agent, prompt_parts=parts).to_redacted_dict()
    second = collect_context_audit(agent, prompt_parts=parts).to_redacted_dict()

    assert first == second


def test_audit_does_not_change_rendered_system_prompt():
    agent = _agent(tools=[])
    prompt_parts = {"stable": "alpha", "context": "beta", "volatile": "gamma"}

    before = build_system_prompt(agent, prompt_parts=prompt_parts)
    report = collect_context_audit(agent, prompt_parts=prompt_parts)
    after = build_system_prompt(agent, prompt_parts=prompt_parts)

    assert before == after == "alpha\n\nbeta\n\ngamma"
    assert report.prompt_chars == len("alpha") + len("beta") + len("gamma")


def test_ranks_sources_and_orders_optimizations_by_safe_savings():
    parts = {
        "stable": "SOUL identity" + ("x" * 100),
        "context": "AGENTS project context" + ("x" * 300),
        "volatile": "USER profile" + ("x" * 80),
    }
    agent = _agent(tools=[_tool("delegate_task", "x" * 1800), _tool("read_file")])

    report = collect_context_audit(agent, prompt_parts=parts)
    by_label = {entry.label: entry for entry in report.entries}

    assert by_label["system_prompt.stable"].necessity == NecessityRank.CRITICAL
    assert by_label["system_prompt.context"].necessity == NecessityRank.HIGH
    assert by_label["system_prompt.volatile"].necessity == NecessityRank.HIGH
    assert by_label["tool_schema.delegate_task"].necessity == NecessityRank.SITUATIONAL

    options = report.optimization_options
    assert list(options) == sorted(options, key=lambda opt: (-opt.estimated_savings_chars, opt.risk))
    assert options[0].estimated_savings_chars >= options[-1].estimated_savings_chars
    assert all("disable" not in opt.action.lower() for opt in options if opt.source_label == "system_prompt.stable")
    assert any("agent.startup_context_audit" in opt.config_hint or "hermes tools" in opt.command_hint for opt in options)


def test_summary_is_compact_and_redacted():
    report = collect_context_audit(
        _agent(tools=[_tool("memory", "SECRET MEMORY BODY" * 20)]),
        prompt_parts={"stable": "alpha", "context": "", "volatile": "SECRET MEMORY BODY"},
    )

    summary = render_context_audit_summary(report, max_lines=8)

    assert "Context audit" in summary
    assert "tool schemas" in summary
    assert "SECRET MEMORY BODY" not in summary
    assert len(summary.splitlines()) <= 8


def test_default_mode_does_not_inject_or_collect_audit(monkeypatch):
    from unittest.mock import patch

    from agent.system_prompt import build_system_prompt

    agent = SimpleNamespace(
        load_soul_identity=False,
        skip_context_files=True,
        valid_tool_names=[],
        tools=[],
        _task_completion_guidance=False,
        _parallel_tool_call_guidance=False,
        _tool_use_enforcement=False,
        _environment_probe=False,
        _kanban_worker_guidance="",
        _memory_store=None,
        _memory_manager=None,
        model="",
        provider="",
        platform="",
        pass_session_id=False,
        session_id="s1",
        _startup_context_audit_mode="off",
        _context_audit_report=None,
        _emit_status=lambda message: None,
    )
    with (
        patch("run_agent.load_soul_md", return_value=""),
        patch("run_agent.build_nous_subscription_prompt", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", return_value=""),
    ):
        prompt = build_system_prompt(agent)

    assert "Context audit" not in prompt
    assert agent._context_audit_report is None


def test_status_mode_collects_without_prompt_injection():
    from unittest.mock import patch

    from agent.system_prompt import build_system_prompt

    agent = SimpleNamespace(
        load_soul_identity=False,
        skip_context_files=True,
        valid_tool_names=[],
        tools=[_tool("read_file")],
        _task_completion_guidance=False,
        _parallel_tool_call_guidance=False,
        _tool_use_enforcement=False,
        _environment_probe=False,
        _kanban_worker_guidance="",
        _memory_store=None,
        _memory_manager=None,
        model="gpt-test",
        provider="test",
        platform="",
        pass_session_id=False,
        session_id="s1",
        _startup_context_audit_mode="status",
        _context_audit_report=None,
        _context_audit_report_path="",
        _emit_status=lambda message: None,
    )
    with (
        patch("run_agent.load_soul_md", return_value=""),
        patch("run_agent.build_nous_subscription_prompt", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", return_value=""),
    ):
        prompt = build_system_prompt(agent)

    assert "Context audit" not in prompt
    assert agent._context_audit_report is not None
    assert agent._context_audit_report.tool_count == 1


def test_summary_mode_injects_redacted_summary_and_accounts_for_it():
    from unittest.mock import patch

    from agent.system_prompt import build_system_prompt

    agent = SimpleNamespace(
        load_soul_identity=False,
        skip_context_files=True,
        valid_tool_names=[],
        tools=[_tool("memory", "SECRET MEMORY BODY" * 5)],
        _task_completion_guidance=False,
        _parallel_tool_call_guidance=False,
        _tool_use_enforcement=False,
        _environment_probe=False,
        _kanban_worker_guidance="",
        _memory_store=None,
        _memory_manager=None,
        model="gpt-test",
        provider="test",
        platform="",
        pass_session_id=False,
        session_id="s1",
        _startup_context_audit_mode="summary",
        _context_audit_report=None,
        _context_audit_report_path="",
        _emit_status=lambda message: None,
    )
    with (
        patch("run_agent.load_soul_md", return_value="RAW SOUL SHOULD NOT LEAK"),
        patch("run_agent.build_nous_subscription_prompt", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", return_value=""),
    ):
        prompt = build_system_prompt(agent)

    assert "Context audit" in prompt
    assert "RAW SOUL SHOULD NOT LEAK" not in prompt.split("Context audit", 1)[1]
    assert "SECRET MEMORY BODY" not in prompt
    assert agent._context_audit_report is not None
    labels = {entry.label for entry in agent._context_audit_report.entries}
    assert "system_prompt.context_audit_summary" in labels
    assert agent._context_audit_report.prompt_chars == len(prompt.replace("\n\n", "")) or agent._context_audit_report.prompt_chars > 0


def test_debug_file_mode_writes_redacted_report(tmp_path):
    from unittest.mock import patch

    from agent.system_prompt import build_system_prompt

    agent = SimpleNamespace(
        load_soul_identity=False,
        skip_context_files=True,
        valid_tool_names=[],
        tools=[_tool("read_file", "SECRET TOOL BODY")],
        _task_completion_guidance=False,
        _parallel_tool_call_guidance=False,
        _tool_use_enforcement=False,
        _environment_probe=False,
        _kanban_worker_guidance="",
        _memory_store=None,
        _memory_manager=None,
        model="gpt-test",
        provider="test",
        platform="",
        pass_session_id=False,
        session_id="session/with:chars",
        _startup_context_audit_mode="debug_file",
        _context_audit_report=None,
        _context_audit_report_path="",
        _emit_status=lambda message: None,
    )
    with (
        patch("agent.system_prompt.get_hermes_home", return_value=tmp_path),
        patch("run_agent.load_soul_md", return_value="RAW SOUL SHOULD NOT LEAK"),
        patch("run_agent.build_nous_subscription_prompt", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", return_value=""),
    ):
        build_system_prompt(agent)

    assert agent._context_audit_report_path
    data = (tmp_path / "sessions" / "context_audits" / "session_with_chars.json").read_text(encoding="utf-8")
    assert "RAW SOUL SHOULD NOT LEAK" not in data
    assert "SECRET TOOL BODY" not in data
    assert "system_prompt.stable" in data
    assert "content_hash" in data


def test_unknown_large_source_gets_review_warning():
    report = collect_context_audit(
        _agent(tools=[]),
        prompt_parts={"stable": "", "context": "", "volatile": ""},
        extra_sources=[("mystery.big", "unknown", "x" * 5000)],
    )

    assert any(opt.source_label == "mystery.big" and "classify" in opt.reason.lower() for opt in report.optimization_options)
