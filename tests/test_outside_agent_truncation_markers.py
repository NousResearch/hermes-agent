"""#121572: non-agent renderers use the non-imitable compression marker."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.compression_marker import (
    _COMPRESSION_MARKER_PREFIX,
    _COMPRESSION_MARKER_RE,
    elide,
    elide_middle,
)
from trajectory_compressor import TrajectoryCompressor


def test_elision_helpers_respect_contracts():
    text = "x" * 10_000
    for limit in (1, 100, 200, 1_000):
        rendered = elide(text, limit)
        assert _COMPRESSION_MARKER_RE.search(rendered)
        kept = rendered.split(_COMPRESSION_MARKER_PREFIX, 1)[0].rstrip()
        assert f"{len(text) - len(kept):,} of {len(text):,} chars omitted" in rendered
        if limit >= 100:
            assert len(rendered) <= limit

    assert elide_middle("abcdef", 3, 0).startswith("abc")
    assert "3 of 6 chars omitted" in elide_middle("abcdef", 3, 0)
    assert "def" not in elide_middle("abcdef", 3, 0)
    assert _COMPRESSION_MARKER_RE.search(elide_middle("abcdef", 0, 0))
    for args in ((-1, 2), (2, -1)):
        try:
            elide_middle(text, *args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {args}")

    for limit in (0, -1):
        try:
            elide(text, limit)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {limit}")


def test_timeout_diagnostic_marks_long_goal_as_non_original(tmp_path, monkeypatch):
    from tools.delegate_tool_child_run import _dump_subagent_timeout_diagnostic

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    goal = "g" * 1200
    child = SimpleNamespace(
        _subagent_id="child",
        valid_tool_names=[],
        get_activity_summary=lambda: {},
    )

    path = _dump_subagent_timeout_diagnostic(
        child=child,
        task_index=0,
        timeout_seconds=300.0,
        duration_seconds=300.0,
        worker_thread=None,
        goal=goal,
    )

    assert path is not None
    text = Path(path).read_text(encoding="utf-8-sig")
    assert _COMPRESSION_MARKER_PREFIX in text
    assert "g" * 100 in text
    assert "...[truncated]" not in text


def test_trajectory_summary_marks_middle_elision_as_non_original():
    compressor = TrajectoryCompressor.__new__(TrajectoryCompressor)
    value = "h" * 1500 + "m" * 2000 + "t" * 500
    rendered = compressor._extract_turn_content_for_summary(
        [{"from": "tool", "value": value}], 0, 1
    )

    assert rendered.startswith("[Turn 0 - TOOL]:\n" + "h" * 1500)
    assert rendered.endswith("t" * 500)
    assert _COMPRESSION_MARKER_PREFIX in rendered
    assert "...[truncated]" not in rendered


def test_session_search_eval_marks_long_tool_output_as_non_original(monkeypatch):
    from evals.session_search_schema import runner

    arm = SimpleNamespace(SESSION_SEARCH_SCHEMA={"name": "session_search", "description": "", "parameters": {}})
    function = SimpleNamespace(
        name="session_search",
        arguments='{"query":"needle"}',
    )
    tool_call = SimpleNamespace(id="call", function=function)
    response = SimpleNamespace(
        usage=None,
        choices=[SimpleNamespace(message=SimpleNamespace(
            content=None,
            tool_calls=[tool_call],
        ))],
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=MagicMock(side_effect=[
                response,
                SimpleNamespace(
                    usage=None,
                    choices=[SimpleNamespace(message=SimpleNamespace(
                        content="done", tool_calls=None
                    ))],
                ),
            ])),
        )
    )
    monkeypatch.setattr(
        runner,
        "exec_tool",
        lambda *args: ("o" * 30001, False),
    )

    result = runner.run_one(
        client=client,
        model="test/model",
        arm_name="base",
        arm_mod=arm,
        task_id="task",
        prompt="prompt",
        oracle=lambda text: text == "done",
        main_db_path=Path("unused.db"),
    )
    tool_message = client.chat.completions.create.call_args_list[0].kwargs["messages"][-1]

    assert result["ok"] is True
    assert _COMPRESSION_MARKER_PREFIX in tool_message["content"]
    assert "...[truncated]" not in tool_message["content"]
