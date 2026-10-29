"""Multiline script bodies must not become synthetic shell-command counts."""
import pytest

from agent.display import build_tool_preview, summarize_shell_command


@pytest.mark.parametrize("command", [
    "python - <<'PY'\nimport os; import sys; print(1)\nPY",
    'cat <<"EOF"\nalpha && beta || gamma; delta\nEOF',
    "cat <<-EOF\r\n\talpha; beta\r\n\tEOF",
    "import os\nprint(1); print(2)",
])
@pytest.mark.parametrize("tool,key", [("terminal", "command"), ("execute_code", "code")])
def test_multiline_script_preview_keeps_body_instead_of_counting_commands(command, tool, key):
    expected = " ".join(command.split())
    assert summarize_shell_command(command) == expected
    args = {key: command}
    assert build_tool_preview(tool, args, max_len=0) == expected
    assert build_tool_preview(tool, args, max_len=100) == expected
    assert args[key] == command


def test_multiline_preview_still_obeys_limit_and_single_line_summary_is_unchanged():
    command = "python - <<'PY'\nprint(1); " + "x" * 150 + "\nPY"
    preview = build_tool_preview("terminal", {"command": command}, max_len=100)
    assert len(preview) == 100
    assert preview.startswith("python - <<'PY' print(1);")
    assert summarize_shell_command("cd /tmp && echo one && echo two") == "echo one + 1 command"
