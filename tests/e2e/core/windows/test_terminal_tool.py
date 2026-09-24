"""The real terminal tool on native Windows, driven by a model turn through ``hermes chat -q``.

On Windows the terminal tool runs every command through Git Bash (``_find_bash``). The
model reaches native shells from there, so each case is one scripted turn:

    model -> terminal(<command>) -> real Git Bash child -> tool result -> model -> answer

and asserts the command's stdout AND its exit code on both the next wire request (what
the model is told) and the persisted ``tool`` row in ``state.db`` (what a resumed session
replays). The Git Bash case also proves the shell is MSYS/MinGW, not WSL.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.e2e.core.windows._helpers import (
    hermes,
    make_home,
    nonce,
    parse_tool_json,
    persisted_messages,
    tool_results,
)
from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall

pytestmark = [pytest.mark.windows_only, pytest.mark.integration]

# shell -> (command template, exit code, extra marker the output must carry)
SHELLS: dict[str, tuple[str, int, str]] = {
    # uname -s under Git Bash is MINGW64_NT-* / MSYS_NT-*; under WSL it would be Linux.
    "git-bash": ('echo "{marker}"; uname -s; exit 3', 3, "_NT-"),
    "powershell": ("powershell.exe -NoProfile -NonInteractive -Command "
                   "\"Write-Output '{marker}'; Write-Output \\$PSVersionTable.PSEdition; exit 5\"", 5, "Desktop"),
    "cmd": ('MSYS_NO_PATHCONV=1 cmd.exe /d /c "echo {marker}& ver& exit /b 7"', 7, "Windows"),
}


@pytest.mark.parametrize("shell", list(SHELLS))
def test_terminal_round_trip_persists_output_and_exit_code(shell: str, tmp_path: Path) -> None:
    template, code, extra = SHELLS[shell]
    marker, answer = nonce(shell.upper()), nonce("DONE")
    with FakeLLMServer([ToolCall("terminal", {"command": template.format(marker=marker)}), Text(answer)]) as srv:
        # `powershell -Command` is flagged as script execution; with no user present to approve,
        # -q blocks it. The documented opt-in keeps this about the tool round trip, not approvals.
        home = make_home(tmp_path, srv.base_url, extra_config="approvals:\n  single_query_mode: approve\n")
        res = hermes(home, "chat", "-q", f"Run the {shell} check.", "-Q")
        assert res.returncode == 0, res.tail()
        assert answer in res.stdout, f"final answer not delivered:\n{res.tail()}"
        results = tool_results(srv)

    assert len(results) == 1, f"expected one tool result on the wire, got {results}"
    wire = parse_tool_json(results[0])
    assert marker in (wire.get("output") or ""), f"{shell}: command output lost on the wire: {wire}"
    assert extra in (wire.get("output") or ""), f"{shell}: output is not from the expected shell: {wire}"
    assert wire.get("exit_code") == code, f"{shell}: exit code lost on the wire: {wire}"

    stored = [r for r in persisted_messages(home) if r["role"] == "tool"]
    assert len(stored) == 1, f"expected one persisted tool row, got {[dict(r) for r in stored]}"
    row = parse_tool_json(stored[0]["content"])
    assert marker in (row.get("output") or "") and row.get("exit_code") == code, (
        f"{shell}: persisted tool row differs from what the model saw: {row}")
