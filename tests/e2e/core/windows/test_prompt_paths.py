"""Prompt assembly with native Windows paths, through real Hermes processes.

* Subdirectory hints: after a terminal command touches ``backend\\src``, the tool result the
  model receives carries ``backend/AGENTS.md`` — for the forward-slash spelling AND the
  native backslash spelling a model emits on Windows (#121150).
* AGENTS.md chain: ``hermes chat -q`` launched in ``pkg\\inner`` of a git repo loads all
  three AGENTS.md files and labels them ``../../AGENTS.md`` / ``../AGENTS.md`` / ``AGENTS.md``
  — one spelling on every OS (#121015).
* ``@folder:`` references through the TUI gateway (the Ink TUI's process): the listing
  header injected into the user message is ``pkg/sub/``, matching its own ``/`` directory
  marker and entry lines, not ``pkg\\sub/`` (#121114).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.e2e.core.windows._helpers import (
    expect,
    hermes,
    known_marks,
    known,
    last_user,
    make_home,
    nonce,
    system_prompt,
    tool_results,
)
from tests.e2e.core.windows._rpc import StdioGateway
from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall

pytestmark = [pytest.mark.windows_only, pytest.mark.integration]

KNOWN: dict[str, str] = {
    "native-backslash": "#121150 subdirectory hints never load from native Windows paths (POSIX shlex)",
    "chain_labels": "#121015 AGENTS.md chain labels use os.path.relpath separators (..\\AGENTS.md)",
    "folder_header": "#121114 @folder listing header mixes separators on Windows (pkg\\sub/)",
}

# spelling -> terminal command the model issues (the issue's own repro commands)
HINT_COMMANDS: dict[str, str] = {
    "forward-slash": "python backend/src/main.py",
    "native-backslash": "python backend\\src\\main.py",
}


@pytest.mark.parametrize("spelling", [
    pytest.param(name, marks=known_marks(name, KNOWN)) for name in HINT_COMMANDS
])
def test_subdirectory_hint_reaches_model(spelling: str, tmp_path: Path) -> None:
    canary = nonce("BACKEND-RULES")
    with FakeLLMServer([ToolCall("terminal", {"command": HINT_COMMANDS[spelling]}), Text("done")]) as srv:
        home = make_home(tmp_path, srv.base_url)
        (home.project / "backend" / "src").mkdir(parents=True)
        (home.project / "backend" / "src" / "main.py").write_text("print('hi')\n", encoding="utf-8")
        (home.project / "backend" / "AGENTS.md").write_text(f"# Backend\n\n{canary}\n", encoding="utf-8")
        res = hermes(home, "chat", "-q", "Run the backend entry point.", "-Q")
        assert res.returncode == 0, res.tail()
        results = tool_results(srv)
    assert len(results) == 1, f"expected one terminal result on the wire, got {results}"
    expect(canary in results[0],
           f"{spelling}: backend/AGENTS.md never reached the model after {HINT_COMMANDS[spelling]!r}:\n"
           f"{results[0][-1500:]}")


@known("chain_labels", KNOWN)
def test_agents_chain_labels_are_os_independent(tmp_path: Path) -> None:
    canaries = {name: nonce(name.upper()) for name in ("root", "pkg", "inner")}
    with FakeLLMServer([Text("done")]) as srv:
        home = make_home(tmp_path, srv.base_url)
        repo = home.project
        subprocess.run(["git", "init", "-q", str(repo)], check=True, timeout=60)
        inner = repo / "pkg" / "inner"
        inner.mkdir(parents=True)
        for directory, name in ((repo, "root"), (repo / "pkg", "pkg"), (inner, "inner")):
            (directory / "AGENTS.md").write_text(f"# {name}\n\n{canaries[name]}\n", encoding="utf-8")
        res = hermes(home, "chat", "-q", "hello", "-Q", cwd=inner)
        assert res.returncode == 0, res.tail()
        prompt = system_prompt(srv.main_requests()[0])
    missing = [name for name, c in canaries.items() if c not in prompt]
    assert not missing, f"AGENTS.md chain members missing from the system prompt: {missing}"
    headings = [line[3:] for line in prompt.splitlines() if line.startswith("## ") and "AGENTS.md" in line]
    expect(headings == ["../../AGENTS.md", "../AGENTS.md", "AGENTS.md"],
           f"AGENTS.md chain headings are not the portable spelling: {headings}")


@known("folder_header", KNOWN)
def test_folder_reference_header_uses_one_separator(tmp_path: Path) -> None:
    body = nonce("FOLDER-FILE")
    with FakeLLMServer([Text("done")]) as srv:
        home = make_home(tmp_path, srv.base_url)
        nested = home.project / "pkg" / "sub"
        nested.mkdir(parents=True)
        (nested / "a.py").write_text(f"# {body}\nx = 1\n", encoding="utf-8")
        gw = StdioGateway(home)
        try:
            gw.turn("Review @folder:pkg/sub")
        finally:
            gw.close()
        user = last_user(srv.main_requests()[0])
    assert "- a.py" in user, f"@folder listing never reached the model:\n{user[-2000:]}"
    # The header is the one non-entry line naming the folder; entries are "- name" lines.
    header = [ln.strip() for ln in user.splitlines() if ln.strip().endswith("sub/") and not ln.lstrip().startswith("-")]
    expect(header == ["pkg/sub/"], f"@folder header lines: {header}\n{user[-1500:]}")
