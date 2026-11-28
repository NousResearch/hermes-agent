"""Context-file caps include the recovery marker, not a fixed percentage reserve."""

import json
import re

import pytest

from agent.prompt_builder import _truncate_content, build_context_files_prompt, drain_truncation_warnings
from agent.subdirectory_hints import SubdirectoryHintTracker, _MAX_HINT_CHARS


@pytest.mark.parametrize("cap", [0, 1, 40, 200, 238, 1000, 19_000, 20_000, 48_000])
@pytest.mark.parametrize("content", [
    pytest.param("α😀" * 13_000 + "中β" * 13_000, id="plain"),
    pytest.param("α😀" * 13_000 + "\n## " + "section " * 1000 + "\n" + "中β" * 13_000,
                 id="long-heading"),
    pytest.param("HEAD\n" + "x" * 10_000 + "\n## " + "A" * 500 + "\n" + "z" * 10_000
                 + "\n## " + "B" * 19_755 + "\n" + "q" * 20_000 + " TAIL", id="oscillating-headings"),
])
def test_truncated_head_tail_and_marker_share_one_budget(cap, content):
    path = "/project/AGENTS.md"
    result = _truncate_content(content, "AGENTS.md", max_chars=cap, read_path=path, queue_warning=False)
    assert len(result) <= cap
    if cap >= 40:
        assert "[...truncated AGENTS.md" in result
    if cap >= 200:
        assert path in result
    if cap >= 1000:
        marker = re.search(r"\n\n\[\.\.\.truncated .*?\]\n\n", result, re.DOTALL)
        assert marker is not None
        head, tail = result[:marker.start()], result[marker.end():]
        assert content.startswith(head) and content.endswith(tail)
        assert f"kept {len(head)}+{len(tail)} of {len(content)} chars" in marker.group()
        assert abs(len(head) * 2 - len(tail) * 7) <= 8
        assert cap - len(result) <= 2
    assert _truncate_content("😀" * cap, "AGENTS.md", max_chars=cap, queue_warning=False) == "😀" * cap


@pytest.mark.parametrize("loader", ["startup", "hint"])
def test_public_loaders_keep_the_cap_and_exact_recovery_path(tmp_path, monkeypatch, loader):
    root = tmp_path / "workspace"
    child = root / "child"
    child.mkdir(parents=True)
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    cap = 2500 if loader == "startup" else _MAX_HINT_CHARS
    (home / "config.yaml").write_text(json.dumps({"context_file_max_chars": cap}), encoding="utf-8")
    body = "HEAD " + "x" * 50_000 + " TAIL"
    (child / "AGENTS.md").write_text(body, encoding="utf-8")
    drain_truncation_warnings()
    if loader == "startup":
        result = build_context_files_prompt(cwd=str(child), skip_soul=True, home_override=home)
        rendered = result[result.index("## AGENTS.md"):]
        path = str(child / "AGENTS.md")
        assert drain_truncation_warnings()
    else:
        tracker = SubdirectoryHintTracker(working_dir=str(root))
        result = tracker.check_tool_call("read_file", {"path": str(child / "source.py")})
        prefix = "\n\n[Subdirectory context discovered: child/AGENTS.md]\n"
        assert result is not None and result.startswith(prefix)
        rendered = result[len(prefix):]
        path = "child/AGENTS.md"
        assert drain_truncation_warnings() == []
        assert tracker.check_tool_call("read_file", {"path": str(child / "other.py")}) is None
    assert cap - 2 <= len(rendered) <= cap
    assert "HEAD " in rendered and rendered.endswith(" TAIL")
    assert f"read_file tool: {path}]" in rendered
