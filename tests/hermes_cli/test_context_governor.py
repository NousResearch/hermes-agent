from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import subprocess


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def test_repo_handoff_writes_current_md_with_bounded_git_state(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    from hermes_cli.context_governor import create_handoff

    repo = tmp_path / "repo"
    subdir = repo / "pkg"
    subdir.mkdir(parents=True)
    _git(repo, "init")
    tracked = repo / "app.py"
    tracked.write_text("print('hello')\n", encoding="utf-8")
    _git(repo, "add", "app.py")
    tracked.write_text("print('changed')\n", encoding="utf-8")
    (repo / "new.txt").write_text("new\n", encoding="utf-8")

    result = create_handoff(
        cwd=subdir,
        session_id="sess-123",
        task_goal="Build Context Governor",
        reason="manual-test",
        now=datetime(2026, 6, 17, 12, 30, tzinfo=timezone.utc),
        state_db_path=tmp_path / "missing-state.db",
    )

    assert result.ok
    assert result.path == repo / "CURRENT.md"
    content = result.path.read_text(encoding="utf-8")
    assert "# CURRENT — Hermes Context Governor Handoff" in content
    assert "- Session ID: `sess-123`" in content
    assert f"- CWD: `{subdir}`" in content
    assert f"- Repo root: `{repo}`" in content
    assert "- Reason: manual-test" in content
    assert "Build Context Governor" in content
    assert "app.py" in content
    assert "new.txt" in content
    assert "Resume this Hermes work" in content


def test_non_repo_handoff_uses_global_fallback(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli.context_governor import create_handoff

    work = tmp_path / "not-a-repo"
    work.mkdir()
    result = create_handoff(
        cwd=work,
        session_id="sess/global",
        task_goal=None,
        now=datetime(2026, 6, 17, 12, 31, tzinfo=timezone.utc),
        state_db_path=tmp_path / "missing.db",
    )

    assert result.ok
    assert result.path.parent == home / "handoffs"
    assert result.path.name.endswith("sess-global.md")
    content = result.path.read_text(encoding="utf-8")
    assert "- Repo root: `(not detected)`" in content
    assert f"- CWD: `{work}`" in content


def test_session_db_hints_extract_goal_and_verification_commands(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_state import SessionDB
    from hermes_cli.context_governor import create_handoff

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.create_session("sess-hints", source="cli")
    db.append_message("sess-hints", "user", "Build the backend handoff writer")
    db.append_message(
        "sess-hints",
        "assistant",
        "running tests",
        tool_calls=[{"function": {"name": "terminal", "arguments": '{"command":"python -m pytest tests/hermes_cli/test_context_governor.py -q"}'}}],
    )
    db.append_message("sess-hints", "tool", '{"output":"passed"}', tool_name="terminal")
    db.close()

    work = tmp_path / "work"
    work.mkdir()
    result = create_handoff(
        cwd=work,
        session_id="sess-hints",
        now=datetime(2026, 6, 17, 12, 32, tzinfo=timezone.utc),
        state_db_path=db_path,
    )

    content = result.path.read_text(encoding="utf-8")
    assert "Build the backend handoff writer" in content
    assert "python -m pytest tests/hermes_cli/test_context_governor.py -q" in content


def test_git_status_parser_keeps_first_character_for_staged_paths(monkeypatch, tmp_path):
    from hermes_cli import context_governor

    def fake_git(_repo, *args):
        if args == ("branch", "--show-current"):
            return "main"
        if args == ("status", "--short"):
            return "M gateway/platforms/telegram.py\n M tools/web_tools.py"
        return None

    monkeypatch.setattr(context_governor, "_run_git", fake_git)

    info = context_governor._collect_git_info(tmp_path)

    assert "gateway/platforms/telegram.py" in info["changed_files"]
    assert "ateway/platforms/telegram.py" not in info["changed_files"]
    assert "tools/web_tools.py" in info["changed_files"]


def test_latest_handoff_prefers_repo_current_over_global(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli.context_governor import find_latest_handoff

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    current = repo / "CURRENT.md"
    current.write_text("# CURRENT\n", encoding="utf-8")
    global_dir = home / "handoffs"
    global_dir.mkdir(parents=True)
    global_handoff = global_dir / "20990101T000000Z-other.md"
    global_handoff.write_text("global\n", encoding="utf-8")

    latest = find_latest_handoff(cwd=repo)

    assert latest == current


def test_context_status_marks_dirty_repo_with_stale_handoff_unsafe(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli.context_governor import get_context_status

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    current = repo / "CURRENT.md"
    current.write_text("old handoff\n", encoding="utf-8")
    changed = repo / "work.py"
    changed.write_text("new work\n", encoding="utf-8")

    status = get_context_status(cwd=repo, state_db_path=tmp_path / "missing.db")

    assert status["latest_handoff_path"] == str(current)
    assert status["handoff_recommended"] is True
    assert status["fresh_session_safe"] is False
    assert "changed files newer than latest handoff" in status["fresh_session_reason"]


def test_fresh_session_prompt_and_command_are_shell_safe(tmp_path):
    from hermes_cli.context_governor import build_fresh_session_prompt, build_fresh_session_command

    handoff = tmp_path / "handoff `weird` $(rm -rf nope).md"
    handoff.write_text("## Next-session prompt\n```text\nContinue safely.\n```\n", encoding="utf-8")

    prompt = build_fresh_session_prompt(handoff)
    command = build_fresh_session_command(handoff)

    assert str(handoff) in prompt
    assert "Continue safely." in prompt
    assert "hermes chat -q" in command
    assert "$(rm -rf nope)" in command
    assert "'" in command  # shell-quoted, not raw executable interpolation


def test_quality_report_compares_handoff_sessions_to_monster_sessions(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_state import SessionDB
    from hermes_cli.context_governor import get_quality_report

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.create_session("monster", source="cli")
    db.append_message("monster", "user", "continue giant session")
    db.update_token_counts("monster", input_tokens=100_000, output_tokens=5_000, cache_read_tokens=400_000, api_call_count=5, absolute=True)
    db.create_session("handoff", source="cli")
    db.append_message("handoff", "user", "[Hermes Context Governor fresh-session handoff] Read /tmp/CURRENT.md and continue")
    db.update_token_counts("handoff", input_tokens=20_000, output_tokens=3_000, cache_read_tokens=40_000, api_call_count=4, absolute=True)
    db.close()

    report = get_quality_report(state_db_path=db_path, days=30)

    assert report["handoff_sessions"]["session_count"] == 1
    assert report["monster_sessions"]["session_count"] == 1
    assert report["handoff_sessions"]["avg_tokens_per_api_call"] < report["monster_sessions"]["avg_tokens_per_api_call"]


def test_context_governor_plugin_registers_hook_and_checkpoint_command(monkeypatch):
    from plugins.context_governor import register

    calls = []

    class FakeContext:
        def register_hook(self, name, callback):
            calls.append(("hook", name, callback))

        def register_command(self, name, handler, description="", args_hint=""):
            calls.append(("command", name, handler, description, args_hint))

        def register_cli_command(self, name, help, setup_fn, handler_fn=None, description=""):
            calls.append(("cli", name, setup_fn, handler_fn, help, description))

    register(FakeContext())

    assert any(kind == "hook" and name == "pre_context_compress" for kind, name, *_ in calls)
    assert any(kind == "command" and name == "checkpoint" for kind, name, *_ in calls)
    assert any(kind == "cli" and name == "checkpoint" for kind, name, *_ in calls)
