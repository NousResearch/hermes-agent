"""CLI surface for the per-task reasoning effort (``--effort``).

The DB layer (``tasks.reasoning_effort``, ``set_reasoning_effort``), the
dispatcher passthrough (``--reasoning <level>``), and the dashboard REST
surface are covered by ``tests/plugins/test_kanban_model_override.py``.
This file covers the operator-facing CLI that drives them:

  * ``create --effort <level>`` persists (and rejects unknown levels).
  * ``set-model <id> --effort <level>`` sets the depth WITHOUT touching an
    existing model override (the independence contract).
  * ``set-model <id> --effort none`` pins thinking OFF (a value, not a clear).
  * ``set-model <id> --effort clear`` resets to the profile default.
  * ``set-model <id> none`` (model clear) leaves the effort intact.
  * ``show`` renders the effort line only when set; ``--json`` carries the
    field; ``list --json`` carries it too.
  * end-to-end: a CLI-set effort lands in the dispatcher's spawn argv.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _created_id(out: str) -> str:
    import re

    m = re.search(r"(t_[a-f0-9]+)", out)
    assert m, f"no task id in output: {out!r}"
    return m.group(1)


# ---------------------------------------------------------------------------
# create --effort
# ---------------------------------------------------------------------------


def test_create_with_effort_persists(kanban_home):
    out = kc.run_slash("create 'deep task' --assignee coder --effort xhigh")
    tid = _created_id(out)
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).reasoning_effort == "xhigh"


def test_create_effort_uppercase_normalized(kanban_home):
    """argparse lowercases via type=str.lower before the choices check."""
    out = kc.run_slash("create 'deep task' --assignee coder --effort XHIGH")
    tid = _created_id(out)
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).reasoning_effort == "xhigh"


def test_create_rejects_unknown_effort(kanban_home):
    out = kc.run_slash("create 'x' --assignee coder --effort turbo")
    assert "usage error" in out or "invalid choice" in out
    # Nothing was created.
    with kb.connect() as conn:
        assert kb.list_tasks(conn) == []


def test_create_without_effort_is_null(kanban_home):
    out = kc.run_slash("create 'plain' --assignee coder")
    tid = _created_id(out)
    with kb.connect() as conn:
        raw = conn.execute(
            "SELECT reasoning_effort FROM tasks WHERE id = ?", (tid,)
        ).fetchone()
    assert raw["reasoning_effort"] is None


# ---------------------------------------------------------------------------
# set-model --effort — set / none / clear / independence
# ---------------------------------------------------------------------------


def test_set_effort_alone_leaves_model_untouched(kanban_home):
    out = kc.run_slash(
        "create 'opus task' --assignee coder "
        "--model claude-opus-4.6 --provider anthropic"
    )
    tid = _created_id(out)
    res = kc.run_slash(f"set-model {tid} --effort xhigh")
    assert "Set reasoning effort" in res
    assert "model override" not in res  # the model knob was not touched
    with kb.connect() as conn:
        t = kb.get_task(conn, tid)
    assert t.reasoning_effort == "xhigh"
    assert t.model_override == "claude-opus-4.6"
    assert t.provider_override == "anthropic"


def test_set_effort_none_is_a_value(kanban_home):
    out = kc.run_slash("create 'x' --assignee coder")
    tid = _created_id(out)
    res = kc.run_slash(f"set-model {tid} --effort none")
    assert "Set reasoning effort" in res
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).reasoning_effort == "none"


def test_set_effort_clear_resets_to_profile(kanban_home):
    out = kc.run_slash("create 'x' --assignee coder --effort high")
    tid = _created_id(out)
    res = kc.run_slash(f"set-model {tid} --effort clear")
    assert "Cleared reasoning effort" in res
    with kb.connect() as conn:
        raw = conn.execute(
            "SELECT reasoning_effort FROM tasks WHERE id = ?", (tid,)
        ).fetchone()
    assert raw["reasoning_effort"] is None  # genuine NULL, not "clear"/"none"


def test_clearing_model_keeps_effort(kanban_home):
    out = kc.run_slash(
        "create 'x' --assignee coder --model glm-5 --effort max"
    )
    tid = _created_id(out)
    res = kc.run_slash(f"set-model {tid} none")
    assert "Cleared model override" in res
    with kb.connect() as conn:
        t = kb.get_task(conn, tid)
    assert t.model_override is None
    assert t.reasoning_effort == "max"


def test_set_model_and_effort_together(kanban_home):
    out = kc.run_slash("create 'x' --assignee coder")
    tid = _created_id(out)
    res = kc.run_slash(f"set-model {tid} claude-opus-4.6 --effort high")
    assert "Set model override" in res
    assert "Set reasoning effort" in res
    with kb.connect() as conn:
        t = kb.get_task(conn, tid)
    assert t.model_override == "claude-opus-4.6"
    assert t.reasoning_effort == "high"


def test_set_model_without_effort_still_clears(kanban_home):
    """The historical contract: bare ``set-model <id>`` clears the model."""
    out = kc.run_slash("create 'x' --assignee coder --model glm-5")
    tid = _created_id(out)
    res = kc.run_slash(f"set-model {tid}")
    assert "Cleared model override" in res
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).model_override is None


def test_set_effort_rejects_unknown_level(kanban_home):
    out = kc.run_slash("create 'x' --assignee coder --effort high")
    tid = _created_id(out)
    res = kc.run_slash(f"set-model {tid} --effort turbo")
    assert "usage error" in res or "invalid choice" in res
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).reasoning_effort == "high"  # unchanged


def test_set_effort_nonexistent_task(kanban_home):
    res = kc.run_slash("set-model t_nope --effort high")
    assert "no such task" in res


def test_provider_with_effort_but_no_model_rejected(kanban_home):
    out = kc.run_slash("create 'x' --assignee coder")
    tid = _created_id(out)
    res = kc.run_slash(f"set-model {tid} --provider anthropic --effort high")
    assert "--provider requires a model" in res
    with kb.connect() as conn:
        t = kb.get_task(conn, tid)
    assert t.provider_override is None
    assert t.reasoning_effort is None  # rejected before any write


# ---------------------------------------------------------------------------
# show / json output
# ---------------------------------------------------------------------------


def test_show_renders_effort_line(kanban_home):
    out = kc.run_slash("create 'x' --assignee coder --effort xhigh")
    tid = _created_id(out)
    show = kc.run_slash(f"show {tid}")
    assert "effort:" in show
    assert "xhigh" in show


def test_show_omits_effort_line_when_unset(kanban_home):
    out = kc.run_slash("create 'plain' --assignee coder")
    tid = _created_id(out)
    show = kc.run_slash(f"show {tid}")
    assert "effort:" not in show


def test_show_json_carries_reasoning_effort(kanban_home):
    out = kc.run_slash("create 'x' --assignee coder --effort minimal")
    tid = _created_id(out)
    payload = json.loads(kc.run_slash(f"show {tid} --json"))
    assert payload["task"]["reasoning_effort"] == "minimal"


def test_list_json_carries_reasoning_effort(kanban_home):
    kc.run_slash("create 'x' --assignee coder --effort low")
    rows = json.loads(kc.run_slash("list --json"))
    assert any(row.get("reasoning_effort") == "low" for row in rows)


# ---------------------------------------------------------------------------
# End-to-end: CLI write → store → reload → dispatcher spawn argv
# ---------------------------------------------------------------------------


def test_cli_set_effort_reaches_spawn_argv(kanban_home, monkeypatch):
    import subprocess

    out = kc.run_slash("create 'x' --assignee coder")
    tid = _created_id(out)
    kc.run_slash(f"set-model {tid} --effort xhigh")

    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.reasoning_effort == "xhigh"

    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    captured = {}

    class FakeProc:
        pid = 4247

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    workspace = kanban_home / "ws"
    workspace.mkdir(exist_ok=True)
    kb._default_spawn(task, str(workspace))
    cmd = captured["cmd"]
    i = cmd.index("--reasoning")
    assert cmd[i + 1] == "xhigh"
