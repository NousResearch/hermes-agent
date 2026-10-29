"""``post_llm_call``'s ``review_signals`` (``agent/review_signals.py``).

Real turns through ``AIAgent.run_conversation`` with a plugin loaded through real discovery from a
temp ``HERMES_HOME``; the provider is a scripted client. The signals a plugin reads must be the
exact decision the finalizer then makes about the background review, turn after turn.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import hermes_yaml as yaml
import pytest

import hermes_cli.plugins as plugins_mod

_TOOLS = [
    {"type": "function", "function": {"name": name, "description": name,
                                      "parameters": {"type": "object", "properties": {}}}}
    for name in ("noop", "skill_manage", "memory")
]


def _install_recorder(rows: Path, *, mutate: bool = False) -> None:
    plugin_dir = Path(os.environ["HERMES_HOME"]) / "plugins" / "recorder"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(yaml.safe_dump({"name": "recorder"}), encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(
        "import json, pathlib\n\n"
        f"ROWS = pathlib.Path({str(rows)!r})\n\n"
        "def register(ctx):\n"
        "    def _vandal(review_signals=None, **_):\n"
        "        try:\n"
        "            review_signals['skill_review_due'] = 'tampered'\n"
        "        except TypeError:\n"
        "            pass\n"
        "    def _record(review_signals=None, **_):\n"
        "        with ROWS.open('a') as fh:\n"
        "            fh.write(json.dumps(dict(review_signals)) + '\\n')\n"
        + ("    ctx.register_hook('post_llm_call', _vandal)\n" if mutate else "")
        + "    ctx.register_hook('post_llm_call', _record)\n",
        encoding="utf-8",
    )
    (Path(os.environ["HERMES_HOME"]) / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": ["recorder"]}}), encoding="utf-8")
    plugins_mod._plugin_manager = plugins_mod.PluginManager()
    plugins_mod.discover_plugins(force=True)


def _reply(text=None, tool=None):
    calls = None
    if tool:
        calls = [SimpleNamespace(id=f"call_{tool}_{id(object())}", type="function",
                                 function=SimpleNamespace(name=tool, arguments="{}"))]
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=text, tool_calls=calls),
        finish_reason="tool_calls" if calls else "stop")], model="test/model", usage=None)


def _run_turns(monkeypatch, scripts, *, skip_background_review=False):
    """Run one turn per entry of ``scripts`` (a list of replies each); return the
    ``_spawn_background_review`` calls, one entry (or None) per turn."""
    from run_agent import AIAgent

    replies = [reply for turn in scripts for reply in turn]
    client = MagicMock()
    client.chat.completions.create.side_effect = lambda **kw: replies.pop(0)
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *a, **k: None)
    spawns: list = []
    with (
        patch("model_tools.get_tool_definitions", return_value=_TOOLS),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI", return_value=client),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        agent = AIAgent(api_key="k", base_url="https://openrouter.ai/api/v1", model="test/model",
                        quiet_mode=True, skip_context_files=True, skip_memory=True)
        agent._cached_system_prompt = "You are helpful."
        agent._memory_store = object()  # the built-in store the memory nudge needs
        agent._memory_nudge_interval = 2
        agent._skill_nudge_interval = 2
        agent.skip_background_review = skip_background_review
        agent._spawn_background_review = lambda **kw: spawns.append(kw)
        history: list = []
        per_turn = []
        for n, _turn in enumerate(scripts):
            before = len(spawns)
            result = agent.run_conversation(f"turn {n}", conversation_history=history)
            history = result["messages"]
            per_turn.append(spawns[before] if len(spawns) > before else None)
    assert replies == []
    return per_turn


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_review_signals_track_the_review_decision_across_turns(monkeypatch):
    rows_path = Path(os.environ["HERMES_HOME"]) / "signals.jsonl"
    _install_recorder(rows_path)
    tool_turn = [_reply(tool="noop"), _reply(tool="noop"), _reply("done")]
    spawns = _run_turns(monkeypatch, [[_reply("a")], tool_turn, [_reply("b")], [_reply("c")]])
    rows = _rows(rows_path)

    assert len(rows) == 4
    for row, spawn in zip(rows, spawns):
        assert row["skill_nudge_interval"] == row["memory_nudge_interval"] == 2
        assert row["background_review_enabled"] is True
        # The hook saw exactly what the finalizer then decided.
        assert (spawn is not None) == (row["skill_review_due"] or row["memory_review_due"])
        if spawn is not None:
            assert spawn["review_skills"] == row["skill_review_due"]
            assert spawn["review_memory"] == row["memory_review_due"]
        assert row["skill_review_due"] == (
            row["tool_iterations_since_skill_review"] >= row["skill_nudge_interval"])

    # Memory: every second user turn; the counter was reset when it fired.
    assert [r["memory_review_due"] for r in rows] == [False, True, False, True]
    assert [r["user_turns_since_memory_review"] for r in rows] == [1, 0, 1, 0]
    # Skills: every loop iteration counts (the tool turn makes three); the counter restarts once
    # the review fired.
    assert [r["tool_iterations_since_skill_review"] for r in rows] == [1, 4, 1, 2]
    assert [r["skill_review_due"] for r in rows] == [False, True, False, True]


def test_review_signals_report_a_suppressed_review(monkeypatch):
    rows_path = Path(os.environ["HERMES_HOME"]) / "signals.jsonl"
    _install_recorder(rows_path)
    spawns = _run_turns(monkeypatch, [[_reply("a")], [_reply("b")]], skip_background_review=True)
    rows = _rows(rows_path)
    assert [r["memory_review_due"] for r in rows] == [False, True]
    assert all(r["background_review_enabled"] is False for r in rows)
    assert spawns == [None, None]


def test_review_signals_disabled_in_config(monkeypatch):
    rows_path = Path(os.environ["HERMES_HOME"]) / "signals.jsonl"
    _install_recorder(rows_path)
    cfg = Path(os.environ["HERMES_HOME"]) / "config.yaml"
    data = yaml.safe_load(cfg.read_text())
    data["auxiliary"] = {"background_review": {"enabled": False}}
    cfg.write_text(yaml.safe_dump(data), encoding="utf-8")
    _run_turns(monkeypatch, [[_reply("a")]])
    assert _rows(rows_path)[0]["background_review_enabled"] is False


def test_one_plugin_cannot_change_what_the_next_one_reads(monkeypatch):
    rows_path = Path(os.environ["HERMES_HOME"]) / "signals.jsonl"
    _install_recorder(rows_path, mutate=True)
    _run_turns(monkeypatch, [[_reply("a")]])
    assert _rows(rows_path)[0]["skill_review_due"] is False


@pytest.fixture(autouse=True)
def _reset_plugins():
    yield
    plugins_mod._plugin_manager = None
