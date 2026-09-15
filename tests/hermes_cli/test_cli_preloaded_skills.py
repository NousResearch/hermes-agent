from __future__ import annotations

import importlib
import os
import sys
from unittest.mock import MagicMock, patch

import pytest


def _make_real_cli(**kwargs):
    clean_config = {
        "model": {
            "default": "anthropic/claude-opus-4.6",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "auto",
        },
        "display": {"compact": False, "tool_progress": "all"},
        "agent": {},
        "terminal": {"env_type": "local"},
    }
    clean_env = {"LLM_MODEL": "", "HERMES_MAX_ITERATIONS": ""}
    prompt_toolkit_stubs = {
        "prompt_toolkit": MagicMock(),
        "prompt_toolkit.history": MagicMock(),
        "prompt_toolkit.styles": MagicMock(),
        "prompt_toolkit.patch_stdout": MagicMock(),
        "prompt_toolkit.application": MagicMock(),
        "prompt_toolkit.layout": MagicMock(),
        "prompt_toolkit.layout.processors": MagicMock(),
        "prompt_toolkit.filters": MagicMock(),
        "prompt_toolkit.layout.dimension": MagicMock(),
        "prompt_toolkit.layout.menus": MagicMock(),
        "prompt_toolkit.widgets": MagicMock(),
        "prompt_toolkit.key_binding": MagicMock(),
        "prompt_toolkit.completion": MagicMock(),
        "prompt_toolkit.formatted_text": MagicMock(),
    }
    with patch.dict(sys.modules, prompt_toolkit_stubs), patch.dict(
        "os.environ", clean_env, clear=False
    ):
        import cli as cli_mod

        cli_mod = importlib.reload(cli_mod)
        with patch.object(cli_mod, "get_tool_definitions", return_value=[]), patch.dict(
            cli_mod.__dict__, {"CLI_CONFIG": clean_config}
        ):
            return cli_mod.HermesCLI(**kwargs)


class _DummyCLI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.session_id = "session-123"
        self.system_prompt = "base prompt"
        self.preloaded_skills = []

    def show_banner(self):
        return None

    def show_tools(self):
        return None

    def show_toolsets(self):
        return None

    def run(self):
        return None


def _real_finalize(cli_obj):
    """Call the real HermesCLI.finalize_preloaded_skills on a dummy object."""
    return _REAL_FINALIZE(cli_obj)


def _capture_real_finalize():
    import cli as cli_mod
    return cli_mod.HermesCLI.__dict__["finalize_preloaded_skills"]


_REAL_FINALIZE = _capture_real_finalize()


def test_main_applies_preloaded_skills_to_system_prompt(monkeypatch):
    import cli as cli_mod

    created = {}

    def fake_cli(**kwargs):
        created["cli"] = _DummyCLI(**kwargs)
        return created["cli"]

    monkeypatch.setattr(cli_mod, "HermesCLI", fake_cli)
    monkeypatch.setattr(
        cli_mod,
        "build_preloaded_skills_prompt",
        lambda skills, task_id=None: ("skill prompt", ["hermes-agent-dev", "github-auth"], []),
    )

    with pytest.raises(SystemExit):
        cli_mod.main(skills="hermes-agent-dev,github-auth", list_tools=True)

    cli_obj = created["cli"]
    # The preload now runs in a background thread and is folded in at agent
    # init via finalize_preloaded_skills() (startup-latency change). Drive
    # the finalize explicitly — the same call _init_agent makes.
    _real_finalize(cli_obj)
    assert cli_obj.system_prompt == "base prompt\n\nskill prompt"
    assert cli_obj.preloaded_skills == ["hermes-agent-dev", "github-auth"]


def test_main_raises_for_unknown_preloaded_skill(monkeypatch):
    import cli as cli_mod

    created = {}

    def fake_cli(**kwargs):
        created["cli"] = _DummyCLI(**kwargs)
        return created["cli"]

    monkeypatch.setattr(cli_mod, "HermesCLI", fake_cli)
    monkeypatch.setattr(
        cli_mod,
        "build_preloaded_skills_prompt",
        lambda skills, task_id=None: ("", [], ["missing-skill"]),
    )

    with pytest.raises(SystemExit):
        cli_mod.main(skills="missing-skill", list_tools=True)

    # The all-skills-unknown hard failure now surfaces when the preload is
    # finalized (agent init), preserving the fail-loud contract.
    with pytest.raises(ValueError, match=r"Unknown skill\(s\): missing-skill"):
        _real_finalize(created["cli"])


def test_dispatched_worker_survives_unknown_pinned_skill(monkeypatch, tmp_path):
    """A kanban card whose pins ALL fail to resolve must not kill its own worker.

    Regression: card t_bc0a9c07 pinned `seo`/`marketing` (category labels, not
    installed skills); agent init raised `Unknown skill(s)` and the worker
    exited before doing any work. In a dispatcher-owned worker context the pin
    is dropped with a warning plus a durable `skill_pin_unresolved` task event
    instead of aborting — the interactive fail-loud path above is unchanged.
    """
    import cli as cli_mod
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_skills as ks

    db_path = tmp_path / "kanban.db"
    kbc.init_db(db_path)
    with kbc.connect(db_path) as conn:
        tid = kb.create_task(conn, title="pinned-card", assignee="worker")

    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "jarvis-os")

    created = {}

    def fake_cli(**kwargs):
        created["cli"] = _DummyCLI(**kwargs)
        return created["cli"]

    monkeypatch.setattr(cli_mod, "HermesCLI", fake_cli)
    monkeypatch.setattr(
        cli_mod,
        "build_preloaded_skills_prompt",
        lambda skills, task_id=None: ("", [], ["seo", "marketing"]),
    )

    with pytest.raises(SystemExit):
        cli_mod.main(skills="seo,marketing", list_tools=True)

    cli_obj = created["cli"]
    assert ks.is_kanban_worker_context() is True
    _real_finalize(cli_obj)  # pre-fix: raises ValueError -> the worker dies
    assert cli_obj.system_prompt == "base prompt"
    assert cli_obj.preloaded_skills == []

    with kbc.connect(db_path) as conn:
        pinned = [e for e in kb.list_events(conn, tid) if e.kind == ks.SKILL_PIN_UNRESOLVED]
    assert len(pinned) == 1, "the dropped pin must leave a durable task event"
    assert pinned[0].payload["skills"] == ["seo", "marketing"]
    assert pinned[0].payload["phase"] == "preload"


def test_interactive_session_without_worker_context_still_fails_loud(monkeypatch):
    """The fail-soft is scoped to dispatched workers: a human typo stays loud."""
    import cli as cli_mod

    created = {}

    def fake_cli(**kwargs):
        created["cli"] = _DummyCLI(**kwargs)
        return created["cli"]

    monkeypatch.setattr(cli_mod, "HermesCLI", fake_cli)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(
        cli_mod,
        "build_preloaded_skills_prompt",
        lambda skills, task_id=None: ("", [], ["seo"]),
    )

    with pytest.raises(SystemExit):
        cli_mod.main(skills="seo", list_tools=True)

    with pytest.raises(ValueError, match=r"Unknown skill\(s\): seo"):
        _real_finalize(created["cli"])


def test_show_banner_does_not_print_skills():
    """show_banner() no longer prints the activated skills line — it moved to run()."""
    cli_obj = _make_real_cli(compact=False)
    cli_obj.preloaded_skills = ["hermes-agent-dev", "github-auth"]
    cli_obj.console = MagicMock()

    with patch("hermes_cli.banner.build_welcome_banner") as mock_banner, patch(
        "shutil.get_terminal_size", return_value=os.terminal_size((120, 40))
    ):
        cli_obj.show_banner()

    print_calls = [
        call.args[0]
        for call in cli_obj.console.print.call_args_list
        if call.args and isinstance(call.args[0], str)
    ]
    startup_lines = [line for line in print_calls if "Activated skills:" in line]
    assert len(startup_lines) == 0
    assert mock_banner.call_count == 1
