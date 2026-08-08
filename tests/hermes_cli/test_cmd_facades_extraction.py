"""Seam tests for the R5-S1 cmd-facade extraction (main.py god-file slice).

Verifies:
  1. Identity re-export: ``hermes_cli.main.cmd_X is hermes_cli.cmd_facades.cmd_X``
     for all 10 moved facades.
  2. Delegation: each facade dispatches to its target implementation module.
  3. ``_require_tty`` lazy seam: the two facades that use it resolve it from
     ``hermes_cli.main`` at call time (keeps the 6 test patch sites on
     ``hermes_cli.main._require_tty`` working).
"""

import importlib

import pytest

import hermes_cli.cmd_facades as cf
import hermes_cli.main as main

FACADES = [
    "cmd_memory",
    "cmd_acp",
    "cmd_tools",
    "cmd_insights",
    "cmd_monitoring",
    "cmd_skills",
    "cmd_pairing",
    "cmd_plugins",
    "cmd_mcp",
    "cmd_claw",
]


@pytest.mark.parametrize("name", FACADES)
def test_identity_reexport(name):
    """main re-exports the exact same callable objects from cmd_facades."""
    main_attr = getattr(main, name)
    cf_attr = getattr(cf, name)
    assert main_attr is cf_attr
    assert callable(main_attr)


def test_module_docstring_links_issue():
    assert "#78647" in cf.__doc__ and "#78631" in cf.__doc__


# --- delegation smokes: facade -> target implementation ---------------------


def test_cmd_tools_delegates_to_tools_config(monkeypatch):
    import hermes_cli.tools_config as tools_config

    seen = {}

    def fake(args):
        seen["args"] = args

    monkeypatch.setattr(tools_config, "tools_disable_enable_command", fake)
    monkeypatch.setattr(main, "_require_tty", lambda *a: None)
    args = type("Args", (), {"tools_action": "list"})()
    main.cmd_tools(args)
    assert seen.get("args") is args


def test_cmd_skills_delegates_to_skills_hub(monkeypatch):
    import hermes_cli.skills_hub as skills_hub

    seen = {}

    def fake(args):
        seen["args"] = args

    monkeypatch.setattr(skills_hub, "skills_command", fake)
    args = type("Args", (), {"skills_action": None})()
    main.cmd_skills(args)
    assert seen.get("args") is args


def test_cmd_plugins_delegates_to_plugins_cmd(monkeypatch):
    import hermes_cli.plugins_cmd as plugins_cmd

    seen = {}

    def fake(args):
        seen["args"] = args

    monkeypatch.setattr(plugins_cmd, "plugins_command", fake)
    args = type("Args", (), {})()
    main.cmd_plugins(args)
    assert seen.get("args") is args


def test_cmd_pairing_delegates_to_pairing(monkeypatch):
    import hermes_cli.pairing as pairing

    seen = {}

    def fake(args):
        seen["args"] = args

    monkeypatch.setattr(pairing, "pairing_command", fake)
    args = type("Args", (), {})()
    main.cmd_pairing(args)
    assert seen.get("args") is args


def test_cmd_mcp_delegates_to_mcp_config(monkeypatch):
    import hermes_cli.mcp_config as mcp_config

    seen = {}

    def fake(args):
        seen["args"] = args

    monkeypatch.setattr(mcp_config, "mcp_command", fake)
    args = type("Args", (), {})()
    main.cmd_mcp(args)
    assert seen.get("args") is args


def test_cmd_claw_delegates_to_claw(monkeypatch):
    import hermes_cli.claw as claw

    seen = {}

    def fake(args):
        seen["args"] = args

    monkeypatch.setattr(claw, "claw_command", fake)
    args = type("Args", (), {})()
    main.cmd_claw(args)
    assert seen.get("args") is args


def test_cmd_insights_uses_sessiondb(monkeypatch):
    """cmd_insights instantiates SessionDB + InsightsEngine and prints a report."""
    import hermes_state

    class FakeDB:
        def close(self):
            pass

    class FakeEngine:
        def __init__(self, db):
            self.db = db

        def generate(self, days, source):
            return {"fake": "report"}

        def format_terminal(self, report):
            return "formatted-report"

    monkeypatch.setattr(hermes_state, "SessionDB", FakeDB)
    monkeypatch.setattr(
        "agent.insights.InsightsEngine", FakeEngine, raising=False
    )
    captured = {}

    def fake_print(*a, **k):
        captured["printed"] = a[0]

    monkeypatch.setattr("builtins.print", fake_print)
    args = type("Args", (), {"days": 7, "source": "all"})()
    main.cmd_insights(args)
    assert captured.get("printed") == "formatted-report"


# --- _require_tty lazy seam -------------------------------------------------


def test_cmd_tools_require_tty_lazy_seam(monkeypatch):
    """cmd_tools resolves _require_tty from main at call time (patch site works)."""
    import hermes_cli.tools_config as tools_config

    calls = []

    def fake_tty(label):
        calls.append(label)

    monkeypatch.setattr(main, "_require_tty", fake_tty)
    monkeypatch.setattr(
        tools_config, "tools_command", lambda args: None
    )
    args = type("Args", (), {"tools_action": None})()
    main.cmd_tools(args)
    assert calls == ["tools"]


def test_cmd_skills_require_tty_lazy_seam(monkeypatch):
    """cmd_skills config branch resolves _require_tty from main at call time."""
    import hermes_cli.skills_config as skills_config

    calls = []

    def fake_tty(label):
        calls.append(label)

    monkeypatch.setattr(main, "_require_tty", fake_tty)
    monkeypatch.setattr(
        skills_config, "skills_command", lambda args: None
    )
    args = type("Args", (), {"skills_action": "config"})()
    main.cmd_skills(args)
    assert calls == ["skills config"]


def test_no_reverse_module_level_import():
    """cmd_facades must not import hermes_cli.main at module level (cycle guard)."""
    import ast

    src = importlib.import_module("hermes_cli.cmd_facades").__file__
    tree = ast.parse(open(src, encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "hermes_cli.main":
            # in-function lazy imports are the sanctioned seam; module-level is a cycle
            assert not isinstance(
                node, ast.ImportFrom
            ) or node.col_offset > 0, "module-level reverse import found"
