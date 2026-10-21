"""`hermes setup`'s "Coming from another agent?" step: it offers only the agents found on disk, installs
the agent-importers plugin through the catalog install only after a yes, then runs that plugin's
import for each pick.

Real step code, a temp HERMES_HOME and a fake HOME; the terminal prompts, the catalog install (network)
and the child `hermes import-agent` process are stand-ins that record what the step asked for.
"""

from __future__ import annotations

import subprocess

import pytest

from hermes_cli import setup_agent_import as step


@pytest.fixture
def homes(tmp_path, monkeypatch):
    user_home, hermes_home = tmp_path / "user", tmp_path / "user" / ".hermes"
    hermes_home.mkdir(parents=True)
    for var in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(var, str(user_home))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("HERMES_NONINTERACTIVE", raising=False)
    calls: list = []
    answers = {"yes": True, "unpick": set()}

    def ask(question, default=True):
        calls.append(("ask", question, default))
        return answers["yes"]

    def checklist(title, items, pre_selected):
        calls.append(("pick", list(items), list(pre_selected)))
        return [i for i in pre_selected if items[i] not in answers["unpick"]]

    def install(name, enable=None, **_):
        calls.append(("install", name, enable))
        (hermes_home / "plugins" / name).mkdir(parents=True)

    def run(argv, env, check=False):
        calls.append(("import", argv[3:], env["HERMES_HOME"]))
        (hermes_home / "config.yaml").write_text(f"imported: {argv[4]}\n", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr("hermes_cli.setup.prompt_yes_no", ask)
    monkeypatch.setattr("hermes_cli.setup.prompt_checklist", checklist)
    monkeypatch.setattr("hermes_cli.plugins_cmd_install.cmd_install", install)
    monkeypatch.setattr(step.subprocess, "run", run)
    return user_home, hermes_home, calls, answers


def test_found_agent_is_offered_then_installed_from_the_catalog_and_imported(homes):
    user_home, hermes_home, calls, _answers = homes
    (user_home / ".openclaw").mkdir()

    assert step.offer_agent_import(hermes_home) == "OpenClaw"

    assert calls == [
        ("ask", "Import your setup from OpenClaw?", True),
        ("install", "agent-importers", True),
        # core's setup migrated OpenClaw's allowlisted secrets after the preview; so does this step
        ("import", ["import-agent", "openclaw", "--migrate-secrets"], str(hermes_home)),
    ]


def test_several_found_are_offered_as_a_checklist_and_only_the_picks_import(homes):
    user_home, hermes_home, calls, answers = homes
    for d in (".claude", ".cursor"):
        (user_home / d).mkdir()
    answers["unpick"] = {"Cursor"}

    assert step.offer_agent_import(hermes_home) == "Claude Code"

    assert calls[1] == ("pick", ["Claude Code", "Cursor"], [0, 1])  # never an agent that is absent
    assert calls[2:] == [("install", "agent-importers", True),
                         ("import", ["import-agent", "claude-code"], str(hermes_home))]


def test_declining_installs_nothing(homes):
    user_home, hermes_home, calls, answers = homes
    (user_home / ".codex").mkdir()
    answers["yes"] = False

    assert step.offer_agent_import(hermes_home) == ""

    assert calls == [("ask", "Import your setup from Codex CLI?", True)]
    assert not (hermes_home / "plugins").exists()


@pytest.mark.parametrize("headless", [True, False])
def test_headless_or_nothing_found_asks_nothing(headless, homes, monkeypatch):
    user_home, hermes_home, calls, _answers = homes
    if headless:  # dashboard/Desktop-spawned setup: no one can answer
        (user_home / ".claude").mkdir()
        monkeypatch.setenv("HERMES_NONINTERACTIVE", "1")

    assert step.offer_agent_import(hermes_home) == ""

    assert calls == []
    assert not (hermes_home / "plugins").exists()
