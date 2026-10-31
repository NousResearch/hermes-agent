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
from hermes_cli.setup import prompt_yes_no as real_prompt_yes_no


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

    def ask(question, default=True, description=None):
        calls.append(("ask", question, default))
        return answers["yes"]

    def checklist(title, items, pre_selected):
        calls.append(("pick", list(items), list(pre_selected)))
        return [i for i in pre_selected if items[i] not in answers["unpick"]]

    def install(name, enable=None, brief=False, **_):
        calls.append(("install", name, enable, brief))
        (hermes_home / "plugins" / name).mkdir(parents=True)

    def run(argv, env, check=False):
        hermes_args = argv[argv.index("hermes_cli.main") + 1:]
        calls.append(("import", hermes_args, env["HERMES_HOME"]))
        (hermes_home / "config.yaml").write_text(f"imported: {hermes_args[-1]}\n", encoding="utf-8")
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
        ("install", "agent-importers", True, True),
        # core's setup migrated OpenClaw's allowlisted secrets after the preview; so does this step.
        # The root home is named explicitly, so a sticky active_profile cannot redirect the child.
        # --from-setup: the plugin words its preview and result as a step of this setup.
        ("import", ["--profile", "default", "import-agent", "openclaw", "--migrate-secrets", "--from-setup"],
         str(hermes_home)),
    ]


def test_several_found_are_offered_as_a_checklist_and_only_the_picks_import(homes):
    user_home, hermes_home, calls, answers = homes
    for d in (".claude", ".cursor"):
        (user_home / d).mkdir()
    answers["unpick"] = {"Cursor"}

    assert step.offer_agent_import(hermes_home) == "Claude Code"

    assert calls[1] == ("pick", ["Claude Code", "Cursor"], [0, 1])  # never an agent that is absent
    assert calls[2:] == [("install", "agent-importers", True, True),
                         ("import", ["--profile", "default", "import-agent", "claude-code", "--from-setup"],
                          str(hermes_home))]


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


def test_a_refused_enable_skips_the_import_and_setup_goes_on(homes, monkeypatch):
    """Installed but disabled, and PM cannot admit the enable: the error is printed, nothing is
    imported, and the step returns so the rest of setup still runs."""
    from hermes_cli.plugins_admission import AdmissionRefused
    user_home, hermes_home, calls, _answers = homes
    (user_home / ".cursor").mkdir()
    (hermes_home / "plugins" / "agent-importers").mkdir(parents=True)

    def refuse(name, **_):
        calls.append(("enable", name))
        raise AdmissionRefused("dependency conflict")
    monkeypatch.setattr("hermes_cli.plugins_cmd.cmd_enable", refuse)

    assert step.offer_agent_import(hermes_home) == ""

    assert calls == [("ask", "Import your setup from Cursor?", True), ("enable", "agent-importers")]


def test_setup_menu_shows_what_was_found_with_the_question(homes, monkeypatch):
    """Setup's yes/no is a full-screen menu that hides what was printed before it, so the detection
    lines travel with the question: "any of them" must point at text the user can see."""
    from hermes_cli import setup
    user_home, hermes_home, calls, _answers = homes
    for d in (".openclaw", ".claude", ".codex"):
        (user_home / d).mkdir()
    menus = []

    def radiolist(title, items, selected=0, *, description=None, **_):
        menus.append((title, items, description))
        return 1  # "No"
    monkeypatch.setattr("hermes_cli.setup.prompt_yes_no", real_prompt_yes_no)
    monkeypatch.setattr("hermes_cli.curses_ui.curses_radiolist", radiolist)

    with setup._setup_navigation_scope():
        assert step.offer_agent_import(hermes_home) == ""

    [(title, items, description)] = menus
    assert (title, items) == ("Import your setup from any of them?", ["Yes", "No"])
    assert ("Found on this computer: OpenClaw (~/.openclaw), Claude Code (~/.claude), Codex CLI (~/.codex)."
            in " ".join(description.split()))
    assert calls == [] and not (hermes_home / "plugins").exists()
