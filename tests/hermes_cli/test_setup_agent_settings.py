"""Tests for agent-settings copy in the interactive setup wizard."""

import pytest

from hermes_cli.setup import setup_agent_settings


def _run_agent_settings(monkeypatch, config: dict, *, prompt_answers, choice_fn):
    """Drive ``setup_agent_settings`` with canned line-prompt answers and ``choice_fn`` standing in
    for the curses menu. Returns the config snapshots handed to ``save_config``."""
    answers = iter(prompt_answers)
    saved: list[dict] = []
    monkeypatch.setattr("hermes_cli.setup.get_env_value", lambda key: "")
    monkeypatch.setattr("hermes_cli.setup.prompt", lambda *args, **kwargs: next(answers))
    monkeypatch.setattr("hermes_cli.setup.prompt_choice", choice_fn)
    monkeypatch.setattr("hermes_cli.setup.save_env_value", lambda *args, **kwargs: None)
    monkeypatch.setattr("hermes_cli.setup.remove_env_value", lambda key: True)
    monkeypatch.setattr("hermes_cli.setup.save_config", lambda cfg, *args, **kwargs: saved.append(dict(cfg)))
    setup_agent_settings(config)
    return saved


def test_setup_agent_settings_prefers_config_over_stale_env(tmp_path, monkeypatch, capsys):
    """Config.yaml wins even when a stale .env value disagrees.

    Regression guard for the bug where `.env HERMES_MAX_ITERATIONS=60`
    from an old `hermes setup` run shadowed `agent.max_turns: 500` in
    config.yaml. The wizard must now display the config value.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    config = {
        "agent": {"max_turns": 500},  # user bumped this in config.yaml
        "display": {"tool_progress": "all"},
        "compression": {"threshold": 0.50},
    }

    prompt_answers = iter(["500", "all", "0.5"])

    # Simulate stale .env value — the wizard must ignore this.
    monkeypatch.setattr(
        "hermes_cli.setup.get_env_value",
        lambda key: "60" if key == "HERMES_MAX_ITERATIONS" else "",
    )
    monkeypatch.setattr("hermes_cli.setup.prompt", lambda *args, **kwargs: next(prompt_answers))
    monkeypatch.setattr("hermes_cli.setup.prompt_choice", lambda question, choices, default=0, **kwargs: default)
    monkeypatch.setattr("hermes_cli.setup.save_env_value", lambda *args, **kwargs: None)

    removed_keys: list[str] = []
    monkeypatch.setattr(
        "hermes_cli.setup.remove_env_value",
        lambda key: (removed_keys.append(key), True)[1],
    )
    monkeypatch.setattr("hermes_cli.setup.save_config", lambda *args, **kwargs: None)

    setup_agent_settings(config)

    out = capsys.readouterr().out
    # Config value wins
    assert "Press Enter to keep 500." in out
    assert "Press Enter to keep 60." not in out
    # And the stale .env entry gets cleaned up
    assert "HERMES_MAX_ITERATIONS" in removed_keys


def test_agent_settings_offers_approval_mode_and_persists_choice(tmp_path, monkeypatch, capsys):
    """The wizard's Agent Settings section offers ``approvals.mode`` (port of goose#12629): the menu
    is preselected on the mode currently in force and the picked mode lands in the saved config."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    seen: dict = {}

    def pick_manual(question, choices, default=0, **kwargs):
        seen.update(question=question, choices=list(choices), default=default)
        return 0

    config = {"agent": {"max_turns": 90}, "approvals": {"mode": "smart"}}
    saved = _run_agent_settings(monkeypatch, config, prompt_answers=["90", "", "0.5"], choice_fn=pick_manual)

    assert seen["question"] == "Approval mode"
    assert [c.split(" ")[0] for c in seen["choices"]] == ["manual", "smart", "off"]
    assert seen["default"] == 1, "menu preselects the mode currently in force (smart)"
    assert saved[-1]["approvals"]["mode"] == "manual"
    assert "Approval mode set to: manual" in capsys.readouterr().out


@pytest.mark.parametrize("stored", ["manual", "off"])
def test_agent_settings_keeps_existing_approval_mode_on_enter(tmp_path, monkeypatch, stored):
    """Enter / Escape on the menu returns its default, and the wizard must then leave an existing
    ``approvals.mode`` untouched: re-running ``hermes setup agent`` never flips a chosen policy."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = {"agent": {"max_turns": 90}, "approvals": {"mode": stored, "deny": ["rm -rf *"]}}
    saved = _run_agent_settings(
        monkeypatch, config, prompt_answers=["90", "", "0.5"],
        choice_fn=lambda question, choices, default=0, **kwargs: default,
    )
    assert saved[-1]["approvals"] == {"mode": stored, "deny": ["rm -rf *"]}
