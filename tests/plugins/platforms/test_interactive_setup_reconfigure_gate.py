"""Every platform plugin's ``interactive_setup`` routes its "already configured? Reconfigure?"
gate through ``hermes_cli.setup_platforms.declines_reconfigure``.

The wizards used to hand-roll the gate (env check + info line + ``prompt_yes_no(..., False)``)
with drifting wording and, for two of them, raw ``input()`` loops. Fixes to the shared gate
(wording, default, non-interactive handling) only reach a wizard that actually calls it, so the
test spies on the real primitive: with the primary env var set and the user answering "no", the
wizard must have asked the shared gate about that env var and returned without persisting or
prompting anything else.
"""

import importlib

import pytest

import hermes_cli.setup_platforms as setup_platforms_mod

# (plugin module, primary env var the shared gate keys on)
_WIZARDS = [
    ("plugins.platforms.buzz.adapter", "BUZZ_RELAY_URL"),
    ("plugins.platforms.dingtalk.adapter", "DINGTALK_CLIENT_ID"),
    ("plugins.platforms.discord.adapter", "DISCORD_BOT_TOKEN"),
    ("plugins.platforms.feishu.adapter", "FEISHU_APP_ID"),
    ("plugins.platforms.google_chat.adapter", "GOOGLE_CHAT_SUBSCRIPTION_NAME"),
    ("plugins.platforms.irc.adapter", "IRC_SERVER"),
    ("plugins.platforms.line.adapter", "LINE_CHANNEL_ACCESS_TOKEN"),
    ("plugins.platforms.matrix.adapter", "MATRIX_PASSWORD"),
    ("plugins.platforms.mattermost.adapter", "MATTERMOST_TOKEN"),
    ("plugins.platforms.raft.adapter", "RAFT_PROFILE"),
    ("plugins.platforms.simplex.adapter", "SIMPLEX_WS_URL"),
    ("plugins.platforms.slack.adapter", "SLACK_BOT_TOKEN"),
    ("plugins.platforms.teams.adapter", "TEAMS_CLIENT_ID"),
    ("plugins.platforms.wecom.adapter", "WECOM_BOT_ID"),
]


@pytest.mark.parametrize("module_name, env_var", _WIZARDS, ids=[m.split(".")[-2] for m, _ in _WIZARDS])
def test_declining_reconfigure_goes_through_shared_gate(monkeypatch, tmp_path, module_name, env_var):
    import hermes_cli.cli_output as cli_output_mod
    import hermes_cli.config as config_mod
    import hermes_cli.setup as setup_mod

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv(env_var, "already-set")

    gated: list[tuple[str, ...]] = []
    real_gate = setup_platforms_mod.declines_reconfigure

    def _spy_gate(label, question, *env_vars):
        gated.append(env_vars)
        return real_gate(label, question, *env_vars)

    def _no_save(*_a, **_kw):
        raise AssertionError("wizard persisted env after the user declined to reconfigure")

    def _no_prompt(*_a, **_kw):
        raise AssertionError("wizard fell through to its own prompts after the user declined")

    # Wizards import the gate lazily from setup_platforms; the gate reads prompt_yes_no through
    # hermes_cli.setup. Everything else a wizard could do after declining must not happen.
    monkeypatch.setattr(setup_platforms_mod, "declines_reconfigure", _spy_gate)
    monkeypatch.setattr(setup_mod, "prompt_yes_no", lambda *_a, **_kw: False)
    monkeypatch.setattr(cli_output_mod, "prompt_yes_no", lambda *_a, **_kw: False)
    for mod in (setup_mod, cli_output_mod):
        monkeypatch.setattr(mod, "prompt", _no_prompt)
        monkeypatch.setattr(mod, "save_env_value", _no_save, raising=False)
    monkeypatch.setattr(config_mod, "save_env_value", _no_save)

    importlib.import_module(module_name).interactive_setup()

    assert gated and env_var in gated[0], f"{module_name}: gate not routed through declines_reconfigure: {gated}"
