"""Signal interactive_setup routes the reconfigure gate through declines_reconfigure."""

import hermes_cli.setup_platforms as setup_platforms_mod
from plugins.platforms.signal.adapter import interactive_setup


def test_declining_reconfigure_goes_through_shared_gate(monkeypatch, tmp_path):
    import agent.secret_scope as ss
    import hermes_cli.cli_output as cli_output_mod
    import hermes_cli.config as config_mod
    import hermes_cli.setup as setup_mod

    ss.set_multiplex_active(False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("SIGNAL_HTTP_URL", "already-set")

    gated: list[tuple[str, ...]] = []
    real_gate = setup_platforms_mod.declines_reconfigure

    def _spy_gate(label, question, *env_vars):
        gated.append(env_vars)
        return real_gate(label, question, *env_vars)

    def _no_save(*_a, **_kw):
        raise AssertionError("wizard persisted env after the user declined to reconfigure")

    def _no_prompt(*_a, **_kw):
        raise AssertionError("wizard fell through to its own prompts after the user declined")

    monkeypatch.setattr(setup_platforms_mod, "declines_reconfigure", _spy_gate)
    monkeypatch.setattr(setup_mod, "prompt_yes_no", lambda *_a, **_kw: False)
    monkeypatch.setattr(cli_output_mod, "prompt_yes_no", lambda *_a, **_kw: False)
    for mod in (setup_mod, cli_output_mod):
        monkeypatch.setattr(mod, "prompt", _no_prompt)
        monkeypatch.setattr(mod, "save_env_value", _no_save, raising=False)
    monkeypatch.setattr(config_mod, "save_env_value", _no_save)

    interactive_setup()

    assert gated and "SIGNAL_HTTP_URL" in gated[0], f"gate not routed through declines_reconfigure: {gated}"
