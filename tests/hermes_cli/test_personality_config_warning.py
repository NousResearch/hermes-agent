"""Configuration visibility for the personality/manual-prompt precedence."""


def test_display_section_warns_when_personality_overrides_manual_prompt(capsys):
    """A config listing must make the ignored manual prompt visible (#136882)."""
    from hermes_cli.config import _show_display_section

    _show_display_section({
        "display": {"personality": "concise"},
        "agent": {"system_prompt": "Always answer in the required format."},
    })

    output = capsys.readouterr().out
    assert "agent.system_prompt is ignored while personality 'concise' is active" in output
    assert "/personality none" in output


def test_display_section_omits_warning_without_manual_prompt(capsys):
    from hermes_cli.config import _show_display_section

    _show_display_section({"display": {"personality": "concise"}, "agent": {}})

    assert "agent.system_prompt is ignored" not in capsys.readouterr().out
