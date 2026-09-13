"""Tests for hermes_cli/moa_cmd.py — MoA (Mixture of Agents) command helpers."""


def test_moa_help_output_not_empty():
    from hermes_cli.moa_cmd import MOA_HELP_TEXT
    assert len(MOA_HELP_TEXT) > 0
    assert "preset" in MOA_HELP_TEXT.lower()


def test_all_actions_known():
    from hermes_cli.moa_cmd import ALL_ACTIONS
    assert isinstance(ALL_ACTIONS, (list, tuple, set))
    assert len(ALL_ACTIONS) >= 1
    for action in ALL_ACTIONS:
        assert isinstance(action, str)
        assert action
