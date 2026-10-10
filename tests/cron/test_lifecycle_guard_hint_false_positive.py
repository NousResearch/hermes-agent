"""Regression for #106723: referenced script hints are not executable commands."""

from cron.lifecycle_guard import contains_gateway_lifecycle_command_or_referenced_script


def test_referenced_script_comment_hint_is_not_blocked(tmp_path):
    script = tmp_path / "hint.sh"
    script.write_text("# To recover, run hermes gateway restart\necho safe\n")

    assert not contains_gateway_lifecycle_command_or_referenced_script(f"bash {script}")


def test_referenced_script_real_lifecycle_command_is_still_blocked(tmp_path):
    script = tmp_path / "restart.sh"
    script.write_text("hermes gateway restart\n")

    assert contains_gateway_lifecycle_command_or_referenced_script(f"bash {script}")


def test_referenced_script_indented_comment_hint_is_not_blocked(tmp_path):
    script = tmp_path / "indented-hint.sh"
    script.write_text("  # hermes gateway restart\necho safe\n")

    assert not contains_gateway_lifecycle_command_or_referenced_script(f"bash {script}")


def test_referenced_script_inline_lifecycle_command_is_still_blocked(tmp_path):
    script = tmp_path / "inline-command.sh"
    script.write_text("echo safe # hermes gateway restart\n")

    assert contains_gateway_lifecycle_command_or_referenced_script(f"bash {script}")
