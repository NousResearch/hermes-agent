"""Behavior coverage for update notices about retained skill edits."""

from hermes_cli.update_cmd import _print_user_modified_skills_notice


def test_user_modified_notice_explains_inspect_and_restore_commands(capsys):
    _print_user_modified_skills_notice(
        {"user_modified": ["edited-one", "edited-two"]}
    )

    output = capsys.readouterr().out
    assert "2 user-modified (kept)" in output
    assert "hermes skills diff <name>" in output
    assert "hermes skills reset <name> --restore" in output
    assert "plain reset" not in output


def test_user_modified_notice_is_silent_when_nothing_was_kept(capsys):
    _print_user_modified_skills_notice({"user_modified": []})

    assert capsys.readouterr().out == ""
