"""Without dedicated Slack tools the prompt must not deny approved helper routes (e.g. post readback)."""

from gateway import session


def test_no_tools_note_points_to_approved_routes_not_a_blanket_denial():
    note = session._SLACK_NO_TOOLS_NOTE.lower()
    assert "you do not have access to slack-specific apis" not in note
    assert "cannot call slack apis yourself" not in note
    assert "inspect the loaded capabilities" in note
    assert "approved cli, helper, or plugin" in note
    assert "do not use raw slack credentials" in note
    assert "undocumented api calls" in note
