"""Plain-language contracts for hermes_state user-facing errors (CLI UX message campaign, cluster D)."""

import hermes_state
from hermes_state import format_session_db_unavailable








def test_db_unavailable_is_one_line_for_chat_surfaces_by_default():
    hermes_state._set_last_init_error("OperationalError: database is locked")
    try:
        text = format_session_db_unavailable(prefix="Cannot resume")
    finally:
        hermes_state._set_last_init_error(None)
    assert "\n" not in text
    assert "Details:" not in text
    assert text.startswith("Cannot resume:")






def test_db_unavailable_commands_are_pinned_to_the_failing_profile(monkeypatch, tmp_path):
    """Both fallbacks that bypass the shared cause table (no cause; network-drive gloss) name the
    profile whose store failed, like the table's actions do."""
    from hermes_constants import profile_cli_selector

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes" / "profiles" / "research"))
    selector = profile_cli_selector()
    assert selector.strip()
    hermes_state._set_last_init_error(None)
    no_cause = format_session_db_unavailable()
    hermes_state._set_last_init_error("OperationalError: locking protocol")
    try:
        network = format_session_db_unavailable()
    finally:
        hermes_state._set_last_init_error(None)
    for text in (no_cause, network):
        assert f"`hermes {selector}doctor`" in text and "`hermes doctor`" not in text
        assert "{profile_arg}" not in text
