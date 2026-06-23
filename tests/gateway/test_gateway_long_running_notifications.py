from gateway.run import _format_long_running_notification


def test_initial_long_running_notice_says_background_and_no_terminal():
    notice = _format_long_running_notification(
        elapsed_seconds=31,
        status_detail="iteration 2/90; terminal",
        initial=True,
    )

    assert "Long-running task" in notice
    assert "still working in this chat" in notice
    assert "Background status updates will stay here" in notice
    assert "no terminal check needed" in notice
    assert "/stop" in notice
    assert "Status: iteration 2/90; terminal." in notice


def test_followup_long_running_notice_is_short_and_keeps_status():
    notice = _format_long_running_notification(
        elapsed_seconds=185,
        status_detail="computer_use",
        initial=False,
    )

    assert notice == "⏳ Still working: 3 min. Status: computer_use."


def test_subminute_long_running_notice_uses_subminute_label():
    notice = _format_long_running_notification(
        elapsed_seconds=12,
        initial=True,
    )

    assert "for <1 min" in notice
