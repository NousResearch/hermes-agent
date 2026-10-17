"""#140706: the client-side peer check should explain an access-denied failure."""

from gateway import runtime_bootstrap_windows as rbw


def test_access_denied_mentions_higher_integrity():
    detail = rbw._open_process_error_detail(5)  # ERROR_ACCESS_DENIED
    assert "higher integrity" in detail
    assert "elevated" in detail


def test_other_errors_stay_generic():
    assert rbw._open_process_error_detail(2) == ""  # ERROR_FILE_NOT_FOUND: the peer is gone
    assert rbw._open_process_error_detail(0) == ""
