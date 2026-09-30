import pytest
import os

@pytest.mark.parametrize("error_number,contended", [(13, True), (5, False)])
def test_scan_lock_classifies_os_errors(tmp_path, monkeypatch, error_number, contended):
    import github_pr_feedback.cli as cli
    if os.name == "nt":
        backend, operation = cli.msvcrt, "locking"
    else:
        backend, operation = cli.fcntl, "flock"
    def fail(*args):
        raise OSError(error_number, "synthetic lock error")
    monkeypatch.setattr(backend, operation, fail)
    if contended:
        with cli._exclusive_scan_lock(tmp_path) as acquired:
            assert acquired is False
    else:
        with pytest.raises(OSError) as caught:
            with cli._exclusive_scan_lock(tmp_path):
                pytest.fail("unexpected acquisition")
        assert caught.value.errno == error_number


def test_scan_lock_excludes_concurrent_scan_and_releases(tmp_path):
    from github_pr_feedback.cli import _exclusive_scan_lock
    with _exclusive_scan_lock(tmp_path) as first:
        assert first is True
        with _exclusive_scan_lock(tmp_path) as second:
            assert second is False
    with _exclusive_scan_lock(tmp_path) as after:
        assert after is True
