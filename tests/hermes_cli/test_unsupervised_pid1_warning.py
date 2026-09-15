"""Tests for the PID-1-with-no-init warning (NousResearch/hermes-agent#111577).

When a deployment overrides the image's ``entrypoint:`` to invoke hermes
directly, ``docker/entrypoint-dispatch.sh`` never runs and hermes itself
becomes PID 1 with no supervisor above it to reap orphaned children —
they accumulate as zombies without bound. ``_warn_if_unsupervised_pid1``
surfaces that trap at startup, mirroring the warning
``entrypoint-dispatch.sh`` already prints on its own non-PID-1 fallback path.
"""

from hermes_cli.main import _warn_if_unsupervised_pid1


class TestWarnIfUnsupervisedPid1:
    def test_warns_when_pid_is_1_on_linux(self, monkeypatch, capsys):
        monkeypatch.setattr("platform.system", lambda: "Linux")
        _warn_if_unsupervised_pid1(pid=1)
        captured = capsys.readouterr()
        assert "PID 1" in captured.err
        assert "zombies" in captured.err

    def test_silent_when_not_pid_1(self, monkeypatch, capsys):
        monkeypatch.setattr("platform.system", lambda: "Linux")
        _warn_if_unsupervised_pid1(pid=4242)
        captured = capsys.readouterr()
        assert captured.err == ""
        assert captured.out == ""

    def test_silent_on_non_linux_even_as_pid_1(self, monkeypatch, capsys):
        monkeypatch.setattr("platform.system", lambda: "Darwin")
        _warn_if_unsupervised_pid1(pid=1)
        captured = capsys.readouterr()
        assert captured.err == ""
