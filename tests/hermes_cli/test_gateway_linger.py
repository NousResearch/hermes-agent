"""Tests for _ensure_linger_enabled() — headless server linger detection."""
import subprocess
from unittest.mock import patch, MagicMock
from pathlib import Path


def _get_fn():
    from hermes_cli.gateway import _ensure_linger_enabled
    return _ensure_linger_enabled


class TestEnsureLingerEnabled:
    def test_linger_already_enabled_via_file(self, tmp_path, capsys):
        """If linger file exists, prints success and skips loginctl."""
        linger_file = tmp_path / "testuser"
        linger_file.touch()

        with patch("hermes_cli.gateway.Path", side_effect=lambda p: Path(p).with_name(p.split("/")[-1])):
            pass  # just verify the import works

        fn = _get_fn()
        with patch("getpass.getpass", return_value="testuser"), \
             patch("getpass.getuser", return_value="testuser"), \
             patch("hermes_cli.gateway.Path") as mock_path, \
             patch("subprocess.run") as mock_run:
            mock_linger = MagicMock()
            mock_linger.exists.return_value = True
            mock_path.return_value = mock_linger
            fn()
            mock_run.assert_not_called()

        out = capsys.readouterr().out
        assert "already enabled" in out

    def test_linger_not_enabled_loginctl_succeeds(self, capsys):
        """If linger file missing but loginctl succeeds, prints success."""
        fn = _get_fn()
        with patch("getpass.getuser", return_value="testuser"), \
             patch("hermes_cli.gateway.Path") as mock_path, \
             patch("subprocess.run") as mock_run:
            mock_linger = MagicMock()
            mock_linger.exists.return_value = False
            mock_path.return_value = mock_linger
            mock_run.return_value = MagicMock(returncode=0)
            fn()

        out = capsys.readouterr().out
        assert "✓ Linger enabled" in out
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "loginctl" in args
        assert "enable-linger" in args
        assert "testuser" in args

    def test_linger_not_enabled_loginctl_fails_shows_warning(self, capsys):
        """If loginctl fails (no sudo), prints clear actionable warning."""
        fn = _get_fn()
        with patch("getpass.getuser", return_value="testuser"), \
             patch("hermes_cli.gateway.Path") as mock_path, \
             patch("subprocess.run") as mock_run:
            mock_linger = MagicMock()
            mock_linger.exists.return_value = False
            mock_path.return_value = mock_linger
            mock_run.return_value = MagicMock(returncode=1, stderr="Permission denied")
            fn()

        out = capsys.readouterr().out
        assert "sudo loginctl enable-linger testuser" in out
        assert "STOP" in out

    def test_linger_called_during_install(self):
        """_ensure_linger_enabled is called inside systemd_install."""
        import inspect
        import hermes_cli.gateway as gw
        src = inspect.getsource(gw.systemd_install)
        assert "_ensure_linger_enabled" in src
