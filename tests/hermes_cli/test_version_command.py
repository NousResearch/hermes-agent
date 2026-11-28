"""Tests for the /version slash command."""

from unittest.mock import patch

from cli import HermesCLI






def test_process_command_version_prints_version_info():
    cli_obj = HermesCLI.__new__(HermesCLI)

    with patch("hermes_cli.main._print_version_info") as mock_print:
        assert cli_obj.process_command("/version") is True

    mock_print.assert_called_once_with(check_updates=True)


def test_global_fast_version_skips_update_check(monkeypatch):
    from hermes_cli import _startup_fast

    calls = []
    monkeypatch.setattr(_startup_fast, "container_mode_may_be_active", lambda: False)
    monkeypatch.setattr(
        _startup_fast,
        "print_fast_version_info",
        lambda **kwargs: calls.append(kwargs),
    )

    assert _startup_fast.try_fast_version(["--version"]) is True
    assert calls == [{"check_updates": False}]
