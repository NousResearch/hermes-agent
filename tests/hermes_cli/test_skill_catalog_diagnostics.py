"""Diagnostics cannot become terminal output or retain per-profile file handles."""
import logging.handlers

import psutil

from hermes_cli.skill_catalog import diagnostics


def test_diagnostic_write_releases_profile_file_descriptor(tmp_path):
    home = tmp_path / 'profile'
    target = home / 'logs/skill-catalog.log'
    diagnostics.write(home, 'fixture diagnostic')
    assert target.read_text().endswith('fixture diagnostic\n')
    assert str(target) not in {row.path for row in psutil.Process().open_files()}


def test_diagnostic_io_failure_never_falls_back_to_stderr(tmp_path, monkeypatch, capfd):
    def failure(self, record):
        raise OSError('fixture private diagnostic path')

    monkeypatch.setattr(logging.handlers.RotatingFileHandler, 'shouldRollover', failure)
    diagnostics.write(tmp_path / 'profile', 'fixture private text')
    output = capfd.readouterr()
    assert output.out == ''
    assert output.err == ''
