"""Cadence failures retain the same warning shown to the operator."""
from unittest.mock import Mock

from hermes_cli.plugins_cadence import run_scheduled_check
from pm import receipt


def test_failed_check_records_warning_and_nonzero_exit(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    log = Mock()

    def fail(_):
        raise OSError("offline")

    run_scheduled_check(run_checks_fn=fail, plugins_dir=tmp_path / "plugins", log=log)
    saved = receipt.latest()
    assert saved["outcome"] == "failed"
    assert saved["exit_code"] != 0
    assert saved["warnings"][0]["message"] == log.warning.call_args_list[0].args[0]
