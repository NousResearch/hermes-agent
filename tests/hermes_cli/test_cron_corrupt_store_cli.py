"""``hermes cron <verb>`` on an unparseable ``jobs.json``.

``load_jobs()`` refuses such a store on purpose (#6797: a corrupt store is never treated as empty
and scheduled against). That refusal must reach the user as one error line and exit status 1, not as
the last line of a traceback (#136723), and the store must be left exactly as it was.
"""

from argparse import Namespace

import pytest

from hermes_cli.cron import cron_command

_TRUNCATED_WRITE = '{"jobs": '


@pytest.fixture()
def corrupt_cron_store(tmp_path, monkeypatch):
    cron_dir = tmp_path / "cron"
    cron_dir.mkdir()
    monkeypatch.setattr("cron.jobs.CRON_DIR", cron_dir)
    monkeypatch.setattr("cron.jobs.JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", cron_dir / "output")
    (cron_dir / "jobs.json").write_text(_TRUNCATED_WRITE, encoding="utf-8")
    return cron_dir / "jobs.json"


@pytest.mark.parametrize("verb", ["list", "status", "doctor"])
def test_corrupt_store_is_one_error_line_not_a_traceback(corrupt_cron_store, verb, capsys):
    rc = cron_command(Namespace(cron_command=verb, all=True, json=False))

    captured = capsys.readouterr()
    assert rc == 1
    assert "Cron database corrupted and unrepairable" in captured.err
    assert "Traceback" not in captured.out + captured.err
    # Fail closed: the refusal never rewrites or "repairs" the store behind the user's back.
    assert corrupt_cron_store.read_text(encoding="utf-8") == _TRUNCATED_WRITE
