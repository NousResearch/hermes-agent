"""A cron job that depends on the scratch dir is warned about at create and edit time.

Regression for #150365: scratch entries are pruned after 24h idle, so a weekly job whose
script reads from ``<HERMES_HOME>/cache/scratch`` (or whose workdir/interpreter lives there)
was saved silently and then failed between runs with FileNotFoundError. The warning is
advisory: the job is still created or updated.
"""

import json
import os
import sys
from types import SimpleNamespace

import pytest

from hermes_constants import get_hermes_home, get_scratch_dir
from tools.cronjob_tools import cronjob


@pytest.fixture(autouse=True)
def _cron_store(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")


def _script(name: str, body: str) -> str:
    scripts = get_hermes_home() / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / name).write_text(body, encoding="utf-8")
    return name


def _create(**kwargs) -> dict:
    return json.loads(cronjob(action="create", schedule="0 0 * * 6", **kwargs))


def test_create_warns_when_a_no_agent_script_reads_from_scratch():
    site = get_scratch_dir(prune=False) / "site"
    name = _script(
        "weekly.py", f'ROOT = "{site}"\nprint(open(ROOT + "/entries.json").read())\n'
    )

    created = _create(script=name, no_agent=True, deliver="local")

    assert created["success"] is True
    assert "script 'weekly.py'" in created["scratch_warning"]
    assert "24h" in created["scratch_warning"]


def test_update_warns_for_a_home_relative_script_and_a_scratch_workdir():
    created = _create(prompt="weekly report", deliver="local")
    assert "scratch_warning" not in created
    name = _script(
        "wrangle.sh", 'WRANGLER="$HERMES_HOME/cache/scratch/app/node_modules/.bin/x"\n'
    )
    workdir = get_scratch_dir(prune=False) / "checkout"
    workdir.mkdir()

    updated = json.loads(
        cronjob(
            action="update", job_id=created["job_id"], script=name, workdir=str(workdir)
        )
    )

    assert updated["success"] is True
    assert "script 'wrangle.sh'" in updated["scratch_warning"]
    assert "workdir" in updated["scratch_warning"]


@pytest.mark.platforms("posix")
def test_a_scratch_venv_interpreter_is_flagged_even_though_it_links_out_of_scratch():
    python = get_scratch_dir(prune=False) / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    os.symlink(sys.executable, python)

    created = _create(
        script=_script("job.py", "print('ok')\n"),
        prompt="x",
        interpreter=str(python),
        deliver="local",
    )

    assert created["success"] is True
    assert "interpreter" in created["scratch_warning"]


def test_persistent_paths_and_per_run_temp_files_are_not_flagged(tmp_path):
    body = (
        "import tempfile\n"
        "work = tempfile.mkdtemp()  # TMPDIR: per-run temp space is fine\n"
        'CACHE = "~/.cache/scratch/state.json"\n'
        'NOTES = "cache/scratchpad.md"\n'
    )
    created = _create(
        script=_script("ok.py", body),
        no_agent=True,
        workdir=str(tmp_path),
        deliver="local",
    )

    assert created["success"] is True
    assert "scratch_warning" not in created


def test_cli_create_prints_the_warning(capsys):
    from hermes_cli import cron as cron_cli

    name = _script("weekly.py", f'ROOT = "{get_scratch_dir(prune=False)}/site"\n')
    args = SimpleNamespace(
        schedule="0 0 * * 6",
        prompt=None,
        name=None,
        deliver="local",
        repeat=None,
        skill=None,
        skills=None,
        script=name,
        workdir=None,
        no_agent=True,
    )

    assert cron_cli.cron_create(args) == 0
    out = capsys.readouterr().out
    assert "Created job" in out
    assert "depends on the scratch dir" in out
