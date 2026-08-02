from __future__ import annotations

import json
import subprocess
import sys


def _run_driver(
    argvs: list[list[str]], hermes_home
) -> subprocess.CompletedProcess[str]:
    driver = r"""
import json
import sys
from pathlib import Path

import cron.jobs as jobs
import hermes_cli.main as main_mod

base = Path(sys.argv[1])
jobs.CRON_DIR = base / "cron"
jobs.JOBS_FILE = jobs.CRON_DIR / "jobs.json"
jobs.OUTPUT_DIR = jobs.CRON_DIR / "output"

results = []
for argv in json.loads(sys.argv[2]):
    sys.argv = ["hermes", *argv]
    try:
        rc = main_mod.main()
        code = int(rc or 0)
    except SystemExit as exc:
        code = int(exc.code or 0)
    results.append({"argv": argv, "code": code})
print(json.dumps(results))
"""
    return subprocess.run(
        [sys.executable, "-c", driver, str(hermes_home), json.dumps(argvs)],
        capture_output=True,
        text=True,
        timeout=120,
        env={
            "HERMES_HOME": str(hermes_home),
            "HOME": str(hermes_home / "host-home"),
            "PATH": "",
            "PYTHONPATH": ".",
        },
    )


def _parsed_results(result: subprocess.CompletedProcess[str]) -> list[dict]:
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_main_doctor_target_missing_returns_one(tmp_path):
    home = tmp_path / "hermes-home"
    (home / "host-home").mkdir(parents=True)

    result = _run_driver([["doctor", "skill", "missing-skill"]], home)

    assert _parsed_results(result)[0]["code"] == 1


def test_main_doctor_invalid_target_selection_returns_two(tmp_path):
    home = tmp_path / "hermes-home"
    (home / "host-home").mkdir(parents=True)

    result = _run_driver([["doctor", "skill"]], home)

    assert _parsed_results(result)[0]["code"] == 2


def test_main_cron_strict_preflight_failure_returns_one_after_save(tmp_path):
    home = tmp_path / "hermes-home"
    (home / "host-home").mkdir(parents=True)

    result = _run_driver(
        [
            [
                "cron",
                "create",
                "every 1h",
                "Run",
                "--name",
                "Strict Fail",
                "--strict-preflight",
                "--skill",
                "missing-skill",
            ]
        ],
        home,
    )

    assert _parsed_results(result)[0]["code"] == 1
    assert "Created job:" in result.stdout
    assert "Job saved. Preflight failed." in result.stdout
