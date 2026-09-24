"""Entrypoint driver for the parity matrix: cron run-now (``hermes cron run <id>``).

Follows the driver contract in ``_drive_cli``. ``hermes cron create`` writes the
job (``--deliver local``: no platform; ``--workdir``: cron's documented
per-job cwd/context channel, see ``hermes cron create --help``), then
``hermes cron run <id>`` executes it: the CLI waits synchronously
(``hermes_cli/cron.py::_job_action`` forces the synchronous path) while the turn
runs in the profile's gateway daemon (``cron.scheduler_authority.run_canonical_job``
→ ``connect_gateway``, which spawns the daemon when none owns the profile) — the
same agent build the ticker uses — and prints ``Ran now: succeeded.``. The job's
saved output file (``cron/output/<id>/<ts>.md``, ``## Response`` section) is what
cron delivers locally, so that is ``final_text``. The daemon outlives the CLI, so
the surface's shutdown ends with ``hermes gateway stop``.

Both CLI calls run from the fake HOME, not the project: a scheduled job's
process cwd is whatever the ticker had, so the workdir is the only channel
that may carry the project context.
"""

from __future__ import annotations

import re
import subprocess

from tests.e2e.core.parity._helpers import (
    TURN_TIMEOUT,
    DriveResult,
    ParityHome,
    hermes_argv,
    stop_profile_gateway,
)
from tests.fakes.fake_llm_provider import FakeLLMServer

# cron/scheduler.py::_resolve_cron_enabled_toolsets -> _get_platform_tools(cfg, "cron") default.
CRON_TOOLSET = "hermes-cron"
_JOB_ID = re.compile(r"Created job: (\S+)")
_RESPONSE = re.compile(r"^## Response\s*\n(.*)\Z", re.S | re.M)


def _cron(ph: ParityHome, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        hermes_argv("cron", *args), cwd=ph.home, env=ph.env(), capture_output=True, text=True,
        timeout=TURN_TIMEOUT, stdin=subprocess.DEVNULL,
    )


def drive_cron(ph: ParityHome, srv: FakeLLMServer, prompt: str) -> DriveResult:
    created = _cron(ph, "create", "1d", prompt, "--name", "parity", "--deliver", "local",
                    "--workdir", str(ph.project))
    match = _JOB_ID.search(created.stdout)
    assert created.returncode == 0 and match, (
        f"hermes cron create exited {created.returncode}: {created.stdout[-1000:]} {created.stderr[-2000:]}")
    job_id = match.group(1)

    ran = _cron(ph, "run", job_id)
    gateway_stopped = stop_profile_gateway(ph)
    assert ran.returncode == 0, f"hermes cron run exited {ran.returncode}: {ran.stderr[-2000:]}"
    # Anything but a synchronous verdict means the run was handed to a ticker/background
    # worker this driver cannot observe — a harness problem, not a parity result.
    assert "Ran now:" in ran.stdout, f"hermes cron run did not execute synchronously: {ran.stdout[-1000:]}"

    outputs = sorted((ph.hermes_home / "cron" / "output" / job_id).glob("*.md"))
    final_text = None
    if outputs:
        body = outputs[-1].read_text(encoding="utf-8")
        m = _RESPONSE.search(body)
        final_text = (m.group(1) if m else body).strip()
    return DriveResult(
        final_text=final_text,
        toolset=CRON_TOOLSET,
        cwd_channel="job workdir",
        # Both CLI invocations exited on their own (subprocess.run, no signal); the daemon took
        # the operator stop.
        graceful_exit=gateway_stopped,
        extra={
            "job_id": job_id,
            "gateway_stopped": gateway_stopped,
            "exit_code": ran.returncode,
            "stderr_tail": ran.stderr[-2000:],
            "run_stdout": ran.stdout.strip()[-500:],
            "run_succeeded": "Ran now: succeeded." in ran.stdout,
            "output_file": str(outputs[-1]) if outputs else None,
        },
    )
