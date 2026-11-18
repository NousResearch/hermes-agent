"""Run the real ``hermes`` CLI entry point in a real process against the engine tests' fake host.

``python -m tests.hermes_cli.backend_migration_race_host gateway migrate --backends ...`` is ``hermes
gateway migrate --backends ...`` with ``SystemHost`` replaced by ``FakeHost`` from
``test_gateway_migrate_backends_apply`` (service manager, process table and network faked; the home's files,
the manifest and the run lock are real). Used by the lock tests to race two real processes.

``RACE_DIR``: with ``RACE_HOLD=1`` the run stops in its preflight (the first read of the old backend),
writes ``RACE_DIR/held-PID`` and waits until ``RACE_DIR/release`` exists. A ``--detach`` spawns the
background run as another real process of this module (its own session), inheriting the environment.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
from pathlib import Path

from hermes_cli import gateway_migrate_backends_apply as engine
from hermes_cli import gateway_migrate_backends_host as host_module
from tests.hermes_cli.test_gateway_migrate_backends_apply import FakeHost

MODULE = "tests.hermes_cli.backend_migration_race_host"


class RaceHost(FakeHost):
    def backends(self):
        race = os.environ.get("RACE_DIR")
        if race and os.environ.get("RACE_HOLD") == "1":
            from tests.waiting import wait_until
            (Path(race) / f"held-{os.getpid()}").write_text("", encoding="utf-8")
            with contextlib.suppress(AssertionError):  # never hang a test run: go on after 60 s
                wait_until((Path(race) / "release").exists, timeout=60)
        return super().backends()

    def spawn_detached(self, flags, log):
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "ab") as fh:
            proc = subprocess.Popen([sys.executable, "-m", MODULE, *flags], stdin=subprocess.DEVNULL,
                                    stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        return {"pid": proc.pid}


def main() -> None:
    engine.VERIFY_SECONDS = 0.0
    engine.ROLLBACK_VERIFY_SECONDS = 0.0
    host_module.SystemHost = RaceHost
    sys.argv = ["hermes", *sys.argv[1:]]
    from hermes_cli.main import main as hermes_main
    hermes_main()


if __name__ == "__main__":
    main()
