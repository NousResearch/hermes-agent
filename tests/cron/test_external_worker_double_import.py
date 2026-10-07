"""``-m cron.scheduler`` must not double-import the module (#132732).

The external worker entry point executes the ``cron`` package first; an eager
``from cron.scheduler import tick`` in ``__init__`` left ``cron.scheduler`` in
``sys.modules`` before runpy re-executed it as ``__main__`` — the classic
``python -m`` + package-imports-submodule collision the worker intermittently
crashed on. The package now resolves ``tick`` lazily (PEP 562), so importing
``cron`` no longer preloads the scheduler and the worker boots as a single
module execution.
"""

import os
from pathlib import Path
import subprocess
from subprocess import run as _spawn
import sys

_REPO_ROOT = str(Path(__file__).resolve().parents[2])

_PRELOAD_PROBE = (
    "import cron, sys; sys.exit(0 if 'cron.scheduler' in sys.modules else 1)"
)
_TICK_PROBE = "from cron import tick; print(callable(tick))"

# Injected via PYTHONPATH so the real ``python -m cron.scheduler`` machinery (not a
# runpy re-implementation) runs with an identity probe registered from interpreter
# startup: ``runpy.run_module`` executes the module in a scratch namespace, where
# ``sys.modules["__main__"]`` never refers to the running module.
_SITECUSTOMIZE_IDENTITY_PROBE = """
import atexit, sys


def _check_scheduler_identity():
    print(
        "SCHEDULER_IDENTITY_CHECK=",
        sys.modules.get("cron.scheduler") is sys.modules["__main__"],
    )


atexit.register(_check_scheduler_identity)
"""


def _env(home):
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("HERMES_", "_HERMES_"))
        and not k.endswith(("_API_KEY", "_TOKEN"))
    }
    env["HERMES_HOME"] = str(home)
    env["PYTHONPATH"] = _REPO_ROOT
    return env


def _run(argv, home):
    return _spawn(
        argv,
        env=_env(home),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_m_cron_scheduler_runs_without_double_import(tmp_path):
    """The worker entry point reaches its payload handling with no module collision.

    ``-W error`` turns the runpy double-import warning into an exception, so the
    pre-fix package died in ``runpy`` before ``__main__`` ever executed. Post-fix
    the process runs to the deterministic missing-payload failure (exit 1).
    """
    payload = tmp_path / "payload.json"
    payload.write_text(
        "not json", encoding="utf-8"
    )  # reaches the worker's own payload handling
    ack = str(tmp_path / "missing.ready")
    argv = [
        sys.executable,
        "-W",
        "error::RuntimeWarning",
        "-m",
        "cron.scheduler",
        "--external-worker-file",
        str(payload),
        "--ack-file",
        ack,
    ]
    result = _run(argv, tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "found in sys.modules" not in result.stderr, result.stderr
    # The worker consumed the payload file itself (its loader unlinks on exit),
    # so ``__main__`` really executed instead of dying inside runpy.
    assert not payload.exists()


def test_m_worker_executes_a_single_module_copy(tmp_path):
    """``cron.scheduler`` seen by importers IS the module runpy is executing.

    The missing-warning assertion above stays green even when the split modules'
    ``from cron import scheduler as _sched`` loads a second, independent copy
    after runpy's check — two ``CronTickYielded`` classes and state the delivery
    helpers cannot see. The identity print runs at interpreter exit, after the
    split modules had every chance to import ``cron.scheduler``.
    """
    probe_dir = tmp_path / "identity-probe"
    probe_dir.mkdir()
    (probe_dir / "sitecustomize.py").write_text(
        _SITECUSTOMIZE_IDENTITY_PROBE, encoding="utf-8"
    )
    payload = tmp_path / "identity-payload.json"
    payload.write_text("not json", encoding="utf-8")
    argv = [
        sys.executable,
        "-m",
        "cron.scheduler",
        "--external-worker-file",
        str(payload),
        "--ack-file",
        str(tmp_path / "missing.ready"),
    ]
    env = _env(tmp_path)
    env["PYTHONPATH"] = f"{probe_dir}{os.pathsep}{_REPO_ROOT}"
    result = _spawn(
        argv,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "SCHEDULER_IDENTITY_CHECK= True" in result.stdout, (
        result.stdout + result.stderr
    )


def test_package_import_does_not_preload_scheduler(tmp_path):
    argv = [sys.executable, "-c", _PRELOAD_PROBE]
    result = _run(argv, tmp_path)
    assert result.returncode == 1, (
        ("import cron preloaded cron.scheduler — the -m double-import is back")
        + result.stdout
        + result.stderr
    )


def test_from_cron_import_tick_still_resolves(tmp_path):
    argv = [sys.executable, "-c", _TICK_PROBE]
    result = _run(argv, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "True"
