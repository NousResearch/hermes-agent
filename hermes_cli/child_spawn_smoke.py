"""Post-publish child-spawn smoke detector.

A supervisor re-spawns the commands an install publishes as
``sys.executable -m hermes_cli.main``: kanban workers
(``hermes_cli/kanban_db_dispatch.py``) and external cron workers
(``cron/scheduler*.py``). That child runs no launcher body and no
``activate_dependencies``, so the interpreter a published launcher embeds has to
import ``hermes_cli`` and every dependency on its own.

On 2026-09-25 a PM publish embedded the dependency-less store interpreter. The
gateway kept working -- ``hermes_bootstrap`` patches ``sys.path`` in process --
so ``hermes --version`` passed while every child died with
``ModuleNotFoundError``; nothing but the cron error stream mentioned it, and the
install was broken for twelve minutes before a human noticed.

This module is the standing detector for that state. After a publish (an update,
a ``hermes pm install``, a launcher repair) it spawns one throwaway child as
``<runtime> -m hermes_cli.main --version`` plus the import probe for the modules
that path needs, both under a clean environment and from a cwd that is not the
checkout. A failure RAISES (``verify_published_runtime``) instead of being
logged: an install whose dispatcher cannot spawn children is a failed install.

It deliberately does NOT assert on ``hermes --version``. That runs the launcher
body, which ``hermes_bootstrap`` completes in-process -- the parent passing is
precisely the state that hid the fault.

Standalone use (same code path the publish uses)::

    python -m hermes_cli.child_spawn_smoke [ROOT] [--launcher PATH] [--json]

Exit 0 when the runtime spawns children, 1 when it does not (with the failure
message on stdout/stderr), 2 when there is nothing to check yet.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

if __name__ == "__main__":
    # Direct execution leaves the checkout off sys.path (same idiom as
    # hermes_cli/_launchers.py): the probe itself is stdlib-only.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: Modules a bare ``sys.executable -m hermes_cli.main`` child must import.
#: ``cron.jobs`` and ``hermes_cli.main`` are the two entry paths the supervisor
#: spawns; ``ruamel.yaml`` is the config reader the cron worker died on in the
#: incident, and stands in for the dependency closure in general.
PROBE_MODULES = ("cron.jobs", "hermes_cli.main", "ruamel.yaml")

#: A child that has not answered in this long is broken, not slow: the same
#: spawn happens on every cron tick.
DEFAULT_TIMEOUT = 120.0

_MISSING_MODULE = re.compile(r"No module named ['\"]([^'\"]+)['\"]")


@dataclass(frozen=True)
class Check:
    """One spawned child: what ran, how it ended, what it said."""

    argv: tuple[str, ...]
    returncode: int | None  # None when the child timed out or could not spawn
    output: str
    missing_module: str | None

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def state(self) -> str:
        return "timed out" if self.returncode is None else f"exited {self.returncode}"

    @property
    def summary(self) -> str:
        """The one line worth putting in an error: the child's own complaint."""
        for line in reversed(self.output.splitlines()):
            if line.strip():
                return line.strip()
        return self.state


@dataclass(frozen=True)
class ProbeResult:
    ok: bool
    python: Path
    checks: tuple[Check, ...]

    @property
    def missing_module(self) -> str | None:
        for check in self.checks:
            if check.missing_module:
                return check.missing_module
        return None

    @property
    def timed_out(self) -> bool:
        return any(check.returncode is None for check in self.checks)

    def lines(self) -> list[str]:
        return [f"{shlex.join(check.argv)} -> {check.state}: {check.summary}"
                for check in self.checks]


def _clean_environment(home: str | Path | None = None) -> dict[str, str]:
    """``env -i`` with the one variable an interpreter needs to find itself.

    The dispatcher's children inherit a stripped environment too (the local
    ``PYTHONPATH`` strip empties theirs), so a probe that needs no ambient help is
    the honest test. ``HOME`` stays because interpreters and config readers
    resolve a home directory at import time.
    """
    environment = {"HOME": str(home) if home else str(Path.home())}
    if os.name == "nt":
        # Windows needs these to start a process at all; the POSIX probe's
        # "clean" has no exact twin there.
        for name in ("SYSTEMROOT", "WINDIR", "PATHEXT", "TEMP", "TMP", "COMSPEC"):
            value = os.environ.get(name)
            if value:
                environment[name] = value
    return environment


def _spawn(argv: list[str], environment: dict[str, str], cwd: Path, timeout: float) -> Check:
    try:
        completed = subprocess.run(
            argv, cwd=str(cwd), env=environment, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return Check(tuple(argv), None, f"no output within {timeout:g}s; killed", None)
    except OSError as exc:
        return Check(tuple(argv), None, f"{type(exc).__name__}: {exc}", None)
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part).strip()
    found = _MISSING_MODULE.search(output)
    return Check(tuple(argv), completed.returncode, output, found.group(1) if found else None)


def probe_child_spawn(python: str | Path, *, home: str | Path | None = None,
                      cwd: str | Path | None = None, timeout: float = DEFAULT_TIMEOUT,
                      ) -> ProbeResult:
    """Spawn the two children the supervisor spawns, from a bare cwd.

    *cwd* defaults to a fresh empty directory: running the child from the
    checkout would let a bare interpreter find ``hermes_cli`` through the current
    directory and pass without proving anything.
    """
    interpreter = Path(python)
    environment = _clean_environment(home)
    if cwd is None:
        workdir = Path(tempfile.mkdtemp(prefix="hermes-child-spawn-"))
    else:
        workdir = Path(cwd)
    checks = (
        _spawn([str(interpreter), "-m", "hermes_cli.main", "--version"], environment, workdir, timeout),
        _spawn([str(interpreter), "-c", f"import {', '.join(PROBE_MODULES)}"], environment, workdir, timeout),
    )
    return ProbeResult(ok=all(check.ok for check in checks), python=interpreter, checks=checks)


def committed_generation_python(root: Path) -> Path | None:
    """The committed dependency generation's interpreter, when one is really there.

    This is the gate, not the answer: a first install has no generation committed
    yet and legitimately publishes the store interpreter, which its launcher body
    completes with ``activate_dependencies``. Only a *usable* generation raises
    the promise this module checks, so an install that is allowed to be in the
    pre-generation state is never failed for it.
    """
    from pm.environments import committed_venv, site_packages, venv_python

    try:
        environment = committed_venv(Path(root))
    except (OSError, RuntimeError, ValueError):
        return None
    if environment is None:
        return None
    try:
        python = venv_python(environment)
        if python.is_file() and site_packages(environment).is_dir():
            return python
    except (OSError, ValueError):
        return None
    return None


def published_runtime(root: Path, *, launcher: Path | None = None) -> Path | None:
    """The interpreter children will inherit, read from the published launcher."""
    from hermes_cli._launchers import published_runtime_python, resolve_launcher_python

    root = Path(root)
    target = Path(launcher) if launcher is not None else root / ".hermes" / "bin" / "hermes"
    return published_runtime_python(target) or resolve_launcher_python(root)


def render_failure(result: ProbeResult, *, launcher: Path) -> tuple[str, str]:
    """(cause, remedy) naming the interpreter and the module, for InstallError."""
    detail = "; ".join(result.lines())
    missing = result.missing_module
    if missing:
        detail += f" (missing module: {missing})"
    cause = (
        f"the launcher {launcher} embeds {result.python}, which cannot spawn a "
        f"child: {detail}. Kanban workers and external cron workers are spawned "
        "as `sys.executable -m hermes_cli.main`, so this install dispatches "
        "nothing while its own CLI and gateway keep running (hermes_bootstrap "
        "completes the parent in-process, which is why `hermes --version` passes)"
    )
    remedy = (
        "run `hermes update` to republish the launchers on this install's "
        "committed dependency generation, or `hermes pm repair` if that "
        "generation is itself incomplete"
    )
    return cause, remedy


def verify_published_runtime(root: Path, *, launcher: Path | None = None,
                            timeout: float = DEFAULT_TIMEOUT, probe=probe_child_spawn,
                            ) -> ProbeResult | None:
    """Prove the published runtime can spawn children; raise when it cannot.

    Returns ``None`` when there is nothing to prove yet (no committed generation
    to be consistent with) or nothing to read the runtime from. Raises
    ``pm.package.InstallError`` when the interpreter the launcher embeds cannot
    run a bare child -- the state that leaves kanban and cron silent.
    """
    root = Path(root)
    target = Path(launcher) if launcher is not None else root / ".hermes" / "bin" / "hermes"
    if committed_generation_python(root) is None:
        log.info("child-spawn smoke check skipped: %s has no usable committed "
                 "dependency generation yet", root)
        return None
    python = published_runtime(root, launcher=target)
    if python is None:
        log.info("child-spawn smoke check skipped: no runtime could be read from %s", target)
        return None
    result = probe(python, timeout=timeout)
    if result.ok:
        log.info("child-spawn smoke check passed: %s runs `-m hermes_cli.main` from a "
                 "clean environment", python)
        return result

    from pm.package import InstallError

    cause, remedy = render_failure(result, launcher=target)
    raise InstallError("child-spawn", cause, remedy)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hermes_cli.child_spawn_smoke",
        description="Spawn a throwaway child the way the dispatcher does, and "
                    "fail loudly when the published runtime cannot import it.",
    )
    parser.add_argument("root", nargs="?", type=Path, default=None,
                        help="install root (default: this file's checkout)")
    parser.add_argument("--launcher", type=Path, default=None,
                        help="launcher file to read the runtime from "
                             "(default: ROOT/.hermes/bin/hermes)")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--json", action="store_true", help="machine-readable verdict")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    root = (args.root or Path(__file__).resolve().parents[1]).resolve()
    launcher = args.launcher or root / ".hermes" / "bin" / "hermes"
    if committed_generation_python(root) is None:
        verdict = {"ok": None, "skipped": "no-usable-committed-generation", "root": str(root)}
        print(json.dumps(verdict) if args.json else
              f"child-spawn smoke check: skipped (no usable committed dependency generation under {root})")
        return 2
    python = published_runtime(root, launcher=launcher)
    if python is None:
        verdict = {"ok": None, "skipped": "no-runtime-in-launcher", "launcher": str(launcher)}
        print(json.dumps(verdict) if args.json else
              f"child-spawn smoke check: skipped (no runtime readable from {launcher})")
        return 2

    result = probe_child_spawn(python, timeout=args.timeout)
    if result.ok:
        print(json.dumps({"ok": True, "python": str(python), "launcher": str(launcher),
                          "checks": [{"argv": list(check.argv), "returncode": check.returncode}
                                     for check in result.checks]}) if args.json else
              f"child-spawn smoke check: OK ({python} spawns `-m hermes_cli.main` from a bare cwd)")
        return 0

    cause, remedy = render_failure(result, launcher=launcher)
    verdict = {"ok": False, "python": str(python), "launcher": str(launcher),
               "missing_module": result.missing_module,
               "checks": [{"argv": list(check.argv), "returncode": check.returncode,
                           "output": check.output} for check in result.checks]}
    if args.json:
        print(json.dumps(verdict))
    else:
        print(f"child-spawn smoke check FAILED\n\n  {cause}\n\n  {remedy}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
