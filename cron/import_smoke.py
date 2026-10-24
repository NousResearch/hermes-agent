"""Import-chain smoke test for the cron dispatch path.

The cron external worker runs as ``sys.executable -m cron.scheduler`` against the
LIVE checkout (``cron/scheduler.py`` puts the repo root on ``sys.path``), so one
module-level error anywhere in the shared gateway/agent import chain kills EVERY
job that touches the gateway delivery path, at once, on the next tick.

Field incident (2026-10-24 02:16Z): ``agent/auxiliary_client.py`` gained a
``defaults=(None,)`` argument on the FUNCTIONAL ``NamedTuple`` form, which
``typing.NamedTuple`` rejects with ``TypeError: Either list of fields or
keywords can be provided to NamedTuple, not both`` — raised at module import.
17 distinct cron jobs failed in the same second, each with the opaque
``Restart-safe cron worker dispatch failed: cron external worker exited before
ownership acknowledgement (exit 1)``. The root cause (one line in one file) was
buried under 17 identical symptoms.

This module turns that class of failure into ONE diagnostic before the first
job is dispatched: the failing module is named, and the traceback is carried
verbatim. The check runs at most once per process per tree state, so a healthy
tree pays one import pass and nothing more.

Why a smoke test and not "just import the module that broke": the worker
imports the LIVE tree on every dispatch, but the GATEWAY process has its own
``sys.modules`` frozen at boot. A module the gateway already imported cannot be
re-imported to reveal a fresh error, so the probe deliberately runs in a
SUBPROCESS against the same tree the worker will use. The subprocess is the only
honest answer to "will the worker's import succeed?".
"""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from tools.environments.local import served_profile_child_env

logger = logging.getLogger("cron.scheduler")

# The chain the external worker walks when it imports the gateway delivery path.
# Ordered outermost-first so the diagnostic names the FIRST module that failed,
# which is the one an operator must fix.
#
# Provenance: the 2026-10-24 incident's failing chain, captured from the
# worker's own stderr:
#     gateway.delivery -> gateway.session -> gateway.session_transcript
#     -> agent.turn_context -> agent.conversation_compression
#     -> agent.auxiliary_client
# `cron.scheduler` itself is the worker's entry module and is imported by
# definition; the modules below are what it pulls in BEYOND itself.
SMOKE_CHAIN: tuple[str, ...] = (
    "gateway.delivery",
    "gateway.session",
    "gateway.session_transcript",
    "agent.turn_context",
    "agent.conversation_compression",
    "agent.auxiliary_client",
)

# Modules whose import failure means "the tree is broken", not "this one
# optional dependency is missing". The probe distinguishes the two by the
# exception type: an ImportError/ModuleNotFoundError for a module in the chain
# is a tree problem; anything else (TypeError, AttributeError, SyntaxError) is
# also a tree problem. A missing THIRD-PARTY dependency (ruamel, httpx) is a
# broken environment, and must not be reported as a broken tree — the worker's
# own boot owns the dependency generation.
_TREE_ERRORS = (TypeError, AttributeError, SyntaxError, NameError, ValueError)


@dataclass(frozen=True)
class SmokeResult:
    """Outcome of one import-chain probe.

    ``ok`` is the verdict. ``failing_module`` names the module that raised (the
    one the operator must fix), ``error_type`` / ``error_message`` name the
    exception, and ``traceback`` carries the worker's own traceback text so the
    diagnostic reads exactly like the stderr the operator would have had to
    dig for.
    """

    ok: bool
    failing_module: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    traceback: Optional[str] = None
    chain: tuple[str, ...] = ()

    def diagnostic(self) -> str:
        """One-line-per-fact summary naming the module and the cause."""
        if self.ok:
            return "cron dispatch import chain is healthy"
        return (
            f"cron dispatch import chain is BROKEN: {self.failing_module} "
            f"failed to import ({self.error_type}: {self.error_message}). "
            f"The external cron worker imports the live checkout, so every job "
            f"on the gateway delivery path will fail until this is fixed."
        )


# Probe result cache. Keyed by the tree fingerprint so a tree that changes
# under a running gateway (the `git pull` case) is re-probed, while an
# unchanged tree is probed once.
_cache_key: Optional[str] = None
_cached_result: Optional[SmokeResult] = None

# Set to True by tests / operators to make the probe loud even when the tree
# looks unchanged. Not part of the public contract.
_FORCE = False


def _tree_fingerprint(repo_root: Path) -> str:
    """Cheap fingerprint of the tree's import surface.

    Hashes the mtime+size of every file in the modules the chain covers, so an
    edit anywhere in the chain re-probes. Deliberately NOT a content hash: the
    probe runs on the gateway's tick, and stat'ing a few dozen files is cheap
    while reading them is not.
    """
    digest = hashlib.sha256()
    roots = ("gateway", "agent", "cron")
    for root in roots:
        base = repo_root / root
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            try:
                stat = path.stat()
            except OSError:
                continue
            digest.update(f"{path}:{stat.st_mtime_ns}:{stat.st_size}".encode())
    return digest.hexdigest()


def _repo_root() -> Path:
    """The checkout the cron worker will import from."""
    return Path(__file__).resolve().parent.parent


def _probe_source(chain: tuple[str, ...], repo_root: Optional[Path] = None) -> str:
    """The child program that imports *chain* and reports the first failure.

    The child runs ``hermes_bootstrap`` FIRST, exactly like the cron worker
    does. That is not optional: ``hermes_bootstrap`` puts the MANAGED
    environment's ``site-packages`` on ``sys.path`` (the generation the worker
    leases). Without it the child cannot import ``ruamel`` — the first thing
    ``hermes_yaml`` needs — and the probe would classify every failure as "a
    missing third-party dependency", i.e. "the environment, not the tree". That
    blind spot is exactly the incident: the tree WAS broken, and the probe
    would have shrugged at it.

    The child also runs with the SAME interpreter as the parent (``sys.executable``),
    which is the interpreter the gateway (and therefore the worker) uses.
    """
    chain_literal = repr(list(chain))
    root_literal = repr(str(repo_root)) if repo_root is not None else repr(str(_repo_root()))
    # `hermes_bootstrap` is INFRASTRUCTURE, not part of the tree under test: it
    # is what puts the managed environment's site-packages on sys.path. It must
    # come from the LIVE checkout even when the tree under test is a scratch
    # copy — a copy of it in the scratch tree cannot resolve the managed
    # environment, and the probe would fail on `ruamel` instead of on the
    # module it was sent to check.
    live_root_literal = repr(str(_repo_root()))
    return (
        "import importlib, json, os, sys, traceback\n"
        # Mirror the worker's boot: the managed environment's site-packages
        # must be on sys.path before any first-party import.
        "os.environ.pop('PYTHONHOME', None)\n"
        "os.environ.pop('PYTHONPATH', None)\n"
        f"sys.path.insert(0, {live_root_literal})\n"
        "try:\n"
        "    import hermes_bootstrap  # noqa: F401\n"
        "except Exception:\n"
        "    pass\n"
        # The tree under test goes FIRST on sys.path, exactly like
        # `cron/scheduler.py` does for itself. This must come AFTER
        # `hermes_bootstrap`: the bootstrap REPLACES sys.path[0] with the
        # live repo root, so a tree inserted before it would be shadowed and
        # the probe would silently test the wrong tree.
        f"sys.path.insert(0, {root_literal})\n"
        f"chain = {chain_literal}\n"
        "for name in chain:\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "    except BaseException as exc:\n"
        "        missing = None\n"
        "        if isinstance(exc, ImportError):\n"
        "            missing = getattr(exc, 'name', None)\n"
        # The chain member that FAILED is the OUTERMOST one (the first import\n"
        "        # that raised), but the MODULE THAT IS BROKEN is the deepest\n"
        "        # frame that lives in the TREE: the chain is a walk, and the last\n"
        "        # tree frame is where the bad code is. The very deepest frame is\n"
        "        # often the stdlib that raised (`typing.NamedTuple` in the\n"
        "        # 2026-10-24 incident) — naming that would point the operator at\n"
        "        # the standard library instead of the file to fix.\n"
        "        culprit = name\n"
        "        tb = exc.__traceback__\n"
        "        frames = []\n"
        "        while tb is not None:\n"
        "            frames.append(tb)\n"
        "            tb = tb.tb_next\n"
        "        if missing:\n"
        "            # A MISSING module: the culprit is the tree frame that ASKED\n"
        "            # for it (the module whose `import` line is wrong), not the\n"
        "            # machinery that raised the ImportError.\n"
        "            for frame in reversed(frames):\n"
        "                fname = frame.tb_frame.f_code.co_filename or ''\n"
        "                if 'site-packages' in fname or 'lib/python' in fname:\n"
        "                    continue\n"
        "                if fname.startswith('<'):\n"
        "                    continue\n"
        "                candidate = (frame.tb_frame.f_globals or {}).get('__name__')\n"
        "                if candidate:\n"
        "                    culprit = candidate\n"
        "                    break\n"
        "        else:\n"
        "            # Any other error: the culprit is the deepest TREE frame —\n"
        "            # the chain is a walk, and the last tree frame is where the\n"
        "            # bad code is. The very deepest frame is often the stdlib\n"
        "            # that raised (`typing.NamedTuple` in the 2026-10-24\n"
        "            # incident) — naming that would point the operator at the\n"
        "            # standard library instead of the file to fix.\n"
        "            for frame in reversed(frames):\n"
        "                code = frame.tb_frame.f_code\n"
        "                fname = code.co_filename or ''\n"
        "                if 'site-packages' in fname or 'lib/python' in fname:\n"
        "                    continue\n"
        "                # Frozen stdlib modules (`<frozen importlib._bootstrap>`)\n"
        "                # are the machinery, not the tree.\n"
        "                if fname.startswith('<'):\n"
        "                    continue\n"
        "                candidate = (frame.tb_frame.f_globals or {}).get('__name__')\n"
        "                if candidate:\n"
        "                    culprit = candidate\n"
        "                    break\n"
        "        sys.stdout.write(json.dumps({\n"
        "            'ok': False,\n"
        "            'failing_module': name,\n"
        "            'culprit_module': culprit,\n"
        "            'missing_name': missing,\n"
        "            'error_type': type(exc).__name__,\n"
        "            'error_message': str(exc),\n"
        "            'traceback': traceback.format_exc(),\n"
        "        }))\n"
        "        raise SystemExit(0)\n"
        "sys.stdout.write(json.dumps({'ok': True}))\n"
    )


def _is_tree_error(
    error_type: str,
    missing_name: Optional[str],
    chain: tuple[str, ...],
    repo_root: Optional[Path] = None,
) -> bool:
    """True when the failure means "the tree is broken", not "env is missing".

    The distinction matters: a missing THIRD-PARTY dependency (ruamel, httpx) is
    the worker's own problem — its PM boot leases the dependency generation —
    and must not be reported as a broken tree. But a chain module that does not
    import IS the tree's problem: the module was deleted, renamed, or never
    existed, and every job on this path is dead until it is fixed.

    ``missing_name`` is the dotted name the ImportError named (``ruamel``,
    ``agent.auxiliary_client``). A name that is not a chain member is resolved
    against the TREE: if the checkout holds a module by that name, the tree is
    broken (it exists on disk but does not import). Only a name the checkout
    does NOT hold at all is an external dependency.
    """
    if error_type in {t.__name__ for t in _TREE_ERRORS}:
        return True
    if error_type not in {"ImportError", "ModuleNotFoundError"}:
        return True
    if not missing_name:
        return True
    # The missing name IS the chain member we tried to import: the module was
    # deleted or renamed. That is the tree.
    if missing_name in chain:
        return True
    # The checkout HOLDS a module by this name but it did not import: the tree
    # is broken (a first-party module at the repo root — ``hermes_yaml``,
    # ``hermes_constants`` — is in this class, and missing it means every job
    # on the path dies).
    if repo_root is not None and _module_exists_in_tree(repo_root, missing_name):
        return True
    top_level = missing_name.split(".")[0]
    # A Hermes package (agent / gateway / cron / hermes_cli / tools) is the
    # tree too. Anything else is an external dependency — the worker's own PM
    # boot leases those, so the tree is not at fault.
    return top_level in {
        "agent", "gateway", "cron", "hermes_cli", "tools", "hermes_state",
    }


def _module_exists_in_tree(repo_root: Path, dotted_name: str) -> bool:
    """True when the checkout holds a module by *dotted_name*.

    Checks a package directory first (``agent/auxiliary_client.py``) and then
    a top-level module (``hermes_yaml.py``). A first-party module that exists
    on disk but fails to import is a BROKEN TREE, not a missing dependency —
    the distinction the incident turned on.
    """
    parts = dotted_name.split(".")
    candidate = repo_root.joinpath(*parts[:-1]) / f"{parts[-1]}.py"
    if candidate.is_file():
        return True
    package = repo_root.joinpath(*parts)
    return package.is_dir() and (package / "__init__.py").is_file()


def _run_probe(chain: tuple[str, ...], repo_root: Path) -> SmokeResult:
    """Run the import probe in a subprocess and parse its verdict.

    The child runs with ``cwd=repo_root`` and the tree's root FIRST on
    ``sys.path`` (via ``PYTHONPATH``), so it imports the tree under test — not
    an installed copy of it.
    """
    script = _probe_source(chain, repo_root)
    # The probe child IMPORTS the chain and nothing else: it needs no
    # credentials, no provider keys, no profile secrets. Build the env with
    # ``served_profile_child_env(inherit_credentials=False)`` so the child
    # cannot inherit the launch profile's dotenv — a plain ``os.environ``
    # copy would hand every provider key to a subprocess that has no use
    # for them.
    env = served_profile_child_env(inherit_credentials=False)
    # The probe must import the SAME tree the worker will import: the repo root
    # first on sys.path, exactly like `cron/scheduler.py` does for itself.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root), *[p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]]
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(repo_root),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
            text=True,
        )
    except subprocess.TimeoutExpired:
        return SmokeResult(
            ok=False,
            failing_module=chain[0] if chain else "<chain>",
            error_type="SmokeTimeout",
            error_message="import-chain probe timed out after 120s",
            chain=chain,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - defensive
        # The probe could not even be LAUNCHED (no interpreter, fork failed,
        # the tree vanished). That is not a verdict about the tree, but it
        # means the operator learns nothing — so report it as a failure with
        # the reason, which is strictly more useful than a bare "not ok".
        return SmokeResult(
            ok=False,
            failing_module=chain[0] if chain else "<chain>",
            error_type=type(exc).__name__,
            error_message=str(exc),
            chain=chain,
        )

    stdout = (completed.stdout or "").strip()
    if not stdout:
        # The probe itself died before it could report — surface its stderr so
        # the operator sees WHY rather than "the chain is broken".
        stderr_tail = (completed.stderr or "").strip().splitlines()
        return SmokeResult(
            ok=False,
            failing_module=chain[0] if chain else "<chain>",
            error_type="SmokeProbeFailed",
            error_message=(
                "import-chain probe exited without a verdict "
                f"(rc={completed.returncode})"
            ),
            traceback="\n".join(stderr_tail[-20:]) if stderr_tail else None,
            chain=chain,
        )

    import json

    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        return SmokeResult(
            ok=False,
            failing_module=chain[0] if chain else "<chain>",
            error_type="SmokeProbeFailed",
            error_message=f"unreadable probe verdict: {exc}",
            traceback=stdout[:4000],
            chain=chain,
        )

    if payload.get("ok"):
        return SmokeResult(ok=True, chain=chain)

    error_type = str(payload.get("error_type") or "")
    missing_name = payload.get("missing_name")
    if not _is_tree_error(error_type, missing_name, chain, repo_root):
        # A missing third-party dependency is the worker's own problem (its
        # PM boot leases the dependency generation). Do NOT fail the tree.
        logger.debug(
            "cron dispatch import smoke: %s failed with %s (environment, not tree): %s",
            payload.get("failing_module"), error_type, payload.get("error_message"),
        )
        return SmokeResult(ok=True, chain=chain)

    return SmokeResult(
        ok=False,
        failing_module=str(
            payload.get("culprit_module")
            or payload.get("failing_module")
            or "<unknown>"
        ),
        error_type=error_type,
        error_message=str(payload.get("error_message") or ""),
        traceback=payload.get("traceback"),
        chain=chain,
    )


def check_import_chain(
    chain: tuple[str, ...] = SMOKE_CHAIN,
    *,
    force: bool = False,
    use_cache: bool = True,
) -> SmokeResult:
    """Probe the cron dispatch import chain; return the verdict.

    Cached per tree fingerprint: an unchanged tree is probed once per process,
    a tree edited under a running gateway is re-probed on the next tick.
    """
    global _cache_key, _cached_result

    repo_root = _repo_root()
    fingerprint = _tree_fingerprint(repo_root)
    if use_cache and not force and not _FORCE:
        if _cache_key == fingerprint and _cached_result is not None:
            return _cached_result

    result = _run_probe(chain, repo_root)
    _cache_key = fingerprint
    _cached_result = result
    return result


def reset_cache() -> None:
    """Drop the cached verdict (tests, and operators who just fixed the tree)."""
    global _cache_key, _cached_result
    _cache_key = None
    _cached_result = None


# Lines that mark a Python traceback's final exception line.
_TRACEBACK_EXCEPTION_MARKERS = (
    "TypeError:", "ImportError:", "ModuleNotFoundError:", "SyntaxError:",
    "AttributeError:", "NameError:", "ValueError:", "IndentationError:",
)


def summarize_worker_import_failure(stderr_text: str) -> Optional[str]:
    """Name the module that failed to import in a worker's captured stderr.

    Defense in depth for the case the tick-level smoke test cannot cover: the
    tree was edited BETWEEN the probe and the spawn, so the probe passed and the
    worker still died at import. The worker's own traceback is the evidence; this
    turns it into one sentence an operator can act on, so the per-job error
    stops reading as "exited before ownership acknowledgement (exit 1)".

    The DEEPEST frame is the failure: the worker walks a chain, so the traceback
    lists every link it walked, and the last one is where it died
    (``agent/auxiliary_client.py`` in the 2026-10-24 incident, not the
    ``gateway/delivery.py`` that started the walk).

    Returns ``None`` when the stderr shows no import-class failure.
    """
    if not stderr_text:
        return None
    text = stderr_text.replace("\r", "\n")
    lines = text.split("\n")
    # The deepest `File ".../x.py", line N, in <module>` frame is where the
    # worker died. The FILE it names is the module that broke — more reliable
    # than the source line under it, which may be a field list or an
    # expression rather than an import statement.
    frames = [
        (index, line) for index, line in enumerate(lines)
        if '", line ' in line and " in <module>" in line
    ]
    failing_file: Optional[str] = None
    if frames:
        _, frame_line = frames[-1]
        start = frame_line.find('File "')
        end = frame_line.find('", line ')
        if start != -1 and end != -1:
            failing_file = frame_line[start + 6:end]
    exception_line = None
    for line in reversed(lines):
        stripped = line.strip()
        if any(stripped.startswith(marker) for marker in _TRACEBACK_EXCEPTION_MARKERS):
            exception_line = stripped
            break
    if exception_line is None:
        return None
    if failing_file:
        return f"import failure in {failing_file}: {exception_line}"
    return f"import failure: {exception_line}"
