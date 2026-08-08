"""npm toolchain helpers — extracted from ``hermes_cli/main.py``.

Mechanical move (main.py decomposition, god-file slice R3 / epic #78647,
target #78631): the four deterministic-npm / subprocess-execution helpers
(``_run_with_idle_timeout``, ``_nixos_build_env``,
``_run_npm_install_deterministic``, ``_run_npm_watching_for_engine_failure``)
are lifted verbatim.  The cluster has zero references to other
``hermes_cli.main`` functions; the only host binding it touches is the module
global ``PROJECT_ROOT``, which is routed through a lazy ``_m()`` main
reference (call-time only) so existing test monkeypatches on
``hermes_cli.main.<name>`` keep reaching this code path, and imports stay
one-way at import time (main.py imports this module, never the reverse).
``main.py`` re-exports all four names (``# noqa: F401``) so callers and test
patches on ``hermes_cli.main`` resolve unchanged.
"""
import os
import shutil
import subprocess
import sys
import threading
import time as _time
from pathlib import Path


def _m():
    """Lazy ``hermes_cli.main`` reference (call-time; keeps patches working)."""
    from hermes_cli import main

    return main


def _run_with_idle_timeout(
    cmd: list[str],
    cwd: Path,
    *,
    idle_timeout_seconds: int = 180,
    indent: str = "    ",
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Run a subprocess that streams output, with an idle-output timeout.

    Issue #33788: ``npm run build`` (Vite) was invoked with
    ``capture_output=True`` and no timeout. On low-memory hosts (notably
    WSL2 with the default 4 GB cap) the build can stall or sit silent for
    minutes; users see a frozen terminal, assume the update is hung, and
    reboot — leaving the editable install in a half-state with the
    ``hermes`` launcher present but ``hermes_cli`` not importable.

    This helper fixes both halves: stdout is streamed (so the user sees
    progress), and if no bytes have appeared on stdout/stderr for
    ``idle_timeout_seconds``, the process is terminated and the call
    returns with a non-zero ``returncode``. The caller's existing
    stale-dist fallback (#23817) takes over from there.

    Returns a ``CompletedProcess`` with merged stdout (text), empty
    stderr, and an integer returncode. Never raises on idle timeout —
    propagation of failure is via the returncode.
    """
    merged_chunks: list[str] = []
    last_output_ts = _time.monotonic()
    lock = threading.Lock()

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
    except OSError as exc:
        # E.g. npm not on PATH between the which() check and now.
        return subprocess.CompletedProcess(cmd, 127, stdout="", stderr=str(exc))

    def _reader() -> None:
        nonlocal last_output_ts
        assert proc.stdout is not None
        for line in proc.stdout:
            try:
                print(f"{indent}{line.rstrip()}", flush=True)
            except UnicodeEncodeError:
                # Windows cp1252 fallback — same pattern as _say().
                enc = getattr(sys.stdout, "encoding", None) or "ascii"
                safe = line.rstrip().encode(enc, errors="replace").decode(enc, errors="replace")
                print(f"{indent}{safe}", flush=True)
            with lock:
                merged_chunks.append(line)
                last_output_ts = _time.monotonic()

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    idle_killed = False
    while True:
        try:
            rc = proc.wait(timeout=5)
            break
        except subprocess.TimeoutExpired:
            with lock:
                idle = _time.monotonic() - last_output_ts
            if idle > idle_timeout_seconds:
                idle_killed = True
                proc.terminate()
                try:
                    rc = proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    rc = proc.wait()
                break

    # Drain reader so we don't leak the stdout file descriptor.
    reader_thread.join(timeout=2)

    combined = "".join(merged_chunks)
    if idle_killed:
        msg = (
            f"\n  ⚠ Build produced no output for {idle_timeout_seconds}s — terminated.\n"
            "    Common causes: out-of-memory on a low-RAM host (WSL/container),\n"
            "    a stuck Node process, or an antivirus scan stalling I/O.\n"
        )
        combined += msg
        # Force a non-zero rc even if terminate() raced with a clean exit.
        if rc == 0:
            rc = 124  # GNU `timeout` convention
    return subprocess.CompletedProcess(cmd, rc, stdout=combined, stderr="")


def _nixos_build_env() -> dict[str, str] | None:
    """Return extra env vars for native module builds on NixOS.

    On NixOS, python3 is typically not on the system PATH (it lives in
    the Nix store and only enters PATH inside a nix-shell or when
    explicitly installed as a system package).  node-gyp uses Python to
    compile native addons like ``node-pty`` and its ``find-python.js``
    does a bare ``PATH`` lookup — which fails on NixOS.

    Two-tier resolution:
    1. Fast path — the hermes venv's python3 (present in managed installs)
    2. Fallback — resolves the absolute python3 path via ``nix-shell``

    Returns an env dict suitable for ``subprocess.run(env=...)`` or
    ``None`` when we are not on NixOS or python3 is already on PATH.
    """
    import re

    try:
        os_release = Path("/etc/os-release").read_text(encoding="utf-8")
    except OSError:
        return None
    if not re.search(r"^ID=nixos$", os_release, re.M):
        return None

    # python3 already on PATH — nothing to do
    if shutil.which("python3"):
        return None

    # Tier 1: fast path — hermes venv python3, no nix-shell overhead
    for venv_name in ("venv", ".venv"):
        venv_python = _m().PROJECT_ROOT / venv_name / "bin" / "python3"
        if venv_python.exists():
            return {**os.environ, "PYTHON": str(venv_python)}

    # Tier 2: nix-shell fallback — resolves the absolute python3 path once.
    # Slower (~2–5 s for the nix-shell eval) but always works, even without
    # a hermes venv (pip / non-managed / bare-git installs).  The resolved
    # path is a self-contained Nix store binary (all deps via RPATH) so it
    # stays valid even after the nix-shell exits.
    try:
        result = subprocess.run(
            ["nix-shell", "-p", "python3", "--run", "which python3"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", check=False, timeout=15,
        )
        if result.returncode == 0:
            python3_path = result.stdout.strip()
            if python3_path and Path(python3_path).exists():
                return {**os.environ, "PYTHON": python3_path}
    except Exception:
        pass  # nix-shell not available — caller will get None

    return None
def _run_npm_install_deterministic(
    npm: str,
    cwd: Path,
    *,
    extra_args: tuple[str, ...] = (),
    capture_output: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Run a deterministic npm install that does not mutate ``package-lock.json``.

    Prefers ``npm ci`` (strict, lockfile-preserving) when a lockfile is present;
    falls back to ``npm install`` only if ``npm ci`` fails (e.g. lockfile out of
    sync on a WIP checkout).  Without this, ``npm install`` on npm ≥ 10 silently
    rewrites committed lockfiles (stripping ``"peer": true`` etc.), which leaves
    the working tree dirty and causes the next ``hermes update`` to stash the
    lockfile — repeatedly.

    ``--include=dev`` is forced on every invocation: the callers are frontend
    builds (web UI / TUI / desktop workspaces), and those builds need the dev
    toolchain (``tsc``, ``vite``, ``electron-builder`` — all
    ``devDependencies``).  If the caller's environment has
    ``NODE_ENV=production`` (or npm config ``omit=dev``) — which leaks in from
    a shell profile, a container image, or the bundled TUI launcher that sets
    ``NODE_ENV=production`` on its subprocess env — npm silently omits
    devDependencies (exit 0, no error), so the build toolchain never installs
    and the subsequent build dies with ``tsc: command not found`` (exit 127).
    The flag overrides both the env var and npm config, unlike scrubbing
    ``NODE_ENV`` from the environment which only fixes the env-leak case.

    ``--no-save`` on the ``npm install`` fallback keeps it true to this
    function's contract: never mutate ``package-lock.json``.  Without it, an
    out-of-sync lockfile gets rewritten by the fallback, which drifts the
    committed lockfile and makes every future ``npm ci`` fail — a
    self-reinforcing cycle where web devDeps never install and a stale dist
    is served on every update (PR #65595).
    """
    # unicode-animations' postinstall animates to /dev/tty (bypasses
    # --silent/capture_output). It no-ops when CI is set — same as the TUI
    # install path and nix/lib.nix npm ci hooks.
    run_env = {**os.environ, **(env or {}), "CI": "1"}

    def _run(cmd: list[str]) -> subprocess.CompletedProcess:
        return _run_npm_watching_for_engine_failure(
            cmd,
            cwd=cwd,
            env=run_env,
            capture_output=capture_output,
        )

    def _attempt(npm_exe: str) -> subprocess.CompletedProcess:
        lockfile = cwd / "package-lock.json"
        if lockfile.exists():
            ci_result = _run([npm_exe, "ci", "--include=dev", *extra_args])
            if ci_result.returncode == 0:
                return ci_result
            # Fall through to `npm install` — lockfile may be out of sync on a
            # WIP fork/branch, or `npm ci` may not be available on very old npm.
        return _run([npm_exe, "install", "--no-save", "--include=dev", *extra_args])

    result = _attempt(npm)
    if result.returncode == 0:
        return result

    # An npm outside the root package.json's `engines.npm` range fails every
    # command here identically (the `npm install` fallback included), so the
    # failure is worth exactly one repair attempt. `maybe_repair_npm_engine`
    # returns the npm to retry with — the same one after an in-place upgrade
    # of a Hermes-managed install, or a freshly provisioned managed npm when
    # the failing npm belongs to the user's own toolchain.
    from hermes_cli.npm_engine import maybe_repair_npm_engine

    combined = f"{result.stdout or ''}\n{result.stderr or ''}"
    repaired_npm = maybe_repair_npm_engine(npm, combined)
    if not repaired_npm:
        return result
    # The repaired npm may be a freshly provisioned managed one whose shebang
    # and lifecycle scripts resolve `node` from PATH — put the managed tree
    # first so they find the managed Node, not the mismatched system one.
    from hermes_constants import with_hermes_node_path

    run_env["PATH"] = with_hermes_node_path(run_env)["PATH"]
    return _attempt(repaired_npm)


def _run_npm_watching_for_engine_failure(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    capture_output: bool,
) -> subprocess.CompletedProcess:
    """Run *cmd*, always retaining stderr so ``EBADENGINE`` stays detectable.

    ``capture_output=False`` callers stream npm's progress live and would
    otherwise hand back a ``CompletedProcess`` with ``stderr=None``, leaving the
    engine-failure recovery nothing to read. Tee stderr instead: each line is
    forwarded to this process's stderr as it arrives (so live output is
    unchanged) and accumulated for the caller.
    """
    if capture_output:
        return subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

    captured: list[str] = []
    with subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    ) as proc:
        if proc.stderr is not None:
            for line in proc.stderr:
                captured.append(line)
                sys.stderr.write(line)
            sys.stderr.flush()
        returncode = proc.wait()
    return subprocess.CompletedProcess(cmd, returncode, None, "".join(captured))
