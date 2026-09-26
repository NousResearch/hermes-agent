"""Shims to stop the old updater doing work until relaunch.

An updater already in memory imports these names after replacing its checkout.
They must not install anything, delegate to PM, or return a falsy value that
would send that old process down its pip fallback. New code must not use them.
"""

from pathlib import Path
from typing import NoReturn

from hermes_cli._old_updater import stop_for_relaunch


def _reload_hermes_constants() -> NoReturn:
    # Shim to suppress old updater work until relaunch. Callers dereference the
    # result, so None crashes. Stop without re-executing live globals.
    stop_for_relaunch()


def ensure_uv(*args, **kwargs) -> NoReturn:
    # Shim to stop the old updater doing work until relaunch. Older releases
    # expect a tuple, newer ones a path. Exit before either can consume it.
    stop_for_relaunch()


def update_managed_uv(*args, **kwargs) -> NoReturn:
    # Shim to stop the old updater doing work until relaunch.
    stop_for_relaunch()


def resolve_uv(*args, **kwargs) -> NoReturn:
    # Shim to stop the old updater doing work until relaunch, not enable pip.
    stop_for_relaunch()


def managed_python_env(
    project_root: Path | None = None, *, install_dir: Path | None = None,
    base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Return a sanitized environment for Hermes-private uv Python commands."""
    target = (
        Path(install_dir) if install_dir is not None else managed_python_install_dir(project_root))
    env = dict(os.environ if base_env is None else base_env)
    for key in (
        "CONDA_DEFAULT_ENV", "CONDA_PREFIX", "UV_PROJECT_ENVIRONMENT", "UV_NO_MANAGED_PYTHON",
        "UV_PYTHON", "UV_PYTHON_DOWNLOADS", "UV_SYSTEM_PYTHON", "VIRTUAL_ENV", "PYTHONHOME",
        "PYTHONPATH"):
        env.pop(key, None)
    env.update({
        "UV_MANAGED_PYTHON": "1", "UV_NO_CONFIG": "1", "UV_PYTHON_INSTALL_BIN": "0",
        "UV_PYTHON_INSTALL_DIR": str(target), "UV_PYTHON_INSTALL_REGISTRY": "0"})
    return env


def _macos_sign_managed_python(python: Path) -> bool:
    """Give a newly downloaded managed Python a stable macOS code identity.

    python-build-standalone binaries are ad-hoc signed, so TCC sees a cdhash-only identity that
    changes every runtime generation; an identifier-pinned designated requirement keeps it stable
    without a Developer ID. Best effort: a missing/incompatible ``codesign`` must not block repair.
    """
    if platform.system() != "Darwin":
        return False
    codesign = shutil.which("codesign")
    if not codesign:
        logger.info("macOS codesign is unavailable; using the downloaded Python signature")
        return False
    requirement = f'=designated => identifier "{_MACOS_MANAGED_PYTHON_IDENTIFIER}"'
    try:
        sign = [
            codesign, "--force", "--deep", "--sign", "-", "--timestamp=none",
            "--identifier", _MACOS_MANAGED_PYTHON_IDENTIFIER,
            "--requirements", requirement, str(python)]
        verify = [codesign, "--verify", "--deep", "--strict", str(python)]
        steps = (
            (sign, "could not stably sign managed Python %s: %s", "codesign failed"),
            (verify, "macOS signature verification failed for managed Python %s: %s",
             "verification failed"))
        for cmd, warning, fallback in steps:
            result = subprocess.run(
                cmd, check=False, capture_output=True, text=True, encoding="utf-8", errors="replace"
            )
            if result.returncode != 0:
                logger.warning(
                    warning, python, (result.stderr or result.stdout or fallback).strip())
                return False
        return True
    except Exception as exc:
        logger.warning("could not sign managed Python %s: %s", python, exc)
        return False


@dataclass(frozen=True)
class RuntimeRepairResult:
    """Outcome of a managed-runtime repair attempt."""

    status: str
    detail: str = ""
    sqlite_before: str = ""
    sqlite_after: str = ""
    backup_venv: Path | None = None

    @property
    def repaired(self) -> bool:
        return self.status == "repaired"


@dataclass(frozen=True)
class _RepairLock:
    path: Path
    fd: int


def _report_runtime_repair_failure(repair: RuntimeRepairResult) -> None:
    if repair.backup_venv is None:
        print("  ℹ Managed Python runtime was not replaced; "
              f"the existing venv is unchanged ({repair.detail}).")
        print("    Sessions stay protected meanwhile: Hermes keeps databases "
              "out of WAL mode on this SQLite build. The next `hermes update` "
              "will retry.")
        return
    print(f"  ✗ Managed Python runtime cutover needs manual recovery: {repair.detail}")
    print(f"    Previous venv: {repair.backup_venv}")


class _UvResult(str):
    """``ensure_uv()`` return value that survives an update boundary. POSIX only: a str subclass
    with an overridden ``__iter__`` is unsafe as a Windows subprocess argument."""

    fresh_bootstrap: bool

    def __new__(cls, path: Optional[str], fresh: bool = False) -> "_UvResult":
        self = super().__new__(cls, path or "")
        self.fresh_bootstrap = fresh
        return self

    def __iter__(self):
        # Tuple-unpacking hook for legacy ``uv_bin, fresh = ensure_uv()`` sites; the first
        # element keeps the historical contract (path string, or None when unavailable).
        return iter(((str(self) or None), self.fresh_bootstrap))


def _ensure_uv_path(
    *, repair_observer: Callable[[RuntimeRepairResult], None] | None = None) -> Optional[str]:
    """Resolve the managed uv path, installing it if necessary (plain ``str``/``None``)."""
    existing = resolve_uv()
    if existing and _uv_runs(existing):
        return existing
    target = managed_uv_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"  → Installing managed uv into {target.parent} ...")
    try:
        _install_uv(target)
    except Exception as exc:
        logger.warning("Managed uv install failed: %s", exc)
        print(f"  ✗ Failed to install managed uv: {exc}")
        return None
    result = resolve_uv()
    if result:
        print(f"  ✓ Managed uv installed ({_uv_version(result)})")
        # Compatibility boundary: an older, already-imported updater calls the freshly pulled
        # ``ensure_uv()``; repairing here lets that first update migrate a vulnerable runtime.
        _run_runtime_repair(result, repair_observer)
    else:
        print("  ✗ Managed uv install appeared to succeed but binary not found")
    return result


def _uv_runs(uv_bin: str) -> bool:
    """``uv --version`` exits 0. A pre-fix installer could salvage a relocated Chocolatey/Scoop shim into
    ``$HERMES_HOME/bin``: it is a file with the executable bit that never runs, so is_file()+X_OK
    alone would keep handing it out forever instead of reinstalling."""
    try:
        return subprocess.run([uv_bin, "--version"], capture_output=True, check=False).returncode == 0
    except OSError:
        return False


def _uv_version(uv_bin: str) -> str:
    return subprocess.run(
        [uv_bin, "--version"],
        capture_output=True, text=True, encoding='utf-8', errors='replace', check=False,
    ).stdout.strip()


def _record_runtime_repair(repair: RuntimeRepairResult) -> None:
    """Put the repair outcome into the update receipt (no-op outside ``hermes update``).

    Receipts are built only from explicit ``record_step``/``record_skip`` calls, so without this
    a failed repair left ``outcome: partial`` with no step naming the reason or the SQLite
    versions. A deferred or not-applicable repair is a skip WITH its reason, not a failed step:
    every pip/non-venv install would otherwise carry a red step in every receipt.
    """
    from hermes_cli.update_receipt import record_skip, record_step

    detail = (
        f"{repair.status}: {repair.detail}" if repair.detail else repair.status
    ) + f" (sqlite {repair.sqlite_before or 'unknown'} → {repair.sqlite_after or 'unknown'})"
    if repair.status in {"skipped", "not-applicable"}:
        record_skip("sqlite_runtime_repair", detail)
    else:
        record_step("sqlite_runtime_repair", repair.status in {"safe", "repaired"}, detail)


def _run_runtime_repair(
    uv_bin: str, repair_observer: Callable[[RuntimeRepairResult], None] | None,
    *, print_skip: bool = False) -> None:
    """Run the vulnerable-runtime repair hook; never raises (repair is non-fatal)."""
    try:
        repair = repair_vulnerable_runtime(uv_bin)
        _record_runtime_repair(repair)
        if repair_observer is not None:
            repair_observer(repair)
        if repair.status == "failed":
            _report_runtime_repair_failure(repair)
    except Exception as exc:
        logger.warning("Managed Python runtime repair failed: %s", exc)
        if print_skip:
            print(f"  ⚠ Managed Python runtime repair skipped: {exc}")


def ensure_uv(
    *, repair_observer: Callable[[RuntimeRepairResult], None] | None = None):
    """Return the managed uv path, installing it first if necessary; falsy on failure, never raises.

    On POSIX the result is a :class:`_UvResult` (``str`` subclass) usable as the path *and*
    unpackable as ``(path, fresh_bootstrap)`` for older call sites.
    """
    result = _ensure_uv_path(repair_observer=repair_observer)
    if platform.system() == "Windows":
        # See _UvResult: the __iter__ override is unsafe as a Windows subprocess argument.
        return result
    return _UvResult(result)


def _uv_self_update_stamp() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "cache" / ".uv_self_update_stamp"


def _uv_self_update_is_fresh(now: float | None = None) -> bool:
    """True when ``uv self update`` ran recently enough to skip.

    uv releases roughly weekly while many users run ``hermes update`` daily; a blocking network
    self-update on every run is waste and, offline, an unbounded hang risk.
    """
    try:
        age = (now if now is not None else time.time()) - _uv_self_update_stamp().stat().st_mtime
        return 0 <= age < UV_SELF_UPDATE_INTERVAL_SECONDS
    except Exception:
        return False


def _touch_uv_self_update_stamp() -> None:
    with contextlib.suppress(OSError):
        stamp = _uv_self_update_stamp()
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.touch()


# uv ships releases ~weekly; refresh the managed binary at most this often.
UV_SELF_UPDATE_INTERVAL_SECONDS = 7 * 24 * 3600
# `uv self update` is a network call with no default timeout; unbounded it can hang forever.
UV_SELF_UPDATE_TIMEOUT_SECONDS = 60


def update_managed_uv(
    *, repair_observer: Callable[[RuntimeRepairResult], None] | None = None, force: bool = False
) -> Optional[str]:
    """Run ``uv self update`` on the managed uv binary; returns its path, or ``None`` if absent.

    The network self-update is skipped when it succeeded within ``UV_SELF_UPDATE_INTERVAL_SECONDS``
    unless ``force=True``; the vulnerable-runtime repair probe ALWAYS runs — CVE-driven repair is
    never gated behind the freshness stamp.
    """
    existing = resolve_uv()
    if not existing:
        # Not installed yet — ensure_uv() will handle that elsewhere.
        return None
    if force or not _uv_self_update_is_fresh():
        try:
            result = subprocess.run(
                [existing, "self", "update"], capture_output=True,
                text=True, encoding='utf-8', errors='replace',
                check=False, timeout=UV_SELF_UPDATE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            logger.debug("uv self update timed out after %ss", UV_SELF_UPDATE_TIMEOUT_SECONDS)
            result = None
        if result is not None and result.returncode == 0:
            _touch_uv_self_update_stamp()
            print(f"  ✓ Managed uv updated ({_uv_version(existing)})")
        elif result is not None:
            # Non-fatal — old uv still works fine.
            logger.debug("uv self update failed (rc=%d): %s", result.returncode, result.stderr)
    # Keep this hook inside the long-standing API: during an update main.py is already imported
    # from the old checkout and ``git pull`` replaces this module before the updater imports it,
    # so calling the repair here is what migrates the runtime on that first update. Non-fatal:
    # the live venv is untouched unless a fully prepared candidate reached cutover.
    _run_runtime_repair(existing, repair_observer, print_skip=True)
    return existing


def _reload_hermes_constants():
    """Re-execute ``hermes_constants`` from disk (the imported one may predate venv_python_path)."""
    import hermes_constants
    return importlib.reload(hermes_constants)


def _venv_python(venv_dir: Path) -> Path:
    try:
        from hermes_constants import venv_python_path
    except ImportError:
        venv_python_path = _reload_hermes_constants().venv_python_path
    return venv_python_path(venv_dir, windows=platform.system() == "Windows")


def _remove_tree(path: Path, *, boundary: Path) -> None:
    """Best-effort removal constrained to a known runtime boundary."""
    try:
        path.resolve().relative_to(boundary.resolve())
    except (OSError, ValueError):
        return
    shutil.rmtree(path, ignore_errors=True)


def _reject(path: Path, boundary: Path, msg: str, *args) -> None:
    """Log a rejected candidate and clean up its tree; always returns ``None``."""
    logger.warning(msg, *args)
    _remove_tree(path, boundary=boundary)
    return None


def _token() -> str:
    return f"{int(time.time())}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def _dotted(parts) -> str:
    return ".".join(str(p) for p in parts)


def _make_world_traversable(path: Path) -> None:
    """Keep root/FHS-managed runtimes executable by non-root callers."""
    with contextlib.suppress(OSError):
        path.chmod(path.stat().st_mode | 0o755)


def _runtime_request(info: SQLiteRuntimeInfo) -> str:
    """Pin the candidate to the current CPython minor line (e.g. ``3.11``): requesting the exact
    patch can never repair installs whose patch has no fixed-SQLite artifact at all."""
    return _dotted(info.python_version[:2])


# Cap on newer patches tried, newest-first, before giving up: each attempt is a real
# download+install+probe+delete cycle, and the fix is almost always in the next patch or two.
_MAX_PATCH_RETRIES = 5


def _list_available_patches(
    uv_bin: str, minor: str, *, cwd: Path, env: dict) -> list[tuple[int, int, int]]:
    """Known patch versions for ``minor`` (e.g. "3.11"), newest first; [] on any failure
    (network, parse), in which case callers fall back to the bare-minor request.

    Queries ``uv python list --all-versions`` rather than trusting the bare minor-line request to resolve to
    the newest patch (issue #71250: on some hosts/uv versions, the resolved candidate for a bare "3.11"
    request can be an older cached/indexed patch that still links a vulnerable SQLite, even when a newer
    non-vulnerable patch is available).
    """
    try:
        result = subprocess.run(
            [
                uv_bin, "python", "list", minor, "--all-versions", "--only-downloads",
                "--output-format", "json", "--no-config"],
            cwd=cwd, env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace", check=False, timeout=15)
        if result.returncode != 0 or not result.stdout.strip():
            return []
        versions: list[tuple[int, int, int]] = []
        for entry in json.loads(result.stdout):
            if not isinstance(entry, dict):
                continue
            # Only default/cpython builds -- skip pypy/graalpy/freethreaded variants.
            if entry.get("implementation") not in (None, "cpython") or (
                entry.get("variant") not in (None, "default")):
                continue
            parts = entry.get("version_parts") or {}
            try:
                versions.append(
                    (int(parts["major"]), int(parts["minor"]), int(parts["patch"])))
            except (KeyError, TypeError, ValueError):
                continue
        # Deduplicate (a version can repeat across platforms/arches) and sort newest-first.
        return sorted(set(versions), reverse=True)
    except Exception:
        return []


def _attempt_install_generation(
    uv_bin: str, request: str, *, project_root: Path, python_root: Path,
    current: SQLiteRuntimeInfo, allow_minor_upgrade: bool = False,
    tried_versions: set[tuple[int, int, int]] | None = None) -> _Provisioned | None:
    """One install+probe attempt for ``request`` (bare minor "3.11" or explicit patch "3.11.15").

    Each attempt gets its own generation directory so a rejected candidate is fully cleaned up
    before the next attempt (--reinstall semantics). Returns None (and cleans up) on any failure.
    """
    generation = python_root / f"generation-{_token()}"
    generation.mkdir(parents=True, exist_ok=False)
    _make_world_traversable(generation)

    reject = partial(_reject, generation, python_root)
    env = managed_python_env(project_root, install_dir=generation)
    run = dict(cwd=project_root, env=env, capture_output=True, text=True,
               encoding="utf-8", errors="replace", check=False)
    install = subprocess.run(
        [uv_bin, "python", "install", request, "--reinstall", "--no-bin", "--no-registry",
         "--no-config"],
        **run)
    if install.returncode != 0:
        return reject(
            "private Python install failed for %s (rc=%d): %s",
            request, install.returncode, (install.stderr or install.stdout or "").strip())
    found = subprocess.run(
        [uv_bin, "python", "find", request, "--managed-python", "--no-config"], **run)
    if found.returncode != 0 or not found.stdout.strip():
        return reject(
            "private Python lookup failed for %s (rc=%d): %s",
            request, found.returncode, (found.stderr or "").strip())
    python = Path(found.stdout.strip().splitlines()[-1])
    try:
        python.resolve().relative_to(generation.resolve())
    except (OSError, ValueError):
        return reject("uv resolved Python outside the Hermes generation: %s", python)
    # Sign before the candidate is probed or promoted so each immutable generation does not look
    # like a new TCC principal on macOS. Non-fatal: the SQLite repair proceeds regardless.
    _macos_sign_managed_python(python)
    candidate = probe_sqlite_runtime(python)
    if candidate is None:
        return reject("could not probe candidate Python runtime: %s", python)
    if tried_versions is not None:
        tried_versions.add(candidate.python_version[:3])
    if allow_minor_upgrade:
        # Falling forward to a higher minor line: only reject downgrades.
        if candidate.python_version < current.python_version:
            return reject(
                "candidate Python downgraded from %s: %s",
                _dotted(current.python_version), candidate.python_version)
    elif candidate.python_version[:2] != current.python_version[:2] or (
        candidate.python_version < current.python_version):
        return reject(
            "candidate Python drifted off the %s minor line or downgraded: %s",
            _dotted(current.python_version[:2]), candidate.python_version)
    if candidate.wal_reset_vulnerable:
        return reject(
            "candidate Python still links vulnerable SQLite %s (%s)",
            candidate.sqlite_version_string, candidate.sqlite_source_id)
    return generation, python, candidate


def _retry_explicit_patches(
    uv_bin: str, request: str, *, project_root: Path, python_root: Path,
    current: SQLiteRuntimeInfo, tried: set[tuple[int, int, int]],
    allow_minor_upgrade: bool = False, skip_at_or_below: tuple[int, int, int] | None = None,
) -> _Provisioned | None:
    """Retry ``request``'s minor line with explicit patches, newest-first, at most
    ``_MAX_PATCH_RETRIES`` attempts, skipping versions already in ``tried`` (a certain rejection
    still costs a full download+install+probe+delete cycle).

    ``skip_at_or_below`` also skips patches at or below that version: only NEWER patches can carry
    the fix and the downgrade guard rejects the rest; on a stale uv catalog the newest indexed
    patch can be the installed one, and the loop would burn every retry walking backwards.
    """
    # The bare minor-line request resolved to a still-vulnerable (or otherwise rejected) candidate. Rather
    # than giving up immediately, query which patches on this minor line uv actually knows about and retry
    # with explicit newer versions, newest-first -- this handles the case where the default resolution for a
    # bare request picks an older cached/indexed patch even though a newer, non-vulnerable one is available
    # (issue #71250).
    env_for_list = managed_python_env(project_root, install_dir=python_root)
    patches = _list_available_patches(uv_bin, request, cwd=project_root, env=env_for_list)
    attempts = 0
    for version_tuple in patches:
        if attempts >= _MAX_PATCH_RETRIES:
            break
        if version_tuple in tried:
            continue
        if skip_at_or_below is not None and version_tuple <= skip_at_or_below:
            continue
        tried.add(version_tuple)
        explicit = _dotted(version_tuple)
        print(f"  → Retrying with explicit patch {explicit}...")
        attempts += 1
        result = _attempt_install_generation(
            uv_bin, explicit, project_root=project_root,
            python_root=python_root, current=current,
            allow_minor_upgrade=allow_minor_upgrade)
        if result is not None:
            return result
    return None


def _provision_line(
    uv_bin: str, request: str, *, tried: set[tuple[int, int, int]],
    allow_minor_upgrade: bool = False, skip_at_or_below: tuple[int, int, int] | None = None,
    **common) -> _Provisioned | None:
    """Try ``request`` once, then its explicit newer patches; None when the whole line fails."""
    result = _attempt_install_generation(
        uv_bin, request, tried_versions=tried, allow_minor_upgrade=allow_minor_upgrade, **common)
    if result is None:
        result = _retry_explicit_patches(
            uv_bin, request, tried=tried, allow_minor_upgrade=allow_minor_upgrade,
            skip_at_or_below=skip_at_or_below, **common)
    return result


def _install_safe_python_generation(
    uv_bin: str, *, project_root: Path, current: SQLiteRuntimeInfo) -> _Provisioned | None:
    runtime_root = project_root / _RUNTIME_DIR_NAME
    python_root = managed_python_install_dir(project_root)
    _make_world_traversable(runtime_root)
    _make_world_traversable(python_root)
    common = dict(project_root=project_root, python_root=python_root, current=current)

    request = _runtime_request(current)
    print(f"  → Provisioning a private Python {request} runtime with fixed SQLite...")
    tried_versions = {current.python_version[:3]}
    # If the bare minor-line request resolves to a still-vulnerable (or otherwise rejected)
    # candidate, the default resolution may have picked an older cached/indexed patch even though
    # a newer, non-vulnerable one exists: retry with explicit newer patches, newest-first.
    result = _provision_line(
        uv_bin, request, tried=tried_versions, skip_at_or_below=current.python_version[:3], **common
    )
    if result is not None:
        return result
    # All patches on the current minor line are vulnerable or rejected. Fall forward to the next
    # supported minor (e.g. 3.11 → 3.12) so the user isn't stuck on every `hermes update`. The
    # requires-python window (>=3.11,<3.14) and the import smoke-test gate compatibility.
    # See #76106.
    cur_major, cur_minor = current.python_version[:2]
    fb_tried: set[tuple[int, int, int]] = set(tried_versions)
    for next_minor in range(cur_minor + 1, 14):  # up to 3.13
        next_request = f"{cur_major}.{next_minor}"
        print(
            f"  → No fixed {cur_major}.{cur_minor} build available; "
            f"trying {next_request} as fallback...")
        result = _provision_line(
            uv_bin, next_request, tried=fb_tried, allow_minor_upgrade=True, **common)
        if result is not None:
            return result
    return None


def _smoke_candidate_venv(venv_dir: Path) -> tuple[bool, str, SQLiteRuntimeInfo | None]:
    """Exercise the candidate interpreter and imports through its real path."""
    python = _venv_python(venv_dir)
    info = probe_sqlite_runtime(python)
    if info is None:
        return False, f"could not execute {python}", None
    if info.wal_reset_vulnerable:
        return False, f"candidate still links vulnerable SQLite {info.sqlite_version_string}", info
    check = (
        "import dotenv, fastapi, openai, prompt_toolkit, pydantic, rich, uvicorn, yaml\n"
        "import hermes_state\n")
    try:
        result = subprocess.run(
            [str(python), "-I", "-c", check], cwd=venv_dir.parent, env=isolated_interpreter_env(),
            capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=90, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc), info
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "core import smoke failed").strip()
        return False, detail.splitlines()[-1] if detail else "core import smoke failed", info
    return True, "", info


# A failed ``uv sync`` prints its diagnosis last, so the tail is the actionable part. Kept
# short: the reason travels into a one-line log entry, the failure report and the receipt step.
_SYNC_TAIL_LINES = 6
_SYNC_REASON_CHARS = 600


def _sync_reason(tail: deque[str]) -> str:
    """The actionable part of a failed sync: uv's ``error:`` line and whatever follows it.

    uv prints progress ("Resolving…", "Resolved 259 packages") before the diagnosis, so the raw
    tail leads with noise; the ``error:``/``hint:`` pair is the part a user can act on.
    """
    parts = [line for line in tail if line.strip()]
    for index, line in enumerate(parts):
        if line.lower().startswith(("error:", "error ")):
            parts = parts[index:]
            break
    else:
        parts = parts[-2:]
    return " | ".join(parts).strip()[:_SYNC_REASON_CHARS]


def _stream_sync(argv: list[str], *, cwd: Path, env: dict[str, str]) -> tuple[int, str]:
    """Run the candidate's locked sync, forwarding output live; return ``(rc, reason)``.

    Streaming is load-bearing, not cosmetic: older desktop update hand-offs drain only the
    child's stdout while it runs, so a full stderr pipe blocks uv forever — stderr is merged
    into stdout and forwarded line by line instead of being captured and reprinted at the end.

    The tail is kept anyway: with inherited stdout the child's diagnosis survived in console
    scrollback only, and the rejection carried a bare exit code — "hermes update says the SQLite
    repair failed and never says why".
    """
    proc = subprocess.Popen(
        list(argv), cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1)
    tail: deque[str] = deque(maxlen=_SYNC_TAIL_LINES)
    stream = proc.stdout
    if stream is not None:
        for line in stream:
            tail.append(line.rstrip())
            sys.stdout.write(line)
            sys.stdout.flush()
    status = proc.wait()
    return status, _sync_reason(tail)


class _CandidateStageError(Exception):
    """A rejected candidate, already cleaned up, with its diagnostic reason."""


def _stage_candidate_venv(
    uv_bin: str, *, project_root: Path, generation: Path, python: Path) -> Path:
    runtime_root = project_root / _RUNTIME_DIR_NAME
    candidate = runtime_root / f"venv-candidate-{_token()}"
    env = managed_python_env(project_root, install_dir=generation)
    env.update({
        "UV_PROJECT_ENVIRONMENT": str(candidate), "UV_PYTHON": str(python),
        "UV_PYTHON_DOWNLOADS": "never", "VIRTUAL_ENV": str(candidate)})

    def reject(message: str, *args) -> None:
        _reject(candidate, runtime_root, message, *args)
        raise _CandidateStageError(message % args if args else message)
    print("  → Building a relocatable replacement environment...")
    created = subprocess.run(
        [
            uv_bin, "venv", str(candidate), "--python", str(python),
            "--managed-python", "--no-python-downloads", "--relocatable", "--no-config"],
        cwd=project_root, env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=False)
    if created.returncode != 0:
        return reject(
            "candidate venv creation failed (rc=%d): %s",
            created.returncode, (created.stderr or created.stdout or "").strip())
    if not (project_root / "uv.lock").is_file():
        return reject("candidate dependency sync refused: uv.lock is missing")
    # Locked sync must see project [tool.uv] exclude-newer; --no-config / UV_NO_CONFIG drops it
    # and uv 0.12+ refuses --locked.
    sync_env = dict(env)
    sync_env.pop("UV_NO_CONFIG", None)
    # stderr=STDOUT: uv writes progress to stderr. Legacy desktop
    # hand-offs (pre scripts/desktop-update/windows.ps1, which drains
    # both pipes) only drain the child's stdout while the child runs; a
    # full stderr pipe (~64KB) blocks uv forever. Merging into stdout
    # keeps the output streaming through the pipe old hand-offs DO
    # drain. This module is imported lazily by update_cmd AFTER the git
    # reset, so even an update running from an old base executes THIS
    # copy — unlike the heartbeat helper (main_install_repair.py), which
    # is imported at startup and only protects bases that ship its twin.
    status, reason = _stream_sync(
        [uv_bin, "sync", "--extra", "all", "--locked", "--python", str(_venv_python(candidate))],
        cwd=project_root, env=sync_env)
    if status != 0:
        # The reason travels with the rejection into RuntimeRepairResult.detail, which the
        # failure report prints and the update receipt records.
        return reject("candidate dependency sync failed (rc=%d): %s", status, reason)
    healthy, detail, _ = _smoke_candidate_venv(candidate)
    if not healthy:
        return reject("candidate venv smoke failed: %s", detail)
    return candidate


def _rename_with_retry(source: Path, destination: Path) -> None:
    for delay in (0.0, 0.1, 0.25, 0.5, 1.0):
        if delay:
            time.sleep(delay)
        try:
            source.rename(destination)
            return
        except OSError as exc:
            last_error = exc
    raise last_error


def _cut_over_candidate(
    candidate: Path, *, project_root: Path, live: Path | None = None
) -> tuple[bool, Path | None, SQLiteRuntimeInfo | None, str]:
    live = live if live is not None else project_root / _VENV_NAME
    runtime_root = project_root / _RUNTIME_DIR_NAME
    token = _token()
    backup = live.with_name(f"{live.name}.stale.runtime-{token}")
    rejected = runtime_root / f"venv-rejected-{token}"
    try:
        try:
            _rename_with_retry(live, backup)
        except OSError as exc:
            return False, None, None, f"could not park the existing venv: {exc}"
        try:
            _rename_with_retry(candidate, live)
        except OSError as promote_error:
            try:
                _rename_with_retry(backup, live)
            except OSError as rollback_error:
                return False, backup, None, (
                    "could not promote the replacement venv "
                    f"({promote_error}); rollback failed ({rollback_error})")
            return False, None, None, f"could not promote the replacement venv: {promote_error}"
        try:
            healthy, detail, info = _smoke_candidate_venv(live)
        except Exception as exc:
            healthy, detail, info = False, f"candidate smoke raised: {exc}", None
        if healthy:
            return True, backup, info, ""
        try:
            _rename_with_retry(live, rejected)
            _rename_with_retry(backup, live)
        except OSError as exc:
            return False, backup, info, (
                "post-cutover smoke failed "
                f"({detail}); rollback failed ({exc}); rejected venv: {rejected}")
        _remove_tree(rejected, boundary=runtime_root)
        return False, None, info, f"post-cutover smoke failed: {detail}"
    except BaseException:
        if not live.exists() and backup.exists():
            try:
                _rename_with_retry(backup, live)
            except OSError as exc:
                logger.error(
                    "interrupted runtime cutover could not restore %s from %s: %s",
                    live, backup, exc)
        raise


def _replace_file_atomically(path: Path, data: bytes) -> None:
    """Replace *path* from a same-directory temporary file."""
    token = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
    temporary = path.with_name(f".{path.name}.runtime-{token}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _cut_over_windows_runtime_config(
    candidate: Path,
    *,
    install_dir: Path | None = None,
    base_env: dict[str, str] | None = None,
    **kwargs,
) -> NoReturn:
    # Shim to stop the old updater doing work until relaunch, not prepare a child.
    stop_for_relaunch()


def rebuild_venv(
    uv_bin: str, venv_dir: Path, python_version: str = "3.11", **kwargs
) -> NoReturn:
    # Shim to stop the old updater doing work until relaunch, not claim a rebuild.
    stop_for_relaunch()
