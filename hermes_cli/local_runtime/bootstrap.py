"""Bootstrap for the managed runtime: config -> installed binaries -> running supervised server.

One public call, ``ensure_local_runtime(config)``, safe at any session start: disabled or
already-running -> no-op; enabled -> serve the installed build under a supervisor. Kept
import-light: callers gate on config before importing so disabled sessions never pay the import.
"""

from __future__ import annotations

from contextlib import contextmanager, suppress
import logging
import os
import time
from pathlib import Path

from hermes_cli.local_runtime.binaries import runtimes_root
from hermes_cli.local_runtime.gguf import SPLIT_PART_RE, model_id_from_stem

logger = logging.getLogger(__name__)

_SUPERVISOR = None  # process-wide singleton; one router per Hermes process
# The engine that singleton runs (or is booting). Budgets price ITS devices: the server is shared by
# every profile this process hosts, so another profile's ``local_runtime.backend`` must not resize it.
_SERVING_ENGINE = None
_EXIT_HOOKED = False


def _stop_at_exit() -> None:
    """The process that boots the server stops it on a clean exit, whatever surface it is (the
    desktop backend also stops it in its shutdown handler; the second call is a no-op). Processes
    that only adopted another's server never stop it. A hard kill still leaves an orphan, which the
    next boot adopts or replaces."""
    global _EXIT_HOOKED
    if not _EXIT_HOOKED:
        import atexit

        atexit.register(shutdown_local_runtime)
        _EXIT_HOOKED = True


def _detect_gpu_vendor() -> str | None:
    """Best-effort GPU vendor for backend selection.

    Preserve NVIDIA's detailed name when nvidia-smi is available, then use the
    platform host facts so Intel and AMD do not silently demote ``auto`` to CPU.
    """
    from hermes_cli.local_runtime.hardware import _cached_nvidia_gpu_query
    from hermes_platform.host.facts import gpu_class

    query = _cached_nvidia_gpu_query()
    if query is not None and query.get("gpu_name"):
        return "nvidia " + query["gpu_name"]
    detected = gpu_class()
    return detected if detected in {"nvidia", "amd", "intel"} else None


def models_dir() -> Path:
    """Machine-scoped, deliberately NOT profile-scoped: a 20 GB GGUF is a machine asset, and every
    profile shares the one managed server that serves it (same rule as runtimes_root())."""
    from hermes_constants import get_default_hermes_root

    return get_default_hermes_root() / "models"


def assets_dir() -> Path:
    """Non-model companion files (mmproj projectors, spec-decode drafts). A subdirectory so the
    router's model listing — and staged_models() — never mistakes an asset for a servable model."""
    return models_dir() / "assets"


def staged_in(models_dir: Path, *, require_complete: bool = True) -> "list[Path]":
    """Servable GGUFs in a directory: single files, plus split GGUFs once by their first part.
    With ``require_complete`` a split counts only when EVERY part is on disk — a mid-download split
    is not servable and must not surface anywhere as a model."""
    files = sorted(models_dir.glob("*.gguf"))
    names = {p.name for p in files}
    out = []
    for p in files:
        m = SPLIT_PART_RE.search(p.name)
        if m is None:
            out.append(p)
            continue
        if m.group(1) != "00001":
            continue
        stem, total = p.name[: m.start()], int(m.group(2))
        if not require_complete or all(f"{stem}-{i:05d}-of-{m.group(2)}.gguf" in names
                                       for i in range(2, total + 1)):
            out.append(p)
    return out


def adopt_legacy_models() -> "list[Path]":
    """Move GGUFs left in the old per-profile ``<profile home>/models`` layout (and its assets/)
    into the machine-scoped dirs, so everything downstream keeps reading one directory.

    ``os.rename`` only: within one filesystem it is instant even for a 20 GB model, while
    ``shutil.move`` silently degrades to a copy across devices. A cross-device profile dir is left
    in place with a warning rather than copying tens of GB at session start. A name that already
    exists in the destination is left alone (check-then-rename: the only window is two processes
    adopting two profiles' same-named file at once, and a same name is the same catalog variant).
    Two processes racing on one file are harmless: the loser's rename finds the source gone and
    skips it. Returns the new paths of the moved files."""
    from hermes_constants import get_default_hermes_root, named_profile_has_identity

    profiles_root = get_default_hermes_root() / "profiles"
    if not profiles_root.is_dir():
        return []
    moved: list[Path] = []
    for home in sorted(profiles_root.iterdir()):
        old = home / "models"
        if home.name.startswith(".") or not old.is_dir() or not named_profile_has_identity(home):
            continue
        for src_dir, dest_dir in ((old, models_dir()), (old / "assets", assets_dir())):
            for src in sorted(src_dir.glob("*.gguf")):
                dest = dest_dir / src.name
                if dest.exists():
                    logger.warning("legacy model %s not moved: %s already exists", src, dest)
                    continue
                try:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    os.rename(src, dest)
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    logger.warning("legacy model %s not moved to %s: %s", src, dest_dir, exc)
                    continue
                moved.append(dest)
        for emptied in (old / "assets", old):
            with suppress(OSError):
                emptied.rmdir()
    if moved:
        logger.info("moved %d legacy model file(s) into %s", len(moved), models_dir())
    return moved


def extra_model_dirs(section: dict | None = None) -> "list[Path]":
    """``local_runtime.model_dirs``: existing read-only GGUF roots served beside ``models_dir()``
    (downloads and deletes stay in ``models_dir()``). ``section`` defaults to the live config."""
    if section is None:
        from hermes_cli.config import load_config_readonly

        section = load_config_readonly().get("local_runtime") or {}
    raw = section.get("model_dirs") if isinstance(section, dict) else None
    dirs = (Path(os.path.expanduser(str(d).strip())) for d in (raw if isinstance(raw, list) else []) if str(d).strip())
    out = []
    for d in dirs:
        if d.is_dir():
            out.append(d)
        else:
            _warn_once(f"local_runtime.model_dirs entry {d} is not a directory; ignoring it")
    return out


_WARNED: set[str] = set()


def _warn_once(message: str) -> None:
    if message not in _WARNED:
        _WARNED.add(message)
        logger.warning("%s", message)


def staged_across(models_dir: Path, extra_dirs: "tuple[Path, ...] | list[Path]" = ()) -> "list[Path]":
    """Servable GGUFs in ``models_dir`` then ``extra_dirs``, one per model id. The preset INI and the
    router name a model by its id, so a root listed twice is walked once and a later file whose id
    is already taken is skipped (the managed dir wins) rather than written as a second section."""
    roots: set[str] = set()
    ids: set[str] = set()
    out = []
    for root in (models_dir, *extra_dirs):
        key = os.path.normcase(str(root.resolve()))
        if key in roots:
            continue
        roots.add(key)
        for gguf in staged_in(root):
            model_id = model_id_from_stem(gguf.stem)
            if model_id in ids:
                _warn_once(f"local_runtime.model_dirs: {gguf} is shadowed by another {model_id!r}; ignoring it")
                continue
            ids.add(model_id)
            out.append(gguf)
    return out


def server_binary(section: dict) -> "tuple[Path, object | None] | None":
    """The llama-server to supervise: ``local_runtime.executable_path`` when set, else the installed
    PM engine's binary (with the engine), else None. The on-demand boot gate asks this too, so a
    configured executable boots without a PM engine."""
    executable = str(section.get("executable_path") or "").strip()
    if executable:
        return Path(os.path.expanduser(executable)), None
    from hermes_cli.local_runtime.binaries import installed_engine

    engine = installed_engine(section.get("backend") or "auto")
    return (engine.binary, engine) if engine is not None else None


def staged_models() -> "list[Path]":
    """Servable staged models (continuation parts, incomplete splits and assets/ never count)."""
    return staged_across(models_dir(), extra_model_dirs())


def staged_model_ids() -> "list[str]":
    return [model_id_from_stem(p.stem) for p in staged_models()]


def _presets_stale() -> bool:
    """True when a staged model has no section in the preset INI — it would autoload with stock
    fit instead of a policy decision."""
    with suppress(Exception):
        from hermes_cli.local_runtime.presets import read_preset_decisions

        known = read_preset_decisions()
        return any(mid not in known or (not known[mid].refusal and not (known[mid].keys or {}).get("model"))
                   for mid in staged_model_ids())
    return False


def _stop_state_server() -> None:
    """Stop the server the state file points at before this process boots a replacement, but only
    an orphan (recorded owner gone): a live owner's watchdog would respawn its router, so that one
    is left running and the replacement boots beside it, as it always has. The process comes from
    the identity-guarded reader, never a bare PID — the endpoint dict callers hold has none."""
    from hermes_cli.local_runtime.recovery import (
        _owner_is_dead,
        is_modern,
        legacy_recorded_process,
        read_state,
        recorded_process,
    )
    from hermes_cli.local_runtime.supervisor import LlamaServerSupervisor

    state = read_state()
    if is_modern(state):
        if not _owner_is_dead(state):
            logger.info("llama-server pid=%s belongs to a live Hermes process; leaving it", state.get("pid"))
            return
        proc = recorded_process(state)
    else:
        proc = legacy_recorded_process(state)
    if proc is None:
        return
    try:
        # Waits for the tree to exit, so the port and the GPU are free for the replacement.
        LlamaServerSupervisor._terminate_tree(proc, verified_root=True)
    except Exception as exc:
        logger.warning("could not stop the incumbent llama-server (pid=%s): %s", proc.pid, exc)


def refresh_local_runtime() -> bool:
    """Restart the managed server so it rescans the models directory. The router's model list is
    SPAWN-ONLY: a GGUF added after start is invisible to GET /models and 400s on completion, so
    anything that changes the staged set while the server runs must bounce it."""
    global _SUPERVISOR
    try:
        from hermes_cli.config import load_config

        if _SUPERVISOR is None:
            from hermes_cli.local_runtime.endpoint import _state_endpoint

            state = _state_endpoint()
            if state is None:
                return False
            _stop_state_server()
        else:
            shutdown_local_runtime()
        return ensure_local_runtime(load_config(), force=True) is not None
    except Exception as exc:
        logger.warning("local runtime refresh failed: %s", exc)
        return False


def _admitted_models_max(mdir: Path, configured: int, extra_dirs: "tuple[Path, ...] | list[Path]" = ()) -> int:
    """Residency cap to hand the router: derived from the hardware budget, ``models_max`` as a ceiling.

    A cap of "four" on a card that holds one model is how a second child ends up paged (WDDM) and
    silently slow — llama.cpp evicts its LRU before an incoming load only when the cap says the
    card is full. A probe miss or an unpriceable model keeps the configured count: this must never
    block a boot.
    """
    try:
        from hermes_cli.local_runtime.hardware import probe_budget
        from hermes_cli.local_runtime.presets import admitted_residency_count

        cap = admitted_residency_count(mdir, probe_budget(planning=True), configured, extra_dirs=extra_dirs)
    except Exception as exc:
        logger.warning("residency cap probe failed (%s); using models_max=%s", exc, configured)
        return configured
    if cap != configured:
        logger.info("residency cap: %s resident model(s) on this card (models_max=%s)",
                    cap, configured)
    return cap


def _generate_presets(mdir: Path, preset_path: Path, section: dict | None = None) -> Path | None:
    """Write the launch-policy INI for every staged model; returns the path to hand the router.

    Priced against CAPACITY, not live free VRAM: this runs while the outgoing server instance may
    still hold the card (restart, refresh after a download), and its memory is freed before the new
    instance loads anything. Pricing against live-free once pinned a fitting model's weights to CPU.

    The window is then narrowed to what fits beside other programs' GPU memory
    (``hardware.launch_budget``), which never moves weights to the CPU.

    Degradation ladder on failure: a STALE policy still beats no policy — stock fit (f16 KV at max
    context, no placement) is the silent-busy-wait failure on Windows. Keep serving with the
    previous INI when one exists; only a first boot with no INI at all falls to stock fit.
    """
    from hermes_cli.local_runtime.hardware import probe_budget
    from hermes_cli.local_runtime.presets import generate_presets

    try:
        capacity = probe_budget(planning=True)
        for entry in generate_presets(mdir, capacity, preset_path, live=_launch_budget(capacity),
                                      **_preset_inputs(section)):
            if entry.refusal:
                logger.warning("model refused by physics check: %s", entry.refusal)
        return preset_path
    except Exception as exc:
        if preset_path.exists():
            logger.error("preset generation failed (%s); serving with the "
                         "PREVIOUS launch policies — models staged since "
                         "the last successful generation run unpoliced "
                         "until this is fixed", exc)
            return preset_path
        logger.error("preset generation failed (%s) and no previous "
                     "policy file exists; router runs stock fit", exc)
        return None


def _preset_inputs(section: dict | None = None) -> dict:
    """The ``local_runtime`` keys every preset plan reads (``model_dirs``, ``model_overrides``), so
    a boot and an idle refit write the same sections. ``section`` defaults to the live config."""
    if section is None:
        from hermes_cli.config import load_config_readonly

        section = load_config_readonly().get("local_runtime") or {}
    overrides = section.get("model_overrides") if isinstance(section, dict) else None
    return {"extra_dirs": extra_model_dirs(section),
            "overrides": overrides if isinstance(overrides, dict) else None}


def _launch_budget(capacity, own_bytes: int = 0):
    """``hardware.launch_budget``, or None when the probe fails: a boot never waits on it."""
    from hermes_cli.local_runtime.hardware import launch_budget

    try:
        return launch_budget(capacity, own_bytes=own_bytes)
    except Exception as exc:
        logger.warning("free GPU memory probe failed (%s); launch windows use capacity", exc)
        return None


# Router statuses that hold GPU memory. The preset file changes only while every model is out of
# them: the router unloads a loaded model whose launch flags change on reload.
_HOLDS_MEMORY = frozenset({"loaded", "loading", "sleeping", "ready"})


def refit_idle_presets(sup) -> bool:
    """Re-plan launch windows against free GPU memory while no model is loaded.

    Boot plans every window once, but models load later (on demand, after an idle unload), and by
    then other programs may hold more or less of the card. Returns True when the router was given
    new presets. A request that starts a load between the last status check and the reload is
    unloaded by the router and fails once; that gap is a few milliseconds, and only when a window
    changes.
    """
    from hermes_cli.local_runtime.hardware import probe_budget
    from hermes_cli.local_runtime.presets import plan_presets, render_presets

    path = sup.preset_path
    if path is None or not path.exists() or _any_holds_memory(sup):
        return False
    capacity = probe_budget(planning=True)
    live = _launch_budget(capacity)
    if live is None:
        return False
    last = sup._refit_usable
    if last is not None and abs(live.usable_vram_bytes - last) < _REFIT_STEP_BYTES:
        return False
    text = render_presets(plan_presets(models_dir(), capacity, live=live, **_preset_inputs()))
    # Held across the write and the reload so a restart can't spawn between them; everything
    # slow ran above.
    with sup._lifecycle_lock:
        if sup.proc is None or sup.proc.poll() is not None or _any_holds_memory(sup):
            return False
        sup._refit_usable = live.usable_vram_bytes
        if text == path.read_text(encoding="utf-8-sig"):
            return False
        from utils import atomic_write_text

        atomic_write_text(path, text, tmp_prefix=f".{path.name}_", mode=0o600)
        sup.reload_presets()
    logger.info("launch windows re-planned for %.1f GiB of free GPU memory",
                live.usable_vram_bytes / (1 << 30))
    return True


# Smaller changes in free memory don't move a window by a ladder rung; skipping them keeps the
# idle loop from re-reading every model header each pass.
_REFIT_STEP_BYTES = 256 << 20


def _any_holds_memory(sup) -> bool:
    return any(status in _HOLDS_MEMORY for status in sup.models(timeout_s=5).values())


def _try_lock_boot_fd(fd: int) -> bool:
    """Non-blocking exclusive attempt; portable across fcntl/msvcrt."""
    if os.name == "nt":
        import msvcrt

        if os.fstat(fd).st_size == 0:
            os.write(fd, b"\0")
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    else:
        import fcntl

        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False


def _unlock_boot_fd(fd: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)


@contextmanager
def _cross_process_boot_lock(timeout_s: float = 130.0):
    """Serialize the state-check-then-spawn sequence across every Hermes process on this
    machine — the ``_SUPERVISOR`` singleton above only rules out a race within ONE process.
    Two profiles booting in the same second each see no ``server.json`` yet and each spawn a
    router on the stable port (#116682); an OS-held lock makes the second caller wait for the
    first to publish its state file, so it re-checks and adopts instead of spawning a duplicate.
    Bounded, not indefinite: never hang session start dead if the lock is somehow stuck, and
    never raise into session start: an unwritable runtimes dir (or a foreign-owned lock file)
    proceeds unlocked with a warning, like the contention timeout."""
    path = runtimes_root() / "boot.lock"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    except OSError as exc:
        logger.warning("boot lock unavailable (%s); proceeding without it", exc)
        yield
        return
    try:
        deadline = time.monotonic() + timeout_s
        while not _try_lock_boot_fd(fd):
            if time.monotonic() >= deadline:
                logger.warning("boot lock contended past %.0fs; proceeding without it", timeout_s)
                break
            time.sleep(0.2)
        try:
            yield
        finally:
            with suppress(OSError):
                _unlock_boot_fd(fd)
    finally:
        os.close(fd)


def ensure_local_runtime(config: dict, force: bool = False) -> "object | None":
    """Idempotent boot of the managed runtime. Returns the supervisor (or None when
    disabled/unavailable). Never raises into a session start — failures log and return None; chat
    falls back to configured providers."""
    global _SUPERVISOR, _SERVING_ENGINE
    section = (config or {}).get("local_runtime") or {}
    if not force and not section.get("enabled"):
        return None
    if _SUPERVISOR is not None:
        return _SUPERVISOR

    try:
        adopt_legacy_models()
    except OSError as exc:  # an unreadable profiles dir must not block serving what's staged
        logger.warning("legacy model adoption failed: %s", exc)
    # Residency: no staged models means nothing to serve — don't boot an empty server (delete
    # your last model and boots stop). force boots as ever.
    if not force and not staged_models():
        logger.info("local runtime enabled but no models staged; not booting")
        return None

    # Another Hermes process may already be supervising — reuse via state, but ONLY while its
    # launch policy still covers every staged model. A server whose preset file predates a
    # download serves the new model with no policy at all (--models-autoload + stock fit). A stale
    # incumbent gets stopped and replaced by a fresh boot with regenerated presets; sessions ride
    # through like any other supervised restart (stable port + persisted key).
    #
    # The state check and the spawn below run under a cross-process lock: two backends racing
    # to boot (#116682) must not both find no state file and both spawn a router on the stable
    # port — the loser waits here, then re-checks state and adopts the winner's server instead.
    from hermes_cli.local_runtime.endpoint import _state_endpoint

    with _cross_process_boot_lock():
        state = _state_endpoint()
        if state is not None:
            if not _presets_stale():
                logger.info("managed llama-server already running (another process)")
                return None
            logger.info("running server's presets predate the staged models; "
                        "replacing it so every model launches with a policy")
            _stop_state_server()

        try:
            from hermes_cli.local_runtime.supervisor import LlamaServerSupervisor

            served = server_binary(section)
            if served is None:
                logger.info("local runtime enabled but no PM engine installed; use the Local Models pane")
                return None
            binary, engine = served
            _SERVING_ENGINE = engine

            mdir = models_dir()
            mdir.mkdir(parents=True, exist_ok=True)
            preset_path = _generate_presets(mdir, runtimes_root() / "presets.ini", section)

            sup = LlamaServerSupervisor(binary, mdir, preset_path=preset_path,
                                        models_max=_admitted_models_max(
                                            mdir, int(section.get("models_max", 4)),
                                            extra_model_dirs(section)),
                                        port=int(section.get("port", 0)) or None)
            try:
                sup.start()
            except Exception:
                # start() can fail after the router process exists (health timeout): leaving it
                # running unsupervised strands its VRAM behind a port nothing will clean up.
                with suppress(Exception):
                    sup.stop()
                raise
            _SUPERVISOR = sup
            _stop_at_exit()
            logger.info("managed llama-server up at %s (backend=%s tag=%s)", sup.base_url,
                        engine.backend if engine else "executable_path", engine.tag if engine else binary)
            _start_idle_sweeper(sup)
            return sup
        except Exception as exc:
            _SERVING_ENGINE = None
            logger.warning("managed local runtime unavailable: %s", exc)
            return None


def shutdown_local_runtime() -> None:
    global _SUPERVISOR, _SERVING_ENGINE
    if _SUPERVISOR is not None:
        _SUPERVISOR.stop()
        _SUPERVISOR = None
    _SERVING_ENGINE = None


def get_supervisor():
    """The process-local supervisor, or None (a server may still run under another process —
    check the state file)."""
    return _SUPERVISOR


def serving_engine():
    """The engine the process-local server runs or is booting, or None when there is none."""
    return _SERVING_ENGINE


_SWEEP_EVERY_S = 120
_REFIT_EVERY_S = 30


def _start_idle_sweeper(sup) -> None:
    """Idle-residency loop: every couple of minutes, unload models idle past the supervisor's
    threshold; every 30 s with nothing loaded, re-plan launch windows against free GPU memory.
    Daemon thread tied to the supervisor's lifetime — exits when the server stops."""
    import threading

    def _loop():
        last_sweep = time.monotonic()
        while sup.proc is not None and sup.proc.poll() is None:
            time.sleep(_REFIT_EVERY_S)
            if time.monotonic() - last_sweep >= _SWEEP_EVERY_S:
                last_sweep = time.monotonic()
                try:
                    sup.sweep_idle()
                except Exception as exc:
                    logger.debug("idle sweep skipped: %s", exc)
            try:
                refit_idle_presets(sup)
            except Exception as exc:
                logger.debug("launch window re-plan skipped: %s", exc)

    threading.Thread(target=_loop, daemon=True, name="local-runtime-idle-sweep").start()
