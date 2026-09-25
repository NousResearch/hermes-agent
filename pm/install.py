"""ensure(): make the installed state match the lockfile for a package,
and hand back its composed environment."""

from __future__ import annotations

import json
import logging
import shutil
import threading
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pm import paths
from pm.downloader import DownloadPaused, ProgressFn
from pm.lock import Facts, Lockfile
from pm.package import InstallError, Package, Runner, StatePackage, compose_env
from pm.plugin_inputs import Candidates, Members, PluginInput, Selection, StagedUpdate
from pm.registry import get_package, walk
from pm.store import Store, current_target, merge_tree, tree_digest

LOG = logging.getLogger(__name__)

# ``progress(stage, done, total, label)`` reports download/unpack/verify;
# multi-archive labels follow lockfile order.


def _prepare_artifacts(package, store, scratch, artifacts, version, target, *,
                       progress=None, pause_event=None, download_progress=None):
    def tick(done, total, ranges):
        if progress is not None:
            active = next(reversed(ranges))
            index = next(i for i, artifact in enumerate(artifacts) if artifact["url"] == active)
            label = f"{index + 1}/{len(artifacts)}" if len(artifacts) > 1 else ""
            progress("download", done, total, label)
        if download_progress is not None:
            download_progress(done, total, ranges)

    staged = scratch / "tree"
    archives = store.fetch_many(artifacts, scratch, progress=tick, pause_event=pause_event)
    for index, archive in enumerate(archives):
        if pause_event is not None and pause_event.is_set():
            raise DownloadPaused("install paused")
        if progress is not None:
            label = f"{index + 1}/{len(artifacts)}" if len(artifacts) > 1 else ""
            progress("unpack", 0, 0, label)
        # unpack empties its destination; extract additional archives separately.
        destination = staged if index == 0 else scratch / f"extra-{index}"
        package.unpack(archive, destination, target)
        if index:
            merge_tree(destination, staged)
    package.stage(store, staged, version, target)
    return staged


def _lockfile() -> Lockfile:
    return Lockfile(paths.lockfile_path())


def _facts() -> Facts:
    return Facts(paths.facts_path())


def _store() -> Store:
    return Store(paths.store_root())


def _installed_location(package: Package, lockfile: Lockfile, target: str, *,
                        verify: bool = False, allow_outdated: bool = False,
                        roots: tuple[Path, ...] | None = None):
    """Prefer the current pin. An explicit read may retain a prior PM install."""
    search_roots = dict.fromkeys(roots if roots is not None else (paths.store_root(), paths.writable_store_root()))
    fallback = None
    for root in search_roots:
        store = Store(root)
        facts = _facts() if root == paths.store_root() else Facts(root / "facts.json")
        fact = facts.get(package.name)
        if not fact or not facts.installed(package.name, None, root):
            continue
        binary = package.binary(store.entry(fact["entry"]), target)
        if binary is not None and not binary.is_file():
            continue
        if verify and not _entry_verified(package, fact, store, target):
            continue
        if facts.installed(package.name, lockfile.version(package.name), root,
                           _identity(lockfile, package.name, target)):
            return facts, store
        if (allow_outdated and fact.get("target") == target
                and fact.get("artifacts") and fact.get("digest")):
            fallback = facts, store
    return fallback


@dataclass(frozen=True)
class InstalledPackage:
    path: Path
    version: str
    binary: Path | None


def installed_package(name: str, *, allow_outdated: bool = False) -> InstalledPackage | None:
    """Read the selected PM entry without installing or changing its facts."""
    package = get_package(name)
    if package.internal:
        raise ValueError(f"{name} is internal PM tooling, not an application package")
    target = current_target()
    location = _installed_location(package, _lockfile(), target, allow_outdated=allow_outdated)
    if location is None:
        return None
    facts, store = location
    fact = facts.get(name)
    entry = store.entry(fact["entry"])
    return InstalledPackage(entry, fact["version"], package.binary(entry, target))


def _identity(lockfile: Lockfile, name: str, target: str):
    """The identity the lock currently pins for `name` on `target`:
    (target, tuple(artifact sha256s)) — or None when the lock pins no
    artifacts (nothing digest-bound to compare)."""
    artifacts = lockfile.artifacts(name, target)
    if not artifacts:
        return None
    return (target, tuple(a["sha256"] for a in artifacts))


def lazy_installs_allowed() -> bool:
    """Policy: may pm install things on demand right now?

    HERMES_DISABLE_LAZY_INSTALLS is an internal bridge var set by the
    official Docker image and the hermetic test harness. The user-facing
    setting is security.allow_lazy_installs in config.yaml; a config
    system that fails to load counts as ALLOWED only when hermes_cli is
    genuinely absent (bootstrap) — config errors fail closed.
    """
    import os

    if os.environ.get("HERMES_DISABLE_LAZY_INSTALLS", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return False
    try:
        from hermes_cli.config import cfg_get, load_config_readonly, require_readable_config_before_write
    except ModuleNotFoundError as exc:
        return exc.name in {"hermes_cli", "hermes_cli.config"}
    except ImportError:
        return False
    try:
        # The normal loader falls back to defaults on invalid YAML. A broken
        # security policy must not grant permission to acquire dependencies.
        require_readable_config_before_write()
        return cfg_get(load_config_readonly(), "security", "allow_lazy_installs", default=True) is True
    except Exception:
        return False


def enabled_extras() -> list[str]:
    """The venv extras recorded in the installed state."""
    fact = Facts(paths.runtime_facts_path()).get("venv") or _facts().get("venv") or {}
    return list(fact.get("extras", []))


def is_installed(name: str) -> bool:
    return _installed_location(get_package(name), _lockfile(), current_target()) is not None


def sealed() -> bool:
    """Sealed = bundled LAYOUT only: the store sits beside the bundle
    manifest and is read-only as shipped."""
    return (paths.store_root().parent / "manifest.json").is_file()


def _refuse_lazy(name: str, what: str) -> InstallError:
    from pm import receipt

    error = InstallError(
        name,
        f"not installed and lazy installs are disabled: {what}",
        "enable security.allow_lazy_installs or run `hermes pm install`",
    )
    receipt.record_refusal("lazy-install", str(error))
    return error


def _remove_entry(store: Store, entry_name: str) -> None:
    """Remove a replaced or failed entry, retrying transient Windows holds.

    Corruption may leave a file where the directory belonged. Failure
    must propagate so recovery never claims to have removed surviving bytes.
    """
    import time

    entry = store.entry(entry_name)
    for attempt in range(5):
        try:
            if entry.is_symlink() or not entry.is_dir():
                entry.unlink(missing_ok=True)
            else:
                shutil.rmtree(entry)
            return
        except FileNotFoundError:
            return
        except OSError as e:
            if attempt == 4:
                raise
            time.sleep(0.2 * (attempt + 1))


def _remove_downloads(store: Store, artifacts: list[dict]) -> None:
    """Release this package's archives after publication, under its store lock."""
    for artifact in artifacts:
        _remove_entry(store, f"fetch-{artifact['sha256']}")


def _entry_verified(package: Package, fact: dict, store: Store, target: str) -> bool:
    """Explicit installs re-check realized bytes; startup keeps its cheap facts check."""
    entry = store.entry(fact["entry"])
    try:
        return not package.verify(entry, target) and fact.get("digest") == tree_digest(entry)
    except OSError:
        return False


def _restore_previous_entry(store: Store, entry, previous) -> None:
    """Keep both versions recoverable until the restore rename succeeds."""
    import uuid

    displaced = store.entry(f".displaced-{uuid.uuid4().hex}")
    had_entry = entry.exists() or entry.is_symlink()
    if had_entry:
        entry.rename(displaced)
    try:
        previous.rename(entry)
    except BaseException:
        if had_entry:
            displaced.rename(entry)
        raise
    if had_entry:
        _remove_entry(store, displaced.name)


@contextmanager
def _publish_entry(package, store, staged, entry, previous_entry, target):
    """Keep rollback live through the caller's native facts commit, if any."""
    if entry.exists() or entry.is_symlink():
        entry.rename(previous_entry)
    try:
        store.publish(staged, entry.name)
        reason = package.verify(entry, target)
        if reason:
            raise InstallError(package.name, f"published entry failed verification: {reason}")
        yield
    except BaseException:
        if previous_entry.exists():
            _restore_previous_entry(store, entry, previous_entry)
        raise
    if previous_entry.exists():
        _remove_entry(store, previous_entry.name)


def _settle_previous_entry(package, store, entry, previous_entry, previous, target) -> None:
    """Finish or undo a publication an earlier install left behind."""
    if not previous_entry.exists():
        return
    # Facts commit last. Stages have no host-side commit record, so
    # an interrupted stage always restores its prior usable bytes.
    if (previous and previous.get("entry") == entry.name
            and _entry_verified(package, previous, store, target)):
        _remove_entry(store, previous_entry.name)
    else:
        _restore_previous_entry(store, entry, previous_entry)


def _entry_current(package, lockfile, facts, store, entry, previous, version, pin, target) -> bool:
    if facts is not None:
        return previous is not None and facts.installed(
            package.name, version, store.root, _identity(lockfile, package.name, target)
        ) and _entry_verified(package, previous, store, target)
    try:
        recorded = (entry / ".pm-stage-pin.json").read_text(encoding="utf-8-sig")
    except OSError:
        recorded = None
    return recorded == pin and not package.verify(entry, target)


def _copy_verified_source(package, lockfile, copy_from, staged, version, target) -> None:
    source_facts, source_store = copy_from
    source = source_facts.get(package.name)
    if (not source_facts.installed(package.name, version, source_store.root,
                                   _identity(lockfile, package.name, target))
            or not _entry_verified(package, source, source_store, target)):
        raise InstallError(package.name, "bundled copy source failed verification")
    shutil.copytree(source_store.entry(source["entry"]), staged, symlinks=True)
    if tree_digest(staged) != source["digest"]:
        raise InstallError(package.name, "copied bytes do not match the bundled source")


def _log_repair(package, previous, version, artifacts) -> None:
    """Work item 6: replacing an ESTABLISHED fact is a repair — log it,
    no transaction system, no receipt file."""
    if not previous or "entry" not in previous:
        return
    old_artifact = (previous.get("artifacts") or ["?"])[0]
    old = f"{previous.get('version', '?')}/{str(old_artifact)[:12]}"
    new = f"{version}/{artifacts[0]['sha256'][:12]}" if artifacts else version
    LOG.info("repair: %s re-realized %s -> %s", package.name, old, new)


def _install(
    package: Package,
    lockfile: Lockfile,
    facts: Facts | None,
    store: Store,
    target: str,
    progress=None,
    pause_event: threading.Event | None = None,
    download_progress: ProgressFn | None = None,
    *,
    copy_from: tuple[Facts, Store] | None = None,
    _lock_held: bool = False,
    _fresh_copy: bool = False,
) -> Path:
    """Realize one pin. Host installs commit facts; cross-target stages carry a marker."""
    version = lockfile.version(package.name)
    if version is None:
        raise InstallError(
            package.name, "not in the lockfile", "add it with `hermes pm lock --bump`"
        )

    reason = package.missing_reason(target)
    if reason is not None:
        raise InstallError(package.name, f"unavailable on {target}: {reason}", "none")

    entry_name = package.store_entry(version, target)
    entry = store.entry(entry_name)
    if getattr(package, "pin_only", False):
        return entry
    artifacts = lockfile.artifacts(package.name, target)
    pin = json.dumps({"target": target, "sha256": [a["sha256"] for a in artifacts]})

    with nullcontext() if _lock_held else store.install_lock():
        if pause_event is not None and pause_event.is_set():
            raise DownloadPaused("install paused")
        if facts is not None:
            facts.reload()
        previous = facts.get(package.name) if facts is not None else None
        previous_entry = store.entry(f".previous-{'stage-' if facts is None else ''}{entry_name}")
        _settle_previous_entry(package, store, entry, previous_entry, previous, target)
        if (_entry_current(package, lockfile, facts, store, entry, previous, version, pin, target)
                and not _fresh_copy):
            _remove_downloads(store, artifacts)
            return entry
        if not artifacts:
            raise InstallError(
                package.name,
                f"no artifact for {target} in the lockfile",
                "run `hermes pm lock --bump` for this package",
            )
        with store.scratch() as scratch:
            staged = scratch / "tree"
            try:
                if copy_from is not None:
                    _copy_verified_source(package, lockfile, copy_from, staged, version, target)
                else:
                    staged = _prepare_artifacts(package, store, scratch, artifacts, version, target,
                                                progress=progress, pause_event=pause_event,
                                                download_progress=download_progress)
                if pause_event is not None and pause_event.is_set():
                    raise DownloadPaused("install paused")
                if progress is not None:
                    progress("verify", 0, 0, "")
                reason = package.verify(staged, target)
                if reason:
                    raise InstallError(package.name, f"staged entry failed verification: {reason}")
                if facts is None:
                    (staged / ".pm-stage-pin.json").write_text(pin, encoding="utf-8")
                with _publish_entry(package, store, staged, entry, previous_entry, target):
                    if facts is not None:
                        facts.record(
                            package.name, version, entry_name, package.env(entry, target), store.root,
                            target=target, artifacts=[a["sha256"] for a in artifacts],
                            digest=tree_digest(entry),
                        )
                _remove_downloads(store, artifacts)
            except (InstallError, DownloadPaused):
                raise
            except Exception as e:
                raise InstallError(package.name, f"install failed: {e}") from e

        _log_repair(package, previous, version, artifacts)
    return entry


def stage_only(
    name: str, target: str, progress=None, *,
    pause_event: threading.Event | None = None,
    download_progress: ProgressFn | None = None,
) -> Path:
    """Realize a cross-target pin without host facts or an executable Runner."""
    return _install(get_package(name), _lockfile(), None, _store(), target,
                    progress=progress, pause_event=pause_event, download_progress=download_progress)


class _InstallOperation:
    """Validity lasts only while this operation holds the publication lock."""

    def __init__(self) -> None:
        self.stack = ExitStack()
        self.store: Store | None = None
        self.checked: set[tuple[str, str | None, str, str]] = set()

    def lock(self) -> Store:
        if self.store is None:
            self.store = Store(paths.writable_store_root())
            self.stack.enter_context(self.store.install_lock())
        return self.store

    def close(self) -> None:
        self.checked.clear()
        self.store = None
        self.stack.close()


@contextmanager
def _install_operation():
    operation = _InstallOperation()
    try:
        yield operation
    finally:
        operation.close()


def ensure(
    name: str,
    *,
    base_env: Optional[dict] = None,
    explicit: bool = False,
    verify: bool = True,
    progress=None,
    pause_event: threading.Event | None = None,
    download_progress: ProgressFn | None = None,
    _operation: _InstallOperation | None = None,
) -> Runner:
    """``explicit`` marks a deliberate install command (`hermes pm
    install`, `hermes pm bundle`) — those ARE the remedy the lazy-install
    policy names, so the policy does not apply to them.

    ``verify`` re-hashes an already-recorded entry and repairs it when the
    bytes moved. A deliberate install keeps that check. Shell activation
    passes ``False``. It trusts the recorded digest, the same check startup
    uses, because hashing every tool tree costs seconds per shell.

    ``progress(stage, done, total, label)`` reports the slow parts of an
    install to a UI, including ordered multi-archive labels.
    """
    if isinstance(get_package(name), StatePackage):
        if _operation is not None:
            # Python construction can provision tools itself. Drop both the
            # lock and its validity before entering that independent operation.
            _operation.close()
        sync_venv(explicit=explicit)
        return Runner(name, compose_env([], base=base_env))

    lockfile = _lockfile()
    target = current_target()
    chain = walk([name])
    checked = _operation.checked if _operation is not None else set()
    if _operation is not None:
        _operation.lock()
    missing = []
    for package in chain:
        identity = (package.name, lockfile.version(package.name), target,
                    json.dumps(_identity(lockfile, package.name, target), sort_keys=True))
        if identity in checked:
            continue
        if _installed_location(package, lockfile, target, verify=explicit and verify) is None:
            missing.append(package)
        else:
            checked.add(identity)
    if missing and not explicit and not lazy_installs_allowed():
        raise _refuse_lazy(name, ", ".join(p.name for p in missing))
    if missing:
        store = _operation.lock() if _operation is not None else Store(paths.writable_store_root())
        facts = _facts() if store.root == paths.store_root() else Facts(store.root / "facts.json")
        for package in missing:
            # Publication may change entries; do not carry observations across it.
            checked.clear()
            _install(package, lockfile, facts, store, target, progress=progress,
                     pause_event=pause_event, download_progress=download_progress,
                     _lock_held=_operation is not None)
    return Runner(name, env_for(name, base_env=base_env))


def env_for(*names: str, base_env: Optional[dict] = None) -> dict[str, str]:
    """Composed env of already-installed packages only. Never installs,
    never raises on missing packages — they contribute nothing."""
    lockfile = _lockfile()
    target = current_target()
    diffs: list[dict] = []
    for name in names:
        try:
            chain = walk([name])
        except KeyError:
            continue
        for package in chain:
            if package.internal:
                continue
            location = _installed_location(package, lockfile, target)
            if location:
                facts, store = location
                diffs.append(facts.env_for(package.name, store.root))
    return compose_env(diffs, base=base_env)


def _runtime_state_matches(fact: dict, stamp: str, *, project_root: Path | None = None) -> bool:
    if not isinstance(fact, dict) or fact.get("stamp") != stamp:
        return False
    from pm.environments import selected_venv

    try:
        environment = selected_venv(paths.repo_root() if project_root is None else project_root)
    except (OSError, RuntimeError, ValueError):
        return False
    recorded = fact.get("environment")
    if recorded is not None and (not isinstance(recorded, str) or Path(recorded).resolve() != environment):
        return False
    return (environment / "pyvenv.cfg").is_file()


def _member_inputs(plugins: PluginInput | None) -> dict:
    """The ``plugin_dirs`` argument of the venv package; empty means config discovery."""
    from pm.publication import candidate_members

    if isinstance(plugins, Candidates):
        return {"plugin_dirs": candidate_members(plugins.dirs)}
    if isinstance(plugins, Members):
        return {"plugin_dirs": plugins.dirs}
    if plugins is None:
        return {}
    raise TypeError(f"{type(plugins).__name__} changes plugin state; only a sync may carry it")


def venv_is_current(*, extras: list[str] | None = None, plugins: Members | Candidates | None = None,
                    project_root: Path | None = None) -> bool:
    """Probe the requested union without changing recorded dependency state."""
    from pm.environments import runtime_facts_path
    from pm.packages import Venv

    root = paths.repo_root() if project_root is None else Path(project_root).absolute()
    package = get_package("venv") if project_root is None else Venv(root)
    fact = Facts(runtime_facts_path(root), strict=True).get("venv")
    if fact is None:
        fact = Facts(paths.facts_path(), strict=True).get("venv")
    if fact is None:
        return False
    if (not isinstance(fact, dict) or not isinstance(fact.get("stamp"), str) or not fact["stamp"]
            or not isinstance(fact.get("extras"), list)
            or any(not isinstance(extra, str) for extra in fact["extras"])):
        raise ValueError("invalid recorded dependency state")
    enabled = sorted(set(fact["extras"]) | set(extras or []))
    stamp = package.expected_stamp(enabled, **_member_inputs(plugins))
    return _runtime_state_matches(fact, stamp, project_root=root)


def _feature_policy(extras: Optional[list[str]], *, repair: bool) -> tuple[list[str] | None, list[str] | None]:
    """Refuse extras this platform or a frozen bundle cannot carry; return (shipped, frozen)."""
    from pm.features import read_features

    if extras:
        from pm.extras import extra_supported
        unsupported = [extra for extra in extras
                       if not extra_supported(extra, importable=lambda _: False)]
        if unsupported:
            raise InstallError("venv", f"extras {unsupported} are not supported by this Python/platform",
                               "choose a supported provider; no dependency environment was changed")
    shipped = read_features()
    frozen = shipped
    # Without a frozen declaration, explicit source setup needs no policy
    # read: the config loader initializes/chmods unrelated user state.
    if frozen is not None and not repair and lazy_installs_allowed():
        frozen = None
    if frozen is not None and extras:
        outside = sorted(set(extras) - set(frozen))
        if outside:
            raise _refuse_lazy(
                "venv",
                f"extras {outside} are outside this bundle's frozen feature "
                "set (security.allow_lazy_installs is false)",
            )
    return shipped, frozen


@contextmanager
def _venv_install_lock(*, patient: bool):
    """Hold the dependency lock, or refuse when an impatient caller would queue."""
    from pm import receipt
    from hermes_cli.runtime_state import INSTALL_LOCK_TIMEOUT_SECONDS, runtime_lock

    # Holding this lock means rebuilding the whole dependency environment, which takes tens of
    # seconds on a bundle. Only an install the user asked for may queue for it; an opportunistic
    # one (a lazy extra at first use, the only non-explicit caller) refuses instead of holding
    # a sibling profile's backend off its port behind a rebuild it did not request.
    with runtime_lock(paths.repo_root(), timeout=None if patient else INSTALL_LOCK_TIMEOUT_SECONDS) as held:
        if not held:
            error = InstallError(
                "venv",
                f"another Hermes process is installing dependencies (waited {INSTALL_LOCK_TIMEOUT_SECONDS:.0f}s)",
                "retry in a moment, or run `hermes pm install` to install explicitly",
            )
            receipt.record_refusal("install-busy", str(error))
            raise error
        yield


def _publication(plugins: PluginInput | None):
    """Snapshot a plugin state change under the held lock, or None for member-only inputs."""
    from pm.publication import PluginSelection, StagedPlugin

    if isinstance(plugins, Selection):
        return PluginSelection(dict(plugins.data))
    if isinstance(plugins, StagedUpdate):
        return StagedPlugin(dict(plugins.data))
    return None


def _publish_inactive(change) -> None:
    """A disabled plugin's code changes without touching the dependency environment."""
    from pm import receipt
    from hermes_cli.runtime_state import finish_publication, recover_publication

    try:
        change.publish(paths.repo_root())
        finish_publication(paths.repo_root())
    except BaseException:
        recover_publication(paths.repo_root())
        raise
    receipt.record_venv_rebuild(False, "inactive plugin")


def _target_selection(package, fact: dict, *, extras, inputs: dict, repair: bool, shipped, frozen):
    """Return (enabled extras, expected stamp, package inputs) this sync must reach."""
    if repair:
        if fact and (not isinstance(fact.get("extras"), list)
                     or any(not isinstance(extra, str) for extra in fact["extras"])
                     or not isinstance(fact.get("stamp"), str) or not fact["stamp"]):
            raise InstallError("venv", "recorded dependency selection is incomplete; refusing to change its graph")
        enabled = list(fact.get("extras", frozen if frozen is not None else ["all"]))
        stamp = fact.get("stamp") or package.expected_stamp(enabled, plugin_dirs=[])
        return enabled, stamp, {"repair": True}
    # The first writable generation replaces, rather than layers on,
    # the payload. Retain its extras until a recorded selection owns them.
    enabled = sorted(set(fact.get("extras", shipped or [])) | set(extras or []))
    return enabled, package.expected_stamp(enabled, **inputs), inputs


def _commit_selection(package, facts: Facts, change, *, enabled: list[str], stamp: str, inputs: dict,
                      current: bool, repair: bool, explicit: bool) -> None:
    """Build (unless current), publish the plugin change, then record the selection."""
    from pm import receipt
    from hermes_cli.runtime_state import finish_publication, recover_publication

    try:
        result = {} if current else (package.apply(enabled, explicit=explicit, **inputs) or {})
        if not repair and package.expected_stamp(enabled, **inputs) != stamp:
            raise ValueError("Dependency inputs changed while preparing publication; retry.")
        if change is not None:
            change.publish(paths.repo_root())
        if not current:
            if result.get("environment") is not None:
                from pm.environments import flush_before_selecting
                flush_before_selecting()
            facts.record_state("venv", stamp, enabled, **result)
        if change is not None:
            finish_publication(paths.repo_root())
        receipt.record_venv_rebuild(not current, "already in sync" if current else "")
    except BaseException:
        recover_publication(paths.repo_root())
        raise


def sync_venv(extras: Optional[list[str]] = None, *, explicit: bool = False,
              plugins: PluginInput | None = None, repair: bool = False) -> None:
    """Make the venv match uv.lock + the enabled extras. Extras union into
    the installed state (one ledger); no-op when the stamp already matches.
    ``repair`` restores the recorded dependency graph into a fresh generation,
    bypassing both that shortcut and config discovery. It cannot add features.
    ``explicit`` marks a deliberate install command (`hermes pm install`,
    `hermes update`) — those are the remedy the lazy-install policy points
    at, so the policy does not apply to them. ``plugins`` names the one
    source of plugin members (see pm.plugin_inputs); None discovers them from config.

    Lazy installs OFF = the frozen feature set: when
    security.allow_lazy_installs is false AND the bundle's
    enabled-features.json exists, the feature list is FROZEN to that file
    — requested extras outside it are refused, and plugin members are
    never installed (the bundle IS the install).

    Receipt contract: EVERY outcome writes a receipt. ``begin`` fires
    BEFORE the frozen/lazy refusals (a refusal is a recorded ``failed``
    outcome, not a silent raise); finalize runs in FINALLY — no-op syncs
    ("ok" with ``venv_rebuild`` false) and refusals ("failed") both get
    a receipt. Plugin selection data is discovered and published by the worker
    under this lock; no executable transaction phases cross the process boundary."""
    from pm import receipt

    token = receipt.begin("sync")
    outcome = "failed"
    try:
        if repair and (extras is not None or plugins is not None):
            raise ValueError("repair restores the recorded environment; it cannot change features or plugins")
        shipped, frozen = _feature_policy(extras, repair=repair)
        package = get_package("venv")
        from hermes_cli.runtime_state import recover_publication
        from pm.publication import StagedPlugin
        with _venv_install_lock(patient=explicit or repair):
            recover_publication(paths.repo_root())
            change = _publication(plugins)
            if isinstance(change, StagedPlugin) and not change.active:
                _publish_inactive(change)
            else:
                inputs = {"plugin_dirs": change.members} if change is not None else _member_inputs(plugins)
                facts = Facts(paths.runtime_facts_path(), strict=repair)
                fact = facts.get("venv") or _facts().get("venv") or {}
                enabled, stamp, inputs = _target_selection(package, fact, extras=extras, inputs=inputs,
                                                           repair=repair, shipped=shipped, frozen=frozen)
                current = not repair and _runtime_state_matches(fact, stamp)
                if not current and not repair and not explicit and not lazy_installs_allowed():
                    raise _refuse_lazy("venv", str(extras) if extras else "venv out of sync")
                receipt.record_feature_list(enabled)
                _commit_selection(package, facts, change, enabled=enabled, stamp=stamp, inputs=inputs,
                                  current=current, repair=repair, explicit=explicit)
        outcome = "ok"
    except BaseException as exc:
        receipt.record_step("dependency-sync", False, f"{type(exc).__name__}: {exc}")
        raise
    finally:
        receipt.finalize(outcome, 0 if outcome == "ok" else 1, token=token)


def drift(*, include_venv: bool = True) -> dict[str, str]:
    """Cheap stamp comparisons of the installed state
    against the lockfile. Maps package names to reasons; empty means healthy. Never
    installs, never touches the network. An install pm has never touched
    (no installed-state file) reports nothing — pm only vouches for what
    it installed. Lockfile packages this build doesn't know (version skew
    during a partial update) are skipped, not fatal."""
    if not paths.facts_path().is_file() and not paths.runtime_facts_path().is_file():
        return {}

    problems: dict[str, str] = {}
    lockfile = _lockfile()
    facts = _facts()
    store = _store()
    target = current_target()
    for name in lockfile.names():
        try:
            package = get_package(name)
        except KeyError:
            continue
        if package.optional or package.internal:
            continue
        if package.missing_reason(target) is not None:
            continue
        if _installed_location(package, lockfile, target) is None:
            problems[name] = "not installed or outdated"
    try:
        venv = get_package("venv")
    except KeyError:
        venv = None
    if include_venv and venv is not None and (paths.runtime_facts_path().is_file() or facts.get("venv") is not None):
        try:
            if not venv_is_current():
                problems["venv"] = "out of sync with uv.lock"
        except (OSError, RuntimeError, ValueError) as exc:
            problems["venv"] = str(exc)
    return problems


def check(*, include_venv: bool = True) -> list[str]:
    """Human-readable startup diagnostics. Use drift() for package identities."""
    return [f"{name}: {reason}" for name, reason in drift(include_venv=include_venv).items()]


def _store_path_dirs() -> list[str]:
    """Composed PATH dirs of all installed (non-internal, on_path) store
    packages, deps-first, deduped. Includes optional packages that are
    *installed* (facts say so) — an installed git/gh must be on PATH even
    though it's not in the root closure. Never installs."""

    lockfile = _lockfile()
    target = current_target()
    dirs: list[str] = []
    for name in lockfile.names():
        try:
            package = get_package(name)
        except KeyError:
            continue
        if package.internal:
            continue
        if not getattr(package, "on_path", True):
            continue
        if package.missing_reason(target) is not None:
            continue
        location = _installed_location(package, lockfile, target)
        if location is None:
            continue
        facts, store = location
        env = facts.env_for(name, store.root)
        path_dirs = env.get("PATH") or []
        if isinstance(path_dirs, str):
            path_dirs = [path_dirs]
        for directory in path_dirs:
            if directory and directory not in dirs:
                dirs.append(str(directory))
    return dirs


def activate(*, allow_incomplete: bool = False) -> list[str]:
    """Make the installed store usable: prepend its tool dirs to
    os.environ['PATH'] so reactive `shutil.which('git'|'bash'|'ffmpeg'|...)`
    resolves the bundled binaries. The gate is `check()` — if the store is
    broken, refuse to inject (fail fast rather than serving a partial PATH).
    Return the check's problems, or an empty list on success, so startup
    callers can report the verdict without checking the store twice.

    ``allow_incomplete`` is the install-time exception: tools are published
    before the venv sync, so a missing venv must not hide the tools the sync
    is about to build against. A missing tool still refuses.

    This is the ONE sanctioned global PATH write: PATH is the discovery
    contract every `which` reads, not a tool-specific env leak. Store-first
    unconditionally — pinned bundled versions win on dev machines too.
    """
    import os

    # The venv verdict is discarded here, and computing it imports application
    # config readers (ruamel) that the bare update interpreter does not carry.
    problems = check(include_venv=not allow_incomplete)
    if problems:
        return problems
    dirs = _store_path_dirs()
    if not dirs:
        return []
    existing = os.environ.get("PATH", "")
    prefix = os.pathsep.join(dirs)
    existing_lower = {p.lower() for p in existing.split(os.pathsep) if p}
    missing = [d for d in dirs if d.lower() not in existing_lower]
    if missing:
        os.environ["PATH"] = os.pathsep.join([*missing, existing]) if existing else os.pathsep.join(missing)
    return []
