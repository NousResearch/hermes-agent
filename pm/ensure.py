"""ensure(): make the installed state match the lockfile for a package,
and hand back its composed environment."""

from __future__ import annotations

import json
import logging
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pm import paths
from pm.downloader import DownloadPaused, ProgressFn
from pm.lock import Facts, Lockfile
from pm.package import InstallError, Package, Runner, StatePackage, compose_env
from pm.registry import get_package, walk
from pm.store import Store, current_target, merge_tree, tree_digest

LOG = logging.getLogger(__name__)

# ``progress(stage, done, total, label)`` — stage is "download" | "unpack",
# label is the archive counter ("1/2") when a package has several. Slow
# lines sit in one stage for minutes, so the byte counters are what prove
# liveness to a UI.


def _artifact_progress(progress, index: int, count: int):
    if progress is None:
        return None
    label = f"{index + 1}/{count}" if count > 1 else ""
    return lambda done, total: progress("download", done, total, label)


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
        from hermes_cli.config import get_config_value
    except ImportError:
        return True
    try:
        return bool(get_config_value("security.allow_lazy_installs", True))
    except Exception:
        return False


def enabled_extras() -> list[str]:
    """The venv extras recorded in the installed state."""
    fact = Facts(paths.runtime_facts_path()).get("venv") or _facts().get("venv") or {}
    return list(fact.get("extras", []))


def is_installed(name: str) -> bool:
    return _installed_location(get_package(name), _lockfile(), current_target()) is not None


def sealed() -> bool:
    """A bundled payload is read-only: its store sits beside the bundle
    manifest. Asking a sealed install for MORE than it shipped is a
    Sealed = bundled LAYOUT only; adoption verification is adopt()'s job."""
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


def _install(
    package: Package,
    lockfile: Lockfile,
    facts: Facts,
    store: Store,
    target: str,
    progress=None,
    pause_event: threading.Event | None = None,
    download_progress: ProgressFn | None = None,
    *,
    copy_from: tuple[Facts, Store] | None = None,
) -> None:
    version = lockfile.version(package.name)
    if version is None:
        raise InstallError(
            package.name, "not in the lockfile", "add it with `hermes pm lock --bump`"
        )

    reason = package.missing_reason(target)
    if reason is not None:
        raise InstallError(package.name, f"unavailable on {target}: {reason}", "none")

    artifacts = lockfile.artifacts(package.name, target)
    entry_name = package.store_entry(version, target)

    with store.install_lock():
        if pause_event is not None and pause_event.is_set():
            raise DownloadPaused("install paused")
        facts.reload()
        entry = store.entry(entry_name)
        previous_entry = store.entry(f".previous-{entry_name}")
        if previous_entry.exists():
            # An interrupted replacement keeps its old bytes outside scratch.
            # Facts commit last; only a verified committed replacement wins.
            fact = facts.get(package.name)
            if fact and fact.get("entry") == entry_name and _entry_verified(package, fact, store, target):
                _remove_entry(store, previous_entry.name)
            else:
                _restore_previous_entry(store, entry, previous_entry)
        if facts.installed(
            package.name, version, store.root, _identity(lockfile, package.name, target)
        ) and _entry_verified(package, facts.get(package.name), store, target):
            _remove_downloads(store, artifacts)
            return
        if not artifacts:
            raise InstallError(
                package.name,
                f"no artifact for {target} in the lockfile",
                "run `hermes pm lock --bump` for this package",
            )
        with store.scratch() as scratch:
            staged = scratch / "tree"
            previous = facts.get(package.name)
            try:
                def tick(done, total, ranges):
                    if progress is not None:
                        active = next(reversed(ranges))
                        index = next(i for i, artifact in enumerate(artifacts) if artifact["url"] == active)
                        label = f"{index + 1}/{len(artifacts)}" if len(artifacts) > 1 else ""
                        progress("download", done, total, label)
                    if download_progress is not None:
                        download_progress(done, total, ranges)

                if copy_from is not None:
                    source_facts, source_store = copy_from
                    source = source_facts.get(package.name)
                    if (not source_facts.installed(package.name, version, source_store.root,
                                                   _identity(lockfile, package.name, target))
                            or not _entry_verified(package, source, source_store, target)):
                        raise InstallError(package.name, "bundled copy source failed verification")
                    shutil.copytree(source_store.entry(source["entry"]), staged, symlinks=True)
                    if tree_digest(staged) != source["digest"]:
                        raise InstallError(package.name, "copied bytes do not match the bundled source")
                else:
                    archives = store.fetch_many(artifacts, scratch, progress=tick, pause_event=pause_event)
                    for index, archive in enumerate(archives):
                        if pause_event is not None and pause_event.is_set():
                            raise DownloadPaused("install paused")
                        label = f"{index + 1}/{len(artifacts)}" if len(artifacts) > 1 else ""
                        if progress is not None:
                            progress("unpack", 0, 0, label)
                        if index == 0:
                            package.unpack(archive, staged, target)
                            continue
                        # unpack() empties its destination; merge additional
                        # archives only after extracting them separately.
                        extra = scratch / f"extra-{index}"
                        package.unpack(archive, extra, target)
                        merge_tree(extra, staged)
                    package.stage(store, staged, version, target)
                if pause_event is not None and pause_event.is_set():
                    raise DownloadPaused("install paused")
                if progress is not None:
                    progress("verify", 0, 0, "")
                reason = package.verify(staged, target)
                if reason:
                    raise InstallError(package.name, f"staged entry failed verification: {reason}")
                if entry.exists():
                    entry.rename(previous_entry)
                try:
                    store.publish(staged, entry_name)
                    reason = package.verify(entry, target)
                    if reason:
                        raise InstallError(package.name, f"published entry failed verification: {reason}")
                    facts.record(
                        package.name, version, entry_name, package.env(entry, target), store.root,
                        target=target, artifacts=[a["sha256"] for a in artifacts],
                        digest=tree_digest(entry),
                    )
                except BaseException:
                    if previous_entry.exists():
                        _restore_previous_entry(store, entry, previous_entry)
                    raise
                if previous_entry.exists():
                    _remove_entry(store, previous_entry.name)
                _remove_downloads(store, artifacts)
            except (InstallError, DownloadPaused):
                raise
            except Exception as e:
                raise InstallError(package.name, f"install failed: {e}") from e

        if previous and "entry" in previous:
            # Work item 6: replacing an ESTABLISHED fact is a repair —
            # log it, no transaction system, no receipt file.
            old_artifact = (previous.get("artifacts") or ["?"])[0]
            old = f"{previous.get('version', '?')}/{str(old_artifact)[:12]}"
            new = (
                f"{version}/{artifacts[0]['sha256'][:12]}"
                if artifacts
                else version
            )
            LOG.info("repair: %s re-realized %s -> %s", package.name, old, new)


def stage_only(name: str, target: str, progress=None) -> "Path":
    """Cross-target staging: publish the pinned (package, version, target)
    entry into the store and return its path. No facts are written and no
    Runner is composed -- the staged binaries belong to ANOTHER machine
    (e.g. linux-arm64-bionic .debs staged on a glibc CI host); this host's
    installed-state must not learn about them. Idempotent: an already
    published + verifying entry is returned as-is.
    """
    lockfile = _lockfile()
    store = _store()
    package = get_package(name)
    version = lockfile.version(package.name)
    if version is None:
        raise InstallError(package.name, "not in the lockfile")
    reason = package.missing_reason(target)
    if reason is not None:
        raise InstallError(package.name, f"unavailable on {target}: {reason}")
    if getattr(package, "pin_only", False):
        # A pure pin (e.g. the termux-docker digest): no bytes, no store
        # entry, nothing to verify locally -- the pin IS the artifact.
        return store.root / package.store_entry(version, target)
    artifacts = lockfile.artifacts(package.name, target)
    entry_name = package.store_entry(version, target)
    # The stage pin marker (same identity shape as a fact's recorded
    # artifacts: target + artifact digests) lets stage_only honor a
    # same-version hash repin without any host-side facts: the entry
    # belongs to ANOTHER machine, so the marker travels inside the entry.
    pin = json.dumps({"target": target, "sha256": [a["sha256"] for a in artifacts]})
    with store.install_lock():
        entry = store.entry(entry_name)
        previous_entry = store.entry(f".previous-stage-{entry_name}")
        if previous_entry.exists():
            # A killed publisher may have installed only part of the new tree.
            _restore_previous_entry(store, entry, previous_entry)
        if store.published(entry_name):
            marker = entry / ".pm-stage-pin.json"
            try:
                recorded = marker.read_text(encoding="utf-8")
            except OSError:
                recorded = None
            if not package.verify(entry, target) and recorded == pin:
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
            for index, artifact in enumerate(artifacts):
                archive = store.fetch(
                    artifact["url"], artifact["sha256"], scratch,
                    progress=_artifact_progress(progress, index, len(artifacts)),
                )
                if index == 0:
                    package.unpack(archive, staged, target)
                else:
                    extra = scratch / f"extra-{index}"
                    package.unpack(archive, extra, target)
                    merge_tree(extra, staged)
            package.stage(store, staged, version, target)
            reason = package.verify(staged, target)
            if reason:
                raise InstallError(package.name, f"staged entry failed verification: {reason}")
            (staged / ".pm-stage-pin.json").write_text(pin, encoding="utf-8")
            # Keep the old pin usable until the replacement has been verified.
            if entry.exists() or entry.is_symlink():
                entry.rename(previous_entry)
            try:
                store.publish(staged, entry_name)
                reason = package.verify(entry, target)
                if reason:
                    raise InstallError(package.name, f"published entry failed verification: {reason}")
            except BaseException:
                if previous_entry.exists():
                    _restore_previous_entry(store, entry, previous_entry)
                raise
            if previous_entry.exists():
                _remove_entry(store, previous_entry.name)
            _remove_downloads(store, artifacts)
    return store.entry(entry_name)


def ensure(
    name: str,
    *,
    base_env: Optional[dict] = None,
    explicit: bool = False,
    progress=None,
    pause_event: threading.Event | None = None,
    download_progress: ProgressFn | None = None,
) -> Runner:
    """``explicit`` marks a deliberate install command (`hermes pm
    install`, `hermes pm bundle`) — those ARE the remedy the lazy-install
    policy names, so the policy does not apply to them.

    ``progress(stage, done, total, label)`` reports the slow parts of an
    install to a UI; see _artifact_progress.
    """
    if isinstance(get_package(name), StatePackage):
        sync_venv(explicit=explicit)
        return Runner(name, compose_env([], base=base_env))

    lockfile = _lockfile()
    target = current_target()
    chain = walk([name])
    missing = [p for p in chain if _installed_location(p, lockfile, target, verify=explicit) is None]
    if missing and not explicit and not lazy_installs_allowed():
        raise _refuse_lazy(name, ", ".join(p.name for p in missing))
    if missing:
        store = Store(paths.writable_store_root())
        facts = _facts() if store.root == paths.store_root() else Facts(store.root / "facts.json")
        for package in missing:
            _install(package, lockfile, facts, store, target, progress=progress,
                     pause_event=pause_event, download_progress=download_progress)
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


def _runtime_state_matches(fact: dict, stamp: str) -> bool:
    if not isinstance(fact, dict) or fact.get("stamp") != stamp:
        return False
    from hermes_cli.runtime_paths import selected_venv

    try:
        environment = selected_venv(paths.repo_root())
    except (OSError, RuntimeError, ValueError):
        return False
    recorded = fact.get("environment")
    if recorded is not None and (not isinstance(recorded, str) or Path(recorded).resolve() != environment):
        return False
    return (environment / "pyvenv.cfg").is_file()


def venv_is_current() -> bool:
    """Use the recorded PM inputs and the boot-time environment selection."""
    fact = Facts(paths.runtime_facts_path(), strict=True).get("venv")
    if fact is None:
        fact = Facts(paths.facts_path(), strict=True).get("venv")
    if fact is None:
        return False
    if (not isinstance(fact, dict) or not isinstance(fact.get("stamp"), str) or not fact["stamp"]
            or not isinstance(fact.get("extras"), list)
            or any(not isinstance(extra, str) for extra in fact["extras"])):
        raise ValueError("invalid recorded dependency state")
    return _runtime_state_matches(fact, get_package("venv").expected_stamp(fact["extras"]))


def sync_venv(extras: Optional[list[str]] = None, *, explicit: bool = False, plugin_dirs=None, before_publish=None, repair: bool = False) -> None:
    """Make the venv match uv.lock + the enabled extras. Extras union into
    the installed state (one ledger); no-op when the stamp already matches.
    ``repair`` restores the recorded dependency graph into a fresh generation,
    bypassing both that shortcut and config discovery. It cannot add features.
    ``explicit`` marks a deliberate install command (`hermes pm install`,
    `hermes update`) — those are the remedy the lazy-install policy points
    at, so the policy does not apply to them.

    Lazy installs OFF = the frozen feature set: when
    security.allow_lazy_installs is false AND the bundle's
    enabled-features.json exists, the feature list is FROZEN to that file
    — requested extras outside it are refused, and plugin members are
    never installed (the bundle IS the install).

    Receipt contract: EVERY outcome writes a receipt. ``begin`` fires
    BEFORE the frozen/lazy refusals (a refusal is a recorded ``failed``
    outcome, not a silent raise); finalize runs in FINALLY — no-op syncs
    ("ok" with ``venv_rebuild`` false) and refusals ("failed") both get
    a receipt. ``before_publish`` is the concrete selection hook: called
    while the install lock is held, AFTER the environment staged and
    BEFORE the facts write — it returns an undo callable that runs if
    the facts write then fails, so a selection committed here is rolled
    back atomically instead of drifting from the surviving environment.
    No module globals, no callback framework — one hook, one consumer."""
    from pm import receipt
    from pm.features import read_features

    token = receipt.begin("sync")
    outcome = "failed"
    try:
        if repair and (extras is not None or plugin_dirs is not None or before_publish is not None):
            raise ValueError("repair restores the recorded environment; it cannot change features or plugins")
        frozen = read_features() if repair or not lazy_installs_allowed() else None
        if frozen is not None and extras:
            outside = sorted(set(extras) - set(frozen))
            if outside:
                raise _refuse_lazy(
                    "venv",
                    f"extras {outside} are outside this bundle's frozen feature "
                    "set (security.allow_lazy_installs is false)",
                )

        package = get_package("venv")
        from hermes_cli.runtime_state import runtime_lock, recover_publication
        with runtime_lock(paths.repo_root()):
            recover_publication(paths.repo_root())
            members = plugin_dirs() if callable(plugin_dirs) else plugin_dirs
            inputs = {} if members is None else {"plugin_dirs": members}
            facts = Facts(paths.runtime_facts_path(), strict=repair)
            fact = facts.get("venv") or _facts().get("venv") or {}
            if repair:
                if fact and (not isinstance(fact.get("extras"), list)
                             or any(not isinstance(extra, str) for extra in fact["extras"])
                             or not isinstance(fact.get("stamp"), str) or not fact["stamp"]):
                    raise InstallError("venv", "recorded dependency selection is incomplete; refusing to change its graph")
                enabled = list(fact.get("extras", frozen if frozen is not None else ["all"]))
                stamp = fact.get("stamp") or package.expected_stamp(enabled, plugin_dirs=[])
                inputs = {"repair": True}
            else:
                enabled = sorted(set(fact.get("extras", [])) | set(extras or []))
                stamp = package.expected_stamp(enabled, **inputs)
            if not repair and not explicit and not lazy_installs_allowed() and not _runtime_state_matches(fact, stamp):
                raise _refuse_lazy("venv", str(extras) if extras else "venv out of sync")
            if not repair and _runtime_state_matches(fact, stamp):
                if before_publish is not None:
                    publication = before_publish()
                    if hasattr(publication, "finish"):
                        publication.finish()
                receipt.record_venv_rebuild(False, "already in sync")
                outcome = "ok"
                return
            receipt.record_feature_list(enabled)
            undo = None
            try:
                result = package.apply(enabled, **inputs) or {}
                if before_publish is not None:
                    undo = before_publish()
                facts.record_state("venv", stamp, enabled, **result)
                if hasattr(undo, "finish"):
                    undo.finish()
                receipt.record_venv_rebuild(True)
            except BaseException:
                if undo is not None:
                    try:
                        undo()
                    except Exception:
                        LOG.exception("pm sync: publish undo failed; config may drift")
                raise
        outcome = "ok"
    except BaseException as exc:
        receipt.record_step("dependency-sync", False, f"{type(exc).__name__}: {exc}")
        raise
    finally:
        receipt.finalize(outcome, 0 if outcome == "ok" else 1, token=token)


def adopt() -> bool:
    """First boot of a bundled install: verify the shipped payload, then
    make it THIS machine's installed state. The payload's own shipped
    ``pm/lock.json`` (inside the repo snapshot) is the offline authority:
    for every package it pins, the shipped fact must record the shipped
    identity, package.verify() must pass, and the recorded realized digest
    must match a fresh tree_digest() over the actual bytes. Any failure:
    log the offending package, do NOT write `.adopted`, return False —
    adopt() refuses to vouch for bytes it could not prove.

    Idempotent and cheap-ish: returns False when there is nothing to
    adopt (no shipped facts, or already adopted)."""
    store = _store()
    facts = _facts()
    if not paths.facts_path().is_file():
        return False

    from hermes_cli.runtime_paths import install_state_dir
    marker = install_state_dir(paths.repo_root()) / ".adopted"
    if marker.is_file():
        return False

    shipped_lock = paths.repo_root() / "pm" / "lock.json"
    if shipped_lock.is_file():
        lockfile = Lockfile(shipped_lock)
        target = current_target()
        for name in lockfile.names():
            try:
                package = get_package(name)
            except KeyError:
                continue
            if isinstance(package, StatePackage):
                continue
            fact = facts.get(name)
            if not facts.installed(
                name, lockfile.version(name), store.root, _identity(lockfile, name, target)
            ):
                LOG.warning(
                    "pm adopt: refusing %s: fact missing, legacy (no recorded "
                    "identity), or does not match the shipped lock",
                    name,
                )
                return False
            entry = store.entry(fact["entry"])
            reason = package.verify(entry, target)
            if reason:
                LOG.warning(
                    "pm adopt: refusing %s: staged entry failed verification: %s",
                    name, reason,
                )
                return False
            if fact.get("digest") != tree_digest(entry):
                LOG.warning(
                    "pm adopt: refusing %s: realized bytes do not match the "
                    "recorded digest",
                    name,
                )
                return False

    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("", encoding="utf-8")
    except OSError:
        LOG.warning("could not record payload verification", exc_info=True)
        return False

    return True


def check() -> list[str]:
    """The startup check: cheap stamp comparisons of the installed state
    against the lockfile. Returns problems; empty means healthy. Never
    installs, never touches the network. An install pm has never touched
    (no installed-state file) reports nothing — pm only vouches for what
    it installed. Lockfile packages this build doesn't know (version skew
    during a partial update) are skipped, not fatal."""
    if not paths.facts_path().is_file() and not paths.runtime_facts_path().is_file():
        return []

    problems: list[str] = []
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
            problems.append(f"{name}: not installed or outdated")
    try:
        venv = get_package("venv")
    except KeyError:
        venv = None
    if venv is not None and (paths.runtime_facts_path().is_file() or facts.get("venv") is not None):
        try:
            if not venv_is_current():
                problems.append("venv: out of sync with uv.lock")
        except (OSError, RuntimeError, ValueError) as exc:
            problems.append(f"venv: {exc}")
    return problems


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


def activate() -> None:
    """Make the installed store usable: prepend its tool dirs to
    os.environ['PATH'] so reactive `shutil.which('git'|'bash'|'ffmpeg'|...)`
    resolves the bundled binaries. The gate is `check()` — if the store is
    broken, refuse to inject (fail fast rather than serving a partial PATH).

    This is the ONE sanctioned global PATH write: PATH is the discovery
    contract every `which` reads, not a tool-specific env leak. Store-first
    unconditionally — pinned bundled versions win on dev machines too.
    """
    import os

    if check():
        return  # broken store → do not provision; callers surface `hermes pm install`
    dirs = _store_path_dirs()
    if not dirs:
        return
    existing = os.environ.get("PATH", "")
    prefix = os.pathsep.join(dirs)
    existing_lower = {p.lower() for p in existing.split(os.pathsep) if p}
    missing = [d for d in dirs if d.lower() not in existing_lower]
    if missing:
        os.environ["PATH"] = os.pathsep.join([*missing, existing]) if existing else os.pathsep.join(missing)
