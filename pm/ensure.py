"""ensure(): make the installed state match the lockfile for a package,
and hand back its composed environment."""

from __future__ import annotations

import logging
import shutil
from typing import Optional

from pm import paths
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
    return list((_facts().get("venv") or {}).get("extras", []))


def is_installed(name: str) -> bool:
    lockfile = _lockfile()
    facts = _facts()
    return facts.installed(
        name,
        lockfile.version(name),
        _store().root,
        _identity(lockfile, name, current_target()),
    )


def sealed() -> bool:
    """A bundled payload is read-only: its store sits beside the bundle
    manifest. Asking a sealed install for MORE than it shipped is a
    Sealed = bundled LAYOUT only; adoption verification is adopt()'s job."""
    return (paths.store_root().parent / "manifest.json").is_file()


def _refuse_lazy(name: str, what: str) -> InstallError:
    if sealed():
        return InstallError(
            name,
            f"this install is sealed and does not ship: {what}",
            "the bundle bakes its entire tree at build time; rebuild the bundle",
        )
    return InstallError(
        name,
        f"not installed and lazy installs are disabled: {what}",
        "enable security.allow_lazy_installs or run `hermes pm install`",
    )


def _remove_entry(store: Store, entry_name: str) -> None:
    """Remove a published store entry before re-realizing it. NOT
    fire-and-forget: a silent failure here would leave the stale entry in
    place and publish() would then keep it (a concurrent-winner guard),
    re-recording facts over bytes the new lock never produced. Defender /
    indexer handles on Windows make removal transiently fail, so retry —
    then fail loudly."""
    import time

    entry = store.entry(entry_name)
    for attempt in range(5):
        try:
            shutil.rmtree(entry)
            return
        except FileNotFoundError:
            return
        except OSError as e:
            if attempt == 4:
                raise
            time.sleep(0.2 * (attempt + 1))


def _install(
    package: Package,
    lockfile: Lockfile,
    facts: Facts,
    store: Store,
    target: str,
    progress=None,
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
        facts.reload()
        if facts.installed(
            package.name, version, store.root, _identity(lockfile, package.name, target)
        ):
            return
        entry = store.entry(entry_name)
        _cond = store.published(entry_name) and package.verify(entry, target) == ""
        if _cond:
            _remove_entry(store, entry_name)
        if not store.published(entry_name):
            if not artifacts:
                raise InstallError(
                    package.name,
                    f"no artifact for {target} in the lockfile",
                    "run `hermes pm lock --bump` for this package",
                )
            with store.scratch() as scratch:
                staged = scratch / "tree"
                try:
                    for index, artifact in enumerate(artifacts):
                        label = f"{index + 1}/{len(artifacts)}" if len(artifacts) > 1 else ""
                        archive = store.fetch(
                            artifact["url"], artifact["sha256"], scratch,
                            progress=_artifact_progress(
                                progress, index, len(artifacts)),
                        )
                        if progress is not None:
                            progress("unpack", 0, 0, label)
                        if index == 0:
                            package.unpack(archive, staged, target)
                            continue
                        # unpack() empties its destination by contract, so
                        # a second archive must be unpacked apart and moved
                        # in — extracting over `staged` would delete the
                        # first archive's files.
                        extra = scratch / f"extra-{index}"
                        package.unpack(archive, extra, target)
                        merge_tree(extra, staged)
                    package.stage(store, staged, version, target)
                    store.publish(staged, entry_name)
                except InstallError:
                    raise
                except Exception as e:
                    raise InstallError(package.name, f"install failed: {e}") from e

        reason = package.verify(entry, target)
        if reason:
            raise InstallError(package.name, f"published entry failed verification: {reason}")

        previous = facts.get(package.name)
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
        env = package.env(entry, target)
        facts.record(
            package.name,
            version,
            entry_name,
            env,
            store.root,
            target=target,
            artifacts=[a["sha256"] for a in artifacts],
            digest=tree_digest(entry),
        )
        if previous and previous.get("version") != version:
            package.migrate(previous["version"], version)


def ensure(
    name: str,
    *,
    base_env: Optional[dict] = None,
    explicit: bool = False,
    progress=None,
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
    facts = _facts()
    store = _store()
    target = current_target()

    chain = walk([name])
    missing = [
        p
        for p in chain
        if not facts.installed(
            p.name, lockfile.version(p.name), store.root, _identity(lockfile, p.name, target)
        )
    ]

    if missing and (sealed() or (not explicit and not lazy_installs_allowed())):
        raise _refuse_lazy(name, ", ".join(p.name for p in missing))

    for package in missing:
        _install(package, lockfile, facts, store, target, progress=progress)
    if missing:
        facts.reload()

    diffs = [facts.env_for(p.name, store.root) for p in chain]
    return Runner(name, compose_env(diffs, base=base_env))


def env_for(*names: str, base_env: Optional[dict] = None) -> dict[str, str]:
    """Composed env of already-installed packages only. Never installs,
    never raises on missing packages — they contribute nothing."""
    facts = _facts()
    store = _store()
    lockfile = _lockfile()
    target = current_target()
    diffs: list[dict] = []
    for name in names:
        try:
            chain = walk([name])
        except KeyError:
            continue
        for package in chain:
            if facts.installed(
                package.name,
                lockfile.version(package.name),
                store.root,
                _identity(lockfile, package.name, target),
            ):
                diffs.append(facts.env_for(package.name, store.root))
    return compose_env(diffs, base=base_env)


def sync_venv(extras: Optional[list[str]] = None, *, explicit: bool = False) -> None:
    """Make the venv match uv.lock + the enabled extras. Extras union into
    the installed state (one ledger); no-op when the stamp already matches.
    ``explicit`` marks a deliberate install command (`hermes pm install`,
    `hermes update`) — those are the remedy the lazy-install policy points
    at, so the policy does not apply to them.

    Lazy installs OFF = the frozen feature set: when
    security.allow_lazy_installs is false AND the bundle's
    enabled-features.json exists, the feature list is FROZEN to that file
    — requested extras outside it are refused, and plugin members are
    never installed (the bundle IS the install)."""
    from pm.features import read_features

    frozen = read_features() if not lazy_installs_allowed() else None
    if frozen is not None and extras:
        outside = sorted(set(extras) - set(frozen))
        if outside:
            raise _refuse_lazy(
                "venv",
                f"extras {outside} are outside this bundle's frozen feature "
                "set (security.allow_lazy_installs is false)",
            )
    package = get_package("venv")
    facts = _facts()
    fact = facts.get("venv") or {}
    enabled = sorted(set(fact.get("extras", [])) | set(extras or []))
    stamp = package.expected_stamp(enabled)
    if fact.get("stamp") == stamp:
        return
    if sealed() or (not explicit and not lazy_installs_allowed()):
        raise _refuse_lazy("venv", str(extras) if extras else "venv out of sync")
    with _store().install_lock():
        facts.reload()
        fact = facts.get("venv") or {}
        enabled = sorted(set(fact.get("extras", [])) | set(extras or []))
        stamp = package.expected_stamp(enabled)
        if fact.get("stamp") == stamp:
            return
        package.apply(enabled)
        facts.record_state("venv", stamp, enabled)


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

    marker = store.root.parent / ".adopted"
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
        marker.write_text("", encoding="utf-8")
    except OSError:
        pass

    # Bootstrap the mutable venv (sealed installs): lazy-off copies the
    # shipped venv out as the seed; lazy-on builds fresh from the shipped
    # uv cache on first sync. Failure is reported, never fatal — a cold
    # sync still converges.
    try:
        venv_package = get_package("venv")
    except KeyError:
        venv_package = None
    seed = getattr(venv_package, "seed_mutable_venv", None) if venv_package else None
    if seed is not None:
        reason = seed()
        if reason:
            LOG.info("pm adopt: mutable venv seed skipped: %s", reason)

    return True


def check() -> list[str]:
    """The startup check: cheap stamp comparisons of the installed state
    against the lockfile. Returns problems; empty means healthy. Never
    installs, never touches the network. An install pm has never touched
    (no installed-state file) reports nothing — pm only vouches for what
    it installed. Lockfile packages this build doesn't know (version skew
    during a partial update) are skipped, not fatal."""
    if not paths.facts_path().is_file():
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
        if not facts.installed(
            name, lockfile.version(name), store.root, _identity(lockfile, name, target)
        ):
            problems.append(f"{name}: not installed or outdated")
    try:
        venv = get_package("venv")
    except KeyError:
        venv = None
    fact = facts.get("venv")
    if venv is not None and fact is not None:
        expected = venv.expected_stamp(fact.get("extras", []))
        if fact.get("stamp") != expected:
            problems.append("venv: out of sync with uv.lock")
    return problems


def _store_path_dirs() -> list[str]:
    """Composed PATH dirs of all installed (non-internal, on_path) store
    packages, deps-first, deduped. Includes optional packages that are
    *installed* (facts say so) — an installed git/gh must be on PATH even
    though it's not in the root closure. Never installs."""
    import os

    if not paths.facts_path().is_file():
        return []
    facts = _facts()
    store = _store()
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
        if not facts.installed(
            name, lockfile.version(name), store.root, _identity(lockfile, name, target)
        ):
            continue
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



def uv(*, venv=None, realize: bool = True):
    """TRANSITIONAL: (uv path, sanitized env) for call sites that still
    drive uv themselves. Two classes remain: update/repair sites (die with
    the update collapse, plan step 4) and side-venv installs — browser-use
    tool venvs (tools_config, browser_use_cli) and hindsight's
    local_embedded daemon — which survive until pm grows the side-venv
    package kind (plan step 5's remaining half). Must not spread."""
    from pm.packages import uv_env

    env = uv_env()
    if venv is not None:
        env["VIRTUAL_ENV"] = str(venv)
        env.pop("UV_NO_CONFIG", None)

    lockfile = _lockfile()
    facts = _facts()
    store = _store()
    package = get_package("uv")
    if not facts.installed(
        "uv", lockfile.version("uv"), store.root, _identity(lockfile, "uv", current_target())
    ):
        if not realize or not lazy_installs_allowed():
            return None, env
        try:
            _install(package, lockfile, facts, store, current_target())
            facts.reload()
        except Exception:
            import logging

            logging.getLogger(__name__).debug("pm.uv: install failed", exc_info=True)
            return None, env
    fact = facts.get("uv")
    if fact is None:
        return None, env
    binary = package.binary(store.entry(fact["entry"]), current_target())
    if binary is None or not binary.is_file():
        return None, env
    return str(binary), env
