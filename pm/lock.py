"""The lockfile (versions + hashes, machine-written) and the installed-state
file (what is actually on this machine).

lock.json:  {"schema": 1, "packages": {name: {"version": ..., "artifacts": {target: {"url": ..., "sha256": ...}}}}}
facts.json: {"schema": 1, "packages": {name: {"entry": ..., "version": ..., "env": ..., "stamp": ...}}}

A target's value is one {"url", "sha256"} object, or a LIST of them when the
package is split across several archives that must land in one directory
(llama.cpp's engine zip and its cudart zip: Windows resolves a DLL from the
loading executable's own directory). Both shapes read back as a list.

lock.json artifacts carry the RESOLVED url beside the hash: the lockfile is
the complete machine interface (nix reads it as pure data), and the python
url templates are consulted only at `pm lock --bump` time. The "any" target
key covers target-independent artifacts (npm's tarball).

facts.json env values hold {{store}} templates so a CI-built file adopts onto
any machine by substitution.
"""

from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from pathlib import Path

SCHEMA = 1
STORE_TOKEN = "{{store}}"


def _read(path: Path, *, strict: bool = False) -> dict:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return {"schema": SCHEMA, "packages": {}}
    except OSError:
        if strict:
            raise
        return {"schema": SCHEMA, "packages": {}}
    try:
        data = json.loads(text)
        if isinstance(data, dict) and data.get("schema") == SCHEMA and isinstance(data.get("packages"), dict):
            return data
    except ValueError:
        pass
    if strict:
        raise ValueError(f"cannot read recorded package state: {path}")
    if text.strip():
        # An unparsable-but-nonempty state file is evidence, not garbage:
        # the next _write would silently discard every installed-state
        # record. Keep the bytes for post-mortem.
        try:
            from pm.paths import store_root
            if path.parent == store_root() and (store_root().parent / "manifest.json").is_file():
                import logging
                logging.getLogger(__name__).warning("invalid shipped state file: %s", path)
            else:
                path.with_suffix(".corrupt").write_text(text, encoding="utf-8")
        except OSError:
            pass
    return {"schema": SCHEMA, "packages": {}}


def _write(path: Path, data: dict) -> None:
    from hermes_cli.runtime_state import _atomic_bytes
    _atomic_bytes(path, (json.dumps(data, indent=2, sort_keys=True) + "\n").encode("utf-8"))


class StaleLockRow(RuntimeError):
    """A concurrent editor changed a row that this writer plans to publish."""


class Lockfile:
    """Snapshot of lock.json with conflict-checked publication of changed pins."""

    def __init__(self, path: Path):
        self.path = Path(path).resolve()
        self._packages = _read(self.path)["packages"]
        self._base = deepcopy(self._packages)
        self._touched: set[str] = set()

    def version(self, name: str) -> str | None:
        return (self._packages.get(name) or {}).get("version")

    def artifacts(self, name: str, target: str) -> list[dict]:
        """Every archive this target needs, in extraction order. One
        artifact and a list of them are the same thing here — the single
        form is just the common case written short."""
        artifacts = (self._packages.get(name) or {}).get("artifacts") or {}
        found = artifacts.get(target)
        if found is None:
            found = artifacts.get("any")
        if found is None:
            return []
        return list(found) if isinstance(found, list) else [found]

    def pinned_artifacts(self, name: str) -> dict:
        """Return the complete pin table, including targets without discovery."""
        return deepcopy((self._packages.get(name) or {}).get("artifacts") or {})

    def names(self) -> list[str]:
        return sorted(self._packages)

    def set_pin(self, name: str, version: str, artifacts: dict[str, dict]) -> None:
        self._packages[name] = {"version": version, "artifacts": deepcopy(artifacts)}
        self._touched.add(name)

    def save(self) -> None:
        """Merge touched rows under the file's lock, or leave all rows unchanged."""
        from hermes_cli.runtime_state import _lock

        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path.with_name(f".{self.path.name}.lock"), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            _lock(fd, wait=True)
            current = _read(self.path, strict=True)["packages"]
            updated = dict(current)
            for name in self._touched:
                intended = self._packages[name]
                if current.get(name) != self._base.get(name) and current.get(name) != intended:
                    raise StaleLockRow(f"{name} changed in {self.path}; read the lockfile again and retry")
                updated[name] = intended
            if updated != current or not self.path.exists():
                _write(self.path, {"schema": SCHEMA, "packages": updated})
            self._packages = updated
            self._base = deepcopy(updated)
            self._touched.clear()
        finally:
            os.close(fd)


def termux_docker_digest() -> str:
    """The pinned termux-docker image digest (pm package `termux-docker`).

    Single reader for every consumer -- the build/deb scripts, the builder
    image script, and the release workflow -- so a lock-schema change lands
    once.
    """
    from pm.paths import lockfile_path

    return Lockfile(lockfile_path()).version("termux-docker")


class Facts:
    """The installed-state file. Written only by pm."""

    def __init__(self, path: Path, *, strict: bool = False):
        self.path = path
        self._packages = _read(path, strict=strict)["packages"]

    def reload(self) -> None:
        self._packages = _read(self.path)["packages"]

    def get(self, name: str) -> dict | None:
        return self._packages.get(name)

    def refresh_digests(self, store_root: Path) -> int:
        """Publish all tool digests after packaging finishes changing their bytes."""
        from pm.store import tree_digest

        packages = _read(self.path, strict=True)["packages"]
        root = store_root.resolve()
        count = 0
        for name, fact in packages.items():
            if not isinstance(fact, dict):
                raise ValueError(f"invalid payload fact: {name}")
            if "entry" not in fact and "stamp" in fact:
                continue
            entry_name = fact.get("entry")
            artifacts = fact.get("artifacts")
            if (not isinstance(entry_name, str) or entry_name in ("", ".", "..")
                    or any(c in entry_name for c in "/\\:")
                    or not isinstance(fact.get("version"), str) or not fact["version"]
                    or not isinstance(fact.get("target"), str) or not fact["target"]
                    or not isinstance(artifacts, list) or not artifacts
                    or any(not isinstance(sha, str) or not re.fullmatch(r"[a-f0-9]{64}", sha)
                           for sha in artifacts)):
                raise ValueError(f"incomplete payload tool: {name}")
            entry = root / entry_name
            if not entry.is_dir() or not entry.resolve().is_relative_to(root):
                raise ValueError(f"incomplete payload tool: {name}")
            fact["digest"] = tree_digest(entry)
            count += 1
        if not count:
            raise ValueError(f"payload facts carry no tool entries: {self.path}")
        _write(self.path, {"schema": SCHEMA, "packages": packages})
        self._packages = packages
        return count

    def installed(
        self,
        name: str,
        expected_version: str | None,
        store_root: Path,
        expected_identity=None,
    ) -> bool:
        """``expected_identity`` is (target, tuple(artifact sha256s)) — the
        identity the lock currently pins. When given, the fact must record
        that exact identity: a same-version re-pin, a different target, or
        a fact written before identity existed all count as NOT installed
        and force a reinstall. ``None`` keeps the version+path check."""
        fact = self._packages.get(name)
        if not fact:
            return False
        if expected_version is not None and fact.get("version") != expected_version:
            return False
        if expected_identity is not None:
            target, shas = expected_identity
            if "target" not in fact or "artifacts" not in fact:
                # Legacy fact: pre-dates digest-bound identity. Not
                # vouchable — force one reinstall.
                return False
            if (fact["target"], tuple(fact["artifacts"])) != (target, tuple(shas)):
                return False
        return (store_root / fact["entry"]).exists()

    def env_for(self, name: str, store_root: Path) -> dict:
        fact = self._packages.get(name) or {}
        return _resolve(fact.get("env", {}), store_root)

    def _merge_and_write(self, name: str, fact: dict) -> None:
        """Read-modify-write against disk so concurrent installs of
        different packages never clobber each other."""
        on_disk = _read(self.path)["packages"]
        for key, value in self._packages.items():
            on_disk.setdefault(key, value)
        self._packages = on_disk
        self._packages[name] = fact
        _write(self.path, {"schema": SCHEMA, "packages": self._packages})

    def record(
        self,
        name: str,
        version: str,
        entry: str,
        env: dict,
        store_root: Path,
        target: str | None = None,
        artifacts: list[str] | None = None,
        digest: str | None = None,
    ) -> None:
        """``target``/``artifacts`` record the identity this install came
        from (work item 1); ``digest`` is the realized tree digest of the
        published entry (work item 2). All three are additive — schema
        stays 1 and older facts without them still read back."""
        fact: dict = {
            "entry": entry,
            "version": version,
            "env": _templatize(env, store_root),
        }
        if target is not None:
            fact["target"] = target
        if artifacts is not None:
            fact["artifacts"] = list(artifacts)
        if digest is not None:
            fact["digest"] = digest
        self._merge_and_write(name, fact)

    def record_state(
        self, name: str, stamp: str, extras: list[str], *,
        environment: Path | None = None, resolved_lock: Path | None = None,
    ) -> None:
        """Commit a verified state package and its selected environment atomically."""
        fact = {"stamp": stamp, "extras": extras}
        if environment is not None:
            fact["environment"] = str(environment.resolve())
        if resolved_lock is not None:
            fact["resolved_lock"] = str(resolved_lock.resolve())
        self._merge_and_write(name, fact)

    def retain(self, names: set[str]) -> None:
        """Drop unselected package facts from an exclusively owned staged store."""
        packages = _read(self.path, strict=True)["packages"]
        retained = {name: fact for name, fact in packages.items() if name in names}
        if retained != packages:
            _write(self.path, {"schema": SCHEMA, "packages": retained})
        self._packages = retained

    def entries_in_use(self) -> set[str]:
        return {f["entry"] for f in self._packages.values() if "entry" in f}


def _templatize(env: dict, store_root: Path) -> dict:
    root = str(store_root)
    out = {}
    for key, value in env.items():
        if isinstance(value, list):
            out[key] = [str(v).replace(root, STORE_TOKEN) for v in value]
        else:
            out[key] = str(value).replace(root, STORE_TOKEN)
    return out


def _resolve(env: dict, store_root: Path) -> dict:
    root = str(store_root)
    out = {}
    for key, value in env.items():
        if isinstance(value, list):
            out[key] = [str(v).replace(STORE_TOKEN, root) for v in value]
        else:
            out[key] = str(value).replace(STORE_TOKEN, root)
    return out
