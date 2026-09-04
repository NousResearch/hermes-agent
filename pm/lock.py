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
from pathlib import Path

SCHEMA = 1
STORE_TOKEN = "{{store}}"


def _read(path: Path) -> dict:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError:
        return {"schema": SCHEMA, "packages": {}}
    try:
        data = json.loads(text)
        if data.get("schema") == SCHEMA and isinstance(data.get("packages"), dict):
            return data
    except ValueError:
        pass
    if text.strip():
        # An unparsable-but-nonempty state file is evidence, not garbage:
        # the next _write would silently discard every installed-state
        # record. Keep the bytes for post-mortem.
        try:
            path.with_suffix(".corrupt").write_text(text, encoding="utf-8")
        except OSError:
            pass
    return {"schema": SCHEMA, "packages": {}}


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class Lockfile:
    """Read side of lock.json. Written only by `pm lock --bump` (cli)."""

    def __init__(self, path: Path):
        self.path = path
        self._packages = _read(path)["packages"]

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

    def names(self) -> list[str]:
        return sorted(self._packages)

    def set_pin(self, name: str, version: str, artifacts: dict[str, dict]) -> None:
        self._packages[name] = {"version": version, "artifacts": artifacts}

    def save(self) -> None:
        _write(self.path, {"schema": SCHEMA, "packages": self._packages})


class Facts:
    """The installed-state file. Written only by pm."""

    def __init__(self, path: Path):
        self.path = path
        self._packages = _read(path)["packages"]

    def reload(self) -> None:
        self._packages = _read(self.path)["packages"]

    def get(self, name: str) -> dict | None:
        return self._packages.get(name)

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

    def record_state(self, name: str, stamp: str, extras: list[str]) -> None:
        """State packages (the venv) have a stamp and extras, no entry."""
        self._merge_and_write(name, {"stamp": stamp, "extras": extras})

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
