"""Vendored config-plane conformance fixtures, run against the agent's client-side rules
(contract.md §13): path grammar, secret-shaped keys, the profile-level write check fed
``expect.locks``, and ``defaultsMerge`` against ``_deep_merge``. The digest test pins the vendored
bytes to the manifest and to ``SOURCE.json``, so an edited fixture fails here, not in production."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from plugins.config_backends.remote import paths as P
from plugins.config_backends.remote.diff import write_check
from plugins.config_backends.remote.values import secret_literal_path

VENDOR = Path(__file__).resolve().parents[3] / "plugins" / "config_backends" / "remote" / "vendor"
MANIFEST = VENDOR / "conformance" / "config-fixtures.sha256"
RESOLUTION = sorted((VENDOR / "conformance" / "config-resolution").glob("*.json"))


def _load(name: str):
    return json.loads((VENDOR / "conformance" / name).read_text(encoding="utf-8"))


def test_vendored_fixtures_match_manifest_and_source():
    source = json.loads((VENDOR / "SOURCE.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(MANIFEST.read_bytes()).hexdigest() == source["fixtureDigest"]
    listed = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        digest, rel = line.split(None, 1)
        listed[rel.strip()] = digest
    for rel, digest in listed.items():
        assert hashlib.sha256((VENDOR / rel).read_bytes()).hexdigest() == digest, rel
    on_disk = {str(p.relative_to(VENDOR)) for p in (VENDOR / "conformance").rglob("*.json")}
    assert on_disk == set(listed), "every vendored fixture is in the manifest and vice versa"
    assert len(RESOLUTION) > 0


@pytest.mark.parametrize("vector", _load("config-paths.json")["valid"], ids=lambda v: v["encoded"])
def test_valid_paths_round_trip(vector):
    assert P.encode(vector["segments"]) == vector["encoded"]
    assert list(P.decode(vector["encoded"])) == vector["segments"]


@pytest.mark.parametrize("encoded", _load("config-paths.json")["invalid"], ids=repr)
def test_invalid_paths_rejected(encoded):
    with pytest.raises(P.PathError):
        P.decode(encoded)


@pytest.mark.parametrize("vector", _load("config-paths.json")["covers"],
                         ids=lambda v: f"{v['lock']}|{v['path']}")
def test_lock_coverage(vector):
    assert P.covers(P.decode(vector["lock"]), P.decode(vector["path"])) is vector["covers"]


@pytest.mark.parametrize("vector", _load("config-secret-keys.json")["vectors"],
                         ids=lambda v: f"{v['path']}={v['value']!r}")
def test_secret_key_vectors(vector):
    path = P.decode(vector["path"])
    rejected = secret_literal_path(path, vector["value"]) is not None
    assert rejected is (vector["expect"] == "reject")


def _profile_writes():
    for f in RESOLUTION:
        fx = json.loads(f.read_text(encoding="utf-8"))
        locks = [(P.decode(lk["path"]), lk["level"]) for lk in fx["expect"].get("locks", [])]
        for i, w in enumerate(fx.get("writes", [])):
            if w.get("by") == "profile" and w.get("op") in {"set", "unset"}:
                yield pytest.param(w, locks, id=f"{f.stem}#{i}")


@pytest.mark.parametrize("write,locks", list(_profile_writes()))
def test_profile_write_check(write, locks):
    path = P.decode(write["path"])
    sets = {path: write.get("value")} if write["op"] == "set" else {}
    unsets = [path] if write["op"] == "unset" else []
    refused = write_check(sets, unsets, locks)
    secret = secret_literal_path(path, write.get("value")) if write["op"] == "set" else None
    expect = write["expect"]
    if expect["result"] == "accept":
        assert refused is None and secret is None
        return
    if expect["error"] == "config_key_locked":
        assert refused is not None
        assert P.encode(refused[0]) == expect["path"] and refused[1] == expect["lockedBy"]
    elif expect["error"] == "config_secret_literal":
        assert refused is None and secret is not None
    else:  # an error the agent does not pre-check (the plane's own validation)
        pytest.skip(f"server-only error {expect['error']}")


def _defaults_merges():
    for f in RESOLUTION:
        fx = json.loads(f.read_text(encoding="utf-8"))
        if "defaultsMerge" in fx:
            yield pytest.param(fx["defaultsMerge"], fx["expect"]["effective"], id=f.stem)


@pytest.mark.parametrize("merge,effective", list(_defaults_merges()))
def test_defaults_merge_matches_deep_merge(merge, effective):
    from hermes_cli.config import _deep_merge
    assert _deep_merge(merge["defaults"], effective) == merge["expect"]
