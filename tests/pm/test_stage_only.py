"""stage_only / _install regressions: a VALID published entry must never be
deleted (verify() returns '' on success), and stage_only must honor a
same-version hash repin (the entry marker design, like facts' identity).

Everything runs inside a temp HERMES_RUNTIME_DIR sandbox: the store and
facts live under tmp_path, and the lockfile + package registry are faked,
so no network and no real install state is touched.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest

from pm.package import InstallError, Package
from pm.store import Store

ensure_mod = importlib.import_module("pm.ensure")


class _FakeLockfile:
    def __init__(self, version: str, artifacts: list[dict]):
        self._version = version
        self._artifacts = artifacts

    def version(self, name: str):
        return self._version

    def artifacts(self, name: str, target: str):
        return self._artifacts


class _FakePackage(Package):
    """unpack() turns the verified archive bytes into one binary file."""

    name = "stage-test"

    def missing_reason(self, target: str):
        return None

    def unpack(self, archive: Path, staged: Path, target: str) -> None:
        staged.mkdir(parents=True, exist_ok=True)
        (staged / "bin").mkdir(exist_ok=True)
        (staged / "bin" / "tool").write_bytes(archive.read_bytes())

    def verify(self, entry: Path, target: str) -> str:
        return "" if (entry / "bin" / "tool").is_file() else "bin/tool missing"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _seed_fetch_cache(store: Store, payload: bytes) -> None:
    """Pre-populate the store's download cache so store.fetch() proves the
    digest from local bytes instead of touching the network."""
    entry = store.entry(f"fetch-{_sha(payload)}")
    entry.mkdir(parents=True)
    (entry / "payload.bin").write_bytes(payload)


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(runtime))
    store = Store(runtime)
    package = _FakePackage()
    monkeypatch.setattr(ensure_mod, "get_package", lambda name: package)
    monkeypatch.setattr(ensure_mod, "_store", lambda: store)
    return store


def _arm_lock(monkeypatch, artifacts: list[dict]):
    lock = _FakeLockfile("1.0", artifacts)
    monkeypatch.setattr(ensure_mod, "_lockfile", lambda: lock)
    return lock


TARGET = "linux-arm64-bionic"
ENTRY = "stage-test-1.0-linux-arm64-bionic"


def test_stage_only_keeps_valid_entry(tmp_path, sandbox, monkeypatch):
    """A published entry that verifies must be returned as-is: the
    verify contract is '' on success, so an inverted predicate here would
    delete and re-fetch a perfectly good entry on every call."""
    payload_a = b"tool-bytes-a"
    _arm_lock(
        monkeypatch,
        [{"url": "https://example.test/tool.deb", "sha256": _sha(payload_a)}],
    )
    _seed_fetch_cache(sandbox, payload_a)

    first = ensure_mod.stage_only("stage-test", TARGET)
    assert (first / "bin" / "tool").read_bytes() == payload_a

    # Mark the published entry: a delete-and-rebuild loses this file.
    (first / "sentinel").write_text("published")

    again = ensure_mod.stage_only("stage-test", TARGET)
    assert again == first
    assert (again / "sentinel").read_text() == "published", (
        "stage_only deleted and rebuilt a VALID published entry"
    )


def test_install_repairs_unproven_entry_and_records_facts(tmp_path, sandbox, monkeypatch):
    """_install with no matching facts must RE-REALIZE the entry: an
    unproven entry (nothing vouches its bytes came from this pin) is
    fail-closed rebuilt, then facts are recorded. Same shape as the
    test_pm_authority repin repair, driven through _install directly."""
    payload_a = b"tool-bytes-a"
    _arm_lock(
        monkeypatch,
        [{"url": "https://example.test/tool.deb", "sha256": _sha(payload_a)}],
    )
    _seed_fetch_cache(sandbox, payload_a)

    package = ensure_mod.get_package("stage-test")
    facts = ensure_mod._facts()
    store = sandbox
    entry_name = package.store_entry("1.0", TARGET)
    sandbox.entry(entry_name).mkdir(parents=True)
    (sandbox.entry(entry_name) / "bin").mkdir()
    (sandbox.entry(entry_name) / "bin" / "tool").write_bytes(b"stale-bytes")
    (sandbox.entry(entry_name) / "sentinel").write_text("unproven")

    ensure_mod._install(package, ensure_mod._lockfile(), facts, store, TARGET)

    assert (sandbox.entry(entry_name) / "bin" / "tool").read_bytes() == payload_a, (
        "_install kept an entry nothing proved against the pin"
    )
    assert not (sandbox.entry(entry_name) / "sentinel").exists()
    recorded = facts.get("stage-test")
    assert recorded and recorded["version"] == "1.0"


def test_stage_only_repin_same_version_rebuilds(tmp_path, sandbox, monkeypatch):
    """Repinning the SAME version to a different sha256 must rebuild the
    staged entry: without an entry marker the stale bytes are handed back."""
    payload_a = b"tool-bytes-a"
    payload_b = b"tool-bytes-b-repin"
    assert payload_a != payload_b

    _arm_lock(
        monkeypatch,
        [{"url": "https://example.test/tool.deb", "sha256": _sha(payload_a)}],
    )
    _seed_fetch_cache(sandbox, payload_a)
    first = ensure_mod.stage_only("stage-test", TARGET)
    assert (first / "bin" / "tool").read_bytes() == payload_a

    # Repin: same version, new digest.
    _arm_lock(
        monkeypatch,
        [{"url": "https://example.test/tool.deb", "sha256": _sha(payload_b)}],
    )
    _seed_fetch_cache(sandbox, payload_b)
    repinned = ensure_mod.stage_only("stage-test", TARGET)
    assert (repinned / "bin" / "tool").read_bytes() == payload_b, (
        "stage_only returned stale bytes after a same-version hash repin"
    )

    # And the rebuilt entry is again idempotent.
    stable = ensure_mod.stage_only("stage-test", TARGET)
    assert (stable / "bin" / "tool").read_bytes() == payload_b


@pytest.mark.parametrize("error_name", ["HashError", "DownloadPaused"])
def test_permanent_or_paused_download_is_not_retried(error_name):
    from pm import downloader
    from pm.network import retry_network

    def fail():
        raise getattr(downloader, error_name)("stop")

    with pytest.raises(getattr(downloader, error_name)):
        retry_network(fail, wait=lambda _: pytest.fail("permanent failure was retried"))


@pytest.mark.parametrize("failure", ["fetch", "publish"])
def test_repin_failure_preserves_previous_staged_entry(sandbox, monkeypatch, failure):
    payload = b"original"
    _arm_lock(monkeypatch, [{"url": "https://example.test/tool", "sha256": _sha(payload)}])
    _seed_fetch_cache(sandbox, payload)
    entry = ensure_mod.stage_only("stage-test", TARGET)
    marker = (entry / ".pm-stage-pin.json").read_bytes()

    replacement = b"replacement"
    _arm_lock(monkeypatch, [{"url": "https://example.test/tool", "sha256": _sha(replacement)}])
    _seed_fetch_cache(sandbox, replacement)

    def fail(*args, **kwargs):
        raise InstallError("stage-test", "injected staging failure")

    if failure == "fetch":
        monkeypatch.setattr(sandbox, "fetch", fail)
    else:
        monkeypatch.setattr(sandbox, "publish", fail)
    with pytest.raises(InstallError, match="injected staging failure"):
        ensure_mod.stage_only("stage-test", TARGET)

    assert (entry / "bin/tool").read_bytes() == payload
    assert (entry / ".pm-stage-pin.json").read_bytes() == marker
