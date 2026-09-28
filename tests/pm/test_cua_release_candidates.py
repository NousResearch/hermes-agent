"""Cua's stable channel is encoded by its tag, not GitHub's monorepo flag."""
import json
from urllib.error import HTTPError

import pytest

from pm import update
from pm.registry import get_package
from pm.store import ALL_TARGETS
from tests.pm._range_server import RangeHandler, dl_server  # noqa: F401
from tests.pm.test_update_request_reuse import upstream  # noqa: F401


PATH = "/repos/trycua/cua/releases?per_page=30&page=1"


def release(tag, **flags):
    return {"tag_name": tag, "prerelease": True, "draft": False, **flags}


def test_cua_stable_candidates_drive_real_multi_target_resolution(upstream):
    calls, _ = upstream
    stable = ["2.3.4", "2.3.3"]
    payload = [release(f"cua-driver-rs-v{v}") for v in stable] + [
        release("cua-driver-rs-v99.0.0", draft=True),
        release("nightly-cua-driver-rs-v99.0.0-nightly.20990101"),
        release("cua-driver-rs-v99.0.0-rc.1"),
        release("cua-driver-rs-v99.0.0-nightly.20990101", prerelease=False),
        release("cua-driver-rs-vsandbox-v99.0.0"),
        release("sandbox-v99.0.0", prerelease=False),
        release("99.0.0", prerelease=False),
        release("latest", prerelease=False),
    ]
    package = get_package("cua-driver")
    targets = [t for t in ALL_TARGETS if package.missing_reason(t) is None]
    RangeHandler.payloads = {PATH: json.dumps(payload[:len(stable)]).encode()}
    assert package.latest_versions(targets[0]) == stable
    calls.clear()
    RangeHandler.payloads = {PATH: json.dumps(payload).encode()}
    decision = update.resolve_package(package, targets, locked="2.3.2")
    assert decision.version == stable[0]
    assert decision.changed
    assert decision.per_target == dict.fromkeys(targets, stable[0])
    assert calls == [(PATH, None)]
    assert package.latest_versions(targets[0]) == stable

    # A future empty/draft-only index never invents a candidate.
    for payload in ([], [release("cua-driver-rs-v2.3.4", draft=True)]):
        RangeHandler.payloads = {PATH: json.dumps(payload).encode()}
        decision = update.resolve_package(package, targets, locked="2.3.2")
        assert not decision.changed
        assert decision.version is None
        assert decision.reason == "no source"

    # Errors remain errors, and a failed resolution does not poison recovery.
    _, failures = upstream
    failures[PATH] = [404]
    RangeHandler.payloads = {PATH: json.dumps([release("cua-driver-rs-v2.3.4")]).encode()}
    with pytest.raises(HTTPError, match="404"):
        update.resolve_package(package, targets, locked="2.3.2")
    assert update.resolve_package(package, targets, locked="2.3.2").version == stable[0]

    # Ordinary stable releases remain eligible, but never downgrade a newer pin.
    RangeHandler.payloads = {PATH: json.dumps([
        release("cua-driver-rs-v2.3.4", prerelease=False),
    ]).encode()}
    decision = update.resolve_package(package, targets, locked="3.0.0")
    assert decision.version == stable[0]
    assert not decision.changed


def test_other_packages_keep_rejecting_prereleases(upstream):
    path = "/repos/BurntSushi/ripgrep/releases?per_page=30&page=1"
    RangeHandler.payloads = {path: json.dumps([
        release("99.0.0"), release("3.2.1", prerelease=False),
        release("99.0.1", prerelease=False, draft=True),
    ]).encode()}
    assert get_package("ripgrep").latest_versions("win32-x64") == ["3.2.1"]
