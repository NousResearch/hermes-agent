"""Independent resolution decisions; HTTP parsing lives in request-reuse tests."""

from __future__ import annotations

import pytest

from pm.update import (
    best_in_minor,
    minor_of,
    resolve_package,
    version_key,
)


def _pkg(pkg_name="node", version_style="semver", **latest):
    """A stub package whose latest_versions returns canned per-target lists."""

    class _P:
        def __init__(self):
            self.name = pkg_name
            self.version_style = version_style
            self._latest = dict(latest)

        def latest_versions(self, target, locked=None):
            return list(self._latest.get(target, []))

    return _P()


def test_version_key_sorts_numeric_and_suffixes():
    assert version_key("2.53.0+5") > version_key("2.53.0+3")
    assert version_key("3.14.7+20260901") > version_key("3.14.7+20260900")
    assert version_key("26.8.1") > version_key("26.7.0")
    assert version_key("10362") > version_key("10361")
    # prerelease-ish segments sort after numerics
    assert version_key("9.0.1") < version_key("9.0.1-rc1")


def test_minor_of():
    assert minor_of("26.7.0") == (26, 7)
    assert minor_of("3.14.7+20260901") == (3, 14)
    assert minor_of("10362") is None  # single component


def test_best_in_minor():
    versions = ["9.0.3", "9.0.1", "9.1.0", "8.4.9"]
    assert best_in_minor(versions, (9, 0)) == "9.0.3"
    assert best_in_minor(versions, (9, 1)) == "9.1.0"
    assert best_in_minor(versions, (10, 0)) is None


@pytest.mark.parametrize("style, candidates, locked, version, per_target, changed, reason", [
    ("semver", {"a": ["10.0.1", "9.0.3"], "b": ["9.0.3", "10.0.1"]}, "9.0.3", "10.0.1", {"a": "10.0.1", "b": "10.0.1"}, True, ""),
    ("semver", {"a": ["26.7.0"]}, "26.7.0", "26.7.0", {"a": "26.7.0"}, False, ""),
    ("semver", {"a": ["2.97.0"], "b": ["2.96.0"]}, "2.95.0", None, {}, False, "no shared version"),
    ("minor", {"a": ["9.1.2", "9.0.1"], "b": ["9.1.0", "9.0.1"]}, "9.0.1", "9.1", {"a": "9.1.2", "b": "9.1.0"}, True, ""),
    ("minor", {"a": ["9.1.2"], "b": ["9.0.3"]}, "9.0.1", None, {}, False, "no shared minor"),
    ("minor", {"a": ["9.1.2"], "b": ["9.1.0"]}, "9.1", "9.1", {"a": "9.1.2", "b": "9.1.0"}, False, ""),
    ("semver", {"a": []}, "1208+145", None, {}, False, "no source"),
    ("semver", {"a": ["26.8.1"], "b": []}, "26.7.0", "26.8.1", {"a": "26.8.1"}, True, ""),
])
def test_resolve_package_decisions(style, candidates, locked, version, per_target, changed, reason):
    result = resolve_package(_pkg(version_style=style, **candidates), list(candidates), locked=locked)
    assert (result.version, result.per_target, result.changed) == (version, per_target, changed)
    if reason:
        assert result.reason == reason
