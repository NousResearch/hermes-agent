"""Native platform policy table; collection wiring lives in test_os_marker_gating."""
import sys

import pytest
from tests._fixtures.platform_gating import _host_matches_platforms, _platform_machine


@pytest.mark.parametrize("specs,hosts", [
    (("linux",), {"linux"}), (("macos",), {"darwin"}),
    (("windows",), {"win32"}), (("WINDOWS",), {"win32"}),
    (("posix",), {"linux", "darwin"}), (("not macos",), {"linux", "win32"}),
    (("not windows",), {"linux", "darwin"}), (("linux", "macos"), {"linux", "darwin"}),
    (("any",), {"linux", "darwin", "win32"}), ((), {"linux", "darwin", "win32"}),
])
def test_native_spec_table(specs, hosts):
    ok, reason = _host_matches_platforms(specs)
    assert ok is (sys.platform in hosts), reason


@pytest.mark.parametrize("specs", [
    ("amiga",), ("not amiga",), ("linx",),
    # A matching spec never excuses a misspelt sibling: the typo would drop the
    # test on the host it was meant for while this host stays green.
    ("linux", "amiga"), ("darwin", "linux"), ("win32", "posix"),
])
def test_unknown_spec_is_a_collection_error_not_a_skip(specs):
    with pytest.raises(pytest.UsageError, match="unknown spec"):
        _host_matches_platforms(specs)


@pytest.mark.parametrize("negate", [False, True])
def test_native_arch_filter(negate):
    machine = _platform_machine()
    for arch, matches in ((machine, True), ("nonexistent-architecture", False)):
        ok, reason = _host_matches_platforms(("any",), arch=arch, arch_negate=negate)
        assert ok is (matches != negate)
        if not ok:
            assert machine in reason
    if machine == "arm64":
        assert _host_matches_platforms(("any",), arch="aarch64", arch_negate=negate)[0] is not negate


@pytest.mark.parametrize("raw,normalized", [
    ("AMD64", "x86_64"), ("x86", "x86_64"), ("aarch64", "arm64"),
    ("arm64", "arm64"), ("x86_64", "x86_64"),
])
def test_machine_alias_normalization(raw, normalized, monkeypatch):
    # Exercise normalization data, never alter sys.platform or interpreter OS behavior.
    monkeypatch.setattr("platform.machine", lambda: raw)
    assert _platform_machine() == normalized
