"""The pinned cua-driver must not predate the macOS Retina backing-scale fix."""

import pm
from pm import paths
from pm.lock import Lockfile
from pm.store import ALL_TARGETS

# trycua/cua#3326: on a scaled macOS panel CGDisplayPixelsWide returns the
# logical width, so drivers before this floor report scale_factor 1.0 for a
# 2x display. coordinate= actions then click at half coordinates while
# element= actions stay correct, which makes the failure silent. The fix
# (trycua/cua#3328) first shipped in cua-driver-rs-v0.22.2.
RETINA_BACKING_SCALE_FLOOR = (0, 22, 2)


def _parse(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split(".")[:3])


def test_cua_driver_pin_carries_the_retina_backing_scale_fix():
    lock = Lockfile(paths.lockfile_path())
    pinned = lock.version("cua-driver")
    assert pinned
    assert _parse(pinned) >= RETINA_BACKING_SCALE_FLOOR


def test_cua_driver_pins_match_advertised_artifacts():
    lock = Lockfile(paths.lockfile_path())
    package = pm.get_package("cua-driver")
    version = lock.version("cua-driver")
    assert version
    for target in ALL_TARGETS:
        artifacts = lock.artifacts("cua-driver", target)
        if package.missing_reason(target):
            assert not artifacts
            continue
        assert artifacts
        assert [a["url"] for a in artifacts] == package.fetch_urls(version, target)
        assert all(len(bytes.fromhex(a["sha256"])) == 32 for a in artifacts)
