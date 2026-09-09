"""Every supported local engine backend installs only reviewed PM artifacts."""

import pm
from pm import paths
from pm.lock import Lockfile
from pm.store import ALL_TARGETS


def test_llamacpp_backends_have_pins_for_their_supported_targets():
    lock = Lockfile(paths.lockfile_path())
    for backend in ("cpu", "cuda", "vulkan", "metal", "hip"):
        name = f"llamacpp-{backend}"
        package = pm.get_package(name)
        version = lock.version(name)
        assert version
        for target in ALL_TARGETS:
            artifacts = lock.artifacts(name, target)
            if package.missing_reason(target):
                assert not artifacts
                continue
            assert artifacts
            assert [a["url"] for a in artifacts] == package.fetch_urls(version, target)
            assert all(len(bytes.fromhex(a["sha256"])) == 32 for a in artifacts)
            if backend == "cuda":
                assert len(artifacts) == 2
                assert any("cudart-" in a["url"] for a in artifacts)
