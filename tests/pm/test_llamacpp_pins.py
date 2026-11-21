"""Every supported local engine backend installs only reviewed PM artifacts."""

import pytest

import pm
from pm import paths
from pm.lock import Lockfile
from pm.store import ALL_TARGETS


@pytest.mark.parametrize(("version", "toolkit"), [("11671", "13.4"), ("12167", "13.4"), ("12170", "13.5")])
def test_cuda_archives_name_the_toolkit_upstream_built_that_tag_with(version, toolkit):
    """Upstream moved its CUDA builds from 13.4 to 13.5 at b12170. An engine installed before the
    package manager is adopted by its own tag's archive names, so an older tag keeps 13.4."""
    package = pm.get_package("llamacpp-cuda")
    for target in package.assets:
        assert all(f"-cuda-{toolkit}-" in name for name in package._asset_names(version, target))


def test_llamacpp_backends_have_pins_for_their_supported_targets():
    lock = Lockfile(paths.lockfile_path())
    backends = ("cpu", "cuda", "vulkan", "metal", "hip")
    # One engine build everywhere: the catalog and presets are calibrated against a single tag.
    assert len({lock.version(f"llamacpp-{backend}") for backend in backends}) == 1
    for backend in backends:
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
                assert any("cudart-" in a["url"] for a in artifacts)
            # Upstream's Linux builds link the system OpenMP runtime, which minimal hosts lack.
            if target in ("linux-x64", "linux-arm64"):
                assert any("/libgomp1_" in a["url"] and a["url"].endswith(".deb") for a in artifacts)
