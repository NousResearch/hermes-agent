"""Every supported local engine backend installs only reviewed PM artifacts."""

import pm
from pm import paths
from pm.lock import Lockfile
from pm.store import ALL_TARGETS


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
                assert len(artifacts) == 2
                assert any("cudart-" in a["url"] for a in artifacts)


def test_llamacpp_cuda_infix_tracks_the_release_not_a_hardcode(monkeypatch):
    """Upstream renames the CUDA line between tags (b10964 built 13.3-x64;
    b11370 moved to 13.4-x64). The infix must come from the release's asset
    list, not a hardcoded string, or the pin step 404s the moment it moves."""
    from pm import packages

    # Offline: the release index is unavailable, so the static default holds.
    monkeypatch.setattr(packages, "_github_release_digests", lambda *a, **k: {})
    monkeypatch.setattr(packages, "_cuda_infix_cache", {})
    cuda = pm.get_package("llamacpp-cuda")
    assert cuda._asset_names("11370", "win32-x64")[0].endswith("win-cuda-13.4-x64.zip")

    # With the release's assets visible, the advertised infix wins even when
    # it differs from the static default.
    monkeypatch.setattr(
        packages,
        "_github_release_digests",
        lambda *a, **k: {"llama-b99999-bin-win-cuda-99.9-x64.zip": "0" * 64},
    )
    monkeypatch.setattr(packages, "_cuda_infix_cache", {})
    assert cuda._asset_names("99999", "win32-x64") == [
        "llama-b99999-bin-win-cuda-99.9-x64.zip",
        "cudart-llama-bin-win-cuda-99.9-x64.zip",
    ]
