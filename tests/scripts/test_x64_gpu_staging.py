"""The floor builder consumes only the two native entries proved in the GPU userland."""
import pytest

from pm.lock import Facts
from pm.package import InstallError
from pm.registry import get_package
from pm.store import tree_digest
from scripts.bundles import native


def test_floor_builder_requires_both_verified_gpu_entries(tmp_path, monkeypatch):
    target = 'linux-x64'
    lock = native._lockfile()
    names = ['llamacpp-cpu', 'llamacpp-vulkan', 'llamacpp-cuda', 'llamacpp-hip']
    store = tmp_path / 'tools'
    for name in names:
        package = get_package(name)
        monkeypatch.setattr(package, 'native_abi_reason', lambda t, p=package: p.host_abi_reason(t, (2, 35)))
    for name in names[2:]:
        package = get_package(name)
        version = lock.version(name)
        assert version is not None
        entry = store / package.store_entry(version, target)
        entry.mkdir(parents=True)
        binary = package.binary(entry, target)
        assert binary is not None
        binary.write_bytes(b'inert pinned fixture')
        Facts(store / 'facts.json').record(name, version, entry.name, package.env(entry, target), store,
                                           target=target, artifacts=[a['sha256'] for a in lock.artifacts(name, target)],
                                           digest=tree_digest(entry))
    assert native._tools_for_builder(names, store, target) == names[:2]
    row = Facts(store / 'facts.json').get('llamacpp-hip')
    assert row is not None
    (store / row['entry'] / 'llama-server').write_bytes(b'changed after the GPU userland probe')
    with pytest.raises(InstallError, match='llamacpp-hip'):
        native._tools_for_builder(names, store, target)
    with pytest.raises(InstallError, match='facts missing'):
        native._tools_for_builder(names, tmp_path / 'empty', target)


def test_compatible_builder_keeps_native_verification_for_every_tool(tmp_path, monkeypatch):
    names = ['llamacpp-cpu', 'llamacpp-vulkan', 'llamacpp-cuda', 'llamacpp-hip']
    for name in names:
        monkeypatch.setattr(get_package(name), 'native_abi_reason', lambda target: None)
    assert native._tools_for_builder(names, tmp_path, 'linux-x64') == names
