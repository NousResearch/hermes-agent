from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
import types
from pathlib import Path

import pytest

from scripts.canary import full_canary_owner_launcher as launcher


RELEASE_SHA = "a" * 40
SOURCE_ARCHIVE = b"release-bound-gateway-and-scripts"
REPOSITORY_ROOT = Path(launcher.__file__).parents[2]


class _PinnedPathStub:
    def __init__(self, path: str) -> None:
        self._path = path

    def absolute_path(self) -> str:
        return self._path


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _unseal(root: Path) -> None:
    for current, directories, files in os.walk(root):
        Path(current).chmod(0o700)
        for name in directories:
            (Path(current) / name).chmod(0o700)
        for name in files:
            path = Path(current) / name
            if not path.is_symlink():
                path.chmod(0o600)


def _reseal_without_following_symlinks(root: Path) -> None:
    for current, directories, files in os.walk(
        root,
        topdown=False,
        followlinks=False,
    ):
        current_path = Path(current)
        for name in files:
            path = current_path / name
            if not path.is_symlink():
                path.chmod(0o400)
        for name in directories:
            path = current_path / name
            if not path.is_symlink():
                path.chmod(0o500)
        current_path.chmod(0o500)


def _support_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(launcher, "_canonical_owner_home", lambda: str(tmp_path))
    root = Path(launcher._trusted_owner_support_paths(RELEASE_SHA)[0])
    source = root / launcher._TRUSTED_OWNER_SUPPORT_SOURCE_RELATIVE
    site = root / launcher._TRUSTED_OWNER_SUPPORT_SITE_RELATIVE

    for package in ("gateway", "scripts"):
        _write(source / package / "__init__.py", b"# release source\n")
    for package in ("cryptography", "yaml", "cffi", "pycparser"):
        _write(site / package / "__init__.py", b"# pinned wheel\n")
    _write(site / "_cffi_backend.so", b"pinned-native-extension")

    manifest = launcher._owner_support_manifest_value(
        RELEASE_SHA,
        source_archive_bytes=len(SOURCE_ARCHIVE),
        source_archive_sha256=hashlib.sha256(SOURCE_ARCHIVE).hexdigest(),
    )
    _write(root / "owner-support.json", _canonical(manifest) + b"\n")
    launcher._seal_owner_support_tree(str(root))
    return root


def _real_import_support_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    monkeypatch.setattr(launcher, "_canonical_owner_home", lambda: str(tmp_path))
    root = Path(launcher._trusted_owner_support_paths(RELEASE_SHA)[0])
    source = root / launcher._TRUSTED_OWNER_SUPPORT_SOURCE_RELATIVE
    site = root / launcher._TRUSTED_OWNER_SUPPORT_SITE_RELATIVE

    def ignored(_directory: str, names: list[str]) -> set[str]:
        return {
            name
            for name in names
            if name == "__pycache__" or name.endswith((".pyc", ".pyo", ".pth"))
        }

    for package in ("gateway", "scripts"):
        shutil.copytree(
            REPOSITORY_ROOT / package,
            source / package,
            ignore=ignored,
            copy_function=shutil.copyfile,
        )
    for package in ("cryptography", "yaml", "cffi", "pycparser"):
        spec = importlib.util.find_spec(package)
        assert spec is not None and spec.origin is not None
        shutil.copytree(
            Path(spec.origin).parent,
            site / package,
            ignore=ignored,
            copy_function=shutil.copyfile,
        )
    backend = importlib.util.find_spec("_cffi_backend")
    assert backend is not None and backend.origin is not None
    site.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(backend.origin, site / Path(backend.origin).name)

    manifest = launcher._owner_support_manifest_value(
        RELEASE_SHA,
        source_archive_bytes=len(SOURCE_ARCHIVE),
        source_archive_sha256=hashlib.sha256(SOURCE_ARCHIVE).hexdigest(),
    )
    _write(root / "owner-support.json", _canonical(manifest) + b"\n")
    launcher._seal_owner_support_tree(str(root))
    return root


def _capture(root: Path) -> tuple[int, int, str]:
    return launcher._capture_owner_support_publication_tree(
        str(root),
        release_sha=RELEASE_SHA,
    )


def _bootstrap_receipt_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[launcher.TrustedGcloudExecutable, Path, dict[str, object]]:
    support_root = _support_tree(tmp_path, monkeypatch)
    support_tree = _capture(support_root)
    support_manifest = launcher._validate_owner_support_manifest(
        str(support_root),
        release_sha=RELEASE_SHA,
    )
    runtime = object.__new__(launcher.TrustedGcloudExecutable)
    runtime._release_sha = RELEASE_SHA
    runtime._launcher_sha256 = "0" * 64
    runtime._sdk_root = "/trusted/google-cloud-sdk"
    runtime._sdk_fingerprint = (11, 12, "1" * 64)
    runtime._sdk_publication_fingerprint = (13, 14, "2" * 64)
    runtime._publication_intent = {
        "publication_release_sha": RELEASE_SHA,
        "intent_sha256": "3" * 64,
    }
    runtime._python = _PinnedPathStub("/trusted/python3.11")
    runtime._python_root = "/trusted/python-root"
    runtime._python_version = launcher._TRUSTED_PYTHON_VERSION
    runtime._python_fingerprint = (15, 16, "4" * 64)
    runtime._python_dependencies = launcher._TRUSTED_PYTHON_DEPENDENCIES
    runtime._owner_support_root = str(support_root)
    runtime._owner_support_source = str(support_root / "source")
    runtime._owner_support_site = str(support_root / "site-packages")
    runtime._owner_support_fingerprint = support_tree
    runtime._owner_support_manifest = support_manifest

    unsigned = {
        **runtime._expected_bootstrap_receipt_fields(),
        "created_at_unix": 1_000,
    }
    receipt: dict[str, object] = {
        **unsigned,
        "receipt_sha256": hashlib.sha256(_canonical(unsigned)).hexdigest(),
    }
    path = (
        tmp_path
        / ".hermes/trusted"
        / f"trusted-runtime-bootstrap-{RELEASE_SHA}.json"
    )
    path.write_bytes(_canonical(receipt) + b"\n")
    path.chmod(0o600)
    return runtime, path, receipt


def test_owner_support_constants_pin_exact_release_inputs() -> None:
    wheels = launcher._TRUSTED_OWNER_SUPPORT_WHEELS

    assert launcher.TRUSTED_OWNER_SUPPORT_TREE_SCHEMA == (
        "muncho-full-canary-owner-support-tree.v1"
    )
    assert [wheel[:2] for wheel in wheels] == [
        ("cryptography", "46.0.7"),
        ("cffi", "2.0.0"),
        ("PyYAML", "6.0.3"),
        ("pycparser", "3.0"),
    ]
    assert [(wheel[2], wheel[4], wheel[5]) for wheel in wheels] == [
        (
            "cryptography-46.0.7-cp311-abi3-macosx_10_9_universal2.whl",
            7_179_869,
            "ea42cbe97209df307fdc3b155f1b6fa2577c0defa8f1f7d3be7d31d189108ad4",
        ),
        (
            "cffi-2.0.0-cp311-cp311-macosx_11_0_arm64.whl",
            180_560,
            "2de9a304e27f7596cd03d16f1b7c72219bd944e99cc52b84d0145aefb07cbd3c",
        ),
        (
            "pyyaml-6.0.3-cp311-cp311-macosx_11_0_arm64.whl",
            175_577,
            "652cb6edd41e718550aad172851962662ff2681490a8a711af6a4d288dd96824",
        ),
        (
            "pycparser-3.0-py3-none-any.whl",
            48_172,
            "b727414169a36b7d524c1c3e31839a521725078d7b2ff038656844266160a992",
        ),
    ]
    assert len({wheel[2] for wheel in wheels}) == len(wheels)
    for _distribution, _version, filename, url, size, digest in wheels:
        assert url.startswith("https://files.pythonhosted.org/packages/")
        assert url.endswith(f"/{filename}")
        assert type(size) is int and size > 0
        assert re.fullmatch(r"[0-9a-f]{64}", digest)
    assert launcher._TRUSTED_OWNER_SUPPORT_MANAGED_MODULES == (
        "gateway",
        "scripts",
        "cryptography",
        "yaml",
        "cffi",
        "pycparser",
        "_cffi_backend",
    )


def test_v2_bootstrap_receipt_binds_owner_support_tree_and_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _path, receipt = _bootstrap_receipt_fixture(tmp_path, monkeypatch)

    assert launcher.TRUSTED_RUNTIME_BOOTSTRAP_RECEIPT_SCHEMA == (
        "muncho-full-canary-owner-trusted-runtime-bootstrap-receipt.v2"
    )
    assert receipt["owner_support_release_sha"] == RELEASE_SHA
    assert receipt["owner_support_tree_sha256"] == runtime._owner_support_fingerprint[2]
    assert receipt["owner_support_manifest_sha256"] == (
        runtime._owner_support_manifest["manifest_sha256"]
    )
    assert receipt["owner_support_source_archive_sha256"] == (
        runtime._owner_support_manifest["source_archive_sha256"]
    )
    assert receipt["owner_support_wheels"] == (
        launcher._owner_support_wheel_receipt_values()
    )
    unsigned = {
        name: value
        for name, value in receipt.items()
        if name != "receipt_sha256"
    }
    assert receipt["receipt_sha256"] == hashlib.sha256(
        _canonical(unsigned)
    ).hexdigest()
    assert runtime._validate_bootstrap_receipt()


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("owner_support_release_sha", "b" * 40),
        ("owner_support_tree_sha256", "f" * 64),
        ("owner_support_manifest_sha256", "e" * 64),
        ("owner_support_source_archive_sha256", "d" * 64),
        ("owner_support_wheels", []),
    ],
)
def test_v2_bootstrap_receipt_rejects_wrong_owner_support_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: object,
) -> None:
    runtime, path, receipt = _bootstrap_receipt_fixture(tmp_path, monkeypatch)
    receipt[field] = replacement
    unsigned = {
        name: value
        for name, value in receipt.items()
        if name != "receipt_sha256"
    }
    receipt["receipt_sha256"] = hashlib.sha256(_canonical(unsigned)).hexdigest()
    path.write_bytes(_canonical(receipt) + b"\n")
    path.chmod(0o600)

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="trusted_runtime_bootstrap_receipt_invalid",
    ):
        runtime._validate_bootstrap_receipt()


def test_v2_bootstrap_receipt_rejects_missing_support_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, path, receipt = _bootstrap_receipt_fixture(tmp_path, monkeypatch)
    del receipt["owner_support_tree_sha256"]
    unsigned = {
        name: value
        for name, value in receipt.items()
        if name != "receipt_sha256"
    }
    receipt["receipt_sha256"] = hashlib.sha256(_canonical(unsigned)).hexdigest()
    path.write_bytes(_canonical(receipt) + b"\n")
    path.chmod(0o600)

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="trusted_runtime_bootstrap_receipt_invalid",
    ):
        runtime._validate_bootstrap_receipt()


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_v2_bootstrap_receipt_rejects_linked_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    link_kind: str,
) -> None:
    runtime, path, _receipt = _bootstrap_receipt_fixture(tmp_path, monkeypatch)
    sibling = path.with_name("receipt-source.json")
    path.rename(sibling)
    if link_kind == "symlink":
        path.symlink_to(sibling)
    else:
        os.link(sibling, path)

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="trusted_runtime_bootstrap_receipt_invalid",
    ):
        runtime._validate_bootstrap_receipt()


def test_manifest_and_tree_digests_are_canonical_and_repeatable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _support_tree(tmp_path, monkeypatch)
    manifest_path = root / "owner-support.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    unsigned = {
        name: value
        for name, value in manifest.items()
        if name != "manifest_sha256"
    }

    assert manifest_path.read_bytes() == _canonical(manifest) + b"\n"
    assert manifest["manifest_sha256"] == hashlib.sha256(
        _canonical(unsigned)
    ).hexdigest()
    first = _capture(root)
    second = _capture(root)
    assert first == second
    assert first[0] > 0
    assert first[1] == sum(
        path.stat().st_size for path in root.rglob("*") if path.is_file()
    )
    assert re.fullmatch(r"[0-9a-f]{64}", first[2])
    assert launcher._validate_owner_support_manifest(
        str(root),
        release_sha=RELEASE_SHA,
    ) == manifest


def test_exact_isolated_activation_imports_all_lazy_owner_module_families(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _real_import_support_tree(tmp_path, monkeypatch)
    tree = _capture(root)
    manifest = launcher._validate_owner_support_manifest(
        str(root),
        release_sha=RELEASE_SHA,
    )
    source = root / "source"
    site = root / "site-packages"
    hostile_site = Path(sys.executable).parents[1] / "lib/python3.11/site-packages"
    probe = f"""
import json
import os
import runpy
import sys

namespace = runpy.run_path({str(launcher.__file__)!r})
namespace["activate_trusted_owner_support"].__globals__["_canonical_owner_home"] = (
    lambda: {str(tmp_path)!r}
)
Runtime = namespace["TrustedGcloudExecutable"]
runtime = object.__new__(Runtime)
runtime._production_runtime = True
runtime._release_sha = {RELEASE_SHA!r}
runtime._owner_support_root = {str(root)!r}
runtime._owner_support_source = {str(source)!r}
runtime._owner_support_site = {str(site)!r}
runtime._owner_support_fingerprint = {tree!r}
runtime._owner_support_manifest = {manifest!r}

activated = namespace["activate_trusted_owner_support"](
    runtime,
    release_sha={RELEASE_SHA!r},
)
required = namespace["_TRUSTED_OWNER_SUPPORT_PREFLIGHT_MODULES"]
modules = [sys.modules[name] for name in required]
if activated != ({str(source)!r}, {str(site)!r}):
    raise RuntimeError("wrong activation roots")
if sys.path[:2] != [{str(source)!r}, {str(site)!r}]:
    raise RuntimeError("wrong sys.path prefix")
for forbidden in ({str(REPOSITORY_ROOT)!r}, {str(hostile_site)!r}):
    for entry in sys.path:
        try:
            if os.path.commonpath((forbidden, entry)) == forbidden:
                raise RuntimeError("ambient path admitted")
        except ValueError:
            pass
for module in modules:
    origin = module.__spec__.origin
    expected = {str(source)!r}
    if os.path.commonpath((expected, origin)) != expected:
        raise RuntimeError("owner module escaped sealed source")
namespace["require_trusted_owner_support_activation"](
    runtime,
    release_sha={RELEASE_SHA!r},
)
print(json.dumps({{
    "isolated": sys.flags.isolated,
    "no_site": sys.flags.no_site,
    "dont_write_bytecode": sys.flags.dont_write_bytecode,
    "safe_path": sys.flags.safe_path,
    "modules": list(required),
}}, sort_keys=True, separators=(",", ":")))
"""
    completed = subprocess.run(
        (
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-X",
            "pycache_prefix=/var/empty/muncho-canary",
            "-c",
            probe,
        ),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={
            "HOME": str(tmp_path),
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "PYTHONPATH": os.pathsep.join(
                (str(REPOSITORY_ROOT), str(hostile_site))
            ),
        },
        check=False,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr.decode(
        "utf-8", errors="replace"
    )
    result = json.loads(completed.stdout)
    assert result == {
        "dont_write_bytecode": 1,
        "isolated": 1,
        "modules": list(launcher._TRUSTED_OWNER_SUPPORT_PREFLIGHT_MODULES),
        "no_site": 1,
        "safe_path": True,
    }
    assert not list(root.rglob("*.pyc"))
    assert not list(root.rglob("__pycache__"))


def _synthetic_publication_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[tuple[object, ...], bytes, bytes, str]:
    wheel_buffer = io.BytesIO()
    with zipfile.ZipFile(wheel_buffer, mode="w") as archive:
        for package in ("cryptography", "yaml", "cffi", "pycparser"):
            archive.writestr(f"{package}/__init__.py", b"# pinned wheel\n")
        archive.writestr("_cffi_backend.so", b"pinned-native-extension")
    wheel_payload = wheel_buffer.getvalue()
    wheel = (
        "owner-support-test",
        "1.0",
        "owner_support_test-1.0-py3-none-any.whl",
        (
            "https://files.pythonhosted.org/packages/test/"
            "owner_support_test-1.0-py3-none-any.whl"
        ),
        len(wheel_payload),
        hashlib.sha256(wheel_payload).hexdigest(),
    )
    monkeypatch.setattr(launcher, "_TRUSTED_OWNER_SUPPORT_WHEELS", (wheel,))

    source_buffer = io.BytesIO()
    with tarfile.open(fileobj=source_buffer, mode="w") as archive:
        for name in ("gateway/__init__.py", "scripts/__init__.py"):
            payload = b"# release source\n"
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    source_payload = source_buffer.getvalue()
    source_sha256 = hashlib.sha256(source_payload).hexdigest()
    return wheel, wheel_payload, source_payload, source_sha256


def test_unreceipted_existing_destination_is_rebuilt_and_compared(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheel, wheel_payload, source_payload, source_sha256 = (
        _synthetic_publication_inputs(monkeypatch)
    )

    root = _support_tree(tmp_path, monkeypatch)
    _unseal(root)
    (root / "source/gateway/__init__.py").write_bytes(b"# preseeded tamper\n")
    manifest = launcher._owner_support_manifest_value(
        RELEASE_SHA,
        source_archive_bytes=len(source_payload),
        source_archive_sha256=source_sha256,
    )
    (root / "owner-support.json").write_bytes(_canonical(manifest) + b"\n")
    launcher._seal_owner_support_tree(str(root))

    hermes_root = tmp_path / ".hermes"
    trusted_root = hermes_root / "trusted"
    hermes_root.chmod(0o700)
    trusted_root.chmod(0o700)
    calls: list[str] = []

    def source_archiver(release_sha: str, destination: str) -> tuple[int, str]:
        assert release_sha == RELEASE_SHA
        calls.append("source")
        archive_path = Path(destination)
        archive_path.write_bytes(source_payload)
        archive_path.chmod(0o600)
        return len(source_payload), source_sha256

    def wheel_downloader(candidate: object, destination: str) -> None:
        assert candidate == wheel
        calls.append("wheel")
        wheel_path = Path(destination)
        wheel_path.write_bytes(wheel_payload)
        wheel_path.chmod(0o600)

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="trusted_owner_support_destination_mismatch",
    ):
        launcher._publish_trusted_owner_support_runtime(
            RELEASE_SHA,
            trusted_root=str(trusted_root),
            source_archiver=source_archiver,
            wheel_downloader=wheel_downloader,
        )

    assert calls == ["source", "wheel"]


def test_fresh_darwin_publication_reseals_the_destination_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheel, wheel_payload, source_payload, source_sha256 = (
        _synthetic_publication_inputs(monkeypatch)
    )
    monkeypatch.setattr(launcher, "_canonical_owner_home", lambda: str(tmp_path))
    hermes_root = tmp_path / ".hermes"
    trusted_root = hermes_root / "trusted"
    trusted_root.mkdir(parents=True)
    hermes_root.chmod(0o700)
    trusted_root.chmod(0o700)
    calls: list[str] = []

    def source_archiver(release_sha: str, destination: str) -> tuple[int, str]:
        assert release_sha == RELEASE_SHA
        calls.append("source")
        archive_path = Path(destination)
        archive_path.write_bytes(source_payload)
        archive_path.chmod(0o600)
        return len(source_payload), source_sha256

    def wheel_downloader(candidate: object, destination: str) -> None:
        assert candidate == wheel
        calls.append("wheel")
        wheel_path = Path(destination)
        wheel_path.write_bytes(wheel_payload)
        wheel_path.chmod(0o600)

    root, tree, manifest = launcher._publish_trusted_owner_support_runtime(
        RELEASE_SHA,
        trusted_root=str(trusted_root),
        source_archiver=source_archiver,
        wheel_downloader=wheel_downloader,
    )

    assert calls == ["source", "wheel"]
    assert root == launcher._trusted_owner_support_paths(RELEASE_SHA)[0]
    assert os.stat(root, follow_symlinks=False).st_mode & 0o777 == 0o500
    assert tree == launcher._capture_owner_support_publication_tree(
        root,
        release_sha=RELEASE_SHA,
    )
    assert manifest == launcher._validate_owner_support_manifest(
        root,
        release_sha=RELEASE_SHA,
    )


def test_retry_recovers_exact_interrupted_post_rename_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheel, wheel_payload, source_payload, source_sha256 = (
        _synthetic_publication_inputs(monkeypatch)
    )
    monkeypatch.setattr(launcher, "_canonical_owner_home", lambda: str(tmp_path))
    hermes_root = tmp_path / ".hermes"
    trusted_root = hermes_root / "trusted"
    trusted_root.mkdir(parents=True)
    hermes_root.chmod(0o700)
    trusted_root.chmod(0o700)
    calls: list[str] = []

    def source_archiver(release_sha: str, destination: str) -> tuple[int, str]:
        assert release_sha == RELEASE_SHA
        calls.append("source")
        archive_path = Path(destination)
        archive_path.write_bytes(source_payload)
        archive_path.chmod(0o600)
        return len(source_payload), source_sha256

    def wheel_downloader(candidate: object, destination: str) -> None:
        assert candidate == wheel
        calls.append("wheel")
        wheel_path = Path(destination)
        wheel_path.write_bytes(wheel_payload)
        wheel_path.chmod(0o600)

    atomic_rename = launcher._darwin_rename_no_replace

    def interrupt_after_rename(*args: object, **kwargs: object) -> None:
        atomic_rename(*args, **kwargs)
        raise launcher.OwnerLauncherError("simulated_post_rename_interruption")

    monkeypatch.setattr(
        launcher,
        "_darwin_rename_no_replace",
        interrupt_after_rename,
    )
    with pytest.raises(launcher.OwnerLauncherError) as interrupted:
        launcher._publish_trusted_owner_support_runtime(
            RELEASE_SHA,
            trusted_root=str(trusted_root),
            source_archiver=source_archiver,
            wheel_downloader=wheel_downloader,
        )
    assert interrupted.value.code == "simulated_post_rename_interruption"

    destination = Path(launcher._trusted_owner_support_paths(RELEASE_SHA)[0])
    assert destination.stat(follow_symlinks=False).st_mode & 0o777 == 0o700

    monkeypatch.setattr(
        launcher,
        "_darwin_rename_no_replace",
        atomic_rename,
    )
    root, tree, manifest = launcher._publish_trusted_owner_support_runtime(
        RELEASE_SHA,
        trusted_root=str(trusted_root),
        source_archiver=source_archiver,
        wheel_downloader=wheel_downloader,
    )

    assert calls == ["source", "wheel", "source", "wheel"]
    assert root == str(destination)
    assert destination.stat(follow_symlinks=False).st_mode & 0o777 == 0o500
    assert tree == launcher._capture_owner_support_publication_tree(
        root,
        release_sha=RELEASE_SHA,
    )
    assert manifest == launcher._validate_owner_support_manifest(
        root,
        release_sha=RELEASE_SHA,
    )


def test_activation_rejects_ambient_managed_module_before_sys_path_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _support_tree(tmp_path, monkeypatch)

    class Runtime:
        @staticmethod
        def trusted_owner_support_paths() -> tuple[str, str]:
            return str(root / "source"), str(root / "site-packages")

    ambient = types.ModuleType("gateway.owner_support_ambient_probe")
    ambient.__file__ = str(launcher.__file__)
    ambient.__spec__ = types.SimpleNamespace(origin=str(launcher.__file__))
    monkeypatch.setitem(sys.modules, ambient.__name__, ambient)
    module_root = launcher._trusted_owner_support_module_root
    monkeypatch.setattr(
        launcher,
        "_trusted_owner_support_module_root",
        lambda name, **roots: (
            module_root(name, **roots) if name == ambient.__name__ else None
        ),
    )
    before = list(sys.path)

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="trusted_owner_support_origin_invalid",
    ):
        launcher.activate_trusted_owner_support(Runtime(), release_sha=RELEASE_SHA)
    assert sys.path == before


def test_module_origin_guard_rejects_ambient_package_search_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _support_tree(tmp_path, monkeypatch)
    source = str(root / "source")
    site = str(root / "site-packages")
    package = types.ModuleType("scripts.owner_support_ambient_package")
    package.__file__ = str(root / "source/scripts/__init__.py")
    package.__spec__ = types.SimpleNamespace(origin=package.__file__)
    package.__path__ = [str(REPOSITORY_ROOT / "scripts")]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    module_root = launcher._trusted_owner_support_module_root
    monkeypatch.setattr(
        launcher,
        "_trusted_owner_support_module_root",
        lambda name, **roots: (
            module_root(name, **roots) if name == package.__name__ else None
        ),
    )

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="trusted_owner_support_origin_invalid",
    ):
        launcher._validate_trusted_owner_support_module_origins(
            source_root=source,
            site_root=site,
        )


def test_tree_is_bound_to_release_and_canonical_owner_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _support_tree(tmp_path, monkeypatch)

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="trusted_owner_support_path_invalid",
    ):
        launcher._capture_owner_support_publication_tree(
            str(root),
            release_sha="b" * 40,
        )

    moved = tmp_path / "wrong-owner-support-root"
    _unseal(root)
    shutil.move(root, moved)
    launcher._seal_owner_support_tree(str(moved))
    with pytest.raises(
        launcher.OwnerLauncherError,
        match="trusted_owner_support_path_invalid",
    ):
        _capture(moved)


@pytest.mark.parametrize(
    "relative",
    [
        "owner-support.json",
        "source/gateway/__init__.py",
        "source/scripts/__init__.py",
        "site-packages/cryptography/__init__.py",
        "site-packages/yaml/__init__.py",
        "site-packages/cffi/__init__.py",
        "site-packages/pycparser/__init__.py",
    ],
)
def test_tree_rejects_missing_required_object(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative: str,
) -> None:
    root = _support_tree(tmp_path, monkeypatch)
    _unseal(root)
    (root / relative).unlink()
    launcher._seal_owner_support_tree(str(root))

    with pytest.raises(launcher.OwnerLauncherError):
        _capture(root)


def test_tree_rejects_symlink_instead_of_release_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _support_tree(tmp_path, monkeypatch)
    target = root / "source/gateway/__init__.py"
    _unseal(root)
    target.unlink()
    target.symlink_to(root / "source/scripts/__init__.py")
    _reseal_without_following_symlinks(root)

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="trusted_owner_support_invalid",
    ):
        _capture(root)


def test_tree_rejects_hardlinked_release_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _support_tree(tmp_path, monkeypatch)
    original = root / "source/gateway/__init__.py"
    alias = root / "source/gateway/alias.py"
    _unseal(root)
    os.link(original, alias)
    launcher._seal_owner_support_tree(str(root))

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="trusted_owner_support_invalid",
    ):
        _capture(root)


@pytest.mark.parametrize(
    "relative",
    [
        "site-packages/ambient.pth",
        "site-packages/injected.pyc",
        "site-packages/injected.pyo",
        "source/gateway/__pycache__/injected.pyc",
    ],
)
def test_tree_rejects_dynamic_site_paths_and_bytecode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative: str,
) -> None:
    root = _support_tree(tmp_path, monkeypatch)
    _unseal(root)
    _write(root / relative, b"forbidden")
    launcher._seal_owner_support_tree(str(root))

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="trusted_owner_support_invalid",
    ):
        _capture(root)


def test_tree_detects_file_toctou_between_directory_walk_and_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _support_tree(tmp_path, monkeypatch)
    target = root / "owner-support.json"
    original_read = launcher._read_pinned_regular_file
    changed = False

    def mutate_then_read(path: str, **kwargs: object):
        nonlocal changed
        if path == str(target) and not changed:
            changed = True
            target.chmod(0o600)
            target.write_bytes(target.read_bytes() + b" ")
            target.chmod(0o400)
        return original_read(path, **kwargs)

    monkeypatch.setattr(launcher, "_read_pinned_regular_file", mutate_then_read)
    with pytest.raises(
        launcher.OwnerLauncherError,
        match="trusted_owner_support_changed",
    ):
        _capture(root)
    assert changed is True


def test_manifest_rejects_wrong_digest_and_noncanonical_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _support_tree(tmp_path, monkeypatch)
    path = root / "owner-support.json"
    _unseal(root)
    value = json.loads(path.read_text(encoding="ascii"))
    value["manifest_sha256"] = "f" * 64
    path.write_bytes(json.dumps(value).encode("ascii") + b"\n")
    launcher._seal_owner_support_tree(str(root))

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="trusted_owner_support_manifest_invalid",
    ):
        launcher._validate_owner_support_manifest(
            str(root),
            release_sha=RELEASE_SHA,
        )
