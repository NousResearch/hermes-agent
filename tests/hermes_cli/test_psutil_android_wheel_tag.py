from __future__ import annotations

import io
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from hermes_cli.psutil_android import (
    MARKER,
    PsutilAndroidInstallError,
    android_wheel_platform_tag,
    prepare_patched_psutil_sdist,
)


def _add_dir(tf: tarfile.TarFile, name: str) -> None:
    info = tarfile.TarInfo(name)
    info.type = tarfile.DIRTYPE
    info.mode = 0o755
    tf.addfile(info)


def _add_file(tf: tarfile.TarFile, name: str, content: str) -> None:
    payload = content.encode()
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mode = 0o644
    tf.addfile(info, io.BytesIO(payload))


def _archive(
    path: Path,
    *,
    setup_cfg: str | None = None,
    setup_py: str | None = None,
) -> None:
    with tarfile.open(path, "w:gz") as tf:
        _add_dir(tf, "psutil-7.2.2")
        _add_dir(tf, "psutil-7.2.2/psutil")
        _add_file(tf, "psutil-7.2.2/psutil/_common.py", f"{MARKER}\n")
        if setup_cfg is not None:
            _add_file(tf, "psutil-7.2.2/setup.cfg", setup_cfg)
        if setup_py is not None:
            _add_file(tf, "psutil-7.2.2/setup.py", setup_py)


@pytest.mark.parametrize(
    ("machine", "expected_abi"),
    [
        ("aarch64", "arm64_v8a"),
        ("arm64", "arm64_v8a"),
        ("armv8l", "armeabi_v7a"),
        ("x86_64", "x86_64"),
        ("i686", "x86"),
    ],
)
def test_android_wheel_platform_tag_maps_termux_architectures(machine, expected_abi):
    assert android_wheel_platform_tag(machine=machine) == f"android_24_{expected_abi}"


def test_android_wheel_platform_ignores_runtime_android_api(monkeypatch):
    monkeypatch.setenv("ANDROID_API_LEVEL", "36")

    assert android_wheel_platform_tag(machine="aarch64") == "android_24_arm64_v8a"


def test_android_wheel_platform_supports_explicit_build_target():
    assert (
        android_wheel_platform_tag(machine="aarch64", api_level=28)
        == "android_28_arm64_v8a"
    )


def test_android_wheel_platform_rejects_unknown_architecture():
    with pytest.raises(
        PsutilAndroidInstallError, match="Unsupported Android architecture"
    ):
        android_wheel_platform_tag(machine="mips64")


def test_prepare_patched_sdist_adds_android_bdist_wheel_config(tmp_path):
    archive = tmp_path / "psutil.tar.gz"
    _archive(archive)

    src = prepare_patched_psutil_sdist(
        archive,
        tmp_path / "extract",
        platform_tag="android_24_arm64_v8a",
    )

    assert (src / "setup.cfg").read_text() == (
        "[bdist_wheel]\nplat_name = android_24_arm64_v8a\n"
    )


def test_prepare_patched_sdist_preserves_existing_setup_config(tmp_path):
    archive = tmp_path / "psutil.tar.gz"
    _archive(
        archive,
        setup_cfg="[metadata]\nlicense_files = LICENSE\n\n[bdist_wheel]\nuniversal = 0\n",
    )

    src = prepare_patched_psutil_sdist(
        archive,
        tmp_path / "extract",
        platform_tag="android_24_arm64_v8a",
    )
    config = (src / "setup.cfg").read_text()

    assert "[metadata]\nlicense_files = LICENSE" in config
    assert "[bdist_wheel]\nplat_name = android_24_arm64_v8a\nuniversal = 0" in config


def test_patched_sdist_builds_an_android_tagged_wheel(tmp_path):
    archive = tmp_path / "psutil.tar.gz"
    _archive(
        archive,
        setup_py=(
            "from setuptools import setup\n"
            "setup(name='psutil', version='7.2.2', packages=['psutil'])\n"
        ),
    )
    src = prepare_patched_psutil_sdist(
        archive,
        tmp_path / "extract",
        platform_tag="android_24_arm64_v8a",
    )

    subprocess.run(
        [sys.executable, "setup.py", "bdist_wheel"],
        cwd=src,
        check=True,
        capture_output=True,
        text=True,
    )

    wheel_names = [wheel.name for wheel in (src / "dist").glob("*.whl")]
    assert wheel_names == ["psutil-7.2.2-py3-none-android_24_arm64_v8a.whl"]
