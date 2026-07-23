from __future__ import annotations

import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from scripts import install_android_wheels
from hermes_cli.android_wheels import (
    AndroidWheelError,
    android_wheel_platform_tag,
    configure_setuptools_android_tag,
    safe_extract_sdist,
)


@pytest.mark.parametrize(
    ("machine", "expected"),
    [
        ("aarch64", "android_24_arm64_v8a"),
        ("armv8l", "android_24_armeabi_v7a"),
        ("x86_64", "android_24_x86_64"),
        ("i686", "android_24_x86"),
    ],
)
def test_android_platform_mappings(machine: str, expected: str) -> None:
    assert android_wheel_platform_tag(machine=machine) == expected


def test_explicit_api_override() -> None:
    assert (
        android_wheel_platform_tag(machine="aarch64", api_level="26")
        == "android_26_arm64_v8a"
    )


def test_runtime_android_api_variable_is_not_used(monkeypatch) -> None:
    monkeypatch.setenv("ANDROID_API_LEVEL", "36")
    assert android_wheel_platform_tag(machine="aarch64") == "android_24_arm64_v8a"


def test_setup_cfg_preserved(tmp_path: Path) -> None:
    root = tmp_path / "pkg"
    root.mkdir()
    cfg = root / "setup.cfg"
    cfg.write_text(
        "[metadata]\nname = demo\n\n[bdist_wheel]\nuniversal = 0\n", encoding="utf-8"
    )
    configure_setuptools_android_tag(root, "android_24_arm64_v8a")
    content = cfg.read_text(encoding="utf-8")
    assert "[metadata]\nname = demo" in content
    assert "universal = 0" in content
    assert "plat_name = android_24_arm64_v8a" in content


def test_configured_bdist_wheel_filename(tmp_path: Path) -> None:
    root = tmp_path / "demo"
    root.mkdir()
    (root / "setup.py").write_text(
        "from setuptools import Extension, setup\n"
        "setup(name='android-tag-demo', version='1.0', py_modules=['demo'], "
        "ext_modules=[Extension('_demo', sources=['demo.c'])])\n",
        encoding="utf-8",
    )
    (root / "demo.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "demo.c").write_text(
        "#include <Python.h>\n"
        'static struct PyModuleDef m={PyModuleDef_HEAD_INIT,"_demo",0,-1,0};\n'
        "PyMODINIT_FUNC PyInit__demo(void){return PyModule_Create(&m);}\n",
        encoding="utf-8",
    )
    configure_setuptools_android_tag(root, "android_24_arm64_v8a")
    result = subprocess.run(
        [sys.executable, "setup.py", "bdist_wheel", "--dist-dir", str(root / "dist")],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 and "invalid command 'bdist_wheel'" in result.stderr:
        pytest.skip("wheel is not installed in this test interpreter")
    assert result.returncode == 0, result.stderr
    wheels = list((root / "dist").glob("*.whl"))
    assert len(wheels) == 1
    assert "android_24_arm64_v8a" in wheels[0].name


def test_safe_extract_rejects_symlink(tmp_path: Path) -> None:
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        directory = tarfile.TarInfo("demo-1.0")
        directory.type = tarfile.DIRTYPE
        tf.addfile(directory)
        link = tarfile.TarInfo("demo-1.0/link")
        link.type = tarfile.SYMTYPE
        link.linkname = "../../outside"
        tf.addfile(link)
    with pytest.raises(AndroidWheelError, match="Unsupported archive member type"):
        safe_extract_sdist(archive, tmp_path / "out")


def test_install_script_forwards_explicit_android_api_level(
    tmp_path: Path, monkeypatch
) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("", encoding="utf-8")
    captured: dict[str, int] = {}

    def fake_platform_tag(*, api_level: int) -> str:
        captured["api_level"] = api_level
        return "android_28_arm64_v8a"

    monkeypatch.setattr(
        install_android_wheels, "android_wheel_platform_tag", fake_platform_tag
    )
    result = install_android_wheels.main([
        "--uv",
        "uv",
        "--python",
        "python",
        "--requirements",
        str(requirements),
        "--android-api-level",
        "28",
        "--package",
        "not-present",
        "--work-dir",
        str(tmp_path / "work"),
    ])

    assert result == 0
    assert captured == {"api_level": 28}
