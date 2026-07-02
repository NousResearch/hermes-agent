"""Test .dist-info style distributions."""

import pathlib
import re
import shutil
import subprocess
import sys
from functools import partial

import pytest

from setuptools.archive_util import unpack_archive

from .textwrap import DALS

read = partial(pathlib.Path.read_text, encoding="utf-8")


class TestDistInfo:
    def test_invalid_version(self, tmp_path):
        """
        Supplying an invalid version crashes dist_info.
        """
        config = "[metadata]\nname=proj\nversion=42\n[egg_info]\ntag_build=invalid!!!\n"
        (tmp_path / "setup.cfg").write_text(config, encoding="utf-8")
        msg = re.compile("invalid version", re.MULTILINE | re.IGNORECASE)
        proc = run_command_inner("dist_info", cwd=tmp_path, check=False)
        assert proc.returncode
        assert msg.search(proc.stdout)
        assert not list(tmp_path.glob("*.dist-info"))

    def test_tag_arguments(self, tmp_path):
        config = """
        [metadata]
        name=proj
        version=42
        [egg_info]
        tag_date=1
        tag_build=.post
        """
        (tmp_path / "setup.cfg").write_text(config, encoding="utf-8")

        print(run_command("dist_info", "--no-date", cwd=tmp_path))
        dist_info = next(tmp_path.glob("*.dist-info"))
        assert dist_info.name.startswith("proj-42")
        shutil.rmtree(dist_info)

        print(run_command("dist_info", "--tag-build", ".a", cwd=tmp_path))
        dist_info = next(tmp_path.glob("*.dist-info"))
        assert dist_info.name.startswith("proj-42a")

    @pytest.mark.parametrize("keep_egg_info", (False, True))
    def test_output_dir(self, tmp_path, keep_egg_info):
        config = "[metadata]\nname=proj\nversion=42\n"
        (tmp_path / "setup.cfg").write_text(config, encoding="utf-8")
        out = tmp_path / "__out"
        out.mkdir()
        opts = ["--keep-egg-info"] if keep_egg_info else []
        run_command("dist_info", "--output-dir", out, *opts, cwd=tmp_path)
        assert len(list(out.glob("*.dist-info"))) == 1
        assert len(list(tmp_path.glob("*.dist-info"))) == 0
        expected_egg_info = int(keep_egg_info)
        assert len(list(out.glob("*.egg-info"))) == expected_egg_info
        assert len(list(tmp_path.glob("*.egg-info"))) == 0
        assert len(list(out.glob("*.__bkp__"))) == 0
        assert len(list(tmp_path.glob("*.__bkp__"))) == 0


class TestWheelCompatibility:
    """Make sure the .dist-info directory produced with the ``dist_info`` command
    is the same as the one produced by ``bdist_wheel``.
    """

    SETUPCFG = DALS(
        """
    [metadata]
    name = {name}
    version = {version}

    [options]
    install_requires =
        foo>=12; sys_platform != "linux"

    [options.extras_require]
    test = pytest

    [options.entry_points]
    console_scripts =
        executable-name = my_package.module:function
    discover =
        myproj = my_package.other_module:function
    """
    )

    EGG_INFO_OPTS = [
        # Related: #3088 #2872
        ("", ""),
        (".post", "[egg_info]\ntag_build = post\n"),
        (".post", "[egg_info]\ntag_build = .post\n"),
        (".post", "[egg_info]\ntag_build = post\ntag_date = 1\n"),
        (".dev", "[egg_info]\ntag_build = .dev\n"),
        (".dev", "[egg_info]\ntag_build = .dev\ntag_date = 1\n"),
        ("a1", "[egg_info]\ntag_build = .a1\n"),
        ("+local", "[egg_info]\ntag_build = +local\n"),
    ]

    @pytest.mark.parametrize("name", "my-proj my_proj my.proj My.Proj".split())
    @pytest.mark.parametrize("version", ["0.42.13"])
    @pytest.mark.parametrize(("suffix", "cfg"), EGG_INFO_OPTS)
    def test_dist_info_is_the_same_as_in_wheel(
        self, name, version, tmp_path, suffix, cfg
    ):
        config = self.SETUPCFG.format(name=name, version=version) + cfg

        for i in "dir_wheel", "dir_dist":
            (tmp_path / i).mkdir()
            (tmp_path / i / "setup.cfg").write_text(config, encoding="utf-8")

        run_command("bdist_wheel", cwd=tmp_path / "dir_wheel")
        wheel = next(tmp_path.glob("dir_wheel/dist/*.whl"))
        unpack_archive(wheel, tmp_path / "unpack")
        wheel_dist_info = next(tmp_path.glob("unpack/*.dist-info"))

        run_command("dist_info", cwd=tmp_path / "dir_dist")
        dist_info = next(tmp_path.glob("dir_dist/*.dist-info"))

        assert dist_info.name == wheel_dist_info.name
        assert dist_info.name.startswith(f"my_proj-{version}{suffix}")
        for file in "METADATA", "entry_points.txt":
            assert read(dist_info / file) == read(wheel_dist_info / file)


def run_command_inner(*cmd, **kwargs):
    opts = {
        "stderr": subprocess.STDOUT,
        "stdout": subprocess.PIPE,
        "text": True,
        "encoding": "utf-8",
        "check": True,
        **kwargs,
    }
    cmd = [sys.executable, "-c", "__import__('setuptools').setup()", *map(str, cmd)]
    return subprocess.run(cmd, **opts)


def run_command(*args, **kwargs):
    return run_command_inner(*args, **kwargs).stdout
