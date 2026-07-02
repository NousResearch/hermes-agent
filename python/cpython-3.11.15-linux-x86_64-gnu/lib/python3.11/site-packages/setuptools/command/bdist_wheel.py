"""
Create a wheel (.whl) distribution.

A wheel is a built archive format.
"""

from __future__ import annotations

import os
import re
import shutil
import struct
import sys
import sysconfig
import warnings
from collections.abc import Iterable, Sequence
from email.generator import BytesGenerator
from glob import iglob
from typing import Literal, cast
from zipfile import ZIP_DEFLATED, ZIP_STORED

from packaging import tags, version as _packaging_version
from wheel.wheelfile import WheelFile

from .. import Command, __version__, _shutil
from .._core_metadata import _safe_license_file
from .._normalization import safer_name
from ..warnings import SetuptoolsDeprecationWarning
from .egg_info import egg_info as egg_info_cls

from distutils import log


def safe_version(version: str) -> str:
    """
    Convert an arbitrary string to a standard version string
    """
    try:
        # normalize the version
        return str(_packaging_version.Version(version))
    except _packaging_version.InvalidVersion:
        version = version.replace(" ", ".")
        return re.sub("[^A-Za-z0-9.]+", "-", version)


setuptools_major_version = int(__version__.split(".")[0])

PY_LIMITED_API_PATTERN = r"cp3\d"


def _is_32bit_interpreter() -> bool:
    return struct.calcsize("P") == 4


def python_tag() -> str:
    return f"py{sys.version_info.major}"


def get_platform(archive_root: str | None) -> str:
    """Return our platform name 'win32', 'linux_x86_64'"""
    result = sysconfig.get_platform()
    if result.startswith("macosx") and archive_root is not None:  # pragma: no cover
        from wheel.macosx_libfile import calculate_macosx_platform_tag

        result = calculate_macosx_platform_tag(archive_root, result)
    elif _is_32bit_interpreter():
        if result == "linux-x86_64":
            # pip pull request #3497
            result = "linux-i686"
        elif result == "linux-aarch64":
            # packaging pull request #234
            # TODO armv8l, packaging pull request #690 => this did not land
            # in pip/packaging yet
            result = "linux-armv7l"

    return result.replace("-", "_")


def get_flag(
    var: str, fallback: bool, expected: bool = True, warn: bool = True
) -> bool:
    """Use a fallback value for determining SOABI flags if the needed config
    var is unset or unavailable."""
    val = sysconfig.get_config_var(var)
    if val is None:
        if warn:
            warnings.warn(
                f"Config variable '{var}' is unset, Python ABI tag may be incorrect",
                RuntimeWarning,
                stacklevel=2,
            )
        return fallback
    return val == expected


def get_abi_tag() -> str | None:
    """Return the ABI tag based on SOABI (if available) or emulate SOABI (PyPy2)."""
    soabi: str = sysconfig.get_config_var("SOABI")
    impl = tags.interpreter_name()
    if not soabi and impl in ("cp", "pp") and hasattr(sys, "maxunicode"):
        d = ""
        u = ""
        if get_flag("Py_DEBUG", hasattr(sys, "gettotalrefcount"), warn=(impl == "cp")):
            d = "d"

        abi = f"{impl}{tags.interpreter_version()}{d}{u}"
    elif soabi and impl == "cp" and soabi.startswith("cpython"):
        # non-Windows
        abi = "cp" + soabi.split("-")[1]
    elif soabi and impl == "cp" and soabi.startswith("cp"):
        # Windows
        abi = soabi.split("-")[0]
        if hasattr(sys, "gettotalrefcount"):
            # using debug build; append "d" flag
            abi += "d"
    elif soabi and impl == "pp":
        # we want something like pypy36-pp73
        abi = "-".join(soabi.split("-")[:2])
        abi = abi.replace(".", "_").replace("-", "_")
    elif soabi and impl == "graalpy":
        abi = "-".join(soabi.split("-")[:3])
        abi = abi.replace(".", "_").replace("-", "_")
    elif soabi:
        abi = soabi.replace(".", "_").replace("-", "_")
    else:
        abi = None

    return abi


def safer_version(version: str) -> str:
    return safe_version(version).replace("-", "_")


class bdist_wheel(Command):
    description = "create a wheel distribution"

    supported_compressions = {
        "stored": ZIP_STORED,
        "deflated": ZIP_DEFLATED,
    }

    user_options = [
        ("bdist-dir=", "b", "temporary directory for creating the distribution"),
        (
            "plat-name=",
            "p",
            "platform name to embed in generated filenames "
            f"[default: {get_platform(None)}]",
        ),
        (
            "keep-temp",
            "k",
            "keep the pseudo-installation tree around after "
            "creating the distribution archive",
        ),
        ("dist-dir=", "d", "directory to put final built distributions in"),
        ("skip-build", None, "skip rebuilding everything (for testing/debugging)"),
        (
            "relative",
            None,
            "build the archive using relative paths [default: false]",
        ),
        (
            "owner=",
            "u",
            "Owner name used when creating a tar file [default: current user]",
        ),
        (
            "group=",
            "g",
            "Group name used when creating a tar file [default: current group]",
        ),
        ("universal", None, "*DEPRECATED* make a universal wheel [default: false]"),
        (
            "compression=",
            None,
            f"zipfile compression (one of: {', '.join(supported_compressions)}) [default: 'deflated']",
        ),
        (
            "python-tag=",
            None,
            f"Python implementation compatibility tag [default: '{python_tag()}']",
        ),
        (
            "build-number=",
            None,
            "Build number for this particular version. "
            "As specified in PEP-0427, this must start with a digit. "
            "[default: None]",
        ),
        (
            "py-limited-api=",
            None,
            "Python tag (cp32|cp33|cpNN) for abi3 wheel tag [default: false]",
        ),
        (
            "dist-info-dir=",
            None,
            "directory where a pre-generated dist-info can be found (e.g. as a "
            "result of calling the PEP517 'prepare_metadata_for_build_wheel' "
            "method)",
        ),
    ]

    boolean_options = ["keep-temp", "skip-build", "relative", "universal"]

    def initialize_options(self) -> None:
        self.bdist_dir: str | None = None
        self.data_dir = ""
        self.plat_name: str | None = None
        self.plat_tag: str | None = None
        self.format = "zip"
        self.keep_temp = False
        self.dist_dir: str | None = None
        self.dist_info_dir = None
        self.egginfo_dir: str | None = None
        self.root_is_pure: bool | None = None
        self.skip_build = False
        self.relative = False
        self.owner = None
        self.group = None
        self.universal = False
        self.compression: str | int = "deflated"
        self.python_tag = python_tag()
        self.build_number: str | None = None
        self.py_limited_api: str | Literal[False] = False
        self.plat_name_supplied = False

    def finalize_options(self) -> None:
        if not self.bdist_dir:
            bdist_base = self.get_finalized_command("bdist").bdist_base
            self.bdist_dir = os.path.join(bdist_base, "wheel")

        if self.dist_info_dir is None:
            egg_info = cast(egg_info_cls, self.distribution.get_command_obj("egg_info"))
            egg_info.ensure_finalized()  # needed for correct `wheel_dist_name`

        self.data_dir = self.wheel_dist_name + ".data"
        self.plat_name_supplied = bool(self.plat_name)

        need_options = ("dist_dir", "plat_name", "skip_build")

        self.set_undefined_options("bdist", *zip(need_options, need_options))

        self.root_is_pure = not (
            self.distribution.has_ext_modules() or self.distribution.has_c_libraries()
        )

        self._validate_py_limited_api()

        # Support legacy [wheel] section for setting universal
        wheel = self.distribution.get_option_dict("wheel")
        if "universal" in wheel:  # pragma: no cover
            # please don't define this in your global configs
            log.warn("The [wheel] section is deprecated. Use [bdist_wheel] instead.")
            val = wheel["universal"][1].strip()
            if val.lower() in ("1", "true", "yes"):
                self.universal = True

        if self.universal:
            SetuptoolsDeprecationWarning.emit(
                "bdist_wheel.universal is deprecated",
                """
                With Python 2.7 end-of-life, support for building universal wheels
                (i.e., wheels that support both Python 2 and Python 3)
                is being obviated.
                Please discontinue using this option, or if you still need it,
                file an issue with pypa/setuptools describing your use case.
                """,
                due_date=(2025, 8, 30),  # Introduced in 2024-08-30
            )

        if self.build_number is not None and not self.build_number[:1].isdigit():
            raise ValueError("Build tag (build-number) must start with a digit.")

    def _validate_py_limited_api(self) -> None:
        if not self.py_limited_api:
            return

        if not re.match(PY_LIMITED_API_PATTERN, self.py_limited_api):
            raise ValueError(f"py-limited-api must match '{PY_LIMITED_API_PATTERN}'")

        if sysconfig.get_config_var("Py_GIL_DISABLED"):
            raise ValueError(
                f"`py_limited_api={self.py_limited_api!r}` not supported. "
                "`Py_LIMITED_API` is currently incompatible with "
                "`Py_GIL_DISABLED`. "
                "See https://github.com/python/cpython/issues/111506."
            )

    @property
    def wheel_dist_name(self) -> str:
        """Return distribution full name with - replaced with _"""
        components = [
            safer_name(self.distribution.get_name()),
            safer_version(self.distribution.get_version()),
        ]
        if self.build_number:
            components.append(self.build_number)
        return "-".join(components)

    def get_tag(self) -> tuple[str, str, str]:
        # bdist sets self.plat_name if unset, we should only use it for purepy
        # wheels if the user supplied it.
        if self.plat_name_supplied and self.plat_name:
            plat_name = self.plat_name
        elif self.root_is_pure:
            plat_name = "any"
        else:
            # macosx contains system version in platform name so need special handle
            if self.plat_name and not self.plat_name.startswith("macosx"):
                plat_name = self.plat_name
            else:
                # on macosx always limit the platform name to comply with any
                # c-extension modules in bdist_dir, since the user can specify
                # a higher MACOSX_DEPLOYMENT_TARGET via tools like CMake

                # on other platforms, and on macosx if there are no c-extension
                # modules, use the default platform name.
                plat_name = get_platform(self.bdist_dir)

            if _is_32bit_interpreter():
                if plat_name in ("linux-x86_64", "linux_x86_64"):
                    plat_name = "linux_i686"
                if plat_name in ("linux-aarch64", "linux_aarch64"):
                    # TODO armv8l, packaging pull request #690 => this did not land
                    # in pip/packaging yet
                    plat_name = "linux_armv7l"

        plat_name = (
            plat_name.lower().replace("-", "_").replace(".", "_").replace(" ", "_")
        )

        if self.root_is_pure:
            if self.universal:
                impl = "py2.py3"
            else:
                impl = self.python_tag
            tag = (impl, "none", plat_name)
        else:
            impl_name = tags.interpreter_name()
            impl_ver = tags.interpreter_version()
            impl = impl_name + impl_ver
            # We don't work on CPython 3.1, 3.0.
            if self.py_limited_api and (impl_name + impl_ver).startswith("cp3"):
                impl = self.py_limited_api
                abi_tag = "abi3"
            else:
                abi_tag = str(get_abi_tag()).lower()
            tag = (impl, abi_tag, plat_name)
            # issue gh-374: allow overriding plat_name
            supported_tags = [
                (t.interpreter, t.abi, plat_name) for t in tags.sys_tags()
            ]
            assert tag in supported_tags, (
                f"would build wheel with unsupported tag {tag}"
            )
        return tag

    def run(self):
        build_scripts = self.reinitialize_command("build_scripts")
        build_scripts.executable = "python"
        build_scripts.force = True

        build_ext = self.reinitialize_command("build_ext")
        build_ext.inplace = False

        if not self.skip_build:
            self.run_command("build")

        install = self.reinitialize_command("install", reinit_subcommands=True)
        install.root = self.bdist_dir
        install.compile = False
        install.skip_build = self.skip_build
        install.warn_dir = False

        # A wheel without setuptools scripts is more cross-platform.
        # Use the (undocumented) `no_ep` option to setuptools'
        # install_scripts command to avoid creating entry point scripts.
        install_scripts = self.reinitialize_command("install_scripts")
        install_scripts.no_ep = True

        # Use a custom scheme for the archive, because we have to decide
        # at installation time which scheme to use.
        for key in ("headers", "scripts", "data", "purelib", "platlib"):
            setattr(install, "install_" + key, os.path.join(self.data_dir, key))

        basedir_observed = ""

        if os.name == "nt":
            # win32 barfs if any of these are ''; could be '.'?
            # (distutils.command.install:change_roots bug)
            basedir_observed = os.path.normpath(os.path.join(self.data_dir, ".."))
            self.install_libbase = self.install_lib = basedir_observed

        setattr(
            install,
            "install_purelib" if self.root_is_pure else "install_platlib",
            basedir_observed,
        )

        log.info(f"installing to {self.bdist_dir}")

        self.run_command("install")

        impl_tag, abi_tag, plat_tag = self.get_tag()
        archive_basename = f"{self.wheel_dist_name}-{impl_tag}-{abi_tag}-{plat_tag}"
        if not self.relative:
            archive_root = self.bdist_dir
        else:
            archive_root = os.path.join(
                self.bdist_dir, self._ensure_relative(install.install_base)
            )

        self.set_undefined_options("install_egg_info", ("target", "egginfo_dir"))
        distinfo_dirname = (
            f"{safer_name(self.distribution.get_name())}-"
            f"{safer_version(self.distribution.get_version())}.dist-info"
        )
        distinfo_dir = os.path.join(self.bdist_dir, distinfo_dirname)
        if self.dist_info_dir:
            # Use the given dist-info directly.
            log.debug(f"reusing {self.dist_info_dir}")
            shutil.copytree(self.dist_info_dir, distinfo_dir)
            # Egg info is still generated, so remove it now to avoid it getting
            # copied into the wheel.
            _shutil.rmtree(self.egginfo_dir)
        else:
            # Convert the generated egg-info into dist-info.
            self.egg2dist(self.egginfo_dir, distinfo_dir)

        self.write_wheelfile(distinfo_dir)

        # Make the archive
        if not os.path.exists(self.dist_dir):
            os.makedirs(self.dist_dir)

        wheel_path = os.path.join(self.dist_dir, archive_basename + ".whl")
        with WheelFile(wheel_path, "w", self._zip_compression()) as wf:
            wf.write_files(archive_root)

        # Add to 'Distribution.dist_files' so that the "upload" command works
        getattr(self.distribution, "dist_files", []).append((
            "bdist_wheel",
            f"{sys.version_info.major}.{sys.version_info.minor}",
            wheel_path,
        ))

        if not self.keep_temp:
            log.info(f"removing {self.bdist_dir}")
            _shutil.rmtree(self.bdist_dir)

    def write_wheelfile(
        self, wheelfile_base: str, generator: str = f"setuptools ({__version__})"
    ) -> None:
        from email.message import Message

        msg = Message()
        msg["Wheel-Version"] = "1.0"  # of the spec
        msg["Generator"] = generator
        msg["Root-Is-Purelib"] = str(self.root_is_pure).lower()
        if self.build_number is not None:
            msg["Build"] = self.build_number

        # Doesn't work for bdist_wininst
        impl_tag, abi_tag, plat_tag = self.get_tag()
        for impl in impl_tag.split("."):
            for abi in abi_tag.split("."):
                for plat in plat_tag.split("."):
                    msg["Tag"] = "-".join((impl, abi, plat))

        wheelfile_path = os.path.join(wheelfile_base, "WHEEL")
        log.info(f"creating {wheelfile_path}")
        with open(wheelfile_path, "wb") as f:
            BytesGenerator(f, maxheaderlen=0).flatten(msg)

    def _ensure_relative(self, path: str) -> str:
        # copied from dir_util, deleted
        drive, path = os.path.splitdrive(path)
        if path[0:1] == os.sep:
            path = drive + path[1:]
        return path

    @property
    def license_paths(self) -> Iterable[str]:
        if setuptools_major_version >= 57:
            # Setuptools has resolved any patterns to actual file names
            return self.distribution.metadata.license_files or ()

        files = set[str]()
        metadata = self.distribution.get_option_dict("metadata")
        if setuptools_major_version >= 42:
            # Setuptools recognizes the license_files option but does not do globbing
            patterns = cast(Sequence[str], self.distribution.metadata.license_files)
        else:
            # Prior to those, wheel is entirely responsible for handling license files
            if "license_files" in metadata:
                patterns = metadata["license_files"][1].split()
            else:
                patterns = ()

        if "license_file" in metadata:
            warnings.warn(
                'The "license_file" option is deprecated. Use "license_files" instead.',
                DeprecationWarning,
                stacklevel=2,
            )
            files.add(metadata["license_file"][1])

        if not files and not patterns and not isinstance(patterns, list):
            patterns = ("LICEN[CS]E*", "COPYING*", "NOTICE*", "AUTHORS*")

        for pattern in patterns:
            for path in iglob(pattern):
                if path.endswith("~"):
                    log.debug(
                        f'ignoring license file "{path}" as it looks like a backup'
                    )
                    continue

                if path not in files and os.path.isfile(path):
                    log.info(
                        f'adding license file "{path}" (matched pattern "{pattern}")'
                    )
                    files.add(path)

        return files

    def egg2dist(self, egginfo_path: str, distinfo_path: str) -> None:
        """Convert an .egg-info directory into a .dist-info directory"""

        def adios(p: str) -> None:
            """Appropriately delete directory, file or link."""
            if os.path.exists(p) and not os.path.islink(p) and os.path.isdir(p):
                _shutil.rmtree(p)
            elif os.path.exists(p):
                os.unlink(p)

        adios(distinfo_path)

        if not os.path.exists(egginfo_path):
            # There is no egg-info. This is probably because the egg-info
            # file/directory is not named matching the distribution name used
            # to name the archive file. Check for this case and report
            # accordingly.
            import glob

            pat = os.path.join(os.path.dirname(egginfo_path), "*.egg-info")
            possible = glob.glob(pat)
            err = f"Egg metadata expected at {egginfo_path} but not found"
            if possible:
                alt = os.path.basename(possible[0])
                err += f" ({alt} found - possible misnamed archive file?)"

            raise ValueError(err)

        # .egg-info is a directory
        pkginfo_path = os.path.join(egginfo_path, "PKG-INFO")

        # ignore common egg metadata that is useless to wheel
        shutil.copytree(
            egginfo_path,
            distinfo_path,
            ignore=lambda x, y: {
                "PKG-INFO",
                "requires.txt",
                "SOURCES.txt",
                "not-zip-safe",
            },
        )

        # delete dependency_links if it is only whitespace
        dependency_links_path = os.path.join(distinfo_path, "dependency_links.txt")
        with open(dependency_links_path, encoding="utf-8") as dependency_links_file:
            dependency_links = dependency_links_file.read().strip()
        if not dependency_links:
            adios(dependency_links_path)

        metadata_path = os.path.join(distinfo_path, "METADATA")
        shutil.copy(pkginfo_path, metadata_path)

        licenses_folder_path = os.path.join(distinfo_path, "licenses")
        for license_path in self.license_paths:
            safe_path = _safe_license_file(license_path)
            dist_info_license_path = os.path.join(licenses_folder_path, safe_path)
            os.makedirs(os.path.dirname(dist_info_license_path), exist_ok=True)
            shutil.copy(license_path, dist_info_license_path)

        adios(egginfo_path)

    def _zip_compression(self) -> int:
        if (
            isinstance(self.compression, int)
            and self.compression in self.supported_compressions.values()
        ):
            return self.compression

        compression = self.supported_compressions.get(str(self.compression))
        if compression is not None:
            return compression

        raise ValueError(f"Unsupported compression: {self.compression!r}")
