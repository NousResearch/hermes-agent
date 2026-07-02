from __future__ import annotations

import fnmatch
import itertools
import operator
import os
import stat
import textwrap
from collections.abc import Iterable, Iterator
from functools import partial
from glob import glob
from pathlib import Path
from typing import Any

from more_itertools import unique_everseen

from .._path import StrPath, StrPathT
from ..dist import Distribution
from ..warnings import SetuptoolsDeprecationWarning

import distutils.command.build_py as orig
import distutils.errors
from distutils.util import convert_path

_IMPLICIT_DATA_FILES = ('*.pyi', 'py.typed')


def make_writable(target) -> None:
    os.chmod(target, os.stat(target).st_mode | stat.S_IWRITE)


class build_py(orig.build_py):
    """Enhanced 'build_py' command that includes data files with packages

    The data files are specified via a 'package_data' argument to 'setup()'.
    See 'setuptools.dist.Distribution' for more details.

    Also, this version of the 'build_py' command allows you to specify both
    'py_modules' and 'packages' in the same setup operation.
    """

    distribution: Distribution  # override distutils.dist.Distribution with setuptools.dist.Distribution
    editable_mode: bool = False
    existing_egg_info_dir: StrPath | None = None  #: Private API, internal use only.

    def finalize_options(self) -> None:
        orig.build_py.finalize_options(self)
        self.package_data = self.distribution.package_data
        self.exclude_package_data = self.distribution.exclude_package_data or {}
        if 'data_files' in self.__dict__:
            del self.__dict__['data_files']

    def copy_file(  # type: ignore[override] # No overload, no bytes support
        self,
        infile: StrPath,
        outfile: StrPathT,
        preserve_mode: bool = True,
        preserve_times: bool = True,
        link: str | None = None,
        level: object = 1,
    ) -> tuple[StrPathT | str, bool]:
        # Overwrite base class to allow using links
        if link:
            infile = str(Path(infile).resolve())
            outfile = str(Path(outfile).resolve())  # type: ignore[assignment] # Re-assigning a str when outfile is StrPath is ok
        return super().copy_file(  # pyright: ignore[reportReturnType] # pypa/distutils#309
            infile, outfile, preserve_mode, preserve_times, link, level
        )

    def run(self) -> None:
        """Build modules, packages, and copy data files to build directory"""
        if not (self.py_modules or self.packages) or self.editable_mode:
            return

        if self.py_modules:
            self.build_modules()

        if self.packages:
            self.build_packages()
            self.build_package_data()

        # Only compile actual .py files, using our base class' idea of what our
        # output files are.
        self.byte_compile(orig.build_py.get_outputs(self, include_bytecode=False))

    # Should return "list[tuple[str, str, str, list[str]]] | Any" but can't do without typed distutils on Python 3.12+
    def __getattr__(self, attr: str) -> Any:
        "lazily compute data files"
        if attr == 'data_files':
            self.data_files = self._get_data_files()
            return self.data_files
        return orig.build_py.__getattr__(self, attr)

    def _get_data_files(self):
        """Generate list of '(package,src_dir,build_dir,filenames)' tuples"""
        self.analyze_manifest()
        return list(map(self._get_pkg_data_files, self.packages or ()))

    def get_data_files_without_manifest(self) -> list[tuple[str, str, str, list[str]]]:
        """
        Generate list of ``(package,src_dir,build_dir,filenames)`` tuples,
        but without triggering any attempt to analyze or build the manifest.
        """
        # Prevent eventual errors from unset `manifest_files`
        # (that would otherwise be set by `analyze_manifest`)
        self.__dict__.setdefault('manifest_files', {})
        return list(map(self._get_pkg_data_files, self.packages or ()))

    def _get_pkg_data_files(self, package: str) -> tuple[str, str, str, list[str]]:
        # Locate package source directory
        src_dir = self.get_package_dir(package)

        # Compute package build directory
        build_dir = os.path.join(*([self.build_lib] + package.split('.')))

        # Strip directory from globbed filenames
        filenames = [
            os.path.relpath(file, src_dir)
            for file in self.find_data_files(package, src_dir)
        ]
        return package, src_dir, build_dir, filenames

    def find_data_files(self, package, src_dir):
        """Return filenames for package's data files in 'src_dir'"""
        patterns = self._get_platform_patterns(
            self.package_data,
            package,
            src_dir,
            extra_patterns=_IMPLICIT_DATA_FILES,
        )
        globs_expanded = map(partial(glob, recursive=True), patterns)
        # flatten the expanded globs into an iterable of matches
        globs_matches = itertools.chain.from_iterable(globs_expanded)
        glob_files = filter(os.path.isfile, globs_matches)
        files = itertools.chain(
            self.manifest_files.get(package, []),
            glob_files,
        )
        return self.exclude_data_files(package, src_dir, files)

    def get_outputs(self, include_bytecode: bool = True) -> list[str]:  # type: ignore[override] # Using a real boolean instead of 0|1
        """See :class:`setuptools.commands.build.SubCommand`"""
        if self.editable_mode:
            return list(self.get_output_mapping().keys())
        return super().get_outputs(include_bytecode)

    def get_output_mapping(self) -> dict[str, str]:
        """See :class:`setuptools.commands.build.SubCommand`"""
        mapping = itertools.chain(
            self._get_package_data_output_mapping(),
            self._get_module_mapping(),
        )
        return dict(sorted(mapping, key=operator.itemgetter(0)))

    def _get_module_mapping(self) -> Iterator[tuple[str, str]]:
        """Iterate over all modules producing (dest, src) pairs."""
        for package, module, module_file in self.find_all_modules():
            package = package.split('.')
            filename = self.get_module_outfile(self.build_lib, package, module)
            yield (filename, module_file)

    def _get_package_data_output_mapping(self) -> Iterator[tuple[str, str]]:
        """Iterate over package data producing (dest, src) pairs."""
        for package, src_dir, build_dir, filenames in self.data_files:
            for filename in filenames:
                target = os.path.join(build_dir, filename)
                srcfile = os.path.join(src_dir, filename)
                yield (target, srcfile)

    def build_package_data(self) -> None:
        """Copy data files into build directory"""
        for target, srcfile in self._get_package_data_output_mapping():
            self.mkpath(os.path.dirname(target))
            _outf, _copied = self.copy_file(srcfile, target)
            make_writable(target)

    def analyze_manifest(self) -> None:
        self.manifest_files: dict[str, list[str]] = {}
        if not self.distribution.include_package_data:
            return
        src_dirs: dict[str, str] = {}
        for package in self.packages or ():
            # Locate package source directory
            src_dirs[assert_relative(self.get_package_dir(package))] = package

        if (
            self.existing_egg_info_dir
            and Path(self.existing_egg_info_dir, "SOURCES.txt").exists()
        ):
            egg_info_dir = self.existing_egg_info_dir
            manifest = Path(egg_info_dir, "SOURCES.txt")
            files = manifest.read_text(encoding="utf-8").splitlines()
        else:
            self.run_command('egg_info')
            ei_cmd = self.get_finalized_command('egg_info')
            egg_info_dir = ei_cmd.egg_info
            files = ei_cmd.filelist.files

        check = _IncludePackageDataAbuse()
        for path in self._filter_build_files(files, egg_info_dir):
            d, f = os.path.split(assert_relative(path))
            prev = None
            oldf = f
            while d and d != prev and d not in src_dirs:
                prev = d
                d, df = os.path.split(d)
                f = os.path.join(df, f)
            if d in src_dirs:
                if f == oldf:
                    if check.is_module(f):
                        continue  # it's a module, not data
                else:
                    importable = check.importable_subpackage(src_dirs[d], f)
                    if importable:
                        check.warn(importable)
                self.manifest_files.setdefault(src_dirs[d], []).append(path)

    def _filter_build_files(
        self, files: Iterable[str], egg_info: StrPath
    ) -> Iterator[str]:
        """
        ``build_meta`` may try to create egg_info outside of the project directory,
        and this can be problematic for certain plugins (reported in issue #3500).

        Extensions might also include between their sources files created on the
        ``build_lib`` and ``build_temp`` directories.

        This function should filter this case of invalid files out.
        """
        build = self.get_finalized_command("build")
        build_dirs = (egg_info, self.build_lib, build.build_temp, build.build_base)
        norm_dirs = [os.path.normpath(p) for p in build_dirs if p]

        for file in files:
            norm_path = os.path.normpath(file)
            if not os.path.isabs(file) or all(d not in norm_path for d in norm_dirs):
                yield file

    def get_data_files(self) -> None:
        pass  # Lazily compute data files in _get_data_files() function.

    def check_package(self, package, package_dir):
        """Check namespace packages' __init__ for declare_namespace"""
        try:
            return self.packages_checked[package]
        except KeyError:
            pass

        init_py = orig.build_py.check_package(self, package, package_dir)
        self.packages_checked[package] = init_py

        if not init_py or not self.distribution.namespace_packages:
            return init_py

        for pkg in self.distribution.namespace_packages:
            if pkg == package or pkg.startswith(package + '.'):
                break
        else:
            return init_py

        with open(init_py, 'rb') as f:
            contents = f.read()
        if b'declare_namespace' not in contents:
            raise distutils.errors.DistutilsError(
                f"Namespace package problem: {package} is a namespace package, but "
                "its\n__init__.py does not call declare_namespace()! Please "
                'fix it.\n(See the setuptools manual under '
                '"Namespace Packages" for details.)\n"'
            )
        return init_py

    def initialize_options(self):
        self.packages_checked = {}
        orig.build_py.initialize_options(self)
        self.editable_mode = False
        self.existing_egg_info_dir = None

    def get_package_dir(self, package: str) -> str:
        res = orig.build_py.get_package_dir(self, package)
        if self.distribution.src_root is not None:
            return os.path.join(self.distribution.src_root, res)
        return res

    def exclude_data_files(self, package, src_dir, files):
        """Filter filenames for package's data files in 'src_dir'"""
        files = list(files)
        patterns = self._get_platform_patterns(
            self.exclude_package_data,
            package,
            src_dir,
        )
        match_groups = (fnmatch.filter(files, pattern) for pattern in patterns)
        # flatten the groups of matches into an iterable of matches
        matches = itertools.chain.from_iterable(match_groups)
        bad = set(matches)
        keepers = (fn for fn in files if fn not in bad)
        # ditch dupes
        return list(unique_everseen(keepers))

    @staticmethod
    def _get_platform_patterns(spec, package, src_dir, extra_patterns=()):
        """
        yield platform-specific path patterns (suitable for glob
        or fn_match) from a glob-based spec (such as
        self.package_data or self.exclude_package_data)
        matching package in src_dir.
        """
        raw_patterns = itertools.chain(
            extra_patterns,
            spec.get('', []),
            spec.get(package, []),
        )
        return (
            # Each pattern has to be converted to a platform-specific path
            os.path.join(src_dir, convert_path(pattern))
            for pattern in raw_patterns
        )


def assert_relative(path):
    if not os.path.isabs(path):
        return path
    from distutils.errors import DistutilsSetupError

    msg = (
        textwrap.dedent(
            """
        Error: setup script specifies an absolute path:

            %s

        setup() arguments must *always* be /-separated paths relative to the
        setup.py directory, *never* absolute paths.
        """
        ).lstrip()
        % path
    )
    raise DistutilsSetupError(msg)


class _IncludePackageDataAbuse:
    """Inform users that package or module is included as 'data file'"""

    class _Warning(SetuptoolsDeprecationWarning):
        _SUMMARY = """
        Package {importable!r} is absent from the `packages` configuration.
        """

        _DETAILS = """
        ############################
        # Package would be ignored #
        ############################
        Python recognizes {importable!r} as an importable package[^1],
        but it is absent from setuptools' `packages` configuration.

        This leads to an ambiguous overall configuration. If you want to distribute this
        package, please make sure that {importable!r} is explicitly added
        to the `packages` configuration field.

        Alternatively, you can also rely on setuptools' discovery methods
        (for example by using `find_namespace_packages(...)`/`find_namespace:`
        instead of `find_packages(...)`/`find:`).

        You can read more about "package discovery" on setuptools documentation page:

        - https://setuptools.pypa.io/en/latest/userguide/package_discovery.html

        If you don't want {importable!r} to be distributed and are
        already explicitly excluding {importable!r} via
        `find_namespace_packages(...)/find_namespace` or `find_packages(...)/find`,
        you can try to use `exclude_package_data`, or `include-package-data=False` in
        combination with a more fine grained `package-data` configuration.

        You can read more about "package data files" on setuptools documentation page:

        - https://setuptools.pypa.io/en/latest/userguide/datafiles.html


        [^1]: For Python, any directory (with suitable naming) can be imported,
              even if it does not contain any `.py` files.
              On the other hand, currently there is no concept of package data
              directory, all directories are treated like packages.
        """
        # _DUE_DATE: still not defined as this is particularly controversial.
        # Warning initially introduced in May 2022. See issue #3340 for discussion.

    def __init__(self) -> None:
        self._already_warned = set[str]()

    def is_module(self, file):
        return file.endswith(".py") and file[: -len(".py")].isidentifier()

    def importable_subpackage(self, parent, file):
        pkg = Path(file).parent
        parts = list(itertools.takewhile(str.isidentifier, pkg.parts))
        if parts:
            return ".".join([parent, *parts])
        return None

    def warn(self, importable):
        if importable not in self._already_warned:
            self._Warning.emit(importable=importable)
            self._already_warned.add(importable)
