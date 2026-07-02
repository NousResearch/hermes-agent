from __future__ import annotations

import contextlib
import os
import re
from collections.abc import Iterator
from itertools import chain
from typing import ClassVar

from .._importlib import metadata
from ..dist import Distribution
from .build import _ORIGINAL_SUBCOMMANDS

import distutils.command.sdist as orig
from distutils import log

_default_revctrl = list


def walk_revctrl(dirname='') -> Iterator:
    """Find all files under revision control"""
    for ep in metadata.entry_points(group='setuptools.file_finders'):
        yield from ep.load()(dirname)


class sdist(orig.sdist):
    """Smart sdist that finds anything supported by revision control"""

    user_options = [
        ('formats=', None, "formats for source distribution (comma-separated list)"),
        (
            'keep-temp',
            'k',
            "keep the distribution tree around after creating archive file(s)",
        ),
        (
            'dist-dir=',
            'd',
            "directory to put the source distribution archive(s) in [default: dist]",
        ),
        (
            'owner=',
            'u',
            "Owner name used when creating a tar file [default: current user]",
        ),
        (
            'group=',
            'g',
            "Group name used when creating a tar file [default: current group]",
        ),
    ]

    distribution: Distribution  # override distutils.dist.Distribution with setuptools.dist.Distribution
    negative_opt: ClassVar[dict[str, str]] = {}

    README_EXTENSIONS = ['', '.rst', '.txt', '.md']
    READMES = tuple(f'README{ext}' for ext in README_EXTENSIONS)

    def run(self) -> None:
        self.run_command('egg_info')
        ei_cmd = self.get_finalized_command('egg_info')
        self.filelist = ei_cmd.filelist
        self.filelist.append(os.path.join(ei_cmd.egg_info, 'SOURCES.txt'))
        self.check_readme()

        # Run sub commands
        for cmd_name in self.get_sub_commands():
            self.run_command(cmd_name)

        self.make_distribution()

        dist_files = getattr(self.distribution, 'dist_files', [])
        for file in self.archive_files:
            data = ('sdist', '', file)
            if data not in dist_files:
                dist_files.append(data)

    def initialize_options(self) -> None:
        orig.sdist.initialize_options(self)

    def make_distribution(self) -> None:
        """
        Workaround for #516
        """
        with self._remove_os_link():
            orig.sdist.make_distribution(self)

    @staticmethod
    @contextlib.contextmanager
    def _remove_os_link():
        """
        In a context, remove and restore os.link if it exists
        """

        class NoValue:
            pass

        orig_val = getattr(os, 'link', NoValue)
        try:
            del os.link
        except Exception:
            pass
        try:
            yield
        finally:
            if orig_val is not NoValue:
                os.link = orig_val

    def add_defaults(self) -> None:
        super().add_defaults()
        self._add_defaults_build_sub_commands()

    def _add_defaults_optional(self):
        super()._add_defaults_optional()
        if os.path.isfile('pyproject.toml'):
            self.filelist.append('pyproject.toml')

    def _add_defaults_python(self):
        """getting python files"""
        if self.distribution.has_pure_modules():
            build_py = self.get_finalized_command('build_py')
            self.filelist.extend(build_py.get_source_files())
            self._add_data_files(self._safe_data_files(build_py))

    def _add_defaults_build_sub_commands(self):
        build = self.get_finalized_command("build")
        missing_cmds = set(build.get_sub_commands()) - _ORIGINAL_SUBCOMMANDS
        # ^-- the original built-in sub-commands are already handled by default.
        cmds = (self.get_finalized_command(c) for c in missing_cmds)
        files = (c.get_source_files() for c in cmds if hasattr(c, "get_source_files"))
        self.filelist.extend(chain.from_iterable(files))

    def _safe_data_files(self, build_py):
        """
        Since the ``sdist`` class is also used to compute the MANIFEST
        (via :obj:`setuptools.command.egg_info.manifest_maker`),
        there might be recursion problems when trying to obtain the list of
        data_files and ``include_package_data=True`` (which in turn depends on
        the files included in the MANIFEST).

        To avoid that, ``manifest_maker`` should be able to overwrite this
        method and avoid recursive attempts to build/analyze the MANIFEST.
        """
        return build_py.data_files

    def _add_data_files(self, data_files):
        """
        Add data files as found in build_py.data_files.
        """
        self.filelist.extend(
            os.path.join(src_dir, name)
            for _, src_dir, _, filenames in data_files
            for name in filenames
        )

    def _add_defaults_data_files(self):
        try:
            super()._add_defaults_data_files()
        except TypeError:
            log.warn("data_files contains unexpected objects")

    def prune_file_list(self) -> None:
        super().prune_file_list()
        # Prevent accidental inclusion of test-related cache dirs at the project root
        sep = re.escape(os.sep)
        self.filelist.exclude_pattern(r"^(\.tox|\.nox|\.venv)" + sep, is_regex=True)

    def check_readme(self) -> None:
        for f in self.READMES:
            if os.path.exists(f):
                return
        else:
            self.warn(
                "standard file not found: should have one of " + ', '.join(self.READMES)
            )

    def make_release_tree(self, base_dir, files) -> None:
        orig.sdist.make_release_tree(self, base_dir, files)

        # Save any egg_info command line options used to create this sdist
        dest = os.path.join(base_dir, 'setup.cfg')
        if hasattr(os, 'link') and os.path.exists(dest):
            # unlink and re-copy, since it might be hard-linked, and
            # we don't want to change the source version
            os.unlink(dest)
            self.copy_file('setup.cfg', dest)

        self.get_finalized_command('egg_info').save_version_info(dest)

    def _manifest_is_not_generated(self):
        # check for special comment used in 2.7.1 and higher
        if not os.path.isfile(self.manifest):
            return False

        with open(self.manifest, 'rb') as fp:
            first_line = fp.readline()
        return first_line != b'# file GENERATED by distutils, do NOT edit\n'

    def read_manifest(self) -> None:
        """Read the manifest file (named by 'self.manifest') and use it to
        fill in 'self.filelist', the list of files to include in the source
        distribution.
        """
        log.info("reading manifest file '%s'", self.manifest)
        manifest = open(self.manifest, 'rb')
        for bytes_line in manifest:
            # The manifest must contain UTF-8. See #303.
            try:
                line = bytes_line.decode('UTF-8')
            except UnicodeDecodeError:
                log.warn(f"{line!r} not UTF-8 decodable -- skipping")
                continue
            # ignore comments and blank lines
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            self.filelist.append(line)
        manifest.close()
