"""setuptools.command.egg_info

Create a distribution's .egg-info directory and contents"""

from __future__ import annotations

import functools
import os
import re
import sys
import time
from collections.abc import Callable

import packaging
import packaging.requirements
import packaging.version

import setuptools.unicode_utils as unicode_utils
from setuptools import Command
from setuptools.command import bdist_egg
from setuptools.command.sdist import sdist, walk_revctrl
from setuptools.command.setopt import edit_config
from setuptools.glob import glob

from .. import _entry_points, _normalization
from .._importlib import metadata
from ..warnings import SetuptoolsDeprecationWarning
from . import _requirestxt

import distutils.errors
import distutils.filelist
from distutils import log
from distutils.errors import DistutilsInternalError
from distutils.filelist import FileList as _FileList
from distutils.util import convert_path

PY_MAJOR = f'{sys.version_info.major}.{sys.version_info.minor}'


def translate_pattern(glob):  # noqa: C901  # is too complex (14)  # FIXME
    """
    Translate a file path glob like '*.txt' in to a regular expression.
    This differs from fnmatch.translate which allows wildcards to match
    directory separators. It also knows about '**/' which matches any number of
    directories.
    """
    pat = ''

    # This will split on '/' within [character classes]. This is deliberate.
    chunks = glob.split(os.path.sep)

    sep = re.escape(os.sep)
    valid_char = f'[^{sep}]'

    for c, chunk in enumerate(chunks):
        last_chunk = c == len(chunks) - 1

        # Chunks that are a literal ** are globstars. They match anything.
        if chunk == '**':
            if last_chunk:
                # Match anything if this is the last component
                pat += '.*'
            else:
                # Match '(name/)*'
                pat += f'(?:{valid_char}+{sep})*'
            continue  # Break here as the whole path component has been handled

        # Find any special characters in the remainder
        i = 0
        chunk_len = len(chunk)
        while i < chunk_len:
            char = chunk[i]
            if char == '*':
                # Match any number of name characters
                pat += valid_char + '*'
            elif char == '?':
                # Match a name character
                pat += valid_char
            elif char == '[':
                # Character class
                inner_i = i + 1
                # Skip initial !/] chars
                if inner_i < chunk_len and chunk[inner_i] == '!':
                    inner_i = inner_i + 1
                if inner_i < chunk_len and chunk[inner_i] == ']':
                    inner_i = inner_i + 1

                # Loop till the closing ] is found
                while inner_i < chunk_len and chunk[inner_i] != ']':
                    inner_i = inner_i + 1

                if inner_i >= chunk_len:
                    # Got to the end of the string without finding a closing ]
                    # Do not treat this as a matching group, but as a literal [
                    pat += re.escape(char)
                else:
                    # Grab the insides of the [brackets]
                    inner = chunk[i + 1 : inner_i]
                    char_class = ''

                    # Class negation
                    if inner[0] == '!':
                        char_class = '^'
                        inner = inner[1:]

                    char_class += re.escape(inner)
                    pat += f'[{char_class}]'

                    # Skip to the end ]
                    i = inner_i
            else:
                pat += re.escape(char)
            i += 1

        # Join each chunk with the dir separator
        if not last_chunk:
            pat += sep

    pat += r'\Z'
    return re.compile(pat, flags=re.MULTILINE | re.DOTALL)


class InfoCommon:
    tag_build = None
    tag_date = None

    @property
    def name(self):
        return _normalization.safe_name(self.distribution.get_name())

    def tagged_version(self):
        tagged = self._maybe_tag(self.distribution.get_version())
        return _normalization.safe_version(tagged)

    def _maybe_tag(self, version):
        """
        egg_info may be called more than once for a distribution,
        in which case the version string already contains all tags.
        """
        return (
            version
            if self.vtags and self._already_tagged(version)
            else version + self.vtags
        )

    def _already_tagged(self, version: str) -> bool:
        # Depending on their format, tags may change with version normalization.
        # So in addition the regular tags, we have to search for the normalized ones.
        return version.endswith((self.vtags, self._safe_tags()))

    def _safe_tags(self) -> str:
        # To implement this we can rely on `safe_version` pretending to be version 0
        # followed by tags. Then we simply discard the starting 0 (fake version number)
        try:
            return _normalization.safe_version(f"0{self.vtags}")[1:]
        except packaging.version.InvalidVersion:
            return _normalization.safe_name(self.vtags.replace(' ', '.'))

    def tags(self) -> str:
        version = ''
        if self.tag_build:
            version += self.tag_build
        if self.tag_date:
            version += time.strftime("%Y%m%d")
        return version

    vtags = property(tags)


class egg_info(InfoCommon, Command):
    description = "create a distribution's .egg-info directory"

    user_options = [
        (
            'egg-base=',
            'e',
            "directory containing .egg-info directories"
            " [default: top of the source tree]",
        ),
        ('tag-date', 'd', "Add date stamp (e.g. 20050528) to version number"),
        ('tag-build=', 'b', "Specify explicit tag to add to version number"),
        ('no-date', 'D', "Don't include date stamp [default]"),
    ]

    boolean_options = ['tag-date']
    negative_opt = {
        'no-date': 'tag-date',
    }

    def initialize_options(self):
        self.egg_base = None
        self.egg_name = None
        self.egg_info = None
        self.egg_version = None
        self.ignore_egg_info_in_manifest = False

    ####################################
    # allow the 'tag_svn_revision' to be detected and
    # set, supporting sdists built on older Setuptools.
    @property
    def tag_svn_revision(self) -> int | None:
        pass

    @tag_svn_revision.setter
    def tag_svn_revision(self, value) -> None:
        pass

    ####################################

    def save_version_info(self, filename) -> None:
        """
        Materialize the value of date into the
        build tag. Install build keys in a deterministic order
        to avoid arbitrary reordering on subsequent builds.
        """
        # follow the order these keys would have been added
        # when PYTHONHASHSEED=0
        egg_info = dict(tag_build=self.tags(), tag_date=0)
        edit_config(filename, dict(egg_info=egg_info))

    def finalize_options(self) -> None:
        # Note: we need to capture the current value returned
        # by `self.tagged_version()`, so we can later update
        # `self.distribution.metadata.version` without
        # repercussions.
        self.egg_name = self.name
        self.egg_version = self.tagged_version()
        parsed_version = packaging.version.Version(self.egg_version)

        try:
            is_version = isinstance(parsed_version, packaging.version.Version)
            spec = "%s==%s" if is_version else "%s===%s"
            packaging.requirements.Requirement(spec % (self.egg_name, self.egg_version))
        except ValueError as e:
            raise distutils.errors.DistutilsOptionError(
                f"Invalid distribution name or version syntax: {self.egg_name}-{self.egg_version}"
            ) from e

        if self.egg_base is None:
            dirs = self.distribution.package_dir
            self.egg_base = (dirs or {}).get('', os.curdir)

        self.ensure_dirname('egg_base')
        self.egg_info = _normalization.filename_component(self.egg_name) + '.egg-info'
        if self.egg_base != os.curdir:
            self.egg_info = os.path.join(self.egg_base, self.egg_info)

        # Set package version for the benefit of dumber commands
        # (e.g. sdist, bdist_wininst, etc.)
        #
        self.distribution.metadata.version = self.egg_version

    def _get_egg_basename(self, py_version=PY_MAJOR, platform=None):
        """Compute filename of the output egg. Private API."""
        return _egg_basename(self.egg_name, self.egg_version, py_version, platform)

    def write_or_delete_file(self, what, filename, data, force: bool = False) -> None:
        """Write `data` to `filename` or delete if empty

        If `data` is non-empty, this routine is the same as ``write_file()``.
        If `data` is empty but not ``None``, this is the same as calling
        ``delete_file(filename)`.  If `data` is ``None``, then this is a no-op
        unless `filename` exists, in which case a warning is issued about the
        orphaned file (if `force` is false), or deleted (if `force` is true).
        """
        if data:
            self.write_file(what, filename, data)
        elif os.path.exists(filename):
            if data is None and not force:
                log.warn("%s not set in setup(), but %s exists", what, filename)
                return
            else:
                self.delete_file(filename)

    def write_file(self, what, filename, data) -> None:
        """Write `data` to `filename` (if not a dry run) after announcing it

        `what` is used in a log message to identify what is being written
        to the file.
        """
        log.info("writing %s to %s", what, filename)
        data = data.encode("utf-8")
        f = open(filename, 'wb')
        f.write(data)
        f.close()

    def delete_file(self, filename) -> None:
        """Delete `filename` (if not a dry run) after announcing it"""
        log.info("deleting %s", filename)
        os.unlink(filename)

    def run(self) -> None:
        # Pre-load to avoid iterating over entry-points while an empty .egg-info
        # exists in sys.path. See pypa/pyproject-hooks#206
        writers = list(metadata.entry_points(group='egg_info.writers'))

        self.mkpath(self.egg_info)
        try:
            os.utime(self.egg_info, None)
        except OSError as e:
            msg = f"Cannot update time stamp of directory '{self.egg_info}'"
            raise distutils.errors.DistutilsFileError(msg) from e
        for ep in writers:
            writer = ep.load()
            writer(self, ep.name, os.path.join(self.egg_info, ep.name))

        # Get rid of native_libs.txt if it was put there by older bdist_egg
        nl = os.path.join(self.egg_info, "native_libs.txt")
        if os.path.exists(nl):
            self.delete_file(nl)

        self.find_sources()

    def find_sources(self) -> None:
        """Generate SOURCES.txt manifest file"""
        manifest_filename = os.path.join(self.egg_info, "SOURCES.txt")
        mm = manifest_maker(self.distribution)
        mm.ignore_egg_info_dir = self.ignore_egg_info_in_manifest
        mm.manifest = manifest_filename
        mm.run()
        self.filelist = mm.filelist


class FileList(_FileList):
    # Implementations of the various MANIFEST.in commands

    def __init__(
        self, warn=None, debug_print=None, ignore_egg_info_dir: bool = False
    ) -> None:
        super().__init__(warn, debug_print)
        self.ignore_egg_info_dir = ignore_egg_info_dir

    def process_template_line(self, line) -> None:
        # Parse the line: split it up, make sure the right number of words
        # is there, and return the relevant words.  'action' is always
        # defined: it's the first word of the line.  Which of the other
        # three are defined depends on the action; it'll be either
        # patterns, (dir and patterns), or (dir_pattern).
        (action, patterns, dir, dir_pattern) = self._parse_template_line(line)

        action_map: dict[str, Callable] = {
            'include': self.include,
            'exclude': self.exclude,
            'global-include': self.global_include,
            'global-exclude': self.global_exclude,
            'recursive-include': functools.partial(
                self.recursive_include,
                dir,
            ),
            'recursive-exclude': functools.partial(
                self.recursive_exclude,
                dir,
            ),
            'graft': self.graft,
            'prune': self.prune,
        }
        log_map = {
            'include': "warning: no files found matching '%s'",
            'exclude': ("warning: no previously-included files found matching '%s'"),
            'global-include': (
                "warning: no files found matching '%s' anywhere in distribution"
            ),
            'global-exclude': (
                "warning: no previously-included files matching "
                "'%s' found anywhere in distribution"
            ),
            'recursive-include': (
                "warning: no files found matching '%s' under directory '%s'"
            ),
            'recursive-exclude': (
                "warning: no previously-included files matching "
                "'%s' found under directory '%s'"
            ),
            'graft': "warning: no directories found matching '%s'",
            'prune': "no previously-included directories found matching '%s'",
        }

        try:
            process_action = action_map[action]
        except KeyError:
            msg = f"Invalid MANIFEST.in: unknown action {action!r} in {line!r}"
            raise DistutilsInternalError(msg) from None

        # OK, now we know that the action is valid and we have the
        # right number of words on the line for that action -- so we
        # can proceed with minimal error-checking.

        action_is_recursive = action.startswith('recursive-')
        if action in {'graft', 'prune'}:
            patterns = [dir_pattern]
        extra_log_args = (dir,) if action_is_recursive else ()
        log_tmpl = log_map[action]

        self.debug_print(
            ' '.join(
                [action] + ([dir] if action_is_recursive else []) + patterns,
            )
        )
        for pattern in patterns:
            if not process_action(pattern):
                log.warn(log_tmpl, pattern, *extra_log_args)

    def _remove_files(self, predicate):
        """
        Remove all files from the file list that match the predicate.
        Return True if any matching files were removed
        """
        found = False
        for i in range(len(self.files) - 1, -1, -1):
            if predicate(self.files[i]):
                self.debug_print(" removing " + self.files[i])
                del self.files[i]
                found = True
        return found

    def include(self, pattern):
        """Include files that match 'pattern'."""
        found = [f for f in glob(pattern) if not os.path.isdir(f)]
        self.extend(found)
        return bool(found)

    def exclude(self, pattern):
        """Exclude files that match 'pattern'."""
        match = translate_pattern(pattern)
        return self._remove_files(match.match)

    def recursive_include(self, dir, pattern):
        """
        Include all files anywhere in 'dir/' that match the pattern.
        """
        full_pattern = os.path.join(dir, '**', pattern)
        found = [f for f in glob(full_pattern, recursive=True) if not os.path.isdir(f)]
        self.extend(found)
        return bool(found)

    def recursive_exclude(self, dir, pattern):
        """
        Exclude any file anywhere in 'dir/' that match the pattern.
        """
        match = translate_pattern(os.path.join(dir, '**', pattern))
        return self._remove_files(match.match)

    def graft(self, dir):
        """Include all files from 'dir/'."""
        found = [
            item
            for match_dir in glob(dir)
            for item in distutils.filelist.findall(match_dir)
        ]
        self.extend(found)
        return bool(found)

    def prune(self, dir):
        """Filter out files from 'dir/'."""
        match = translate_pattern(os.path.join(dir, '**'))
        return self._remove_files(match.match)

    def global_include(self, pattern):
        """
        Include all files anywhere in the current directory that match the
        pattern. This is very inefficient on large file trees.
        """
        if self.allfiles is None:
            self.findall()
        match = translate_pattern(os.path.join('**', pattern))
        found = [f for f in self.allfiles if match.match(f)]
        self.extend(found)
        return bool(found)

    def global_exclude(self, pattern):
        """
        Exclude all files anywhere that match the pattern.
        """
        match = translate_pattern(os.path.join('**', pattern))
        return self._remove_files(match.match)

    def append(self, item) -> None:
        item = item.removesuffix('\r')  # Fix older sdists built on Windows
        path = convert_path(item)

        if self._safe_path(path):
            self.files.append(path)

    def extend(self, paths) -> None:
        self.files.extend(filter(self._safe_path, paths))

    def _repair(self):
        """
        Replace self.files with only safe paths

        Because some owners of FileList manipulate the underlying
        ``files`` attribute directly, this method must be called to
        repair those paths.
        """
        self.files = list(filter(self._safe_path, self.files))

    def _safe_path(self, path):
        enc_warn = "'%s' not %s encodable -- skipping"

        # To avoid accidental trans-codings errors, first to unicode
        u_path = unicode_utils.filesys_decode(path)
        if u_path is None:
            log.warn(f"'{path}' in unexpected encoding -- skipping")
            return False

        # Must ensure utf-8 encodability
        utf8_path = unicode_utils.try_encode(u_path, "utf-8")
        if utf8_path is None:
            log.warn(enc_warn, path, 'utf-8')
            return False

        try:
            # ignore egg-info paths
            is_egg_info = ".egg-info" in u_path or b".egg-info" in utf8_path
            if self.ignore_egg_info_dir and is_egg_info:
                return False
            # accept is either way checks out
            if os.path.exists(u_path) or os.path.exists(utf8_path):
                return True
        # this will catch any encode errors decoding u_path
        except UnicodeEncodeError:
            log.warn(enc_warn, path, sys.getfilesystemencoding())


class manifest_maker(sdist):
    template = "MANIFEST.in"

    def initialize_options(self) -> None:
        self.use_defaults = True
        self.prune = True
        self.manifest_only = True
        self.force_manifest = True
        self.ignore_egg_info_dir = False

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        self.filelist = FileList(ignore_egg_info_dir=self.ignore_egg_info_dir)
        if not os.path.exists(self.manifest):
            self.write_manifest()  # it must exist so it'll get in the list
        self.add_defaults()
        if os.path.exists(self.template):
            self.read_template()
        self.add_license_files()
        self._add_referenced_files()
        self.prune_file_list()
        self.filelist.sort()
        self.filelist.remove_duplicates()
        self.write_manifest()

    def _manifest_normalize(self, path):
        path = unicode_utils.filesys_decode(path)
        return path.replace(os.sep, '/')

    def write_manifest(self) -> None:
        """
        Write the file list in 'self.filelist' to the manifest file
        named by 'self.manifest'.
        """
        self.filelist._repair()

        # Now _repairs should encodability, but not unicode
        files = [self._manifest_normalize(f) for f in self.filelist.files]
        msg = f"writing manifest file '{self.manifest}'"
        self.execute(write_file, (self.manifest, files), msg)

    def warn(self, msg) -> None:
        if not self._should_suppress_warning(msg):
            sdist.warn(self, msg)

    @staticmethod
    def _should_suppress_warning(msg):
        """
        suppress missing-file warnings from sdist
        """
        return re.match(r"standard file .*not found", msg)

    def add_defaults(self) -> None:
        sdist.add_defaults(self)
        self.filelist.append(self.template)
        self.filelist.append(self.manifest)
        rcfiles = list(walk_revctrl())
        if rcfiles:
            self.filelist.extend(rcfiles)
        elif os.path.exists(self.manifest):
            self.read_manifest()

        if os.path.exists("setup.py"):
            # setup.py should be included by default, even if it's not
            # the script called to create the sdist
            self.filelist.append("setup.py")

        ei_cmd = self.get_finalized_command('egg_info')
        self.filelist.graft(ei_cmd.egg_info)

    def add_license_files(self) -> None:
        license_files = self.distribution.metadata.license_files or []
        for lf in license_files:
            log.info("adding license file '%s'", lf)
        self.filelist.extend(license_files)

    def _add_referenced_files(self):
        """Add files referenced by the config (e.g. `file:` directive) to filelist"""
        referenced = getattr(self.distribution, '_referenced_files', [])
        # ^-- fallback if dist comes from distutils or is a custom class
        for rf in referenced:
            log.debug("adding file referenced by config '%s'", rf)
        self.filelist.extend(referenced)

    def _safe_data_files(self, build_py):
        """
        The parent class implementation of this method
        (``sdist``) will try to include data files, which
        might cause recursion problems when
        ``include_package_data=True``.

        Therefore, avoid triggering any attempt of
        analyzing/building the manifest again.
        """
        if hasattr(build_py, 'get_data_files_without_manifest'):
            return build_py.get_data_files_without_manifest()

        SetuptoolsDeprecationWarning.emit(
            "`build_py` command does not inherit from setuptools' `build_py`.",
            """
            Custom 'build_py' does not implement 'get_data_files_without_manifest'.
            Please extend command classes from setuptools instead of distutils.
            """,
            see_url="https://peps.python.org/pep-0632/",
            # due_date not defined yet, old projects might still do it?
        )
        return build_py.get_data_files()


def write_file(filename, contents) -> None:
    """Create a file with the specified name and write 'contents' (a
    sequence of strings without line terminators) to it.
    """
    contents = "\n".join(contents)

    # assuming the contents has been vetted for utf-8 encoding
    contents = contents.encode("utf-8")

    with open(filename, "wb") as f:  # always write POSIX-style manifest
        f.write(contents)


def write_pkg_info(cmd, basename, filename) -> None:
    log.info("writing %s", filename)
    metadata = cmd.distribution.metadata
    metadata.version, oldver = cmd.egg_version, metadata.version
    metadata.name, oldname = cmd.egg_name, metadata.name

    try:
        metadata.write_pkg_info(cmd.egg_info)
    finally:
        metadata.name, metadata.version = oldname, oldver

    safe = getattr(cmd.distribution, 'zip_safe', None)

    bdist_egg.write_safety_flag(cmd.egg_info, safe)


def warn_depends_obsolete(cmd, basename, filename) -> None:
    """
    Unused: left to avoid errors when updating (from source) from <= 67.8.
    Old installations have a .dist-info directory with the entry-point
    ``depends.txt = setuptools.command.egg_info:warn_depends_obsolete``.
    This may trigger errors when running the first egg_info in build_meta.
    TODO: Remove this function in a version sufficiently > 68.
    """


# Export API used in entry_points
write_requirements = _requirestxt.write_requirements
write_setup_requirements = _requirestxt.write_setup_requirements


def write_toplevel_names(cmd, basename, filename) -> None:
    pkgs = dict.fromkeys([
        k.split('.', 1)[0] for k in cmd.distribution.iter_distribution_names()
    ])
    cmd.write_file("top-level names", filename, '\n'.join(sorted(pkgs)) + '\n')


def overwrite_arg(cmd, basename, filename) -> None:
    write_arg(cmd, basename, filename, True)


def write_arg(cmd, basename, filename, force: bool = False) -> None:
    argname = os.path.splitext(basename)[0]
    value = getattr(cmd.distribution, argname, None)
    if value is not None:
        value = '\n'.join(value) + '\n'
    cmd.write_or_delete_file(argname, filename, value, force)


def write_entries(cmd, basename, filename) -> None:
    eps = _entry_points.load(cmd.distribution.entry_points)
    defn = _entry_points.render(eps)
    cmd.write_or_delete_file('entry points', filename, defn, True)


def _egg_basename(egg_name, egg_version, py_version=None, platform=None):
    """Compute filename of the output egg. Private API."""
    name = _normalization.filename_component(egg_name)
    version = _normalization.filename_component(egg_version)
    egg = f"{name}-{version}-py{py_version or PY_MAJOR}"
    if platform:
        egg += f"-{platform}"
    return egg


class EggInfoDeprecationWarning(SetuptoolsDeprecationWarning):
    """Deprecated behavior warning for EggInfo, bypassing suppression."""
