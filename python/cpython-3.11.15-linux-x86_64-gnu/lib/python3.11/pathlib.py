import fnmatch
import functools
import io
import ntpath
import os
import posixpath
import re
import sys
import warnings
from _collections_abc import Sequence
from errno import ENOENT, ENOTDIR, EBADF, ELOOP
from operator import attrgetter
from stat import S_ISDIR, S_ISLNK, S_ISREG, S_ISSOCK, S_ISBLK, S_ISCHR, S_ISFIFO
from urllib.parse import quote_from_bytes as urlquote_from_bytes


__all__ = [
    "PurePath", "PurePosixPath", "PureWindowsPath",
    "Path", "PosixPath", "WindowsPath",
    ]

#
# Internals
#

_WINERROR_NOT_READY = 21  # drive exists but is not accessible
_WINERROR_INVALID_NAME = 123  # fix for bpo-35306
_WINERROR_CANT_RESOLVE_FILENAME = 1921  # broken symlink pointing to itself

# EBADF - guard against macOS `stat` throwing EBADF
_IGNORED_ERRNOS = (ENOENT, ENOTDIR, EBADF, ELOOP)

_IGNORED_WINERRORS = (
    _WINERROR_NOT_READY,
    _WINERROR_INVALID_NAME,
    _WINERROR_CANT_RESOLVE_FILENAME)

def _ignore_error(exception):
    return (getattr(exception, 'errno', None) in _IGNORED_ERRNOS or
            getattr(exception, 'winerror', None) in _IGNORED_WINERRORS)


def _is_wildcard_pattern(pat):
    # Whether this pattern needs actual matching using fnmatch, or can
    # be looked up directly as a file.
    return "*" in pat or "?" in pat or "[" in pat


class _Flavour(object):
    """A flavour implements a particular (platform-specific) set of path
    semantics."""

    def __init__(self):
        self.join = self.sep.join

    def parse_parts(self, parts):
        parsed = []
        sep = self.sep
        altsep = self.altsep
        drv = root = ''
        it = reversed(parts)
        for part in it:
            if not part:
                continue
            if altsep:
                part = part.replace(altsep, sep)
            drv, root, rel = self.splitroot(part)
            if sep in rel:
                for x in reversed(rel.split(sep)):
                    if x and x != '.':
                        parsed.append(sys.intern(x))
            else:
                if rel and rel != '.':
                    parsed.append(sys.intern(rel))
            if drv or root:
                if not drv:
                    # If no drive is present, try to find one in the previous
                    # parts. This makes the result of parsing e.g.
                    # ("C:", "/", "a") reasonably intuitive.
                    for part in it:
                        if not part:
                            continue
                        if altsep:
                            part = part.replace(altsep, sep)
                        drv = self.splitroot(part)[0]
                        if drv:
                            break
                break
        if drv or root:
            parsed.append(drv + root)
        parsed.reverse()
        return drv, root, parsed

    def join_parsed_parts(self, drv, root, parts, drv2, root2, parts2):
        """
        Join the two paths represented by the respective
        (drive, root, parts) tuples.  Return a new (drive, root, parts) tuple.
        """
        if root2:
            if not drv2 and drv:
                return drv, root2, [drv + root2] + parts2[1:]
        elif drv2:
            if drv2 == drv or self.casefold(drv2) == self.casefold(drv):
                # Same drive => second path is relative to the first
                return drv, root, parts + parts2[1:]
        else:
            # Second path is non-anchored (common case)
            return drv, root, parts + parts2
        return drv2, root2, parts2


class _WindowsFlavour(_Flavour):
    # Reference for Windows paths can be found at
    # http://msdn.microsoft.com/en-us/library/aa365247%28v=vs.85%29.aspx

    sep = '\\'
    altsep = '/'
    has_drv = True
    pathmod = ntpath

    is_supported = (os.name == 'nt')

    drive_letters = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
    ext_namespace_prefix = '\\\\?\\'

    reserved_names = (
        {'CON', 'PRN', 'AUX', 'NUL', 'CONIN$', 'CONOUT$'} |
        {'COM%s' % c for c in '123456789\xb9\xb2\xb3'} |
        {'LPT%s' % c for c in '123456789\xb9\xb2\xb3'}
        )

    # Interesting findings about extended paths:
    # * '\\?\c:\a' is an extended path, which bypasses normal Windows API
    #   path processing. Thus relative paths are not resolved and slash is not
    #   translated to backslash. It has the native NT path limit of 32767
    #   characters, but a bit less after resolving device symbolic links,
    #   such as '\??\C:' => '\Device\HarddiskVolume2'.
    # * '\\?\c:/a' looks for a device named 'C:/a' because slash is a
    #   regular name character in the object namespace.
    # * '\\?\c:\foo/bar' is invalid because '/' is illegal in NT filesystems.
    #   The only path separator at the filesystem level is backslash.
    # * '//?/c:\a' and '//?/c:/a' are effectively equivalent to '\\.\c:\a' and
    #   thus limited to MAX_PATH.
    # * Prior to Windows 8, ANSI API bytes paths are limited to MAX_PATH,
    #   even with the '\\?\' prefix.

    def splitroot(self, part, sep=sep):
        first = part[0:1]
        second = part[1:2]
        if (second == sep and first == sep):
            # XXX extended paths should also disable the collapsing of "."
            # components (according to MSDN docs).
            prefix, part = self._split_extended_path(part)
            first = part[0:1]
            second = part[1:2]
        else:
            prefix = ''
        third = part[2:3]
        if (second == sep and first == sep and third != sep):
            # is a UNC path:
            # vvvvvvvvvvvvvvvvvvvvv root
            # \\machine\mountpoint\directory\etc\...
            #            directory ^^^^^^^^^^^^^^
            index = part.find(sep, 2)
            if index != -1:
                index2 = part.find(sep, index + 1)
                # a UNC path can't have two slashes in a row
                # (after the initial two)
                if index2 != index + 1:
                    if index2 == -1:
                        index2 = len(part)
                    if prefix:
                        return prefix + part[1:index2], sep, part[index2+1:]
                    else:
                        return part[:index2], sep, part[index2+1:]
        drv = root = ''
        if second == ':' and first in self.drive_letters:
            drv = part[:2]
            part = part[2:]
            first = third
        if first == sep:
            root = first
            part = part.lstrip(sep)
        return prefix + drv, root, part

    def casefold(self, s):
        return s.lower()

    def casefold_parts(self, parts):
        return [p.lower() for p in parts]

    def compile_pattern(self, pattern):
        return re.compile(fnmatch.translate(pattern), re.IGNORECASE).fullmatch

    def _split_extended_path(self, s, ext_prefix=ext_namespace_prefix):
        prefix = ''
        if s.startswith(ext_prefix):
            prefix = s[:4]
            s = s[4:]
            if s.startswith('UNC\\'):
                prefix += s[:3]
                s = '\\' + s[3:]
        return prefix, s

    def is_reserved(self, parts):
        # NOTE: the rules for reserved names seem somewhat complicated
        # (e.g. r"..\NUL" is reserved but not r"foo\NUL" if "foo" does not
        # exist). We err on the side of caution and return True for paths
        # which are not considered reserved by Windows.
        if not parts:
            return False
        if parts[0].startswith('\\\\'):
            # UNC paths are never reserved
            return False
        name = parts[-1].partition('.')[0].partition(':')[0].rstrip(' ')
        return name.upper() in self.reserved_names

    def make_uri(self, path):
        # Under Windows, file URIs use the UTF-8 encoding.
        drive = path.drive
        if len(drive) == 2 and drive[1] == ':':
            # It's a path on a local drive => 'file:///c:/a/b'
            rest = path.as_posix()[2:].lstrip('/')
            return 'file:///%s/%s' % (
                drive, urlquote_from_bytes(rest.encode('utf-8')))
        else:
            # It's a path on a network drive => 'file://host/share/a/b'
            return 'file:' + urlquote_from_bytes(path.as_posix().encode('utf-8'))


class _PosixFlavour(_Flavour):
    sep = '/'
    altsep = ''
    has_drv = False
    pathmod = posixpath

    is_supported = (os.name != 'nt')

    def splitroot(self, part, sep=sep):
        if part and part[0] == sep:
            stripped_part = part.lstrip(sep)
            # According to POSIX path resolution:
            # http://pubs.opengroup.org/onlinepubs/009695399/basedefs/xbd_chap04.html#tag_04_11
            # "A pathname that begins with two successive slashes may be
            # interpreted in an implementation-defined manner, although more
            # than two leading slashes shall be treated as a single slash".
            if len(part) - len(stripped_part) == 2:
                return '', sep * 2, stripped_part
            else:
                return '', sep, stripped_part
        else:
            return '', '', part

    def casefold(self, s):
        return s

    def casefold_parts(self, parts):
        return parts

    def compile_pattern(self, pattern):
        return re.compile(fnmatch.translate(pattern)).fullmatch

    def is_reserved(self, parts):
        return False

    def make_uri(self, path):
        # We represent the path using the local filesystem encoding,
        # for portability to other applications.
        bpath = bytes(path)
        return 'file://' + urlquote_from_bytes(bpath)


_windows_flavour = _WindowsFlavour()
_posix_flavour = _PosixFlavour()


#
# Globbing helpers
#

def _make_selector(pattern_parts, flavour):
    pat = pattern_parts[0]
    child_parts = pattern_parts[1:]
    if not pat:
        return _TerminatingSelector()
    if pat == '**':
        cls = _RecursiveWildcardSelector
    elif '**' in pat:
        raise ValueError("Invalid pattern: '**' can only be an entire path component")
    elif _is_wildcard_pattern(pat):
        cls = _WildcardSelector
    else:
        cls = _PreciseSelector
    return cls(pat, child_parts, flavour)

if hasattr(functools, "lru_cache"):
    _make_selector = functools.lru_cache()(_make_selector)


class _Selector:
    """A selector matches a specific glob pattern part against the children
    of a given path."""

    def __init__(self, child_parts, flavour):
        self.child_parts = child_parts
        if child_parts:
            self.successor = _make_selector(child_parts, flavour)
            self.dironly = True
        else:
            self.successor = _TerminatingSelector()
            self.dironly = False

    def select_from(self, parent_path):
        """Iterate over all child paths of `parent_path` matched by this
        selector.  This can contain parent_path itself."""
        path_cls = type(parent_path)
        is_dir = path_cls.is_dir
        exists = path_cls.exists
        scandir = path_cls._scandir
        if not is_dir(parent_path):
            return iter([])
        return self._select_from(parent_path, is_dir, exists, scandir)


class _TerminatingSelector:

    def _select_from(self, parent_path, is_dir, exists, scandir):
        yield parent_path


class _PreciseSelector(_Selector):

    def __init__(self, name, child_parts, flavour):
        self.name = name
        _Selector.__init__(self, child_parts, flavour)

    def _select_from(self, parent_path, is_dir, exists, scandir):
        try:
            path = parent_path._make_child_relpath(self.name)
            if (is_dir if self.dironly else exists)(path):
                for p in self.successor._select_from(path, is_dir, exists, scandir):
                    yield p
        except PermissionError:
            return


class _WildcardSelector(_Selector):

    def __init__(self, pat, child_parts, flavour):
        self.match = flavour.compile_pattern(pat)
        _Selector.__init__(self, child_parts, flavour)

    def _select_from(self, parent_path, is_dir, exists, scandir):
        try:
            with scandir(parent_path) as scandir_it:
                entries = list(scandir_it)
            for entry in entries:
                if self.dironly:
                    try:
                        # "entry.is_dir()" can raise PermissionError
                        # in some cases (see bpo-38894), which is not
                        # among the errors ignored by _ignore_error()
                        if not entry.is_dir():
                            continue
                    except OSError as e:
                        if not _ignore_error(e):
                            raise
                        continue
                name = entry.name
                if self.match(name):
                    path = parent_path._make_child_relpath(name)
                    for p in self.successor._select_from(path, is_dir, exists, scandir):
                        yield p
        except PermissionError:
            return


class _RecursiveWildcardSelector(_Selector):

    def __init__(self, pat, child_parts, flavour):
        _Selector.__init__(self, child_parts, flavour)

    def _iterate_directories(self, parent_path, is_dir, scandir):
        yield parent_path
        try:
            with scandir(parent_path) as scandir_it:
                entries = list(scandir_it)
            for entry in entries:
                entry_is_dir = False
                try:
                    entry_is_dir = entry.is_dir(follow_symlinks=False)
                except OSError as e:
                    if not _ignore_error(e):
                        raise
                if entry_is_dir:
                    path = parent_path._make_child_relpath(entry.name)
                    for p in self._iterate_directories(path, is_dir, scandir):
                        yield p
        except PermissionError:
            return

    def _select_from(self, parent_path, is_dir, exists, scandir):
        try:
            yielded = set()
            try:
                successor_select = self.successor._select_from
                for starting_point in self._iterate_directories(parent_path, is_dir, scandir):
                    for p in successor_select(starting_point, is_dir, exists, scandir):
                        if p not in yielded:
                            yield p
                            yielded.add(p)
            finally:
                yielded.clear()
        except PermissionError:
            return


#
# Public API
#

class _PathParents(Sequence):
    """This object provides sequence-like access to the logical ancestors
    of a path.  Don't try to construct it yourself."""
    __slots__ = ('_pathcls', '_drv', '_root', '_parts')

    def __init__(self, path):
        # We don't store the instance to avoid reference cycles
        self._pathcls = type(path)
        self._drv = path._drv
        self._root = path._root
        self._parts = path._parts

    def __len__(self):
        if self._drv or self._root:
            return len(self._parts) - 1
        else:
            return len(self._parts)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return tuple(self[i] for i in range(*idx.indices(len(self))))

        if idx >= len(self) or idx < -len(self):
            raise IndexError(idx)
        if idx < 0:
            idx += len(self)
        return self._pathcls._from_parsed_parts(self._drv, self._root,
                                                self._parts[:-idx - 1])

    def __repr__(self):
        return "<{}.parents>".format(self._pathcls.__name__)


class PurePath(object):
    """Base class for manipulating paths without I/O.

    PurePath represents a filesystem path and offers operations which
    don't imply any actual filesystem I/O.  Depending on your system,
    instantiating a PurePath will return either a PurePosixPath or a
    PureWindowsPath object.  You can also instantiate either of these classes
    directly, regardless of your system.
    """
    __slots__ = (
        '_drv', '_root', '_parts',
        '_str', '_hash', '_pparts', '_cached_cparts',
    )

    def __new__(cls, *args):
        """Construct a PurePath from one or several strings and or existing
        PurePath objects.  The strings and path objects are combined so as
        to yield a canonicalized path, which is incorporated into the
        new PurePath object.
        """
        if cls is PurePath:
            cls = PureWindowsPath if os.name == 'nt' else PurePosixPath
        return cls._from_parts(args)

    def __reduce__(self):
        # Using the parts tuple helps share interned path parts
        # when pickling related paths.
        return (self.__class__, tuple(self._parts))

    @classmethod
    def _parse_args(cls, args):
        # This is useful when you don't want to create an instance, just
        # canonicalize some constructor arguments.
        parts = []
        for a in args:
            if isinstance(a, PurePath):
                parts += a._parts
            else:
                a = os.fspath(a)
                if isinstance(a, str):
                    # Force-cast str subclasses to str (issue #21127)
                    parts.append(str(a))
                else:
                    raise TypeError(
                        "argument should be a str object or an os.PathLike "
                        "object returning str, not %r"
                        % type(a))
        return cls._flavour.parse_parts(parts)

    @classmethod
    def _from_parts(cls, args):
        # We need to call _parse_args on the instance, so as to get the
        # right flavour.
        self = object.__new__(cls)
        drv, root, parts = self._parse_args(args)
        self._drv = drv
        self._root = root
        self._parts = parts
        return self

    @classmethod
    def _from_parsed_parts(cls, drv, root, parts):
        self = object.__new__(cls)
        self._drv = drv
        self._root = root
        self._parts = parts
        return self

    @classmethod
    def _format_parsed_parts(cls, drv, root, parts):
        if drv or root:
            return drv + root + cls._flavour.join(parts[1:])
        else:
            return cls._flavour.join(parts)

    def _make_child(self, args):
        drv, root, parts = self._parse_args(args)
        drv, root, parts = self._flavour.join_parsed_parts(
            self._drv, self._root, self._parts, drv, root, parts)
        return self._from_parsed_parts(drv, root, parts)

    def __str__(self):
        """Return the string representation of the path, suitable for
        passing to system calls."""
        try:
            return self._str
        except AttributeError:
            self._str = self._format_parsed_parts(self._drv, self._root,
                                                  self._parts) or '.'
            return self._str

    def __fspath__(self):
        return str(self)

    def as_posix(self):
        """Return the string representation of the path with forward (/)
        slashes."""
        f = self._flavour
        return str(self).replace(f.sep, '/')

    def __bytes__(self):
        """Return the bytes representation of the path.  This is only
        recommended to use under Unix."""
        return os.fsencode(self)

    def __repr__(self):
        return "{}({!r})".format(self.__class__.__name__, self.as_posix())

    def as_uri(self):
        """Return the path as a 'file' URI."""
        if not self.is_absolute():
            raise ValueError("relative path can't be expressed as a file URI")
        return self._flavour.make_uri(self)

    @property
    def _cparts(self):
        # Cached casefolded parts, for hashing and comparison
        try:
            return self._cached_cparts
        except AttributeError:
            self._cached_cparts = self._flavour.casefold_parts(self._parts)
            return self._cached_cparts

    def __eq__(self, other):
        if not isinstance(other, PurePath):
            return NotImplemented
        return self._cparts == other._cparts and self._flavour is other._flavour

    def __hash__(self):
        try:
            return self._hash
        except AttributeError:
            self._hash = hash(tuple(self._cparts))
            return self._hash

    def __lt__(self, other):
        if not isinstance(other, PurePath) or self._flavour is not other._flavour:
            return NotImplemented
        return self._cparts < other._cparts

    def __le__(self, other):
        if not isinstance(other, PurePath) or self._flavour is not other._flavour:
            return NotImplemented
        return self._cparts <= other._cparts

    def __gt__(self, other):
        if not isinstance(other, PurePath) or self._flavour is not other._flavour:
            return NotImplemented
        return self._cparts > other._cparts

    def __ge__(self, other):
        if not isinstance(other, PurePath) or self._flavour is not other._flavour:
            return NotImplemented
        return self._cparts >= other._cparts

    drive = property(attrgetter('_drv'),
                     doc="""The drive prefix (letter or UNC path), if any.""")

    root = property(attrgetter('_root'),
                    doc="""The root of the path, if any.""")

    @property
    def anchor(self):
        """The concatenation of the drive and root, or ''."""
        anchor = self._drv + self._root
        return anchor

    @property
    def name(self):
        """The final path component, if any."""
        parts = self._parts
        if len(parts) == (1 if (self._drv or self._root) else 0):
            return ''
        return parts[-1]

    @property
    def suffix(self):
        """
        The final component's last suffix, if any.

        This includes the leading period. For example: '.txt'
        """
        name = self.name
        i = name.rfind('.')
        if 0 < i < len(name) - 1:
            return name[i:]
        else:
            return ''

    @property
    def suffixes(self):
        """
        A list of the final component's suffixes, if any.

        These include the leading periods. For example: ['.tar', '.gz']
        """
        name = self.name
        if name.endswith('.'):
            return []
        name = name.lstrip('.')
        return ['.' + suffix for suffix in name.split('.')[1:]]

    @property
    def stem(self):
        """The final path component, minus its last suffix."""
        name = self.name
        i = name.rfind('.')
        if 0 < i < len(name) - 1:
            return name[:i]
        else:
            return name

    def with_name(self, name):
        """Return a new path with the file name changed."""
        if not self.name:
            raise ValueError("%r has an empty name" % (self,))
        drv, root, parts = self._flavour.parse_parts((name,))
        if (not name or name[-1] in [self._flavour.sep, self._flavour.altsep]
            or drv or root or len(parts) != 1):
            raise ValueError("Invalid name %r" % (name))
        return self._from_parsed_parts(self._drv, self._root,
                                       self._parts[:-1] + [name])

    def with_stem(self, stem):
        """Return a new path with the stem changed."""
        return self.with_name(stem + self.suffix)

    def with_suffix(self, suffix):
        """Return a new path with the file suffix changed.  If the path
        has no suffix, add given suffix.  If the given suffix is an empty
        string, remove the suffix from the path.
        """
        f = self._flavour
        if f.sep in suffix or f.altsep and f.altsep in suffix:
            raise ValueError("Invalid suffix %r" % (suffix,))
        if suffix and not suffix.startswith('.') or suffix == '.':
            raise ValueError("Invalid suffix %r" % (suffix))
        name = self.name
        if not name:
            raise ValueError("%r has an empty name" % (self,))
        old_suffix = self.suffix
        if not old_suffix:
            name = name + suffix
        else:
            name = name[:-len(old_suffix)] + suffix
        return self._from_parsed_parts(self._drv, self._root,
                                       self._parts[:-1] + [name])

    def relative_to(self, *other):
        """Return the relative path to another path identified by the passed
        arguments.  If the operation is not possible (because this is not
        a subpath of the other path), raise ValueError.
        """
        # For the purpose of this method, drive and root are considered
        # separate parts, i.e.:
        #   Path('c:/').relative_to('c:')  gives Path('/')
        #   Path('c:/').relative_to('/')   raise ValueError
        if not other:
            raise TypeError("need at least one argument")
        parts = self._parts
        drv = self._drv
        root = self._root
        if root:
            abs_parts = [drv, root] + parts[1:]
        else:
            abs_parts = parts
        to_drv, to_root, to_parts = self._parse_args(other)
        if to_root:
            to_abs_parts = [to_drv, to_root] + to_parts[1:]
        else:
            to_abs_parts = to_parts
        n = len(to_abs_parts)
        cf = self._flavour.casefold_parts
        if (root or drv) if n == 0 else cf(abs_parts[:n]) != cf(to_abs_parts):
            formatted = self._format_parsed_parts(to_drv, to_root, to_parts)
            raise ValueError("{!r} is not in the subpath of {!r}"
                    " OR one path is relative and the other is absolute."
                             .format(str(self), str(formatted)))
        return self._from_parsed_parts('', root if n == 1 else '',
                                       abs_parts[n:])

    def is_relative_to(self, *other):
        """Return True if the path is relative to another path or False.
        """
        try:
            self.relative_to(*other)
            return True
        except ValueError:
            return False

    @property
    def parts(self):
        """An object providing sequence-like access to the
        components in the filesystem path."""
        # We cache the tuple to avoid building a new one each time .parts
        # is accessed.  XXX is this necessary?
        try:
            return self._pparts
        except AttributeError:
            self._pparts = tuple(self._parts)
            return self._pparts

    def joinpath(self, *args):
        """Combine this path with one or several arguments, and return a
        new path representing either a subpath (if all arguments are relative
        paths) or a totally different path (if one of the arguments is
        anchored).
        """
        return self._make_child(args)

    def __truediv__(self, key):
        try:
            return self._make_child((key,))
        except TypeError:
            return NotImplemented

    def __rtruediv__(self, key):
        try:
            return self._from_parts([key] + self._parts)
        except TypeError:
            return NotImplemented

    @property
    def parent(self):
        """The logical parent of the path."""
        drv = self._drv
        root = self._root
        parts = self._parts
        if len(parts) == 1 and (drv or root):
            return self
        return self._from_parsed_parts(drv, root, parts[:-1])

    @property
    def parents(self):
        """A sequence of this path's logical parents."""
        return _PathParents(self)

    def is_absolute(self):
        """True if the path is absolute (has both a root and, if applicable,
        a drive)."""
        if not self._root:
            return False
        return not self._flavour.has_drv or bool(self._drv)

    def is_reserved(self):
        """Return True if the path contains one of the special names reserved
        by the system, if any."""
        return self._flavour.is_reserved(self._parts)

    def match(self, path_pattern):
        """
        Return True if this path matches the given pattern.
        """
        cf = self._flavour.casefold
        path_pattern = cf(path_pattern)
        drv, root, pat_parts = self._flavour.parse_parts((path_pattern,))
        if not pat_parts:
            raise ValueError("empty pattern")
        if drv and drv != cf(self._drv):
            return False
        if root and root != cf(self._root):
            return False
        parts = self._cparts
        if drv or root:
            if len(pat_parts) != len(parts):
                return False
            pat_parts = pat_parts[1:]
        elif len(pat_parts) > len(parts):
            return False
        for part, pat in zip(reversed(parts), reversed(pat_parts)):
            if not fnmatch.fnmatchcase(part, pat):
                return False
        return True

# Can't subclass os.PathLike from PurePath and keep the constructor
# optimizations in PurePath._parse_args().
os.PathLike.register(PurePath)


class PurePosixPath(PurePath):
    """PurePath subclass for non-Windows systems.

    On a POSIX system, instantiating a PurePath should return this object.
    However, you can also instantiate it directly on any system.
    """
    _flavour = _posix_flavour
    __slots__ = ()


class PureWindowsPath(PurePath):
    """PurePath subclass for Windows systems.

    On a Windows system, instantiating a PurePath should return this object.
    However, you can also instantiate it directly on any system.
    """
    _flavour = _windows_flavour
    __slots__ = ()


# Filesystem-accessing classes


class Path(PurePath):
    """PurePath subclass that can make system calls.

    Path represents a filesystem path but unlike PurePath, also offers
    methods to do system calls on path objects. Depending on your system,
    instantiating a Path will return either a PosixPath or a WindowsPath
    object. You can also instantiate a PosixPath or WindowsPath directly,
    but cannot instantiate a WindowsPath on a POSIX system or vice versa.
    """
    __slots__ = ()

    def __new__(cls, *args, **kwargs):
        if cls is Path:
            cls = WindowsPath if os.name == 'nt' else PosixPath
        self = cls._from_parts(args)
        if not self._flavour.is_supported:
            raise NotImplementedError("cannot instantiate %r on your system"
                                      % (cls.__name__,))
        return self

    def _make_child_relpath(self, part):
        # This is an optimization used for dir walking.  `part` must be
        # a single part relative to this path.
        parts = self._parts + [part]
        return self._from_parsed_parts(self._drv, self._root, parts)

    def __enter__(self):
        # In previous versions of pathlib, __exit__() marked this path as
        # closed; subsequent attempts to perform I/O would raise an IOError.
        # This functionality was never documented, and had the effect of
        # making Path objects mutable, contrary to PEP 428.
        # In Python 3.9 __exit__() was made a no-op.
        # In Python 3.11 __enter__() began emitting DeprecationWarning.
        # In Python 3.13 __enter__() and __exit__() should be removed.
        warnings.warn("pathlib.Path.__enter__() is deprecated and scheduled "
                      "for removal in Python 3.13; Path objects as a context "
                      "manager is a no-op",
                      DeprecationWarning, stacklevel=2)
        return self

    def __exit__(self, t, v, tb):
        pass

    # Public API

    @classmethod
    def cwd(cls):
        """Return a new path pointing to the current working directory
        (as returned by os.getcwd()).
        """
        return cls(os.getcwd())

    @classmethod
    def home(cls):
        """Return a new path pointing to the user's home directory (as
        returned by os.path.expanduser('~')).
        """
        return cls("~").expanduser()

    def samefile(self, other_path):
        """Return whether other_path is the same or not as this file
        (as returned by os.path.samefile()).
        """
        st = self.stat()
        try:
            other_st = other_path.stat()
        except AttributeError:
            other_st = self.__class__(other_path).stat()
        return os.path.samestat(st, other_st)

    def iterdir(self):
        """Iterate over the files in this directory.  Does not yield any
        result for the special paths '.' and '..'.
        """
        for name in os.listdir(self):
            yield self._make_child_relpath(name)

    def _scandir(self):
        # bpo-24132: a future version of pathlib will support subclassing of
        # pathlib.Path to customize how the filesystem is accessed. This
        # includes scandir(), which is used to implement glob().
        return os.scandir(self)

    def glob(self, pattern):
        """Iterate over this subtree and yield all existing files (of any
        kind, including directories) matching the given relative pattern.
        """
        sys.audit("pathlib.Path.glob", self, pattern)
        if not pattern:
            raise ValueError("Unacceptable pattern: {!r}".format(pattern))
        drv, root, pattern_parts = self._flavour.parse_parts((pattern,))
        if drv or root:
            raise NotImplementedError("Non-relative patterns are unsupported")
        if pattern[-1] in (self._flavour.sep, self._flavour.altsep):
            pattern_parts.append('')
        selector = _make_selector(tuple(pattern_parts), self._flavour)
        for p in selector.select_from(self):
            yield p

    def rglob(self, pattern):
        """Recursively yield all existing files (of any kind, including
        directories) matching the given relative pattern, anywhere in
        this subtree.
        """
        sys.audit("pathlib.Path.rglob", self, pattern)
        drv, root, pattern_parts = self._flavour.parse_parts((pattern,))
        if drv or root:
            raise NotImplementedError("Non-relative patterns are unsupported")
        if pattern and pattern[-1] in (self._flavour.sep, self._flavour.altsep):
            pattern_parts.append('')
        selector = _make_selector(("**",) + tuple(pattern_parts), self._flavour)
        for p in selector.select_from(self):
            yield p

    def absolute(self):
        """Return an absolute version of this path by prepending the current
        working directory. No normalization or symlink resolution is performed.

        Use resolve() to get the canonical path to a file.
        """
        if self.is_absolute():
            return self
        return self._from_parts([self.cwd()] + self._parts)

    def resolve(self, strict=False):
        """
        Make the path absolute, resolving all symlinks on the way and also
        normalizing it.
        """

        def check_eloop(e):
            winerror = getattr(e, 'winerror', 0)
            if e.errno == ELOOP or winerror == _WINERROR_CANT_RESOLVE_FILENAME:
                raise RuntimeError("Symlink loop from %r" % e.filename)

        try:
            s = os.path.realpath(self, strict=strict)
        except OSError as e:
            check_eloop(e)
            raise
        p = self._from_parts((s,))

        # In non-strict mode, realpath() doesn't raise on symlink loops.
        # Ensure we get an exception by calling stat()
        if not strict:
            try:
                p.stat()
            except OSError as e:
                check_eloop(e)
        return p

    def stat(self, *, follow_symlinks=True):
        """
        Return the result of the stat() system call on this path, like
        os.stat() does.
        """
        return os.stat(self, follow_symlinks=follow_symlinks)

    def owner(self):
        """
        Return the login name of the file owner.
        """
        try:
            import pwd
            return pwd.getpwuid(self.stat().st_uid).pw_name
        except ImportError:
            raise NotImplementedError("Path.owner() is unsupported on this system")

    def group(self):
        """
        Return the group name of the file gid.
        """

        try:
            import grp
            return grp.getgrgid(self.stat().st_gid).gr_name
        except ImportError:
            raise NotImplementedError("Path.group() is unsupported on this system")

    def open(self, mode='r', buffering=-1, encoding=None,
             errors=None, newline=None):
        """
        Open the file pointed by this path and return a file object, as
        the built-in open() function does.
        """
        if "b" not in mode:
            encoding = io.text_encoding(encoding)
        return io.open(self, mode, buffering, encoding, errors, newline)

    def read_bytes(self):
        """
        Open the file in bytes mode, read it, and close the file.
        """
        with self.open(mode='rb') as f:
            return f.read()

    def read_text(self, encoding=None, errors=None):
        """
        Open the file in text mode, read it, and close the file.
        """
        encoding = io.text_encoding(encoding)
        with self.open(mode='r', encoding=encoding, errors=errors) as f:
            return f.read()

    def write_bytes(self, data):
        """
        Open the file in bytes mode, write to it, and close the file.
        """
        # type-check for the buffer interface before truncating the file
        view = memoryview(data)
        with self.open(mode='wb') as f:
            return f.write(view)

    def write_text(self, data, encoding=None, errors=None, newline=None):
        """
        Open the file in text mode, write to it, and close the file.
        """
        if not isinstance(data, str):
            raise TypeError('data must be str, not %s' %
                            data.__class__.__name__)
        encoding = io.text_encoding(encoding)
        with self.open(mode='w', encoding=encoding, errors=errors, newline=newline) as f:
            return f.write(data)

    def readlink(self):
        """
        Return the path to which the symbolic link points.
        """
        if not hasattr(os, "readlink"):
            raise NotImplementedError("os.readlink() not available on this system")
        return self._from_parts((os.readlink(self),))

    def touch(self, mode=0o666, exist_ok=True):
        """
        Create this file with the given access mode, if it doesn't exist.
        """

        if exist_ok:
            # First try to bump modification time
            # Implementation note: GNU touch uses the UTIME_NOW option of
            # the utimensat() / futimens() functions.
            try:
                os.utime(self, None)
            except OSError:
                # Avoid exception chaining
                pass
            else:
                return
        flags = os.O_CREAT | os.O_WRONLY
        if not exist_ok:
            flags |= os.O_EXCL
        fd = os.open(self, flags, mode)
        os.close(fd)

    def mkdir(self, mode=0o777, parents=False, exist_ok=False):
        """
        Create a new directory at this given path.
        """
        try:
            os.mkdir(self, mode)
        except FileNotFoundError:
            if not parents or self.parent == self:
                raise
            self.parent.mkdir(parents=True, exist_ok=True)
            self.mkdir(mode, parents=False, exist_ok=exist_ok)
        except OSError:
            # Cannot rely on checking for EEXIST, since the operating system
            # could give priority to other errors like EACCES or EROFS
            if not exist_ok or not self.is_dir():
                raise

    def chmod(self, mode, *, follow_symlinks=True):
        """
        Change the permissions of the path, like os.chmod().
        """
        os.chmod(self, mode, follow_symlinks=follow_symlinks)

    def lchmod(self, mode):
        """
        Like chmod(), except if the path points to a symlink, the symlink's
        permissions are changed, rather than its target's.
        """
        self.chmod(mode, follow_symlinks=False)

    def unlink(self, missing_ok=False):
        """
        Remove this file or link.
        If the path is a directory, use rmdir() instead.
        """
        try:
            os.unlink(self)
        except FileNotFoundError:
            if not missing_ok:
                raise

    def rmdir(self):
        """
        Remove this directory.  The directory must be empty.
        """
        os.rmdir(self)

    def lstat(self):
        """
        Like stat(), except if the path points to a symlink, the symlink's
        status information is returned, rather than its target's.
        """
        return self.stat(follow_symlinks=False)

    def rename(self, target):
        """
        Rename this path to the target path.

        The target path may be absolute or relative. Relative paths are
        interpreted relative to the current working directory, *not* the
        directory of the Path object.

        Returns the new Path instance pointing to the target path.
        """
        os.rename(self, target)
        return self.__class__(target)

    def replace(self, target):
        """
        Rename this path to the target path, overwriting if that path exists.

        The target path may be absolute or relative. Relative paths are
        interpreted relative to the current working directory, *not* the
        directory of the Path object.

        Returns the new Path instance pointing to the target path.
        """
        os.replace(self, target)
        return self.__class__(target)

    def symlink_to(self, target, target_is_directory=False):
        """
        Make this path a symlink pointing to the target path.
        Note the order of arguments (link, target) is the reverse of os.symlink.
        """
        if not hasattr(os, "symlink"):
            raise NotImplementedError("os.symlink() not available on this system")
        os.symlink(target, self, target_is_directory)

    def hardlink_to(self, target):
        """
        Make this path a hard link pointing to the same file as *target*.

        Note the order of arguments (self, target) is the reverse of os.link's.
        """
        if not hasattr(os, "link"):
            raise NotImplementedError("os.link() not available on this system")
        os.link(target, self)

    def link_to(self, target):
        """
        Make the target path a hard link pointing to this path.

        Note this function does not make this path a hard link to *target*,
        despite the implication of the function and argument names. The order
        of arguments (target, link) is the reverse of Path.symlink_to, but
        matches that of os.link.

        Deprecated since Python 3.10 and scheduled for removal in Python 3.12.
        Use `hardlink_to()` instead.
        """
        warnings.warn("pathlib.Path.link_to() is deprecated and is scheduled "
                      "for removal in Python 3.12. "
                      "Use pathlib.Path.hardlink_to() instead.",
                      DeprecationWarning, stacklevel=2)
        self.__class__(target).hardlink_to(self)

    # Convenience functions for querying the stat results

    def exists(self):
        """
        Whether this path exists.
        """
        try:
            self.stat()
        except OSError as e:
            if not _ignore_error(e):
                raise
            return False
        except ValueError:
            # Non-encodable path
            return False
        return True

    def is_dir(self):
        """
        Whether this path is a directory.
        """
        try:
            return S_ISDIR(self.stat().st_mode)
        except OSError as e:
            if not _ignore_error(e):
                raise
            # Path doesn't exist or is a broken symlink
            # (see http://web.archive.org/web/20200623061726/https://bitbucket.org/pitrou/pathlib/issues/12/ )
            return False
        except ValueError:
            # Non-encodable path
            return False

    def is_file(self):
        """
        Whether this path is a regular file (also True for symlinks pointing
        to regular files).
        """
        try:
            return S_ISREG(self.stat().st_mode)
        except OSError as e:
            if not _ignore_error(e):
                raise
            # Path doesn't exist or is a broken symlink
            # (see http://web.archive.org/web/20200623061726/https://bitbucket.org/pitrou/pathlib/issues/12/ )
            return False
        except ValueError:
            # Non-encodable path
            return False

    def is_mount(self):
        """
        Check if this path is a POSIX mount point
        """
        # Need to exist and be a dir
        if not self.exists() or not self.is_dir():
            return False

        try:
            parent_dev = self.parent.stat().st_dev
        except OSError:
            return False

        dev = self.stat().st_dev
        if dev != parent_dev:
            return True
        ino = self.stat().st_ino
        parent_ino = self.parent.stat().st_ino
        return ino == parent_ino

    def is_symlink(self):
        """
        Whether this path is a symbolic link.
        """
        try:
            return S_ISLNK(self.lstat().st_mode)
        except OSError as e:
            if not _ignore_error(e):
                raise
            # Path doesn't exist
            return False
        except ValueError:
            # Non-encodable path
            return False

    def is_block_device(self):
        """
        Whether this path is a block device.
        """
        try:
            return S_ISBLK(self.stat().st_mode)
        except OSError as e:
            if not _ignore_error(e):
                raise
            # Path doesn't exist or is a broken symlink
            # (see http://web.archive.org/web/20200623061726/https://bitbucket.org/pitrou/pathlib/issues/12/ )
            return False
        except ValueError:
            # Non-encodable path
            return False

    def is_char_device(self):
        """
        Whether this path is a character device.
        """
        try:
            return S_ISCHR(self.stat().st_mode)
        except OSError as e:
            if not _ignore_error(e):
                raise
            # Path doesn't exist or is a broken symlink
            # (see http://web.archive.org/web/20200623061726/https://bitbucket.org/pitrou/pathlib/issues/12/ )
            return False
        except ValueError:
            # Non-encodable path
            return False

    def is_fifo(self):
        """
        Whether this path is a FIFO.
        """
        try:
            return S_ISFIFO(self.stat().st_mode)
        except OSError as e:
            if not _ignore_error(e):
                raise
            # Path doesn't exist or is a broken symlink
            # (see http://web.archive.org/web/20200623061726/https://bitbucket.org/pitrou/pathlib/issues/12/ )
            return False
        except ValueError:
            # Non-encodable path
            return False

    def is_socket(self):
        """
        Whether this path is a socket.
        """
        try:
            return S_ISSOCK(self.stat().st_mode)
        except OSError as e:
            if not _ignore_error(e):
                raise
            # Path doesn't exist or is a broken symlink
            # (see http://web.archive.org/web/20200623061726/https://bitbucket.org/pitrou/pathlib/issues/12/ )
            return False
        except ValueError:
            # Non-encodable path
            return False

    def expanduser(self):
        """ Return a new path with expanded ~ and ~user constructs
        (as returned by os.path.expanduser)
        """
        if (not (self._drv or self._root) and
            self._parts and self._parts[0][:1] == '~'):
            homedir = os.path.expanduser(self._parts[0])
            if homedir[:1] == "~":
                raise RuntimeError("Could not determine home directory.")
            return self._from_parts([homedir] + self._parts[1:])

        return self


class PosixPath(Path, PurePosixPath):
    """Path subclass for non-Windows systems.

    On a POSIX system, instantiating a Path should return this object.
    """
    __slots__ = ()

class WindowsPath(Path, PureWindowsPath):
    """Path subclass for Windows systems.

    On a Windows system, instantiating a Path should return this object.
    """
    __slots__ = ()

    def is_mount(self):
        raise NotImplementedError("Path.is_mount() is unsupported on this system")
