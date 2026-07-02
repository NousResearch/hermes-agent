# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.
"""
.. testsetup::

    from packaging.version import parse, Version
"""

from __future__ import annotations

import re
import sys
import typing
from typing import (
    Any,
    Callable,
    Literal,
    NamedTuple,
    SupportsInt,
    Tuple,
    TypedDict,
    Union,
)

from ._structures import Infinity, InfinityType, NegativeInfinity, NegativeInfinityType

if typing.TYPE_CHECKING:
    from typing_extensions import Self, Unpack

if sys.version_info >= (3, 13):  # pragma: no cover
    from warnings import deprecated as _deprecated
elif typing.TYPE_CHECKING:
    from typing_extensions import deprecated as _deprecated
else:  # pragma: no cover
    import functools
    import warnings

    def _deprecated(message: str) -> object:
        def decorator(func: object) -> object:
            @functools.wraps(func)
            def wrapper(*args: object, **kwargs: object) -> object:
                warnings.warn(
                    message,
                    category=DeprecationWarning,
                    stacklevel=2,
                )
                return func(*args, **kwargs)

            return wrapper

        return decorator


_LETTER_NORMALIZATION = {
    "alpha": "a",
    "beta": "b",
    "c": "rc",
    "pre": "rc",
    "preview": "rc",
    "rev": "post",
    "r": "post",
}

__all__ = ["VERSION_PATTERN", "InvalidVersion", "Version", "parse"]

LocalType = Tuple[Union[int, str], ...]

CmpPrePostDevType = Union[InfinityType, NegativeInfinityType, Tuple[str, int]]
CmpLocalType = Union[
    NegativeInfinityType,
    Tuple[Union[Tuple[int, str], Tuple[NegativeInfinityType, Union[int, str]]], ...],
]
CmpKey = Tuple[
    int,
    Tuple[int, ...],
    CmpPrePostDevType,
    CmpPrePostDevType,
    CmpPrePostDevType,
    CmpLocalType,
]
VersionComparisonMethod = Callable[[CmpKey, CmpKey], bool]


class _VersionReplace(TypedDict, total=False):
    epoch: int | None
    release: tuple[int, ...] | None
    pre: tuple[Literal["a", "b", "rc"], int] | None
    post: int | None
    dev: int | None
    local: str | None


def parse(version: str) -> Version:
    """Parse the given version string.

    >>> parse('1.0.dev1')
    <Version('1.0.dev1')>

    :param version: The version string to parse.
    :raises InvalidVersion: When the version string is not a valid version.
    """
    return Version(version)


class InvalidVersion(ValueError):
    """Raised when a version string is not a valid version.

    >>> Version("invalid")
    Traceback (most recent call last):
        ...
    packaging.version.InvalidVersion: Invalid version: 'invalid'
    """


class _BaseVersion:
    __slots__ = ()

    # This can also be a normal member (see the packaging_legacy package);
    # we are just requiring it to be readable. Actually defining a property
    # has runtime effect on subclasses, so it's typing only.
    if typing.TYPE_CHECKING:

        @property
        def _key(self) -> tuple[Any, ...]: ...

    def __hash__(self) -> int:
        return hash(self._key)

    # Please keep the duplicated `isinstance` check
    # in the six comparisons hereunder
    # unless you find a way to avoid adding overhead function calls.
    def __lt__(self, other: _BaseVersion) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key < other._key

    def __le__(self, other: _BaseVersion) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key <= other._key

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key == other._key

    def __ge__(self, other: _BaseVersion) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key >= other._key

    def __gt__(self, other: _BaseVersion) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key > other._key

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key != other._key


# Deliberately not anchored to the start and end of the string, to make it
# easier for 3rd party code to reuse

# Note that ++ doesn't behave identically on CPython and PyPy, so not using it here
_VERSION_PATTERN = r"""
    v?+                                                   # optional leading v
    (?:
        (?:(?P<epoch>[0-9]+)!)?+                          # epoch
        (?P<release>[0-9]+(?:\.[0-9]+)*+)                 # release segment
        (?P<pre>                                          # pre-release
            [._-]?+
            (?P<pre_l>alpha|a|beta|b|preview|pre|c|rc)
            [._-]?+
            (?P<pre_n>[0-9]+)?
        )?+
        (?P<post>                                         # post release
            (?:-(?P<post_n1>[0-9]+))
            |
            (?:
                [._-]?
                (?P<post_l>post|rev|r)
                [._-]?
                (?P<post_n2>[0-9]+)?
            )
        )?+
        (?P<dev>                                          # dev release
            [._-]?+
            (?P<dev_l>dev)
            [._-]?+
            (?P<dev_n>[0-9]+)?
        )?+
    )
    (?:\+
        (?P<local>                                        # local version
            [a-z0-9]+
            (?:[._-][a-z0-9]+)*+
        )
    )?+
"""

_VERSION_PATTERN_OLD = _VERSION_PATTERN.replace("*+", "*").replace("?+", "?")

# Possessive qualifiers were added in Python 3.11.
# CPython 3.11.0-3.11.4 had a bug: https://github.com/python/cpython/pull/107795
# Older PyPy also had a bug.
VERSION_PATTERN = (
    _VERSION_PATTERN_OLD
    if (sys.implementation.name == "cpython" and sys.version_info < (3, 11, 5))
    or (sys.implementation.name == "pypy" and sys.version_info < (3, 11, 13))
    or sys.version_info < (3, 11)
    else _VERSION_PATTERN
)
"""
A string containing the regular expression used to match a valid version.

The pattern is not anchored at either end, and is intended for embedding in larger
expressions (for example, matching a version number as part of a file name). The
regular expression should be compiled with the ``re.VERBOSE`` and ``re.IGNORECASE``
flags set.

:meta hide-value:
"""


# Validation pattern for local version in replace()
_LOCAL_PATTERN = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*", re.IGNORECASE)


def _validate_epoch(value: object, /) -> int:
    epoch = value or 0
    if isinstance(epoch, int) and epoch >= 0:
        return epoch
    msg = f"epoch must be non-negative integer, got {epoch}"
    raise InvalidVersion(msg)


def _validate_release(value: object, /) -> tuple[int, ...]:
    release = (0,) if value is None else value
    if (
        isinstance(release, tuple)
        and len(release) > 0
        and all(isinstance(i, int) and i >= 0 for i in release)
    ):
        return release
    msg = f"release must be a non-empty tuple of non-negative integers, got {release}"
    raise InvalidVersion(msg)


def _validate_pre(value: object, /) -> tuple[Literal["a", "b", "rc"], int] | None:
    if value is None:
        return value
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and value[0] in ("a", "b", "rc")
        and isinstance(value[1], int)
        and value[1] >= 0
    ):
        return value
    msg = f"pre must be a tuple of ('a'|'b'|'rc', non-negative int), got {value}"
    raise InvalidVersion(msg)


def _validate_post(value: object, /) -> tuple[Literal["post"], int] | None:
    if value is None:
        return value
    if isinstance(value, int) and value >= 0:
        return ("post", value)
    msg = f"post must be non-negative integer, got {value}"
    raise InvalidVersion(msg)


def _validate_dev(value: object, /) -> tuple[Literal["dev"], int] | None:
    if value is None:
        return value
    if isinstance(value, int) and value >= 0:
        return ("dev", value)
    msg = f"dev must be non-negative integer, got {value}"
    raise InvalidVersion(msg)


def _validate_local(value: object, /) -> LocalType | None:
    if value is None:
        return value
    if isinstance(value, str) and _LOCAL_PATTERN.fullmatch(value):
        return _parse_local_version(value)
    msg = f"local must be a valid version string, got {value!r}"
    raise InvalidVersion(msg)


# Backward compatibility for internals before 26.0. Do not use.
class _Version(NamedTuple):
    epoch: int
    release: tuple[int, ...]
    dev: tuple[str, int] | None
    pre: tuple[str, int] | None
    post: tuple[str, int] | None
    local: LocalType | None


class Version(_BaseVersion):
    """This class abstracts handling of a project's versions.

    A :class:`Version` instance is comparison aware and can be compared and
    sorted using the standard Python interfaces.

    >>> v1 = Version("1.0a5")
    >>> v2 = Version("1.0")
    >>> v1
    <Version('1.0a5')>
    >>> v2
    <Version('1.0')>
    >>> v1 < v2
    True
    >>> v1 == v2
    False
    >>> v1 > v2
    False
    >>> v1 >= v2
    False
    >>> v1 <= v2
    True
    """

    __slots__ = ("_dev", "_epoch", "_key_cache", "_local", "_post", "_pre", "_release")
    __match_args__ = ("_str",)

    _regex = re.compile(r"\s*" + VERSION_PATTERN + r"\s*", re.VERBOSE | re.IGNORECASE)

    _epoch: int
    _release: tuple[int, ...]
    _dev: tuple[str, int] | None
    _pre: tuple[str, int] | None
    _post: tuple[str, int] | None
    _local: LocalType | None

    _key_cache: CmpKey | None

    def __init__(self, version: str) -> None:
        """Initialize a Version object.

        :param version:
            The string representation of a version which will be parsed and normalized
            before use.
        :raises InvalidVersion:
            If the ``version`` does not conform to PEP 440 in any way then this
            exception will be raised.
        """
        # Validate the version and parse it into pieces
        match = self._regex.fullmatch(version)
        if not match:
            raise InvalidVersion(f"Invalid version: {version!r}")
        self._epoch = int(match.group("epoch")) if match.group("epoch") else 0
        self._release = tuple(map(int, match.group("release").split(".")))
        self._pre = _parse_letter_version(match.group("pre_l"), match.group("pre_n"))
        self._post = _parse_letter_version(
            match.group("post_l"), match.group("post_n1") or match.group("post_n2")
        )
        self._dev = _parse_letter_version(match.group("dev_l"), match.group("dev_n"))
        self._local = _parse_local_version(match.group("local"))

        # Key which will be used for sorting
        self._key_cache = None

    def __replace__(self, **kwargs: Unpack[_VersionReplace]) -> Self:
        epoch = _validate_epoch(kwargs["epoch"]) if "epoch" in kwargs else self._epoch
        release = (
            _validate_release(kwargs["release"])
            if "release" in kwargs
            else self._release
        )
        pre = _validate_pre(kwargs["pre"]) if "pre" in kwargs else self._pre
        post = _validate_post(kwargs["post"]) if "post" in kwargs else self._post
        dev = _validate_dev(kwargs["dev"]) if "dev" in kwargs else self._dev
        local = _validate_local(kwargs["local"]) if "local" in kwargs else self._local

        if (
            epoch == self._epoch
            and release == self._release
            and pre == self._pre
            and post == self._post
            and dev == self._dev
            and local == self._local
        ):
            return self

        new_version = self.__class__.__new__(self.__class__)
        new_version._key_cache = None
        new_version._epoch = epoch
        new_version._release = release
        new_version._pre = pre
        new_version._post = post
        new_version._dev = dev
        new_version._local = local

        return new_version

    @property
    def _key(self) -> CmpKey:
        if self._key_cache is None:
            self._key_cache = _cmpkey(
                self._epoch,
                self._release,
                self._pre,
                self._post,
                self._dev,
                self._local,
            )
        return self._key_cache

    @property
    @_deprecated("Version._version is private and will be removed soon")
    def _version(self) -> _Version:
        return _Version(
            self._epoch, self._release, self._dev, self._pre, self._post, self._local
        )

    @_version.setter
    @_deprecated("Version._version is private and will be removed soon")
    def _version(self, value: _Version) -> None:
        self._epoch = value.epoch
        self._release = value.release
        self._dev = value.dev
        self._pre = value.pre
        self._post = value.post
        self._local = value.local
        self._key_cache = None

    def __repr__(self) -> str:
        """A representation of the Version that shows all internal state.

        >>> Version('1.0.0')
        <Version('1.0.0')>
        """
        return f"<Version('{self}')>"

    def __str__(self) -> str:
        """A string representation of the version that can be round-tripped.

        >>> str(Version("1.0a5"))
        '1.0a5'
        """
        # This is a hot function, so not calling self.base_version
        version = ".".join(map(str, self.release))

        # Epoch
        if self.epoch:
            version = f"{self.epoch}!{version}"

        # Pre-release
        if self.pre is not None:
            version += "".join(map(str, self.pre))

        # Post-release
        if self.post is not None:
            version += f".post{self.post}"

        # Development release
        if self.dev is not None:
            version += f".dev{self.dev}"

        # Local version segment
        if self.local is not None:
            version += f"+{self.local}"

        return version

    @property
    def _str(self) -> str:
        """Internal property for match_args"""
        return str(self)

    @property
    def epoch(self) -> int:
        """The epoch of the version.

        >>> Version("2.0.0").epoch
        0
        >>> Version("1!2.0.0").epoch
        1
        """
        return self._epoch

    @property
    def release(self) -> tuple[int, ...]:
        """The components of the "release" segment of the version.

        >>> Version("1.2.3").release
        (1, 2, 3)
        >>> Version("2.0.0").release
        (2, 0, 0)
        >>> Version("1!2.0.0.post0").release
        (2, 0, 0)

        Includes trailing zeroes but not the epoch or any pre-release / development /
        post-release suffixes.
        """
        return self._release

    @property
    def pre(self) -> tuple[str, int] | None:
        """The pre-release segment of the version.

        >>> print(Version("1.2.3").pre)
        None
        >>> Version("1.2.3a1").pre
        ('a', 1)
        >>> Version("1.2.3b1").pre
        ('b', 1)
        >>> Version("1.2.3rc1").pre
        ('rc', 1)
        """
        return self._pre

    @property
    def post(self) -> int | None:
        """The post-release number of the version.

        >>> print(Version("1.2.3").post)
        None
        >>> Version("1.2.3.post1").post
        1
        """
        return self._post[1] if self._post else None

    @property
    def dev(self) -> int | None:
        """The development number of the version.

        >>> print(Version("1.2.3").dev)
        None
        >>> Version("1.2.3.dev1").dev
        1
        """
        return self._dev[1] if self._dev else None

    @property
    def local(self) -> str | None:
        """The local version segment of the version.

        >>> print(Version("1.2.3").local)
        None
        >>> Version("1.2.3+abc").local
        'abc'
        """
        if self._local:
            return ".".join(str(x) for x in self._local)
        else:
            return None

    @property
    def public(self) -> str:
        """The public portion of the version.

        >>> Version("1.2.3").public
        '1.2.3'
        >>> Version("1.2.3+abc").public
        '1.2.3'
        >>> Version("1!1.2.3dev1+abc").public
        '1!1.2.3.dev1'
        """
        return str(self).split("+", 1)[0]

    @property
    def base_version(self) -> str:
        """The "base version" of the version.

        >>> Version("1.2.3").base_version
        '1.2.3'
        >>> Version("1.2.3+abc").base_version
        '1.2.3'
        >>> Version("1!1.2.3dev1+abc").base_version
        '1!1.2.3'

        The "base version" is the public version of the project without any pre or post
        release markers.
        """
        release_segment = ".".join(map(str, self.release))
        return f"{self.epoch}!{release_segment}" if self.epoch else release_segment

    @property
    def is_prerelease(self) -> bool:
        """Whether this version is a pre-release.

        >>> Version("1.2.3").is_prerelease
        False
        >>> Version("1.2.3a1").is_prerelease
        True
        >>> Version("1.2.3b1").is_prerelease
        True
        >>> Version("1.2.3rc1").is_prerelease
        True
        >>> Version("1.2.3dev1").is_prerelease
        True
        """
        return self.dev is not None or self.pre is not None

    @property
    def is_postrelease(self) -> bool:
        """Whether this version is a post-release.

        >>> Version("1.2.3").is_postrelease
        False
        >>> Version("1.2.3.post1").is_postrelease
        True
        """
        return self.post is not None

    @property
    def is_devrelease(self) -> bool:
        """Whether this version is a development release.

        >>> Version("1.2.3").is_devrelease
        False
        >>> Version("1.2.3.dev1").is_devrelease
        True
        """
        return self.dev is not None

    @property
    def major(self) -> int:
        """The first item of :attr:`release` or ``0`` if unavailable.

        >>> Version("1.2.3").major
        1
        """
        return self.release[0] if len(self.release) >= 1 else 0

    @property
    def minor(self) -> int:
        """The second item of :attr:`release` or ``0`` if unavailable.

        >>> Version("1.2.3").minor
        2
        >>> Version("1").minor
        0
        """
        return self.release[1] if len(self.release) >= 2 else 0

    @property
    def micro(self) -> int:
        """The third item of :attr:`release` or ``0`` if unavailable.

        >>> Version("1.2.3").micro
        3
        >>> Version("1").micro
        0
        """
        return self.release[2] if len(self.release) >= 3 else 0


class _TrimmedRelease(Version):
    __slots__ = ()

    def __init__(self, version: str | Version) -> None:
        if isinstance(version, Version):
            self._epoch = version._epoch
            self._release = version._release
            self._dev = version._dev
            self._pre = version._pre
            self._post = version._post
            self._local = version._local
            self._key_cache = version._key_cache
            return
        super().__init__(version)  # pragma: no cover

    @property
    def release(self) -> tuple[int, ...]:
        """
        Release segment without any trailing zeros.

        >>> _TrimmedRelease('1.0.0').release
        (1,)
        >>> _TrimmedRelease('0.0').release
        (0,)
        """
        # This leaves one 0.
        rel = super().release
        len_release = len(rel)
        i = len_release
        while i > 1 and rel[i - 1] == 0:
            i -= 1
        return rel if i == len_release else rel[:i]


def _parse_letter_version(
    letter: str | None, number: str | bytes | SupportsInt | None
) -> tuple[str, int] | None:
    if letter:
        # We normalize any letters to their lower case form
        letter = letter.lower()

        # We consider some words to be alternate spellings of other words and
        # in those cases we want to normalize the spellings to our preferred
        # spelling.
        letter = _LETTER_NORMALIZATION.get(letter, letter)

        # We consider there to be an implicit 0 in a pre-release if there is
        # not a numeral associated with it.
        return letter, int(number or 0)

    if number:
        # We assume if we are given a number, but we are not given a letter
        # then this is using the implicit post release syntax (e.g. 1.0-1)
        return "post", int(number)

    return None


_local_version_separators = re.compile(r"[\._-]")


def _parse_local_version(local: str | None) -> LocalType | None:
    """
    Takes a string like abc.1.twelve and turns it into ("abc", 1, "twelve").
    """
    if local is not None:
        return tuple(
            part.lower() if not part.isdigit() else int(part)
            for part in _local_version_separators.split(local)
        )
    return None


def _cmpkey(
    epoch: int,
    release: tuple[int, ...],
    pre: tuple[str, int] | None,
    post: tuple[str, int] | None,
    dev: tuple[str, int] | None,
    local: LocalType | None,
) -> CmpKey:
    # When we compare a release version, we want to compare it with all of the
    # trailing zeros removed. We will use this for our sorting key.
    len_release = len(release)
    i = len_release
    while i and release[i - 1] == 0:
        i -= 1
    _release = release if i == len_release else release[:i]

    # We need to "trick" the sorting algorithm to put 1.0.dev0 before 1.0a0.
    # We'll do this by abusing the pre segment, but we _only_ want to do this
    # if there is not a pre or a post segment. If we have one of those then
    # the normal sorting rules will handle this case correctly.
    if pre is None and post is None and dev is not None:
        _pre: CmpPrePostDevType = NegativeInfinity
    # Versions without a pre-release (except as noted above) should sort after
    # those with one.
    elif pre is None:
        _pre = Infinity
    else:
        _pre = pre

    # Versions without a post segment should sort before those with one.
    if post is None:
        _post: CmpPrePostDevType = NegativeInfinity

    else:
        _post = post

    # Versions without a development segment should sort after those with one.
    if dev is None:
        _dev: CmpPrePostDevType = Infinity

    else:
        _dev = dev

    if local is None:
        # Versions without a local segment should sort before those with one.
        _local: CmpLocalType = NegativeInfinity
    else:
        # Versions with a local segment need that segment parsed to implement
        # the sorting rules in PEP440.
        # - Alpha numeric segments sort before numeric segments
        # - Alpha numeric segments sort lexicographically
        # - Numeric segments sort numerically
        # - Shorter versions sort before longer versions when the prefixes
        #   match exactly
        _local = tuple(
            (i, "") if isinstance(i, int) else (NegativeInfinity, i) for i in local
        )

    return epoch, _release, _pre, _post, _dev, _local
