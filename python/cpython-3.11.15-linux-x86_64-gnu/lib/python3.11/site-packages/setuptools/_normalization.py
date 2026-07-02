"""
Helpers for normalization as expected in wheel/sdist/module file names
and core metadata
"""

import re
from typing import TYPE_CHECKING

import packaging

# https://packaging.python.org/en/latest/specifications/core-metadata/#name
_VALID_NAME = re.compile(r"^([A-Z0-9]|[A-Z0-9][A-Z0-9._-]*[A-Z0-9])$", re.IGNORECASE)
_UNSAFE_NAME_CHARS = re.compile(r"[^A-Z0-9._-]+", re.IGNORECASE)
_NON_ALPHANUMERIC = re.compile(r"[^A-Z0-9]+", re.IGNORECASE)
_PEP440_FALLBACK = re.compile(
    r"^v?(?P<safe>(?:[0-9]+!)?[0-9]+(?:\.[0-9]+)*)", re.IGNORECASE
)


def safe_identifier(name: str) -> str:
    """Make a string safe to be used as Python identifier.
    >>> safe_identifier("12abc")
    '_12abc'
    >>> safe_identifier("__editable__.myns.pkg-78.9.3_local")
    '__editable___myns_pkg_78_9_3_local'
    """
    safe = re.sub(r'\W|^(?=\d)', '_', name)
    assert safe.isidentifier()
    return safe


def safe_name(component: str) -> str:
    """Escape a component used as a project name according to Core Metadata.
    >>> safe_name("hello world")
    'hello-world'
    >>> safe_name("hello?world")
    'hello-world'
    >>> safe_name("hello_world")
    'hello_world'
    """
    return _UNSAFE_NAME_CHARS.sub("-", component)


def safe_version(version: str) -> str:
    """Convert an arbitrary string into a valid version string.
    Can still raise an ``InvalidVersion`` exception.
    To avoid exceptions use ``best_effort_version``.
    >>> safe_version("1988 12 25")
    '1988.12.25'
    >>> safe_version("v0.2.1")
    '0.2.1'
    >>> safe_version("v0.2?beta")
    '0.2b0'
    >>> safe_version("v0.2 beta")
    '0.2b0'
    >>> safe_version("ubuntu lts")
    Traceback (most recent call last):
    ...
    packaging.version.InvalidVersion: Invalid version: 'ubuntu.lts'
    """
    v = version.replace(' ', '.')
    try:
        return str(packaging.version.Version(v))
    except packaging.version.InvalidVersion:
        attempt = _UNSAFE_NAME_CHARS.sub("-", v)
        return str(packaging.version.Version(attempt))


def best_effort_version(version: str) -> str:
    """Convert an arbitrary string into a version-like string.
    Fallback when ``safe_version`` is not safe enough.
    >>> best_effort_version("v0.2 beta")
    '0.2b0'
    >>> best_effort_version("ubuntu lts")
    '0.dev0+sanitized.ubuntu.lts'
    >>> best_effort_version("0.23ubuntu1")
    '0.23.dev0+sanitized.ubuntu1'
    >>> best_effort_version("0.23-")
    '0.23.dev0+sanitized'
    >>> best_effort_version("0.-_")
    '0.dev0+sanitized'
    >>> best_effort_version("42.+?1")
    '42.dev0+sanitized.1'
    """
    try:
        return safe_version(version)
    except packaging.version.InvalidVersion:
        v = version.replace(' ', '.')
        match = _PEP440_FALLBACK.search(v)
        if match:
            safe = match["safe"]
            rest = v[len(safe) :]
        else:
            safe = "0"
            rest = version
        safe_rest = _NON_ALPHANUMERIC.sub(".", rest).strip(".")
        local = f"sanitized.{safe_rest}".strip(".")
        return safe_version(f"{safe}.dev0+{local}")


def safe_extra(extra: str) -> str:
    """Normalize extra name according to PEP 685
    >>> safe_extra("_FrIeNdLy-._.-bArD")
    'friendly-bard'
    >>> safe_extra("FrIeNdLy-._.-bArD__._-")
    'friendly-bard'
    """
    return _NON_ALPHANUMERIC.sub("-", extra).strip("-").lower()


def filename_component(value: str) -> str:
    """Normalize each component of a filename (e.g. distribution/version part of wheel)
    Note: ``value`` needs to be already normalized.
    >>> filename_component("my-pkg")
    'my_pkg'
    """
    return value.replace("-", "_").strip("_")


def filename_component_broken(value: str) -> str:
    """
    Produce the incorrect filename component for compatibility.

    See pypa/setuptools#4167 for detailed analysis.

    TODO: replace this with filename_component after pip 24 is
    nearly-ubiquitous.

    >>> filename_component_broken('foo_bar-baz')
    'foo-bar-baz'
    """
    return value.replace('_', '-')


def safer_name(value: str) -> str:
    """Like ``safe_name`` but can be used as filename component for wheel"""
    # See bdist_wheel.safer_name
    return (
        # Per https://packaging.python.org/en/latest/specifications/name-normalization/#name-normalization
        re
        .sub(r"[-_.]+", "-", safe_name(value))
        .lower()
        # Per https://packaging.python.org/en/latest/specifications/binary-distribution-format/#escaping-and-unicode
        .replace("-", "_")
    )


def safer_best_effort_version(value: str) -> str:
    """Like ``best_effort_version`` but can be used as filename component for wheel"""
    # See bdist_wheel.safer_verion
    # TODO: Replace with only safe_version in the future (no need for best effort)
    return filename_component(best_effort_version(value))


def _missing_canonicalize_license_expression(expression: str) -> str:
    """
    Defer import error to affect only users that actually use it
    https://github.com/pypa/setuptools/issues/4894
    >>> _missing_canonicalize_license_expression("a OR b")
    Traceback (most recent call last):
    ...
    ImportError: ...Cannot import `packaging.licenses`...
    """
    raise ImportError(
        "Cannot import `packaging.licenses`."
        """
        Setuptools>=77.0.0 requires "packaging>=24.2" to work properly.
        Please make sure you have a suitable version installed.
        """
    )


try:
    from packaging.licenses import (
        canonicalize_license_expression as _canonicalize_license_expression,
    )
except ImportError:  # pragma: nocover
    if not TYPE_CHECKING:
        # XXX: pyright is still upset even with # pyright: ignore[reportAssignmentType]
        _canonicalize_license_expression = _missing_canonicalize_license_expression
