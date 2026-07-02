# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.

from __future__ import annotations

import operator
import os
import platform
import sys
from typing import AbstractSet, Callable, Literal, Mapping, TypedDict, Union, cast

from ._parser import MarkerAtom, MarkerList, Op, Value, Variable
from ._parser import parse_marker as _parse_marker
from ._tokenizer import ParserSyntaxError
from .specifiers import InvalidSpecifier, Specifier
from .utils import canonicalize_name

__all__ = [
    "Environment",
    "EvaluateContext",
    "InvalidMarker",
    "Marker",
    "UndefinedComparison",
    "UndefinedEnvironmentName",
    "default_environment",
]


def __dir__() -> list[str]:
    return __all__


Operator = Callable[[str, Union[str, AbstractSet[str]]], bool]
EvaluateContext = Literal["metadata", "lock_file", "requirement"]
"""A ``typing.Literal`` enumerating valid marker evaluation contexts.

Valid values for the ``context`` passed to :meth:`Marker.evaluate` are:

* ``"metadata"`` (for core metadata; default)
* ``"lock_file"`` (for lock files)
* ``"requirement"`` (i.e. all other situations)
"""

MARKERS_ALLOWING_SET = {"extras", "dependency_groups"}
MARKERS_REQUIRING_VERSION = {
    "implementation_version",
    "platform_release",
    "python_full_version",
    "python_version",
}


class InvalidMarker(ValueError):
    """Raised when attempting to create a :class:`Marker` from invalid input.

    This error indicates that the given marker string does not conform to the
    :ref:`specification of dependency specifiers <pypug:dependency-specifiers>`.
    """


class UndefinedComparison(ValueError):
    """Raised when evaluating an unsupported marker comparison.

    This can happen when marker values are compared as versions but do not
    conform to the :ref:`specification of version specifiers
    <pypug:version-specifiers>`.
    """


class UndefinedEnvironmentName(ValueError):
    """Raised when evaluating a marker that references a missing environment key."""


class Environment(TypedDict):
    """
    A dictionary that represents a Python environment as captured by
    :func:`default_environment`. All fields are required.
    """

    implementation_name: str
    """The implementation's identifier, e.g. ``'cpython'``."""

    implementation_version: str
    """
    The implementation's version, e.g. ``'3.13.0a2'`` for CPython 3.13.0a2, or
    ``'7.3.13'`` for PyPy3.10 v7.3.13.
    """

    os_name: str
    """
    The value of :py:data:`os.name`. The name of the operating system dependent module
    imported, e.g. ``'posix'``.
    """

    platform_machine: str
    """
    Returns the machine type, e.g. ``'i386'``.

    An empty string if the value cannot be determined.
    """

    platform_release: str
    """
    The system's release, e.g. ``'2.2.0'`` or ``'NT'``.

    An empty string if the value cannot be determined.
    """

    platform_system: str
    """
    The system/OS name, e.g. ``'Linux'``, ``'Windows'`` or ``'Java'``.

    An empty string if the value cannot be determined.
    """

    platform_version: str
    """
    The system's release version, e.g. ``'#3 on degas'``.

    An empty string if the value cannot be determined.
    """

    python_full_version: str
    """
    The Python version as string ``'major.minor.patchlevel'``.

    Note that unlike the Python :py:data:`sys.version`, this value will always include
    the patchlevel (it defaults to 0).
    """

    platform_python_implementation: str
    """
    A string identifying the Python implementation, e.g. ``'CPython'``.
    """

    python_version: str
    """The Python version as string ``'major.minor'``."""

    sys_platform: str
    """
    This string contains a platform identifier that can be used to append
    platform-specific components to :py:data:`sys.path`, for instance.

    For Unix systems, except on Linux and AIX, this is the lowercased OS name as
    returned by ``uname -s`` with the first part of the version as returned by
    ``uname -r`` appended, e.g. ``'sunos5'`` or ``'freebsd8'``, at the time when Python
    was built.
    """


def _normalize_extras(
    result: MarkerList | MarkerAtom | str,
) -> MarkerList | MarkerAtom | str:
    if not isinstance(result, tuple):
        return result

    lhs, op, rhs = result
    if isinstance(lhs, Variable) and lhs.value == "extra":
        normalized_extra = canonicalize_name(rhs.value)
        rhs = Value(normalized_extra)
    elif isinstance(rhs, Variable) and rhs.value == "extra":
        normalized_extra = canonicalize_name(lhs.value)
        lhs = Value(normalized_extra)
    return lhs, op, rhs


def _normalize_extra_values(results: MarkerList) -> MarkerList:
    """
    Normalize extra values.
    """

    return [_normalize_extras(r) for r in results]


def _format_marker(
    marker: list[str] | MarkerAtom | str, first: bool | None = True
) -> str:
    assert isinstance(marker, (list, tuple, str))

    # Sometimes we have a structure like [[...]] which is a single item list
    # where the single item is itself it's own list. In that case we want skip
    # the rest of this function so that we don't get extraneous () on the
    # outside.
    if (
        isinstance(marker, list)
        and len(marker) == 1
        and isinstance(marker[0], (list, tuple))
    ):
        return _format_marker(marker[0])

    if isinstance(marker, list):
        inner = (_format_marker(m, first=False) for m in marker)
        if first:
            return " ".join(inner)
        else:
            return "(" + " ".join(inner) + ")"
    elif isinstance(marker, tuple):
        return " ".join([m.serialize() for m in marker])
    else:
        return marker


_operators: dict[str, Operator] = {
    "in": lambda lhs, rhs: lhs in rhs,
    "not in": lambda lhs, rhs: lhs not in rhs,
    "<": lambda _lhs, _rhs: False,
    "<=": operator.eq,
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.eq,
    ">": lambda _lhs, _rhs: False,
}


def _eval_op(lhs: str, op: Op, rhs: str | AbstractSet[str], *, key: str) -> bool:
    op_str = op.serialize()
    if key in MARKERS_REQUIRING_VERSION:
        try:
            spec = Specifier(f"{op_str}{rhs}")
        except InvalidSpecifier:
            pass
        else:
            return spec.contains(lhs, prereleases=True)

    oper: Operator | None = _operators.get(op_str)
    if oper is None:
        raise UndefinedComparison(f"Undefined {op!r} on {lhs!r} and {rhs!r}.")

    return oper(lhs, rhs)


def _normalize(
    lhs: str, rhs: str | AbstractSet[str], key: str
) -> tuple[str, str | AbstractSet[str]]:
    # PEP 685 - Comparison of extra names for optional distribution dependencies
    # https://peps.python.org/pep-0685/
    # > When comparing extra names, tools MUST normalize the names being
    # > compared using the semantics outlined in PEP 503 for names
    if key == "extra":
        assert isinstance(rhs, str), "extra value must be a string"
        # Both sides are normalized at this point already
        return (lhs, rhs)
    if key in MARKERS_ALLOWING_SET:
        if isinstance(rhs, str):  # pragma: no cover
            return (canonicalize_name(lhs), canonicalize_name(rhs))
        else:
            return (canonicalize_name(lhs), {canonicalize_name(v) for v in rhs})

    # other environment markers don't have such standards
    return lhs, rhs


def _evaluate_markers(
    markers: MarkerList, environment: dict[str, str | AbstractSet[str]]
) -> bool:
    groups: list[list[bool]] = [[]]

    for marker in markers:
        if isinstance(marker, list):
            groups[-1].append(_evaluate_markers(marker, environment))
        elif isinstance(marker, tuple):
            lhs, op, rhs = marker

            if isinstance(lhs, Variable):
                environment_key = lhs.value
                lhs_value = environment[environment_key]
                rhs_value = rhs.value
            else:
                lhs_value = lhs.value
                environment_key = rhs.value
                rhs_value = environment[environment_key]

            assert isinstance(lhs_value, str), "lhs must be a string"
            lhs_value, rhs_value = _normalize(lhs_value, rhs_value, key=environment_key)
            groups[-1].append(_eval_op(lhs_value, op, rhs_value, key=environment_key))
        elif marker == "or":
            groups.append([])
        elif marker == "and":
            pass
        else:  # pragma: nocover
            raise TypeError(f"Unexpected marker {marker!r}")

    return any(all(item) for item in groups)


def _format_full_version(info: sys._version_info) -> str:
    version = f"{info.major}.{info.minor}.{info.micro}"
    kind = info.releaselevel
    if kind != "final":
        version += kind[0] + str(info.serial)
    return version


def default_environment() -> Environment:
    """Return the default marker environment for the current Python process.

    This is the base environment used by :meth:`Marker.evaluate`.
    """
    iver = _format_full_version(sys.implementation.version)
    implementation_name = sys.implementation.name
    return {
        "implementation_name": implementation_name,
        "implementation_version": iver,
        "os_name": os.name,
        "platform_machine": platform.machine(),
        "platform_release": platform.release(),
        "platform_system": platform.system(),
        "platform_version": platform.version(),
        "python_full_version": platform.python_version(),
        "platform_python_implementation": platform.python_implementation(),
        "python_version": ".".join(platform.python_version_tuple()[:2]),
        "sys_platform": sys.platform,
    }


class Marker:
    """Represents a parsed dependency marker expression.

    Marker expressions are parsed according to the
    :ref:`specification of dependency specifiers <pypug:dependency-specifiers>`.

    :param marker: The string representation of a marker expression.
    :raises InvalidMarker: If ``marker`` cannot be parsed.

    Instances are safe to serialize with :mod:`pickle`. They use a stable
    format so the same pickle can be loaded in future packaging releases.

    .. versionchanged:: 26.2

        Added a stable pickle format. Pickles created with packaging 26.2+ can
        be unpickled with future releases.  Backward compatibility with pickles
        from pip._vendor.packaging < 26.2 is supported but may be removed in a future
        release.
    """

    __slots__ = ("_markers",)

    def __init__(self, marker: str) -> None:
        # Note: We create a Marker object without calling this constructor in
        #       packaging.requirements.Requirement. If any additional logic is
        #       added here, make sure to mirror/adapt Requirement.

        # If this fails and throws an error, the repr still expects _markers to
        # be defined.
        self._markers: MarkerList = []

        try:
            self._markers = _normalize_extra_values(_parse_marker(marker))
            # The attribute `_markers` can be described in terms of a recursive type:
            # MarkerList = List[Union[Tuple[Node, ...], str, MarkerList]]
            #
            # For example, the following expression:
            # python_version > "3.6" or (python_version == "3.6" and os_name == "unix")
            #
            # is parsed into:
            # [
            #     (<Variable('python_version')>, <Op('>')>, <Value('3.6')>),
            #     'and',
            #     [
            #         (<Variable('python_version')>, <Op('==')>, <Value('3.6')>),
            #         'or',
            #         (<Variable('os_name')>, <Op('==')>, <Value('unix')>)
            #     ]
            # ]
        except ParserSyntaxError as e:
            raise InvalidMarker(str(e)) from e

    @classmethod
    def _from_markers(cls, markers: MarkerList) -> Marker:
        """Create a Marker instance from a pre-parsed marker tree.

        This avoids re-parsing serialised marker strings when combining markers.
        """
        new = cls.__new__(cls)
        new._markers = markers
        return new

    def __str__(self) -> str:
        return _format_marker(self._markers)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}({str(self)!r})>"

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Marker):
            return NotImplemented

        return str(self) == str(other)

    def __getstate__(self) -> str:
        # Return the marker expression string for compactness and stability.
        # Internal Node objects are excluded; the string is re-parsed on load.
        return str(self)

    def __setstate__(self, state: object) -> None:
        if isinstance(state, str):
            # New format (26.2+): just the marker expression string.
            try:
                self._markers = _normalize_extra_values(_parse_marker(state))
            except ParserSyntaxError as exc:
                raise TypeError(f"Cannot restore Marker from {state!r}") from exc
            return
        if isinstance(state, dict) and "_markers" in state:
            # Old format (packaging <= 26.1, no __slots__): plain __dict__.
            markers = state["_markers"]
            if isinstance(markers, list):
                self._markers = markers
                return
        if isinstance(state, tuple) and len(state) == 2:
            # Old format (packaging <= 26.1, __slots__): (None, {slot: value}).
            _, slot_dict = state
            if isinstance(slot_dict, dict) and "_markers" in slot_dict:
                markers = slot_dict["_markers"]
                if isinstance(markers, list):
                    self._markers = markers
                    return
        raise TypeError(f"Cannot restore Marker from {state!r}")

    def __and__(self, other: Marker) -> Marker:
        if not isinstance(other, Marker):
            return NotImplemented
        return self._from_markers([self._markers, "and", other._markers])

    def __or__(self, other: Marker) -> Marker:
        if not isinstance(other, Marker):
            return NotImplemented
        return self._from_markers([self._markers, "or", other._markers])

    def evaluate(
        self,
        environment: Mapping[str, str | AbstractSet[str]] | None = None,
        context: EvaluateContext = "metadata",
    ) -> bool:
        """Evaluate a marker.

        Return the boolean from evaluating this marker against the environment.
        The environment is determined from the current Python process unless
        passed in explicitly.

        :param environment: Mapping containing keys and values to override the
           detected environment.
        :param EvaluateContext context: The context in which the marker is
            evaluated, which influences what marker names are considered valid.
            Accepted values are ``"metadata"`` (for core metadata; default),
            ``"lock_file"``, and ``"requirement"`` (i.e. all other situations).
        :raises UndefinedComparison: If the marker uses a comparison on values
            that are not valid versions per the :ref:`specification of version
            specifiers <pypug:version-specifiers>`.
        :raises UndefinedEnvironmentName: If the marker references a value that
            is missing from the evaluation environment.
        :returns: ``True`` if the marker matches, otherwise ``False``.

        """
        current_environment = cast(
            "dict[str, str | AbstractSet[str]]", default_environment()
        )
        if context == "lock_file":
            current_environment.update(
                extras=frozenset(), dependency_groups=frozenset()
            )
        elif context == "metadata":
            current_environment["extra"] = ""

        if environment is not None:
            current_environment.update(environment)
            if "extra" in current_environment:
                # The API used to allow setting extra to None. We need to handle
                # this case for backwards compatibility. Also skip running
                # normalize name if extra is empty.
                extra = cast("str | None", current_environment["extra"])
                current_environment["extra"] = canonicalize_name(extra) if extra else ""

        return _evaluate_markers(
            self._markers, _repair_python_full_version(current_environment)
        )


def _repair_python_full_version(
    env: dict[str, str | AbstractSet[str]],
) -> dict[str, str | AbstractSet[str]]:
    """
    Work around platform.python_version() returning something that is not PEP 440
    compliant for non-tagged Python builds.
    """
    python_full_version = cast("str", env["python_full_version"])
    if python_full_version.endswith("+"):
        env["python_full_version"] = f"{python_full_version}local"
    return env
