from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from .errors import _ErrorCollector
from .requirements import Requirement

__all__ = [
    "CyclicDependencyGroup",
    "DependencyGroupInclude",
    "DependencyGroupResolver",
    "DuplicateGroupNames",
    "InvalidDependencyGroupObject",
    "resolve_dependency_groups",
]


def __dir__() -> list[str]:
    return __all__


# -----------
# Error Types
# -----------


class DuplicateGroupNames(ValueError):
    """
    The same dependency groups were defined twice, with different non-normalized names.
    """


class CyclicDependencyGroup(ValueError):
    """
    The dependency group includes form a cycle.
    """

    def __init__(self, requested_group: str, group: str, include_group: str) -> None:
        self.requested_group = requested_group
        self.group = group
        self.include_group = include_group

        if include_group == group:
            reason = f"{group} includes itself"
        else:
            reason = f"{include_group} -> {group}, {group} -> {include_group}"
        super().__init__(
            "Cyclic dependency group include while resolving "
            f"{requested_group}: {reason}"
        )


# in the PEP 735 spec, the tables in dependency group lists were described as
# "Dependency Object Specifiers", but the only defined type of object was a
# "Dependency Group Include" -- hence the naming of this error as "Object"
class InvalidDependencyGroupObject(ValueError):
    """
    A member of a dependency group was identified as a dict, but was not in a valid
    format.
    """


# ------------------------
# Object Model & Interface
# ------------------------


class DependencyGroupInclude:
    __slots__ = ("include_group",)

    def __init__(self, include_group: str) -> None:
        """
        Initialize a DependencyGroupInclude.

        :param include_group: The name of the group referred to by this include.
        """
        self.include_group = include_group

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.include_group!r})"


class DependencyGroupResolver:
    """
    A resolver for Dependency Group data.

    This class handles caching, name normalization, cycle detection, and other
    parsing requirements. There are only two public methods for exploring the data:
    ``lookup()`` and ``resolve()``.

    :param dependency_groups: A mapping, as provided via pyproject
        ``[dependency-groups]``.
    """

    def __init__(
        self,
        dependency_groups: Mapping[str, Sequence[str | Mapping[str, str]]],
    ) -> None:
        errors = _ErrorCollector()

        self.dependency_groups = _normalize_group_names(dependency_groups, errors)

        # a map of group names to parsed data
        self._parsed_groups: dict[
            str, tuple[Requirement | DependencyGroupInclude, ...]
        ] = {}
        # a map of group names to their ancestors, used for cycle detection
        self._include_graph_ancestors: dict[str, tuple[str, ...]] = {}
        # a cache of completed resolutions to Requirement lists
        self._resolve_cache: dict[str, tuple[Requirement, ...]] = {}

        errors.finalize("[dependency-groups] data was invalid")

    def lookup(self, group: str) -> tuple[Requirement | DependencyGroupInclude, ...]:
        """
        Lookup a group name, returning the parsed dependency data for that group.
        This will not resolve includes.

        :param group: the name of the group to lookup
        """
        group = _normalize_name(group)

        with _ErrorCollector().on_exit(
            f"[dependency-groups] data for {group!r} was malformed"
        ) as errors:
            return self._parse_group(group, errors)

    def resolve(self, group: str) -> tuple[Requirement, ...]:
        """
        Resolve a dependency group to a list of requirements.

        :param group: the name of the group to resolve
        """
        group = _normalize_name(group)

        with _ErrorCollector().on_exit(
            f"[dependency-groups] data for {group!r} was malformed"
        ) as errors:
            return self._resolve(group, group, errors)

    def _resolve(
        self, group: str, requested_group: str, errors: _ErrorCollector
    ) -> tuple[Requirement, ...]:
        """
        This is a helper for cached resolution to strings. It preserves the name of the
        group which the user initially requested in order to present a clearer error in
        the event that a cycle is detected.

        :param group: The normalized name of the group to resolve.
        :param requested_group: The group which was used in the original, user-facing
            request.
        """
        if group in self._resolve_cache:
            return self._resolve_cache[group]

        parsed = self._parse_group(group, errors)

        resolved_group = []

        for item in parsed:
            if isinstance(item, Requirement):
                resolved_group.append(item)
            elif isinstance(item, DependencyGroupInclude):
                include_group = _normalize_name(item.include_group)

                # if a group is cyclic, record the error
                # otherwise, follow the include_group reference
                #
                # this allows us to examine all includes in a group, even in the
                # presence of errors
                if include_group in self._include_graph_ancestors.get(group, ()):
                    errors.error(
                        CyclicDependencyGroup(
                            requested_group, group, item.include_group
                        )
                    )
                else:
                    self._include_graph_ancestors[include_group] = (
                        *self._include_graph_ancestors.get(group, ()),
                        group,
                    )
                    resolved_group.extend(
                        self._resolve(include_group, requested_group, errors)
                    )
            else:  # pragma: no cover
                raise NotImplementedError(
                    f"Invalid dependency group item after parse: {item}"
                )

        # in the event that errors were detected, present the group as empty and do not
        # cache the result
        # this ensures that repeated access to a cyclic group will raise multiple errors
        if errors.errors:
            return ()

        self._resolve_cache[group] = tuple(resolved_group)
        return self._resolve_cache[group]

    def _parse_group(
        self, group: str, errors: _ErrorCollector
    ) -> tuple[Requirement | DependencyGroupInclude, ...]:
        # short circuit -- never do the work twice
        if group in self._parsed_groups:
            return self._parsed_groups[group]

        if group not in self.dependency_groups:
            errors.error(LookupError(f"Dependency group '{group}' not found"))
            return ()

        raw_group = self.dependency_groups[group]
        if isinstance(raw_group, str):
            errors.error(
                TypeError(
                    f"Dependency group {group!r} contained a string rather than a list."
                )
            )
            return ()

        if not isinstance(raw_group, Sequence):
            errors.error(
                TypeError(f"Dependency group {group!r} is not a sequence type.")
            )
            return ()

        elements: list[Requirement | DependencyGroupInclude] = []
        for item in raw_group:
            if isinstance(item, str):
                # packaging.requirements.Requirement parsing ensures that this is a
                # valid PEP 508 Dependency Specifier
                # raises InvalidRequirement on failure
                elements.append(Requirement(item))
            elif isinstance(item, Mapping):
                if tuple(item.keys()) != ("include-group",):
                    errors.error(
                        InvalidDependencyGroupObject(
                            f"Invalid dependency group item: {item!r}"
                        )
                    )
                else:
                    include_group = item["include-group"]
                    elements.append(DependencyGroupInclude(include_group=include_group))
            else:
                errors.error(TypeError(f"Invalid dependency group item: {item!r}"))

        self._parsed_groups[group] = tuple(elements)
        return self._parsed_groups[group]


# --------------------
# Functional Interface
# --------------------


def resolve_dependency_groups(
    dependency_groups: Mapping[str, Sequence[str | Mapping[str, str]]], /, *groups: str
) -> tuple[str, ...]:
    """
    Resolve a dependency group to a tuple of requirements, as strings.

    :param dependency_groups: the parsed contents of the ``[dependency-groups]`` table
        from ``pyproject.toml``
    :param groups: the name of the group(s) to resolve
    """
    resolver = DependencyGroupResolver(dependency_groups)
    return tuple(str(r) for group in groups for r in resolver.resolve(group))


# ----------------
# internal helpers
# ----------------


_NORMALIZE_PATTERN = re.compile(r"[-_.]+")


def _normalize_name(name: str) -> str:
    return _NORMALIZE_PATTERN.sub("-", name).lower()


def _normalize_group_names(
    dependency_groups: Mapping[str, Sequence[str | Mapping[str, str]]],
    errors: _ErrorCollector,
) -> dict[str, Sequence[str | Mapping[str, str]]]:
    original_names: dict[str, list[str]] = {}
    normalized_groups: dict[str, Sequence[str | Mapping[str, str]]] = {}

    for group_name, value in dependency_groups.items():
        normed_group_name = _normalize_name(group_name)
        original_names.setdefault(normed_group_name, []).append(group_name)
        normalized_groups[normed_group_name] = value

    for normed_name, names in original_names.items():
        if len(names) > 1:
            errors.error(
                DuplicateGroupNames(
                    "Duplicate dependency group names: "
                    f"{normed_name} ({', '.join(names)})"
                )
            )

    return normalized_groups
