"""The purpose of this module is implement PEP 621 validations that are
difficult to express as a JSON Schema (or that are not supported by the current
JSON Schema library).
"""

import collections
import itertools
from inspect import cleandoc
from typing import Generator, Iterable, Mapping, TypeVar

from .error_reporting import ValidationError

T = TypeVar("T", bound=Mapping)


class RedefiningStaticFieldAsDynamic(ValidationError):
    _DESC = """According to PEP 621:

    Build back-ends MUST raise an error if the metadata specifies a field
    statically as well as being listed in dynamic.
    """
    __doc__ = _DESC
    _URL = (
        "https://packaging.python.org/en/latest/specifications/pyproject-toml/#dynamic"
    )


class IncludedDependencyGroupMustExist(ValidationError):
    _DESC = """An included dependency group must exist and must not be cyclic.
    """
    __doc__ = _DESC
    _URL = "https://peps.python.org/pep-0735/"


class ImportNameCollision(ValidationError):
    _DESC = """According to PEP 794:

    All import-names and import-namespaces items must be unique.
    """
    __doc__ = _DESC
    _URL = "https://peps.python.org/pep-0794/"


class ImportNameMissing(ValidationError):
    _DESC = """According to PEP 794:

    An import name must have all parents listed.
    """
    __doc__ = _DESC
    _URL = "https://peps.python.org/pep-0794/"


def validate_project_dynamic(pyproject: T) -> T:
    project_table = pyproject.get("project", {})
    dynamic = project_table.get("dynamic", [])

    for field in dynamic:
        if field in project_table:
            raise RedefiningStaticFieldAsDynamic(
                message=f"You cannot provide a value for `project.{field}` and "
                "list it under `project.dynamic` at the same time",
                value={
                    field: project_table[field],
                    "...": " # ...",
                    "dynamic": dynamic,
                },
                name=f"data.project.{field}",
                definition={
                    "description": cleandoc(RedefiningStaticFieldAsDynamic._DESC),
                    "see": RedefiningStaticFieldAsDynamic._URL,
                },
                rule="PEP 621",
            )

    return pyproject


def validate_include_depenency(pyproject: T) -> T:
    dependency_groups = pyproject.get("dependency-groups", {})
    for key, value in dependency_groups.items():
        for each in value:
            if (
                isinstance(each, dict)
                and (include_group := each.get("include-group"))
                and include_group not in dependency_groups
            ):
                raise IncludedDependencyGroupMustExist(
                    message=f"The included dependency group {include_group} doesn't exist",
                    value=each,
                    name=f"data.dependency_groups.{key}",
                    definition={
                        "description": cleandoc(IncludedDependencyGroupMustExist._DESC),
                        "see": IncludedDependencyGroupMustExist._URL,
                    },
                    rule="PEP 735",
                )
    # TODO: check for `include-group` cycles (can be conditional to graphlib)
    return pyproject


def _remove_private(items: Iterable[str]) -> Generator[str, None, None]:
    for item in items:
        yield item.partition(";")[0].rstrip()


def validate_import_name_issues(pyproject: T) -> T:
    project = pyproject.get("project", {})
    import_names = collections.Counter(_remove_private(project.get("import-names", [])))
    import_namespaces = collections.Counter(
        _remove_private(project.get("import-namespaces", []))
    )

    duplicated = [k for k, v in (import_names + import_namespaces).items() if v > 1]

    if duplicated:
        raise ImportNameCollision(
            message="Duplicated names are not allowed in import-names/import-namespaces",
            value=duplicated,
            name="data.project.importnames(paces)",
            definition={
                "description": cleandoc(ImportNameCollision._DESC),
                "see": ImportNameCollision._URL,
            },
            rule="PEP 794",
        )

    names = frozenset(import_names + import_namespaces)
    for name in names:
        for parent in itertools.accumulate(
            name.split(".")[:-1], lambda a, b: f"{a}.{b}"
        ):
            if parent not in names:
                raise ImportNameMissing(
                    message="All parents of an import name must also be listed in import-namespace/import-names",
                    value=name,
                    name="data.project.importnames(paces)",
                    definition={
                        "description": cleandoc(ImportNameMissing._DESC),
                        "see": ImportNameMissing._URL,
                    },
                    rule="PEP 794",
                )

    return pyproject


EXTRA_VALIDATIONS = (
    validate_project_dynamic,
    validate_include_depenency,
    validate_import_name_issues,
)
