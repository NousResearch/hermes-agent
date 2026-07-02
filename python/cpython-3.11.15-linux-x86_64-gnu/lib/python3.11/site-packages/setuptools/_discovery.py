import functools
import operator

import packaging.requirements


# from coherent.build.discovery
def extras_from_dep(dep):
    try:
        markers = packaging.requirements.Requirement(dep).marker._markers
    except AttributeError:
        markers = ()
    return set(
        marker[2].value
        for marker in markers
        if isinstance(marker, tuple) and marker[0].value == 'extra'
    )


def extras_from_deps(deps):
    """
    >>> extras_from_deps(['requests'])
    set()
    >>> extras_from_deps(['pytest; extra == "test"'])
    {'test'}
    >>> sorted(extras_from_deps([
    ...     'requests',
    ...     'pytest; extra == "test"',
    ...     'pytest-cov; extra == "test"',
    ...     'sphinx; extra=="doc"']))
    ['doc', 'test']
    """
    return functools.reduce(operator.or_, map(extras_from_dep, deps), set())
