from typing import TYPE_CHECKING
from warnings import warn

warn(
    "The 'wheel' package is no longer the canonical location of the 'bdist_wheel' "
    "command, and will be removed in a future release. Please update to setuptools "
    "v70.1 or later which contains an integrated version of this command.",
    FutureWarning,
    stacklevel=1,
)

if TYPE_CHECKING:
    from ._bdist_wheel import bdist_wheel as bdist_wheel
else:
    try:
        # Better integration/compatibility with setuptools:
        # in the case new fixes or PEPs are implemented in setuptools
        # there is no need to backport them to the deprecated code base.
        # This is useful in the case of old packages in the ecosystem
        # that are still used but have low maintenance.
        from setuptools.command.bdist_wheel import bdist_wheel
    except ImportError:
        # Only used in the case of old setuptools versions.
        # If the user wants to get the latest fixes/PEPs,
        # they are encouraged to address the deprecation warning.
        from ._bdist_wheel import bdist_wheel as bdist_wheel
