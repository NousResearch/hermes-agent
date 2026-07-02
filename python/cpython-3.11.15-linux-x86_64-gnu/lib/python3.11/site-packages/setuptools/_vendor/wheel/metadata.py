from warnings import warn

from ._metadata import convert_requirements as convert_requirements
from ._metadata import generate_requirements as generate_requirements
from ._metadata import pkginfo_to_metadata as pkginfo_to_metadata
from ._metadata import requires_to_requires_dist as requires_to_requires_dist
from ._metadata import safe_extra as safe_extra
from ._metadata import safe_name as safe_name
from ._metadata import split_sections as split_sections

warn(
    f"The {__name__!r} package has been made private and should no longer be imported. "
    f"Please either copy the code or find an alternative library to import it from, as "
    f"this warning will be removed in a future version of 'wheel'.",
    DeprecationWarning,
    stacklevel=2,
)
