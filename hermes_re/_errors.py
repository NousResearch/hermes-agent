"""The facade's exception class.

``error`` subclasses stdlib ``re.error`` so that existing ``except re.error:`` handlers (and code
holding a reference to the stdlib class) catch it. This is the ONLY place the facade imports
stdlib ``re``, and it takes nothing from it but that one class: no matching ever goes to stdlib.
"""
from __future__ import annotations

from re import error as _stdlib_error  # the base class only; see the module docstring


class error(_stdlib_error):
    """A pattern RE2 rejects, a pattern the facade cannot translate, or a bad replacement template.

    ``msg`` and ``pattern`` as in stdlib; ``pos``, ``lineno`` and ``colno`` are always ``None``.
    """

    def __init__(self, msg, pattern=None):
        super().__init__(msg, pattern)


PatternError = error  # the name stdlib gives this class since Python 3.13
