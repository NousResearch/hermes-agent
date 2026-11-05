"""The suites' gate for an operation a registered backend has not built yet.

``operation(store, attribute, name)`` returns ``store.<attribute>.<name>``; when the backend's repository for ``attribute`` is absent or does not define ``name``, the case skips with the registration gate's reason, so the accounting lists it as staged rather than passing it.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest


def operation(store: Any, attribute: str, name: str) -> Callable[..., Any]:
    """``store.<attribute>.<name>``, or a skip naming the operation when the backend does not offer it yet."""
    repository = getattr(store, attribute, None)
    if repository is None or not callable(getattr(type(repository), name, None)):
        pytest.skip(f"{attribute}.{name} branch not folded on this backend yet")
    return getattr(repository, name)
