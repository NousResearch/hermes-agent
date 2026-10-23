"""Guard the uvicorn pin against reintroducing issue #140860.

uvicorn <= 0.52.0 decided "the client asked to close" by comparing the request's
``Connection`` header against the exact tuple ``(b"connection", b"close")``.
Chrome on Windows sends the value padded (``Connection: close     ``), so the
match failed and uvicorn neither closed the connection after the response nor
served further requests on it — browsing a LAN-bound dashboard
(``hermes dashboard --host 0.0.0.0``) from that browser hung on the login page
forever. uvicorn 0.53.0 switched to the parser's own ``should_keep_alive()``,
which handles case, surrounding whitespace and comma lists.

The dashboard runs on a pinned dependency, so the regression guard is the pin
itself: this test fails if ``[web]`` allows a uvicorn older than 0.53.0.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from packaging.requirements import Requirement

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"

# First uvicorn release whose keep-alive decision tolerates a padded
# `Connection: close`. See issue #140860.
MINIMUM_UVICORN = (0, 53, 0)

# Operators that put a hard floor on what a resolver may pick.
_FLOOR_OPERATORS = frozenset({"==", "===", ">=", "~="})


def _as_tuple(release: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in release.split("."):
        digits = ""
        for char in chunk:
            if not char.isdigit():
                break
            digits += char
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def _uvicorn_floor() -> tuple[int, ...]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    specs = [
        spec
        for spec in data["project"]["optional-dependencies"].get("web", [])
        if spec.split(";")[0].strip().lower().startswith("uvicorn")
    ]
    assert specs, "the [web] extra should pin uvicorn"
    requirement = Requirement(specs[0].split(";")[0].strip())

    floors = [
        _as_tuple(spec.version)
        for spec in requirement.specifier
        if spec.operator in _FLOOR_OPERATORS
    ]
    assert floors, f"could not derive a version floor from {specs[0]!r}"
    return min(floors)


def test_web_extra_pins_uvicorn_at_least_053() -> None:
    floor = _uvicorn_floor()
    assert floor >= MINIMUM_UVICORN, (
        f"[web] allows uvicorn {'.'.join(map(str, floor))}, which can resolve below "
        "0.53.0 and hang the dashboard in Chrome over a LAN bind (#140860)"
    )
