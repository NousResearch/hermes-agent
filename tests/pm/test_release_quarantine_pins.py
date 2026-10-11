"""A pinned dependency has no float room, so the release quarantine must not filter it.

Regression for #133876: ``pilk==0.2.4`` (the whole ``silk`` extra) is a 2023
upload whose simple-index dates PyPI no longer serves reliably, so uv's
``exclude-newer`` filter dropped the only satisfiable version ("no publish
time") and every ``hermes pm lock`` / update dependency-sync failed with an
unsatisfiable workspace. pyproject.toml's own table comment says an exact pin
gets zero float protection from the cutoff — the exemption list is what keeps
such a pin resolvable, so the pins of the silk extra stay under this guard.
"""

from __future__ import annotations

import re
import tomllib

import pm.paths


def _repo_pyproject() -> dict:
    text = (pm.paths.repo_root() / "pyproject.toml").read_text(encoding="utf-8-sig")
    return tomllib.loads(text)


def _normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def test_silk_extra_pins_are_exempt_from_release_quarantine():
    document = _repo_pyproject()
    pinned = {
        match.group(1)
        for requirement in document["project"]["optional-dependencies"]["silk"]
        if (match := re.match(r"([A-Za-z0-9_.-]+)\s*==\s*[0-9]", requirement))
    }
    assert pinned, "the silk extra moved off exact pins; this guard needs a new trigger"
    exempt = document["tool"]["uv"]["exclude-newer-package"]
    missing = sorted(name for name in pinned if _normalized(name) not in exempt)
    assert not missing, (
        f"exact-pinned silk extras {missing} are missing from [tool.uv.exclude-newer-package]: "
        "a version the cutoff filters leaves the pin unsatisfiable "
        "(uv reports it as having no publish time)"
    )
