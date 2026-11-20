"""The strict lint config may only rewrite code to syntax the oldest supported Python runs.

ruff's version-gated autofixes (UP037 strips annotation quotes, UP017/UP006/UP007 swap in newer
spellings) are safe only for ``target-version`` and newer. The Python floor is ``requires-python``
in pyproject.toml; before 3.14 annotations are evaluated eagerly, so a quote stripped for a py314
target (a method annotated with its own class, a later-defined class) raises NameError at import
on 3.11-3.13. Pinning the target to the floor keeps every such fix valid on every supported Python.
"""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_strict_ruff_target_is_the_supported_python_floor():
    floor = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["requires-python"]
    major, minor = re.search(r">=\s*(\d+)\.(\d+)", floor).groups()
    strict = tomllib.loads((ROOT / "ruff.toml").read_text(encoding="utf-8"))
    assert strict["target-version"] == f"py{major}{minor}", (
        f"ruff.strict.toml target-version {strict['target-version']!r} is not the requires-python "
        f"floor {floor!r}: its autofixes would emit syntax the oldest supported Python cannot import")
