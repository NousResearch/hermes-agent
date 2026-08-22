"""Guard: skill frontmatter must reference real toolset names.

`requires_toolsets` / `fallback_for_toolsets` are matched by exact string against
the toolsets a tool registered with (`registry.register(..., toolset="file")`).
A typo does not raise and is not reported anywhere — `_skill_should_show()` simply
finds the name absent from the available set and hides the skill *unconditionally*,
on every platform, for every user. The skill silently stops existing.

That is what happened to `research-paper-writing`, which declared
`requires_toolsets: [terminal, files]`. The registered toolset is `file`
(singular, `tools/file_tools.py`), so the skill could never appear:

    >>> _skill_should_show({"requires_toolsets": ["terminal", "file"]},  set(), {"terminal", "file", "web"})
    True
    >>> _skill_should_show({"requires_toolsets": ["terminal", "files"]}, set(), {"terminal", "file", "web"})
    False

This is a behaviour contract, not a snapshot: valid names are DISCOVERED from the
`registry.register(...)` calls rather than frozen in a literal, so adding or
renaming a toolset cannot make this test stale, and it fails only on a name that
no tool actually registers.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"
SKILLS_DIR = REPO_ROOT / "skills"

TOOLSET_KEYS = ("requires_toolsets", "fallback_for_toolsets")

# registry.register(name="read_file", toolset="file", ...) — the call is often
# spread over multiple lines, hence DOTALL and the non-greedy prefix.
_REGISTER_TOOLSET = re.compile(
    r"registry\.register\([^)]*?toolset\s*=\s*[\"\']([a-z_][a-z0-9_]*)[\"\']",
    re.DOTALL,
)


def _registered_toolsets() -> set[str]:
    """Toolset names some tool actually registers under.

    Scanned statically rather than by importing every tool module: several carry
    optional third-party dependencies, and a test that silently skips when an
    import fails would stop guarding anything.
    """
    names: set[str] = set()
    for py in sorted(TOOLS_DIR.glob("*.py")):
        names.update(_REGISTER_TOOLSET.findall(py.read_text(encoding="utf-8")))
    return names


def _frontmatter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8-sig")
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    try:
        data = yaml.safe_load(text[3:end])
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def _declared_toolsets(path: Path) -> list[tuple[str, str]]:
    """[(frontmatter_key, toolset_name)] declared by one SKILL.md."""
    fm = _frontmatter(path)
    hermes = ((fm.get("metadata") or {}).get("hermes") or {}) if isinstance(fm, dict) else {}
    out: list[tuple[str, str]] = []
    for source in (fm, hermes):
        if not isinstance(source, dict):
            continue
        for key in TOOLSET_KEYS:
            value = source.get(key)
            if isinstance(value, str):
                value = [v.strip() for v in value.split(",") if v.strip()]
            if isinstance(value, list):
                out.extend((key, str(v)) for v in value)
    return out


def test_toolset_scan_finds_the_known_toolsets():
    """The scan itself must work, or the guard below passes vacuously."""
    registered = _registered_toolsets()
    assert "file" in registered, f"static scan failed to find the file toolset: {sorted(registered)}"
    assert "web" in registered
    assert len(registered) > 5, f"suspiciously few toolsets discovered: {sorted(registered)}"
    # The specific typo this test exists to catch is not a real toolset.
    assert "files" not in registered


@pytest.mark.parametrize(
    "skill_file",
    sorted(SKILLS_DIR.rglob("SKILL.md")),
    ids=lambda p: str(p.relative_to(SKILLS_DIR).parent) if isinstance(p, Path) else str(p),
)
def test_bundled_skill_toolsets_are_real(skill_file: Path):
    registered = _registered_toolsets()
    for key, name in _declared_toolsets(skill_file):
        assert name in registered, (
            f"{skill_file.relative_to(REPO_ROOT)}: {key} references unknown toolset "
            f"{name!r}. A name no tool registers hides the skill unconditionally "
            f"(see _skill_should_show). Registered: {sorted(registered)}"
        )


