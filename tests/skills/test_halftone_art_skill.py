"""Invariant tests for the optional halftone-art skill.

Covers ``optional-skills/creative/halftone-art`` — a browser-side halftone engine
(canvas + CSS + SVG) with die-cut sticker cutouts. These assert the contracts the
skill's own documentation depends on: frontmatter shape, every path SKILL.md
points at, the engine's public API, and alpha-preserving export. They are not
snapshots of prose, so the skill can be edited freely as long as the contracts hold.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import hermes_yaml as yaml

REPO = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO / "optional-skills" / "creative" / "halftone-art"
SKILL_MD = SKILL_DIR / "SKILL.md"

MARKETING = re.compile(
    r"\b(powerful|comprehensive|seamless|revolutionary|cutting-edge|state-of-the-art)\b",
    re.IGNORECASE,
)
MACHINE_LOCAL = re.compile(r"/home/(?!runner\b)[a-z0-9_-]+/|[A-Z]:\\\\+Users\\\\+|/Users/[a-z0-9_.-]+/")

# Templates are the skill's runnable half; each must stay a single self-contained
# ES-module page so a reader can open it without a build step.
TEMPLATES = ["starter.html", "subjects.html", "stickers.html", "css-techniques.html"]

# The engine's documented surface. SKILL.md and the React bindings import these
# names, so a rename is a breaking change to the skill, not an implementation detail.
PUBLIC_API = [
    "export function renderHalftone(",
    "export function renderHalftoneField(",
    "export function buildSubjectMask(",
    "export function plateCoverage(",
    "export function cmykCoverage(",
    "export const PRESETS",
    "export const DEFAULTS",
]

# Options the sticker workflow depends on. Documented in SKILL.md and Quick Reference.
CUTOUT_OPTIONS = ["cutout", "cutoutMode", "cutoutFill", "cutoutOutline", "cutoutTolerance"]


def _skill_text() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


def _frontmatter() -> tuple[dict, str]:
    content = _skill_text()
    assert content.startswith("---"), "SKILL.md must start with ---"
    match = re.search(r"\n---\s*\n", content[3:])
    assert match, "unclosed frontmatter"
    front = yaml.safe_load(content[3 : match.start() + 3])
    assert isinstance(front, dict), "frontmatter must be a YAML mapping"
    return front, content


def test_frontmatter_contract() -> None:
    front, _ = _frontmatter()
    missing = [
        field
        for field in ("name", "description", "version", "author", "license", "platforms")
        if field not in front
    ]
    assert not missing, f"missing frontmatter fields: {missing}"
    assert front["name"] == SKILL_DIR.name, "frontmatter name must match the directory"
    # Credit a human first: contributed skills must not read as agent-authored.
    assert "Sakib Dalal" in str(front["author"]), "author must credit the human contributor"
    assert re.fullmatch(r"\d+\.\d+\.\d+", str(front["version"])), "version must be semver"
    assert set(str(p) for p in front["platforms"]) == {"linux", "macos", "windows"}


def test_description_is_index_safe() -> None:
    front, _ = _frontmatter()
    description = str(front["description"])
    # The skill index clips the description; anything longer reaches the model cut off.
    assert len(description) <= 60, f"description is {len(description)} chars (max 60)"
    assert description.rstrip().endswith("."), "description must end with a period"
    match = MARKETING.search(description)
    assert not match, f"marketing word in description: {match.group(0)!r}"


def test_documented_files_exist() -> None:
    content = _skill_text()
    referenced = set(
        re.findall(r"`((?:references|templates)/[A-Za-z0-9_.-]+)`", content)
    )
    assert referenced, "SKILL.md should point at its references/ and templates/"
    missing = sorted(rel for rel in referenced if not (SKILL_DIR / rel).is_file())
    assert not missing, f"SKILL.md references missing files: {missing}"


def test_engine_public_api_is_intact() -> None:
    engine = (SKILL_DIR / "templates" / "halftone-core.js").read_text(encoding="utf-8")
    missing = [name for name in PUBLIC_API if name not in engine]
    assert not missing, f"engine is missing documented exports: {missing}"
    # Dependency-free is a claim the skill makes in prose; keep it true.
    assert not re.search(r"^\s*import\s.+from\s+['\"](?!\.)", engine, re.MULTILINE), (
        "halftone-core.js must not import third-party modules"
    )


def test_cutout_options_are_documented_and_defaulted() -> None:
    engine = (SKILL_DIR / "templates" / "halftone-core.js").read_text(encoding="utf-8")
    content = _skill_text()
    defaults_block = engine[engine.index("export const DEFAULTS") :]
    missing_defaults = [opt for opt in CUTOUT_OPTIONS if f"{opt}:" not in defaults_block]
    assert not missing_defaults, f"cutout options missing from DEFAULTS: {missing_defaults}"
    missing_docs = [opt for opt in CUTOUT_OPTIONS if opt not in content]
    assert not missing_docs, f"cutout options missing from SKILL.md: {missing_docs}"


@pytest.mark.parametrize("name", TEMPLATES)
def test_templates_are_self_contained(name: str) -> None:
    path = SKILL_DIR / "templates" / name
    assert path.is_file(), f"missing template: {name}"
    html = path.read_text(encoding="utf-8")
    # No CDN scripts: the pages must run from a local folder with no build step.
    external = re.findall(r"<script[^>]+src=[\"'](https?://[^\"']+)", html)
    assert not external, f"{name} loads remote scripts: {external}"
    # Canvas pixel reads need a local origin; the templates document serving over HTTP.
    if "<canvas" in html:
        assert "halftone-core.js" in html, f"{name} draws canvases but does not import the engine"


def test_sticker_export_preserves_alpha() -> None:
    stickers = (SKILL_DIR / "templates" / "stickers.html").read_text(encoding="utf-8")
    assert "image/png" in stickers, "sticker export must encode PNG (alpha is the point)"
    assert "image/jpeg" not in stickers, "JPEG would flatten the transparency"


def test_no_machine_local_paths() -> None:
    for path in SKILL_DIR.rglob("*"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        match = MACHINE_LOCAL.search(text)
        assert not match, f"{path.relative_to(SKILL_DIR)}: machine-local path {match.group(0)!r}"
