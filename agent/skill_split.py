"""Oversized-skill SPLIT operation (curator.split_over_kb).

Carves an over-threshold ``SKILL.md`` into ``references/*.md`` section files
plus a lean pointer ``SKILL.md`` — content-preserving and reversible.

Contract (see the curator-scope-shared-skills spec, Invariant 5 / Phase 4):

- Only fires when ``SKILL.md`` is larger than the threshold.
- Frontmatter bytes are IDENTICAL before and after the split (a mangled
  ``name:`` would silently break discovery).
- Every carved section is replaced in-place by a pointer line matching the
  FIXED template ``### <title> -> see references/<slug>.md`` (the whitelist
  is a fixed regex, never an open-ended "lines the splitter injected" set).
- Reconstruction proof: splicing each carved file's exact bytes back over
  its pointer line reproduces the original body byte-for-byte. This is an
  ordered, duplicate-sensitive equality — strictly stronger than a set of
  line hashes (which is order- and duplicate-blind and explicitly rejected).
- ``references/.split-manifest.json`` records the carve map so a future
  join is mechanical (``join_split_skill``).
- TERMINAL FAILURE: when frontmatter + the kept preamble + the pointer list
  alone cannot fit under the cap, the skill is left UNSPLIT and reported —
  never mangle frontmatter or loop forever.
- A slug colliding with an existing ``references/`` file is deduped, never
  overwritten.
- A ``SKILL.md`` with no ``## `` headings falls back to size-chunked
  ``references/part-NN.md`` carves with the same pointer/splice contract.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# The one and only pointer-line template. Verification removes/replaces ONLY
# lines matching this regex — a fixed whitelist, asserted to match exactly
# the number of carved sections.
POINTER_RE = re.compile(r"^### .+ -> see references/[\w-]+\.md$")
POINTER_TEMPLATE = "### {title} -> see references/{slug}.md"

MANIFEST_NAME = ".split-manifest.json"

# Hard ceiling independent of the configured threshold: the skill_manage
# patch cap (tools/skill_manager_tool.MAX_SKILL_CONTENT_CHARS).
PATCH_CAP_BYTES = 100_000

# Sections whose headings look session-specific / long-tail are carved first.
_CARVE_FIRST_RE = re.compile(
    r"pending-|lesson|repro|\d{4}-\d{2}-\d{2}", re.IGNORECASE
)

_HEADING_RE = re.compile(r"^## +(.+?)\s*$", re.MULTILINE)

# Fallback chunk size for heading-less bodies.
_PART_CHUNK_BYTES = 60_000


def split_frontmatter(text: str) -> Tuple[str, str]:
    """Split raw SKILL.md text into (frontmatter_bytes, body_bytes).

    The frontmatter string includes both ``---`` fences and the trailing
    newline so ``frontmatter + body == text`` exactly. Returns ``("", text)``
    when there is no frontmatter.
    """
    if not text.startswith("---\n"):
        return "", text
    end = text.find("\n---\n", 4)
    if end == -1:
        return "", text
    cut = end + len("\n---\n")
    return text[:cut], text[cut:]


def _slugify(title: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", title).strip("-").lower()
    return slug or "section"


def _dedupe_slug(slug: str, taken: set) -> str:
    if slug not in taken:
        return slug
    n = 2
    while f"{slug}-{n}" in taken:
        n += 1
    return f"{slug}-{n}"


def _parse_sections(body: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Parse *body* into (preamble, sections).

    A section starts at a ``## `` heading and runs to the next one (or EOF).
    The preamble is everything before the first heading. Section text
    includes the heading line — ``preamble + "".join(s["text"])`` equals
    *body* exactly.
    """
    matches = list(_HEADING_RE.finditer(body))
    if not matches:
        return body, []
    preamble = body[: matches[0].start()]
    sections: List[Dict[str, Any]] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        sections.append({"title": m.group(1), "text": body[start:end]})
    return preamble, sections


def _pointer_line(title: str, slug: str) -> str:
    # Titles containing newlines are impossible (regex is single-line); strip
    # anything that could break the single-line pointer contract.
    clean = re.sub(r"\s+", " ", title).strip() or "section"
    return POINTER_TEMPLATE.format(title=clean, slug=slug)


def verify_split(original_text: str, lean_text: str,
                 carves: List[Dict[str, str]]) -> Optional[str]:
    """Verify the content-preservation invariant. Returns an error or None.

    - frontmatter bytes identical
    - splicing each carve's exact content over its pointer line (in document
      order, exact-match, at line start) reproduces the original body
      byte-for-byte — an ORDERED, duplicate-sensitive equality, strictly
      stronger than a set of line hashes
    - every pointer matches the FIXED template regex, and exactly one pointer
      is spliced per carve (removed-line count == carve count)

    Pre-existing pointer-style lines in the original content are unaffected:
    only the exact pointer strings this split minted are spliced (slug
    dedupe guarantees they are unique within the document).
    """
    orig_fm, orig_body = split_frontmatter(original_text)
    lean_fm, lean_body = split_frontmatter(lean_text)
    if orig_fm != lean_fm:
        return "frontmatter changed by split"

    reconstructed = lean_body
    search_from = 0
    for carve in carves:
        pointer = carve["pointer"]
        if not POINTER_RE.match(pointer):
            return f"carve pointer does not match the fixed template: {pointer!r}"
        needle = pointer + "\n"
        idx = reconstructed.find(needle, search_from)
        if idx == -1:
            return f"pointer line not found in lean body: {pointer!r}"
        if idx > 0 and reconstructed[idx - 1] != "\n":
            return f"pointer not at line start: {pointer!r}"
        content = carve["content"]
        reconstructed = (
            reconstructed[:idx] + content + reconstructed[idx + len(needle):]
        )
        search_from = idx + len(content)
    if reconstructed != orig_body:
        return "reconstructed body differs from original"
    return None


def _build_lean_and_carves(
    frontmatter: str,
    preamble: str,
    sections: List[Dict[str, Any]],
    carve_titles: List[int],
    slugs: Dict[int, str],
) -> Tuple[str, List[Dict[str, str]]]:
    """Assemble the lean SKILL.md text + ordered carve records."""
    carve_set = set(carve_titles)
    parts: List[str] = [preamble]
    carves: List[Dict[str, str]] = []
    for idx, sec in enumerate(sections):
        if idx in carve_set:
            slug = slugs[idx]
            pointer = _pointer_line(sec["title"], slug)
            # Pointer replaces the section; keep a single newline after it so
            # the document stays readable. Splice contract: the pointer LINE
            # (sans newline) maps to the section's exact bytes.
            parts.append(pointer + "\n")
            carves.append({
                "slug": slug,
                "file": f"references/{slug}.md",
                "title": sec["title"],
                "pointer": pointer,
                "content": sec["text"],
            })
        else:
            parts.append(sec["text"])
    return frontmatter + "".join(parts), carves


def plan_split(skill_dir: Path, threshold_kb: int) -> Dict[str, Any]:
    """Compute a split plan for *skill_dir* without writing anything.

    Returns a dict with ``action`` one of ``skip`` (under threshold),
    ``split`` (plan ready) or ``unsplittable`` (terminal failure — leave
    unsplit, report).
    """
    skill_md = skill_dir / "SKILL.md"
    result: Dict[str, Any] = {"skill_dir": str(skill_dir), "action": "skip"}
    if threshold_kb <= 0 or not skill_md.exists():
        return result
    raw = skill_md.read_text(encoding="utf-8")
    size = len(raw.encode("utf-8"))
    threshold = min(threshold_kb * 1024, PATCH_CAP_BYTES)
    result["size"] = size
    if size <= threshold:
        return result

    frontmatter, body = split_frontmatter(raw)
    preamble, sections = _parse_sections(body)

    refs_dir = skill_dir / "references"
    taken = set()
    if refs_dir.is_dir():
        taken = {p.stem for p in refs_dir.glob("*.md")}

    if not sections:
        # Heading-less blob: size-chunk the body beyond the preamble head.
        return _plan_partnn(
            result, frontmatter, body, threshold, taken
        )

    # Carve order: session-specific/long-tail headings first (largest first),
    # then remaining sections largest first.
    flagged = [i for i, s in enumerate(sections)
               if _CARVE_FIRST_RE.search(s["title"])]
    rest = [i for i in range(len(sections)) if i not in set(flagged)]
    flagged.sort(key=lambda i: len(sections[i]["text"]), reverse=True)
    rest.sort(key=lambda i: len(sections[i]["text"]), reverse=True)
    order = flagged + rest

    carved: List[int] = []
    slugs: Dict[int, str] = {}
    lean_text = raw
    carves: List[Dict[str, str]] = []
    for idx in order:
        slug = _dedupe_slug(_slugify(sections[idx]["title"]),
                            taken | {s for s in slugs.values()})
        slugs[idx] = slug
        carved.append(idx)
        # carve records must be in DOCUMENT order for the splice proof
        doc_order = sorted(carved)
        lean_text, carves = _build_lean_and_carves(
            frontmatter, preamble, sections, doc_order,
            slugs,
        )
        if len(lean_text.encode("utf-8")) <= threshold:
            break

    lean_size = len(lean_text.encode("utf-8"))
    if lean_size > threshold:
        # Terminal failure: even carving everything can't fit.
        result["action"] = "unsplittable"
        result["reason"] = (
            f"lean SKILL.md would still be {lean_size} B > {threshold} B "
            "after carving every section; leaving unsplit"
        )
        return result

    err = verify_split(raw, lean_text, carves)
    if err is not None:
        result["action"] = "unsplittable"
        result["reason"] = f"content-preservation verification failed: {err}"
        return result

    result.update({
        "action": "split",
        "lean_text": lean_text,
        "carves": carves,
        "post_size": lean_size,
        "original_text": raw,
    })
    return result


def _plan_partnn(result: Dict[str, Any], frontmatter: str, body: str,
                 threshold: int, taken: set) -> Dict[str, Any]:
    """Fallback plan for heading-less bodies: references/part-NN.md chunks."""
    overhead = len(frontmatter.encode("utf-8"))
    keep = max(threshold // 4, 4096)
    head = body[:keep]
    # Never split mid-line: retreat to the last newline in the kept head.
    nl = head.rfind("\n")
    if nl > 0:
        head = head[: nl + 1]
    remainder = body[len(head):]
    if not remainder:
        result["action"] = "unsplittable"
        result["reason"] = "no headings and nothing carvable"
        return result

    carves: List[Dict[str, str]] = []
    pointers: List[str] = []
    chunk_no = 1
    pos = 0
    while pos < len(remainder):
        end = min(pos + _PART_CHUNK_BYTES, len(remainder))
        if end < len(remainder):
            nl = remainder.rfind("\n", pos, end)
            if nl > pos:
                end = nl + 1
        chunk = remainder[pos:end]
        slug = _dedupe_slug(f"part-{chunk_no:02d}",
                            taken | {c["slug"] for c in carves})
        pointer = _pointer_line(f"Part {chunk_no:02d}", slug)
        carves.append({
            "slug": slug,
            "file": f"references/{slug}.md",
            "title": f"Part {chunk_no:02d}",
            "pointer": pointer,
            "content": chunk,
        })
        pointers.append(pointer + "\n")
        pos = end
        chunk_no += 1

    lean_text = frontmatter + head + "".join(pointers)
    lean_size = len(lean_text.encode("utf-8"))
    if lean_size > threshold:
        result["action"] = "unsplittable"
        result["reason"] = (
            f"heading-less fallback lean size {lean_size} B > {threshold} B"
        )
        return result

    original = frontmatter + body
    err = verify_split(original, lean_text, carves)
    if err is not None:
        result["action"] = "unsplittable"
        result["reason"] = f"content-preservation verification failed: {err}"
        return result

    result.update({
        "action": "split",
        "lean_text": lean_text,
        "carves": carves,
        "post_size": lean_size,
        "original_text": original,
        "fallback": "part-NN",
    })
    return result


def execute_split(skill_dir: Path, plan: Dict[str, Any]) -> List[Path]:
    """Apply a ``plan_split`` plan. Returns the list of files written.

    Verification already passed at plan time; re-verify defensively before
    touching disk (the plan could be stale if the file changed since).
    """
    skill_md = skill_dir / "SKILL.md"
    current = skill_md.read_text(encoding="utf-8")
    if current != plan.get("original_text"):
        raise RuntimeError(
            f"SKILL.md changed since the split was planned: {skill_md}"
        )
    err = verify_split(current, plan["lean_text"], plan["carves"])
    if err is not None:
        raise RuntimeError(f"split verification failed at execute: {err}")

    refs_dir = skill_dir / "references"
    refs_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for carve in plan["carves"]:
        dest = skill_dir / carve["file"]
        if dest.exists():
            raise RuntimeError(f"carve target already exists: {dest}")
        dest.write_text(carve["content"], encoding="utf-8")
        written.append(dest)

    manifest_path = refs_dir / MANIFEST_NAME
    manifest = {
        "version": 1,
        "carves": [
            {k: c[k] for k in ("slug", "file", "title", "pointer")}
            for c in plan["carves"]
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    written.append(manifest_path)

    skill_md.write_text(plan["lean_text"], encoding="utf-8")
    written.append(skill_md)
    return written


def join_split_skill(skill_dir: Path) -> bool:
    """Mechanically reverse a split using the manifest. Returns True on join.

    Splices every carved file back over its pointer line, deletes the carve
    files + manifest. The inverse of ``execute_split`` — used by tests to
    prove reversibility and available for manual recovery.
    """
    refs_dir = skill_dir / "references"
    manifest_path = refs_dir / MANIFEST_NAME
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    carves = manifest.get("carves") or []
    skill_md = skill_dir / "SKILL.md"
    text = skill_md.read_text(encoding="utf-8")
    fm, body = split_frontmatter(text)

    # P0 FIX: Validate every carve path stays inside skill_dir (path traversal protection)
    skill_dir_resolved = skill_dir.resolve()
    contents: List[Dict[str, str]] = []
    for c in carves:
        carve_path_str = c["file"]
        # Reject absolute paths
        if Path(carve_path_str).is_absolute():
            raise RuntimeError(
                f"join_split_skill: absolute path in manifest rejected: {carve_path_str}"
            )
        src = skill_dir / carve_path_str
        # Check for symlinks BEFORE resolving (lstat doesn't follow symlinks)
        try:
            if src.lstat().st_mode & 0o170000 == 0o120000:  # S_IFLNK
                # It's a symlink; resolve and verify containment
                resolved = src.resolve(strict=True)
                try:
                    resolved.relative_to(skill_dir_resolved)
                except ValueError:
                    raise RuntimeError(
                        f"join_split_skill: symlink escape rejected: {carve_path_str} -> {resolved}"
                    )
        except (OSError, ValueError):
            pass  # File doesn't exist yet or resolve failed; containment check below will catch it
        
        # Require the resolved path to be contained within skill_dir
        try:
            src_resolved = src.resolve(strict=False)
            src_resolved.relative_to(skill_dir_resolved)
        except ValueError:
            raise RuntimeError(
                f"join_split_skill: path traversal rejected: {carve_path_str} escapes skill_dir"
            )
        
        contents.append({
            **c,
            "content": src.read_text(encoding="utf-8"),
        })

    joined = body
    search_from = 0
    hit = 0
    for c in contents:
        needle = c["pointer"] + "\n"
        idx = joined.find(needle, search_from)
        if idx == -1:
            break
        joined = joined[:idx] + c["content"] + joined[idx + len(needle):]
        search_from = idx + len(c["content"])
        hit += 1
    if hit != len(contents):
        raise RuntimeError(
            f"join found {hit} pointers for {len(contents)} carves"
        )
    skill_md.write_text(fm + joined, encoding="utf-8")
    for c in contents:
        try:
            # Re-validate before unlink (defense in depth)
            carve_path = skill_dir / c["file"]
            carve_resolved = carve_path.resolve(strict=False)
            carve_resolved.relative_to(skill_dir_resolved)
            carve_path.unlink()
        except (OSError, ValueError):
            pass
    try:
        manifest_path.unlink()
    except OSError:
        pass
    return True
