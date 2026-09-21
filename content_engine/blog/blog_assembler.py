"""Blog assembler - emit MDX with valid frontmatter into SahilBlog.

Writes approved:false draft posts to SAHILBLOG_REPO/src/content/blog/<slug>.mdx
with frontmatter matching the VERIFIED content.config.ts schema. Images are
copied to public/blog/<slug>/ and inline image refs rewritten to /blog/<slug>/NN.png.

Frontmatter rules (from the verified contract):
  - title, description: quoted strings
  - pubDate: UNQUOTED YYYY-MM-DD (Astro z.date() parses this; quoted strings fail)
  - tags: YAML list
  - tier: "pm" | "builder" (the only two valid values)
  - format: "essay" | "note" | "review" | "brief"
  - approved: false (always on generation; flip on approval)
  - source: "research-paper" | "gitradar" | "manual"
  - heroImage: optional, "/blog/<slug>/hero.png"
"""
from __future__ import annotations

import re
import shutil
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

from config import SAHILBLOG_REPO
from blog.blog_slug import slugify
from blog.schema_contract import normalise_frontmatter


SEMANTIC_DUPLICATE_THRESHOLD = 0.85


def _check_semantic_duplicate(
    title: str,
    posts_dir: Path,
    threshold: float = SEMANTIC_DUPLICATE_THRESHOLD,
    exclude: Optional[Path] = None,
) -> Optional[str]:
    """Check if a title is a fuzzy match for any existing post's title.

    Uses SequenceMatcher on normalised title strings (lowercase, stripped of
    punctuation). Returns the existing post's slug if a match above threshold
    is found, or None if no duplicate.

    Args:
        title: The proposed post title.
        posts_dir: Directory containing existing .mdx posts.
        threshold: Similarity ratio 0-1 (default 0.85).
        exclude: Path to exclude from the check (the post being assembled).

    Returns:
        The slug of the duplicate post, or None.
    """
    if not posts_dir.exists():
        return None

    def _normalise(t: str) -> str:
        return re.sub(r"[^\w\s]", "", t.lower()).strip()

    norm_title = _normalise(title)

    for mdx in sorted(posts_dir.glob("*.mdx")):
        if exclude and mdx.resolve() == exclude.resolve():
            continue
        try:
            text = mdx.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # Extract title from frontmatter.
        m = re.search(r'^title:\s*"?([^"\n]+?)"?\s*$', text, re.MULTILINE)
        if not m:
            continue
        existing_title = _normalise(m.group(1).strip())
        if not existing_title:
            continue
        ratio = SequenceMatcher(None, norm_title, existing_title).ratio()
        if ratio >= threshold:
            return mdx.stem  # slug

    return None


def _yaml_value(val) -> str:
    """Emit a YAML value: quoted string, unquoted bool, or list."""
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, list):
        if not val:
            return "[]"
        inner = ", ".join(_yaml_scalar(v) for v in val)
        return f"[{inner}]"
    return _yaml_scalar(val)


def _yaml_scalar(val) -> str:
    """Emit a quoted YAML scalar."""
    if val is None:
        return '""'
    s = str(val).replace('"', '\\"')
    return f'"{s}"'


def _frontmatter(fm: dict) -> str:
    """Emit YAML frontmatter block (between --- markers)."""
    lines = ["---"]
    # Order matters for readability, not for Astro (it's a map).
    if "title" in fm:
        lines.append(f"title: {_yaml_scalar(fm['title'])}")
    if "description" in fm:
        lines.append(f"description: {_yaml_scalar(fm['description'])}")
    # pubDate: unquoted YYYY-MM-DD (Astro z.date() requires this).
    pub = fm.get("pubDate")
    if pub:
        lines.append(f"pubDate: {pub}")
    if "updatedDate" in fm and fm["updatedDate"]:
        lines.append(f"updatedDate: {fm['updatedDate']}")
    if "heroImage" in fm and fm["heroImage"]:
        lines.append(f'heroImage: "{fm["heroImage"]}"')
    if "tags" in fm:
        lines.append(f"tags: {_yaml_value(fm['tags'])}")
    if "tier" in fm:
        # Enum values: emit unquoted for Astro's z.enum() compatibility.
        lines.append(f"tier: {fm['tier']}")
    if "format" in fm:
        lines.append(f"format: {fm['format']}")
    if "approved" in fm:
        lines.append(f"approved: {_yaml_value(fm['approved'])}")
    if "source" in fm:
        lines.append(f"source: {fm['source']}")
    lines.append("---")
    return "\n".join(lines)



def _has_mermaid_diagram(body_md: str) -> bool:
    """Check if the body contains a Mermaid code block."""
    import re as _re
    return bool(_re.search(r"```mermaid\n", body_md))


def _has_mapping_table(body_md: str) -> bool:
    """Check if the body contains a markdown table (primitive mapping table)."""
    # Markdown tables have | header | header | and |---|---| separator rows.
    import re as _re
    return bool(_re.search(r"\|.*\|.*\n\|[-:|\s]+\|", body_md))


def validate_blueprint(draft: dict) -> tuple[str, list[str]]:
    """Validate a blueprint-format draft has required elements.

    Returns (status, issues):
      status: "ok" | "fail"
      issues: list of missing-element strings
    """
    body = draft.get("body_md", "") or ""
    issues: list[str] = []
    if not _has_mermaid_diagram(body):
        issues.append("Blueprint post missing Mermaid diagram")
    if not _has_mapping_table(body):
        issues.append("Blueprint post missing primitive mapping table")
    return ("ok" if not issues else "fail", issues)


def _insert_section_images(body_md: str, section_paths: dict, slug: str,
                           imgs_dir: Path) -> str:
    """Copy section images and insert inline refs after their H2 headings.

    section_paths is {h2_heading: source_image_path}. Images are copied to
    imgs_dir as NN.png (01, 02, ...) and inline ![alt](/blog/<slug>/NN.png)
    refs are inserted after the matching H2 heading.
    """
    if not section_paths:
        return body_md
    out_lines = []
    idx = 0
    for line in body_md.splitlines():
        out_lines.append(line)
        m = re.match(r"^##\s+(.+?)\s*$", line)
        if m:
            heading = m.group(1).strip()
            # Match by heading text (case-insensitive, trimmed).
            for h, src in section_paths.items():
                if h.strip().lower() == heading.lower():
                    idx += 1
                    dst_name = f"{idx:02d}.png"
                    dst = imgs_dir / dst_name
                    try:
                        shutil.copy2(src, dst)
                    except Exception as exc:
                        print(f"[blog_assembler] section image copy failed: {exc}")
                        continue
                    out_lines.append("")
                    out_lines.append(f"![{heading}](/blog/{slug}/{dst_name})")
                    break
    return "\n".join(out_lines)


def _resolve_collision(posts_dir: Path, slug: str) -> Path:
    """If <slug>.mdx exists, append -N until free."""
    base = posts_dir / f"{slug}.mdx"
    if not base.exists():
        return base
    n = 2
    while True:
        candidate = posts_dir / f"{slug}-{n}.mdx"
        if not candidate.exists():
            return candidate
        n += 1


def assemble(draft: dict, images: dict, repo: Optional[str] = None,
             pub_date: Optional[str] = None) -> Path:
    """Write the MDX post into the SahilBlog repo.

    draft: {title, description, body_md, tier, tags, format, source, stream, slug}
    images: {hero_path: str|None, section_paths: {h2_heading: path}}
    repo: path to the SahilBlog repo (defaults to config.SAHILBLOG_REPO).
    pub_date: optional YYYY-MM-DD string; defaults to today.

    Returns the Path to the written MDX file.
    """
    repo_path = Path(repo) if repo else Path(SAHILBLOG_REPO)
    slug = draft.get("slug") or slugify(draft.get("title", "post"))
    posts_dir = repo_path / "src/content/blog"
    imgs_dir = repo_path / "public" / "blog" / slug
    posts_dir.mkdir(parents=True, exist_ok=True)
    imgs_dir.mkdir(parents=True, exist_ok=True)

    fm = {
        "title": draft.get("title", ""),
        "description": draft.get("description", ""),
        "pubDate": pub_date or date.today().isoformat(),
        "tags": draft.get("tags", []),
        "tier": draft.get("tier", "pm"),
        "format": draft.get("format", "essay"),
        "approved": False,
        "source": draft.get("source", "manual"),
    }
    # Enforce the Astro schema contract: clamp source/tier/format/tags to the
    # enums in content.config.ts so a draft can never break the production build.
    fm = normalise_frontmatter(fm, repo=str(repo_path))

    # Hero image.
    hero = images.get("hero_path")
    if hero and Path(hero).exists():
        shutil.copy2(hero, imgs_dir / "hero.png")
        fm["heroImage"] = f"/blog/{slug}/hero.png"

    # Section images + inline refs.
    body = _insert_section_images(
        draft.get("body_md", ""), images.get("section_paths", {}), slug, imgs_dir,
    )

    # Semantic duplicate check — prevent near-identical titles from entering.
    dup_slug = _check_semantic_duplicate(draft.get("title", ""), posts_dir)
    if dup_slug:
        raise ValueError(
            f"Semantic duplicate: '{draft.get('title', '')}' is too similar to "
            f"existing post slug '{dup_slug}' (threshold {SEMANTIC_DUPLICATE_THRESHOLD})"
        )

    mdx_path = _resolve_collision(posts_dir, slug)
    mdx_path.write_text(_frontmatter(fm) + "\n\n" + body, encoding="utf-8")
    return mdx_path