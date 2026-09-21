"""Article assembler — produce the publish bundle on disk.

Layout (mirrors the baoyu-article-illustrator skill):
  output/articles/<YYYY-MM-DD>-<slug>/
  ├── article.md         (paste-ready markdown, hero first)
  ├── outline.md         (reproducibility / regen record)
  ├── imgs/              (NN-<type>-<slug>.png copies)
  └── prompts/           (NN-<type>-<slug>.md, written by the illustrator)

The bundle is a value object: {dir, article_md, article_md_path,
image_paths, title, lede}. dry_run=True returns the shape without any
disk write.
"""
from __future__ import annotations
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional


OUTPUT_ROOT = Path(__file__).resolve().parent / "output" / "articles"


def _slug_from_title(title: str) -> str:
    """2-4 words, kebab-case, lower-case. Falls back to 'article'."""
    words = re.findall(r"[a-z0-9]+", (title or "").lower())
    if len(words) < 2:
        return "article"
    return "-".join(words[:4]) or "article"


def _resolve_collision(base: Path) -> Path:
    """Append -YYYYMMDD-HHMMSS to a colliding path so repeated bundles don't clobber."""
    if not base.exists():
        return base
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return base.with_name(f"{base.name}-{stamp}")


def _lede(body_md: str, n: int = 240) -> str:
    """First 1-3 non-heading paragraphs, truncated to n chars."""
    parts: list[str] = []
    for line in body_md.splitlines():
        s = line.strip()
        if not s:
            if parts:
                break
            continue
        if s.startswith("#"):
            continue
        parts.append(s)
        if len(" ".join(parts)) > n:
            break
    text = " ".join(parts)
    return text[:n]


def _date_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


@dataclass
class ArticleBundle:
    dir: Path
    article_md: str
    article_md_path: Path
    image_paths: List[Path] = field(default_factory=list)
    title: str = ""
    lede: str = ""
    mode: str = ""
    pillar: str = ""


def bundle(illustrated: dict, draft: dict, out_root: Optional[Path] = None,
           dry_run: bool = False) -> ArticleBundle:
    """Compose the publish bundle. Returns the value object.

    `illustrated` is the dict returned by article_illustrator.illustrate:
      {body_md, images, outline_path}.
    `draft` is the ArticleDraft from article_generator.write.
    """
    out_root = Path(out_root) if out_root else OUTPUT_ROOT
    title = (draft.get("title") or illustrated.get("title") or "Article").strip()
    slug = draft.get("slug") or _slug_from_title(title)
    base = out_root / f"{_date_stamp()}-{slug}"
    body = (illustrated.get("body_md") or "").strip()
    lede = _lede(body)

    if dry_run:
        # Resolve a virtual path; do not create anything.
        return ArticleBundle(
            dir=_resolve_collision(base),
            article_md=body,
            article_md_path=base / "article.md",
            image_paths=[Path(p) for p in illustrated.get("images", [])],
            title=title, lede=lede,
            mode=draft.get("mode", ""), pillar=draft.get("pillar", ""),
        )

    target = _resolve_collision(base)
    target.mkdir(parents=True, exist_ok=True)
    (target / "imgs").mkdir(exist_ok=True)
    (target / "prompts").mkdir(exist_ok=True)

    # Copy outline.md (reproducibility) if the illustrator wrote one
    # alongside its tmp area.
    outline_src = illustrated.get("outline_path")
    if outline_src and Path(outline_src).exists():
        shutil.copy2(outline_src, target / "outline.md")

    # Copy images.
    image_paths: List[Path] = []
    for src in illustrated.get("images", []) or []:
        src_p = Path(src)
        if not src_p.exists():
            continue
        dst = target / "imgs" / src_p.name
        try:
            shutil.copy2(src_p, dst)
        except Exception as exc:  # noqa: BLE001
            print(f"[article_assembler] image copy failed: {exc}")
            continue
        image_paths.append(dst)

    # Write the paste-ready article.
    article_path = target / "article.md"
    article_path.write_text(body, encoding="utf-8")

    return ArticleBundle(
        dir=target, article_md=body, article_md_path=article_path,
        image_paths=image_paths, title=title, lede=lede,
        mode=draft.get("mode", ""), pillar=draft.get("pillar", ""),
    )
