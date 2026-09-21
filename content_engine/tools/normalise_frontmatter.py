"""One-time frontmatter normalisation for existing SahilBlog MDX files.

Normalises:
  - Unquote enum values: font: "essay" → font: essay
  - Quote title/description: title: Value → title: "Value"
  - Ensure pubDate is unquoted YYYY-MM-DD
  - Ensure approved/source use correct YAML type

Run with: python -m tools.normalise_frontmatter --posts-dir <dir> [--dry-run]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


# Fields that should be unquoted enum values.
_ENUM_FIELDS = {"format", "tier", "source", "approved"}
# Fields that should be quoted strings.
_STRING_FIELDS = {"title", "description", "heroImage"}
# Fields that should be unquoted dates (YYYY-MM-DD).
_DATE_FIELDS = {"pubDate", "updatedDate"}


def _normalise_field(key: str, val: str) -> str:
    """Normalise a single frontmatter field value."""
    val = val.strip()

    if key in _ENUM_FIELDS:
        # Strip quotes: "essay" → essay
        return val.strip('"').strip("'")

    if key in _STRING_FIELDS:
        # Ensure quoted.
        if not (val.startswith('"') and val.endswith('"')):
            escaped = val.replace('"', '\\"')
            return f'"{escaped}"'
        return val

    if key in _DATE_FIELDS:
        # Ensure unquoted YYYY-MM-DD.
        return val.strip('"').strip("'")

    if key == "tags":
        # Leave tags as-is (already valid YAML list).
        return val

    return val


def _normalise_frontmatter_block(text: str) -> tuple[str, list[str]]:
    """Normalise frontmatter in an MDX file.

    Returns (normalised_text, list_of_changes).
    """
    if not text.startswith("---"):
        return text, []

    end = text.find("---", 3)
    if end == -1:
        return text, []

    fm_block = text[3:end]
    changes: list[str] = []

    lines = fm_block.splitlines()
    new_lines = []
    for line in lines:
        if ":" not in line or line.strip().startswith("#"):
            new_lines.append(line)
            continue

        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()

        normalised = _normalise_field(key, val)
        if normalised != val:
            changes.append(f"  {key}: {val!r} → {normalised!r}")
        new_lines.append(f"{key}: {normalised}")

    new_fm = "\n".join(new_lines)
    new_text = "---" + new_fm + text[end:]
    return new_text, changes


def normalise_file(mdx_path: Path, dry_run: bool = False) -> list[str]:
    """Normalise a single MDX file. Returns list of change descriptions."""
    text = mdx_path.read_text(encoding="utf-8", errors="replace")
    new_text, changes = _normalise_frontmatter_block(text)

    if changes and not dry_run:
        mdx_path.write_text(new_text, encoding="utf-8")

    return changes


def normalise_all(posts_dir: Path, dry_run: bool = False) -> dict:
    """Normalise all MDX files in a directory.

    Returns {"total": int, "changed": int, "details": {slug: [changes]}}.
    """
    details: dict[str, list[str]] = {}
    changed = 0

    for mdx in sorted(posts_dir.glob("*.mdx")):
        changes = normalise_file(mdx, dry_run=dry_run)
        if changes:
            details[mdx.stem] = changes
            changed += 1

    return {
        "total": len(list(posts_dir.glob("*.mdx"))),
        "changed": changed,
        "details": details,
    }


def main():
    """CLI entry."""
    import argparse
    parser = argparse.ArgumentParser(description="Normalise MDX frontmatter")
    parser.add_argument("--posts-dir", required=True, help="Path to src/content/blog")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing")
    args = parser.parse_args()

    posts_dir = Path(args.posts_dir)
    if not posts_dir.exists():
        print(f"Error: {posts_dir} does not exist", file=sys.stderr)
        return 1

    report = normalise_all(posts_dir, dry_run=args.dry_run)
    mode = "DRY RUN" if args.dry_run else "LIVE"
    print(f"Frontmatter normalisation ({mode})")
    print(f"  Total files: {report['total']}")
    print(f"  Changed: {report['changed']}")

    for slug, changes in report["details"].items():
        print(f"\n  {slug}:")
        for c in changes:
            print(f"    {c}")

    return 0


if __name__ == "__main__":
    sys.exit(main())