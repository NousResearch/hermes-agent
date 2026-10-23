#!/usr/bin/env python3
"""Sync memory tree output to Obsidian vault."""

import argparse
import os
import json
from datetime import datetime
from pathlib import Path


def write_markdown(vault_path: str, rel_path: str, content: str, title: str = ""):
    """Write a markdown file to Obsidian vault."""
    full_path = Path(vault_path) / rel_path
    full_path.parent.mkdir(parents=True, exist_ok=True)

    # Build frontmatter
    today = datetime.now().strftime("%Y-%m-%d")
    frontmatter = f"""---
date: {today}
title: "{title or full_path.stem}"
tags: [hermes-session, memory-tree]
---

"""
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(frontmatter + content)

    print(f"[memory-tree] Written: {full_path}")
    return str(full_path)


def update_index(vault_path: str, session_path: str, title: str):
    """Update the master index with a link to this session."""
    index_path = Path(vault_path) / "Hermes Sessions" / "INDEX.md"
    today = datetime.now().strftime("%Y-%m-%d")

    entry = f"- [{today}] {title} — [[{Path(session_path).stem}]]\n"

    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            existing = f.read()
        # Prepend to keep most recent at top
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(entry + existing)
    else:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(f"# Hermes Session Index\n\n{entry}")

    print(f"[memory-tree] Updated index: {index_path}")


def extract_topics(content: str) -> list:
    """Naive topic extraction from markdown headers."""
    import re
    topics = re.findall(r'^### (.+)$', content, re.MULTILINE)
    return topics


def create_backlinks(vault_path: str, topics: list, session_stem: str):
    """Add #hermes-session backlinks to existing topic notes."""
    for topic in topics:
        topic_note = Path(vault_path) / f"{topic}.md"
        if topic_note.exists():
            backlink = f"\n- [[{session_stem}]] (session)\n"
            with open(topic_note, "r", encoding="utf-8") as f:
                existing = f.read()
            if session_stem not in existing:
                with open(topic_note, "a", encoding="utf-8") as f:
                    f.write(backlink)
                print(f"[memory-tree] Backlinked: {topic} ← {session_stem}")


def main():
    parser = argparse.ArgumentParser(description="Sync memory tree to Obsidian")
    parser.add_argument("--vault", required=True, help="Path to Obsidian vault")
    parser.add_argument("--output", required=True, help="Relative path in vault (e.g., Hermes Sessions/session.md)")
    parser.add_argument("--content", help="Markdown content string")
    parser.add_argument("--input", help="Read content from file instead")
    parser.add_argument("--title", help="Session title", default="")
    parser.add_argument("--no-index", action="store_true", help="Skip INDEX.md update")
    parser.add_argument("--no-backlinks", action="store_true", help="Skip backlink creation")
    args = parser.parse_args()

    content = args.content
    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            content = f.read()

    if not content:
        print("[memory-tree] Error: no content provided", file=sys.stderr)
        sys.exit(1)

    title = args.title or f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    session_path = write_markdown(args.vault, args.output, content, title)

    if not args.no_index:
        update_index(args.vault, args.output, title)

    if not args.no_backlinks:
        topics = extract_topics(content)
        session_stem = Path(args.output).stem
        create_backlinks(args.vault, topics, session_stem)

    print(f"[memory-tree] Synced: {title} → {session_path}")


if __name__ == "__main__":
    import sys
    main()
