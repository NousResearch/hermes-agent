#!/usr/bin/env python3
"""
zotero_note.py — Create, update, list, and tag notes on Zotero items.

Notes in Zotero are items with itemType "note" that are children of a parent item.
The note body is stored as HTML; this script handles plain text / markdown conversion.

Usage:
  python zotero_note.py list ITEM_KEY
  python zotero_note.py show NOTE_KEY
  python zotero_note.py create ITEM_KEY --title "Reading notes" --body "Key points..."
  python zotero_note.py create ITEM_KEY --title "Summary" --file notes.md
  python zotero_note.py create ITEM_KEY --template reading
  python zotero_note.py update NOTE_KEY --body "Replacement text"
  python zotero_note.py update NOTE_KEY --append "Additional paragraph"
  python zotero_note.py status ITEM_KEY unread
  python zotero_note.py status ITEM_KEY reading
  python zotero_note.py status ITEM_KEY done
  python zotero_note.py delete NOTE_KEY
"""

import argparse
import html
import os
import re
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("requests is required: pip install requests")

API_BASE = "https://api.zotero.org"
HEADERS: dict = {}
USER_ID: str = ""

PROGRESS_TAGS = {"unread", "reading", "done"}


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

TEMPLATES = {
    "reading": """## Summary
[2-3 sentence overview of the work]

## Key Points
- 
- 
- 

## Notable Quotes
> "..." (p. X)

## Questions / Gaps
- 

## Related Work
- 

## My Assessment
""",
    "book": """## Overview
[What is this book about?]

## Chapter Notes
### Chapter 1: 

## Key Takeaways
1. 
2. 
3. 

## Favourite Passages
> "..." (p. X)

## Would Recommend To
""",
    "quick": """## Quick Notes
[Stream-of-consciousness first impressions]

## Follow-up
- [ ] 
""",
}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def get_env() -> tuple[str, str]:
    api_key = os.environ.get("ZOTERO_API_KEY", "")
    user_id = os.environ.get("ZOTERO_USER_ID", "")
    if not api_key or not user_id:
        sys.exit(
            "Set ZOTERO_API_KEY and ZOTERO_USER_ID environment variables.\n"
            "  Get them at: https://www.zotero.org/settings/keys"
        )
    return api_key, user_id


def build_headers(api_key: str) -> dict:
    return {
        "Zotero-API-Version": "3",
        "Zotero-API-Key": api_key,
        "Content-Type": "application/json",
    }


# ---------------------------------------------------------------------------
# HTML / text helpers
# ---------------------------------------------------------------------------

def text_to_html(text: str) -> str:
    """Convert plain text / basic markdown to Zotero note HTML."""
    lines = text.split("\n")
    html_parts = []
    in_list = False

    for line in lines:
        stripped = line.strip()

        # Headings
        if stripped.startswith("### "):
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            html_parts.append(f"<h3>{html.escape(stripped[4:])}</h3>")
        elif stripped.startswith("## "):
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            html_parts.append(f"<h2>{html.escape(stripped[3:])}</h2>")
        elif stripped.startswith("# "):
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            html_parts.append(f"<h1>{html.escape(stripped[2:])}</h1>")
        # Blockquote
        elif stripped.startswith("> "):
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            html_parts.append(f"<blockquote><p>{html.escape(stripped[2:])}</p></blockquote>")
        # List items
        elif stripped.startswith("- ") or stripped.startswith("* "):
            if not in_list:
                html_parts.append("<ul>")
                in_list = True
            content = stripped[2:]
            # Handle checkboxes
            if content.startswith("[ ] "):
                content = "☐ " + content[4:]
            elif content.startswith("[x] ") or content.startswith("[X] "):
                content = "☑ " + content[4:]
            html_parts.append(f"<li>{html.escape(content)}</li>")
        # Numbered list
        elif re.match(r"^\d+\. ", stripped):
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            content = re.sub(r"^\d+\. ", "", stripped)
            html_parts.append(f"<p>{html.escape(content)}</p>")
        # Blank line
        elif not stripped:
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            html_parts.append("")
        # Regular paragraph
        else:
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            # Inline bold/italic
            content = html.escape(stripped)
            content = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", content)
            content = re.sub(r"\*(.+?)\*", r"<em>\1</em>", content)
            content = re.sub(r"`(.+?)`", r"<code>\1</code>", content)
            html_parts.append(f"<p>{content}</p>")

    if in_list:
        html_parts.append("</ul>")

    return "\n".join(p for p in html_parts if p != "")


def html_to_text(html_str: str) -> str:
    """Minimal HTML → plain text for display."""
    if not html_str:
        return ""
    text = re.sub(r"<h[1-3][^>]*>(.*?)</h[1-3]>", lambda m: "\n## " + m.group(1) + "\n", html_str, flags=re.DOTALL)
    text = re.sub(r"<blockquote[^>]*>(.*?)</blockquote>", lambda m: "\n> " + m.group(1).strip() + "\n", text, flags=re.DOTALL)
    text = re.sub(r"<li[^>]*>(.*?)</li>", lambda m: "  - " + m.group(1).strip(), text, flags=re.DOTALL)
    text = re.sub(r"<p[^>]*>(.*?)</p>", lambda m: m.group(1).strip() + "\n", text, flags=re.DOTALL)
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_note_html(title: str | None, body: str) -> str:
    """Wrap body HTML with an optional title heading."""
    body_html = text_to_html(body)
    if title:
        return f"<h1>{html.escape(title)}</h1>\n{body_html}"
    return body_html


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def get_item(item_key: str) -> dict:
    resp = requests.get(f"{API_BASE}/users/{USER_ID}/items/{item_key}", headers=HEADERS)
    if resp.status_code == 404:
        sys.exit(f"Item not found: {item_key}")
    resp.raise_for_status()
    return resp.json()


def get_children(item_key: str) -> list[dict]:
    resp = requests.get(f"{API_BASE}/users/{USER_ID}/items/{item_key}/children", headers=HEADERS)
    resp.raise_for_status()
    return resp.json()


def create_note(parent_key: str, note_html: str, tags: list[str] | None = None) -> str:
    """Create a child note. Returns the new note's key."""
    payload = [{
        "itemType": "note",
        "parentItem": parent_key,
        "note": note_html,
        "tags": [{"tag": t} for t in (tags or [])],
    }]
    resp = requests.post(
        f"{API_BASE}/users/{USER_ID}/items",
        headers=HEADERS,
        json=payload,
    )
    resp.raise_for_status()
    result = resp.json()
    created = result.get("successful", {}).get("0", {})
    return created.get("data", {}).get("key") or created.get("key", "")


def update_note(note_key: str, note_html: str) -> None:
    """Replace a note's content."""
    item = get_item(note_key)
    data = item["data"]
    if data.get("itemType") != "note":
        sys.exit(f"{note_key} is not a note item (type: {data.get('itemType')})")
    version = data["version"]
    patch_headers = {**HEADERS, "If-Unmodified-Since-Version": str(version)}
    resp = requests.patch(
        f"{API_BASE}/users/{USER_ID}/items/{note_key}",
        headers=patch_headers,
        json={"note": note_html, "version": version},
    )
    resp.raise_for_status()


def append_to_note(note_key: str, append_html: str) -> None:
    """Append content to an existing note."""
    item = get_item(note_key)
    data = item["data"]
    if data.get("itemType") != "note":
        sys.exit(f"{note_key} is not a note item")
    existing = data.get("note", "")
    new_html = existing + "\n" + append_html
    version = data["version"]
    patch_headers = {**HEADERS, "If-Unmodified-Since-Version": str(version)}
    resp = requests.patch(
        f"{API_BASE}/users/{USER_ID}/items/{note_key}",
        headers=patch_headers,
        json={"note": new_html, "version": version},
    )
    resp.raise_for_status()


def set_tags(item_key: str, tags_to_set: list[str], replace_progress: bool = False) -> None:
    """Set tags on an item. If replace_progress=True, removes existing PROGRESS_TAGS first."""
    item = get_item(item_key)
    data = item["data"]
    version = data["version"]
    current_tags = data.get("tags", [])

    if replace_progress:
        current_tags = [t for t in current_tags if t["tag"] not in PROGRESS_TAGS]

    existing_names = {t["tag"] for t in current_tags}
    for tag in tags_to_set:
        if tag not in existing_names:
            current_tags.append({"tag": tag})

    patch_headers = {**HEADERS, "If-Unmodified-Since-Version": str(version)}
    resp = requests.patch(
        f"{API_BASE}/users/{USER_ID}/items/{item_key}",
        headers=patch_headers,
        json={"tags": current_tags, "version": version},
    )
    resp.raise_for_status()



# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_list(item_key: str) -> None:
    children = get_children(item_key)
    notes = [c for c in children if c["data"].get("itemType") == "note"]
    if not notes:
        print(f"No notes on item {item_key}")
        return
    print(f"\n{len(notes)} note(s) on {item_key}:\n")
    for note in notes:
        data = note["data"]
        key = data["key"]
        raw = data.get("note", "")
        preview = html_to_text(raw)[:100].replace("\n", " ")
        tags = [t["tag"] for t in data.get("tags", [])]
        tag_str = f"  [{', '.join(tags)}]" if tags else ""
        date = data.get("dateModified", "")[:10]
        print(f"  [{key}] {preview}...{tag_str}  (modified: {date})")
    print()


def cmd_show(note_key: str) -> None:
    item = get_item(note_key)
    data = item["data"]
    if data.get("itemType") != "note":
        print(f"Warning: {note_key} has type '{data.get('itemType')}', not 'note'")
    raw = data.get("note", "")
    text = html_to_text(raw)
    tags = [t["tag"] for t in data.get("tags", [])]
    print(f"\n[{note_key}]  modified: {data.get('dateModified', '')[:10]}")
    if tags:
        print(f"Tags: {', '.join(tags)}")
    print(f"{'─' * 60}")
    print(text)
    print(f"{'─' * 60}\n")


def cmd_create(args: argparse.Namespace) -> None:
    # Get body text
    if args.template:
        body = TEMPLATES.get(args.template, "")
        if not body:
            sys.exit(f"Unknown template: {args.template}. Available: {', '.join(TEMPLATES)}")
        if args.body:
            body = args.body + "\n\n" + body
    elif args.file:
        body = Path(args.file).read_text(encoding="utf-8")
    elif args.body:
        body = args.body
    else:
        # Read from stdin
        print("Enter note body (Ctrl+D to finish):")
        try:
            body = sys.stdin.read()
        except KeyboardInterrupt:
            sys.exit("\nCancelled")

    note_html = build_note_html(args.title, body)
    tags = args.tag or []

    print(f"Creating note on item {args.item_key}...")
    key = create_note(args.item_key, note_html, tags=tags)
    if key:
        print(f"✓ Created note: {key}")
    else:
        print("Note created (key unavailable in response)")


def cmd_update(args: argparse.Namespace) -> None:
    if args.append:
        append_html = text_to_html(args.append)
        print(f"Appending to note {args.note_key}...")
        append_to_note(args.note_key, append_html)
        print("✓ Note updated")
    elif args.body:
        note_html = build_note_html(args.title, args.body)
        print(f"Updating note {args.note_key}...")
        update_note(args.note_key, note_html)
        print("✓ Note updated")
    elif args.file:
        body = Path(args.file).read_text(encoding="utf-8")
        note_html = build_note_html(args.title, body)
        print(f"Updating note {args.note_key} from file...")
        update_note(args.note_key, note_html)
        print("✓ Note updated")
    else:
        sys.exit("Provide --body, --append, or --file")


def cmd_status(item_key: str, status: str) -> None:
    if status not in PROGRESS_TAGS:
        sys.exit(f"Status must be one of: {', '.join(sorted(PROGRESS_TAGS))}")
    print(f"Setting status '{status}' on {item_key}...")
    set_tags(item_key, [status], replace_progress=True)
    print(f"✓ Tagged as '{status}'")


def cmd_delete(note_key: str) -> None:
    print(
        f"Deletion is disabled in this skill to prevent accidental data loss.\n"
        f"To delete note {note_key}, open the Zotero desktop app, right-click the item, and choose Delete."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Manage Zotero notes")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list
    p_list = subparsers.add_parser("list", help="List notes on an item")
    p_list.add_argument("item_key")

    # show
    p_show = subparsers.add_parser("show", help="Show note content")
    p_show.add_argument("note_key")

    # create
    p_create = subparsers.add_parser("create", help="Create a note on an item")
    p_create.add_argument("item_key")
    p_create.add_argument("--title", metavar="TITLE", help="Note title (H1 heading)")
    p_create.add_argument("--body", metavar="TEXT", help="Note body text (markdown)")
    p_create.add_argument("--file", metavar="FILE", help="Read body from a file")
    p_create.add_argument("--template", choices=list(TEMPLATES), metavar="TEMPLATE",
                          help=f"Use a template: {', '.join(TEMPLATES)}")
    p_create.add_argument("--tag", action="append", default=[], metavar="TAG")

    # update
    p_update = subparsers.add_parser("update", help="Update an existing note")
    p_update.add_argument("note_key")
    p_update.add_argument("--title", metavar="TITLE")
    p_update.add_argument("--body", metavar="TEXT", help="Replace note body")
    p_update.add_argument("--append", metavar="TEXT", help="Append to note")
    p_update.add_argument("--file", metavar="FILE", help="Replace body with file content")

    # status
    p_status = subparsers.add_parser("status", help="Set reading progress tag")
    p_status.add_argument("item_key")
    p_status.add_argument("status", choices=sorted(PROGRESS_TAGS))

    # delete
    p_delete = subparsers.add_parser("delete", help="Delete a note")
    p_delete.add_argument("note_key")

    args = parser.parse_args()

    api_key, user_id = get_env()
    global HEADERS, USER_ID
    HEADERS = build_headers(api_key)
    USER_ID = user_id

    if args.command == "list":
        cmd_list(args.item_key)
    elif args.command == "show":
        cmd_show(args.note_key)
    elif args.command == "create":
        cmd_create(args)
    elif args.command == "update":
        cmd_update(args)
    elif args.command == "status":
        cmd_status(args.item_key, args.status)
    elif args.command == "delete":
        cmd_delete(args.note_key)


if __name__ == "__main__":
    main()
