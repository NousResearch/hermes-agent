#!/usr/bin/env python3
"""Native Apple Notes helper for the ``apple-notes`` skill.

All operations are parameterized and noninteractive: every command takes
explicit arguments and never prompts the user. The AppleScript that drives
Notes.app is assembled by pure functions (``build_*``) and executed by
``run_applescript``, so the string-building logic is unit-testable without
macOS or a live Notes database.

Invoke through the Hermes ``terminal`` tool, for example::

    terminal: python3 skills/apple/apple-notes/scripts/apple_notes.py list-folders
"""

from __future__ import annotations

import argparse
import html
import subprocess
import sys
from typing import Optional

OSASCRIPT_TIMEOUT = 30


def escape_applescript_string(value: str) -> str:
    """Escape a string for an AppleScript double-quoted literal.

    AppleScript requires backslash and double-quote to be escaped with a
    leading backslash. Other characters pass through unchanged because body
    text is first converted by ``text_to_notes_html``.
    """
    return value.replace("\\", "\\\\").replace('"', '\\"')


def text_to_notes_html(text: str) -> str:
    """Convert plain text into Notes body HTML.

    HTML-special characters are escaped so user content cannot inject markup,
    then newlines become ``<br>`` line breaks.
    """
    escaped = html.escape(text, quote=False)
    return escaped.replace("\n", "<br>")


def _q(value: str) -> str:
    """Wrap an escaped value in AppleScript double quotes."""
    return '"' + escape_applescript_string(value) + '"'


def build_list_folders_script() -> str:
    return 'tell application "Notes" to get name of every folder'


def build_list_notes_script(folder: str) -> str:
    # Scope with a tell block: a one-liner like `every note of first folder
    # whose name is F` parses the `whose` against the note, not the folder,
    # and silently returns nothing.
    f = _q(folder)
    return (
        'tell application "Notes"\n'
        "    tell first folder whose name is " + f + "\n"
        "        get name of every note\n"
        "    end tell\n"
        "end tell"
    )


def build_search_script(query: str) -> str:
    # Title-only search: Notes `whose` on the rich-text body is unreliable.
    return (
        'tell application "Notes" to get name of every note '
        "whose name contains " + _q(query)
    )


def build_read_note_script(title: str, folder: Optional[str] = None) -> str:
    t = _q(title)
    if folder:
        f = _q(folder)
        return (
            'tell application "Notes"\n'
            "    tell first folder whose name is " + f + "\n"
            "        get body of first note whose name is " + t + "\n"
            "    end tell\n"
            "end tell"
        )
    return 'tell application "Notes" to get body of first note whose name is ' + t


def build_create_note_script(title: str, body_html: str, folder: Optional[str] = None) -> str:
    t = _q(title)
    b = _q(body_html)
    props = "{name:" + t + ", body:" + b + "}"
    if folder:
        f = _q(folder)
        return (
            'tell application "Notes"\n'
            "    tell first folder whose name is " + f + "\n"
            "        make new note with properties " + props + "\n"
            "    end tell\n"
            "end tell"
        )
    return (
        'tell application "Notes"\n'
        "    tell first folder\n"
        "        make new note with properties " + props + "\n"
        "    end tell\n"
        "end tell"
    )


def build_append_note_script(title: str, body_html: str, folder: Optional[str] = None) -> str:
    t = _q(title)
    b = _q(body_html)
    if folder:
        f = _q(folder)
        return (
            'tell application "Notes"\n'
            "    tell first folder whose name is " + f + "\n"
            "        set n to first note whose name is " + t + "\n"
            "        set body of n to (body of n) & " + b + "\n"
            "    end tell\n"
            "end tell"
        )
    return (
        'tell application "Notes"\n'
        "    set n to first note whose name is " + t + "\n"
        "    set body of n to (body of n) & " + b + "\n"
        "end tell"
    )


def build_create_folder_script(name: str) -> str:
    return 'tell application "Notes" to make new folder with properties {name:' + _q(name) + "}"


def build_move_note_script(title: str, dest: str, src: Optional[str] = None) -> str:
    t = _q(title)
    d = _q(dest)
    if src:
        s = _q(src)
        return (
            'tell application "Notes"\n'
            "    set destFolder to first folder whose name is " + d + "\n"
            "    tell first folder whose name is " + s + "\n"
            "        move first note whose name is " + t + " to destFolder\n"
            "    end tell\n"
            "end tell"
        )
    return (
        'tell application "Notes"\n'
        "    set destFolder to first folder whose name is " + d + "\n"
        "    move first note whose name is " + t + " to destFolder\n"
        "end tell"
    )


def run_applescript(script: str) -> str:
    """Run an AppleScript via ``osascript -`` (stdin) and return stdout."""
    result = subprocess.run(
        ["osascript", "-"],
        input=script,
        capture_output=True,
        text=True,
        timeout=OSASCRIPT_TIMEOUT,
    )
    if result.returncode != 0:
        raise RuntimeError("osascript failed: " + result.stderr.strip())
    return result.stdout


def _emit(text: str) -> int:
    if text:
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")
    return 0


def _body_arg(args: argparse.Namespace) -> str:
    if args.body_html is not None:
        return args.body_html
    return text_to_notes_html(args.body)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="apple_notes.py",
        description="Native, noninteractive Apple Notes helper.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-folders", help="List every Notes folder.")

    p = sub.add_parser("list-notes", help="List note titles in a folder.")
    p.add_argument("--folder", required=True)

    p = sub.add_parser("search", help="Search note titles by substring.")
    p.add_argument("--query", required=True)

    p = sub.add_parser("read", help="Read a note body.")
    p.add_argument("--title", required=True)
    p.add_argument("--folder")

    p = sub.add_parser("create", help="Create a note.")
    p.add_argument("--title", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--body", help="Plain-text body (converted to Notes HTML).")
    g.add_argument("--body-html", help="Raw HTML body (no conversion).")
    p.add_argument("--folder")

    p = sub.add_parser("append", help="Append HTML to an existing note body.")
    p.add_argument("--title", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--body", help="Plain-text body (converted to Notes HTML).")
    g.add_argument("--body-html", help="Raw HTML body (no conversion).")
    p.add_argument("--folder")

    p = sub.add_parser("create-folder", help="Create a new folder.")
    p.add_argument("--name", required=True)

    p = sub.add_parser("move", help="Move a note to another folder.")
    p.add_argument("--title", required=True)
    p.add_argument("--dest", required=True)
    p.add_argument("--src", help="Source folder (defaults to searching all folders).")
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    cmd = args.command

    if cmd == "list-folders":
        return _emit(run_applescript(build_list_folders_script()))
    if cmd == "list-notes":
        return _emit(run_applescript(build_list_notes_script(args.folder)))
    if cmd == "search":
        return _emit(run_applescript(build_search_script(args.query)))
    if cmd == "read":
        return _emit(run_applescript(build_read_note_script(args.title, args.folder)))
    if cmd == "create":
        return _emit(run_applescript(build_create_note_script(args.title, _body_arg(args), args.folder)))
    if cmd == "append":
        return _emit(run_applescript(build_append_note_script(args.title, _body_arg(args), args.folder)))
    if cmd == "create-folder":
        return _emit(run_applescript(build_create_folder_script(args.name)))
    if cmd == "move":
        return _emit(run_applescript(build_move_note_script(args.title, args.dest, args.src)))
    return 2


if __name__ == "__main__":
    sys.exit(main())
