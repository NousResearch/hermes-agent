"""``hermes clipboard image``: the clipboard image extractor as a device-local command.

Runs ``hermes_cli.clipboard.save_clipboard_image`` (the extractor the CLI and Ink use) and reports
the outcome. It reads this machine's clipboard only: no gateway, no session store, no profile
files, so a client whose gateway is remote still reads the clipboard where it runs.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

EXIT_OK = 0
EXIT_NO_IMAGE = 1
EXIT_NO_TOOL = 3
EXIT_WRITE_FAILED = 4

_SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", "image/png"), (b"\xff\xd8\xff", "image/jpeg"), (b"GIF8", "image/gif"),
    (b"BM", "image/bmp"), (b"II*\x00", "image/tiff"), (b"MM\x00*", "image/tiff"))


def _mime(head: bytes) -> str:
    """The image type by magic bytes: the extractor writes PNG, but a copied image *file*
    (Windows/WSL file drop) is copied byte for byte."""
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    return next((mime for sig, mime in _SIGNATURES if head.startswith(sig)), "application/octet-stream")


def extract_clipboard_image(dest: Path) -> tuple[int, dict]:
    """Write the clipboard image to *dest*; ``(exit code, JSON reply)``."""
    from hermes_cli import clipboard
    from hermes_platform.resolver import locate_command

    dest = dest.expanduser().absolute()
    written = None
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not os.access(dest.parent, os.W_OK):
            raise PermissionError("directory is not writable")
        if os.path.lexists(dest) and (dest.is_dir() or not os.access(dest, os.W_OK)):
            raise PermissionError("file is not writable")
        # Extract into a private directory beside dest, then rename over it: the extractor
        # deletes its target on failure, so dest itself is only ever replaced by a full image.
        with tempfile.TemporaryDirectory(prefix=".hermes-clipboard-", dir=dest.parent) as tmp:
            staged = Path(tmp) / dest.name
            if clipboard.save_clipboard_image(staged):
                with staged.open("rb") as f:
                    written = {"ok": True, "path": str(dest), "mime": _mime(f.read(16)),
                               "bytes": staged.stat().st_size}
                if os.path.exists(dest):  # keep an overwritten file's mode: a 0600 temp file stays private
                    os.chmod(staged, os.stat(dest).st_mode & 0o7777)
                os.replace(staged, dest)
    except OSError as exc:
        return EXIT_WRITE_FAILED, {
            "ok": False, "reason": "write_failed", "message": f"cannot write {dest}: {exc}"}
    if written:
        return EXIT_OK, written
    tools = clipboard.clipboard_image_tools()
    if not any(locate_command(tool).command for tool in tools):
        return EXIT_NO_TOOL, {
            "ok": False, "reason": "no_tool", "tools": list(tools),
            "message": "no clipboard tool found; install one of: " + ", ".join(tools)}
    return EXIT_NO_IMAGE, {"ok": False, "reason": "no_image", "message": "the clipboard holds no image"}


def cmd_clipboard(args) -> int:
    """``hermes clipboard image --out PATH [--json]``."""
    code, reply = extract_clipboard_image(Path(args.out))
    if args.json:
        print(json.dumps(reply))
    elif reply["ok"]:
        print(reply["path"])
    else:
        print(reply["message"], file=sys.stderr)
    return code
