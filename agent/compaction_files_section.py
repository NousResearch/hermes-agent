"""File access metadata for lean compaction handoffs, without file I/O."""

import json
import re
from typing import Any


FILES_HEADING = "## Files Read or Edited (newest first; contents not in context)"
_SECTION_RE = re.compile(r"(?ms)^" + re.escape(FILES_HEADING) + r"$.*?(?=^## |\Z)")
_PATCH_HEADER_RE = re.compile(r"^\*\*\*[ \t]*(Update|Add|Delete|Move)[ \t]+File:[ \t]*(.+?)[ \t\r]*$", re.MULTILINE)
_REOPEN = "\nRe-open with read_file before relying on file contents."


def without_files_section(summary: str) -> str:
    """Replace model-written sections only when their heading occupies a whole line."""
    return _SECTION_RE.sub("", summary)


def build_files_section(turns: list[dict[str, Any]]) -> str:
    """Newest distinct paths from file-tool arguments; edits win over reads."""
    from agent.context_compressor import _redact_compaction_text, _tc_get

    paths: dict[str, str] = {}
    for msg in turns:
        if msg.get("role") != "assistant":
            continue
        for call in msg.get("tool_calls") or []:
            fn = _tc_get(call, "function", {})
            name = _tc_get(fn, "name")
            if name not in {"read_file", "write_file", "patch"}:
                continue
            args = _tc_get(fn, "arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    continue
            if not isinstance(args, dict):
                continue
            accessed = [args.get("path")]
            if name == "patch" and args.get("mode") == "patch" and isinstance(args.get("patch"), str):
                for match in _PATCH_HEADER_RE.finditer(args["patch"]):
                    accessed.extend([p.strip() for p in match.group(2).split("->", 1)]
                                    if match.group(1) == "Move" else [match.group(2)])
            for path in accessed:
                if not isinstance(path, str) or not path.strip():
                    continue
                previous = paths.pop(path, "read")
                paths[path] = "edited" if name != "read_file" else previous
    if not paths:
        return ""
    section = "\n\n" + FILES_HEADING + "\n"
    for path, tag in list(paths.items())[::-1][:12]:
        path = _redact_compaction_text(path).replace("\r", "\\r").replace("\n", "\\n")
        remaining = 1_200 - len(section) - len(_REOPEN) - len(f"- {tag}: \n")
        if remaining < 40:
            break
        limit = min(200, remaining)  # a long path keeps its tail: the file name is what read_file needs
        section += f"- {tag}: {path if len(path) <= limit else '…' + path[-(limit - 1):]}\n"
    return section.rstrip("\n") + _REOPEN
