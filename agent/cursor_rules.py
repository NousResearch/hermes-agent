"""Cursor project rules (``.cursor/rules/*.mdc``) loaded by their own rule types.

Each ``.mdc`` file opens with a small front matter block that says when the rule applies
(https://cursor.com/docs/rules): ``alwaysApply: true`` rules are always in context, ``globs`` rules attach
when the agent works on a matching file, ``description`` rules are fetched by the agent when the
description fits the task, and the rest are manual. The front matter is Cursor's own dialect, not YAML
(``globs: *.tsx`` is a YAML alias error), so it is read line by line here; a trailing ``# comment`` is
dropped from a value, as Cursor itself does.

Startup (``prompt_builder._load_cursorrules``) puts only always-apply bodies in the system prompt plus a
one-line index of the others, so every rule stays reachable through ``read_file``. Glob rules are
attached to the tool result of the first tool call that touches a matching file
(``subdirectory_hints.SubdirectoryHintTracker``), which keeps the system prompt, and its cache, unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Optional

_TRUE = frozenset({"true", "yes", "on", "1"})


@dataclass(frozen=True)
class CursorRule:
    label: str  # ".cursor/rules/react.mdc"
    body: str  # markdown after the front matter
    description: str = ""
    globs: tuple[str, ...] = ()
    always: bool = True

    def summary(self) -> str:
        """One index line: label, description, and the file patterns that attach it."""
        text = f"- {self.label}"
        if self.description:
            text += f" — {self.description}"
        if self.globs:
            text += f" (attached for files: {', '.join(self.globs)})"
        elif not self.description:
            text += " (manual rule)"
        return text


def _strip_comment(value: str) -> str:
    """Drop a ``# comment`` that starts after whitespace outside quotes (``globs: *.ts  # web``)."""
    quote = ""
    for i, ch in enumerate(value):
        if quote:
            quote = "" if ch == quote else quote
        elif ch in "\"'":
            quote = ch
        elif ch == "#" and (i == 0 or value[i - 1].isspace()):
            return value[:i].rstrip()
    return value.strip()


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _parse_front_matter(block: str) -> dict[str, list[str]]:
    """``key: value`` lines plus indented ``- item`` continuation lines, every value comment-stripped."""
    meta: dict[str, list[str]] = {}
    key = None
    for line in block.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line[0].isspace() and key and line.strip().startswith("-"):
            meta[key].append(_strip_comment(line.strip()[1:]))
            continue
        name, sep, value = line.partition(":")
        if not sep or not name.strip():
            continue
        key = name.strip()
        value = _strip_comment(value)
        meta[key] = [value] if value else []
    return meta


def _split_globs(values: Iterable[str]) -> tuple[str, ...]:
    out: list[str] = []
    for value in values:
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            value = value[1:-1]
        out += [g for g in (_unquote(part) for part in value.split(",")) if g]
    return tuple(out)


def parse_rule(label: str, content: str) -> CursorRule:
    """A rule from one ``.mdc`` file. No (or unterminated) front matter means always-apply, as before."""
    text = content.lstrip("\ufeff")
    end = text.find("\n---", 3) if text.startswith("---") else -1
    if end == -1:
        return CursorRule(label, text.strip())
    meta = _parse_front_matter(text[3:end])
    body = text[end + 4:].lstrip("\n").strip()
    description = _unquote(" ".join(meta.get("description", [])))
    globs = _split_globs(meta.get("globs", []))
    always_values = meta.get("alwaysApply")
    if always_values is None:
        always = not (description or globs)  # empty front matter: nothing says it is scoped
    else:
        always = _unquote(" ".join(always_values)).lower() in _TRUE
    return CursorRule(label, body, description, globs, always)


@lru_cache(maxsize=256)
def _glob_regex(pattern: str) -> re.Pattern:
    """Cursor/gitignore-style glob: ``**`` spans directories, ``*`` and ``?`` stay inside one segment."""
    out, i = [], 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out) + r"\Z")


def glob_matches(pattern: str, rel_path: str) -> bool:
    """*rel_path* is POSIX and relative to the working directory; a slash-free pattern matches any basename."""
    pattern = pattern.strip().lstrip("/")
    pattern = pattern.removeprefix("./")
    if not pattern:
        return False
    target = rel_path if "/" in pattern else rel_path.rsplit("/", 1)[-1]
    return bool(_glob_regex(pattern).match(target))


def matching_rules(rules: Iterable[CursorRule], rel_path: str) -> list[CursorRule]:
    """Glob-scoped (not always-apply) rules whose patterns match *rel_path*."""
    return [r for r in rules if not r.always and r.globs and any(glob_matches(g, rel_path) for g in r.globs)]


def render_startup(rules: list[CursorRule]) -> tuple[list[CursorRule], Optional[str]]:
    """Always-apply rules for the system prompt, plus the index text for every other rule (or None)."""
    always = [r for r in rules if r.always and r.body]
    scoped = [r for r in rules if not r.always and r.body]
    if not scoped:
        return always, None
    index = ("These project rules are not loaded. Read one with read_file when its description fits the task; "
             "a rule with file patterns is also attached to the tool result when you touch a matching file.\n"
             + "\n".join(r.summary() for r in scoped))
    return always, index
