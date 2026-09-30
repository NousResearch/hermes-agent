"""Import sessions from foreign coding agents (Claude Code, Codex CLI, Cursor). Foreign files are only ever read;
imported history must satisfy the provider role-alternation invariant (see ``_merge_turns``)."""

from __future__ import annotations

import contextlib
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from stat import S_ISREG
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote, urlparse

from hermes_state_ids import new_session_id

# User-message texts that are really injected context wrappers, not typed input.
_WRAPPER_TAG_RE = re.compile(
    r"^<(?:user_instructions|environment_context|recommended_plugins|"
    r"skills_instructions|permissions[_-]instructions|turn_context|"
    r"command-name|command-message|local-command-stdout|system-reminder)\b", re.IGNORECASE)

_TITLE_MAX = 60
_SOURCE_LABELS = {"claude": "Claude Code", "codex": "Codex CLI", "cursor": "Cursor"}
_SOURCE_DB_NAMES = {"claude": "claude-code", "codex": "codex-cli", "cursor": "cursor"}
# Cursor agent turns that are injected by the IDE, not typed by the user.
_CURSOR_NOISE_PREFIXES = (
    "Briefly inform the user about the task result",
    "Available subagent_types",
    "You have access to tools through dynamic namespaces",
    "Side chat boundary",
)
_CURSOR_DATE_ONLY_RE = re.compile(
    r"^(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday), \w+ \d{1,2}, \d{4}, .+$")
_CURSOR_QUERY_RE = re.compile(r"<user_query>\s*(.*?)\s*</user_query>", re.S)


@dataclass
class ForeignSession:
    """A discoverable session in another tool's on-disk store."""

    source: str  # "claude" | "codex" | "cursor"
    path: Path
    mtime: float
    cwd: Optional[str] = None
    title_guess: Optional[str] = None
    turn_count: int = 0
    session_id: Optional[str] = None  # the foreign tool's own id

    @property
    def label(self) -> str:
        title = (self.title_guess or "").strip() or self.path.stem
        return f"[{_SOURCE_LABELS.get(self.source, self.source)}] {title[:_TITLE_MAX]}"


def _read_json_lines(path: Path):
    """Yield parsed JSON objects, silently skipping unparseable lines."""
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if isinstance(obj, dict):
                    yield obj
    except OSError:
        return


def _block_text(block: Any) -> str:
    """Plain text of one content block; tool_result, thinking/reasoning and unknown types yield ''."""
    if isinstance(block, str):
        return block
    if not isinstance(block, dict):
        return ""
    btype = block.get("type")
    if btype in ("text", "input_text", "output_text"):
        return text if isinstance(text := block.get("text"), str) else ""
    if btype == "tool_use":  # Claude Code assistant block
        return f"[ran tool: {block.get('name') or 'tool'}]"
    return "[image]" if btype == "image" else ""


def _flatten_blocks(content: Any) -> str:
    """Flatten a message ``content`` (string or block list) to plain text."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n\n".join(p for p in (_block_text(b).strip() for b in content) if p)


def _merge_turns(raw_turns: List[Tuple[str, str]]) -> List[Dict[str, str]]:
    """Merge consecutive same-role turns; guarantee strict alternation.

    A leading assistant turn (session began before the log window) gets a minimal user stub so the
    first message is always ``user``; this is the only place a stub is ever inserted.
    """
    merged: List[Dict[str, str]] = []
    for role, text in raw_turns:
        if not (text := text.strip()):
            continue
        if merged and merged[-1]["role"] == role:
            merged[-1]["content"] += "\n\n" + text
        else:
            merged.append({"role": role, "content": text})
    if merged and merged[0]["role"] == "assistant":
        merged.insert(0, {"role": "user", "content": "(imported conversation begins with an assistant reply)"})
    return merged


def _message_turn(message: Any) -> Optional[Tuple[str, str]]:
    """Normalize one message dict into a ``(role, text)`` turn, or None when it is not importable."""
    role = message.get("role") if isinstance(message, dict) else None
    if role not in ("user", "assistant"):
        return None
    text = _flatten_blocks(message.get("content"))
    return None if not text or (role == "user" and _WRAPPER_TAG_RE.match(text.lstrip())) else (role, text)


def _first_user_line(turns: List[Tuple[str, str]]) -> Optional[str]:
    for role, text in turns:
        if role == "user" and (line := text.strip().partition("\n")[0].strip()):
            return line[:_TITLE_MAX * 2]
    return None


def _parsed(turns: List[Tuple[str, str]], cwd: Optional[str], session_id: Optional[str],
            title: Optional[str] = None) -> Dict[str, Any]:
    return {"turns": _merge_turns(turns), "cwd": cwd, "title_guess": title or _first_user_line(turns),
            "session_id": session_id}


def parse_claude_session(path: Path) -> Dict[str, Any]:
    """Parse one Claude Code session JSONL into normalized turns + meta."""
    turns: List[Tuple[str, str]] = []
    cwd = summary = session_id = None
    for obj in _read_json_lines(path):
        otype = obj.get("type")
        if otype == "summary":
            if isinstance(s := obj.get("summary"), str) and s.strip():
                summary = s.strip()
        elif otype in ("user", "assistant") and not (obj.get("isSidechain") or obj.get("isMeta")):
            if cwd is None and isinstance(obj.get("cwd"), str):
                cwd = obj["cwd"]
            if session_id is None and isinstance(obj.get("sessionId"), str):
                session_id = obj["sessionId"]
            if turn := _message_turn(obj.get("message")):
                turns.append(turn)
    return _parsed(turns, cwd, session_id, summary)


def parse_codex_session(path: Path) -> Dict[str, Any]:
    """Parse one Codex CLI rollout JSONL into normalized turns + meta."""
    turns: List[Tuple[str, str]] = []
    cwd = session_id = None
    for obj in _read_json_lines(path):
        otype, payload = obj.get("type"), obj.get("payload")
        if not isinstance(payload, dict):
            continue
        if otype == "session_meta":
            if isinstance(payload.get("cwd"), str):
                cwd = payload["cwd"]
            if isinstance(sid := payload.get("session_id") or payload.get("id"), str):
                session_id = sid
        elif otype == "response_item":
            ptype = payload.get("type")
            if ptype == "message" and (turn := _message_turn(payload)):  # developer/system payloads skipped
                turns.append(turn)
            elif ptype in ("custom_tool_call", "function_call", "local_shell_call"):
                # Assistant activity; merged into neighbours later. Tool outputs / reasoning skipped.
                name = payload.get("name") or payload.get("tool") or "tool"
                turns.append(("assistant", f"[ran tool: {name}]"))
    return _parsed(turns, cwd, session_id)


def _cursor_user_text(content: Any) -> str:
    """The text the user typed, minus Cursor's timestamp wrapper and IDE injections."""
    blob = _flatten_blocks(content)
    queries = []
    for raw in _CURSOR_QUERY_RE.findall(blob):
        text = raw.strip()
        if not text or text.startswith(_CURSOR_NOISE_PREFIXES) or _CURSOR_DATE_ONLY_RE.match(text):
            continue
        queries.append(text)
    if not queries:
        # An attachment with no typed text is still a user turn, same as Claude/Codex.
        return "[image attached]" if _cursor_has_image(blob) else ""
    text = "\n\n".join(queries)
    if _cursor_has_image(blob):
        text += "\n\n[image attached]"
    return text


def _cursor_has_image(blob: str) -> bool:
    return "<image_files>" in blob or "[Image]" in blob or "[image]" in blob


def _cursor_project_slug(path: Path) -> Optional[str]:
    """``<projects>/<slug>/agent-transcripts/<id>/<id>.jsonl`` → slug. Anything else is unset."""
    try:
        if path.parent.parent.name == "agent-transcripts" and path.parent.name == path.stem:
            return path.parent.parent.parent.name
    except (IndexError, OSError):
        return None
    return None


def _fs_casefold() -> bool:
    """macOS and Windows fold directory case. Linux does not."""
    return os.name == "nt" or sys.platform == "darwin"


def _cursor_workspace_storage(platform: str, *, environ: Optional[dict] = None, home: Optional[Path] = None) -> Path:
    """Platform path of Cursor's ``workspaceStorage``. ``platform`` is ``nt``, ``darwin``, or anything else (Linux)."""
    env = os.environ if environ is None else environ
    home = Path.home() if home is None else home
    if platform == "nt":
        base = env.get("APPDATA") or str(home / "AppData" / "Roaming")
        return Path(base) / "Cursor" / "User" / "workspaceStorage"
    if platform == "darwin":
        return home / "Library" / "Application Support" / "Cursor" / "User" / "workspaceStorage"
    base = env.get("XDG_CONFIG_HOME") or str(home / ".config")
    return Path(base) / "Cursor" / "User" / "workspaceStorage"


def cursor_workspace_storage() -> Path:
    """Cursor's recorded workspace list. This is the cwd source, not the transcript.

    Same role as the ``cwd`` field Claude Code and Codex write into the log.
    """
    platform = "nt" if os.name == "nt" else sys.platform
    return _cursor_workspace_storage(platform)


def cursor_slug_for_path(path: str) -> str:
    """Encode a filesystem path the way Cursor names ``~/.cursor/projects/<slug>``.

    Separators and other ASCII punctuation become ``-``. Non-ASCII is dropped
    (``Università`` → ``Universit``), which is why the slug alone cannot rebuild
    that path. Existing hyphens are kept, so ``fisso-llm`` stays one component.
    """
    text = path.replace("\\", "/")
    if len(text) >= 2 and text[1] == ":":
        text = text[0] + text[2:]
    out = []
    for ch in text.lstrip("/"):
        if ch.isascii() and (ch.isalnum() or ch == "-"):
            out.append(ch)
        elif ch.isascii():
            out.append("-")
    slug = "".join(out)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")


def _cursor_uri_path(folder: str) -> Optional[str]:
    """``workspace.json`` ``folder`` → filesystem path. Remote URIs keep the remote path."""
    if folder.startswith("file://"):
        raw = unquote(urlparse(folder).path)
        if re.match(r"^/[A-Za-z]:/", raw):
            raw = raw[1:]
    elif folder.startswith("vscode-remote://"):
        rest = folder.split("vscode-remote://", 1)[1]
        idx = rest.find("/")
        if idx < 0:
            return None
        raw = unquote(rest[idx:])
    else:
        return None
    return raw if raw and raw != "/" else None


def _slugs_equal(left: str, right: str) -> bool:
    if _fs_casefold():
        return left.casefold() == right.casefold()
    return left == right


def _recorded_cursor_cwd(slug: str, storage: Optional[Path] = None) -> Optional[str]:
    """Path Cursor itself recorded for *slug*, or None when the record is missing or ambiguous."""
    root = Path(storage) if storage is not None else cursor_workspace_storage()
    if not root.is_dir():
        return None
    hits: List[str] = []
    try:
        entries = list(root.iterdir())
    except OSError:
        return None
    for entry in entries:
        ws = entry / "workspace.json"
        if not ws.is_file():
            continue
        try:
            data = json.loads(ws.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError, UnicodeError):
            continue
        folder = data.get("folder") if isinstance(data, dict) else None
        if not isinstance(folder, str):
            continue
        path = _cursor_uri_path(folder)
        if path and _slugs_equal(cursor_slug_for_path(path), slug):
            hits.append(path)
    if not hits:
        return None
    existing = [path for path in hits if Path(path).is_dir()]
    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:
        return None
    return hits[0] if len(set(hits)) == 1 else None


def _use_recorded_cwd(storage: Optional[Path], transcript: Optional[Path]) -> bool:
    """Recorded lookup is for a real Cursor install, or a test that pins ``storage``.

    A fixture transcript outside the projects dir must not pick up the developer's
    own ``workspaceStorage``.
    """
    if storage is not None:
        return True
    if transcript is None:
        return False
    try:
        projects = _default_root("cursor").resolve()
        return Path(transcript).resolve().is_relative_to(projects)
    except (OSError, ValueError):
        return False


def _cursor_walk_starts(
    slug: str,
    root: Optional[Path],
    *,
    platform: Optional[str] = None,
    mounts: Optional[Path] = None,
) -> List[Tuple[Path, str]]:
    """Where a slug walk begins. ``root`` pins one start (tests).

    A Windows slug ``C-Users-…`` starts at ``C:/``. On Linux/WSL the same slug is
    also tried under ``/mnt/<drive>`` when that mount exists. ``mounts`` overrides
    that directory so the WSL branch can be tested without a real ``/mnt``.
    """
    platform = os.name if platform is None else platform
    if root is not None:
        return [(root, slug)]
    if platform == "nt" and len(slug) >= 3 and slug[0].isalpha() and slug[1] == "-":
        return [(Path(f"{slug[0].upper()}:/"), slug[2:])]
    starts = [(Path("/"), slug)]
    if platform != "nt" and len(slug) >= 3 and slug[0].isalpha() and slug[1] == "-":
        mount = (Path(mounts) if mounts is not None else Path("/mnt")) / slug[0].lower()
        if mount.is_dir():
            starts.append((mount, slug[2:]))
    return starts


def _slug_remainder(name: str, rest: str, *, casefold: bool) -> Optional[str]:
    """Remainder of *rest* after consuming *name* as one path component, or None."""
    if not name or len(name) > len(rest):
        return None
    head, tail = rest[:len(name)], rest[len(name):]
    if casefold:
        if head.casefold() != name.casefold():
            return None
    elif head != name:
        return None
    if tail.startswith("-"):
        return tail[1:]
    return tail if not tail else None


def _walk_cursor_slug(
    slug: str,
    *,
    root: Optional[Path] = None,
    casefold: Optional[bool] = None,
    mounts: Optional[Path] = None,
) -> Optional[str]:
    """Resolve *slug* by walking directories. None unless exactly one full path matches.

    Longest-name-first is not enough: ``foo`` + ``bar-baz`` and ``foo-bar`` + ``baz``
    can both be prefixes. Every branch is tried. Two complete paths means the slug
    is ambiguous and the cwd is left unset, same as a missing Claude/Codex field.
    """
    if casefold is None:
        casefold = _fs_casefold()
    found: List[str] = []

    def rec(node: Path, rest: str) -> None:
        if len(found) > 1:
            return
        if not rest:
            found.append(str(node))
            return
        if not node.is_dir():
            return
        try:
            children = [child for child in node.iterdir() if child.is_dir()]
        except OSError:
            return
        children.sort(key=lambda child: len(child.name), reverse=True)
        for child in children:
            nxt = _slug_remainder(child.name, rest, casefold=casefold)
            if nxt is None:
                continue
            rec(child, nxt)

    for start, rest in _cursor_walk_starts(slug, root, mounts=mounts):
        if rest:
            rec(start, rest)
        if len(found) > 1:
            return None
    return found[0] if len(found) == 1 else None


def recover_cursor_cwd(
    slug: str,
    *,
    root: Optional[Path] = None,
    storage: Optional[Path] = None,
    transcript: Optional[Path] = None,
) -> Optional[str]:
    """Turn a Cursor project-dir slug back into a directory, or None.

    Prefer Cursor's own workspace record (the analogue of Claude/Codex ``cwd``).
    That record still has spaces, dots, and non-ASCII that the slug deleted.
    Fall back to a filesystem walk only when the record is missing, and only
    when exactly one directory consumes the slug. ``root`` / ``storage`` are for tests.
    """
    if not slug or "/" in slug or "\\" in slug or ".." in slug.split("-"):
        return None
    if root is None and _use_recorded_cwd(storage, transcript):
        recorded = _recorded_cursor_cwd(slug, storage)
        if recorded:
            return recorded
    return _walk_cursor_slug(slug, root=root)


def parse_cursor_session(path: Path) -> Dict[str, Any]:
    """Parse one Cursor agent transcript JSONL into normalized turns + meta.

    Discovery lists only ``<id>/<id>.jsonl``. A subagent file passed explicitly
    still parses; it is just not offered in the picker.
    """
    turns: List[Tuple[str, str]] = []
    for obj in _read_json_lines(path):
        role = obj.get("role")
        message = obj.get("message")
        # Cursor stores the role on the line, not inside ``message``. ``_message_turn``
        # reads the inner role, so stamp it without mutating the parsed object.
        if role == "assistant" and isinstance(message, dict):
            if message.get("role") != "assistant":
                message = {**message, "role": "assistant"}
            if turn := _message_turn(message):
                turns.append(turn)
        elif role == "user" and isinstance(message, dict) and (text := _cursor_user_text(message.get("content"))):
            turns.append(("user", text))
    slug = _cursor_project_slug(path)
    cwd = recover_cursor_cwd(slug, transcript=path) if slug else None
    return _parsed(turns, cwd, path.stem)


def foreign_source_names() -> Tuple[str, ...]:
    """Source tokens accepted by ``--from`` and ``--resume @<token>``."""
    return tuple(_SOURCE_LABELS)


def foreign_resume_source(token: Any) -> Optional[str]:
    """``@claude`` / ``@codex`` / ``@cursor`` → source name. Anything else is None."""
    if not isinstance(token, str):
        return None
    name = token.strip().lower()
    if not name.startswith("@"):
        return None
    name = name[1:]
    return name if name in _SOURCE_LABELS else None


def infer_foreign_source(path: str) -> Optional[str]:
    """Guess the tool from a session-file path. Cursor wins over the loose ``.jsonl`` rule."""
    p = str(path).replace("\\", "/")
    name = Path(path).name
    if "/.cursor/" in p or "/agent-transcripts/" in p:
        return "cursor"
    if "/.codex/" in p or f"{os.sep}.codex{os.sep}" in p or name.startswith("rollout-"):
        return "codex"
    if "/.claude/" in p or f"{os.sep}.claude{os.sep}" in p or (p.endswith(".jsonl") and "claude" in p):
        return "claude"
    return None


# source -> (default root under ~, env override var, subdir under the env root, glob pattern,
#            recursive, parser)
_SOURCES = {
    "claude": ((".claude", "projects"), "CLAUDE_CONFIG_DIR", "projects", "*/*.jsonl", False,
               parse_claude_session),
    "codex": ((".codex", "sessions"), "CODEX_HOME", "sessions", "rollout-*.jsonl", True,
              parse_codex_session),
    # Parent chats only: ``<slug>/agent-transcripts/<id>/<id>.jsonl``. Subagents sit one
    # level deeper and are not listed. Cursor has no official relocation var; CURSOR_CONFIG_DIR
    # is an optional override of the config dir (the parent of ``projects``), blank = ~/.cursor.
    "cursor": ((".cursor", "projects"), "CURSOR_CONFIG_DIR", "projects",
               "*/agent-transcripts/*/*.jsonl", False, parse_cursor_session),
}


def _parser(source: str):
    return _SOURCES[source][5]


def _default_root(source: str) -> Path:
    """Default session store for *source*, honoring the tool's own relocation env var.

    Claude Code moves its whole config dir with ``CLAUDE_CONFIG_DIR``; Codex CLI with
    ``CODEX_HOME``. A blank/whitespace value is treated as unset (an empty override must not
    resolve to a relative ``"projects"`` under the CWD). Ported from cline/cline#13827."""
    default_parts, env_var, env_subdir, *_ = _SOURCES[source]
    override = os.environ.get(env_var, "").strip()
    if override:
        return Path(override).expanduser() / env_subdir
    return Path.home().joinpath(*default_parts)


def _walk(source: str, root: Optional[Path] = None) -> List[Tuple[Path, os.stat_result]]:
    """Regular log files of *source* under *root* (default: the tool's env-aware store, see
    ``_default_root``) as ``(path, stat)``, newest first. Symlinks escaping the root and
    unreadable/rotated entries are skipped, so one bad file never hides the rest. Shared by the
    CLI picker and the desktop browser."""
    pattern, recursive = _SOURCES[source][3], _SOURCES[source][4]
    root = (Path(root) if root else _default_root(source)).resolve()
    found: List[Tuple[Path, os.stat_result]] = []
    for path in (root.rglob(pattern) if recursive else root.glob(pattern)) if root.is_dir() else ():
        try:
            resolved = path.resolve()
            st = resolved.stat()
        except OSError:
            continue
        if resolved.is_relative_to(root) and S_ISREG(st.st_mode):
            found.append((resolved, st))
    found.sort(key=lambda item: item[1].st_mtime, reverse=True)
    return found


def _list_sessions(source: str, root: Optional[Path]) -> List[ForeignSession]:
    parse = _parser(source)
    results: List[ForeignSession] = []
    for path, st in _walk(source, root):
        parsed = parse(path)
        if parsed["turns"]:
            results.append(ForeignSession(source, path, st.st_mtime, parsed["cwd"], parsed["title_guess"],
                                          len(parsed["turns"]), parsed["session_id"]))
    return results


def import_foreign_session(source: str, path, db=None) -> str:
    """Import one foreign session into the Hermes SessionDB; returns the Hermes session id.

    A second import of the same foreign id reopens the existing copy. Raises
    ``ValueError`` on unknown source or a session with no usable conversation turns.
    """
    source = (source or "").strip().lower().lstrip("@")
    if source not in _SOURCE_LABELS:
        raise ValueError(f"Unknown foreign session source: {source!r}")
    path = Path(path).expanduser()
    if not path.is_file():
        raise ValueError(f"Session file not found: {path}")
    parsed = _parser(source)(path)
    turns = parsed["turns"]
    if not turns:
        raise ValueError(f"No user/assistant conversation turns found in {path}")
    first_user = _first_user_line([(t["role"], t["content"]) for t in turns]) or path.stem
    if len(first_user) > _TITLE_MAX:
        first_user = first_user[: _TITLE_MAX - 1] + "…"
    tool = _SOURCE_DB_NAMES[source]
    owns_db = db is None
    if owns_db:
        from hermes_state_registry import acquire
        db = acquire()  # the CLI resume that follows acquires this same handle
    try:
        origin_payload = {"tool": tool, "path": str(path), "foreign_session_id": parsed.get("session_id")}
        if existing := db.find_foreign_import(origin_payload):
            return existing
        session_id = new_session_id()
        db.create_session(session_id, source=tool, cwd=parsed.get("cwd"),
                          origin_json=json.dumps({"imported_from": origin_payload}))
        for turn in turns:
            db.append_message(session_id, turn["role"], turn["content"])
        with contextlib.suppress(Exception):  # title is cosmetic; the import itself succeeded
            db.set_session_title(session_id, f"Imported from {_SOURCE_LABELS[source]}: {first_user}")
        return session_id
    finally:
        if owns_db:
            with contextlib.suppress(Exception):
                db.close()


def gather_foreign_sessions(source: Optional[str] = None, *, claude_root: Optional[Path] = None,
                            codex_root: Optional[Path] = None, cursor_root: Optional[Path] = None,
                            limit: int = 25) -> List[ForeignSession]:
    """List foreign sessions across sources, newest first."""
    roots = {"claude": claude_root, "codex": codex_root, "cursor": cursor_root}
    sessions = [s for name in _SOURCES if source in (None, name) for s in _list_sessions(name, roots.get(name))]
    sessions.sort(key=lambda s: s.mtime, reverse=True)
    return sessions[:limit] if limit else sessions


def pick_foreign_session(source: Optional[str] = None, *, limit: int = 25) -> Optional[ForeignSession]:
    """Interactive numbered picker. Returns None when nothing was chosen."""
    sessions = gather_foreign_sessions(source, limit=limit)
    if not sessions:
        where = _SOURCE_LABELS.get(source or "", "Claude Code, Codex CLI, or Cursor")
        print(f"No {where} sessions found on this machine.")
        return None
    print("Foreign sessions (newest first):")
    for i, s in enumerate(sessions, 1):
        ws = f"  ({os.path.basename(s.cwd.rstrip('/')) or s.cwd})" if s.cwd else ""
        print(f"  {i:>2}. {datetime.fromtimestamp(s.mtime):%Y-%m-%d %H:%M}  {s.label}{ws}  [{s.turn_count} turns]")
    if not sys.stdin.isatty():
        names = "|".join(foreign_source_names())
        print("Non-interactive terminal — pass the file path directly:\n"
              f"  hermes sessions import --from {names} <path>")
        return None
    try:
        raw = input(f"Import which session? [1-{len(sessions)}, empty to cancel] ").strip()
        idx = int(raw) if raw else None
    except (EOFError, KeyboardInterrupt):
        return None
    except ValueError:
        print(f"Not a number: {raw}")
        return None
    if idx is not None and 1 <= idx <= len(sessions):
        return sessions[idx - 1]
    if idx is not None:
        print(f"Out of range: {idx}")
    return None


def run_sessions_import(args, db=None) -> Optional[str]:
    """`hermes sessions import` entry point. Returns new session id or None."""
    source = getattr(args, "from_source", None)
    path = getattr(args, "path", None)
    if path:
        # A missing file is reported as such, not as the misleading "cannot infer source".
        if not Path(path).exists():
            print(f"Error: file not found: {path}")
            return None
        if not source:  # guess from the path shape; a more specific match wins
            source = infer_foreign_source(str(path))
        if not source:
            print(f"Cannot infer source from path; pass --from {'|'.join(foreign_source_names())}.")
            return None
        chosen_path = Path(path)
    else:
        if (picked := pick_foreign_session(source)) is None:
            return None
        source, chosen_path = picked.source, picked.path
    try:
        session_id = import_foreign_session(source, chosen_path, db=db)
    except ValueError as e:
        print(f"Error: {e}")
        return None
    print(f"✓ {_SOURCE_LABELS.get(source, source)} session is {session_id}")
    print(f"  Continue it with:  hermes --resume {session_id}")
    return session_id
