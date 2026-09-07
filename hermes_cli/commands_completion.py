"""prompt_toolkit completer + inline auto-suggest for slash commands. Kept out of
:mod:`hermes_cli.commands` (which re-exports both classes) so the registry module stays
prompt_toolkit-free for the gateway."""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
import time
from collections.abc import Callable, Iterable, Mapping
from itertools import chain
from typing import Any, Dict, Optional, Tuple

from prompt_toolkit.auto_suggest import AutoSuggest, Suggestion
from prompt_toolkit.completion import Completer, Completion

from hermes_cli.commands import COMMANDS, SUBCOMMANDS

# (config-file signature, personalities) memo for /personality completion.
_personalities_memo: Optional[
    Tuple[Tuple[Optional[str], Optional[int], Optional[int]], Dict[str, Any]]
] = None


def _personalities_from_cli_config() -> Dict[str, Any]:
    """``available_personalities(load_cli_config())`` memoised on config path+mtime+size:
    load_cli_config() is a full YAML parse + deep merge and the completer runs per keystroke.
    Falls back to a fresh load when the file cannot be stat'ed."""
    global _personalities_memo
    from cli import load_cli_config
    from hermes_cli.personality import available_personalities
    try:
        from hermes_cli.config import get_config_path
        cfg_path = get_config_path()
        st = cfg_path.stat()
        sig = (str(cfg_path), st.st_mtime_ns, st.st_size)
    except Exception:
        sig = (None, None, None)
    if _personalities_memo is None or _personalities_memo[0] != sig:
        _personalities_memo = (sig, available_personalities(load_cli_config()))
    return _personalities_memo[1]


def _short_desc(info: Mapping[str, Any], default: str) -> str:
    """50-char description preview used in completion menus."""
    description = str(info.get("description", default))
    return description[:50] + ("..." if len(description) > 50 else "")


def _file_size_label(path: str) -> str:
    """Return a compact human-readable file size, or '' on error."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return ""
    if size < 1024:
        return f"{size}B"
    if size < 1024 ** 2:
        return f"{size / 1024:.0f}K"
    return f"{size / 1024 ** 2:.1f}M" if size < 1024 ** 3 else f"{size / 1024 ** 3:.1f}G"


def _completion(text: str, word: str, display: str, meta) -> Completion:
    """``Completion`` replacing the *word* under the cursor."""
    return Completion(text, start_position=-len(word), display=display, display_meta=meta)


def _prefix_completions(
    rows: Iterable[tuple[str, Any]], partial: str, *, skip_exact: bool = True):
    """A Completion per ``(name, meta)`` whose name starts with the case-folded *partial*."""
    lowered = partial.lower()
    for name, meta in rows:
        if name.startswith(lowered) and not (skip_exact and name == lowered):
            yield _completion(name, partial, name, meta)


def _split_args(sub_text: str) -> tuple[list[str], str]:
    """``(completed_words, partial)``; a trailing space means a fresh word."""
    parts = sub_text.split()
    if sub_text.endswith(" ") or not parts:
        return parts, ""
    return parts[:-1], parts[-1]


# Dynamic argument completers: (sub_text, sub_lower) -> Completion iterator
def _quiet(gen_fn):
    """Generator decorator: any exception while producing completions just ends the stream."""
    @functools.wraps(gen_fn)
    def wrapper(*args):
        try:
            yield from gen_fn(*args)
        except Exception:
            return
    return wrapper


@_quiet
def _skin_completions(sub_text: str, sub_lower: str):
    """/skin — available skins."""
    from hermes_cli.skin_engine import list_skins
    rows = ((s["name"], s.get("description", "") or s.get("source", "")) for s in list_skins())
    yield from _prefix_completions(rows, sub_text)


@_quiet
def _personality_completions(sub_text: str, sub_lower: str):
    """/personality — ``none`` plus configured personalities."""
    from hermes_cli.personality import describe_personality
    personalities = _personalities_from_cli_config()
    rows = chain(
        [("none", "clear personality overlay")],
        ((name, describe_personality(prompt)) for name, prompt in personalities.items()))
    yield from _prefix_completions(rows, sub_text)


@_quiet
def _tools_completions(sub_text: str, sub_lower: str):
    """/tools — subcommand, then toolset / MCP-server names for enable|disable. Toolsets are
    offered only when the subcommand would change their state; MCP server prefixes always."""
    completed, partial = _split_args(sub_text)
    if not completed:
        yield from _prefix_completions(((s, None) for s in ("list", "disable", "enable")), partial)
        return
    subcommand = completed[0].lower()
    if subcommand not in ("enable", "disable"):
        return
    already = set(completed[1:])
    from hermes_cli.config import load_config_readonly
    from hermes_cli.tools_config import (
        CONFIGURABLE_TOOLSETS, _get_platform_tools, _get_plugin_toolset_keys)
    # Readonly loader: per keystroke and never mutates, so skip load_config()'s deepcopy.
    # Read-only path: the completer only inspects the config (toolset enable state + MCP server names) — it
    # never mutates it. Use the readonly loader so the per-keystroke completion doesn't pay the defensive
    # deepcopy (perf(agent) #74322 converted 29 call sites to the readonly loader; this per-keystroke site
    # was missed).
    config = load_config_readonly()
    enabled = _get_platform_tools(config, "cli", include_default_mcp_servers=False)
    mcp_servers = config.get("mcp_servers") or {}
    want_enabled = subcommand != "enable"
    rows = [(k, label) for k, label, _d in CONFIGURABLE_TOOLSETS]
    rows += [(k, "plugin toolset") for k in sorted(_get_plugin_toolset_keys())]
    rows = [(k, m) for k, m in rows if (k in enabled) == want_enabled]
    if isinstance(mcp_servers, dict):
        rows += [(f"{srv}:", f"MCP server '{srv}'") for srv in sorted(mcp_servers)]
    yield from _prefix_completions(
        ((k, m) for k, m in rows if k not in already), partial, skip_exact=False)


def _handoff_completions(sub_text: str, sub_lower: str):
    """/handoff — connected gateway platforms, first arg only. A home channel is not required
    (often learned at runtime); the meta hints whether one is set."""
    completed, partial = _split_args(sub_text)
    if completed:
        return
    try:
        from gateway.config import load_gateway_config
        gw = load_gateway_config()
        platforms = gw.get_connected_platforms()
    except Exception:
        return
    for platform in platforms:
        name = platform.value
        if not name.startswith(partial.lower()):
            continue
        try:
            home = gw.get_home_channel(platform)
        except Exception:
            home = None
        home_name = getattr(home, "name", None) if home else None
        yield _completion(
            name, partial, name, f"→ {home_name}" if home_name else "send this session here")


# base command -> (handler(sub_text, sub_lower), single_word_only). Single-word handlers only
# run while the first argument is typed; /tools and /handoff parse multi-word input themselves.
_DYNAMIC_COMPLETIONS: dict[str, tuple[Callable[..., Any], bool]] = {
    "/skin": (_skin_completions, True),
    "/personality": (_personality_completions, True),
    "/tools": (_tools_completions, False),
    "/handoff": (_handoff_completions, False)}


def _extract_path_word(text: str) -> str | None:
    """Word under the cursor when it contains ``/`` and no ``://`` scheme (URLs aren't paths)."""
    word = text.rpartition(" ")[2]
    return word if word and "://" not in word and "/" in word else None


def _dir_completions(
    expanded: str, word: str, limit: int, text_for: Callable[[str], str],
    want_dir: bool | None = None):
    """Directory-listing completions for *expanded*: entries matched case-insensitively on the
    typed basename (all entries after a trailing ``/``), sorted, limited to *limit*.
    ``text_for(full_path)`` builds the completion text (without trailing ``/``); *want_dir*
    restricts to dirs / files."""
    if expanded.endswith("/"):
        search_dir, prefix = expanded, ""
    else:
        search_dir, prefix = os.path.dirname(expanded) or ".", os.path.basename(expanded)
    try:
        entries = os.listdir(search_dir)
    except OSError:
        return
    prefix_lower = prefix.lower()
    count = 0
    for entry in sorted(entries):
        if not entry.lower().startswith(prefix_lower):
            continue
        full_path = os.path.join(search_dir, entry)
        is_dir = os.path.isdir(full_path)
        if want_dir is not None and want_dir != is_dir:
            continue
        if count >= limit:
            break
        suffix = "/" if is_dir else ""
        yield _completion(
            text_for(full_path) + suffix, word, entry + suffix,
            "dir" if is_dir else _file_size_label(full_path))
        count += 1


def _path_completions(word: str, limit: int = 30):
    """Path completions for *word*, keeping the user's style (~, absolute, relative)."""
    if word.startswith("~"):
        text_for = lambda fp: "~/" + os.path.relpath(fp, os.path.expanduser("~"))  # noqa: E731
    else:
        text_for = str if os.path.isabs(word) else os.path.relpath
    yield from _dir_completions(os.path.expanduser(word), word, limit, text_for)


_STATIC_CONTEXT_REFS = (
    ("@diff", "Git working tree diff"),
    ("@staged", "Git staged diff"),
    ("@file:", "Attach a file"),
    ("@folder:", "Attach a folder"),
    ("@git:", "Git log with diffs (e.g. @git:5)"),
    ("@url:", "Fetch web content"))


def _score_path(filepath: str, query: str) -> int:
    """Score a file path against a fuzzy query. Higher = better; 0 = no match; 1 = empty query."""
    if not query:
        return 1
    lower_file = os.path.basename(filepath).lower()
    lower_q = query.lower()
    for score, hit in ((100, lower_file == lower_q), (80, lower_file.startswith(lower_q)),
                       (60, lower_q in lower_file), (40, lower_q in filepath.lower())):
        if hit:
            return score
    # Abbreviation: query chars in order in the filename ("fo" ~ "file_operations"); bonus when
    # >= half land on word boundaries (_-./).
    qi = boundary_hits = 0
    prev = "_"  # treat start as boundary
    for c in lower_file:
        if qi < len(lower_q) and c == lower_q[qi]:
            boundary_hits += prev in "_-./"
            qi += 1
        prev = c
    return 0 if qi < len(lower_q) else 35 if boundary_hits >= len(lower_q) * 0.5 else 25


class SlashCommandCompleter(Completer):
    """Autocomplete for built-in slash commands, subcommands, and skill commands."""

    # Bare-run picker commands get no trailing space: "/model " would block the picker.
    _PICKER_COMMANDS = frozenset({"model", "skin", "personality"})

    # Module-level helpers exposed as staticmethods for existing callers/tests.
    _extract_path_word = staticmethod(_extract_path_word)
    _path_completions = staticmethod(_path_completions)
    _personality_completions = staticmethod(_personality_completions)
    _tools_completions = staticmethod(_tools_completions)

    def __init__(
        self,
        skill_commands_provider: Callable[[], Mapping[str, dict[str, Any]]] | None = None,
        command_filter: Callable[[str], bool] | None = None,
        skill_bundles_provider: Callable[[], Mapping[str, dict[str, Any]]] | None = None) -> None:
        self._skill_commands_provider = skill_commands_provider
        self._command_filter = command_filter
        self._skill_bundles_provider = skill_bundles_provider
        # Cached project file list for fuzzy @ completions
        self._file_cache: list[str] = []
        self._file_cache_time: float = 0.0
        self._file_cache_cwd: str = ""

    def _command_allowed(self, slash_command: str) -> bool:
        try:
            return self._command_filter is None or bool(self._command_filter(slash_command))
        except Exception:
            return True

    @staticmethod
    def _call_provider(provider) -> Mapping[str, dict[str, Any]]:
        try:
            return (provider() if provider is not None else None) or {}
        except Exception:
            return {}

    def _iter_skill_commands(self) -> Mapping[str, dict[str, Any]]:
        return self._call_provider(self._skill_commands_provider)

    @staticmethod
    def _normalize_skill_token(token: str) -> str:
        """Canonical hyphenated /slug; mirrors resolve_skill_command_key() (``_`` == ``-``)."""
        return "/" + token.lstrip("/").replace("_", "-").lower()

    def _is_skill_command(self, token: str) -> bool:
        return self._normalize_skill_token(token) in self._iter_skill_commands()

    def _stacked_skill_completions(self, text: str):
        """Skill-command completions for stacked invocations (``/skill-a /skill-b do XYZ``): only
        while every completed token is a distinct skill command, the cap is not reached, and the
        current word starts with ``/`` — instruction text must never get skill suggestions."""
        try:
            from agent.skill_commands import _MAX_STACKED_SKILLS as _cap
        except Exception:
            _cap = 5
        completed, current_word = _split_args(text)
        skill_cmds = self._iter_skill_commands()
        seen: set[str] = set()
        for token in completed:
            key = self._normalize_skill_token(token)
            if key not in skill_cmds or key in seen:
                return
            seen.add(key)
        if len(seen) >= _cap or not current_word.startswith("/"):
            return  # a bare space after the chain may start the instruction
        word_key = self._normalize_skill_token(current_word)
        for cmd, info in skill_cmds.items():
            if cmd in seen or not cmd.startswith(word_key):
                continue
            # Exact match: trailing space keeps the dropdown open for the next stacked token.
            yield _completion(
                f"{cmd} " if cmd == word_key else cmd, current_word, cmd,
                f"⚡ {_short_desc(info, 'Skill command')}")

    @staticmethod
    def _completion_text(cmd_name: str, word: str) -> str:
        """Replacement text; exact matches get a trailing space (else prompt_toolkit hides the
        menu on a no-op replacement) — except _PICKER_COMMANDS."""
        exact = cmd_name == word and cmd_name not in SlashCommandCompleter._PICKER_COMMANDS
        return f"{cmd_name} " if exact else cmd_name

    @staticmethod
    def _extract_context_word(text: str) -> str | None:
        """Extract a bare ``@`` token for context reference completions."""
        word = text.rpartition(" ")[2]
        return word if word.startswith("@") else None

    def _context_completions(self, word: str, limit: int = 30):
        """@ completions: static refs, ``@file:``/``@folder:`` paths, else fuzzy project files."""
        lowered = word.lower()
        for candidate, meta in _STATIC_CONTEXT_REFS:
            if candidate.startswith(lowered) and candidate != lowered:
                yield _completion(candidate, word, candidate, meta)
        # Bare `@file` / `@folder` (no colon yet) already opens the picker.
        for prefix in ("@file:", "@folder:"):
            bare = prefix[:-1]
            if word == bare or word.startswith(prefix):
                expanded = os.path.expanduser("" if word == bare else word[len(prefix):])
                if not expanded or expanded == ".":
                    expanded = "./"
                # `@folder:` = dirs only, `@file:` = files only (else `@folder:` lists dotfiles).
                yield from _dir_completions(
                    expanded, word, limit, lambda fp: f"{prefix}{os.path.relpath(fp)}",
                    want_dir=(prefix == "@folder:"))
                return
        yield from self._fuzzy_file_completions(word, word[1:], limit)

    def _get_project_files(self) -> list[str]:
        """Cached project file list (5s TTL); rg (gitignore-aware) then fd."""
        cwd = os.getcwd()
        now = time.monotonic()
        if self._file_cache and self._file_cache_cwd == cwd and now - self._file_cache_time < 5.0:
            return self._file_cache
        files: list[str] = []
        for cmd in (
            ["rg", "--files", "--sortr=modified", cwd],
            ["rg", "--files", cwd],
            ["fd", "--type", "f", "--base-directory", cwd]):
            if not shutil.which(cmd[0]):
                continue
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=2,
                    cwd=cwd, encoding="utf-8", errors="replace")
            except (subprocess.TimeoutExpired, OSError):
                continue
            if proc.returncode != 0 or not proc.stdout.strip():
                continue
            for p in proc.stdout.strip().split("\n")[:5000]:
                try:
                    files.append(os.path.relpath(p, cwd) if os.path.isabs(p) else p)
                except ValueError:
                    continue  # Windows: relpath raises across mounts/drive letters
            break
        self._file_cache, self._file_cache_time, self._file_cache_cwd = files, now, cwd
        return files

    def _fuzzy_file_completions(self, word: str, query: str, limit: int = 20):
        """Fuzzy file completions for bare @query (no query = recently modified files)."""
        files = self._get_project_files()
        if query:
            scored = sorted(
                ((s, fp) for fp in files if (s := _score_path(fp, query)) > 0),
                key=lambda x: (-x[0], x[1]))
            files = [fp for _, fp in scored]
        for fp in files[:limit]:
            is_dir = fp.endswith("/")
            meta = "dir" if is_dir else _file_size_label(os.path.join(os.getcwd(), fp))
            if query:
                meta = f"{fp}  {meta}" if meta else fp
            yield _completion(
                f"@{'folder' if is_dir else 'file'}:{fp}", word, os.path.basename(fp), meta)

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            ctx_word = self._extract_context_word(text)
            path_word = _extract_path_word(text)
            if ctx_word is not None:
                yield from self._context_completions(ctx_word)
            elif path_word is not None:
                yield from _path_completions(path_word)
            return
        parts = text.split(maxsplit=1)
        base_cmd = parts[0].lower()
        if len(parts) > 1 or text.endswith(" "):
            # Completing arguments: base command already typed.
            sub_text = parts[1] if len(parts) > 1 else ""
            first_arg = " " not in sub_text
            # Stacked slash-skill chain (see split_stacked_skill_commands in agent/skill_commands).
            if self._is_skill_command(base_cmd):
                yield from self._stacked_skill_completions(text)
                return
            handler, single_word = _DYNAMIC_COMPLETIONS.get(base_cmd, (None, False))
            if handler is not None and (not single_word or first_arg):
                yield from handler(sub_text, sub_text.lower())
            elif first_arg and base_cmd in SUBCOMMANDS and self._command_allowed(base_cmd):
                yield from _prefix_completions(
                    ((s, None) for s in SUBCOMMANDS[base_cmd]), sub_text)
            return
        word = text[1:]

        def _cmd_completion(cmd_name: str, meta: str):
            return _completion(self._completion_text(cmd_name, word), word, f"/{cmd_name}", meta)

        for cmd, desc in COMMANDS.items():
            if self._command_allowed(cmd) and cmd[1:].startswith(word):
                yield _cmd_completion(cmd[1:], desc)
        for cmd, info in self._call_provider(self._skill_bundles_provider).items():
            if cmd[1:].startswith(word):
                skill_count = len(info.get("skills", []))
                yield _cmd_completion(
                    cmd[1:], f"▣ {_short_desc(info, 'Skill bundle')} ({skill_count} skills)")
        for cmd, info in self._iter_skill_commands().items():
            if cmd[1:].startswith(word):
                yield _cmd_completion(cmd[1:], f"⚡ {_short_desc(info, 'Skill command')}")
        try:
            from hermes_cli.plugins import get_plugin_commands
            for cmd_name, cmd_info in get_plugin_commands().items():
                if cmd_name.startswith(word):
                    yield _cmd_completion(
                        cmd_name, f"🔌 {_short_desc(cmd_info, 'Plugin command')}")
        except Exception:
            pass


class SlashCommandAutoSuggest(AutoSuggest):
    """Inline ghost-text for slash commands and subcommands; history fallback for other input."""

    def __init__(
        self, history_suggest: AutoSuggest | None = None,
        completer: SlashCommandCompleter | None = None) -> None:
        self._history = history_suggest
        self._completer = completer  # Reuse its model cache

    def _allowed(self, cmd: str) -> bool:
        return self._completer is None or self._completer._command_allowed(cmd)

    def _history_suggestion(self, buffer, document):
        return self._history.get_suggestion(buffer, document) if self._history else None

    def get_suggestion(self, buffer, document):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return self._history_suggestion(buffer, document)
        parts = text.split(maxsplit=1)
        base_cmd = parts[0].lower()
        if len(parts) == 1 and not text.endswith(" "):
            # Still typing the name: prefer the SHORTEST match so /he ghosts "lp", not "artbeat".
            word = text[1:].lower()
            for cmd in sorted(COMMANDS, key=len):
                cmd_name = cmd[1:]
                if self._allowed(cmd) and cmd_name.startswith(word) and cmd_name != word:
                    return Suggestion(cmd_name[len(word):])
            return None

        sub_text = parts[1] if len(parts) > 1 else ""
        sub_lower = sub_text.lower()
        # Stacked skill chain: ghost the rest of the next skill name, else history fallback.
        if self._completer is not None and self._completer._is_skill_command(base_cmd):
            for completion in self._completer._stacked_skill_completions(text):
                start = completion.start_position
                remainder = completion.text[-start:] if start else completion.text
                if remainder.strip():
                    return Suggestion(remainder)
        if not self._allowed(base_cmd):
            return None
        if " " not in sub_text:
            for sub in SUBCOMMANDS.get(base_cmd, ()):
                if sub.startswith(sub_lower) and sub != sub_lower:
                    return Suggestion(sub[len(sub_text):])
        return self._history_suggestion(buffer, document)
