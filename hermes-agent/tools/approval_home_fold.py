"""Fold the resolved user home and HERMES_HOME in a command to ``~/`` / ``~/.hermes/``.

Detection-time normalization for :mod:`tools.approval_detection`: the dangerous-command patterns
name Hermes's own files through ``~/.hermes`` / ``$HERMES_HOME`` spellings, so every spelling a shell
uses for the resolved homes (native, Git Bash ``/c/…``, ``~``, ``$HOME``, ``$LOCALAPPDATA``, …) is
rewritten into those forms first.
"""

import functools
import os
import re

from hermes_platform.host import facts


# Shell metacharacters, quotes, and whitespace that terminate a path token.
_PATH_TOKEN_STOP = r"""\s'"`;|&<>()"""
_PATH_TAIL = r"(?P<tail>(?:[/\\][^/\\" + _PATH_TOKEN_STOP + r"]*)+)"


@functools.lru_cache(maxsize=64)
def _home_prefix_fold_regex(path: str):
    """Compile a regex matching *path* as an absolute directory prefix.
    Components match with either separator so native Windows, forward-slash, and mixed forms all
    fold; the caller normalizes the tail's backslashes to ``/``. A non-empty tail is required, so a
    bare home is never folded. Returns ``None`` for an unset/degenerate path (fewer than two
    components: ``/``, ``C:\\``, ``""``) so a stray HOME cannot rewrite unrelated prefixes."""
    components = [c for c in re.split(r"[/\\]+", path) if c] if path else []
    if len(components) < 2:
        return None
    # Optional leading root separator; a Windows drive letter is a component.
    return re.compile(r"[/\\]*" + r"[/\\]+".join(re.escape(c) for c in components) + _PATH_TAIL)


def _fold_home_prefixes(command: str, paths, replacement: str) -> str:
    """Fold each resolved home prefix in *command* to *replacement* (no trailing separator; the tail
    supplies it). Longest first so a deeper home folds before a shorter overlapping one that would clobber it."""
    for path in dict.fromkeys(sorted((p for p in paths if p), key=len, reverse=True)):
        pattern = _home_prefix_fold_regex(path)
        if pattern is not None:
            command = pattern.sub(lambda m: replacement + m.group("tail").replace("\\", "/"), command)
    return command


_MSYS_DRIVE_PATH_RE = re.compile(r"^([A-Za-z]):[/\\](.+)$")


def _msys_drive_spellings(path: str) -> list[str]:
    """``/c/Users/me`` (and the capital-drive form) for ``C:\\Users\\me``; empty without a drive."""
    match = _MSYS_DRIVE_PATH_RE.match(path) if path else None
    if match is None:
        return []
    letter, rest = match.group(1), match.group(2)
    return [f"/{letter.lower()}/{rest}", f"/{letter.upper()}/{rest}"]


def _path_below(root: str, path: str) -> str | None:
    """*path* relative to *root* when it sits underneath (either separator; Windows folds case),
    else ``None``. Only the comparison folds, so the caller gets the tail as it was written."""
    root_parts = [p for p in re.split(r"[/\\]+", root) if p]
    path_parts = [p for p in re.split(r"[/\\]+", path) if p]
    if not root_parts or len(path_parts) <= len(root_parts):
        return None
    head, wanted = path_parts[:len(root_parts)], root_parts
    if facts.os_family() == "win32":
        head, wanted = [p.lower() for p in head], [p.lower() for p in wanted]
    if head != wanted:
        return None
    return "/".join(path_parts[len(root_parts):])


def _home_spellings(paths, command: str) -> list[str]:
    """Every spelling a shell uses for one of *paths*: as written (native and Git Bash drive form)
    and the same location written through a home variable — ``~``, ``$HOME``, ``$LOCALAPPDATA`` /
    ``%LOCALAPPDATA%``, ``$USERPROFILE``. Windows's default HERMES_HOME nests under the profile,
    which Git Bash reaches as ``/c/Users/<me>/AppData/Local/hermes``; folding only the absolute
    native form left that spelling — and its ``~/…`` and ``$LOCALAPPDATA/…`` siblings — unmatched.
    The variable spellings are only derived when *command* writes one of ``~``, ``$``, ``%``, so a
    command with no variable syntax costs nothing beyond its own absolute prefixes."""
    absolute = list(dict.fromkeys(s for path in paths for s in (path, *_msys_drive_spellings(path))))
    if not any(marker in command for marker in ("~", "$", "%")):
        return absolute  # no variable syntax, so no variable spelling can match either
    # `~` is $HOME in Git Bash, so the variable wins over the native reading of it.
    home = os.environ.get("HOME") or os.path.expanduser("~")
    local_appdata = os.environ.get("LOCALAPPDATA", "")
    roaming = os.environ.get("APPDATA", "")
    profile = os.environ.get("USERPROFILE", "")
    named = (("HOME", home), ("LOCALAPPDATA", local_appdata), ("APPDATA", roaming), ("USERPROFILE", profile))
    roots = [("~", home)] + [(spelled, root) for name, root in named
                             for spelled in (f"${name}", f"${{{name}}}", f"%{name}%")]
    relative = [f"{prefix}/{tail}"
                for prefix, root in roots
                for spelled_root in dict.fromkeys([root, *_msys_drive_spellings(root)])
                for candidate in absolute if (tail := _path_below(spelled_root, candidate))]
    return absolute + list(dict.fromkeys(relative))


def _rewrite_resolved_user_home(command: str) -> str:
    """User home (expanduser / realpath / $HOME) -> ``~/``; no-op when unset, degenerate, or unresolvable."""
    try:
        # expanduser, realpath, and an explicit HOME — Windows expanduser uses USERPROFILE, not HOME.
        home = os.path.expanduser("~")
        paths = [home, os.path.realpath(home), os.environ.get("HOME", "")]
        paths += [spelling for path in list(paths) if path for spelling in _msys_drive_spellings(path)]
    except Exception:
        return command
    return _fold_home_prefixes(command, paths, "~")


def _rewrite_resolved_hermes_home(command: str) -> str:
    """Resolved HERMES_HOME -> ``~/.hermes/`` so the _HERMES_CONFIG_PATH / _HERMES_ENV_PATH rules
    match Docker/gateway deployments and the platform default in every spelling a shell uses for it
    (native, Git Bash ``/c/…``, ``~/AppData/Local/hermes``, ``$LOCALAPPDATA/hermes``)."""
    try:
        from hermes_constants import get_hermes_home
        home = get_hermes_home().expanduser()
        paths = _home_spellings([str(home), str(home.resolve(strict=False))], command)
    except Exception:
        return command
    return _fold_home_prefixes(command, paths, "~/.hermes")
