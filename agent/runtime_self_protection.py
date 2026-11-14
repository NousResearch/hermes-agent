"""The running Hermes runtime's own interpreter/venv is not agent-deletable.

Deleting the Python interpreter the current process boots from — or the base
interpreter its venv depends on — bricks the install: the next boot fails with
``uv trampoline failed to spawn Python child process`` and no amount of agent
work can repair it, because the agent itself no longer starts (#58748). The
same applies to overwriting the interpreter through the file tools.

Two questions, one coordinate system:

* :func:`command_deletes_runtime` — would a shell command delete a protected
  path? Wired into the approval floor (``tools.approval._floor_block``), so it
  cannot be bypassed by yolo / approvals.mode=off / cron approve mode.
* :func:`is_protected_path` — is a filesystem path protected? Wired into the
  file-safety write classifier, covering write/patch/move/delete.

Protected set (all resolved through the same normalizer so shell spellings —
native, git-bash ``/c/...``, WSL ``/mnt/c/...``, ``$HOME`` — compare equal):

* ``sys.executable`` and ``sys.prefix`` (the runtime's own venv),
* the base interpreter from ``pyvenv.cfg``'s ``home =`` / ``sys._base_executable``,
* the uv-managed install directory holding that base (matched by version for
  ``uv python uninstall``).

Deliberately NOT protected: any other venv or interpreter on the machine — the
agent may freely manage project environments. Only the runtime it is itself
running from is off-limits.
"""

from __future__ import annotations

import os
import re
import shlex
import sys
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple, Optional

# Split a command line into individually-analyzable commands. Newlines, `;`,
# `&&`, `||` and `|` each start a fresh command whose own name decides whether
# it deletes anything.
_SEGMENT_SPLIT_RE = re.compile(r"\n|&&|\|\||[;|]")
# The same split keeping the separators: whether a `cd` moved the shell for the
# NEXT segment depends on how they are joined.
_SEGMENT_SEP_SPLIT_RE = re.compile(r"(\n|&&|\|\||[;|])")

# Words whose argument tail is a *different* command (sudo rm ...) or an
# environment assignment prefix (FOO=1 rm ...).
_COMMAND_PREFIXES = frozenset({"sudo", "command", "nohup", "exec", "env", "nice", "time"})

# Direct-deletion command names (basename, lowercase). `rm` covers POSIX and
# git-bash; the Windows-native spellings cover cmd (`rd`, `del`, `rmdir /s`)
# and PowerShell (`Remove-Item`, and its `ri`/`rm`/`del`/`erase` aliases).
_DELETING_COMMANDS = frozenset({
    "rm", "rmdir", "rd", "del", "erase", "remove-item", "ri",
})

# `cd`/`pushd` move the shell before the segments after them run; `popd` returns
# to a directory only the shell knows. Tracking `cd` is what makes a relative
# delete target resolve against the directory `rm` will actually act in (#143691).
_DIRECTORY_CHANGE_COMMANDS = frozenset({"cd", "pushd"})
_DIRECTORY_POP_COMMANDS = frozenset({"popd"})

# `find <roots...> -delete` (or `-exec rm ...`) deletes its search roots.
_FIND_DELETE_RE = re.compile(r"(?:^|\s)-(?:delete|exec\s+rm\b)", re.IGNORECASE)

# uv-managed install layout: .../uv/python/cpython-3.14.7-arm64-apple-darwin/...
_UV_INSTALL_DIR_RE = re.compile(
    r"(?i)(?P<root>.*[/\\](?:uv[/\\])?python[/\\]cpython-(?P<version>\d+(?:\.\d+)*)(?:[-_][^/\\]*)?)(?:[/\\].*)?$"
)

_WIN_DRIVE_FROM_POSIX_RE = re.compile(r"^/(?:(mnt)/)?([a-zA-Z])/(.+)$")


def _normalize_path(raw: str) -> str:
    """One canonical form for both shell spellings and runtime paths.

    Expands ``~``/``$HOME``, maps git-bash (``/c/...``) and WSL (``/mnt/c/...``)
    drive forms to native Windows paths, then ``realpath``s so symlink mirrors
    (macOS ``/private``) and relocated installs compare equal. Missing tails are
    fine: POSIX ``realpath`` resolves the existing prefix lexically.
    """
    path = str(raw or "").strip()
    if not path:
        return ""
    if len(path) >= 2 and path[0] == path[-1] and path[0] in "\"'":
        path = path[1:-1]
    try:
        path = os.path.expandvars(os.path.expanduser(path))
    except Exception:
        try:
            path = os.path.expanduser(path)
        except Exception:
            pass
    if os.name == "nt":
        m = _WIN_DRIVE_FROM_POSIX_RE.match(path)
        if m:
            path = f"{m.group(2)}:\\{m.group(3)}"
        path = path.replace("/", "\\")
    from tools.environments.sandbox_scope import sandbox_lexical_path
    if (lexical := sandbox_lexical_path(path)) is not None:
        return lexical  # a hosted agent's sandbox path: never resolved against this host
    try:
        return os.path.normcase(os.path.realpath(os.path.normpath(path)))
    except Exception:
        return os.path.normcase(os.path.normpath(path))


def _pyvenv_home(prefix: str) -> str:
    """The ``home =`` base-interpreter directory from ``<prefix>/pyvenv.cfg``."""
    try:
        lines = (Path(prefix) / "pyvenv.cfg").read_text(encoding="utf-8-sig", errors="replace").splitlines()
    except Exception:
        return ""
    for line in lines:
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip().lower() == "home":
            return value.strip()
    return ""


@lru_cache(maxsize=8)
def _protected_snapshot(executable: str, prefix: str) -> tuple[tuple[str, str], ...]:
    """Resolved ``(path, description)`` pairs for the runtime's own interpreter.

    Cached per (executable, prefix) — the pair cannot change while a process
    runs, and keying on the inputs lets tests swap ``sys`` attributes freely.
    """
    entries: list[tuple[str, str]] = []

    exe = _normalize_path(executable)
    if exe:
        entries.append((exe, "the Python interpreter this Hermes runtime is running from"))

    venv = _normalize_path(prefix)
    if venv and venv != exe:
        entries.append((venv, "this Hermes runtime's own virtualenv"))

    base_dir = _normalize_path(_pyvenv_home(prefix)) or _normalize_path(getattr(sys, "_base_executable", "") or "")
    if base_dir:
        entries.append((base_dir, "the base interpreter this Hermes venv depends on"))
        m = _UV_INSTALL_DIR_RE.match(base_dir.replace("\\", "/"))
        if m:
            uv_root = _normalize_path(m.group("root"))
            if uv_root:
                entries.append((uv_root, f"the uv-managed Python install this Hermes venv depends on ({m.group('version')})"))

    return tuple((path, desc) for path, desc in entries if path)


def _protected() -> tuple[tuple[str, str], ...]:
    return _protected_snapshot(getattr(sys, "executable", "") or "", getattr(sys, "prefix", "") or "")


def _overlaps(a: str, b: str) -> bool:
    """True when ``a`` and ``b`` are the same path or one is an ancestor of the other."""
    if not a or not b:
        return False
    sep = os.sep
    return a == b or a.startswith(b + sep) or b.startswith(a + sep)


def split_entry(path: str) -> tuple[str, str]:
    """``os.path.split`` for a directory entry, trailing separators and ``.`` components
    dropped first: an empty or ``.`` leaf (``dir/link/``, ``dir/link/.``) would make entry
    checks degenerate to the link's target, while pathlib/rm still act on ``link``.
    A bare root (``/``, ``C:\\``) is kept as is."""
    drive, tail = os.path.splitdrive(path)
    parent, leaf = os.path.split(drive + (tail.rstrip(os.sep + (os.altsep or "")) or tail[:1]))
    if leaf == "." and parent:
        return split_entry(parent)
    return parent, leaf


def is_protected_path(path: str, *, follow: bool = True) -> Optional[str]:
    """Description of the protected runtime path ``path`` touches, else ``None``.

    ``follow=False`` keeps the final component unresolved (the entry itself, for
    ops that unlink/rename a symlink rather than its target)."""
    if follow:
        resolved = _normalize_path(path)
    else:
        parent, leaf = split_entry(path)
        resolved = _normalize_path(parent or ".")
        resolved = resolved and os.path.normcase(os.path.join(resolved, leaf))
    if not resolved:
        return None
    for protected, description in _protected():
        if _overlaps(resolved, protected):
            return description
    return None


def _split_words(segment: str) -> list[str]:
    try:
        return shlex.split(segment)
    except ValueError:
        return segment.split()


def _strip_prefixes(words: list[str]) -> list[str]:
    out = list(words)
    while out:
        first = out[0]
        if first in _COMMAND_PREFIXES:
            out = out[1:]
            continue
        if "=" in first and not first.startswith(("-", "/")) and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", first):
            out = out[1:]
            continue
        break
    return out


def _version_spec_matches(spec: str, version: str, uv_root_name: str) -> bool:
    """``uv python uninstall`` spec against the running base's version/dir."""
    spec = spec.strip().lower()
    if not spec:
        return False
    if spec in ("--all", "-a", "all"):
        return True
    if spec.startswith("-"):
        return False
    if uv_root_name.lower().startswith(spec):
        return True
    return spec == version or (version.startswith(spec) and version[len(spec):len(spec) + 1] == ".")


def _resolve_target(raw: str, cwd: str) -> str:
    """A delete target as the shell would spell it from ``cwd``.

    An absolute word (or one whose ``~`` expands to an absolute path) is already
    unambiguous; anything else is joined onto ``cwd`` so a preceding ``cd`` is
    honoured. When ``cwd`` is unknown (``""``) the word is returned verbatim and
    ``_normalize_path`` resolves it against the process cwd, as it always did.
    """
    word = (raw or "").strip().strip("\"'")
    if not word or os.path.isabs(os.path.expandvars(os.path.expanduser(word))):
        return word  # `$HOME/...` is as absolute as `~/...`
    return os.path.join(cwd, word) if cwd else word


class _DirectoryChange(NamedTuple):
    """One ``cd``/``pushd``/``popd`` segment, parsed once."""

    target: str  # absolute, or relative to the shell's cwd; ``""`` when only the shell knows
    plain: bool  # one operand and no option bash may reject: it moves iff ``target`` is enterable
    physical: bool = False  # `-P`: a later `..` may leave the logical path, so the old reading stays too

    def destination(self, cwd: str) -> str:
        if not self.target or os.path.isabs(self.target):
            return self.target
        return os.path.join(cwd, self.target) if cwd else ""


_UNKNOWN_DIRECTORY = _DirectoryChange("", plain=False)


def _parse_directory_change(words: list[str], *, segment: str, home_known: bool) -> _DirectoryChange:
    """Where a ``cd``/``pushd``/``popd`` segment moves the shell.

    ``cd -`` (OLDPWD), ``cd "$VAR"``, globs, ``popd`` and stack rotation depend on
    shell state that is not in the command string. Those answer ``""`` so later
    segments keep the process-cwd reading instead of a guessed directory that
    would both block and miss real deletes.
    """
    name = words[0].lower()
    if name in _DIRECTORY_POP_COMMANDS:
        return _UNKNOWN_DIRECTORY  # popd returns to a directory only the shell knows
    if name == "pushd" and (len(words) != 2 or words[1][:1] in "+-"):
        return _UNKNOWN_DIRECTORY  # a bare `pushd`, `pushd +N` / `-N` or `-n` rotates the stack
    options, operands = [], []
    for word in words[1:]:
        if word in ("-", "--"):
            return _UNKNOWN_DIRECTORY  # `cd -` is OLDPWD; after `--` an operand may look like an option
        if word.startswith("-") and not operands:
            options.append(word)
        elif os.name == "nt" and re.match(r"^/[A-Za-z]$", word):
            options.append(word)  # cmd's `cd /d <drive>`
        else:
            operands.append(word)
    target = operands[0] if operands else "~"  # a bare `cd` goes home
    if "'" in target or '"' in target:
        return _UNKNOWN_DIRECTORY  # quotes survived the split: it broke inside a quoted word
    if re.search(r"[$`%*?[]", target):
        return _UNKNOWN_DIRECTORY  # expansions and globs only the shell resolves
    expanded = os.path.expanduser(target)
    if target.startswith("~") and (
        expanded.startswith("~") or not home_known or re.search(r"[\"'\\]", segment)
    ):
        # `~+`, `~-`, an unknown user, a reassigned HOME, or a quoted `~` (`cd ""` too): bash does not go home
        return _UNKNOWN_DIRECTORY
    plain = len(operands) <= 1 and all(o == "-L" for o in options)
    # `cd -P` resolves symlinks; `realpath` cannot stand in for it (non-strict, it
    # collapses `missing/../x`), so the logical path and the process cwd both stay.
    physical = any("P" in o for o in options if o.startswith("-"))
    return _DirectoryChange(expanded, plain=plain, physical=physical)


def _uv_uninstall_target(words: list[str]) -> Optional[str]:
    """Description when ``uv python uninstall`` would remove the running base."""
    protected = _protected()
    uv_entries = [
        (root, desc) for root, desc in protected
        if desc.startswith("the uv-managed Python install")
    ]
    if not uv_entries:
        return None
    lowered = [w.lower() for w in words]
    try:
        python_at = lowered.index("python")
        uninstall_at = lowered.index("uninstall", python_at + 1)
    except ValueError:
        return None
    specs = words[uninstall_at + 1:]
    for root, description in uv_entries:
        m = _UV_INSTALL_DIR_RE.match(root.replace("\\", "/"))
        version = m.group("version") if m else ""
        root_name = os.path.basename(root)
        if any(_version_spec_matches(spec, version, root_name) for spec in specs):
            return description
    return None


def command_deletes_runtime(command: str) -> Optional[str]:
    """Description of the runtime path ``command`` would delete, else ``None``.

    Covers ``rm``/``rmdir``/``rd``/``del``/``erase``/``Remove-Item`` (any flags —
    the interpreter is a file, so no recursion is needed to kill it), the
    ``find <roots> -delete`` form, and ``uv python uninstall``. A relative
    target is resolved against the directory an earlier ``cd``/``pushd`` moved
    the shell to, falling back to the process cwd when that cannot be known
    statically (``cd -``, ``cd "$DIR"``, ``popd``). Known limits: paths
    assembled by ``xargs``/command substitution are invisible here, and
    ``shutil.rmtree`` inside executed *code* is a different sandbox.
    """
    if not command or not command.strip():
        return None
    protected = _protected()
    if not protected:
        return None

    try:
        start = _normalize_path(os.getcwd())
    except OSError:  # the process cwd was removed under us
        start = ""
    walk = _ShellWalk(
        (start,),
        cdpath="CDPATH" in command or os.environ.get("CDPATH", "") != "",
        commented=re.search(r"(?:^|[\s;&|()])#", command) is not None,
    )
    home_known = re.search(r"\bHOME\b", command) is None  # `HOME=X; cd` goes to X
    parts = _SEGMENT_SEP_SPLIT_RE.split(command)
    for index in range(0, len(parts), 2):
        segment = parts[index]
        joiner = parts[index + 1] if index + 1 < len(parts) else ""
        raw = _split_words(segment)
        words = _strip_prefixes(raw)
        if not words or words == ["\\"]:
            # After `|` a newline or `\` still leaves the next command in the pipeline.
            walk.prev_joiner = "|" if walk.prev_joiner == "|" else joiner
            continue
        if _is_directory_change(raw) and "|" not in (walk.prev_joiner, joiner):
            change = _parse_directory_change(raw, segment=segment, home_known=home_known)
            walk.step(*walk.after_directory_change(change), joiner)
            continue
        hit = _segment_deletes_runtime(segment, words, os.path.basename(words[0]).lower(), walk.cwds)
        if hit:
            return hit
        walk.trusted = False  # this command may have created or removed directories
        # `builtin cd`, `then cd ..`, `{ cd ..; }`, `eval 'cd ..'` or a function may still move the shell.
        unmodeled_cd = re.search(r"(?<![\w./-])(?:cd|pushd|popd)(?![\w./-])", segment)
        walk.step(_cap(walk.cwds + ("",)) if unmodeled_cd else walk.cwds, False, joiner)

    return None


# Past this many candidate directories the walk collapses to the process cwd
# plus the newest few: `;`-joined cds into missing relative dirs otherwise
# double the set per cd, and this runs before every terminal command.
_MAX_CANDIDATE_CWDS = 8


def _cap(cwds: tuple[str, ...]) -> tuple[str, ...]:
    cwds = tuple(dict.fromkeys(cwds))
    if len(cwds) <= _MAX_CANDIDATE_CWDS:
        return cwds
    return tuple(dict.fromkeys(("",) + cwds[-(_MAX_CANDIDATE_CWDS // 2):]))


def _is_directory_change(raw: list[str]) -> bool:
    """Whether the segment runs the shell's own ``cd``/``pushd``/``popd`` builtin.

    ``sudo cd``, ``env cd``, ``FOO=1 cd``, ``/usr/bin/cd`` and ``CD`` (POSIX) run
    something else, which never moves the shell, so they do not count.
    """
    if not raw:
        return False
    name = raw[0].lower() if os.name == "nt" else raw[0]
    return name in _DIRECTORY_CHANGE_COMMANDS or name in _DIRECTORY_POP_COMMANDS


class _ShellWalk:
    """Directories the shell may be in as it walks a command's segments.

    ``cwds`` is where the current segment may run; ``list_cwds`` is every
    directory seen since the current and/or-list began, because a failure
    anywhere in an ``&&`` chain skips the rest and a later ``||``, ``;`` or
    newline segment then runs in whatever directory the shell had reached.
    """

    def __init__(self, cwds: tuple[str, ...], *, cdpath: bool, commented: bool):
        self.cwds = cwds
        self.list_cwds = cwds
        self.list_certain = True  # every segment of the current list surely succeeded
        self.list_has_or = False  # a `||` earlier in the list can skip a later cd
        self.trusted = True  # nothing has run yet that could change the filesystem
        self.cdpath = cdpath  # CDPATH makes a relative cd target unknowable
        self.commented = commented  # a comment can hide the separators around a cd
        self.prev_joiner = ""

    def after_directory_change(self, change: _DirectoryChange) -> tuple[tuple[str, ...], bool]:
        """Directories the shell may be in after ``change`` ran, and whether it surely moved."""
        moved = tuple(change.destination(cwd) for cwd in self.cwds) + (("",) if change.physical else ())
        if self.cdpath and change.target and not os.path.isabs(change.target):
            return _cap(self.cwds + moved + ("",)), False
        if self.trusted:
            # Nothing ran yet that could create a directory, so a cd into a missing
            # one fails and leaves the shell where it was (already in ``list_cwds``).
            reachable = tuple(t for t in moved if not t or _enterable(t))
        else:
            reachable = moved
        if self.list_has_or or self.commented:  # `true || cd X && rm ...` skips the cd and still runs rm
            return _cap(self.cwds + reachable), False
        # Existence is only trusted before anything ran that could change it,
        # and `cd a b` / `cd -x a` fail without moving the shell.
        certain = self.trusted and change.plain and bool(moved) and all(moved) and reachable == moved
        return _cap(reachable), certain

    def step(self, after: tuple[str, ...], certain: bool, joiner: str) -> None:
        self.list_certain = self.list_certain and certain
        self.list_cwds = _cap(self.list_cwds + after)
        if joiner in ("&&", "|"):
            self.cwds = after
        elif joiner == "||":
            self.cwds = self.list_cwds
            self.list_has_or = True
        else:  # `;`, newline, end
            self.cwds = after if self.list_certain else self.list_cwds
            self.list_cwds = self.cwds
            self.list_certain = True
            self.list_has_or = False
        self.prev_joiner = joiner


def _enterable(path: str) -> bool:
    return os.path.isdir(path) and os.access(path, os.X_OK)


def _segment_deletes_runtime(segment: str, words: list[str], name: str, cwds: tuple[str, ...]) -> Optional[str]:
    """Protected path one command segment deletes, its relative targets read from each of ``cwds``."""
    if name == "uv":
        return _uv_uninstall_target(words)
    if name not in _DELETING_COMMANDS and not (name == "find" and _FIND_DELETE_RE.search(segment)):
        return None
    for word in (word for word in words[1:] if not word.startswith("-")):
        for cwd in cwds:
            hit = is_protected_path(_resolve_target(word, cwd))
            if hit:
                return hit
    return None
