"""The local skills index backend: today's per-profile ``skills/`` directory behind the port.

:class:`LocalSkillIndex` serves one profile's skills directory, at today's path
(``<profile home>/skills/``) and in today's layout (``skills/[category/]<name>/SKILL.md`` plus
the supporting directories), scanned with today's code (``agent.skill_utils``). Each operation
is the same read or write the code makes today, after one check of the pin: the bound scope
(``hermes_scope.current_scope()``) must be a local one whose agent is the index's own profile
home. A local index holds one agent and only agent-level records, so any other pin (another
tenant, another profile) is refused with ``StoragePinRefused`` before anything is read or
written, and a write naming another level is refused with ``SkillsLevelRefused``. With no scope
bound, the pin is the ambient profile (``get_hermes_home()``), as every local caller has it
today.

The index's identity is frozen when it opens: the canonical skills directory and the profile
home that holds it, taken from the absolute path it was given, never recomputed against a later
working directory (a relative path is refused). A directory alias of the profile (a symlinked
home) is the same profile; a skills directory that is a link into another directory is refused at
open, since its files belong to whichever profile owns that directory. That identity is
re-verified before every operation: the directory must still resolve to the one the index opened
(its real path, and — when one existed at open — the same directory, ``st_dev``/``st_ino``), so a
root renamed away and replaced by a link into another profile is refused with
``StoragePinRefused`` before anything is read or written, not served as this profile's skills.
The check and the operation's own I/O are two moments: a link swapped in between them is
outside the local threat model (the machine's owner rearranging their own files under a running
process), as with the local state store.

Offer-time gates (platform, environment, app availability, the disabled list), the same-name
tier resolution across project, create-dir and external roots, and the write gates of
``skill_manage`` stay with today's callers: this backend is the agent's own skills, the level
the port models. A read of a skill that exists only under those other roots resolves here as
missing, as it would for any other agent's skill.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path, PurePosixPath
from typing import Any, Optional

from hermes_scope import LOCAL_TENANT, current_scope
from hermes_skills_port import (
    AGENT_LEVEL, SKILL_LEVELS, SkillsIndexing, SkillsIndexProvider, SkillsLevelRefused, SkillIndex,
    StoragePinRefused,
)

# Today's budgets and rules (``tools/skill_manager_tool``), shared by the port's writes.
MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
VALID_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
ALLOWED_SUBDIRS = frozenset({"references", "templates", "scripts", "assets"})
_NAME_RULE = "Use lowercase letters, numbers, hyphens, dots, and underscores."


def index_identity(skills_dir) -> tuple[Path, Path]:
    """``(canonical skills directory, owning profile home)`` of the directory this index serves.
    Refuses (``StoragePinRefused``) a relative path (which directory it names cannot be
    established later) and a directory that resolves outside the one it is named in (a link
    into another profile: its files belong to that profile's agent)."""
    named = Path(os.fspath(skills_dir))
    if not os.path.isabs(str(named)):
        raise StoragePinRefused(
            f"skills index handle {str(named)!r} does not name an absolute directory")
    home = named.parent.resolve()
    canonical = named.resolve()
    if canonical.parent != home:
        raise StoragePinRefused(
            f"skills index {named} is a link into {canonical.parent}, not a directory of its "
            f"profile {home}")
    return canonical, home


def root_stat(path) -> Optional[tuple[int, int]]:
    """``(st_dev, st_ino)`` of a skills root directory, or None: the profile may have no skills
    directory yet (the first write creates it at that path), and a filesystem that reports no
    stable identity cannot anchor one."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_dev, st.st_ino) if st.st_dev and st.st_ino else None


def _validate_name(name: str) -> Optional[str]:
    if not name:
        return "Skill name is required."
    if len(name) > MAX_NAME_LENGTH:
        return f"Skill name exceeds {MAX_NAME_LENGTH} characters."
    return None if VALID_NAME_RE.match(name) else (
        f"Invalid skill name {name!r}. {_NAME_RULE} Must start with a letter or digit.")


def _validate_category(category: Optional[str]) -> Optional[str]:
    if category is None or (isinstance(category, str) and not category.strip()):
        return None
    if not isinstance(category, str):
        return "Category must be a string."
    category = category.strip()
    invalid = f"Invalid category {category!r}. {_NAME_RULE} Categories must be a single directory name."
    if "/" in category or "\\" in category:
        return invalid
    if len(category) > MAX_NAME_LENGTH:
        return f"Category exceeds {MAX_NAME_LENGTH} characters."
    return None if VALID_NAME_RE.match(category) else invalid


def _validate_file_path(file_path: str) -> Optional[str]:
    """A supporting-file path: relative, under one support directory, no escape. ``SKILL.md``
    itself is accepted at the skill root, as today's write_file does."""
    if not file_path:
        return "file_path is required."
    candidate = PurePosixPath(file_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        return "Path traversal is not allowed; the file must be a relative path within the skill."
    parts = candidate.parts
    if parts and parts[-1] == "SKILL.md" and len(parts) in (1, 2):
        return None
    if not parts or parts[0] not in ALLOWED_SUBDIRS:
        allowed = ", ".join(sorted(ALLOWED_SUBDIRS))
        return f"File must be under one of: {allowed}. Got: {file_path!r}"
    if len(parts) < 2:
        return f"Provide a file path, not just a directory. Example: '{parts[0]}/myfile.md'"
    return None


class LocalSkillIndex(SkillIndex):
    """Today's skills directory, pinned to its own profile."""

    #: A local index stores the agent level only; writes naming another level are refused.
    levels = frozenset({AGENT_LEVEL})

    def __init__(self, skills_dir) -> None:
        from hermes_constants import hermes_home_key
        self.skills_dir, self.home = index_identity(skills_dir)
        self._home_key = hermes_home_key(self.home)
        # The directory this handle serves, as it resolved at open: its real path, and the
        # directory itself when one was already there (st_dev/st_ino — a directory replaced
        # in place is a different one). None: the profile has no skills directory yet; the
        # first write creates it at that path, never elsewhere.
        self._root_realpath = os.path.realpath(str(self.skills_dir))
        self._root_stat = root_stat(self.skills_dir)

    # --- the pin and the level

    def _check_root(self) -> None:
        """Refuse (``StoragePinRefused``) before anything is read or written when the skills
        directory no longer resolves to the one this index opened: its real path changed (a
        root renamed away and replaced by a link names another profile's directory — the
        refusal a fresh open of that root would get), or, for a root that existed at open, it
        is another directory now (st_dev/st_ino; a root replaced in place is not the one it
        opened). A root that did not exist at open is the path itself: still absent, or created
        there by a later write, it stays this index's root; resolving anywhere else it does
        not. The check and the operation's I/O are two moments: a link swapped in between them
        is outside the local threat model (the machine's owner rearranging their own files
        under a running process), as with the local state store."""
        if os.path.realpath(str(self.skills_dir)) != self._root_realpath or (
                self._root_stat is not None and root_stat(self.skills_dir) != self._root_stat):
            raise StoragePinRefused(
                f"skills directory {str(self.skills_dir)!r} no longer resolves to the one "
                "this index opened")

    def _pin(self) -> None:
        self._check_root()
        from hermes_constants import get_hermes_home, hermes_home_key
        scope = current_scope()
        if scope is not None and scope.tenant != LOCAL_TENANT:
            raise StoragePinRefused(f"a local skills index serves tenant {LOCAL_TENANT!r} only")
        agent = scope.agent if scope is not None else get_hermes_home()
        # Both sides canonical and absolute: a relative agent is the directory it names now.
        if hermes_home_key(agent) != self._home_key:
            raise StoragePinRefused("the bound agent is not this skills index's profile")

    def _store_level(self, level: str) -> None:
        """Refuse a level this backend does not store, before anything is read or written."""
        if level not in SKILL_LEVELS:
            raise SkillsLevelRefused(f"a skills index stores levels {sorted(SKILL_LEVELS)} only")
        if level not in self.levels:
            raise SkillsLevelRefused(
                f"a local skills index stores the agent level only; {level!r} skills are refused")

    # --- reads

    def _scan(self) -> list[dict[str, Any]]:
        """Today's scan of today's directory: every ``SKILL.md`` under the profile skills dir,
        parsed and resolved (status / load_name) by ``agent.skill_utils.resolve_skill_catalog``
        over the one tier this backend holds."""
        from agent.skill_utils import iter_skill_index_files, parse_frontmatter, resolve_skill_catalog
        scanned: list[dict[str, Any]] = []
        for skill_md in iter_skill_index_files(self.skills_dir, "SKILL.md"):
            try:
                text = skill_md.read_text(encoding="utf-8-sig", errors="replace")[:4000]
                frontmatter, body = parse_frontmatter(text)
            except OSError:
                continue
            description = str(frontmatter.get("description", ""))
            if not description:  # first non-heading body line, as today's catalog scan takes it
                description = next((ln for ln in map(str.strip, body.strip().split("\n"))
                                    if ln and not ln.startswith("#")), "")
            if len(description) > MAX_DESCRIPTION_LENGTH:
                description = description[:MAX_DESCRIPTION_LENGTH - 3] + "..."
            relative = skill_md.parent.relative_to(self.skills_dir)
            scanned.append({
                "name": str(frontmatter.get("name") or skill_md.parent.name)[:MAX_NAME_LENGTH],
                "description": description,
                "category": relative.parts[0] if len(relative.parts) > 1 else None,
                "tier": 0, "root": self.skills_dir, "path": skill_md,
            })
        return resolve_skill_catalog(scanned)

    def _row(self, entry: dict[str, Any]) -> dict[str, Any]:
        return {"name": entry["name"], "description": entry["description"],
                "category": entry["category"], "status": entry["status"],
                "load_name": entry["load_name"], "level": AGENT_LEVEL}

    def _locate(self, name: str) -> Optional[dict[str, Any]]:
        """The entry the index resolves *name* to, by the same aliases today's resolver accepts
        (declared name, directory name, path relative to the skills root), selected by today's
        candidate resolution: identical copies of one skill under one root resolve to the
        shallowest, and a name two different skills share is refused with today's ambiguity
        error, never guessed. ``None`` for a name the agent does not offer."""
        from agent.skill_utils import (
            AMBIGUOUS_SKILL_PREFIX, pick_skill_candidate, skill_candidate_rank,
        )
        matches = [entry for entry in self._scan()
                   if name in {str(entry["name"]), Path(entry["path"]).parent.name,
                               entry["relative_path"]}]
        if len(matches) > 1:
            won, _contenders = pick_skill_candidate([
                (entry["tier"], str(entry["root"]), skill_candidate_rank(entry["path"], entry["root"]),
                 entry["path"]) for entry in matches])
            if won is None:
                load_names = sorted({entry["relative_path"] for entry in matches} - {name})
                raise ValueError(
                    f"{AMBIGUOUS_SKILL_PREFIX}'{name}': "
                    + (f"use one of {', '.join(load_names)}. " if load_names else "")
                    + f"{len(matches)} different skills share this name in the same skills "
                    "directory tier; refusing to guess.")
            matches = [matches[won]]
        return matches[0] if matches else None

    def list(self) -> list[dict[str, Any]]:
        self._pin()
        rows = [self._row(entry) for entry in self._scan() if entry["load_name"]]
        return sorted(rows, key=lambda row: (row["category"] or "", row["name"]))

    def get(self, name: str) -> Optional[dict[str, Any]]:
        self._pin()
        entry = self._locate(name)
        return None if entry is None else self._row(entry)

    def read_body(self, name: str, file_path: Optional[str] = None) -> Optional[str]:
        self._pin()
        entry = self._locate(name)
        if entry is None:
            return None
        skill_dir = Path(entry["path"]).parent
        target = skill_dir / "SKILL.md"
        if file_path is not None:
            if _validate_file_path(file_path):
                return None
            target = skill_dir / PurePosixPath(file_path)
        try:
            return target.read_text(encoding="utf-8-sig", errors="replace")
        except OSError:
            return None

    # --- writes

    def put(self, name: str, content: str, *, category: Optional[str] = None,
            level: str = AGENT_LEVEL) -> dict[str, Any]:
        self._pin()
        self._store_level(level)
        # The same identifiers every read accepts, first: an exact path is how a same-name
        # skill is addressed, and the ambiguity refusal a bare shared name gets is the one
        # today's resolver gives. A bare name is validated only where it is used, to create
        # a skill that does not exist yet.
        existing = self._locate(name)
        if existing is not None:  # a replace lands where the skill lives today
            skill_dir = Path(existing["path"]).parent
        else:
            for error in (_validate_name(name), _validate_category(category)):
                if error:
                    raise ValueError(error)
            skill_dir = self.skills_dir / (category or "") / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
        written = next((entry for entry in self._scan()
                        if Path(entry["path"]).parent == skill_dir), None)
        assert written is not None, "a skill just written must scan"
        return self._row(written)

    def put_file(self, name: str, file_path: str, content: str, *, level: str = AGENT_LEVEL) -> None:
        self._pin()
        self._store_level(level)
        if error := _validate_file_path(file_path):
            raise ValueError(error)
        entry = self._locate(name)
        if entry is None:
            raise LookupError(f"skill {name!r} not found; create it first")
        target = Path(entry["path"]).parent / PurePosixPath(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def delete(self, name: str, *, level: str = AGENT_LEVEL) -> bool:
        self._pin()
        self._store_level(level)
        entry = self._locate(name)
        if entry is None:
            return False
        skill_dir = Path(entry["path"]).parent
        shutil.rmtree(skill_dir)
        self._rmdir_if_empty(skill_dir.parent, stop=self.skills_dir)
        return True

    def delete_file(self, name: str, file_path: str, *, level: str = AGENT_LEVEL) -> bool:
        self._pin()
        self._store_level(level)
        if error := _validate_file_path(file_path):
            raise ValueError(error)
        entry = self._locate(name)
        if entry is None:
            return False
        skill_dir = Path(entry["path"]).parent
        target = skill_dir / PurePosixPath(file_path)
        try:
            target.unlink()
        except OSError:
            return False
        self._rmdir_if_empty(target.parent, stop=skill_dir)
        return True

    @staticmethod
    def _rmdir_if_empty(parent: Path, stop: Optional[Path] = None) -> None:
        """An emptied category directory goes; the skills root never does (today's rule)."""
        if parent != stop and parent != parent.parent:
            try:
                if parent.exists() and not any(parent.iterdir()):
                    parent.rmdir()
            except OSError:
                pass


class LocalSkillsIndexProvider(SkillsIndexProvider):
    """Today's skills: one ``skills/`` directory per profile home, holding that agent's records
    only."""

    levels = frozenset({AGENT_LEVEL})

    @property
    def name(self) -> str:
        return "local"

    def open(self, skills_dir: Any = None) -> SkillsIndexing:
        """The index over *skills_dir*, or over the bound profile's own ``skills/`` directory
        (``get_skills_dir()``). The index's directory and profile are frozen here
        (``index_identity``); the pin is checked by every operation."""
        from hermes_constants import get_skills_dir
        return SkillsIndexing(LocalSkillIndex(skills_dir if skills_dir is not None else get_skills_dir()))

    def assert_isolating(self) -> None:
        """A local index is one directory per profile and refuses every pin but its own; there
        is no shared role that could bypass that."""
        return
