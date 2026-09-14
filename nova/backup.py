"""Backing up and restoring the state NOVA owns.

NOVA writes profiles, a knowledge index, an audit log and a work board, and until now
offered no way to preserve or move any of it. ``backups`` appeared in the codebase only on
``NEVER_WRITE``.

The design question that decides everything here is whether a backup carries credentials.

**It does not, by default.** ``<profile>/.env`` is the one file NOVA is forbidden to write,
and a backup that swept it up would turn every archive into a secret-bearing artifact —
copied to laptops, attached to tickets, kept in object storage with whatever policy that
bucket happens to have. The whole deployment seam exists so NOVA never holds a credential;
an archive that quietly did would undo it.

So a backup records **which variables each agent needs** and not their values, and restore
reports what must be re-provisioned. That is the same trade ``nova doctor`` makes: NOVA
cannot supply a secret, so it says precisely what is missing and where it goes.

``--include-secrets`` exists because a disaster-recovery copy that cannot be restored
without a separate secret store is a real operational burden, and pretending otherwise
would just push people to ``tar`` the directory themselves and lose the manifest as well.
It warns loudly, records the choice in the manifest, and writes the archive ``0600``.

**What is not backed up.** The work board (``kanban.db``) is the runtime's, shared with its
own CLI and dashboard; NOVA does not own it and will not restore over it. Conversation
history, memories and session state are the customer's and are on ``NEVER_WRITE``.
"""

from __future__ import annotations

import json
import os
import tarfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from nova._env import read_env_file

MANIFEST_NAME = "nova-backup.json"
BACKUP_VERSION = 1

#: Directories and files inside a NOVA home that NOVA owns and can restore.
OWNED = ("profiles", "skins", "nova", "nova-knowledge.db", "control-principals.yaml")

#: Never included unless the operator explicitly asks. These are credentials.
SECRET_NAMES = (".env", ".op.env")

#: Never included, regardless of runtime: these names mean the same thing everywhere.
#: Anything else a particular runtime owns comes from :meth:`AgentRuntime.never_archive`,
#: because which files belong to a runtime is the adapter's knowledge — the same reason
#: ``NEVER_WRITE`` lives beside the materializer rather than here.
EXCLUDED = (
    "memories", "sessions", "logs", "workspace", "checkpoints", "local", "backups",
)


@dataclass
class BackupManifest:
    """What an archive contains, and what it deliberately does not."""

    version: int = BACKUP_VERSION
    tenant_id: str = ""
    created_at: str = ""
    nova_version: str = ""
    includes_secrets: bool = False
    files: int = 0
    agents: tuple[str, ...] = ()
    #: Variable NAMES each agent needs. Never values — that is the point of the file.
    required_env: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "tenant_id": self.tenant_id,
            "created_at": self.created_at,
            "nova_version": self.nova_version,
            "includes_secrets": self.includes_secrets,
            "files": self.files,
            "agents": list(self.agents),
            "required_env": {k: list(v) for k, v in self.required_env.items()},
            "_comment": (
                "A NOVA backup. Credentials are NOT included unless includes_secrets is "
                "true; required_env names the variables each agent needs so they can be "
                "re-provisioned in <profile>/.env after restore."
            ),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "BackupManifest":
        if not isinstance(data, dict):
            raise ValueError("a backup manifest must be a JSON object")
        return cls(
            version=int(data.get("version", 0)),
            tenant_id=str(data.get("tenant_id", "")),
            created_at=str(data.get("created_at", "")),
            nova_version=str(data.get("nova_version", "")),
            includes_secrets=bool(data.get("includes_secrets", False)),
            files=int(data.get("files", 0)),
            agents=tuple(data.get("agents") or ()),
            required_env={
                str(k): [str(x) for x in v]
                for k, v in (data.get("required_env") or {}).items()
            },
        )


def _members(
    home: Path, include_secrets: bool, never: tuple[str, ...] = ()
) -> list[Path]:
    """Every file NOVA owns inside ``home``, in a stable order."""
    excluded = set(EXCLUDED) | set(never)
    found: list[Path] = []
    for name in OWNED:
        target = home / name
        if not target.exists():
            continue
        if target.is_file():
            found.append(target)
            continue
        for path in sorted(target.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            if any(part in excluded for part in path.relative_to(home).parts):
                continue
            if path.name in SECRET_NAMES and not include_secrets:
                continue
            found.append(path)
    return found


def create(
    home: Path,
    destination: Path,
    *,
    tenant_id: str = "",
    agents: Iterable[str] = (),
    required_env: Optional[dict[str, list[str]]] = None,
    include_secrets: bool = False,
    never_archive: tuple[str, ...] = (),
) -> BackupManifest:
    """Write a ``.tar.gz`` of the state NOVA owns. Returns what went into it.

    ``never_archive`` is the adapter's list of files that belong to its runtime rather than
    to NOVA — asked for rather than assumed, so a second adapter's state is excluded by its
    own say-so instead of by a list here that would silently not mention it.
    """
    from nova import __version__

    members = _members(home, include_secrets, never_archive)
    manifest = BackupManifest(
        tenant_id=tenant_id,
        created_at=datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        nova_version=__version__,
        includes_secrets=include_secrets,
        files=len(members),
        agents=tuple(agents),
        required_env=dict(required_env or {}),
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    # 0600 before a byte is written. An archive that briefly exists world-readable is
    # world-readable, and this one may contain credentials.
    handle = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(handle, "wb") as raw, tarfile.open(fileobj=raw, mode="w:gz") as archive:
            payload = json.dumps(manifest.to_dict(), indent=2, sort_keys=True).encode("utf-8")
            info = tarfile.TarInfo(MANIFEST_NAME)
            info.size = len(payload)
            info.mode = 0o600
            archive.addfile(info, __import__("io").BytesIO(payload))
            for path in members:
                archive.add(path, arcname=str(Path("home") / path.relative_to(home)))
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return manifest


def read_manifest(archive_path: Path) -> BackupManifest:
    """The manifest, without extracting anything else."""
    with tarfile.open(archive_path, mode="r:gz") as archive:
        member = archive.extractfile(MANIFEST_NAME)
        if member is None:
            raise ValueError(f"{archive_path} has no {MANIFEST_NAME}; not a NOVA backup")
        return BackupManifest.from_dict(json.loads(member.read().decode("utf-8")))


@dataclass
class RestoreReport:
    """What a restore did, and what the operator must still do."""

    manifest: BackupManifest
    restored: int = 0
    skipped: tuple[str, ...] = ()
    missing_env: dict[str, list[str]] = field(default_factory=dict)
    dry_run: bool = False

    def summary(self) -> str:
        verb = "would restore" if self.dry_run else "restored"
        line = f"{verb} {self.restored} file(s) from a {self.manifest.created_at} backup"
        if self.missing_env:
            agents = len(self.missing_env)
            line += f"; {agents} agent(s) still need credentials"
        return line

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest": self.manifest.to_dict(),
            "restored": self.restored,
            "skipped": list(self.skipped),
            "missing_env": {k: list(v) for k, v in self.missing_env.items()},
            "dry_run": self.dry_run,
        }


def _safe_members(archive: tarfile.TarFile, home: Path) -> list[tarfile.TarInfo]:
    """Members that land inside ``home``, refusing anything that escapes it.

    An archive is untrusted input even when NOVA wrote it — it may have been edited, or
    swapped, between backup and restore. A member named ``../../etc/cron.d/x`` is the
    oldest trick there is, and ``tarfile`` will happily follow it.
    """
    root = (home / "home").resolve()
    safe: list[tarfile.TarInfo] = []
    for member in archive.getmembers():
        if member.name == MANIFEST_NAME:
            continue
        if member.issym() or member.islnk():
            # A link in a restore archive has no legitimate use and every illegitimate one.
            continue
        if not member.isfile():
            continue
        target = (home / Path(member.name).relative_to("home")).resolve() if \
            member.name.startswith("home/") else None
        if target is None:
            continue
        try:
            target.relative_to(home.resolve())
        except ValueError:
            continue
        safe.append(member)
    return safe


def restore(
    archive_path: Path,
    home: Path,
    *,
    dry_run: bool = False,
    environ: Optional[dict] = None,
) -> RestoreReport:
    """Extract a backup into ``home``, then report what credentials are still missing."""
    manifest = read_manifest(archive_path)
    if manifest.version > BACKUP_VERSION:
        raise ValueError(
            f"{archive_path} was written by a newer NOVA (backup version "
            f"{manifest.version}; this one understands {BACKUP_VERSION})"
        )

    home.mkdir(parents=True, exist_ok=True)
    restored = 0
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = _safe_members(archive, home)
        skipped = tuple(
            member.name
            for member in archive.getmembers()
            if member.name != MANIFEST_NAME and member not in members
        )
        if not dry_run:
            for member in members:
                destination = home / Path(member.name).relative_to("home")
                destination.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    continue
                data = source.read()
                # Credentials keep 0600 on the way back in, whatever the archive said.
                mode = 0o600 if destination.name in SECRET_NAMES else 0o644
                fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
                try:
                    os.write(fd, data)
                finally:
                    os.close(fd)
                restored += 1
        else:
            restored = len(members)

    environ = os.environ if environ is None else environ
    missing: dict[str, list[str]] = {}
    for agent, names in manifest.required_env.items():
        present = read_env_file(home / "profiles" / agent / ".env")
        absent = [n for n in names if n not in present and not environ.get(n)]
        if absent:
            missing[agent] = absent

    return RestoreReport(
        manifest=manifest,
        restored=restored,
        skipped=skipped,
        missing_env=missing,
        dry_run=dry_run,
    )
