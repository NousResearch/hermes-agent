"""``hermes import-agent --sync`` — keep previously imported Claude Code / Codex setups current.

Every successful ``hermes import-agent`` run records its source in ``HERMES_HOME/import-sync.json``
(the sync manifest); ``--sync`` re-imports every registered source whose files changed since the
last run. Change detection is a content digest over exactly the files the importer reads, so an
unchanged source is a cheap no-op and credential files (never read by the importer) can never
trigger a sync. Sync runs prompt-free — it only re-applies sources the user already imported once.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from utils import atomic_write_text

logger = logging.getLogger(__name__)

SYNC_MANIFEST_NAME = "import-sync.json"
# Credential stores that live inside the source trees; never part of the digest.
_CREDENTIAL_FILENAMES = frozenset({".credentials.json", "auth.json", "credentials.json"})


def sync_manifest_path(target_root: Path) -> Path:
    return Path(target_root) / SYNC_MANIFEST_NAME


def _iter_sync_files(agent: str, source_root: Path) -> Iterator[Path]:
    """Yield the source files whose content determines the sync digest — exactly what
    :class:`~hermes_cli.agent_import.AgentImporter` reads for ``agent``."""
    if agent == "claude-code":
        # ~/.claude.json lives NEXT TO ~/.claude/ (see AgentImporter._run_claude_code)
        paths = [source_root / "CLAUDE.md", source_root / "settings.json",
                 source_root.parent / ".claude.json"]
    else:
        paths = [source_root / "AGENTS.md", source_root / "config.toml"]
        paths.extend(sorted((source_root / "memories").glob("*.md")))
    paths.extend(p for p in sorted((source_root / "skills").rglob("*")) if p.is_file())
    for path in paths:
        if path.name not in _CREDENTIAL_FILENAMES and path.is_file():
            yield path


def compute_source_digest(agent: str, source_root: Path) -> str:
    """Content digest of everything the importer would read from ``source_root``."""
    source_root = Path(source_root)
    digest = hashlib.sha256()
    for path in _iter_sync_files(agent, source_root):
        try:
            rel = str(path.relative_to(source_root))
        except ValueError:
            rel = f"../{path.name}"
        try:
            content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            content_hash = "unreadable"
        digest.update(f"{rel}\0{content_hash}\0".encode("utf-8", "replace"))
    return digest.hexdigest()


def skill_tree_digest(skill_dir: Path) -> str:
    """Content digest of an installed skill directory (relative path + bytes of every file)."""
    digest = hashlib.sha256()
    for path in sorted(p for p in Path(skill_dir).rglob("*") if p.is_file()):
        try:
            digest.update(f"{path.relative_to(skill_dir)}\0{hashlib.sha256(path.read_bytes()).hexdigest()}\0".encode("utf-8", "replace"))
        except OSError:
            digest.update(b"unreadable\0")
    return digest.hexdigest()


def managed_skills(entry: Any) -> Dict[str, Optional[str]]:
    """``imported_skills`` as ``{name: digest-at-import}`` (a pre-digest list maps to ``None`` = trusted)."""
    skills = entry.get("imported_skills") if isinstance(entry, dict) else None
    if isinstance(skills, dict):
        return dict(skills)
    return {name: None for name in skills} if isinstance(skills, list) else {}


def load_sync_manifest(target_root: Path) -> Dict[str, Any]:
    """Read the manifest; a missing, unreadable or malformed file yields an empty manifest."""
    path = sync_manifest_path(target_root)
    if not path.exists():
        return {"version": 1, "agents": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Unreadable %s — starting a fresh sync manifest", path)
        return {"version": 1, "agents": {}}
    if not isinstance(data, dict) or not isinstance(data.get("agents"), dict):
        return {"version": 1, "agents": {}}
    return data


def save_sync_manifest(target_root: Path, manifest: Dict[str, Any]) -> None:
    atomic_write_text(sync_manifest_path(target_root),
                      json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def update_sync_manifest(agent: str, source_root: Path, target_root: Path,
                         overwrite: bool, report: Dict[str, Any], *, refresh_digest: bool = True) -> None:
    """Record/refresh the sync entry for ``agent`` after a real (non-dry-run) import.

    ``imported_skills`` maps every skill this command ever copied for the agent to the digest of
    the copy it wrote: a later sync replaces the destination only while it still matches that
    digest, so a skill the user edited (or created) under the import category is never clobbered.
    ``refresh_digest=False`` keeps the previous source digest so a run with errors is retried.
    """
    manifest = load_sync_manifest(target_root)
    agents = manifest.setdefault("agents", {})
    entry = agents.get(agent)
    if not isinstance(entry, dict):
        entry = {}
    skills = managed_skills(entry)
    for item in report.get("items", []):
        if item.get("kind") == "skill" and item.get("status") == "imported" and item.get("destination"):
            skills[Path(item["destination"]).name] = skill_tree_digest(Path(item["destination"]))
    entry.update({"source": str(source_root), "overwrite": bool(overwrite),
                  "last_import": int(time.time()), "imported_skills": dict(sorted(skills.items()))})
    if refresh_digest:
        entry["digest"] = compute_source_digest(agent, Path(source_root))
    agents[agent] = entry
    save_sync_manifest(target_root, manifest)


def sync_imported_agents(args) -> None:
    """Handle ``hermes import-agent --sync``: re-import every registered source whose digest
    changed. Prompt-free (cron-friendly); ``--dry-run`` previews without touching the manifest."""
    from hermes_cli.agent_import import AgentImporter, print_import_report
    from hermes_cli.setup import print_error, print_header, print_info, print_success
    from hermes_constants import get_hermes_home

    dry_run = bool(getattr(args, "dry_run", False))
    hermes_home = get_hermes_home().resolve()
    agents: Dict[str, Any] = load_sync_manifest(hermes_home).get("agents", {})
    if not agents:
        print()
        print_info("No import sources registered yet.")
        print_info("Run 'hermes import-agent' first — successful imports are "
                   "registered for sync automatically.")
        return

    print()
    print_header("Import Sync")
    synced = unchanged = failed = 0
    for agent_name in sorted(agents):
        entry = agents[agent_name]
        if not isinstance(entry, dict) or not entry.get("source"):
            continue
        source_dir = Path(entry["source"])
        if not source_dir.is_dir():
            print_info(f"{agent_name}: source {source_dir} no longer exists — skipped")
            continue
        if compute_source_digest(agent_name, source_dir) == entry.get("digest"):
            print_info(f"{agent_name}: unchanged since last import")
            unchanged += 1
            continue
        print_info(f"{agent_name}: changes detected in {source_dir}"
                   + (" (dry run)" if dry_run else " — re-importing"))
        overwrite = bool(entry.get("overwrite", False))
        try:
            report = AgentImporter(agent_name, source_dir, hermes_home, execute=not dry_run,
                                   overwrite=overwrite, sync_skills=managed_skills(entry)).run()
        except Exception as exc:  # noqa: BLE001 — keep syncing the other sources
            print_error(f"{agent_name}: sync failed: {exc}")
            logger.debug("import-agent sync error", exc_info=True)
            failed += 1
            continue
        print_import_report(report, dry_run=dry_run)
        had_errors = bool(report.get("summary", {}).get("error"))
        if not dry_run:
            try:
                # Errors keep the old source digest so the next sync retries the failed items.
                update_sync_manifest(agent_name, source_dir, hermes_home, overwrite, report,
                                     refresh_digest=not had_errors)
            except OSError as exc:
                logger.warning("Could not update import sync manifest: %s", exc)
                had_errors = True
        if had_errors:
            failed += 1
        else:
            synced += 1

    print()
    if synced == 0 and failed == 0:
        print_success(f"All {unchanged} source(s) already up to date.")
        return
    parts = [f"{synced} synced"] + [f"{n} {label}" for n, label in
                                    ((unchanged, "unchanged"), (failed, "failed")) if n]
    message = "Sync complete: " + ", ".join(parts) + "."
    if failed:
        print_error(message)
        raise SystemExit(1)
    print_success(message)
