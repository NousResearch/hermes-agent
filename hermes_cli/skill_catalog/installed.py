"""Path-based installed-row extension for live, authorized catalogs.

No registry entries means no provider calls and no extra wire fields. Metadata
comes from bounded local installed reads, not a remote inspect/search per skill.
"""
import time
from pathlib import Path

from pm.filesystem import is_junction

from hermes_constants import get_hermes_home
from hermes_cli.skill_catalog.registry import snapshot
from hermes_cli.skill_catalog.validation import CatalogError
from hermes_cli.skill_catalog.worker import call


def target_path(home: Path, relative: str) -> Path:
    """Resolve one validated target without following redirected ancestors."""
    from hermes_cli.skill_catalog.validation import _path
    try:
        _path(relative)
        root = home / "skills"
        if root.is_symlink():
            raise ValueError("Redirected skills root")
        current = root
        for part in relative.split("/"):
            if part.startswith("."):
                raise ValueError("Reserved target")
            current = current / part
            if current.is_symlink() or is_junction(current):
                raise ValueError("Redirected target")
        if not current.resolve().is_relative_to(root.resolve()):
            raise ValueError("Target outside skills root")
        return current
    except (ValueError, OSError):
        raise CatalogError("ownership_conflict", "Catalog target is not a local skill path.") from None


def installed_records(entry, *, deadline=None):
    """Read a complete bounded local record set; partial pages have no authority."""
    deadline = min(deadline, time.monotonic() + 3) if deadline else time.monotonic() + 3
    cursor = None
    cursors = set()
    records = []
    for _ in range(100):
        page = call(entry, "installed", {"limit": 100, "cursor": cursor}, deadline=deadline)
        records.extend(page["items"])
        cursor = page["next_cursor"]
        if cursor is None:
            return records
        if cursor in cursors:
            break
        cursors.add(cursor)
    raise CatalogError("invalid_response", "Catalog installed pagination is incomplete.")


def extend_installed_rows(rows):
    """Add D4 flat fields by actual path while retaining stock row names/order."""
    home = get_hermes_home()
    from hermes_cli.skill_catalog.projection import Projection
    projection = Projection(home).read()
    entries = snapshot()
    if not entries:
        if projection['registrations']:
            raise CatalogError('ownership_state_unavailable', 'Catalog owner is in another serving process.')
        return rows
    from tools.skills_tool import _skill_catalog
    paths = {}
    for skill in _skill_catalog(skip_disabled=True):
        path = skill["path"].parent
        try:
            relative = path.relative_to(home / "skills").as_posix()
            if target_path(home, relative) == path:
                paths[skill["load_name"]] = relative
        except (ValueError, CatalogError):
            continue
    claims = {}
    conflicts = set()
    for entry in entries:
        if not entry.authorized():
            continue
        try:
            records = installed_records(entry)
        except CatalogError:
            raise CatalogError('ownership_state_unavailable', 'Catalog installed records are unavailable.') from None
        owner = projection['registrations'].get(entry.registration_id)
        if owner is None or owner['state'] != 'ready':
            raise CatalogError('ownership_state_unavailable', 'Catalog ownership is not ready.')
        for record in records:
            relative = record["target_relative_path"]
            try:
                target_path(home, relative)
            except CatalogError:
                continue
            if record["managed_state"] not in ("installed", "pending"):
                continue
            if relative in claims:
                conflicts.add(relative)
            claims[relative] = (entry, record)
    for row in rows:
        relative = paths.get(row["name"])
        if relative not in claims or relative in conflicts:
            continue
        entry, record = claims[relative]
        if entry.stopped or not entry.authorized():
            continue
        row.update({key: value for key, value in record.items() if key != "name"})
        settlement = 'reconciling' if any(
            reservation['registration_id'] == entry.registration_id
            and reservation['target_relative_path'] == relative
            for reservation in projection['reservations'].values()) else 'settled'
        row.update(provenance="provider", settlement=settlement, provider={
            "id": entry.descriptor.name, "plugin_id": entry.plugin_key, "label": entry.descriptor.label,
            "version": record["installed_version"], "author": None,
            "managed_state": record["managed_state"], "scan_verdict": "unknown", "actions": record["actions"]})
    return rows
