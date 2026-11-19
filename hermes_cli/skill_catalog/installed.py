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


def _unavailable():
    return CatalogError('ownership_state_unavailable', 'Catalog installed authority changed or is unavailable.')


def _authority(doc, entries):
    """Only relevant owner/claim/reservation changes invalidate an installed read."""
    if set(doc['registrations']) != {entry.registration_id for entry in entries}:
        raise _unavailable()
    for entry in entries:
        owner = doc['registrations'].get(entry.registration_id)
        if (entry.stopped or not entry.authorized() or owner is None or owner['state'] != 'ready'
                or owner['source_id'] != entry.source_id or owner['generation'] != entry.generation
                or owner['plugin_key'] != entry.plugin_key
                or owner['implementation_revision'] != entry.descriptor.implementation_revision):
            raise _unavailable()
    return {key: doc[key] for key in ('registrations', 'targets', 'reservations')}


def _records_by_claim(home, entry, records, projection):
    expected = {path: claim for path, claim in projection['targets'].items()
                if claim['registration_id'] == entry.registration_id}
    claims = {}
    for record in records:
        relative = record['target_relative_path']
        target = target_path(home, relative)
        if record['managed_state'] not in ('installed', 'pending') or not target.exists():
            continue
        claim = expected.get(relative)
        if (relative in claims or claim is None or any(
                record[key] != claim[key] for key in ('identifier', 'installation_revision', 'managed_state'))):
            raise _unavailable()
        claims[relative] = (entry, record)
    if claims.keys() != expected.keys():
        raise _unavailable()
    return claims


def _decorate(rows, paths, claims, projection):
    for row in rows:
        relative = paths.get(row['name'])
        if relative not in claims:
            continue
        entry, record = claims[relative]
        row.update({key: value for key, value in record.items() if key != 'name'})
        settlement = 'reconciling' if any(
            reservation['registration_id'] == entry.registration_id
            and reservation['target_relative_path'] == relative
            for reservation in projection['reservations'].values()) else 'settled'
        row.update(provenance='provider', settlement=settlement, provider={
            'id': entry.descriptor.name, 'plugin_id': entry.plugin_key, 'label': entry.descriptor.label,
            'version': record['installed_version'], 'author': None,
            'managed_state': record['managed_state'], 'scan_verdict': 'unknown', 'actions': record['actions']})
    return rows


def extend_installed_rows(rows):
    """Join provider metadata only against unchanged, matching live target claims.

    Provider reads run outside the mutation gate. Final identity/claim validation
    and row decoration share the gate, so a concurrent disable or settlement never
    turns stale provider metadata into current ownership (or editable local rows).
    """
    home = get_hermes_home()
    from hermes_cli.skill_catalog.projection import Projection
    store = Projection(home)
    projection = store.read()
    entries = snapshot()
    before = _authority(projection, entries)
    if not entries:
        return rows
    from tools.skills_tool import _skill_catalog
    paths = {}
    for skill in _skill_catalog(skip_disabled=True):
        path = skill['path'].parent
        try:
            relative = path.relative_to(home / 'skills').as_posix()
            if target_path(home, relative) == path:
                paths[skill['load_name']] = relative
        except (ValueError, CatalogError):
            continue
    claims = {}
    deadline = time.monotonic() + 3
    for entry in entries:
        try:
            records = installed_records(entry, deadline=deadline)
            claims.update(_records_by_claim(home, entry, records, projection))
        except CatalogError:
            raise _unavailable() from None
    with store.transaction() as current:
        if _authority(current, entries) != before:
            raise _unavailable()
        return _decorate(rows, paths, claims, current)
