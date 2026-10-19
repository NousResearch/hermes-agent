"""Hand-written profile deletion tombstones, for state the production writer never produces.

``hermes_cli.profile_lifecycle.mark_profile_deleting`` is the one writer, and tests that only need
"a deleted profile" call it. This writes the file directly at the lexical
``profiles/.deleted/<name>`` path instead: the tokenless ``deleted\\n`` fence older releases left
(``read_profile_deletion_incarnation`` still accepts it), or a tombstone under a fixture's
profiles root that is not yet recognisable as a Hermes home, where the writer refuses.
"""

from pathlib import Path

from hermes_constants import profile_tombstone_path


def write_legacy_profile_tombstone(profile_home: Path, content: str = "deleted\n") -> Path:
    marker = profile_tombstone_path(Path(profile_home))
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(content, encoding="utf-8")
    return marker
