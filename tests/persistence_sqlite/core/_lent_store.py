"""A store handle served through a store holder the test already holds."""

from persistence import ProfileScope
from persistence.sqlite import SQLiteBackend


def lent_store(holder):
    """The SQLite store handle on *holder*'s file, served through *holder* alone; the caller keeps its reference."""
    scope = ProfileScope(profile_dir=holder.db_path.parent, name=holder.db_path.parent.name)
    return SQLiteBackend.lending(scope.hermes_home_key, holder).open(scope)
