"""A text marker for the update downgrade guard of v0.22.7..v0.23.1, and the title constants outside plugins import.

Those updaters read the target tree's root ``hermes_state*.py`` for the runtime ledger table below and refuse the
update as a downgrade when no file carries it. The session store lives in ``persistence/sqlite/``; this file keeps
the text so a host on one of those releases can still update.

The Honcho memory plugin (plugin-catalog/honcho.yaml) imports the two title-provenance constants from here at module
top level, so without them it never registers. They are the values stored in ``sessions.title_source``, so they
cannot change without a data migration; the storage layer keeps its own copy and the code outside it may not import it.
"""

TITLE_SOURCE_DERIVED = "derived"
TITLE_SOURCE_LLM = "llm"

__all__ = ["TITLE_SOURCE_DERIVED", "TITLE_SOURCE_LLM"]

# CREATE TABLE IF NOT EXISTS session_admissions
