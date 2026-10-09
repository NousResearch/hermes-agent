"""Frozen pre-PM updater compatibility for process-identity imports.

Runtime ownership lives in :mod:`runtime.process_identity`. Current application code must import
that canonical owner directly. This module exists only because an already-running historical
`hermes update` can lazy-import these names after replacing its checkout.
"""

from runtime.process_identity import (
    REAPABLE_PURPOSES as REAPABLE_PURPOSES,
    ledger_entries as ledger_entries,
    spawner_is_dead as spawner_is_dead,
)

__all__ = ["REAPABLE_PURPOSES", "ledger_entries", "spawner_is_dead"]
