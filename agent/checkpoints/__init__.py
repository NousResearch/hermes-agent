"""Snapshots of the working directory that ``/rollback``, ``/undo`` and ``hermes checkpoints`` use.

- ``manager``: ``CheckpointManager``; takes, lists, diffs and restores snapshots in the shared store.
- ``maintenance``: store retention, orphan pruning, status and cleanup.
- ``pruning``: the store lock and checked history rewrites shared by snapshot limits and store maintenance.
- ``refs``: reads a ``/rollback`` argument as a list number or a checkpoint hash.
- ``undo_hint``: ``/undo``'s line naming the files the undone turns changed, and the rollback rewind rule.
- ``manager_profile_rename``: rekeys a profile's checkpoint projects after the profile directory moves.
"""
