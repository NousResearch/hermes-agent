# Rollback guide

Back up the target database before a release rollback. Use a copy for rehearsal
and never overwrite the only user database.

## Schema

`communication_core.schema.rollback(path, target_version)` applies ordered down
migrations in one transaction per version. Tests cover rollback to zero and a
failing migration script with no partial state. Restore the backup if a newer
consumer wrote data that the down migration cannot represent.

## Facebook migration

Run `hermes communication migration facebook-rollback <run-id>`. Rollback is
scoped by source system, account, and source hash; it deletes only canonical
rows/mappings/legacy archives created by that run and marks it rolled back. The
legacy source was opened read-only and remains unchanged.

## Adapter or CLI/skill cutover

Disable the affected connected account. The legacy files remain on disk for
forensics/rollback, but do not re-enable a sender. Revert the consumer to the
old verified read-only command only after comparing IDs/counts. The Core
database can remain untouched while an adapter is disabled.

## Merge/unmerge

Use `people unmerge <merge-audit-id>`. Identity, event, commitment, group,
journey, route, and preference ownership is restored from the audit snapshot;
raw messages/conversations retain their stable IDs throughout.

## Telegram News shared abstractions

News owns its database and can roll back its intelligence schema independently.
Communication Core stores only public-reference suggestions/drafts. Remove a
suggestion/draft locally if needed; never copy private content into News as a
rollback shortcut.
