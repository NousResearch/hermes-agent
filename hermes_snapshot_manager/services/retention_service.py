from __future__ import annotations

from hermes_snapshot_manager.services.snapshot_service import SnapshotService


class RetentionService:
    def __init__(self, snapshot_service: SnapshotService):
        self.snapshot_service = snapshot_service

    def cleanup(self) -> dict:
        snapshots = self.snapshot_service.list_snapshots()
        keep = self.snapshot_service.settings.retention_daily
        protected = {snapshot.id for snapshot in snapshots[:keep]}
        protected.update(snapshot.id for snapshot in snapshots if snapshot.is_known_good)

        deleted: list[str] = []
        for snapshot in snapshots:
            if snapshot.id in protected:
                continue
            self.snapshot_service.delete_snapshot(snapshot.id)
            deleted.append(snapshot.id)
        return {"deleted": deleted, "retained": len(snapshots) - len(deleted), "protected": sorted(protected)}
