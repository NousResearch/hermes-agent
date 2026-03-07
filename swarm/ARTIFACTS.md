# Swarm Artifact Store

Manages shared artifacts produced by swarm tasks with provenance tracking.

## Storage Layout

```
base_dir/
  {task_id}/
    {artifact_id}_{name}
```

Each artifact gets a unique ID. Files are copied into the store on `store()` / `store_bytes()`, so the original can be deleted without affecting the store.

## API

| Method | Description |
|---|---|
| `store(task_id, path, mime_type?)` | Copy a file into the store, return `ArtifactRef` |
| `store_bytes(task_id, data, name, mime_type?)` | Store raw bytes, return `ArtifactRef` |
| `get(artifact_id)` | Get filesystem path to a stored artifact |
| `get_by_task(task_id)` | List all artifacts produced by a task |
| `list_all()` | List every artifact in the store |
| `delete(artifact_id)` | Remove a single artifact |
| `cleanup_task(task_id)` | Remove all artifacts for a task |
| `export_manifest()` | JSON-serializable dict of all artifacts with provenance |

## Provenance

Every `ArtifactRef` records: originating `task_id`, SHA-256 `checksum`, `size_bytes`, `mime_type`, and `created_at` timestamp.

## Thread Safety

All index mutations are guarded by a `threading.Lock`. File I/O is scoped per task directory to avoid cross-task contention.
