# Snapshot Structural Reference

## Two Snapshot Formats

The snapshot creation process changed format over time. Both exist in the wild.

### Full Format (9 files) — Pre-May 25 2026

Complete environment backup, ~14-15 GB each (dominated by state.db):

| File | Size | Description |
|------|------|-------------|
| state.db | 14-15 GB | SQLite session/state database (the bulk) |
| config.yaml | ~20 KB | Hermes agent config |
| auth.json | ~8 KB | Auth tokens |
| channel_directory.json | ~12 KB | Messaging channel config |
| cron/ | ~150 KB | Cron job definitions (directory) |
| gateway_state.json | ~4 KB | Gateway runtime state |
| manifest.json | ~4 KB | Manifest |
| processes.json | ~4 KB | Process state |
| .env | KB-scale | Environment variables |

### Minimal Format (4-5 files) — May 25 2026+

State-only backup, MB to sub-GB scale:

| File | Size | Description |
|------|------|-------------|
| state.db | 16 MB - 751 MB | Shrinks as sessions age out |
| state.db-journal | ~4 KB | SQLite journal |
| auth.json | 0-8 KB | Auth tokens |
| config.yaml | 16-20 KB | Config |
| .env | KB-scale | Environment variables |

## Why This Matters

- **Deletion decisions**: Full-format snapshots are 14-15 GB each. Minimal-format are negligible. Prioritize full-format for deletion.
- **Content overlap**: Full-format snapshots contain everything minimal-format does plus operational state (cron, gateway, channels). A newer minimal snapshot + live state.db covers the same recovery needs.
- **Atime is reliable**: Snapshot files are read-only after creation. Access times reflect actual read operations (e.g., listing, stat), not writes.

## Quick Analysis Commands

List snapshots with size and file count:
```
for d in /root/.hermes/state-snapshots/*/; do
  echo "$(basename "$d") | $(du -sh "$d" | cut -f1) | $(find "$d" -type f | wc -l) files"
done
```

Check access times for deletion decisions:
```
for d in /root/.hermes/state-snapshots/*/; do
  name=$(basename "$d")
  dir_atime=$(stat -c '%x' "$d")
  newest=$(find "$d" -type f -printf '%A+' | sort | tail -1)
  echo "$name | dir: $dir_atime | last file access: $newest"
done
```

Inspect structure (9-file vs 5-file):
```
for d in /root/.hermes/state-snapshots/*/; do
  echo "=== $(basename "$d") ==="
  ls -la "$d"
done
```
