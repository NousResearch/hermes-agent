# Managed board enrollment for memory stores

`install.py` writes this board-only block to the default memory store and to
existing non-bot profile stores. Replace `<session_coord_path>` with the
absolute installed CLI path. Keep both markers so future upgrades can replace
only the installer-owned text.

```text
<!-- BEGIN session-coord managed board-v2 -->
STANDING RULE — session-coord (board-wire v2): Before mutating shared resources (files, skills, memory, cron store, remote boxes, or desktop UI), first run `python3 '<session_coord_path>' status`. Register once per task with `ID=$(python3 '<session_coord_path>' register --task '...' --surface <surface>)`, then atomically claim every required resource with `python3 '<session_coord_path>' claim --id $ID --res <key> [--res <key2>]`. Continue only after CLAIMED; when HELD/QUEUED, do not mutate the requested resources—use `check`, `inbox`, or one bounded `--wait` shell call and wait for release. Run `python3 '<session_coord_path>' done --id $ID` at task end. Priorities are user-set; never `steal` without explicit approval. Full protocol: skill multi-session-coordination. Off-switch: `python3 '<session_coord_path>' disable`.
<!-- END session-coord managed board-v2 -->
```

This block deliberately does not use `--yield` or promise automatic resume. The
optional native block is written separately by `hermes_setup.py` only after the
selected profile passes real Hermes capability and policy checks.

## Upgrade behavior

- Exact shipped `session-coord (wire v1)` blocks are migrated in place,
  including the newer local-native variant.
- The surrounding memory entries remain unchanged and the previous file is
  copied to `MEMORY.md.bak-<timestamp>`.
- A customized, duplicate, or malformed block is preserved and reported as
  `ACTION NEEDED`; compare it manually rather than deleting user-authored text.
- A current managed block is a no-op on rerun.

Preview with `python3 install.py --check`. Install with `python3 install.py`.
