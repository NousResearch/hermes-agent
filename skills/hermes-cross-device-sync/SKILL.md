---
name: hermes-cross-device-sync
description: Sync Hermes Agent data (memories, skills, sessions) across devices via Syncthing — covers Termux↔WSL setup, Android storage workarounds, and common pitfalls.
version: 1.0.0
author: jasonchang2025
license: MIT
metadata:
  hermes:
    tags: [syncthing, cross-device, sync, termux, android, wsl, devops]
    category: devops
    requires_toolsets: [terminal, cronjob]
---

# Syncthing Hermes Sync

Sync Hermes Agent's `~/.hermes/` data directories across devices using Syncthing. Covers the common multi-device scenario: Android Termux ↔ Windows (WSL) ↔ Linux.

## Architecture

### Default (Syncthing inside WSL)

```
Phone (Termux)                          Shared Storage                           Syncthing                    Desktop (WSL/Linux)
~/.hermes/memories/  ←rsync→  ~/storage/shared/HermesSync/memories/  ←→  Syncthing  ←→  /home/<user>/.hermes/memories/
~/.hermes/skills/    ←rsync→  ~/storage/shared/HermesSync/skills/                    /home/<user>/.hermes/skills/
~/.hermes/sessions/  ←rsync→  ~/storage/shared/HermesSync/sessions/                  /home/<user>/.hermes/sessions/
```

### Alternative (Syncthing on Windows, polling via \\wsl$)

```
Phone (Termux)                          Shared Storage                           Syncthing (Windows)              Desktop (WSL via \\wsl$\ + poll)
~/.hermes/memories/  ←rsync→  ~/storage/shared/HermesSync/memories/  ←→  C:\Windows\...\Syncthing  ←poll→  \\wsl$\Ubuntu\home\user\.hermes\memories\
```

### Dual Hermes Agent Scenario

When two Hermes Agent instances (phone + desktop) both use the `memory` tool, they both write to the same
`MEMORY.md` and `USER.md` files. Syncthing sees these as the same file and syncs them — the **last writer wins**.
There is no content merge. The `DIALOGUE_LOG.md` is safer for cross-device logging since agents can append
without overwriting each other.

## Step 1: Identify Hermes Paths

Find paths on each device:
- **Termux**: `~/.hermes/` → `/data/data/com.termux/files/home/.hermes/`
- **WSL**: `~/.hermes/` → `/home/<user>/.hermes/`
- **Linux**: `~/.hermes/` → `/home/<user>/.hermes/`
- **Windows (native)**: Likely under `C:\\Users\\<user>\\.hermes\\`

Verify Hermes installation: `which hermes-agent && pip show hermes-agent | grep -E 'Location|Editable'`

If path looks like `/home/<user>/.hermes/` but the machine is Windows, check WSL first:
```powershell
wsl -l -v          # List WSL distros and their state
```
Then verify inside WSL: `wsl ls -la /home/<user>/.hermes/`

## Step 2: Termux — Expose to Shared Storage

Android's Syncthing can only access `/storage/emulated/0/`. Termux's private data dir is invisible. Workaround:

Create the sync directories from the **Android file manager** (not Termux, to avoid FUSE visibility issues):

```
/storage/emulated/0/HermesSync/
  ├── memories/
  ├── skills/
  └── sessions/
```

Then use a bidirectional sync script (rsync + cron) to keep `~/.hermes/` and `/storage/emulated/0/HermesSync/` in sync. **Requires `termux-api`** (see Step 6).

See `references/termux-bridge-sync.md` for the sync script and `scripts/sync_hermes.sh` for a ready-to-deploy version.

## Step 3: WSL Path Configuration

### Option A (Recommended): Run Syncthing INSIDE WSL

**Install and run Syncthing inside WSL itself:**

```bash
# In WSL
sudo apt install syncthing
syncthing   # starts on http://localhost:8384
```

Syncthing inside WSL accesses `/home/<user>/.hermes/` natively — no path issues, no monitoring errors.

### Option B (Alternative): Windows Syncthing + \\wsl$ + Polling

**Critical limitation**: Windows-native Syncthing CANNOT monitor `\\wsl$` paths with filesystem watchers — it will throw "Incorrect function" errors. However, you CAN still use Windows Syncthing by **disabling filesystem watching** and relying on periodic rescan:

In your Windows Syncthing config.xml (at `%LOCALAPPDATA%\Syncthing\config.xml`), for each WSL folder:
```xml
<folder ... fsWatcherEnabled="false" rescanIntervalS="30" ...>
```

The config achieves this by setting `fsWatcherEnabled="false"` and using `rescanIntervalS="30"` (poll every 30 seconds). This avoids the `\\wsl$` monitoring issue entirely. The folder path would be:
```
\\wsl$\Ubuntu\home\USER\.hermes\memories
```

**Trade-offs:**
- Pros: No Syncthing install needed inside WSL; simpler Windows-side management
- Cons: 30-second sync delay; higher CPU from polling; Windows must be running for sync to work

**To find the config on Windows:**
```powershell
Get-Content "$env:LOCALAPPDATA\Syncthing\config.xml"
```

## Step 4: Pair Devices & Share Folders

1. Get device IDs from each Syncthing Web UI (Actions → Show ID)
2. Add remote devices on each side
3. Create folder shares with paths:
   - **Phone**: `/storage/emulated/0/HermesSync/memories/` (label: `hermes-memories`)
   - **WSL**: `/home/<user>/.hermes/memories/` (label: `hermes-memories`)
4. Repeat for `skills`, `sessions`

## Step 5: Selective Sync (Do NOT Sync Everything)

| Sync | Do Not Sync |
|------|-------------|
| `memories/` | `state.db`, `state.db-wal`, `state.db-shm` |
| `skills/` | `.venv/` |
| `sessions/` | `cache/` |
| | `logs/` |

`config.yaml` — evaluate carefully; device-specific paths/keys may differ.

## Verification

### Method 1: File Manager (recommended for initial setup)

**Test Syncthing tunnel using the Android file manager — NOT Termux.** Termux's `~/storage/shared/` is a separate FUSE view that behaves unpredictably with Syncthing (observed: Termux-written files may be visible to Syncthing, but Syncthing-written files are invisible to Termux). Rely on the Android file manager for reliable testing:

1. **Desktop → Phone**: Create a test file on desktop (in WSL: `/home/<user>/.hermes/memories/test.txt`). Check Android file manager in `/storage/emulated/0/HermesSync/memories/`.
2. **Phone → Desktop**: In Android file manager, create a test file in the same directory. Check desktop.
3. If both directions work, the Syncthing tunnel is confirmed.

Only then proceed to the Termux bridge sync.

### Method 2: Sync Test Markers in MEMORY.md (Proven Bidirectional)

For bidirectional testing between two Hermes Agents sharing `memories/` — **verified working 2026-05-04** on WSL2 desktop ↔ Android Termux phone:

1. **Desktop → Phone**: Insert a test marker line into `MEMORY.md`:
   ```bash
   echo "【SyncTest-电脑→手机】$(date '+%Y-%m-%d %H:%M') 下行同步测试" >> ~/.hermes/memories/MEMORY.md
   ```
2. Wait ~30 seconds for Syncthing polling (or check the syncthing.log for "Updated file").
3. On the phone, check the synced `MEMORY.md` for the marker.
4. **Phone → Desktop**: From the phone's Hermes Agent, append a reverse marker:
   ```bash
   echo "【SyncTest-手机→电脑】$(date '+%Y-%m-%d %H:%M') 上行同步测试" >> ~/.hermes/memories/MEMORY.md
   ```
5. Back on the desktop, read `MEMORY.md` — the phone's marker should appear.

**Real-world result (2026-05-04):** Both directions worked. The phone Hermes Agent also independently updated MEMORY.md with detailed sync architecture notes while testing, confirming the "both agents write to the same file" scenario is functional. **No conflict files were generated** — Syncthing resolved with "last writer wins" cleanly. Key observation: after sync, `Change time (ctime)` updates on every poll cycle even when content didn't change, while `Modify time (mtime)` only updates on actual new content — use `stat --format='%y (modify) | %z (change)'` to distinguish.

### Method 3: Check Syncthing Log from WSL

The Windows Syncthing log at `%LOCALAPPDATA%\\Syncthing\\syncthing.log` is accessible from WSL at:
```bash
grep -E "Deleted|Updated|Received|folder" /mnt/c/Users/USER/AppData/Local/Syncthing/syncthing.log
```

**Real log observations (2026-05-04):** The log showed hundreds of `Deleted file` entries for old session files from April 17-28, plus `Deleted directory` for skill directories the phone side had removed. This confirms:
- Syncthing processes remote deletions (phone→desktop propagation works)
- Old sessions accumulate quickly — the phone agent may be configured differently and prunes aggressively
- The `Failed to delete directory` error for non-empty dirs indicates ignored files on one side but not the other (see Pitfalls)

The log shows each sync event with folder label, file name, and timestamp. Look for:
- `Updated file` — successful inbound sync
- `Deleted file` — remote deletion propagated
- `folder.label=homehermesmemories` etc. — which Hermes directory was affected
- `Failed to delete directory ... error="directory has been deleted on a remote device but is not empty"` — mismatch in ignore rules between devices

### Method 4: Timestamp Comparison

On the desktop, check modification times of synced files:
```bash
stat ~/.hermes/memories/MEMORY.md
# Look at Modify: vs Change: — Change time updates when Syncthing polls the file
```

If `Change` time (ctime) is newer than `Modify` time (mtime), Syncthing is actively polling the file even if no new content arrived. Note: on Windows Syncthing with `\\wsl$` polling, ctime updates on every poll cycle regardless of content change.

### Method 5: Verify Windows Syncthing Config

The actual config can be read from WSL:
```bash
cat /mnt/c/Users/USER/AppData/Local/Syncthing/config.xml
```

Look for:
- `folder` elements with `path="\\\\wsl$\\Ubuntu\\..."`
- `fsWatcherEnabled="false"` — confirms polling mode (required for `\\wsl$` paths)
- `rescanIntervalS` — polling interval (usually 30s by default)
- `type="sendreceive"` — bidirectional sync
- Device IDs — verify both phone and desktop are configured

## Step 6: Termux Bridge Sync (One-Way Until termux-api)

Due to Android's FUSE isolation, this bridge is **one-way** by default: Termux → shared storage → Syncthing.

The reverse direction (Syncthing → shared storage → Termux) requires `termux-api` installed so Termux can read files written by external apps via the Storage Access Framework. Without it, Termux **cannot read** files that Syncthing writes to shared storage.

**Install termux-api for full bidirectional access:**
```bash
pkg install termux-api
# Also install the Termux:API companion APK on Android (from F-Droid)
```
With `termux-api`, the bridge script in `references/termux-bridge-sync.md` becomes fully bidirectional.

Without `termux-api`, only use the script's first `rsync` line (Termux→shared storage) — the reverse rsync will silently find nothing to copy since Termux can't see Syncthing-written files.

### Verification That Does NOT Work

For testing, do these from the **Android system file manager**, never from Termux. Termux-created files in `~/storage/shared/` are invisible to both Syncthing and the Android file manager (they go through separate FUSE layers). Similarly, files placed by Syncthing or the file manager are invisible to Termux without `termux-api`.

## Pitfalls

1. **Windows Syncthing + `\\wsl$` = filesystem watcher breaks**. Use `fsWatcherEnabled="false"` + polling as described in Step 3 Option B if you must run Syncthing on Windows.

2. **Termux private dir invisible to Android apps**. Must bridge via `~/storage/shared/`.

3. **FUSE isolation is asymmetric with `~/storage/shared/`**. Termux-written files through `~/storage/shared/` MAY be visible to Syncthing (observed on Android 13+), but Syncthing-written files are NEVER visible to Termux via `~/storage/shared/`. The path `/storage/emulated/0/` is the canonical Android shared storage and works correctly with `termux-api` for both read and write. Always use `/storage/emulated/0/` (requires termux-api + Termux:API APK) instead of `~/storage/shared/` for the bridge script.

4. **Syncthing Android "create folder" may fail even when dir exists**. If the file picker shows the path but creation errors out: (a) check Syncthing has "Files and media" permission in Android Settings → Apps → Syncthing. On some Android versions this permission option is absent — in that case, (b) create all directories from the Android file manager FIRST, then in Syncthing select the already-existing directory. Do NOT attempt to create new folders from within Syncthing if the permission is missing.

5. **Syncthing may not auto-start on Termux**. Consider `Termux:Boot` plugin for persistent background.

6. **`state.db` is a 40+ MB SQLite file with WAL**. Syncthing will sync it literally, causing conflicts. Exclude it.

7. **Two Hermes Agents writing to the same MEMORY.md = last writer wins, no merge**. When both agents use the `memory` tool, they both write to `MEMORY.md` (and `USER.md`). Syncthing keeps the most recent version; there are no merge semantics. If the desktop agent writes at 10:00 and the phone writes at 10:01, the phone's version replaces the desktop's. **Real example (2026-05-04):** The desktop added a "SyncTest" marker line, then the phone Hermes Agent independently wrote a batch of updated sync config notes to the same file. When Syncthing synced, the phone's full-file write (via its `memory` tool) overwrote the desktop's marker. Result: the phone's content survived, the desktop's marker was lost — but no sync-conflict file was generated because Syncthing sees it as the same file with a newer timestamp.

   To prevent data loss:
   - Use test markers (see Verification Method 2) to confirm both directions work before relying on it.
   - Consider using `DIALOGUE_LOG.md` for append-only per-entry logging (both agents can append without overwriting each other).
   - The `current_task.json` state file protocol is an alternative: it's a single structured file that agents can read for current context, while MEMORY.md accumulates persistent facts.
   - **Coordination strategy:** Avoid both agents running `memory` simultaneously. If you must, have one agent be the "memory writer" and the other read-only.

8. **File ctime vs mtime confusion**. After a Syncthing poll, the file's `Change` time (ctime) updates even if content didn't change. Always check `Modify` time (mtime) to determine actual content freshness: `stat --format='%y (modify) | %z (change)' file`.

9. **`current_task.json` is NOT synced** by default — it lives in `~/.hermes/state/`, outside the three synced directories (memories/skills/sessions). To make task state cross-device, either:
   - Add `~/.hermes/state/` as a fourth Syncthing folder
   - Or symlink `~/.hermes/state/` into the memories folder
   - Or use the multi-agent memory sync protocol: set `~/.hermes/state/current_task.json` as a shared state file injected into system prompts

10. **Skills created via `skill_manage` can be silently overwritten by cross-device sync.** When both agents sync `skills/` bidirectionally and the phone agent already has a skill with the same name (or creates one before the desktop agent sees the sync), the desktop agent's newly-created skill will be replaced. **Real example (2026-05-04):** The desktop Hermes Agent created `devops/cross-device-sync` via `skill_manage`. Before the desktop's skill directory structure propagated to the phone, the phone agent independently wrote an updated `syncthing-hermes-sync` skill to the same `skills/` directory tree. When Syncthing synced, the phone's version (with its own directory and SKILL.md) overwrote the desktop's newly-created skill. The desktop's `cross-device-sync` skill was entirely replaced by the phone's `syncthing-hermes-sync` — no conflict file generated, no warning. Mitigations:
    - Before creating a new skill, check `skills_list()` to see if a similar skill name might exist on the other device
    - Give skills distinct names that won't collide with what the other agent might independently name
    - If using dual Hermes Agents, designate one agent (e.g., desktop) as the primary skill author and the other as read-only for skills
    - After creating a skill, wait for a sync cycle before assuming it's persistent

## GitHub Repository

The `hermes-cross-device-sync` skill is also available as a standalone GitHub repository:

- **URL**: https://github.com/jasonchang2025/hermes-cross-device-sync
- **Contents**: `SKILL.md` (this document), `README.md`, `references/termux-bridge-sync.md`, `scripts/sync_hermes.sh`, `LICENSE` (MIT)
- **Purpose**: Easier sharing, forking, and contribution. Pull requests welcome for additional device combinations or improved scripts.

## What's NOT Shared (Design Limits)

| Item | Reason |
|------|--------|
| `~/.hermes/state/current_task.json` | Lives outside synced directories |
| Agent runtime memory state | Each Agent session loads independently |
| Real-time conversation relay | No WebSocket or event bus between agents |
| `~/.hermes/config.yaml` | Device-specific (keys, providers, models differ) |
| Skill indices / caches | `.skills_index.json`, `state.db`, etc. are local |

## Future Roadmap

- [ ] Add `~/.hermes/state/` as a fourth sync folder for `current_task.json` + task context
- [ ] Conflict-auto-merge script (chronological append for MEMORY.md, or structured merge for JSON)
- [ ] DIALOGUE_LOG.md cross-device append (avoids full-file overwrite)
- [ ] Agent health check: detect peer's last sync timestamp and report staleness
- [ ] Optional: WebSocket relay for near-real-time context sharing