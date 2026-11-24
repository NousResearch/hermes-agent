# Discord Slash Command Persistence Solution

## Problem
Slash commands would "disappear" (fail to sync or get rate-limited) when:
- Switching models
- Restarting the gateway
- Changing language settings
- Any event causing the command fingerprint to change

The root cause: command descriptions were resolved from translation catalogs at runtime, so the Discord sync fingerprint changed whenever translations changed.

## Solution: Stable Command Manifest

Created `plugins/platforms/discord/discord_command_manifest.py` - a version-controlled manifest defining ALL native Discord slash commands with **stable English descriptions**.

### Key Changes

1. **New Manifest File** (`discord_command_manifest.py`)
   - Defines 31 native commands with stable English descriptions
   - Includes options, choices, templates, and followup keys
   - Version-controlled in Git - changes are tracked
   - Single source of truth for command definitions

2. **Updated Adapter** (`adapter.py`)
   - `_native_slash_commands()` now reads from manifest
   - Descriptions no longer depend on `t()` translation calls
   - Fingerprint stable across language changes for native commands
   - Backward compatibility maintained for tests (`_NATIVE_SLASH_COMMAND_SPECS`)

3. **CLI Tools** for Git workflow:
   - `discord_command_cli.py export` - Export manifest to JSON for Git commit
   - `discord_command_cli.py import` - Validate imported manifest
   - `verify_fingerprint.py` - Verify fingerprint stability

## Files Created/Modified

| File | Purpose |
|------|---------|
| `discord_command_manifest.py` | **NEW** - Stable command manifest (source of truth) |
| `discord_command_cli.py` | **NEW** - Export/import manifest for Git backup |
| `verify_fingerprint.py` | **NEW** - Verify fingerprint matches stored state |
| `adapter.py` | **MODIFIED** - Uses manifest instead of translation keys |

## Usage

### Export manifest for Git commit
```bash
cd hermes-agent
PYTHONPATH=. python plugins/platforms/discord/discord_command_cli.py export
# Creates discord_commands.json - commit this to your repo
```

### Verify fingerprint stability
```bash
PYTHONPATH=. python plugins/platforms/discord/verify_fingerprint.py
# Shows manifest fingerprint vs stored sync state
```

### Import/validate from Git
```bash
PYTHONPATH=. python plugins/platforms/discord/discord_command_cli.py import discord_commands.json
```

## Benefits

1. **Fingerprint Stability**: Native commands (31) now have immutable descriptions
2. **Git History**: Every command change tracked in version control
3. **Disaster Recovery**: Can restore command state from Git
4. **No More "Disappearing" Commands**: Model switches/restarts won't trigger unnecessary re-syncs for native commands
5. **Audit Trail**: Know exactly when/why commands changed

## What's Still Dynamic

The following still use translations and can affect fingerprint:
- `COMMAND_REGISTRY` commands (CLI commands exposed as slash)
- Plugin commands
- Skills (via `/skill` autocomplete)

These are expected to change as plugins/skills are added/removed.

## Migration Notes

- No breaking changes - existing sync state preserved
- Next sync will use new stable descriptions
- Followup messages still use translation keys (resolved at runtime)
- Language changes only affect followups, not command identity