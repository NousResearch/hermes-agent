# Profile Mismatch Reproduction Case

*Date: 2026-07-22*
*Symptom: Slack bot shows "Hermes" instead of "Poncho"*

## Initial State

- Gateway was running under the **sixty61** profile (no SOUL.md → defaulted to "Hermes")
- Desktop chat was running under the **poncho** profile (has SOUL.md → "Poncho")
- Gateway process was alive and Slack was connected, but with wrong identity

## Log Evidence

```
2026-07-22 05:37:52 INFO gateway.run: Active profile: sixty61
2026-07-22 05:37:52 INFO [Slack] Authenticated as @hermes in workspace Sixty61
```

## Root Cause

The gateway had been started from a shell where the active profile was `sixty61`. It cached that profile at startup and never re-checked. The `poncho` profile's SOUL.md ("You are Poncho") was never loaded by the gateway.

## Fix Applied

1. Re-ran `hermes gateway install` from the poncho profile
2. Gateway restarted with correct profile
3. Slack re-authenticated as `@poncho`

## After Fix

```
2026-07-22 10:52:19 INFO gateway.run: Active profile: poncho
2026-07-22 10:52:20 INFO [Slack] Authenticated as @poncho in workspace Sixty61
```

## Key Commands

```bash
# Check current gateway status
hermes gateway status

# Check which profile the gateway is using
grep "Active profile" ~/.hermes/logs/gateway.log | tail -3

# Check authenticated name on Slack
grep "Authenticated as @" ~/.hermes/logs/gateway.log | tail -3

# (Re)start gateway to pick up current profile
hermes gateway run
# Or for persistent service
hermes gateway install
```
