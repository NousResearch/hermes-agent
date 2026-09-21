---
name: clawsec-advisory-guardian
description: Detect advisory matches for installed skills and require explicit user approval before any removal action.
metadata: { "openclaw": { "events": ["agent:bootstrap", "command:new"] } }
---

# ClawSec Advisory Guardian Hook

This hook checks the ClawSec advisory feed against locally installed skills on:

- `agent:bootstrap`
- `command:new`

When it detects an advisory affecting an installed skill, it posts an alert message.
If the advisory looks malicious or removal-oriented, it explicitly recommends removal
and asks for user approval first.

## Safety Contract

- The hook does not delete or modify skills.
- It only reports findings and requests explicit approval before removal.
- Alerts are deduplicated using `~/.openclaw/clawsec-suite-feed-state.json`.

## Optional Environment Variables

- `CLAWSEC_FEED_URL`: override remote feed URL.
- `CLAWSEC_FEED_SIG_URL`: override detached remote feed signature URL (default `${CLAWSEC_FEED_URL}.sig`).
- `CLAWSEC_FEED_CHECKSUMS_URL`: override remote checksum manifest URL (default sibling `checksums.json`).
- `CLAWSEC_FEED_CHECKSUMS_SIG_URL`: override detached remote checksum manifest signature URL.
- `CLAWSEC_FEED_PUBLIC_KEY`: path to pinned feed-signing public key PEM.
- `CLAWSEC_LOCAL_FEED`: override local fallback feed file.
- `CLAWSEC_LOCAL_FEED_SIG`: override local detached feed signature path.
- `CLAWSEC_LOCAL_FEED_CHECKSUMS`: override local checksum manifest path.
- `CLAWSEC_LOCAL_FEED_CHECKSUMS_SIG`: override local checksum manifest signature path.
- `CLAWSEC_VERIFY_CHECKSUM_MANIFEST`: set to `0` only for emergency troubleshooting (default verifies checksums).
- `CLAWSEC_ALLOW_UNSIGNED_FEED`: set to `1` only for temporary migration compatibility; bypasses signature/checksum verification.
- `CLAWSEC_SUITE_STATE_FILE`: override state file path.
- `CLAWSEC_INSTALL_ROOT`: override installed skills root.
- `CLAWSEC_SUITE_DIR`: override clawsec-suite install path.
- `CLAWSEC_HOOK_INTERVAL_SECONDS`: minimum interval between hook scans (default `300`).
