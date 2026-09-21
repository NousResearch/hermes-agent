---
name: clawsec-suite
version: 0.1.7
description: ClawSec suite manager with embedded advisory-feed monitoring, cryptographic signature verification, approval-gated malicious-skill response, and guided setup for additional security skills.
homepage: https://clawsec.prompt.security
clawdis:
  emoji: "📦"
  requires:
    bins: [node, npx, openclaw, curl, jq, shasum, openssl, unzip]
adoption_status: permanent
---

# ClawSec Suite

## Operational Notes

- Required runtime: `node`, `npx`, `openclaw`, `curl`, `jq`, `shasum`, `openssl`, `unzip`
- Side effects: setup scripts install an advisory hook under `~/.openclaw/hooks`, optionally create an unattended `openclaw cron` job, and use `npx clawhub@latest install` for guarded installs
- Network behavior: fetches signed advisory feed artifacts and remote catalog metadata unless you pin local paths
- Trust model: the suite can recommend removal or block risky installs, but removal/install overrides stay approval-gated

This means `clawsec-suite` can:
- monitor the ClawSec advisory feed,
- track which advisories are new since last check,
- cross-reference advisories against locally installed skills,
- recommend removal for malicious-skill advisories and require explicit user approval first,
- and still act as the setup/management entrypoint for other ClawSec protections.

## Included vs Optional Protections

### Built into clawsec-suite
- Embedded feed seed file: `advisories/feed.json`
- Portable heartbeat workflow in `HEARTBEAT.md`
- Advisory polling + state tracking + affected-skill checks
- OpenClaw advisory guardian hook package: `hooks/clawsec-advisory-guardian/`
- Setup scripts for hook and optional cron scheduling: `scripts/`
- Guarded installer: `scripts/guarded_skill_install.mjs`
- Dynamic catalog discovery for installable skills: `scripts/discover_skill_catalog.mjs`

### Installed separately (dynamic catalog)
`clawsec-suite` does not hard-code add-on skill names in this document.

Discover the current catalog from the authoritative index (`https://clawsec.prompt.security/skills/index.json`) at runtime:

```bash
SUITE_DIR="${INSTALL_ROOT:-$HOME/.openclaw/skills}/clawsec-suite"
node "$SUITE_DIR/scripts/discover_skill_catalog.mjs"
```

Fallback behavior:
- If the remote catalog index is reachable and valid, the suite uses it.
- If the remote index is unavailable or malformed, the script falls back to suite-local catalog metadata in `skill.json`.

## Installation

### Cross-shell path note

- In `bash`/`zsh`, keep path variables expandable (for example, `INSTALL_ROOT="$HOME/.openclaw/skills"`).
- Do not single-quote home-variable paths (avoid `'$HOME/.openclaw/skills'`).
- In PowerShell, set an explicit path:
  - `$env:INSTALL_ROOT = Join-Path $HOME ".openclaw\\skills"`
- If a path is passed with unresolved tokens (like `\$HOME/...`), suite scripts now fail fast with a clear error.

### Option A: Via clawhub (recommended)

```bash
npx clawhub@latest install clawsec-suite
```

### Option B: Manual download with signature + checksum verification

```bash
set -euo pipefail

VERSION="${SKILL_VERSION:?Set SKILL_VERSION (e.g. 0.0.8)}"
INSTALL_ROOT="${INSTALL_ROOT:-$HOME/.openclaw/skills}"
DEST="$INSTALL_ROOT/clawsec-suite"
BASE="https://github.com/prompt-security/clawsec/releases/download/clawsec-suite-v${VERSION}"

TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TEMP_DIR"' EXIT

# Pinned release-signing public key (verify fingerprint out-of-band on first use)
# Fingerprint (SHA-256 of SPKI DER): 711424e4535f84093fefb024cd1ca4ec87439e53907b305b79a631d5befba9c8
RELEASE_PUBKEY_SHA256="711424e4535f84093fefb024cd1ca4ec87439e53907b305b79a631d5befba9c8"
cat > "$TEMP_DIR/release-signing-public.pem" <<'PEM'
-----BEGIN PUBLIC KEY-----
MCowBQYDK2VwAyEAS7nijfMcUoOBCj4yOXJX+GYGv2pFl2Yaha1P4v5Cm6A=
-----END PUBLIC KEY-----
PEM

ACTUAL_KEY_SHA256="$(openssl pkey -pubin -in "$TEMP_DIR/release-signing-public.pem" -outform DER | shasum -a 256 | awk '{print $1}')"
if [ "$ACTUAL_KEY_SHA256" != "$RELEASE_PUBKEY_SHA256" ]; then
  echo "ERROR: Release public key fingerprint mismatch" >&2
  exit 1
fi

ZIP_NAME="clawsec-suite-v${VERSION}.zip"

# 1) Download release archive + signed checksums manifest + signing public key
curl -fsSL "$BASE/$ZIP_NAME" -o "$TEMP_DIR/$ZIP_NAME"
curl -fsSL "$BASE/checksums.json" -o "$TEMP_DIR/checksums.json"
curl -fsSL "$BASE/checksums.sig" -o "$TEMP_DIR/checksums.sig"

# 2) Verify checksums manifest signature before trusting any hashes
openssl base64 -d -A -in "$TEMP_DIR/checksums.sig" -out "$TEMP_DIR/checksums.sig.bin"
if ! openssl pkeyutl -verify \
  -pubin \
  -inkey "$TEMP_DIR/release-signing-public.pem" \
  -sigfile "$TEMP_DIR/checksums.sig.bin" \
  -rawin \
  -in "$TEMP_DIR/checksums.json" >/dev/null 2>&1; then
  echo "ERROR: checksums.json signature verification failed" >&2
  exit 1
fi

EXPECTED_ZIP_SHA="$(jq -r '.archive.sha256 // empty' "$TEMP_DIR/checksums.json")"
if [ -z "$EXPECTED_ZIP_SHA" ]; then
  echo "ERROR: checksums.json missing archive.sha256" >&2
  exit 1
fi

if command -v shasum >/dev/null 2>&1; then
  ACTUAL_ZIP_SHA="$(shasum -a 256 "$TEMP_DIR/$ZIP_NAME" | awk '{print $1}')"
else
  ACTUAL_ZIP_SHA="$(sha256sum "$TEMP_DIR/$ZIP_NAME" | awk '{print $1}')"
fi

if [ "$EXPECTED_ZIP_SHA" != "$ACTUAL_ZIP_SHA" ]; then
  echo "ERROR: Archive checksum mismatch for $ZIP_NAME" >&2
  exit 1
fi

echo "Checksums manifest signature and archive hash verified."

# 3) Install verified archive
mkdir -p "$INSTALL_ROOT"
rm -rf "$DEST"
unzip -q "$TEMP_DIR/$ZIP_NAME" -d "$INSTALL_ROOT"

chmod 600 "$DEST/skill.json"
find "$DEST" -type f ! -name "skill.json" -exec chmod 644 {} \;

echo "Installed clawsec-suite v${VERSION} to: $DEST"
echo "Next step (OpenClaw): node \"\$DEST/scripts/setup_advisory_hook.mjs\""
```

## OpenClaw Automation (Hook + Optional Cron)

After installing the suite, enable the advisory guardian hook:

```bash
SUITE_DIR="${INSTALL_ROOT:-$HOME/.openclaw/skills}/clawsec-suite"
node "$SUITE_DIR/scripts/setup_advisory_hook.mjs"
```

The setup script prints a preflight review before it installs and enables the persistent hook.

Optional: create/update a periodic cron nudge (default every `6h`) that triggers a main-session advisory scan:

```bash
SUITE_DIR="${INSTALL_ROOT:-$HOME/.openclaw/skills}/clawsec-suite"
node "$SUITE_DIR/scripts/setup_advisory_cron.mjs"
```

The cron setup script prints a preflight review before it creates or updates the unattended job.

What this adds:
- scan on `agent:bootstrap` and `/new` (`command:new`),
- compare advisory `affected` entries against installed skills,
- consider advisories with `application: "openclaw"` (and legacy entries without `application` for backward compatibility),
- notify when new matches appear,
- and ask for explicit user approval before any removal flow.

Restart the OpenClaw gateway after enabling the hook. Then run `/new` once to force an immediate scan in the next session context.

## Guarded Skill Install Flow (Double Confirmation)

When the user asks to install a skill, treat that as the first request and run a guarded install check:

```bash
SUITE_DIR="${INSTALL_ROOT:-$HOME/.openclaw/skills}/clawsec-suite"
node "$SUITE_DIR/scripts/guarded_skill_install.mjs" --skill helper-plus --version 1.0.1
```

Behavior:
- If no advisory match is found, install proceeds.
- If `--version` is omitted, matching is conservative: any advisory that references the skill name is treated as a match.
- If advisory match is found, the script prints advisory context and exits with code `42`.
- Then require an explicit second confirmation from the user and rerun with `--confirm-advisory`:

```bash
node "$SUITE_DIR/scripts/guarded_skill_install.mjs" --skill helper-plus --version 1.0.1 --confirm-advisory
```

This enforces:
1. First confirmation: user asked to install.
2. Second confirmation: user explicitly approves install after seeing advisory details.

## Embedded Advisory Feed Behavior

The embedded feed logic uses these defaults:

- Remote feed URL: `https://clawsec.prompt.security/advisories/feed.json`
- Remote feed signature URL: `${CLAWSEC_FEED_URL}.sig` (override with `CLAWSEC_FEED_SIG_URL`)
- Remote checksums manifest URL: sibling `checksums.json` (override with `CLAWSEC_FEED_CHECKSUMS_URL`)
- Local seed fallback: `~/.openclaw/skills/clawsec-suite/advisories/feed.json`
- Local feed signature: `${CLAWSEC_LOCAL_FEED}.sig` (override with `CLAWSEC_LOCAL_FEED_SIG`)
- Local checksums manifest: `~/.openclaw/skills/clawsec-suite/advisories/checksums.json`
- Pinned feed signing key: `~/.openclaw/skills/clawsec-suite/advisories/feed-signing-public.pem` (override with `CLAWSEC_FEED_PUBLIC_KEY`)
- State file: `~/.openclaw/clawsec-suite-feed-state.json`
- Hook rate-limit env (OpenClaw hook): `CLAWSEC_HOOK_INTERVAL_SECONDS` (default `300`)

**Fail-closed verification:** Feed signatures are required by default. Checksum manifests are verified when companion checksum artifacts are available. Set `CLAWSEC_ALLOW_UNSIGNED_FEED=1` only as a temporary migration bypass when adopting this version before signed feed artifacts are available upstream.

### Quick feed check

```bash
FEED_URL="${CLAWSEC_FEED_URL:-https://clawsec.prompt.security/advisories/feed.json}"
STATE_FILE="${CLAWSEC_SUITE_STATE_FILE:-$HOME/.openclaw/clawsec-suite-feed-state.json}"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

if ! curl -fsSLo "$TMP/feed.json" "$FEED_URL"; then
  echo "ERROR: Failed to fetch advisory feed"
  exit 1
fi

if ! jq -e '.version and (.advisories | type == "array")' "$TMP/feed.json" >/dev/null; then
  echo "ERROR: Invalid advisory feed format"
  exit 1
fi

mkdir -p "$(dirname "$STATE_FILE")"
if [ ! -f "$STATE_FILE" ]; then
  echo '{"schema_version":"1.0","known_advisories":[],"last_feed_check":null,"last_feed_updated":null}' > "$STATE_FILE"
  chmod 600 "$STATE_FILE"
fi

NEW_IDS_FILE="$TMP/new_ids.txt"
jq -r --argfile state "$STATE_FILE" '($state.known_advisories // []) as $known | [.advisories[]?.id | select(. != null and ($known | index(.) | not))] | .[]?' "$TMP/feed.json" > "$NEW_IDS_FILE"

if [ -s "$NEW_IDS_FILE" ]; then
  echo "New advisories detected:"
  while IFS= read -r id; do
    [ -z "$id" ] && continue
    jq -r --arg id "$id" '.advisories[] | select(.id == $id) | "- [\(.severity | ascii_upcase)] \(.id): \(.title)"' "$TMP/feed.json"
    jq -r --arg id "$id" '.advisories[] | select(.id == $id) | "  Exploitability: \(.exploitability_score // "unknown" | ascii_upcase)"' "$TMP/feed.json"
  done < "$NEW_IDS_FILE"
else
  echo "FEED_OK - no new advisories"
fi
```

## Exploitability Context

Advisories in the feed can include `exploitability_score` and `exploitability_rationale` fields to help agents prioritize real-world threats:

- **Exploitability scores**: `high`, `medium`, `low`, or `unknown`
- **Context-aware assessment**: Considers attack vector, authentication requirements, and AI agent deployment patterns
- **Exploit availability**: Detects public exploits and weaponization status

When processing advisories, prioritize by exploitability in addition to severity. A HIGH severity + HIGH exploitability CVE is more urgent than a CRITICAL severity + LOW exploitability CVE.

For detailed methodology, see the [exploitability scoring documentation](../../wiki/exploitability-scoring.md).

## Heartbeat Integration

Use the suite heartbeat script as the single periodic security check entrypoint:

- `skills/clawsec-suite/HEARTBEAT.md`

It handles:
- suite update checks,
- feed polling,
- new-advisory detection,
- affected-skill cross-referencing,
- approval-gated response guidance for malicious/removal advisories,
- and persistent state updates.

## Approval-Gated Response Contract

If an advisory indicates a malicious or removal-recommended skill and that skill is installed:

1. Notify the user immediately with advisory details and severity.
2. Recommend removing or disabling the affected skill.
3. Treat the original install request as first intent only.
4. Ask for explicit second confirmation before deletion/disable action (or before proceeding with risky install).
5. Only proceed after that second confirmation.

The suite hook and heartbeat guidance are intentionally non-destructive by default.

## Advisory Suppression / Allowlist

The advisory guardian pipeline supports opt-in suppression for advisories that have been reviewed and accepted by your security team. This is useful for first-party tooling or advisories that do not apply to your deployment.

### Activation

Advisory suppression requires a single gate: the configuration file must contain `"enabledFor"` with `"advisory"` in the array. No CLI flag is needed -- the sentinel in the config file IS the opt-in gate.

If the `enabledFor` array is missing, empty, or does not include `"advisory"`, all advisories are reported normally.

### Config File Resolution (4-tier)

The advisory guardian resolves the suppression config using the same priority order as the audit pipeline:

1. Explicit `--config <path>` argument
2. `OPENCLAW_AUDIT_CONFIG` environment variable
3. `~/.openclaw/security-audit.json`
4. `.clawsec/allowlist.json`

### Config Format

```json
{
  "enabledFor": ["advisory"],
  "suppressions": [
    {
      "checkId": "CVE-2026-25593",
      "skill": "clawsec-suite",
      "reason": "First-party security tooling — reviewed by security team",
      "suppressedAt": "2026-02-15"
    },
    {
      "checkId": "CLAW-2026-0001",
      "skill": "example-skill",
      "reason": "Advisory does not apply to our deployment configuration",
      "suppressedAt": "2026-02-16"
    }
  ]
}
```

### Sentinel Semantics

- `"enabledFor": ["advisory"]` -- only advisory suppression active
- `"enabledFor": ["audit"]` -- only audit suppression active (no effect on advisory pipeline)
- `"enabledFor": ["audit", "advisory"]` -- both pipelines honor suppressions
- Missing or empty `enabledFor` -- no suppression active (safe default)

### Matching Rules

- **checkId:** exact match against the advisory ID (e.g., `CVE-2026-25593` or `CLAW-2026-0001`)
- **skill:** case-insensitive match against the affected skill name from the advisory
- Both fields must match for an advisory to be suppressed

### Required Fields per Suppression Entry

| Field | Description | Example |
|-------|-------------|---------|
| `checkId` | Advisory ID to suppress | `CVE-2026-25593` |
| `skill` | Affected skill name | `clawsec-suite` |
| `reason` | Justification for audit trail (required) | `First-party tooling, reviewed by security team` |
| `suppressedAt` | ISO 8601 date (YYYY-MM-DD) | `2026-02-15` |

### Shared Config with Audit Pipeline

The advisory and audit pipelines share the same config file. Use the `enabledFor` array to control which pipelines honor the suppression list:

```json
{
  "enabledFor": ["audit", "advisory"],
  "suppressions": [
    {
      "checkId": "skills.code_safety",
      "skill": "clawsec-suite",
      "reason": "First-party tooling — audit finding accepted",
      "suppressedAt": "2026-02-15"
    },
    {
      "checkId": "CVE-2026-25593",
      "skill": "clawsec-suite",
      "reason": "First-party tooling — advisory reviewed",
      "suppressedAt": "2026-02-15"
    }
  ]
}
```

Audit entries (with check identifiers like `skills.code_safety`) are only matched by the audit pipeline. Advisory entries (with advisory IDs like `CVE-2026-25593` or `CLAW-2026-0001`) are only matched by the advisory pipeline. Each pipeline filters for its own relevant entries.

## Optional Skill Installation

Discover currently available installable skills dynamically, then install the ones you want:

```bash
SUITE_DIR="${INSTALL_ROOT:-$HOME/.openclaw/skills}/clawsec-suite"
node "$SUITE_DIR/scripts/discover_skill_catalog.mjs"

# then install any discovered skill by name
npx clawhub@latest install <skill-name>
```

Machine-readable output is also available for automation:

```bash
node "$SUITE_DIR/scripts/discover_skill_catalog.mjs" --json
```

## Security Notes

- Always verify `checksums.json` signature before trusting its file URLs/hashes, then verify each file checksum.
- Verify advisory feed detached signatures; do not enable `CLAWSEC_ALLOW_UNSIGNED_FEED` outside temporary migration windows.
- Keep advisory polling rate-limited (at least 5 minutes between checks).
- Treat `critical` and `high` advisories affecting installed skills as immediate action items.
- If you migrate off standalone `clawsec-feed`, keep one canonical state file to avoid duplicate notifications.
- Pin and verify public key fingerprints out-of-band before first use.
