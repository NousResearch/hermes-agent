# ClawSec Suite — Heartbeat Check

Run this periodically (cron/systemd/CI/agent scheduler). It assumes POSIX shell, `curl`, and `jq`.

## Goals

1. Check whether `clawsec-suite` has an update available.
2. Poll the advisory feed.
3. Report new advisories, highlight affected installed skills, and require approval before removal actions.

---

## Configuration

```bash
INSTALL_ROOT="${INSTALL_ROOT:-$HOME/.openclaw/skills}"
SUITE_DIR="$INSTALL_ROOT/clawsec-suite"
GITHUB_RELEASES_API="${GITHUB_RELEASES_API:-https://api.github.com/repos/prompt-security/clawsec/releases?per_page=100}"
RELEASE_DOWNLOAD_BASE_URL="${RELEASE_DOWNLOAD_BASE_URL:-https://github.com/prompt-security/clawsec/releases/download}"
FEED_URL="${CLAWSEC_FEED_URL:-https://clawsec.prompt.security/advisories/feed.json}"
STATE_FILE="${CLAWSEC_SUITE_STATE_FILE:-$HOME/.openclaw/clawsec-suite-feed-state.json}"
MIN_FEED_INTERVAL_SECONDS="${MIN_FEED_INTERVAL_SECONDS:-300}"
```

---

## Step 0 — Basic sanity

```bash
set -euo pipefail

test -d "$SUITE_DIR"
test -f "$SUITE_DIR/skill.json"

echo "=== ClawSec Suite Heartbeat ==="
echo "When:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Suite: $SUITE_DIR"
```

---

## Step 1 — Check suite version updates

```bash
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

INSTALLED_VER="$(jq -r '.version // ""' "$SUITE_DIR/skill.json" 2>/dev/null || true)"
LATEST_TAG=""
LATEST_VER=""

if curl -fsSLo "$TMP/releases.json" "$GITHUB_RELEASES_API"; then
  LATEST_TAG="$(jq -r '[.[] | select(.tag_name | startswith("clawsec-suite-v"))][0].tag_name // ""' "$TMP/releases.json" 2>/dev/null || true)"
fi

if [ -n "$LATEST_TAG" ]; then
  if curl -fsSLo "$TMP/remote-skill.json" "$RELEASE_DOWNLOAD_BASE_URL/$LATEST_TAG/skill.json"; then
    LATEST_VER="$(jq -r '.version // ""' "$TMP/remote-skill.json" 2>/dev/null || true)"
  fi
fi

echo "Installed suite: ${INSTALLED_VER:-unknown}"
echo "Latest suite:    ${LATEST_VER:-unknown}"

if [ -z "$LATEST_VER" ]; then
  echo "WARNING: Could not determine latest suite version from release metadata."
elif [ "$LATEST_VER" != "$INSTALLED_VER" ]; then
  echo "UPDATE AVAILABLE: clawsec-suite ${INSTALLED_VER:-unknown} -> $LATEST_VER"
else
  echo "Suite appears up to date."
fi
```

---

## Step 2 — Initialize advisory state

```bash
mkdir -p "$(dirname "$STATE_FILE")"

if [ ! -f "$STATE_FILE" ]; then
  echo '{"schema_version":"1.0","known_advisories":[],"last_feed_check":null,"last_feed_updated":null}' > "$STATE_FILE"
  chmod 600 "$STATE_FILE"
fi

if ! jq -e '.schema_version and .known_advisories' "$STATE_FILE" >/dev/null 2>&1; then
  echo "WARNING: Invalid state file, resetting: $STATE_FILE"
  cp "$STATE_FILE" "${STATE_FILE}.bak.$(date -u +%Y%m%d%H%M%S)" 2>/dev/null || true
  echo '{"schema_version":"1.0","known_advisories":[],"last_feed_check":null,"last_feed_updated":null}' > "$STATE_FILE"
  chmod 600 "$STATE_FILE"
fi
```

---

## Step 3 — Advisory feed check (embedded clawsec-feed)

```bash
now_epoch="$(date -u +%s)"
last_check="$(jq -r '.last_feed_check // "1970-01-01T00:00:00Z"' "$STATE_FILE")"
last_epoch="$(date -u -d "$last_check" +%s 2>/dev/null || date -u -j -f "%Y-%m-%dT%H:%M:%SZ" "$last_check" +%s 2>/dev/null || echo 0)"

if [ $((now_epoch - last_epoch)) -lt "$MIN_FEED_INTERVAL_SECONDS" ]; then
  echo "Feed check skipped (rate limit: ${MIN_FEED_INTERVAL_SECONDS}s)."
else
  FEED_TMP="$TMP/feed.json"
  FEED_SOURCE="$FEED_URL"

  if ! curl -fsSLo "$FEED_TMP" "$FEED_URL"; then
    if [ -f "$SUITE_DIR/advisories/feed.json" ]; then
      cp "$SUITE_DIR/advisories/feed.json" "$FEED_TMP"
      FEED_SOURCE="$SUITE_DIR/advisories/feed.json (local fallback)"
      echo "WARNING: Remote feed unavailable, using local fallback."
    else
      echo "ERROR: Remote feed unavailable and no local fallback feed found."
      exit 1
    fi
  fi

  if ! jq -e '.version and (.advisories | type == "array")' "$FEED_TMP" >/dev/null 2>&1; then
    echo "ERROR: Advisory feed has invalid format."
    exit 1
  fi

  echo "Feed source: $FEED_SOURCE"
  echo "Feed updated: $(jq -r '.updated // "unknown"' "$FEED_TMP")"

  NEW_IDS_FILE="$TMP/new_ids.txt"
  jq -r --argfile state "$STATE_FILE" '($state.known_advisories // []) as $known | [.advisories[]?.id | select(. != null and ($known | index(.) | not))] | .[]?' "$FEED_TMP" > "$NEW_IDS_FILE"

  if [ -s "$NEW_IDS_FILE" ]; then
    echo "New advisories:"
    while IFS= read -r id; do
      [ -z "$id" ] && continue
      jq -r --arg id "$id" '.advisories[] | select(.id == $id) | "- [\(.severity | ascii_upcase)] \(.id): \(.title)"' "$FEED_TMP"
      jq -r --arg id "$id" '.advisories[] | select(.id == $id) | "  Exploitability: \(.exploitability_score // "unknown" | ascii_upcase)"' "$FEED_TMP"
      jq -r --arg id "$id" '.advisories[] | select(.id == $id) | "  Action: \(.action // "Review advisory details")"' "$FEED_TMP"
    done < "$NEW_IDS_FILE"
  else
    echo "FEED_OK - no new advisories"
  fi

  echo "Affected installed skills (if any):"
  found_affected=0
  removal_recommended=0
  for skill_path in "$INSTALL_ROOT"/*; do
    [ -d "$skill_path" ] || continue
    skill_name="$(basename "$skill_path")"

    skill_hits="$(jq -r --arg skill_prefix "${skill_name}@" '
      [.advisories[]
      | select(any(.affected[]?; startswith($skill_prefix)))
      | "- [\(.severity | ascii_upcase)] \(.id): \(.title)\n  Action: \(.action // "Review advisory details")"
      ] | .[]?
    ' "$FEED_TMP")"

    if [ -n "$skill_hits" ]; then
      found_affected=1
      echo "- $skill_name is referenced by advisory feed entries"
      printf "%s\n" "$skill_hits"

      if jq -e --arg skill_prefix "${skill_name}@" '
        any(
          .advisories[];
          any(.affected[]?; startswith($skill_prefix))
          and (
            ((.type // "" | ascii_downcase) == "malicious_skill")
            or ((.title // "" | ascii_downcase | test("malicious|exfiltrat|backdoor|trojan|stealer")))
            or ((.description // "" | ascii_downcase | test("malicious|exfiltrat|backdoor|trojan|stealer")))
            or ((.action // "" | ascii_downcase | test("remove|uninstall|disable|do not use|quarantine")))
          )
        )
      ' "$FEED_TMP" >/dev/null 2>&1; then
        removal_recommended=1
      fi
    fi
  done

  if [ "$found_affected" -eq 0 ]; then
    echo "- none"
  fi

  if [ "$removal_recommended" -eq 1 ]; then
    echo "Approval required: ask the user for explicit approval before removing any skill."
    echo "Double-confirmation policy: install request is first intent; require a second explicit confirmation with advisory context."
  fi

  # Persist state
  current_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  state_tmp="$TMP/state.json"

  jq --arg t "$current_utc" --arg updated "$(jq -r '.updated // ""' "$FEED_TMP")" --argfile feed "$FEED_TMP" '
    .last_feed_check = $t
    | .last_feed_updated = (if $updated == "" then .last_feed_updated else $updated end)
    | .known_advisories = ((.known_advisories // []) + [$feed.advisories[]?.id] | map(select(. != null)) | unique)
  ' "$STATE_FILE" > "$state_tmp"

  mv "$state_tmp" "$STATE_FILE"
  chmod 600 "$STATE_FILE"
fi
```

---

## Output Summary

Heartbeat output should include:
- suite version status,
- advisory feed status,
- new advisory list (if any) with exploitability scores,
- installed skills that appear in advisory `affected` lists,
- and a double-confirmation reminder before risky install/remove actions.

### Exploitability-Based Prioritization

When alerting on advisories, prioritize by **exploitability score** in addition to severity:

- `high` exploitability: Trivially or easily exploitable with public tooling, immediate action required
- `medium` exploitability: Exploitable with specific conditions, standard priority
- `low` exploitability: Difficult to exploit or theoretical, low priority

**Priority Rule**: A HIGH severity + HIGH exploitability CVE should be treated more urgently than a CRITICAL severity + LOW exploitability CVE.

If your runtime sends alerts, treat `high` exploitability advisories affecting installed skills as immediate notifications, regardless of severity rating.
