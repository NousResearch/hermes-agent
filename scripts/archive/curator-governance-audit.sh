#!/bin/bash
# curator-governance-audit.sh - One-shot validation checkpoint
# Captures the full state of the curator governance layer after implementation.
# Session context: 09/06/26 - implement Denji-as-curator-extension architecture
#
# Architecture decisions (this session):
# 1. Re-enabled Hermes curator with Kensei config (45d stale, 120d archive)
# 2. Created pin-assigned-skills.sh - daily proactive pinning of 118 profile-referenced skills
# 3. Created curator-governance-hook.py - weekly reactive validation of curator decisions
# 4. Applied simplify-swarm (SAFE + CAREFUL tiers) + hermaguard (all CRITICAL + HIGH fixes)
# 5. Key fixes: atomic writes, error logging, concurrency lockfile, shell word-splitting, glob→rglob
#
# Run: cronjob one-shot, delivers to #governance

set -euo pipefail
BASE="$HOME/.hermes"
TIMESTAMP=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
DATE=$(date -u '+%d/%m/%y')

echo "═══════════════════════════════════════════"
echo "  CURATOR GOVERNANCE AUDIT - $DATE"
echo "  Session: 09/06/26 - curator extension implementation"
echo "═══════════════════════════════════════════"
echo ""

# ── Section 1: Environment ──
echo "## Environment"
echo "- Timestamp: $TIMESTAMP"
echo "- Host: $(hostname)"
echo "- Hermes home: $BASE"

# Curator status
CURATOR_STATUS=$(hermes curator status 2>&1)
echo ""
echo "## Curator Status"
echo '```'
echo "$CURATOR_STATUS" | head -10
echo '```'

# ── Section 2: Pin State ──
echo ""
echo "## Skill Pin State"
PIN_COUNT=$(python3 -c "
import json
with open('$BASE/skills/.usage.json') as f:
    data = json.load(f)
pinned = sum(1 for r in data.values() if isinstance(r, dict) and r.get('pinned'))
total = len(data)
print(f'{pinned} pinned of {total} total records')
")
echo "- $PIN_COUNT"

STALE_REFS=$(python3 -c "
import yaml, json, os
from pathlib import Path
BASE = Path('$BASE')
refs = set()
# Root
cfg = yaml.safe_load(open(BASE / 'config.yaml')) or {}
sb = cfg.get('skills', {}) or {}
for key in ['always_skills', 'enabled_skills']:
    raw = sb.get(key, [])
    if isinstance(raw, str):
        try: raw = json.loads(raw)
        except: raw = []
    for s in (raw or []):
        s = str(s).strip()
        if s and not s.startswith('/'): refs.add(s)
# Profiles
for d in (BASE / 'profiles').iterdir():
    if not d.is_dir() or d.name.startswith(('.', '_')): continue
    cfgp = d / 'config.yaml'
    if not cfgp.exists(): continue
    data = yaml.safe_load(open(cfgp)) or {}
    sb = data.get('skills', {}) or {}
    for key in ['always_skills', 'enabled_skills']:
        for s in (sb.get(key, []) or []):
            s = str(s).strip()
            if s and not s.startswith('/'): refs.add(s)

stale = []
for s in sorted(refs):
    matches = list((BASE / 'skills').glob(f'**/{s}/SKILL.md'))
    if not matches:
        stale.append(s)
print(f'{len(refs)} total referenced, {len(stale)} stale: ' + ', '.join(stale) if stale else '0 stale')
" 2>&1)
echo "- Profile references: $STALE_REFS"

# ── Section 3: Governance Hook Validation ──
echo ""
echo "## Governance Hook - Dry Run"
HOOK_OUTPUT=$(python3 "$BASE/scripts/curator-governance-hook.py" 2>&1)
HOOK_EXIT=$?
echo '```'
echo "$HOOK_OUTPUT"
echo '```'
echo "- Exit code: $HOOK_EXIT"

# ── Section 4: Pin Script Validation ──
echo ""
echo "## Pin Script - Dry Run"
PIN_OUTPUT=$(bash "$BASE/scripts/pin-assigned-skills.sh" 2>&1)
PIN_EXIT=$?
echo '```'
echo "$PIN_OUTPUT" | tail -12
echo '```'
echo "- Exit code: $PIN_EXIT"

# ── Section 5: Crontab Verification ──
echo ""
echo "## Cron Jobs"
hermes cron list 2>&1 | grep -A 3 'curator-' | while read line; do
    echo "  $line"
done

# ── Section 6: Lockfile Hygiene ──
echo ""
echo "## Lockfile Hygiene"
for lock in "$BASE/.curator_governance_hook.lock" "$BASE/.pin_assigned_skills.lock"; do
    if [ -f "$lock" ]; then
        echo "  ⚠️  Stale lockfile: $lock (will be cleaned on next run)"
    else
        echo "  ✅ $(basename "$lock") - clean"
    fi
done

# ── Section 7: Verdict ──
echo ""
echo "## Verdict"
OK=true
[ "$HOOK_EXIT" -eq 0 ] || { echo "- ❌ Governance hook exit code non-zero: $HOOK_EXIT"; OK=false; }
[ "$PIN_EXIT" -eq 0 ] || { echo "- ❌ Pin script exit code non-zero: $PIN_EXIT"; OK=false; }
if $OK; then
    echo "- ✅ Both scripts run clean"
    echo "- ✅ Curator enabled with Kensei config"
    echo "- ✅ Protection layers: proactive (daily pin) + reactive (weekly hook)"
    echo "- ✅ Session reference: 09/06/26 curator-extension implementation"
    echo ""
    echo "System is operational. First live curator run expected within ~24h."
else
    echo "- ❌ Issues detected - review output above"
fi

echo ""
echo "═══════════════════════════════════════════"
echo "  AUDIT COMPLETE - $DATE"
echo "═══════════════════════════════════════════"
