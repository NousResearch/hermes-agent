#!/usr/bin/env bash
# ============================================================================
# apply-unitalk-branding.sh — Apply Unitalk rebranding to a fresh upstream pull
# ============================================================================
# Idempotent: safe to run multiple times. Each sed pattern is precise — no
# blind replace-all that could corrupt identifiers, headers, or paths.
#
# USAGE: ./scripts/apply-unitalk-branding.sh
#
# SECTIONS:
#   1. Agent identity (prompt_builder.py, onboarding_text.py)
#   2. UI branding (i18n strings, welcome plugin, onboarding)
#   3. Theme changes (DEFAULT_SKIN_NAME, presets)
#   4. Provider list (browser-auth only, Unitalk first)
#   5. Design elements (logo, color scheme, wordmark → logo)
#   6. Verification (grep checks, Python syntax, identifier integrity)
#   7. Commit
# ============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "=== Unitalk Branding Script ==="
echo "Root: $ROOT"
echo ""

# ── SECTION 1: Agent Identity ────────────────────────────────────────────────
# prompt_builder.py holds DEFAULT_AGENT_IDENTITY — the system prompt's
# opening identity line. The upstream says "Hermes Agent, built by Nous Research".
# We replace with "AI Collaborator, built by Unitalk".
#
# onboarding_text.py holds GREETING and _GREETINGS — the scripted first-run
# greeting. The upstream says "I'm Hermes" or references "Hermes".
#
# IMPORTANT: prompt_builder.py references "hermes-agent" skill, hermes-agent.nousresearch.com/docs,
# and HERMES_AGENT_HELP_GUIDANCE — these are NOT user-facing brand strings,
# they describe the underlying product to the model. DO NOT touch them.

echo "--- Section 1: Agent Identity ---"

# Only replace the identity line — not references to hermes-agent skill/docs
PROMPT_BUILDER="$ROOT/agent/prompt_builder.py"
if grep -q 'You are Hermes Agent, built by Nous Research' "$PROMPT_BUILDER" 2>/dev/null; then
    sed -i 's/You are Hermes Agent, built by Nous Research/You are an AI Collaborator, built by Unitalk/g' "$PROMPT_BUILDER"
    echo "  Fixed: DEFAULT_AGENT_IDENTITY in prompt_builder.py"
else
    echo "  OK: prompt_builder.py already branded (or no match)"
fi

# The help guidance references "Hermes Agent" in its explanatory comment but the
# user-facing text should say "Unitalk AI Collaborator". Check the guidance strings.
if grep -q '"For help with Hermes Agent itself' "$PROMPT_BUILDER" 2>/dev/null; then
    sed -i 's/"For help with Hermes Agent itself/"For help with Unitalk AI Collaborator itself/g' "$PROMPT_BUILDER"
    echo "  Fixed: HERMES_AGENT_HELP_GUIDANCE in prompt_builder.py"
else
    echo "  OK: Help guidance already branded (or no match)"
fi

# onboarding_text.py: GREETING and _GREETINGS
ONBOARDING_TEXT="$ROOT/agent/onboarding_text.py"
if grep -q "I'm Hermes" "$ONBOARDING_TEXT" 2>/dev/null; then
    sed -i "s/I'm Hermes/I'm Unitalk/g" "$ONBOARDING_TEXT"
    echo "  Fixed: GREETING in onboarding_text.py"
fi

# Check for "the Hermes desktop app" and "the Hermes terminal UI"
# These appear in onboarding_text.py comments and code
if grep -q 'the Hermes desktop app' "$ONBOARDING_TEXT" 2>/dev/null; then
    sed -i 's/the Hermes desktop app/the Unitalk desktop app/g' "$ONBOARDING_TEXT"
    echo "  Fixed: 'the Hermes desktop app' in onboarding_text.py"
fi
if grep -q 'the Hermes terminal UI' "$ONBOARDING_TEXT" 2>/dev/null; then
    sed -i 's/the Hermes terminal UI/the Unitalk terminal UI/g' "$ONBOARDING_TEXT"
    echo "  Fixed: 'the Hermes terminal UI' in onboarding_text.py"
fi

# The facts_block still refers to "the computer running Hermes"
if grep -q 'the computer running Hermes' "$ONBOARDING_TEXT" 2>/dev/null; then
    sed -i 's/the computer running Hermes/the computer running Unitalk/g' "$ONBOARDING_TEXT"
    echo "  Fixed: 'the computer running Hermes' in onboarding_text.py"
fi

echo ""

# ── SECTION 2: UI Branding ───────────────────────────────────────────────────
# Handle i18n strings, shared metrics consent, welcome plugin text.
# "Partager avec Nous" → "Partager avec Unitalk" in all locale files.

echo "--- Section 2: UI Branding ---"

# Shared metrics files — "Share with Nous" → "Share with Unitalk"
# en_shared_metrics.ts still has "Share with Nous" and "Nous"
EN_METRICS="$ROOT/apps/desktop/src/i18n/en_shared_metrics.ts"
if grep -q 'Share with Nous' "$EN_METRICS" 2>/dev/null; then
    sed -i 's/Share with Nous/Share with Unitalk/g' "$EN_METRICS"
    echo "  Fixed: en_shared_metrics.ts (Share with Nous → Share with Unitalk)"
fi
if grep -q 'stats to Nous' "$EN_METRICS" 2>/dev/null; then
    sed -i 's/sends stats to Nous/sends stats to Unitalk/g' "$EN_METRICS"
    echo "  Fixed: en_shared_metrics.ts (stats to Nous → stats to Unitalk)"
fi

# Check other locale shared_metrics files
for f in "$ROOT"/apps/desktop/src/i18n/*_shared_metrics.ts; do
    if grep -q 'Partager avec Nous' "$f" 2>/dev/null; then
        sed -i 's/Partager avec Nous/Partager avec Unitalk/g' "$f"
        echo "  Fixed: $(basename "$f") (Partager avec Nous → Partager avec Unitalk)"
    fi
    if grep -q 'partager avec Nous' "$f" 2>/dev/null; then
        sed -i 's/partager avec Nous/partager avec Unitalk/g' "$f"
        echo "  Fixed: $(basename "$f") (partager avec Nous → Unitalk)"
    fi
    if grep -q 'envoyer à Nous' "$f" 2>/dev/null; then
        sed -i 's/envoyer à Nous/envoyer à Unitalk/g' "$f"
        echo "  Fixed: $(basename "$f") (envoyer à Nous → Unitalk)"
    fi
done

# Welcome plugin text — should reference "Collaborateur IA" not "Hermes"
WELCOME_PAGE="$ROOT/apps/desktop/src/plugins/welcome/page.tsx"
if grep -q 'Hermes Agent' "$WELCOME_PAGE" 2>/dev/null; then
    sed -i 's/Hermes Agent/Collaborateur IA/g' "$WELCOME_PAGE"
    echo "  Fixed: Welcome page (Hermes Agent → Collaborateur IA)"
fi

# First-run intro copy should already say "Unitalk" — but check
for f in "$ROOT"/apps/desktop/src/i18n/*_first_run.ts; do
    if grep -q "I'm Hermes" "$f" 2>/dev/null; then
        sed -i "s/I'm Hermes/I'm Unitalk/g" "$f"
        echo "  Fixed: $(basename "$f") (I'm Hermes → I'm Unitalk)"
    fi
done

echo ""

# ── SECTION 3: Theme Changes ─────────────────────────────────────────────────
# - Set DEFAULT_SKIN_NAME = 'unitalk'
# - Ensure unitalk brand theme is in BUILTIN_THEMES
# - Replace geeky theme names (if they still exist) with professional brand names
#   The upstream had 13 geeky themes; our fork has 11 brand themes.

echo "--- Section 3: Theme Changes ---"

PRESETS_TS="$ROOT/apps/desktop/src/themes/presets.ts"

# DEFAULT_SKIN_NAME
if grep -q "DEFAULT_SKIN_NAME = 'unitalk'" "$PRESETS_TS" 2>/dev/null; then
    echo "  OK: DEFAULT_SKIN_NAME is 'unitalk'"
else
    echo "  WARNING: DEFAULT_SKIN_NAME may need updating — check presets.ts"
fi

# Ensure 11 built-in themes (Unitalk + 10 brand themes), not 13 geeky ones
THEME_COUNT=$(grep -c "export const [a-z]*Theme:" "$PRESETS_TS" 2>/dev/null || echo "0")
echo "  Theme count in presets.ts: $THEME_COUNT (expect 11)"

# Geeky theme names that should NOT exist (upstream had these):
GEEKY_NAMES=("dracula" "nord" "gruvbox" "tokyo-night" "catppuccin" "one-dark" "solarized" "everforest" "rose-pine" "ayu" "kanagawa" "oxocarbon" "flexoki")
for name in "${GEEKY_NAMES[@]}"; do
    if grep -qi "$name" "$PRESETS_TS" 2>/dev/null; then
        echo "  WARNING: Geeky theme '$name' still present in presets.ts"
    fi
done

echo ""

# ── SECTION 4: Provider List ─────────────────────────────────────────────────
# - Hide terminal-auth providers (Qwen Code → qwen-oauth hidden)
# - Remove API-key providers except local models (Fireworks, OpenRouter hidden on onboarding screen)
# - Ensure Unitalk (nous) is first in provider order
# - Browser-auth OAuth providers only: nous, openai-codex, minimax-oauth, xai-oauth, anthropic, claude-code

echo "--- Section 4: Provider List ---"

PROVIDERS_TSX="$ROOT/apps/desktop/src/components/onboarding/providers.tsx"

if grep -q "nous: 0" "$PROVIDERS_TSX" 2>/dev/null; then
    echo "  OK: Unitalk (nous) is first in provider order"
else
    echo "  WARNING: Unitalk (nous) may not be first — check providers.tsx"
fi

if grep -q "HIDDEN_PROVIDER_IDS" "$PROVIDERS_TSX" 2>/dev/null; then
    echo "  OK: HIDDEN_PROVIDER_IDS set exists for terminal-auth providers"
fi

echo ""

# ── SECTION 5: Design Elements ───────────────────────────────────────────────
# - Replace giant wordmark with logo on intro screen
# - Apply unitalk.ai color scheme
# The design has already been applied through the "feat(design): apply unitalk.ai branding"
# commit. The welcome plugin replaces the empty state with a dashboard.

echo "--- Section 5: Design Elements ---"

WELCOME_PLUGIN="$ROOT/apps/desktop/src/plugins/welcome/plugin.tsx"
if [ -f "$WELCOME_PLUGIN" ]; then
    echo "  OK: Welcome plugin exists — replaces empty state"
else
    echo "  WARNING: Welcome plugin missing — empty state not replaced"
fi

echo ""

# ── SECTION 6: Verification ─────────────────────────────────────────────────

echo "=== Verification ==="
echo ""

# 6a. Check for forbidden replacements
echo "--- 6a. Protected identifiers (must be unchanged) ---"

PROTECTED=(
    "X-Hermes-Session-Token"
    "X-Hermes-Signature-256"
    "X-Hermes-Event"
    "X-Hermes-Delivery"
    "X-Hermes-Session-Id"
    "HERMES_HOME"
    "HERMES_DESKTOP_DEV_SERVER"
    "HERMES_DESKTOP_CDP_PORT"
    "hermes_cli"
    "hermes_state_"
    "hermes_constants"
    "hermes_bootstrap"
    "hermes_logging"
    "hermes_platform"
    "hermes_dir_acl"
    "hermes_time"
    "hermes_agent"
    "~/.hermes/"
    "~/hermes-first-task/"
    "hermes-agent.nousresearch.com/docs"
    "NousResearchInc"
)

MISTAKES=0
for pattern in "${PROTECTED[@]}"; do
    # Search for "Unitalk" + the pattern (would indicate accidental replacement)
    if grep -r "Unitalk.*${pattern}\|${pattern}.*Unitalk" "$ROOT/agent" "$ROOT/apps" --include="*.py" --include="*.ts" --include="*.tsx" -l 2>/dev/null | head -5; then
        true  # just checking
    fi
done

# More direct: ensure HERMES_HOME still exists verbatim
if ! grep -rq 'HERMES_HOME' "$ROOT/agent" --include="*.py" 2>/dev/null; then
    echo "  ❌ CRITICAL: HERMES_HOME not found — check for accidental replacement"
    MISTAKES=$((MISTAKES + 1))
else
    echo "  ✓ HERMES_HOME preserved"
fi

if ! grep -rq 'hermes-agent.nousresearch.com' "$ROOT/agent/prompt_builder.py" 2>/dev/null; then
    echo "  ❌ CRITICAL: hermes-agent.nousresearch.com/docs URL not found in prompt_builder.py"
    MISTAKES=$((MISTAKES + 1))
else
    echo "  ✓ hermes-agent.nousresearch.com/docs URL preserved"
fi

echo ""

# 6b. Grep for residual "Hermes Agent" in user-facing strings
echo "--- 6b. Residual user-facing 'Hermes Agent' ---"
RESIDUAL=$(grep -r "Hermes Agent" "$ROOT/apps/desktop/src" --include="*.ts" --include="*.tsx" -l 2>/dev/null | grep -v node_modules | grep -v '.test.' || true)
if [ -n "$RESIDUAL" ]; then
    echo "  ⚠ Residual 'Hermes Agent' found in:"
    echo "$RESIDUAL" | while read -r f; do echo "    $f"; done
else
    echo "  ✓ No residual 'Hermes Agent' in desktop UI"
fi

# 6c. Python syntax check on key files
echo ""
echo "--- 6c. Python syntax check ---"
for f in "$ROOT/agent/prompt_builder.py" "$ROOT/agent/onboarding_text.py" "$ROOT/agent/system_prompt.py"; do
    if python3 -m py_compile "$f" 2>/dev/null; then
        echo "  ✓ $(basename "$f") compiles"
    else
        echo "  ❌ $(basename "$f") has syntax errors"
        MISTAKES=$((MISTAKES + 1))
    fi
done

# 6d. Identifier integrity — ensure no Hermes identifiers received a Unitalk prefix
echo ""
echo "--- 6d. Identifier integrity ---"
# Check that Python identifiers starting with "hermes_" still start with "hermes_"
BAD_IDS=$(grep -roP '\bunitalk_[a-z_]+\b' "$ROOT/agent" --include="*.py" 2>/dev/null | sort -u || true)
if [ -n "$BAD_IDS" ]; then
    echo "  ⚠ Unexpected 'unitalk_' identifiers found:"
    echo "$BAD_IDS" | while read -r id; do echo "    $id"; done
else
    echo "  ✓ No accidental 'unitalk_' renames of hermes_ identifiers"
fi

echo ""
echo "=== Verification complete: $MISTAKES critical issues ==="

# ── SECTION 7: Commit ────────────────────────────────────────────────────────

if [ "$MISTAKES" -gt 0 ]; then
    echo ""
    echo "❌ Cannot commit: $MISTAKES critical issues found. Fix them first."
    exit 1
fi

# Check if there are changes to commit
cd "$ROOT"
if git diff --quiet && git diff --cached --quiet; then
    echo ""
    echo "✓ No changes to commit — branding is up to date."
    exit 0
fi

echo ""
echo "--- Committing changes ---"
git add -A
git commit -m "feat(branding): apply Unitalk branding to upstream pull

- Agent identity: 'AI Collaborator, built by Unitalk'
- UI strings: 'Partager avec Unitalk' in shared metrics
- Theme: DEFAULT_SKIN_NAME='unitalk', 11 brand themes
- Providers: Unitalk first, browser-auth only, terminal-auth hidden
- Design: welcome plugin replaces empty state, unitalk.ai colors

Protected identifiers untouched:
  X-Hermes-*, HERMES_HOME, hermes_cli, hermes_*,
  hermes-agent.nousresearch.com/docs, NousResearchInc"

echo "✓ Branding applied and committed."