#!/usr/bin/env bash
# Engagement Scout — batched X scanning (5 batches of 5 accounts)
# Each run processes one batch, rotating through all 5 batches across the day.
# Keeps each run under the 300s cron scheduler timeout.
set -euo pipefail

PROJECT_DIR="/home/kensei/repos/KenseiAgent"
ENGAGEMENT_PY="$PROJECT_DIR/content_engine/engagement_suggester.py"
CE_DIR="$PROJECT_DIR/content_engine"
STATE_FILE="/home/kensei/.hermes/data/engagement-scout-batch.txt"

# Source env for DISCORD_BOT_TOKEN + OLLAMA_API_KEY
set -a
source /home/kensei/.hermes/.env 2>/dev/null || true
set +a

export PYTHONPATH="${CE_DIR}:${PYTHONPATH:-}"

# Define 5 batches of accounts
BATCH_0="marc_louvion,levelsio,tahseen_rahman,shadcn,rauchg"
BATCH_1="theo,swyx,ryan_c_harris,kentcdodds,pk_hal"
BATCH_2="AnthropicAI,claude_code,alexalbert__,amasad,TheAthleticFC"
BATCH_3="utdreport,ManUtd,GaryLineker,naval,paulg"
BATCH_4="sweatystartup,NousResearch,teknium,hermesagent,Sahil_Saghir"

# Determine current batch (rotate 0→1→2→3→4→0...)
mkdir -p "$(dirname "$STATE_FILE")"
CURRENT=0
if [ -f "$STATE_FILE" ]; then
    CURRENT=$(cat "$STATE_FILE")
fi

# Select the batch
case "$CURRENT" in
    0) BATCH="$BATCH_0" ;;
    1) BATCH="$BATCH_1" ;;
    2) BATCH="$BATCH_2" ;;
    3) BATCH="$BATCH_3" ;;
    4) BATCH="$BATCH_4" ;;
    *) BATCH="$BATCH_0"; CURRENT=0 ;;
esac

# Advance for next run
NEXT=$(( (CURRENT + 1) % 5 ))
echo "$NEXT" > "$STATE_FILE"

export ENGAGEMENT_BATCH_ACCOUNTS="$BATCH"

# Self-heal: ensure Playwright chromium is installed. The engagement pipeline
# uses Playwright headless to scrape X.com; if the browser binary is missing
# (e.g. after a `pip install` that bumped the Playwright version, or a
# sandbox reset), the script otherwise dies with
#   playwright._impl._errors.Error: BrowserType.launch: Executable doesn't exist
# and the run exits non-zero. Detect the missing binary and reinstall once
# before paying the full python startup cost. We do this with a hard cap so a
# broken install can't blow the 300s cron budget.
ensure_playwright_browser() {
    local venv_py="/home/kensei/repos/KenseiAgent/.venv/bin/python"
    # The engagement pipeline launches with headless=True, which (since
    # Playwright 1.49) prefers a separate chromium_headless_shell install
    # rather than the full chromium browser. If the headless_shell binary
    # is missing, the cron dies with
    #   BrowserType.launch: Executable doesn't exist at
    #   .../chromium_headless_shell-1223/chrome-linux/headless_shell
    # even if the full chromium is present. So we detect by checking both
    # binaries at the playwright browsers cache, and reinstall via
    # `playwright install chromium` if either is missing. The detection
    # does NOT launch the browser, so it's cheap.
    local status_file
    status_file="$(mktemp)"
    ( cd "$CE_DIR" && "$venv_py" <<'PY' >"$status_file" 2>/dev/null
from playwright.sync_api import sync_playwright
import os
with sync_playwright() as pw:
    p_full = pw.chromium.executable_path
    # p_full looks like .../chromium-<ver>/chrome-linux/chrome. The
    # headless_shell lives at .../chromium_headless_shell-<ver>/chrome-linux/
    # headless_shell — sibling directory, same version segment.
    cache_root = os.path.dirname(os.path.dirname(os.path.dirname(p_full)))
    ver = os.path.basename(os.path.dirname(os.path.dirname(p_full))).replace('chromium-', '')
    p_headless = os.path.join(cache_root, f'chromium_headless_shell-{ver}',
                              'chrome-linux', 'headless_shell')
    full_ok = os.path.isfile(p_full) and os.access(p_full, os.X_OK)
    head_ok = os.path.isfile(p_headless) and os.access(p_headless, os.X_OK)
    print('OK' if (full_ok and head_ok) else 'NEED_INSTALL')
    print(f'full: {p_full} ({"ok" if full_ok else "MISSING"})')
    print(f'headless: {p_headless} ({"ok" if head_ok else "MISSING"})')
PY
    ) || true
    local first_line
    first_line="$(head -n1 "$status_file" 2>/dev/null || true)"
    rm -f "$status_file"

    if [ "$first_line" = "OK" ]; then
        return 0
    fi

    echo "[engagement-scout] Playwright browser incomplete; running 'playwright install chromium' (self-heal)..." >&2
    # 240s cap leaves headroom under the 300s outer timeout for the real scan.
    if timeout 240 "$venv_py" -m playwright install chromium >/tmp/engagement_playwright_install.log 2>&1; then
        echo "[engagement-scout] Playwright install complete." >&2
        return 0
    fi
    echo "[engagement-scout] Playwright install FAILED; continuing — scan will likely error and surface the real cause." >&2
    return 0  # do not block the scan; let it fail with the original error
}

ensure_playwright_browser

cd "$CE_DIR"
echo "[engagement-scout] Batch $CURRENT — accounts: $BATCH"
exec /home/kensei/repos/KenseiAgent/.venv/bin/python "$ENGAGEMENT_PY" scan 2>&1
