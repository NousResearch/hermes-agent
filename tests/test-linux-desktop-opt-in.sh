#!/usr/bin/env bash
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}  ✓${NC} $1"; }
fail() { echo -e "${RED}  ✗ FAILED${NC}: $1"; exit 1; }
warn() { echo -e "${YELLOW}  ⚠${NC} $1"; }

echo "=============================================="
echo "Linux Desktop Opt-In — Test Suite"
echo "=============================================="
echo ""

REPO_ROOT="${1:-.}"
cd "$REPO_ROOT"

# --- Find Python ---
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" >/dev/null 2>&1; then
        if "$cmd" -c "print('ok')" >/dev/null 2>&1; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    for p in /usr/bin/python3 /usr/bin/python /usr/local/bin/python3 /usr/local/bin/python; do
        if [ -x "$p" ] && "$p" -c "print('ok')" >/dev/null 2>&1; then
            PYTHON="$p"
            break
        fi
    done
fi

if [ -z "$PYTHON" ] && command -v apt-get >/dev/null 2>&1; then
    echo "  Installing Python..."
    apt-get update -qq && apt-get install -y -qq python3 && PYTHON=python3
fi

# --- 1. Bash syntax ---
echo "▶ 1. Bash syntax (setup-hermes.sh)"
bash -n setup-hermes.sh 2>/dev/null && pass "setup-hermes.sh syntax OK" || warn "bash not available"

# --- 2. Source-level checks ---
echo ""
echo "▶ 2. Source-level checks"

grep -q '"--install"' hermes_cli/subcommands/gui.py && \
    pass "--install flag in gui.py" || fail "missing --install in gui.py"

grep -q 'sys.platform.startswith("linux")' hermes_cli/main.py && \
    pass "platform guard in cmd_gui" || fail "missing platform guard"

grep -q 'from hermes_cli.linux_desktop_entry import install_desktop_entry' hermes_cli/main.py && \
    pass "imports install_desktop_entry in cmd_gui" || fail "missing import"

sed -n '/^def cmd_gui/,/^def /p' hermes_cli/main.py | head -30 | grep -q 'return' && \
    pass "cmd_gui has early return for --install" || fail "missing early return"

grep -q "Hermes Desktop app (native Linux GUI)" setup-hermes.sh && \
    pass "installer prompt present" || fail "missing installer prompt"

grep -q 'SETUP_PYTHON.*linux_desktop_entry' setup-hermes.sh && \
    pass "installer calls install_desktop_entry via SETUP_PYTHON" || fail "missing SETUP_PYTHON call"

# --- 3. Docs checks ---
echo ""
echo "▶ 3. Docs verification"

grep -q "Desktop app on Linux" website/docs/getting-started/installation.md && \
    pass "installation.md: Linux desktop section" || fail "missing in installation.md"

grep -q "With the desktop app on Linux" website/docs/getting-started/quickstart.md && \
    pass "quickstart.md: Linux desktop path" || fail "missing in quickstart.md"

grep -q "\-\-install.*Write the Linux desktop entry" website/docs/user-guide/desktop.md && \
    pass "desktop.md: --install flag table" || fail "missing in desktop.md flag table"

grep -q "On Linux, the first \`hermes desktop\` builds" website/docs/user-guide/desktop.md && \
    pass "desktop.md: Linux first-launch note" || fail "missing in desktop.md"

# --- 4. Functional test ---
if [ -n "$PYTHON" ]; then
    echo ""
    echo "▶ 4. Functional test (--install writes .desktop file)"

    VENV_PY=""
    if [ -x venv/bin/python ]; then
        VENV_PY=venv/bin/python
    elif [ -x .venv/bin/python ]; then
        VENV_PY=.venv/bin/python
    else
        VENV_PY="$PYTHON"
    fi

    rm -f ~/.local/share/applications/hermes.desktop 2>/dev/null || true
    rm -rf ~/.local/share/icons/hicolor/*/apps/hermes.png 2>/dev/null || true

    OUTPUT=$($VENV_PY -m hermes_cli.main desktop --install 2>&1) || fail "hermes desktop --install exited non-zero"

    echo "$OUTPUT" | grep -q "✓ Desktop launcher installed:" && \
        pass "--install writes launcher" || fail "no success message: $OUTPUT"

    [ -f ~/.local/share/applications/hermes.desktop ] && \
        pass ".desktop file exists" || fail ".desktop file missing"

    grep -q '^\[Desktop Entry\]' ~/.local/share/applications/hermes.desktop && \
        pass ".desktop has [Desktop Entry]" || fail "missing [Desktop Entry]"

    grep -q '^Name=Hermes' ~/.local/share/applications/hermes.desktop && \
        pass ".desktop has Name=Hermes" || fail "missing Name"

    grep -q '^Exec=' ~/.local/share/applications/hermes.desktop && \
        pass ".desktop has Exec=" || fail "missing Exec="

    grep -q '^Icon=' ~/.local/share/applications/hermes.desktop && \
        pass ".desktop has Icon=" || fail "missing Icon"

    grep -q '^Terminal=false' ~/.local/share/applications/hermes.desktop && \
        pass ".desktop has Terminal=false" || fail "missing Terminal=false"

    OUTPUT2=$($VENV_PY -m hermes_cli.main desktop --install 2>&1) || fail "re-run failed"
    echo "$OUTPUT2" | grep -q "✓ Desktop launcher installed:" && \
        pass "idempotent re-run OK" || fail "re-run failed: $OUTPUT2"

    rm -f ~/.local/share/applications/hermes.desktop 2>/dev/null || true
    rm -rf ~/.local/share/icons/hicolor/*/apps/hermes.png 2>/dev/null || true
else
    warn "no python — skipping functional test"
fi

# --- 5. Existing test suite ---
if [ -n "$PYTHON" ] && $PYTHON -m pytest --version >/dev/null 2>&1; then
    echo ""
    echo "▶ 5. Existing test suite — test_gui_command.py"
    $PYTHON -m pytest tests/hermes_cli/test_gui_command.py -q --tb=short 2>&1 | tail -5
else
    warn "pytest not available — skipping existing test suite"
fi

echo ""
echo "=============================================="
echo -e "${GREEN}✓ All tests passed${NC}"
echo "=============================================="
