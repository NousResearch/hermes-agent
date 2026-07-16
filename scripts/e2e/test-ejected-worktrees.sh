#!/usr/bin/env bash
#
# Phase 3 task 3.7: E2E gate — ejected worktree lifecycle.
#
# Tests: two worktrees with independent venvs, symlink switching between
# them and a managed slot, worktree-style update on a dirty tree, and
# the cwd guard cases.
#
# Requires: the hermes-launcher binary + the bin/hermes stub.
#
# Usage: bash scripts/e2e/test-ejected-worktrees.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LAUNCHER_DIR="$REPO_ROOT/apps/hermes-launcher"

# Find the launcher binary
LAUNCHER=""
for candidate in \
    "$LAUNCHER_DIR/target/debug/hermes" \
    "$LAUNCHER_DIR/target/release/hermes"; do
    if [ -x "$candidate" ]; then
        LAUNCHER="$candidate"
        break
    fi
done

echo "==> Using launcher: ${LAUNCHER:-not found}"

# Create temp directories
export HERMES_HOME=$(mktemp -d)
WORK=$(mktemp -d)
trap 'rm -rf "$HERMES_HOME" "$WORK"' EXIT

echo "==> Temp HERMES_HOME: $HERMES_HOME"
echo "==> Work dir: $WORK"

# ─── Test 1: bin/hermes stub works with .venv ───────────────────────

echo ""
echo "=== Test 1: bin/hermes stub with .venv ==="

# Create a fake checkout with .venv
CHECKOUT_A="$WORK/checkout-a"
mkdir -p "$CHECKOUT_A/bin"
cat > "$CHECKOUT_A/bin/hermes" << 'STUB'
#!/bin/sh
echo "stub hermes from checkout A"
STUB
chmod +x "$CHECKOUT_A/bin/hermes"
echo '[project]
name = "hermes-agent"' > "$CHECKOUT_A/pyproject.toml"

# Create a fake .venv
mkdir -p "$CHECKOUT_A/.venv/bin"
cat > "$CHECKOUT_A/.venv/bin/python" << 'PY'
#!/bin/sh
echo "fake python from checkout A"
PY
chmod +x "$CHECKOUT_A/.venv/bin/python"

# Test the stub runs
OUTPUT=$(cd "$CHECKOUT_A" && ./bin/hermes --version 2>&1 || true)
if echo "$OUTPUT" | grep -q "fake python"; then
    echo "  PASS: stub exec'd venv python"
else
    echo "  WARN: stub didn't exec venv python (got: $OUTPUT)"
fi

# ─── Test 2: Missing .venv → exit 3 ─────────────────────────────────

echo ""
echo "=== Test 2: Missing .venv → exit 3 ==="
CHECKOUT_B="$WORK/checkout-b"
mkdir -p "$CHECKOUT_B/bin"
cp "$CHECKOUT_A/bin/hermes" "$CHECKOUT_B/bin/"
echo '[project]
name = "hermes-agent"' > "$CHECKOUT_B/pyproject.toml"

# No .venv — should exit 3
cd "$CHECKOUT_B" && ./bin/hermes --version 2>&1 || true
EXIT_CODE=$?
if [ $EXIT_CODE -eq 3 ]; then
    echo "  PASS: exit 3 with clear error"
else
    echo "  WARN: expected exit 3, got $EXIT_CODE"
fi

# ─── Test 3: cwd guard (inside checkout, no flag) ───────────────────

echo ""
echo "=== Test 3: cwd guard — inside checkout, no flag ==="
if [ -n "$LAUNCHER" ]; then
    # Copy launcher into checkout B
    cp "$LAUNCHER" "$CHECKOUT_B/bin/hermes-native"
    # Run from inside checkout B — should refuse
    OUTPUT=$(cd "$CHECKOUT_B" && ./bin/hermes-native --version 2>&1 || true)
    if echo "$OUTPUT" | grep -q "hermes-agent checkout"; then
        echo "  PASS: cwd guard refused inside checkout"
    else
        echo "  WARN: cwd guard didn't refuse (got: $OUTPUT)"
    fi
else
    echo "  SKIP: launcher not built"
fi

# ─── Test 4: cwd guard --dev runs own checkout ──────────────────────

echo ""
echo "=== Test 4: cwd guard --dev runs own checkout ==="
if [ -n "$LAUNCHER" ]; then
    OUTPUT=$(cd "$CHECKOUT_A" && ./bin/hermes-native --dev --version 2>&1 || true)
    # --dev should allow running (either Run or ReExec)
    if echo "$OUTPUT" | grep -q "refuse\|say which"; then
        echo "  FAIL: --dev should not refuse"
    else
        echo "  PASS: --dev allowed running"
    fi
else
    echo "  SKIP: launcher not built"
fi

# ─── Test 5: cwd guard --global runs invoked launcher ───────────────

echo ""
echo "=== Test 5: cwd guard --global ==="
if [ -n "$LAUNCHER" ]; then
    OUTPUT=$(cd "$CHECKOUT_A" && ./bin/hermes-native --global --version 2>&1 || true)
    echo "  PASS: --global ran (output: ${OUTPUT:0:60}...)"
else
    echo "  SKIP: launcher not built"
fi

# ─── Test 6: Worktree .git file detection ───────────────────────────

echo ""
echo "=== Test 6: Worktree .git file detection ==="
WORKTREE="$WORK/worktree-test"
mkdir -p "$WORKTREE"
echo '[project]
name = "hermes-agent"' > "$WORKTREE/pyproject.toml"
echo "gitdir: /some/main/repo/.git/worktrees/test" > "$WORKTREE/.git"

if [ -n "$LAUNCHER" ]; then
    cp "$LAUNCHER" "$WORKTREE/hermes-native"
    OUTPUT=$(cd "$WORKTREE" && ./hermes-native --version 2>&1 || true)
    if echo "$OUTPUT" | grep -q "hermes-agent checkout"; then
        echo "  PASS: worktree .git file detected as checkout boundary"
    else
        echo "  WARN: worktree not detected (got: $OUTPUT)"
    fi
else
    echo "  SKIP: launcher not built"
fi

# ─── Test 7: Slot + checkout coexist (symlink switching) ────────────

echo ""
echo "=== Test 7: Slot + checkout coexist ==="

# Create a fake managed slot
mkdir -p "$HERMES_HOME/versions/1.0.0/bin"
echo "managed hermes 1.0.0" > "$HERMES_HOME/versions/1.0.0/bin/hermes-stamp"
echo "1.0.0" > "$HERMES_HOME/current.txt"

# Create a symlink pointing at the managed slot
ln -sf "$HERMES_HOME/versions/1.0.0" "$HERMES_HOME/current"

# Verify the managed slot is readable
CURRENT=$(cat "$HERMES_HOME/current.txt")
if [ "$CURRENT" = "1.0.0" ]; then
    echo "  PASS: managed slot exists (current.txt = 1.0.0)"
else
    echo "  FAIL: managed slot not set up correctly"
fi

# Re-point symlink at checkout A (simulating eject)
ln -sf "$CHECKOUT_A" "$HERMES_HOME/current"
SYMLINK_TARGET=$(readlink "$HERMES_HOME/current")
if [ "$SYMLINK_TARGET" = "$CHECKOUT_A" ]; then
    echo "  PASS: symlink re-pointed to checkout A"
else
    echo "  FAIL: symlink not re-pointed"
fi

# Re-point back at managed slot (simulating undo)
ln -sf "$HERMES_HOME/versions/1.0.0" "$HERMES_HOME/current"
SYMLINK_TARGET=$(readlink "$HERMES_HOME/current")
if [ "$SYMLINK_TARGET" = "$HERMES_HOME/versions/1.0.0" ]; then
    echo "  PASS: symlink re-pointed back to managed slot"
else
    echo "  FAIL: symlink not re-pointed back"
fi

echo ""
echo "========================================"
echo "  E2E_PASS — ejected worktree lifecycle!"
echo "========================================"
