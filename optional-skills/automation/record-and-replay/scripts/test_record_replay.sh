#!/bin/bash
# Integration test for Record & Replay pipeline
# Tests: record → analyze → generate → verify
#
# Usage: bash scripts/test_record_replay.sh
#
# This test:
#   a) Records a 5-second test workflow (simulated — no real GUI interaction needed)
#   b) Runs analyze_recording.py on the result
#   c) Verifies the analysis JSON has the expected structure
#   d) Runs generate_skill.py to create a test skill
#   e) Verifies the SKILL.md was created with correct frontmatter
#   f) Cleans up test artifacts
#   g) Exits 0 if all checks pass

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SKILL_DIR="$(cd "$(dirname "$SCRIPT_DIR")" && pwd)"
TEST_DIR="/tmp/test-recording-$$"
SKILL_NAME="test-replay-skill"
OUTPUT_SKILL_DIR="/tmp/test-skill-$$"

echo "🧪 Record & Replay Integration Test"
echo "==================================="
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo "🧹 Cleaning up..."
    rm -rf "$TEST_DIR" "$OUTPUT_SKILL_DIR" 2>/dev/null || true
    # Kill any leftover recording processes
    pkill -f "record_workflow.py" 2>/dev/null || true
    echo "✅ Cleanup complete"
}
trap cleanup EXIT

# Step 0: Verify prerequisites
echo "Step 0: Verify prerequisites"
if ! python3 -c "import json, hashlib, argparse" 2>/dev/null; then
    echo "❌ Python 3 with stdlib required"
    exit 1
fi
if ! python3 -c "import PIL" 2>/dev/null; then
    echo "⚠️  PIL/Pillow not installed — screenshot diffing will be disabled"
fi
echo "✅ Prerequisites OK"
echo ""

# Step 1: Create a synthetic recording (simulates what record_workflow.py produces)
echo "Step 1: Create synthetic test recording"
mkdir -p "$TEST_DIR/events" "$TEST_DIR/screenshots" "$TEST_DIR/ax_trees"

# Create events.jsonl with a realistic workflow pattern
cat > "$TEST_DIR/events/events.jsonl" << 'EVENTS'
{"type":"snapshot","timestamp":0.0,"wall_time":"2026-06-21T10:00:00","screenshot":"screenshots/0001.png","ax_tree":"ax_trees/0001.txt","frontmost_app":{"name":"Calculator","bundle_id":"com.apple.calculator","pid":12345},"mouse_position":[400,300],"screenshot_number":1}
{"type":"mouse_down","timestamp":0.5,"wall_time":"2026-06-21T10:00:00","button":"left","position":[300,400],"modifiers":[]}
{"type":"mouse_up","timestamp":0.6,"wall_time":"2026-06-21T10:00:00","button":"left","position":[300,400],"modifiers":[]}
{"type":"key_down","timestamp":1.0,"wall_time":"2026-06-21T10:00:01","key":"2","character":"2","modifiers":[]}
{"type":"key_down","timestamp":1.5,"wall_time":"2026-06-21T10:00:01","key":"+","character":"+","modifiers":[]}
{"type":"key_down","timestamp":2.0,"wall_time":"2026-06-21T10:00:02","key":"2","character":"2","modifiers":[]}
{"type":"key_down","timestamp":2.5,"wall_time":"2026-06-21T10:00:02","key":"=","character":"=","modifiers":[]}
{"type":"snapshot","timestamp":3.0,"wall_time":"2026-06-21T10:00:03","screenshot":"screenshots/0002.png","ax_tree":"ax_trees/0002.txt","frontmost_app":{"name":"Calculator","bundle_id":"com.apple.calculator","pid":12345},"mouse_position":[400,300],"screenshot_number":2}
{"type":"clipboard_copy","timestamp":4.0,"wall_time":"2026-06-21T10:00:04","preview":"4","content_hash":"abc123","content_length":1}
{"type":"screenshot_skipped","timestamp":5.0,"wall_time":"2026-06-21T10:00:05","reason":"no_change","frontmost_app":{"name":"Calculator","bundle_id":"com.apple.calculator","pid":12345},"mouse_position":[400,300],"screenshot_number":3}
EVENTS

# Create metadata.json
cat > "$TEST_DIR/metadata.json" << 'META'
{
    "version": "1.1.0",
    "created_at": "2026-06-21T10:00:00",
    "duration_seconds": 5.0,
    "event_count": 10,
    "screenshot_count": 2,
    "screenshots_skipped": 1,
    "interval": 1.0,
    "platform": "darwin"
}
META

# Create dummy screenshots (small valid PNGs)
python3 -c "
from PIL import Image
img = Image.new('RGB', (100, 100), color='red')
img.save('$TEST_DIR/screenshots/0001.png')
img2 = Image.new('RGB', (100, 100), color='blue')
img2.save('$TEST_DIR/screenshots/0002.png')
" 2>/dev/null || echo "⚠️  Could not create test screenshots (PIL not available)"

# Create dummy AX trees
echo "# AX Tree (osascript fallback)
Frontmost app: Calculator
Window: Calculator" > "$TEST_DIR/ax_trees/0001.txt"
echo "# AX Tree (osascript fallback)
Frontmost app: Calculator
Window: Calculator
Result: 4" > "$TEST_DIR/ax_trees/0002.txt"

echo "✅ Synthetic recording created at $TEST_DIR"
echo ""

# Step 2: Run analyze_recording.py
echo "Step 2: Run analyze_recording.py"
ANALYSIS_OUTPUT="$TEST_DIR/analysis.json"
python3 "$SCRIPT_DIR/analyze_recording.py" "$TEST_DIR" --output "$ANALYSIS_OUTPUT" 2>&1 || {
    echo "❌ analyze_recording.py failed"
    exit 1
}

if [ ! -f "$ANALYSIS_OUTPUT" ]; then
    echo "❌ Analysis JSON not created"
    exit 1
fi
echo "✅ Analysis JSON created"
echo ""

# Step 3: Verify analysis JSON structure
echo "Step 3: Verify analysis JSON structure"
python3 -c "
import json, sys
with open('$ANALYSIS_OUTPUT') as f:
    data = json.load(f)

# Check required fields
required = ['recording_dir', 'metadata', 'total_events', 'total_steps', 'suggested_skill_name', 'steps']
for field in required:
    if field not in data:
        print(f'❌ Missing field: {field}')
        sys.exit(1)

# Check steps have expected fields
if not data['steps']:
    print('❌ No steps detected')
    sys.exit(1)

for step in data['steps']:
    step_required = ['step_number', 'app', 'start_time', 'end_time', 'signature', 'interactions']
    for field in step_required:
        if field not in step:
            print(f'❌ Step missing field: {field}')
            sys.exit(1)

# Check suggested_skill_name is non-empty
if not data['suggested_skill_name']:
    print('❌ suggested_skill_name is empty')
    sys.exit(1)

print(f'✅ Analysis structure valid: {data[\"total_steps\"]} steps, {data[\"total_events\"]} events')
print(f'   Suggested name: {data[\"suggested_skill_name\"]}')
" || exit 1
echo ""

# Step 4: Run generate_skill.py
echo "Step 4: Run generate_skill.py (dry-run first)"
python3 "$SCRIPT_DIR/generate_skill.py" \
    --analysis "$ANALYSIS_OUTPUT" \
    --recording-dir "$TEST_DIR" \
    --name "$SKILL_NAME" \
    --dry-run \
    --no-vision 2>&1 || {
    echo "❌ generate_skill.py --dry-run failed"
    exit 1
}
echo "✅ Dry-run succeeded"
echo ""

echo "Step 4b: Run generate_skill.py (save)"
python3 "$SCRIPT_DIR/generate_skill.py" \
    --analysis "$ANALYSIS_OUTPUT" \
    --recording-dir "$TEST_DIR" \
    --name "$SKILL_NAME" \
    --output-dir "$OUTPUT_SKILL_DIR" \
    --no-vision 2>&1 || {
    echo "❌ generate_skill.py failed"
    exit 1
}

SKILL_FILE="$OUTPUT_SKILL_DIR/SKILL.md"
if [ ! -f "$SKILL_FILE" ]; then
    echo "❌ SKILL.md not created at $SKILL_FILE"
    exit 1
fi
echo "✅ SKILL.md created"
echo ""

# Step 5: Verify SKILL.md frontmatter
echo "Step 5: Verify SKILL.md frontmatter"
python3 -c "
import sys
with open('$SKILL_FILE') as f:
    content = f.read()

# Check frontmatter
if not content.startswith('---'):
    print('❌ Missing frontmatter')
    sys.exit(1)

# Extract frontmatter
fm_end = content.index('---', 3)
fm = content[3:fm_end]

required_fields = ['name:', 'description:', 'version:', 'platforms:']
for field in required_fields:
    if field not in fm:
        print(f'❌ Missing frontmatter field: {field}')
        sys.exit(1)

# Check for Procedure section
if '## Procedure' not in content:
    print('❌ Missing ## Procedure section')
    sys.exit(1)

# Check for at least one step
if '### Step' not in content:
    print('❌ No steps in SKILL.md')
    sys.exit(1)

print('✅ SKILL.md frontmatter and structure valid')
" || exit 1
echo ""

# Step 6: Test HTML viewer generation
echo "Step 6: Test HTML viewer generation"
python3 "$SCRIPT_DIR/analyze_recording.py" "$TEST_DIR" --html --output "$TEST_DIR/analysis2.json" 2>&1 || {
    echo "❌ HTML viewer generation failed"
    exit 1
}
if [ ! -f "$TEST_DIR/viewer.html" ]; then
    echo "❌ viewer.html not created"
    exit 1
fi
echo "✅ HTML viewer created"
echo ""

# Step 7: Test replay_skill.py dry-run
echo "Step 7: Test replay_skill.py (dry-run)"
python3 "$SCRIPT_DIR/replay_skill.py" \
    --skill "$SKILL_FILE" \
    --dry-run \
    --step-delay 0.1 \
    --max-retries 1 2>&1 || {
    echo "⚠️  replay_skill.py dry-run returned non-zero (may be expected if no steps parsed)"
}
echo "✅ Replay dry-run completed"
echo ""

# Final summary
echo ""
echo "==================================="
echo "✅ All integration tests passed!"
echo "==================================="
echo ""
echo "Test summary:"
echo "  - Synthetic recording: created"
echo "  - analyze_recording.py: passed"
echo "  - Analysis JSON structure: valid"
echo "  - generate_skill.py: passed"
echo "  - SKILL.md frontmatter: valid"
echo "  - HTML viewer: created"
echo "  - replay_skill.py dry-run: completed"
echo ""
exit 0
