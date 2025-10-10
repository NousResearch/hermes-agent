#!/bin/bash
#
# Test Script for Mock Web Tools & WebSocket Reconnection
#
# This script tests:
# 1. Mock web tools (no API calls, fake data)
# 2. WebSocket timeout/reconnection during long operations
# 3. Complete logging capture
#
# Perfect for development/testing without wasting API credits!

set -e

cd "$(dirname "$0")"

echo "=========================================="
echo "🧪 Mock Mode Test Script"
echo "=========================================="
echo ""

# Check if logging server is running
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "⚠️  Logging server not detected!"
    echo "   Starting logging server in background..."
    python api_endpoint/logging_server.py &
    SERVER_PID=$!
    echo "   Server PID: $SERVER_PID"
    sleep 3
else
    echo "✅ Logging server already running"
    SERVER_PID=""
fi

echo ""
echo "📋 Test Configuration:"
echo "   - Mock web tools: ENABLED"
echo "   - Mock delay: 60 seconds (triggers WebSocket timeout)"
echo "   - WebSocket logging: ENABLED"
echo "   - Expected behavior: Connection timeout + auto-reconnect"
echo ""
echo "🔄 Running agent with mock mode..."
echo "   (This will take ~60 seconds to test reconnection)"
echo ""

# Run agent with mock mode
python run_agent.py \
  --enabled_toolsets web \
  --enable_websocket_logging \
  --mock_web_tools \
  --mock_delay 60 \
  --query "Find publicly traded water companies benefiting from AI data centers"

echo ""
echo "=========================================="
echo "✅ Test Complete!"
echo "=========================================="
echo ""

# Find most recent log file
LATEST_LOG=$(ls -t api_endpoint/logs/realtime/session_*.json 2>/dev/null | head -1)

if [ -n "$LATEST_LOG" ]; then
    echo "📊 Log Analysis:"
    echo "   File: $LATEST_LOG"
    echo ""
    
    # Count events
    echo "   Event Counts:"
    python3 -c "
import json
import sys

with open('$LATEST_LOG') as f:
    data = json.load(f)
    events = data.get('events', [])
    
    # Count by type
    counts = {}
    for e in events:
        etype = e.get('type', 'unknown')
        counts[etype] = counts.get(etype, 0) + 1
    
    for etype, count in sorted(counts.items()):
        print(f'     - {etype}: {count}')
    
    # Check completeness
    has_complete = any(e.get('type') == 'complete' for e in events)
    print()
    if has_complete:
        print('   ✅ Session completed successfully!')
    else:
        print('   ⚠️  Session incomplete (may have been interrupted)')
    
    # Check for reconnections
    tool_results = [e for e in events if e.get('type') == 'tool_result']
    tool_calls = [e for e in events if e.get('type') == 'tool_call']
    
    if len(tool_results) == len(tool_calls):
        print('   ✅ All tool calls have results (no missing events)')
    else:
        print(f'   ⚠️  Tool calls: {len(tool_calls)}, Results: {len(tool_results)}')
"
else
    echo "⚠️  No log files found"
fi

# Cleanup
if [ -n "$SERVER_PID" ]; then
    echo ""
    echo "🛑 Stopping logging server (PID: $SERVER_PID)..."
    kill $SERVER_PID 2>/dev/null || true
fi

echo ""
echo "💡 Key Observations to Look For:"
echo "   1. '[MOCK]' prefix on tool execution messages"
echo "   2. '🔄 Reconnecting to logging server' after long tool"
echo "   3. '✅ Reconnected successfully!' confirmation"
echo "   4. Complete log file with all events captured"
echo ""
echo "🎉 Mock mode test completed!"

