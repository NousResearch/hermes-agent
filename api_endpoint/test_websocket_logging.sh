#!/bin/bash
# Test script for WebSocket logging system
#
# This script demonstrates the complete WebSocket logging workflow:
# 1. Starts the logging server
# 2. Runs the agent with WebSocket logging enabled
# 3. Shows the logged data
#
# Usage: ./test_websocket_logging.sh

set -e  # Exit on error

echo "🧪 Testing WebSocket Logging System"
echo "===================================="
echo ""

# Check if required packages are installed
echo "📦 Checking dependencies..."
python -c "import fastapi; import uvicorn; import websockets" 2>/dev/null || {
    echo "❌ Missing dependencies. Installing..."
    pip install fastapi uvicorn websockets
}
echo "✅ Dependencies OK"
echo ""

# Start the logging server in the background
echo "🚀 Starting logging server..."
python api_endpoint/logging_server.py --port 8000 &
SERVER_PID=$!

# Give server time to start
sleep 2

# Check if server is running
if ps -p $SERVER_PID > /dev/null; then
    echo "✅ Logging server started (PID: $SERVER_PID)"
else
    echo "❌ Failed to start logging server"
    exit 1
fi

echo ""
echo "🤖 Running agent with WebSocket logging..."
echo ""

# Run the agent with WebSocket logging
python run_agent.py \
    --enabled_toolsets web \
    --enable_websocket_logging \
    --query "What are the top 3 programming languages in 2025?" \
    --max_turns 5

echo ""
echo "✅ Agent execution complete!"
echo ""

# Show the most recent log file
echo "📊 Viewing logged session data..."
echo ""

LATEST_LOG=$(ls -t logs/realtime/session_*.json 2>/dev/null | head -1)

if [ -f "$LATEST_LOG" ]; then
    echo "📄 Log file: $LATEST_LOG"
    echo ""
    
    # Pretty print the JSON if jq is available
    if command -v jq &> /dev/null; then
        echo "Event summary:"
        jq '.events[] | {type: .type, timestamp: .timestamp}' "$LATEST_LOG"
        echo ""
        echo "Total events: $(jq '.events | length' "$LATEST_LOG")"
    else
        echo "Content (install 'jq' for pretty printing):"
        cat "$LATEST_LOG"
    fi
else
    echo "⚠️  No log files found in logs/realtime/"
fi

echo ""
echo "🛑 Stopping logging server..."
kill $SERVER_PID 2>/dev/null || true

echo "✅ Test complete!"
echo ""
echo "Next steps:"
echo "  1. Start server: python api_endpoint/logging_server.py"
echo "  2. Run agent: python run_agent.py --enable_websocket_logging --query \"...\""
echo "  3. View logs: http://localhost:8000/sessions"

