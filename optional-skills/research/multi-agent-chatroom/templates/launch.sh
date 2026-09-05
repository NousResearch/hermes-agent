#!/bin/bash
# Multi-Agent Chatroom for AI2050-OpenOne
# 一键启动全部 Agent
# 
# 配置: config.yaml
#   coding:     编程 Agent (执行研究任务)
#   reviewers:  审核 Agent 列表 (评审+综合)
#   supervisor: 调度 Agent (任务管理)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Parse config to show active models
python -c "
import yaml
c = yaml.safe_load(open('config.yaml'))
coding = c.get('coding', {})
reviewers = c.get('reviewers', [])
sup = c.get('supervisor', {})
print(f'编程: {coding.get(\"name\")} ({coding.get(\"provider\")}/{coding.get(\"model\")})')
for r in reviewers:
    print(f'审核: {r.get(\"name\")} ({r.get(\"provider\")}/{r.get(\"model\")}) [{r.get(\"role\")}]')
print(f'调度: {sup.get(\"name\")} ({sup.get(\"provider\")}/{sup.get(\"model\")})')
" 2>/dev/null || echo "⚠️  config.yaml parse warning"

echo ""

# Check Python deps
python -c "import fastapi, websockets, httpx, yaml" 2>/dev/null || {
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt -q
}

# Create workspace if needed
WORKDIR=$(python -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(c.get('project',{}).get('workdir','../Ai2050-OpenOne'))")
WORKDIR=$(realpath "$WORKDIR" 2>/dev/null || echo "$WORKDIR")
if [ ! -d "$WORKDIR" ]; then
    echo "⚠️  Workdir $WORKDIR not found."
    echo "   Clone your research repository first, then update config.yaml"
    echo ""
    echo "   Or edit config.yaml → project.workdir to point to your project"
    exit 1
fi

echo "📁 Workdir: $WORKDIR"

# Start server
echo ""
echo "🚀 Starting chatroom server..."
python cli/server.py &
SERVER_PID=$!
sleep 2

# Check server health
if curl -s http://localhost:8765/health > /dev/null 2>&1; then
    echo "✅ Server running on ws://localhost:8765"
else
    echo "❌ Server failed to start"
    exit 1
fi

# Start supervisor
echo "👔 Starting supervisor..."
python cli/supervisor.py &
SUPERVISOR_PID=$!
sleep 1

# Start coding agent
echo "🤖 Starting coding agent..."
python cli/deepseek.py &
CODING_PID=$!
sleep 1

# Start all reviewer agents
echo "🔍 Starting reviewer agents..."
python cli/gpt_reviewer.py &
REVIEWER1_PID=$!
sleep 1

python cli/claude_reviewer.py &
REVIEWER2_PID=$!

echo ""
echo "═══════════════════════════════════"
echo "  All agents running!"
echo "  Server:  ws://localhost:8765"
echo "  PIDs:    server=$SERVER_PID sup=$SUPERVISOR_PID coding=$CODING_PID reviewer1=$REVIEWER1_PID reviewer2=$REVIEWER2_PID"
echo "  Press Ctrl+C to stop all"
echo "═══════════════════════════════════"
echo ""

# Cleanup on exit
trap "echo 'Shutting down...'; kill $SERVER_PID $SUPERVISOR_PID $CODING_PID $REVIEWER1_PID $REVIEWER2_PID 2>/dev/null; echo 'Done.'; exit 0" INT TERM

wait
