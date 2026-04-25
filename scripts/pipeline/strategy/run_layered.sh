#!/bin/bash
# ============================================================================
# Strategy Deep Research - 分層執行
# 
# 執行流程:
#   L1 (dev)     → 資料收集
#   L2 (analyst) → 分析共通點、勝率、加減碼
#   L3 (writer)  → 格式化輸出、發送 Discord
# ============================================================================

CACHE_DIR="/home/ubuntu/.openclaw/cache/strategy"
LOG_FILE="/tmp/pipeline-strategy.log"

echo "=============================================="
echo "📊 策略深度研究 - 分層執行"
echo "=============================================="

mkdir -p $CACHE_DIR

# L1: 資料收集 (dev)
echo ""
echo "【L1】資料收集 (dev)..."
python3 /home/ubuntu/.openclaw/scripts/pipeline/strategy/L1_collect.py >> $LOG_FILE 2>&1
if [ $? -eq 0 ]; then
    echo "✅ L1 完成"
else
    echo "❌ L1 失敗"
    exit 1
fi

# L2: 分析層 (analyst)
echo ""
echo "【L2】分析層 (analyst)..."
python3 /home/ubuntu/.openclaw/scripts/pipeline/strategy/L2_analyze.py >> $LOG_FILE 2>&1
if [ $? -eq 0 ]; then
    echo "✅ L2 完成"
else
    echo "❌ L2 失敗"
    exit 1
fi

# L3: 輸出層 (writer)
echo ""
echo "【L3】輸出層 (writer)..."
python3 /home/ubuntu/.openclaw/scripts/pipeline/strategy/L3_format.py >> $LOG_FILE 2>&1
if [ $? -eq 0 ]; then
    echo "✅ L3 完成"
else
    echo "❌ L3 失敗"
    exit 1
fi

echo ""
echo "=============================================="
echo "✅ 分層執行完成"
echo "=============================================="
echo ""
echo "產出檔案:"
ls -la $CACHE_DIR/