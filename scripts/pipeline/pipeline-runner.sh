#!/bin/bash
# ============================================================================
# Pipeline Runner - 統一 Pipeline 執行器
# 用法: ./pipeline-runner.sh <pipeline> [layer]
# ============================================================================

set -e

BASE="/home/ubuntu/.openclaw/scripts/pipeline"
LOG_DIR="/tmp/pipeline-logs"
mkdir -p "$LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

run_pipeline() {
    local pipeline="$1"
    local layer="${2:-all}"
    local log_file="$LOG_DIR/${pipeline}-$(date +%Y%m%d-%H%M).log"
    
    log "=== $pipeline Pipeline (layer=$layer) ==="
    
    case "$pipeline" in
        "stock")
            case "$layer" in
                "1"|"fetch") bash "$BASE/stock/fetch.sh" >> "$log_file" 2>&1 ;;
                "2"|"alert") bash "$BASE/stock/alert.sh" >> "$log_file" 2>&1 ;;
                "3"|"format") bash "$BASE/stock/format.sh" >> "$log_file" 2>&1 ;;
                *) bash "$BASE/stock/stock-pipeline.sh" >> "$log_file" 2>&1 ;;
            esac
            ;;
        "news")
            case "$layer" in
                "1"|"fetch") bash "$BASE/news/fetch.py" >> "$log_file" 2>&1 ;;
                "2"|"classify") bash "$BASE/news/classify.sh" >> "$log_file" 2>&1 ;;
                "3"|"format") bash "$BASE/news/format.sh" >> "$log_file" 2>&1 ;;
                *) 
                    bash "$BASE/news/fetch.py" >> "$log_file" 2>&1
                    bash "$BASE/news/classify.sh" >> "$log_file" 2>&1
                    bash "$BASE/news/format.sh" >> "$log_file" 2>&1
                    ;;
            esac
            ;;
        "morning")
            bash "$BASE/morning/fetch-all.sh" >> "$log_file" 2>&1
            bash "$BASE/morning/format.sh" >> "$log_file" 2>&1
            ;;
        "task")
            bash "$BASE/task/fetch.sh" >> "$log_file" 2>&1
            bash "$BASE/task/analyze.sh" >> "$log_file" 2>&1
            bash "$BASE/task/format.sh" >> "$log_file" 2>&1
            ;;
        "memory")
            bash "$BASE/memory/fetch.sh" >> "$log_file" 2>&1
            bash "$BASE/memory/distill.sh" >> "$log_file" 2>&1
            ;;
        "research")
            bash "$BASE/research/fetch.py" >> "$log_file" 2>&1
            bash "$BASE/research/format.sh" >> "$log_file" 2>&1
            ;;
        "analysis")
            bash "$BASE/analysis/run.sh" >> "$log_file" 2>&1
            ;;
        "health")
            bash "$BASE/health/dashboard.sh" >> "$log_file" 2>&1
            bash "$BASE/health/auto-heal.sh" >> "$log_file" 2>&1
            ;;
        "all")
            # 執行所有 essentials
            bash "$BASE/morning/fetch-all.sh" >> "$LOG_DIR/morning-fetch.log" 2>&1
            bash "$BASE/morning/format.sh" >> "$LOG_DIR/morning-format.log" 2>&1
            bash "$BASE/task/fetch.sh" >> "$LOG_DIR/task-fetch.log" 2>&1
            bash "$BASE/task/analyze.sh" >> "$LOG_DIR/task-analyze.log" 2>&1
            bash "$BASE/task/format.sh" >> "$LOG_DIR/task-format.log" 2>&1
            bash "$BASE/health/dashboard.sh" >> "$LOG_DIR/dashboard.log" 2>&1
            bash "$BASE/health/auto-heal.sh" >> "$LOG_DIR/auto-heal.log" 2>&1
            ;;
        *)
            log "未知 Pipeline: $pipeline"
            return 1
            ;;
    esac
    
    log "=== $pipeline 完成 ==="
}

# 主流程
case "${1:-help}" in
    "help"|"-h")
        echo "Pipeline Runner"
        echo "=============="
        echo ""
        echo "用法:"
        echo "  $0 <pipeline> [layer]"
        echo ""
        echo "可用 Pipeline:"
        echo "  stock, news, morning, task, memory, research, analysis, health, all"
        echo ""
        echo "Layer:"
        echo "  1/fetch, 2/alert, 3/format, all (預設)"
        ;;
    *)
        run_pipeline "$1" "$2"
        ;;
esac
