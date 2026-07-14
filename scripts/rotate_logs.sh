#!/bin/bash
# Hermes 日志轮转脚本 - 手动清理大日志文件

LOG_DIR="/c/Users/1/AppData/Local/hermes/logs"

echo "🗑️  开始清理 Hermes 日志文件..."

# 备份并清空大于 1MB 的日志文件
for logfile in "$LOG_DIR"/*.log; do
    if [ -f "$logfile" ]; then
        size=$(stat -c%s "$logfile" 2>/dev/null || echo 0)
        if [ "$size" -gt 1048576 ]; then  # 大于 1MB
            echo "  📦 备份并清空: $(basename "$logfile") ($(numfmt --to=iec-i --suffix=B $size 2>/dev/null || echo $size bytes))"
            mv "$logfile" "$logfile.$(date +%Y%m%d_%H%M%S)"
            touch "$logfile"
        fi
    fi
done

# 删除 7 天前的 .log.YYYYMMDD_HHMMSS 备份文件
find "$LOG_DIR" -name "*.log.*" -type f -mtime +7 -delete 2>/dev/null

echo "✅ 日志轮转完成"
ls -lh "$LOG_DIR"/*.log 2>/dev/null | head -10
