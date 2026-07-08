#!/bin/bash
# send-media-safe.sh v2 — MEDIA: 发送前强制路径校正 + 文件存在性验证 (JSON 输出)
# 
# 输入：任意文件路径
# 输出：JSON（stdout）: {"status":"success","media_path":"...","size_bytes":N} 或 {"status":"error","message":"..."}
# 退出码：0=成功，1=文件不存在，2=路径不在白名单
#
# v2 变更（2026-07-02）：JSON 输出替代文本 MEDIA: 行——供 media-delivery-gate 插件结构化消费
# 设计原则：脚本执法 > Agent 自觉

set -euo pipefail

MEDIA_DIR="${HOME}/media"
ALLOW_DIRS=("${HOME}/media" "${HOME}/local" "${HOME}/kb")

if [ $# -lt 1 ]; then
    echo '{"status":"error","message":"用法: send-media-safe.sh <文件路径>"}'
    exit 2
fi

INPUT="$1"
INPUT="${INPUT/#\~/$HOME}"

# 文件存在性验证
if [ ! -f "$INPUT" ]; then
    echo "{\"status\":\"error\",\"message\":\"文件不存在: $INPUT\"}"
    exit 1
fi

ABS_PATH="$(realpath "$INPUT")"
FILENAME="$(basename "$ABS_PATH")"
SIZE_BYTES="$(stat --format=%s "$ABS_PATH")"

# 检查是否已在白名单目录
for dir in "${ALLOW_DIRS[@]}"; do
    if [[ "$ABS_PATH" == "$dir"/* ]]; then
        echo "{\"status\":\"success\",\"media_path\":\"${ABS_PATH}\",\"size_bytes\":${SIZE_BYTES}}"
        exit 0
    fi
done

# 不在白名单 → cp 到 media
DEST="${MEDIA_DIR}/${FILENAME}"

if [ -f "$DEST" ]; then
    TIMESTAMP="$(date +%Y%m%d%H%M%S)"
    BASENAME="${FILENAME%.*}"
    EXT="${FILENAME##*.}"
    if [ "$BASENAME" = "$FILENAME" ]; then
        DEST="${MEDIA_DIR}/${FILENAME}_${TIMESTAMP}"
    else
        DEST="${MEDIA_DIR}/${BASENAME}_${TIMESTAMP}.${EXT}"
    fi
fi

cp "$ABS_PATH" "$DEST"
SIZE_BYTES="$(stat --format=%s "$DEST")"
echo "{\"status\":\"success\",\"media_path\":\"${DEST}\",\"size_bytes\":${SIZE_BYTES}}"
exit 0
