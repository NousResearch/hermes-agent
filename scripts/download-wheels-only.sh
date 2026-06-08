#!/usr/bin/env bash
# 仅补充下载 Python wheels（跳过二进制、源码等）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${1:-$PROJECT_ROOT/hermes-offline-bundle}"
WHEEL_DIR="$OUTPUT_DIR/python-wheels"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

mkdir -p "$WHEEL_DIR"

echo "==> 导出依赖列表 ..."
cd "$PROJECT_ROOT"
REQ_FILE="$OUTPUT_DIR/requirements.txt"
uv export --extra all --no-hashes -o "$REQ_FILE" 2>/dev/null
# 移除 -e . (editable install，不能和 --target 一起用)
grep -v '^\-e \.' "$REQ_FILE" > "$REQ_FILE.tmp" && mv "$REQ_FILE.tmp" "$REQ_FILE"
echo "    导出 $(grep -c '==' "$REQ_FILE") 个包"

echo "==> 下载 [all] extras wheels ..."
uv pip install \
    --python-version "$PYTHON_VERSION" \
    --only-binary :all: \
    --target "$WHEEL_DIR" \
    -r "$REQ_FILE" 2>&1 | tail -5 || {
    echo "    重试不限制二进制 ..."
    uv pip install \
        --python-version "$PYTHON_VERSION" \
        --target "$WHEEL_DIR" \
        -r "$REQ_FILE" 2>&1 | tail -5 || true
}

echo "==> 下载 lazy deps wheels ..."
LAZY_REQ="$OUTPUT_DIR/requirements-lazy.txt"
cat > "$LAZY_REQ" << 'EOF'
anthropic==0.87.0
boto3==1.42.89
azure-identity==1.25.3
exa-py==2.10.2
firecrawl-py==4.17.0
parallel-web==0.4.2
mistralai==2.4.8
edge-tts==7.2.7
elevenlabs==1.59.0
faster-whisper==1.2.1
sounddevice==0.5.5
numpy==2.4.3
fal-client==0.13.1
honcho-ai==2.0.1
hindsight-client==0.6.1
python-telegram-bot[webhooks]==22.6
discord.py[voice]==2.7.1
brotlicffi==1.2.0.1
slack-bolt==1.27.0
slack-sdk==3.40.1
aiohttp==3.13.4
mautrix[encryption]==0.21.0
Markdown==3.10.2
aiosqlite==0.22.1
asyncpg==0.31.0
aiohttp-socks==0.11.0
dingtalk-stream==0.24.3
alibabacloud-dingtalk==2.2.42
qrcode==7.4.2
lark-oapi==1.5.3
defusedxml==0.7.1
modal==1.3.4
daytona==0.155.0
google-api-python-client==2.194.0
google-auth-oauthlib==1.3.1
google-auth-httplib2==0.3.1
youtube-transcript-api==1.2.4
agent-client-protocol==0.9.0
EOF

uv pip install \
    --python-version "$PYTHON_VERSION" \
    --only-binary :all: \
    --target "$WHEEL_DIR" \
    -r "$LAZY_REQ" 2>&1 | tail -5 || {
    echo "    重试不限制二进制 ..."
    uv pip install \
        --python-version "$PYTHON_VERSION" \
        --target "$WHEEL_DIR" \
        -r "$LAZY_REQ" 2>&1 | tail -5 || true
}

WHEEL_COUNT=$(find "$WHEEL_DIR" -name "*.whl" 2>/dev/null | wc -l)
WHEEL_SIZE=$(du -sm "$WHEEL_DIR" 2>/dev/null | cut -f1)
echo ""
echo "==> 完成: $WHEEL_COUNT 个 wheel 包, ${WHEEL_SIZE} MB"
echo "    位置: $WHEEL_DIR"
