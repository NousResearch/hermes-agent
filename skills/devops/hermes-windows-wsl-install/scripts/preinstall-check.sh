#!/usr/bin/env bash
# preinstall-check.sh - WSL 端 pre-install 状态一键检查（7 探针）
# 用法: bash preinstall-check.sh
# 输出: 7 探针 PASS/WARN/FAIL + 修法

set -u

PASS=0
WARN=0
FAIL=0

# 颜色
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

check_pass() { echo -e "${GREEN}[PASS]${NC} $1"; PASS=$((PASS+1)); }
check_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; WARN=$((WARN+1)); }
check_fail() { echo -e "${RED}[FAIL]${NC} $1"; FAIL=$((FAIL+1)); }

echo "=== Hermes Pre-Install 状态检查（WSL 端）==="
echo ""

# 探针 1: pip 清华镜像
echo -n "1. pip 清华镜像 ... "
if grep -q "pypi.tuna.tsinghua.edu.cn" /etc/pip.conf 2>/dev/null; then
    check_pass "/etc/pip.conf 已配清华"
elif grep -q "pypi.tuna.tsinghua.edu.cn" ~/.pip/pip.conf 2>/dev/null; then
    check_pass "~/.pip/pip.conf 已配清华"
else
    check_fail "未配 pip 清华镜像 — 跑: sudo tee /etc/pip.conf <<< 'index-url=https://pypi.tuna.tsinghua.edu.cn/simple'"
fi

# 探针 2: apt 阿里镜像
echo -n "2. apt 阿里镜像 ... "
if grep -q "mirrors.aliyun.com" /etc/apt/sources.list.d/*.sources 2>/dev/null; then
    check_pass "/etc/apt/sources.list.d/ 已配阿里"
else
    check_warn "未配 apt 阿里镜像 — 跑: sudo sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/ubuntu.sources"
fi

# 探针 3: gh-proxy.com 可达
echo -n "3. gh-proxy.com 可达 ... "
if curl -s -o /dev/null -w "%{http_code}" --max-time 5 https://gh-proxy.com/https://raw.githubusercontent.com/NousResearch/hermes-agent/main/README.md | grep -q 200; then
    check_pass "GitHub 镜像可达"
else
    check_fail "gh-proxy.com 不可达 — 公司代理可能 block 全部 github 镜像"
fi

# 探针 4: raw.githubusercontent.com（不依赖镜像）
echo -n "4. raw.githubusercontent.com 不可达 ... "
if ! curl -s -o /dev/null -w "%{http_code}" --max-time 5 https://raw.githubusercontent.com/NousResearch/hermes-agent/main/README.md 2>&1 | grep -q 200; then
    check_warn "raw.githubusercontent.com GFW block — 必走 gh-proxy / gitee"
else
    check_pass "raw.githubusercontent.com 可达（罕见，可能没 GFW）"
fi

# 探针 5: WSL 2 + localhostForwarding
echo -n "5. WSL 2 + localhostForwarding ... "
if grep -q "localhostForwarding=true" /etc/wsl.conf 2>/dev/null; then
    check_pass "/etc/wsl.conf 已配 localhostForwarding=true"
else
    check_warn "未显式配 localhostForwarding（WSL 2 默认 true）"
fi

# 探针 6: 端口 9119 未占用
echo -n "6. 端口 9119 未占用 ... "
if ! ss -tlnp 2>/dev/null | grep -q ":9119"; then
    check_pass "9119 端口空闲"
else
    check_warn "9119 已被占用 — 杀旧: pkill -f 'dashboard.*9119'"
fi

# 探针 7: hermes 路径
echo -n "7. hermes CLI 路径 ... "
HERMES_PATH=$(which hermes 2>/dev/null)
if [ -n "$HERMES_PATH" ]; then
    check_pass "hermes 在 $HERMES_PATH"
else
    check_fail "hermes 不在 PATH — 跑: pip install -e ~/.hermes/hermes-agent"
fi

echo ""
echo "=== 总结 ==="
echo "  PASS: $PASS / 7"
echo "  WARN: $WARN / 7"
echo "  FAIL: $FAIL / 7"
echo ""

if [ $FAIL -gt 0 ]; then
    echo "❌ 有 $FAIL 项必修复，按上面 [FAIL] 行的修法跑"
    exit 1
fi

if [ $WARN -gt 0 ]; then
    echo "⚠️  有 $WARN 项建议优化，不影响安装"
    exit 0
fi

echo "✅ 7/7 PASS，pre-install 环境完美！"
echo ""
echo "下一步："
echo "  1. 看 references/preinstall-sop.md 走 10 步安装"
echo "  2. 跑 scripts/setup-shared-memory.sh 配置共同一个记忆"
exit 0
