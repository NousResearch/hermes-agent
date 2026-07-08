#!/bin/bash
# security_baseline.sh v3 — HOST_MAIN 安全基线审计（P1-1）
# v3: +cron凭据扫描(#17) +端口合并汇报(#22)
# v2: 管道子shell修复 + user_profile/.env + IPv6 + config regex + kb-backup权限
# 用法: bash security_baseline.sh
set -euo pipefail

PASS=0; FAIL=0; INFO=0
GREEN='\033[32m'; RED='\033[31m'; YELLOW='\033[33m'; NC='\033[0m'

pass() { PASS=$((PASS+1)); echo -e "${GREEN}[PASS]${NC} $1"; }
fail() { FAIL=$((FAIL+1)); echo -e "${RED}[FAIL]${NC} $1"; }
info() { INFO=$((INFO+1)); echo -e "${YELLOW}[INFO]${NC} $1"; }

echo "============================================="
echo " HOST_MAIN 安全基线审计 v3 — $(date +%Y-%m-%dT%H:%M:%S)"
echo "============================================="

# ═══════════════════════════════════════════════
# 维度 1: 文件权限基线
# ═══════════════════════════════════════════════
echo ""
echo "── 维度1: 文件权限基线 ──"

declare -A PERM_CHECKS=(
    ["$HOME/.hermes/credentials/.key"]="600"
    ["$HOME/.hermes/.provider_b_key"]="600"
    ["$HOME/.hermes/.env"]="600"
    ["$HOME/.hermes/profiles/profile_B/.env"]="600"
    ["$HOME/hermes-data/user_profile/.env"]="600"
    ["$HOME/hermes-data/user_profile/.provider_b_key"]="600"
    ["$HOME/.hermes/memories/SOUL.md"]="644"
    ["$HOME/.hermes/memories/MEMORY.md"]="644"
    ["$HOME/.hermes/config.yaml"]="644"
    ["$HOME/.hermes/weixin/accounts/ACCT_A@platform.bot.context-tokens.json"]="600"
    ["$HOME/.hermes/weixin/accounts/ACCT_B@platform.bot.context-tokens.json"]="600"
    ["$HOME/.config/LarkShell-XXXXX/logout_token"]="600"
    ["$HOME/external-app/data/default-user/secrets.json"]="600"
    ["$HOME/external-app/data/cookie-secret.txt"]="600"
)

for path in "${!PERM_CHECKS[@]}"; do
    max_perm="${PERM_CHECKS[$path]}"
    if [ -f "$path" ]; then
        actual=$(stat -c '%a' "$path" 2>/dev/null || echo "MISSING")
        if [ "$actual" -le "$max_perm" ] 2>/dev/null; then
            pass "$path = $actual"
        else
            fail "$path = $actual (期望 ≤$max_perm)"
        fi
    fi
done

# 额外扫描凭据文件（排除代码/文档）
echo "  …额外凭据文件扫描…"
while read -r perm path; do
    [ -z "$perm" ] && continue
    if [ "$perm" != "600" ] && [ "$perm" != "400" ]; then
        fail "$path = $perm (应 600/400)"
    fi
done < <(find "$HOME" -maxdepth 4 -type f \
    \( -name '*key*' -o -name '*token*' -o -name '*secret*' -o -name '.provider_b_key' -o -name '*.pem' \) \
    ! -path '*/.cache/*' ! -path '*/node_modules/*' ! -path '*/venv/*' ! -path '*/.git/*' \
    ! -path '*/external-app/*' ! -path '*/hermes-agent/*' ! -path '*/<media-dir>/*' \
    ! -path '*/kb/*' ! -path '*/local-backup/*' \
    ! -name '*.md' ! -name '*.py' ! -name '*.js' ! -name '*.json' ! -name '*.css' ! -name '*.txt' \
    ! -name '_check_output_keywords.sh' ! -path '*/.fluxbox/*' \
    -exec stat -c '%a %n' {} \; 2>/dev/null)

# 目录权限检查
for dir in "$HOME/.hermes/credentials" "$HOME/local-backup" "$HOME/hermes-data/user_profile"; do
    if [ -d "$dir" ]; then
        perm=$(stat -c '%a' "$dir")
        if [ "$perm" -le 700 ] 2>/dev/null; then
            pass "$dir/ = $perm"
        else
            fail "$dir/ = $perm (期望 ≤700)"
        fi
    fi
done

# ═══════════════════════════════════════════════
# 维度 2: 网络暴露基线（含 IPv6）
# ═══════════════════════════════════════════════
echo ""
echo "── 维度2: 网络暴露基线 ──"

info "监听端口（0.0.0.0 + [::] + *）:"
UNKNOWN_PORTS=()
while read -r line; do
    port=$(echo "$line" | awk '{print $4}' | rev | cut -d: -f1 | rev)
    case "$port" in
        22)   info "  :$port (SSH)" ;;
        53)   info "  :$port (DNS stub)" ;;
        631)  info "  :$port (CUPS)" ;;
        111)  info "  :$port (rpcbind)" ;;
        3389) fail "  :$port (xrdp — 应限制到 LAN)" ;;
        3493) info "  :$port (localhost)" ;;
        8643) fail "  :$port (api_server — 应限制到 ZT)" ;;
        8000) info "  :$port (node dev)" ;;
        8001) info "  :$port (未知)" ;;
        9993) info "  :$port (ZeroTier)" ;;
        60938) info "  :$port (remote-access)" ;;
        *)    UNKNOWN_PORTS+=("$port") ;;
    esac
done < <(ss -tlnp 2>/dev/null | grep -E '0\.0\.0\.0|\[::\]|\*')

# 未识别端口合并汇报（#22 修复——不单独 FAIL）
if [ ${#UNKNOWN_PORTS[@]} -gt 0 ]; then
    info "未识别端口: ${UNKNOWN_PORTS[*]}（请确认是否需要加入白名单或收紧）"
fi

# UFW 规则
info "UFW 规则:"
while read -r line; do
    rule=$(echo "$line" | awk '{print $1}')
    case "$rule" in
        22/tcp)   info "  $line (SSH)" ;;
        8643/tcp) fail "  $line (api_server → 应限制到 ZT)" ;;
        8642/tcp) fail "  $line (旧 api_server 端口 → 应删除)" ;;
        3389/tcp) fail "  $line (xrdp → 应限制到 LAN)" ;;
        9993/udp) info "  $line (ZT)" ;;
        *)        fail "  $line (未预期)" ;;
    esac
done < <(sudo ufw status verbose 2>/dev/null | grep 'ALLOW IN.*Anywhere')

# UFW 默认策略
def_policy=$(sudo ufw status verbose 2>/dev/null | grep '^Default:')
if echo "$def_policy" | grep -q 'deny (incoming)'; then
    pass "UFW default: deny incoming"
else
    fail "UFW default: $def_policy"
fi

# ═══════════════════════════════════════════════
# 维度 3: 凭据存储基线
# ═══════════════════════════════════════════════
echo ""
echo "── 维度3: 凭据存储基线 ──"

info "systemd unit 密钥扫描:"
for unit in hermes-gateway hermes-gateway-profile_B; do
    if systemctl cat "$unit" 2>/dev/null | grep -qiE 'API_KEY|SECRET|TOKEN|PASSWORD|PASSWD'; then
        matches=$(systemctl cat "$unit" 2>/dev/null | grep -iE 'API_KEY|SECRET|TOKEN|PASSWORD|PASSWD')
        fail "$unit.service 含疑似硬编码密钥"
    else
        pass "$unit.service: 无硬编码密钥"
    fi
done

info "session 文件密钥残留:"
count=$(grep -rl 'API_SERVER_KEY\|43a47e096d5bd9' "$HOME/.hermes/sessions/" 2>/dev/null | wc -l)
if [ "$count" -eq 0 ]; then
    pass "session JSON: 无 API_SERVER_KEY 残留"
else
    fail "session JSON: $count 个文件含 API_SERVER_KEY（历史残留，新 session 不再增长）"
fi

info "config.yaml 密钥扫描:"
for conf in "$HOME/.hermes/config.yaml" "$HOME/.hermes/profiles/profile_B/config.yaml"; do
    if [ -f "$conf" ]; then
        if grep -qE "api_key:\s*['\"]?[a-zA-Z0-9_-]{20,}" "$conf" 2>/dev/null; then
            fail "$conf: 疑似含明文 api_key（20+ 字符）"
        else
            pass "$conf: 无明文密钥"
        fi
    fi
done

# ═══════════════════════════════════════════════
# 维度 3e: cron jobs.json 凭据扫描（#17）
# ═══════════════════════════════════════════════

info "cron jobs.json 凭据扫描:"
# 扫描所有 profile 的 jobs.json
KEY_REGEX='["'"'"']?api_key["'"'"']?\s*[:=]\s*['"'"'"]?[a-zA-Z0-9_-]{20,}'
HAS_LEAK=0
for jobs_file in "$HOME/.hermes/cron/jobs.json" "$HOME/.hermes/profiles/"*/cron/jobs.json; do
    [ -f "$jobs_file" ] || continue
    if command -v jq &>/dev/null; then
        while read -r id desc; do
            # 扫描 prompt 和 script 字段
            prompt=$(jq -r ".[] | select(.id==\"$id\" or .job_id==\"$id\") | .prompt // empty" "$jobs_file" 2>/dev/null)
            script=$(jq -r ".[] | select(.id==\"$id\" or .job_id==\"$id\") | .script // empty" "$jobs_file" 2>/dev/null)
            if echo "$prompt$script" | grep -qiE "$KEY_REGEX"; then
                fail "cron $id ($desc): prompt/script 含疑似 api_key"
                HAS_LEAK=1
            fi
        done < <(jq -r '.[] | "\(.id // .job_id) \(.name // "unnamed")"' "$jobs_file" 2>/dev/null)
    else
        # 降级：grep 全文
        if grep -qiE "$KEY_REGEX" "$jobs_file" 2>/dev/null; then
            fail "$jobs_file: 含疑似 api_key"
            HAS_LEAK=1
        fi
    fi
done
if [ "$HAS_LEAK" -eq 0 ]; then
    pass "所有 cron jobs: 无硬编码凭据"
fi


# ═══════════════════════════════════════════════
# 维度 4: sudo 权限基线
# ═══════════════════════════════════════════════
echo ""
echo "── 维度4: sudo 权限基线 ──"

info "NOPASSWD 条目:"
while read -r line; do
    if echo "$line" | grep -q 'NOPASSWD: ALL'; then
        fail "sudo NOPASSWD: ALL — $(echo $line | head -c 80)"
    else
        pass "$(echo $line | head -c 80)"
    fi
done < <(sudo grep -rh 'NOPASSWD' /etc/sudoers /etc/sudoers.d/ 2>/dev/null)

for f in /etc/sudoers.d/*; do
    name=$(basename "$f")
    case "$name" in
        README) ;;
        USER|hermes-fix-permissions) pass "sudoers.d/$name: 已知" ;;
        *) fail "sudoers.d/$name: 未预期的遗留文件" ;;
    esac
done

# ═══════════════════════════════════════════════
# 维度 5: systemd 安全加固
# ═══════════════════════════════════════════════
echo ""
echo "── 维度5: systemd 安全加固 ──"

for unit in hermes-gateway hermes-gateway-profile_B; do
    info "$unit.service:"
    content=$(systemctl cat "$unit" 2>/dev/null)
    for directive in ProtectHome ProtectSystem NoNewPrivileges PrivateTmp ReadOnlyPaths; do
        if echo "$content" | grep -q "^$directive="; then
            pass "  $unit: $directive 已设置"
        else
            fail "  $unit: 缺少 $directive"
        fi
    done
done

# ═══════════════════════════════════════════════
# 汇总
# ═══════════════════════════════════════════════
echo ""
echo "============================================="
echo " 结果: ${GREEN}$PASS PASS${NC} | ${RED}$FAIL FAIL${NC} | ${YELLOW}$INFO INFO${NC}"
echo "============================================="
[ "$FAIL" -eq 0 ] && echo "✅ 所有安全基线检查通过" || echo "❌ 存在 $FAIL 项不合规"
