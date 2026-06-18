#!/bin/bash
# Hermes Agent 每日健康检查 (v3 — host-side, 64 only)
# 报告本机(64) + 公网端点状态。在 host 上跑,通过 docker exec 发飞书。
# 风险阈值: disk>85% 红色 / >75% 黄色, mem<300M 红色, 服务 down 红色
set -u

LOG="/home/admin/.hermes/logs/healthcheck.log"
ALERT="/home/admin/.hermes/logs/healthcheck_alert.txt"
mkdir -p "$(dirname "$LOG")"
TS=$(date '+%Y-%m-%d %H:%M:%S')
ALERTS=()

# 颜色
R='\033[0;31m'; Y='\033[1;33m'; G='\033[0;32m'; N='\033[0m'
ok()   { printf "  ${G}✓${N} %s\n" "$1"; }
warn() { printf "  ${Y}⚠${N} %s\n" "$1"; ALERTS+=("$1"); }
err()  { printf "  ${R}✗${N} %s\n" "$1"; ALERTS+=("$1"); }

echo "════════════════════════════════════════" | tee -a "$LOG"
echo "  Hermes Health Check — $TS"            | tee -a "$LOG"
echo "════════════════════════════════════════" | tee -a "$LOG"

# ─── 1. 本机 (64.176.42.24 — 主机) ──────────────
echo "" | tee -a "$LOG"
echo "▸ 本机 64.176.42.24 (Vultr Osaka)" | tee -a "$LOG"

# 磁盘
USEPCT=$(df / | awk 'NR==2 {gsub("%",""); print $5}')
AVAIL=$(df -h / | awk 'NR==2 {print $4}')
if   [ "$USEPCT" -ge 85 ]; then err  "磁盘 ${USEPCT}% (剩 ${AVAIL}) — 严重"
elif [ "$USEPCT" -ge 75 ]; then warn "磁盘 ${USEPCT}% (剩 ${AVAIL}) — 关注"
else                              ok   "磁盘 ${USEPCT}% (剩 ${AVAIL})"
fi

# 内存
MEM_AVAIL_MB=$(free -m | awk '/^Mem:/ {print $7}')
if   [ "$MEM_AVAIL_MB" -lt 300 ]; then err  "内存仅剩 ${MEM_AVAIL_MB}MB"
elif [ "$MEM_AVAIL_MB" -lt 800 ]; then warn "内存剩 ${MEM_AVAIL_MB}MB"
else                                    ok   "内存剩 ${MEM_AVAIL_MB}MB"
fi

# Load
LOAD=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | tr -d ',')
CORES=$(nproc)
LOAD_INT=${LOAD%.*}
if   [ "$LOAD_INT" -ge $((CORES * 2)) ]; then err  "load=${LOAD} (cores=${CORES}) 过高"
elif [ "$LOAD_INT" -ge "$CORES" ];       then warn "load=${LOAD} (cores=${CORES})"
else                                          ok   "load=${LOAD} (cores=${CORES})"
fi

# Hermes gateway/dashboard (检查容器)
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^hermes$'; then
  ok "hermes gateway 容器运行中"
else
  err "hermes gateway 容器未运行"
fi
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^hermes-dashboard$'; then
  ok "hermes dashboard 容器运行中"
else
  warn "hermes dashboard 容器未运行"
fi

# Nginx
NGINX=$(systemctl is-active nginx 2>/dev/null)
[ "$NGINX" = "active" ] && ok "nginx active" || err "nginx ${NGINX:-DOWN}"

# Docker 容器
for c in open-webui filebrowser; do
  docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${c}$" && ok "docker: ${c} 运行中" || err "docker: ${c} 未运行"
done

# 监听端口 (host network 含 hermes 9119)
PORTS=$(ss -tlnp 2>/dev/null | grep -cE ':(80|3000|8081|8642|9119) ')
[ "${PORTS:-0}" -ge 4 ] && ok "监听端口 ${PORTS} 个" || warn "监听端口仅 ${PORTS} 个"

# ─── 2. 安全审计 ─────────────────────────────────
echo "" | tee -a "$LOG"
echo "▸ 安全审计" | tee -a "$LOG"
AUDIT=/root/security-audit-2026-06-16
KNOWN_AK_HASH=$(cat $AUDIT/known-good/authorized_keys.sha256 2>/dev/null)
if [ -n "$KNOWN_AK_HASH" ]; then
  AK=/home/admin/.ssh/authorized_keys
  CUR_HASH=$(sha256sum $AK | awk '{print $1}')
  if [ "$CUR_HASH" = "$KNOWN_AK_HASH" ]; then
    ok "authorized_keys 未变 (sha256=${CUR_HASH:0:12}…)"
  else
    err "authorized_keys 已变更! 当前=${CUR_HASH:0:12}…, 已知=${KNOWN_AK_HASH:0:12}… (review /root/security-audit-2026-06-16/)"
  fi
else
  warn "无 authorized_keys baseline, 跳过 hash 检查"
fi
# dpkg 完整性
if dpkg --audit 2>/dev/null | grep -q .; then
  warn "dpkg --audit 发现异常 (包未完成配置 / 损坏)"
else
  ok "dpkg --audit 干净"
fi
# 异常 SUID — 首次跑没 prev,跳过对比避免假阳性
SUSPECT_SUID=$(find / -xdev -perm -4000 -type f 2>/dev/null | sort > /tmp/.suid.now; wc -l < /tmp/.suid.now)
echo "  → 当前 SUID 二进制 $SUSPECT_SUID 个"
if [ -f /tmp/.suid.prev ]; then
  if diff -q /tmp/.suid.prev /tmp/.suid.now >/dev/null 2>&1; then
    ok "SUID 列表无变化"
  else
    warn "SUID 列表变化 (见 /tmp/.suid.{prev,now})"
  fi
else
  ok "SUID baseline 已建立 ($SUSPECT_SUID 个,首次跑不对比)"
fi
mv /tmp/.suid.now /tmp/.suid.prev

# ─── 3. 公网端点 ───────────────────────────────
echo "" | tee -a "$LOG"
echo "▸ 公网端点" | tee -a "$LOG"
for pair in \
  "https://chat.leimengde.net|Open WebUI" \
  "https://file.leimengde.net|Filebrowser" \
  ; do
  name="${pair##*|}"; url="${pair%%|*}"
  code=$(curl -sS -o /dev/null -m 8 -L -w "%{http_code}" "$url" 2>/dev/null || echo 000)
  if [ "$code" = "200" ] || [ "$code" = "401" ]; then ok "$name ($code)"; else err "$name ($code)"; fi
done

# ─── 4. 总结 ────────────────────────────────────
echo "" | tee -a "$LOG"
if [ ${#ALERTS[@]} -eq 0 ]; then
  echo "✓ 全部正常 — $TS" | tee -a "$LOG"
  > "$ALERT"
  exit 0
else
  echo "⚠ ${#ALERTS[@]} 项风险需要关注:" | tee -a "$LOG"
  printf '  - %s\n' "${ALERTS[@]}" | tee -a "$LOG" | tee "$ALERT"
  exit 1
fi