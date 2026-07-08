#!/bin/bash
# boy_scout_scan.sh — 童子军规则扫描
# 用法: bash boy_scout_scan.sh <任务描述关键词>
# 输出: 全量清单（分组折叠），不自动修任何东西

set -o pipefail

KEYWORD="$1"
[ -z "$KEYWORD" ] && { echo "用法: bash boy_scout_scan.sh <关键词>"; exit 1; }

SCRIPTS_DIR="/home/ohtok/.hermes/scripts"
SKILLS_DIR="/home/ohtok/.hermes/skills/personal"
LINT_DIR="/home/ohtok/.hermes/lint"
SECURITY=()
STYLE=()

# ── 判断扫描范围 ──

# lint 脚本相关 → 扫 lint/ 下其他脚本
if echo "$KEYWORD" | grep -qP "lint|[0-9]{2}-|pipefail|架构品味|restic|password|nas|notebook|credential"; then
    # 扫 pipefail
    for f in $(find "$SCRIPTS_DIR" "$LINT_DIR" -name "*.sh" -type f 2>/dev/null); do
        bn=$(basename "$f")
        [[ "$bn" == "boy_scout_scan.sh" ]] && continue
        head -10 "$f" 2>/dev/null | grep -q "pipefail" && continue
        [ $(wc -l < "$f" 2>/dev/null) -lt 10 ] && continue
        STYLE+=("🟡 $bn: 缺 set -o pipefail")
    done
    
    # 扫凭据变量
    for f in $(find "$SCRIPTS_DIR" -name "*.sh" -type f 2>/dev/null); do
        bn=$(basename "$f")
        if grep -qP '\$(PWFILE|BAILIAN_KEY|BAILIAN_API_KEY)' "$f" 2>/dev/null; then
            SECURITY+=("🔴 $bn: 遮蔽变量名 \$PWFILE/\$BAILIAN_KEY")
        fi
    done
    
    # 扫 restic 密码
    for f in $(grep -rl "restic" "$SCRIPTS_DIR" 2>/dev/null); do
        bn=$(basename "$f")
        if ! grep -q "\-\-password-file" "$f" 2>/dev/null; then
            SECURITY+=("🔴 $bn: restic 未用 --password-file")
        fi
    done
fi

# skill 相关 → 扫同类 skill
if echo "$KEYWORD" | grep -qP "skill|SKILL|trigger|description|frontmatter|合规"; then
    for d in $(find "$SKILLS_DIR" -maxdepth 1 -type d 2>/dev/null); do
        [ "$d" = "$SKILLS_DIR" ] && continue
        skill_md="$d/SKILL.md"
        [ ! -f "$skill_md" ] && continue
        bn=$(basename "$d")
        
        # 缺 triggers (非 reference/bridge)
        if ! grep -qP '^triggers:' "$skill_md" 2>/dev/null; then
            tp=$(grep -oP '^type:\s*\K\S+' "$skill_md" 2>/dev/null)
            if [ "$tp" != "reference" ] && [ "$tp" != "bridge" ]; then
                STYLE+=("🟡 $bn: type=$tp 缺 triggers")
            fi
        fi
        
        # 缺 status
        if ! grep -qP '^status:' "$skill_md" 2>/dev/null; then
            STYLE+=("🟡 $bn: 缺 status 字段")
        fi
    done
fi

# cron/脚本相关 → 扫 at 回退
if echo "$KEYWORD" | grep -qP "cron|script|脚本|systemctl|restart|gateway"; then
    for f in $(find "$SCRIPTS_DIR" -name "*.sh" -type f 2>/dev/null); do
        bn=$(basename "$f")
        if grep -q "systemctl.*restart" "$f" 2>/dev/null && ! grep -q "at now" "$f" 2>/dev/null; then
            SECURITY+=("🔴 $bn: systemctl restart 未设 at 回退")
        fi
    done
fi

# NAS 路径相关 → 扫非备份脚本
if echo "$KEYWORD" | grep -qP "NAS|nas|/mnt|mount"; then
    for f in $(find "$SCRIPTS_DIR" "$SKILLS_DIR" -name "*.sh" -type f 2>/dev/null); do
        bn=$(basename "$f")
        case "$bn" in *backup*|*备份*|*restic*|*sync*|*S1*|*S2*|*nas_*) continue ;; esac
        if grep -n "/mnt/nas" "$f" 2>/dev/null | grep -v '^\s*#' | grep -v 'echo\|printf' | grep -qE '(cp |mv |tar |rsync |>|>>)'; then
            SECURITY+=("🔴 $bn: 非备份脚本写 /mnt/nas")
        fi
    done
fi

# ── 输出 ──

TOTAL_SECURITY=${#SECURITY[@]}
TOTAL_STYLE=${#STYLE[@]}
TOTAL=$((TOTAL_SECURITY + TOTAL_STYLE))

if [ $TOTAL -eq 0 ]; then
    echo "✅ 童子军扫描: 周边无可顺手修的项目"
    exit 0
fi

echo "童子军扫描: $KEYWORD"
echo ""

if [ $TOTAL_SECURITY -gt 0 ]; then
    echo "🔴 安全隐患 ($TOTAL_SECURITY)"
    for item in "${SECURITY[@]}"; do
        echo "  $item"
    done
    echo ""
fi

if [ $TOTAL_STYLE -gt 0 ]; then
    MAX_SHOW=15
    echo "🟡 规范问题 ($TOTAL_STYLE)"
    for i in "${!STYLE[@]}"; do
        if [ $i -lt $MAX_SHOW ]; then
            echo "  ${STYLE[$i]}"
        fi
    done
    if [ $TOTAL_STYLE -gt $MAX_SHOW ]; then
        echo "  ... 还有 $((TOTAL_STYLE - MAX_SHOW)) 条（说"展开"看全量）"
    fi
    echo ""
fi

echo "共 $TOTAL 个可顺手修的项目。修哪些？"
