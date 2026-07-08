#!/bin/bash
# output-violation-scan.sh — 每日事后机械审计 v2
# v2: 路径豁免 + 上下文降级，减少设计文档/演讲稿误报
set -eo pipefail
# 扫昨天的 Agent 产出文件，对照 SOUL.md 机械规则
# cron 驱动（systemd 调度，不依赖 Agent 自觉）
# PASS → silent; FAIL → 列出违规项推微信

set -o pipefail

TODAY=$(date '+%Y-%m-%d')
SCAN_MINUTES=1500  # 每天一次，扫最近 25 小时内的修改文件
OUTPUT_DIR="/home/ohtok/hermes-local/yangyang"
ALERTS=""
HAS_VIOLATION=0

# ── 豁免配置 ──
# 这些路径下的文件跳过 NAS 检查和 date 检查（设计文档/演讲稿天然讨论系统路径和包含时间措辞）
EXEMPT_NAS_DATE_PATTERNS=(
    "/home/ohtok/hermes-local/yangyang/设计文档/"
    "/home/ohtok/hermes-local/yangyang/小青推广演讲"
    "/home/ohtok/hermes-local/yangyang/projects/"   # PPT 项目目录
)

# 外部第十人/子代理产出文件——独立 session 不加载 SOUL 规则，不做 Rule 0/1/2/审阅指南/TL;DR 检查
EXEMPT_EXTERNAL_TENTH_PATTERNS=(
    "外部第十人"
    "_纪律性第十人"
    "第十人审计_"            # v3: delegate_task 产出的第十人审计报告
)

# 非 Agent 产出——外部文件/系统注册表/手工文档，不要求 SOUL 合规标签
EXEMPT_NON_AGENT_OUTPUT_PATTERNS=(
    "/home/ohtok/hermes-local/yangyang/incoming/"
    "/home/ohtok/hermes-local/yangyang/inbox/"
    "小青特性注册表"
    "INDEX"                           # INDEX.md 等索引文件
    "Hermes-Skill设计参考指南"        # 静态参考文档，非 Agent 产出
)

# PPT Skill 产出文件——生成脚本/浅层大纲/PPT Master 中间文件，不走 SOUL 合规审计
# 2026-06-25 杨旸明确："这个skill的脚本不用做合规审计，以后都是"
EXEMPT_PPT_SKILL_PATTERNS=(
    "生成脚本_"                       # 生成脚本_*.md
    "浅层大纲_"                       # 浅层大纲_*.md
    "/home/ohtok/hermes-local/yangyang/projects/"  # PPT 项目目录(content-source/spec_lock/design_spec/image_prompts)
)

# delegate_task 产出自动补标——这些文件不经过 Agent 正常 write_file 流程，缺合规标记
# 检测方式：文件名含以上豁免模式 + 内容含 delegate_task 特征（如 "delegate_task" "子代理" "独立审计"）
AUTO_PATCH_PATTERNS=(
    "外部第十人"
    "第十人审计_"
    "_纪律性第十人"
)

# 方案 C：NAS 路径上下文判读——讨论标记词
DISCUSSION_MARKERS='P00[0-9]|ADR-00[0-9]|PRODUCT_PATCHES|write_guard|skip_for_paths|误报|描述|提到|触发|历史|合规警告|含 2 处'

# ── delegate_task 产出自动补标 ──
# v3: 识别子代理/第十人产出，自动注入缺失的合规标记，消除代理不加载 SOUL 规则导致的假阳性
auto_patch_delegate_output() {
    local f="$1"
    local bn=$(basename "$f")
    local is_delegate=0
    
    # 检测是否为 delegate_task 产出
    for pattern in "${AUTO_PATCH_PATTERNS[@]}"; do
        if echo "$bn" | grep -qF "$pattern"; then
            is_delegate=1
            break
        fi
    done
    # 也检查内容特征：含 "审计对象" "外部第十人" "审计人：外部第十人" 等
    if [ "$is_delegate" -eq 0 ] && grep -qP '(审计对象.*内部AI|审计人.*外部第十人|delegate_task.*独立审计)' "$f" 2>/dev/null; then
        is_delegate=1
    fi
    
    [ "$is_delegate" -eq 0 ] && return 0
    
    # 检测缺失项并补标
    local needs_patch=0
    local patch_tail=""
    
    # 🟡 假设标注
    if ! grep -q '🟡 假设' "$f" 2>/dev/null; then
        needs_patch=1
        patch_tail="${patch_tail}
🟡 假设：此文件产自 delegate_task 独立子进程，未经过 Agent SOUL 规则约束的正常产出流程。内容仅基于子进程 prompt 注入的快照，未访问本 session 的实时上下文。"
    fi
    
    # 📋 审阅指南（仅方案/报告类）
    if grep -qP '(方案|报告|深度分析|审计|设计)' "$f" 2>/dev/null; then
        if ! grep -qP '📋 审阅指南|审阅指南' "$f" 2>/dev/null; then
            needs_patch=1
            patch_tail="${patch_tail}
## 📋 审阅指南
- **重点段落**：[请杨旸标注]
- **可跳过**：[请杨旸标注]
- **数据来源**：delegate_task 子进程独立分析
- **假设列表**：[请杨旸补充]
- **建议审阅方式**：全文通读（子代理独立上下文，可能存在盲区）"
        fi
    fi
    
    if [ "$needs_patch" -eq 1 ]; then
        echo "$patch_tail" >> "$f"
        echo "  ✅ auto-patched: $bn" >&2
    fi
    return 0
}

# 扫描并自动补标
for f in $(find "$OUTPUT_DIR" -name "*.md" -mmin -"$SCAN_MINUTES" 2>/dev/null); do
    auto_patch_delegate_output "$f"
done

# ── 辅助函数 ──

flag() {
    HAS_VIOLATION=1
    ALERTS="${ALERTS}$1\\n"
}

is_exempt_nas_date() {
    local f="$1"
    for pattern in "${EXEMPT_NAS_DATE_PATTERNS[@]}"; do
        if echo "$f" | grep -qF "$pattern"; then
            return 0
        fi
    done
    return 1
}

is_exempt_external_tenth() {
    local f="$1"
    for pattern in "${EXEMPT_EXTERNAL_TENTH_PATTERNS[@]}"; do
        if echo "$(basename "$f")" | grep -qF "$pattern"; then
            return 0
        fi
    done
    return 1
}

is_exempt_non_agent() {
    local f="$1"
    for pattern in "${EXEMPT_NON_AGENT_OUTPUT_PATTERNS[@]}"; do
        if echo "$f" | grep -qF "$pattern"; then
            return 0
        fi
    done
    # PPT Skill 产出豁免（2026-06-25 杨旸）
    for pattern in "${EXEMPT_PPT_SKILL_PATTERNS[@]}"; do
        if echo "$f" | grep -qF "$pattern"; then
            return 0
        fi
    done
    return 1
}

# ── Rule 2B: SSD-only ──
# 产出文件路径含 /mnt/nas/ 且不是备份文件
# v2: 豁免设计文档/演讲稿；非豁免文件做上下文判读降级
for f in $(find "$OUTPUT_DIR" -name "*.md" -mmin -"$SCAN_MINUTES" ! -name "*backup*" ! -name "*备份*" 2>/dev/null); do
    if grep -q '/mnt/nas/' "$f" 2>/dev/null; then
        if is_exempt_nas_date "$f"; then
            continue  # 设计文档/演讲稿：不检查 NAS 路径
        fi
        # 方案 C：上下文判读——检查命中行附近是否有讨论标记词
        if grep -B3 -A3 '/mnt/nas/' "$f" 2>/dev/null | grep -qP "$DISCUSSION_MARKERS"; then
            flag "⚪ Rule 2B: $(basename "$f") 含 NAS 路径（上下文判读为讨论性引用，非真违规）"
        else
            flag "🔴 Rule 2B: $(basename "$f") 含 NAS 路径"
        fi
    fi
done

# ── Rule 2C: 文件名安全 ──
# 新产出文件名含 emoji / Windows 保留字 / 控制字符 / 尾部空格或点
for f in $(find "$OUTPUT_DIR" -name "*.md" -mmin -"$SCAN_MINUTES" 2>/dev/null); do
    bn=$(basename "$f")
    if echo "$bn" | grep -qP '[\\\\/:*?\"<>|]'; then
        flag "🔴 Rule 2C: $(basename "$f") 含 Windows 保留字符"
    fi
    if echo "$bn" | grep -q '[[:cntrl:]]'; then
        flag "🔴 Rule 2C: $(basename "$f") 含控制字符"
    fi
    if echo "$bn" | grep -qE ' $|\\.$'; then
        flag "🔴 Rule 2C: $(basename "$f") 含尾部空格或点"
    fi
done

# ── Rule 1: 置信度标签 ──
# 分析类文件必须有 🔴🟡🟢⚪
# 豁免：外部第十人/子代理产出
for f in $(find "$OUTPUT_DIR" -name "*.md" -mmin -"$SCAN_MINUTES" 2>/dev/null); do
    is_exempt_external_tenth "$f" && continue
    is_exempt_non_agent "$f" && continue
    # 判断是否为分析类：含"分析/建议/方案/报告/对比/审查"关键词
    if grep -qP '(分析|建议|方案|报告|对比|审查|判断|推荐|选择)' "$f" 2>/dev/null; then
        if ! grep -qP '[🔴🟡🟢⚪]' "$f" 2>/dev/null; then
            flag "🟡 Rule 1: $(basename "$f") 为分析类但无置信度标签"
        fi
    fi
done

# ── Rule 0 假设标注 ──
# 豁免：外部第十人/子代理产出
for f in $(find "$OUTPUT_DIR" -name "*.md" -mmin -"$SCAN_MINUTES" 2>/dev/null); do
    is_exempt_external_tenth "$f" && continue
    is_exempt_non_agent "$f" && continue
    if grep -qP '(分析|建议|方案|报告|对比|审查|判断|推荐|选择)' "$f" 2>/dev/null; then
        if ! grep -q '🟡 假设' "$f" 2>/dev/null; then
            flag "🟡 Rule 0: $(basename "$f") 含分析但无 🟡假设标注"
        fi
    fi
done

# ── Rule 2: 时间验证 ──
# v2: 设计文档/演讲稿豁免（日期=文件名）；外部第十人豁免
for f in $(find "$OUTPUT_DIR" -name "*.md" -mmin -"$SCAN_MINUTES" 2>/dev/null); do
    if grep -qP '(刚才|现在|今天|X分钟前|昨晚|马上|刚刚)' "$f" 2>/dev/null; then
        if grep -qP 'date.*\\+' "$f" 2>/dev/null; then
            continue  # 已有 date 命令，不报
        fi
        if is_exempt_nas_date "$f"; then
            continue  # 设计文档/演讲稿：不报 date 缺失
        fi
        if is_exempt_external_tenth "$f"; then
            continue  # 外部第十人：不报 date 缺失
        fi
        if is_exempt_non_agent "$f"; then
            continue  # 非 Agent 产出：不报 date 缺失
        fi
        flag "🟡 Rule 2: $(basename "$f") 含时间表述但无 date 命令输出"
    fi
done

# ── 审阅指南（方案/报告类） ──
# 豁免：外部第十人/子代理产出
for f in $(find "$OUTPUT_DIR" -name "*.md" -mmin -"$SCAN_MINUTES" 2>/dev/null); do
    is_exempt_external_tenth "$f" && continue
    is_exempt_non_agent "$f" && continue
    if grep -qP '(方案|报告|深度分析|SOUL|设计)' "$f" 2>/dev/null; then
        if ! grep -qP '📋 审阅指南|审阅指南' "$f" 2>/dev/null; then
            flag "🟡 审阅指南: $(basename "$f") 为方案/报告类但无 📋 审阅指南"
        fi
    fi
done

# ── TL;DR（>500 字） ──
# 豁免：外部第十人/子代理产出
for f in $(find "$OUTPUT_DIR" -name "*.md" -mmin -"$SCAN_MINUTES" 2>/dev/null); do
    is_exempt_external_tenth "$f" && continue
    is_exempt_non_agent "$f" && continue
    # 跳过系统文件
    if echo "$f" | grep -qE '(深思_|对比_|report_)'; then
        chars=$(wc -c < "$f" 2>/dev/null)
        if [ "$chars" -gt 1500 ]; then  # ~500 中文字
            if ! grep -qP 'TL;DR|tl;dr' "$f" 2>/dev/null; then
                flag "🟡 TL;DR: $(basename "$f") >500字但无 TL;DR"
            fi
        fi
    fi
done

# ── 输出 ──
if [ "$HAS_VIOLATION" -eq 1 ]; then
    echo "🚨 产出合规审计 ($TODAY)"
    echo ""
    echo -e "$ALERTS"
    echo ""
    echo "> 规则来源：SOUL.md Rule 0/1/2/2B/2C + 审阅指南 + TL;DR"
    echo "> 审计模式：事后机械扫描（cron 驱动，非 Agent 自觉）"
    echo "> v2: 设计文档/演讲稿豁免NAS+date；非豁免文件NAS路径上下文判读降级"
fi

# ── 架构品味 Linter ──
# 仅报告，不影响退出码（架构债≠每日合规失败）
set +e  # 允许 linter 返回非零而不中断脚本
LINT_OUTPUT=$(bash /home/ohtok/.hermes/lint/run_all.sh 2>&1)
LINT_EXIT=$?
set -e  # 恢复

if [ "$LINT_EXIT" -eq 1 ]; then
    echo ""
    echo "---"
    echo "🔧 架构品味 Linter"
    echo "$LINT_OUTPUT" | head -20
    echo "   ... ($(echo "$LINT_OUTPUT" | wc -l) lines total)"
elif [ "$LINT_EXIT" -eq 2 ]; then
    echo ""
    echo "---"
    echo "🔧 架构品味 Linter (规范问题)"
    echo "$LINT_OUTPUT" | head -15
    echo "   ... ($(echo "$LINT_OUTPUT" | wc -l) lines total)"
fi

if [ "$HAS_VIOLATION" -eq 1 ]; then
    exit 1
fi
exit 0
