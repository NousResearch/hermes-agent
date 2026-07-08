#!/bin/bash
# pre_flight_check.sh — 脚本部署前机械验证所有外部依赖存在
# 用法: bash pre_flight_check.sh <script_file>
# 退出码: 0 = PASS或WARN（不阻断）, 1 = FAIL（阻断部署）
# 已知限制: heredoc内嵌代码不检测、控制流不分析、只查一级node_modules
# 创建: 2026-06-27

set -uo pipefail
# 不使用 set -e——check_file/check_binary 通过 return 传递结果，
# 由末尾的 fail 计数器统一决定 exit code。

SCRIPT="$1"
if [ ! -f "$SCRIPT" ]; then
    echo "❌ pre_flight_check FAIL — $SCRIPT 不存在"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$SCRIPT")" && pwd)"

total=0
fail=0
warn=0
pass=0

# === 展开 ~ 、$HOME 和相对路径 ===
expand_path() {
    local p="$1"
    # 相对路径 → 基于脚本所在目录解析
    if [[ ! "$p" =~ ^[/~$] ]]; then
        p="$SCRIPT_DIR/$p"
    fi
    p="${p/#\~/$HOME}"
    p="${p/\$HOME/$HOME}"
    echo "$p"
}

# === 检查文件存在 ===
check_file() {
    local raw_path="$1"
    local label="$2"
    local expanded
    expanded=$(expand_path "$raw_path")
    ((total++)) || true
    if [ -f "$expanded" ]; then
        echo "  ✅ $label: $raw_path → 存在"
        ((pass++)) || true
        return 0
    else
        echo "  ❌ $label: $raw_path → 文件不存在 ($expanded)"
        ((fail++)) || true
        return 1
    fi
}

# === 检查二进制在 PATH ===
check_binary() {
    local cmd="$1"
    ((total++)) || true
    if command -v "$cmd" &>/dev/null; then
        echo "  ✅ 二进制: $cmd → $(command -v "$cmd")"
        ((pass++)) || true
        return 0
    else
        echo "  ❌ 二进制: $cmd → 不在PATH"
        ((fail++)) || true
        return 1
    fi
}

# === 检查 node_modules 就近依赖 ===
check_node_modules() {
    local script_path="$1"
    local expanded
    expanded=$(expand_path "$script_path")
    local dir
    dir=$(dirname "$expanded")
    if [ -d "$dir/node_modules" ]; then
        echo "  ✅ node_modules: $dir/node_modules/ → 存在"
        ((pass++)) || true
    else
        echo "  ⚠️  node_modules: $dir/ → 同级未找到（Node.js可能从上级目录解析——v1只查一级）"
        ((warn++)) || true
    fi
}

echo "🔍 pre_flight_check: $SCRIPT"
echo ""

# === 预处理：删除所有 heredoc 块（避免正则误入） ===
CLEANED=$(sed -E '/<<[[:space:]]*["'\'']?[A-Za-z_][A-Za-z0-9_]*["'\'']?/,/^[[:space:]]*[A-Za-z_][A-Za-z0-9_]*[[:space:]]*$/d' "$SCRIPT")

# === 逐行扫描 ===
in_heredoc=false
heredoc_delim=""

while IFS= read -r line || [ -n "$line" ]; do
    # 跳过空行
    [[ -z "${line// }" ]] && continue
    
    # 跳过纯注释行
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    
    # 去掉行尾注释（# 后面是注释，但不在字符串内）
    stripped="${line%%#*}"
    [ -z "${stripped// }" ] && continue
    
    # ── 检测1: node <path> ──
    if [[ "$stripped" =~ node[[:space:]]+[\"\']?([^[:space:]\"\']+\.(mjs|js))[\"\']? ]]; then
        raw="${BASH_REMATCH[1]}"
        if [[ "$raw" =~ ^\\$ ]]; then
            echo "  ⚠️  node: $line → 变量引用，无法静态验证"
            ((warn++)) || true
        else
            check_file "$raw" "node"
            # 检查 node_modules
            expanded=$(expand_path "$raw")
            if [ -f "$expanded" ]; then
                check_node_modules "$raw"
            fi
        fi
    fi
    
    # ── 检测2: python3/python <path> ──
    if [[ "$stripped" =~ python3?[[:space:]]+[\"\']?([^[:space:]\"\']+\.py)[\"\']? ]]; then
        raw="${BASH_REMATCH[1]}"
        if [[ "$raw" =~ ^\\$ ]]; then
            echo "  ⚠️  python: $line → 变量引用，无法静态验证"
            ((warn++)) || true
        else
            check_file "$raw" "python"
        fi
    fi
    
    # ── 检测3: bash/sh <path> ──
    if [[ "$stripped" =~ (bash|sh)[[:space:]]+[\"\']?([^[:space:]\"\']+\.sh)[\"\']? ]]; then
        raw="${BASH_REMATCH[2]}"
        if [[ "$raw" =~ ^\\$ ]]; then
            echo "  ⚠️  bash/sh: $line → 变量引用，无法静态验证"
            ((warn++)) || true
        else
            check_file "$raw" "bash/sh"
        fi
    fi
    
    # ── 检测4: 变量引用 node/python/bash（WARN非FAIL）──
    if [[ "$stripped" =~ (node|python3?|bash|sh)[[:space:]]+[\"\']?(\$[A-Za-z_][A-Za-z0-9_]*|\$\{[A-Za-z_][A-Za-z0-9_]*\}) ]]; then
        echo "  ⚠️  ${BASH_REMATCH[1]}: $line → 变量引用，无法静态验证"
        ((warn++)) || true
    fi
    
    # ── 检测5: 固定文件路径引用（非 node/python/bash 上下文的文件路径）──
    # 扫描 ~/path/file.ext 或 /path/file.ext 模式（含常见扩展名）
    if [[ "$stripped" =~ (^|[[:space:];|&\<\>\(\)])(~?/[^[:space:];|&\<\>\(\)]*\.(pdf|pptx|docx|xlsx|csv|json|yaml|yml|xml|txt|png|jpg|jpeg|gif|svg|md|rst|cfg|conf|toml|ini|env)) ]]; then
        raw="${BASH_REMATCH[2]}"
        # 排除已被 node/python/bash 检测处理过的
        if [[ ! "$raw" =~ \.(mjs|js|py|sh)$ ]]; then
            # 排除注释和 echo/printf 打印中的路径
            if [[ ! "$stripped" =~ (echo|printf|cat[[:space:]]*\<\<) ]]; then
                check_file "$raw" "文件"
            fi
        fi
    fi
    
    # ── 检测6: 裸二进制命令 ──
    # 只扫描看起来像命令执行的行政（不含赋值、括号、引号开头的行）
    # 排除常见的 shell 编程语法噪声
    if [[ "$stripped" =~ ^[[:space:]]*[a-z][a-z0-9_-]*[[:space:]] && ! "$stripped" =~ [\(\)\{\}=] && ! "$stripped" =~ ^[[:space:]]*[\"\'] ]]; then
        first_word=$(echo "$stripped" | awk '{print $1}')
        case "$first_word" in
            if|then|else|elif|fi|for|while|do|done|case|esac|in|function|return|exit|echo|cd|export|source|.|eval|exec|set|unset|alias|local|declare|readonly|shift|wait|test|true|false|node|python|python3|bash|sh|cat|grep|awk|sed|find|ls|cp|mv|rm|mkdir|timeout|sudo|from|import|print|def|class|try|except|with|as|yield|raise|pass|break) ;;
            *)
                check_binary "$first_word"
                ;;
        esac
    fi
    
done <<< "$CLEANED"

echo ""
echo "──────────────────────────"
if [ "$fail" -gt 0 ]; then
    echo "❌ pre_flight_check FAIL — ${fail}项缺失 / ${total}项检查"
    exit 1
elif [ "$warn" -gt 0 ]; then
    echo "⚠️  pre_flight_check PASS — ${pass}项通过, ${warn}项WARN（静态验证不可用）"
    exit 0
else
    echo "✅ pre_flight_check PASS — ${pass}项依赖全部存在"
    exit 0
fi
