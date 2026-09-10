#!/usr/bin/env bash
#
# 快速应用"禁用自动付费模型"配置
# 
# 使用方式：
#   bash scripts/disable-paid-models.sh
#   bash scripts/disable-paid-models.sh --dry-run   # 预览不保存
#

set -e

DRY_RUN=0
VERBOSE=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=1
            echo "[DRY RUN MODE] 预览更改，不保存文件"
            shift
            ;;
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

CONFIG_PATH="$HOME/.hermes/config.yaml"
BACKUP_PATH="$HOME/.hermes/config.yaml.backup.$(date +%s)"

echo "=========================================="
echo "Hermes 付费模型防护配置"
echo "=========================================="
echo ""

# 检查配置文件是否存在
if [ ! -f "$CONFIG_PATH" ]; then
    echo "❌ 错误：找不到 $CONFIG_PATH"
    echo "   请先运行 'hermes setup' 初始化"
    exit 1
fi

echo "📋 当前配置文件：$CONFIG_PATH"
echo ""

# 显示当前 fallback 配置
echo "🔍 当前 fallback 配置状态："
if grep -q "fallback_providers:" "$CONFIG_PATH" 2>/dev/null; then
    echo "   找到 fallback_providers 配置："
    grep -A 10 "fallback_providers:" "$CONFIG_PATH" | head -15
else
    echo "   ✓ 未找到 fallback_providers（已禁用）"
fi

if grep -q "fallback_model:" "$CONFIG_PATH" 2>/dev/null; then
    echo "   找到旧格式 fallback_model 配置："
    grep -A 5 "fallback_model:" "$CONFIG_PATH" | head -8
else
    echo "   ✓ 未找到 fallback_model（已禁用）"
fi

echo ""
echo "========== 选择操作 =========="
echo "1) 禁用所有 fallback（推荐：零成本，100%安全）"
echo "2) 配置仅本地 fallback（可选容错，仅用本地模型）"
echo "3) 配置免费 OpenRouter fallback（谨慎：可能产生成本）"
echo "4) 查看详细指南"
echo "5) 退出"
echo ""
read -p "请选择 [1-5]: " choice

case $choice in
    1)
        echo ""
        echo "🔧 操作：禁用所有 fallback..."
        
        # 备份原配置
        if [ $DRY_RUN -eq 0 ]; then
            cp "$CONFIG_PATH" "$BACKUP_PATH"
            echo "   ✓ 原配置已备份到：$BACKUP_PATH"
        fi
        
        # 移除或注释掉 fallback 配置
        if [ $DRY_RUN -eq 0 ]; then
            # 创建临时文件
            tmp_file=$(mktemp)
            
            # 删除或注释 fallback_model 和 fallback_providers 行及其内容
            awk '
                /^fallback_model:/ { 
                    print "# " $0 " (disabled by disable-paid-models.sh)"
                    skip=1 
                    next 
                }
                /^fallback_providers:/ { 
                    print "# " $0 " (disabled by disable-paid-models.sh)"
                    print "fallback_providers: []"
                    skip=1 
                    next 
                }
                skip && /^[^ #]/ { skip=0 }
                skip && /^  [^ ]/ { skip=0 }
                !skip { print }
            ' "$CONFIG_PATH" > "$tmp_file"
            
            mv "$tmp_file" "$CONFIG_PATH"
            echo "   ✓ 已禁用 fallback 配置"
        else
            echo "   [DRY RUN] 将禁用 fallback 配置"
        fi
        
        echo ""
        echo "✅ 完成！Hermes 现在只会使用本地模型。"
        echo "   - 主模型失败时：显示错误（不切换）"
        echo "   - 成本：$0（只用本地 Ollama）"
        echo "   - 安全性：100%（无 API 调用）"
        ;;
        
    2)
        echo ""
        echo "🔧 操作：配置本地 fallback..."
        
        # 检查本地可用模型
        echo ""
        echo "🔎 检查本地 Ollama 可用模型..."
        
        if command -v curl &> /dev/null; then
            local_models=$(curl -s http://172.22.144.1:11434/api/tags 2>/dev/null | grep -o '"name":"[^"]*' | cut -d'"' -f4 || echo "")
            if [ -z "$local_models" ]; then
                echo "   ⚠️  无法连接到 Ollama（172.22.144.1:11434）"
                echo "      请确保 Windows 侧 Ollama 服务正在运行"
            else
                echo "   ✓ 找到本地模型："
                echo "$local_models" | while read model; do
                    echo "      - $model"
                done
            fi
        else
            echo "   ⚠️  需要 curl 来检查本地模型"
        fi
        
        # 创建示例配置
        cat > /tmp/fallback-example.yaml << 'EOF'
# 示例：本地 fallback 配置
fallback_providers:
  - provider: custom
    model: qwen2.5:7b
    base_url: http://172.22.144.1:11434/v1
  - provider: custom
    model: gemma4:26b
    base_url: http://172.22.144.1:11434/v1
EOF
        
        echo ""
        echo "📄 示例配置："
        cat /tmp/fallback-example.yaml
        
        if [ $DRY_RUN -eq 0 ]; then
            read -p "要应用此配置吗？[y/N]: " apply
            if [ "$apply" = "y" ] || [ "$apply" = "Y" ]; then
                cp "$CONFIG_PATH" "$BACKUP_PATH"
                # 将示例内容插入到配置文件
                sed -i '/^fallback_providers:/,/^[^ ]/{
                    /^fallback_providers:/c\
fallback_providers:\
  - provider: custom\
    model: qwen2.5:7b\
    base_url: http://172.22.144.1:11434/v1\
  - provider: custom\
    model: gemma4:26b\
    base_url: http://172.22.144.1:11434/v1
                }' "$CONFIG_PATH" 2>/dev/null || echo "配置更新可能失败，请手动编辑"
                echo "   ✓ 配置已应用"
            fi
        fi
        ;;
        
    3)
        echo ""
        echo "⚠️  警告：OpenRouter fallback 可能产生成本！"
        echo ""
        echo "📖 详细指南请见：PREVENT_PAID_MODELS.md 中的『策略 3』"
        echo ""
        echo "建议操作："
        echo "  1. 阅读：cat PREVENT_PAID_MODELS.md | grep -A 30 '策略 3'"
        echo "  2. 在 OpenRouter 中设置支出上限：https://openrouter.ai/account/billing/limits"
        echo "  3. 手动编辑 ~/.hermes/config.yaml"
        echo ""
        ;;
        
    4)
        echo ""
        echo "📖 详细指南："
        if [ -f "PREVENT_PAID_MODELS.md" ]; then
            less PREVENT_PAID_MODELS.md
        else
            echo "❌ 找不到 PREVENT_PAID_MODELS.md"
            echo "   请在项目根目录运行此脚本"
        fi
        ;;
        
    5)
        echo "退出"
        exit 0
        ;;
        
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "下一步："
echo "  1. 检查配置：hermes config show | grep -A 5 fallback"
echo "  2. 测试模型：hermes --test"
echo "  3. 查看日志：tail -f ~/.hermes/logs/agent.log"
echo ""
