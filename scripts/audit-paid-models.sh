#!/usr/bin/env bash
#
# 检查 Hermes 是否被配置为自动调用 OpenRouter 付费模型
# 
# 使用方式：
#   bash scripts/audit-paid-models.sh
#

set -e

CONFIG_PATH="$HOME/.hermes/config.yaml"
LOG_PATH="$HOME/.hermes/logs/agent.log"

echo "=========================================="
echo "Hermes 付费模型审计报告"
echo "=========================================="
echo ""
echo "⏱️  生成时间：$(date)"
echo ""

# ============================================================================
# 检查 1：配置文件中的 fallback 设置
# ============================================================================

echo "🔍 检查 1：Fallback 配置"
echo "-" | awk '{for(i=0;i<40;i++)printf "-"} END {print ""}'

if [ ! -f "$CONFIG_PATH" ]; then
    echo "❌ 错误：找不到 $CONFIG_PATH"
    exit 1
fi

fallback_model=$(grep -A 3 "^fallback_model:" "$CONFIG_PATH" 2>/dev/null || echo "")
fallback_providers=$(grep -A 10 "^fallback_providers:" "$CONFIG_PATH" 2>/dev/null || echo "")

if [ -z "$fallback_model" ] && [ -z "$fallback_providers" ]; then
    echo "✅ 好消息：未找到 fallback_model 或 fallback_providers"
    echo "   → 禁用自动 fallback"
    FALLBACK_STATUS="DISABLED"
elif echo "$fallback_providers" | grep -q "fallback_providers: \[\]"; then
    echo "✅ 好消息：fallback_providers 为空列表"
    echo "   → 禁用自动 fallback"
    FALLBACK_STATUS="DISABLED"
else
    echo "⚠️  警告：检测到 fallback 配置"
    FALLBACK_STATUS="ENABLED"
    if [ -n "$fallback_model" ]; then
        echo ""
        echo "   fallback_model："
        echo "$fallback_model" | sed 's/^/     /'
    fi
    if [ -n "$fallback_providers" ]; then
        echo ""
        echo "   fallback_providers："
        echo "$fallback_providers" | sed 's/^/     /'
    fi
fi

echo ""

# ============================================================================
# 检查 2：OpenRouter 相关配置
# ============================================================================

echo "🔍 检查 2：OpenRouter 配置"
echo "-" | awk '{for(i=0;i<40;i++)printf "-"} END {print ""}'

openrouter_config=$(grep -i "openrouter" "$CONFIG_PATH" 2>/dev/null || echo "")

if [ -z "$openrouter_config" ]; then
    echo "✅ 好消息：配置中未找到 OpenRouter 引用"
    OPENROUTER_CONFIGURED="NO"
else
    echo "⚠️  检测到 OpenRouter 配置："
    echo "$openrouter_config" | sed 's/^/   /'
    OPENROUTER_CONFIGURED="YES"
fi

echo ""

# ============================================================================
# 检查 3：最近的日志中的 provider 使用
# ============================================================================

echo "🔍 检查 3：最近日志中的 Provider 使用"
echo "-" | awk '{for(i=0;i<40;i++)printf "-"} END {print ""}'

if [ ! -f "$LOG_PATH" ]; then
    echo "⚠️  日志文件不存在：$LOG_PATH"
    echo "   首次运行 Hermes 后会生成日志"
    RECENT_PROVIDERS="UNKNOWN"
else
    recent_logs=$(tail -500 "$LOG_PATH" 2>/dev/null || echo "")
    
    # 检查最近 500 行中的 provider 信息
    custom_usage=$(echo "$recent_logs" | grep -ci "provider.*custom\|provider.*local" || echo "0")
    openrouter_usage=$(echo "$recent_logs" | grep -ci "provider.*openrouter\|openrouter.*api" || echo "0")
    openai_usage=$(echo "$recent_logs" | grep -ci "provider.*openai\|openai.*api" || echo "0")
    anthropic_usage=$(echo "$recent_logs" | grep -ci "provider.*anthropic\|claude.*api" || echo "0")
    
    echo "   最近 500 行日志中的 provider 使用次数："
    echo "   - custom/local: $custom_usage 次"
    echo "   - OpenRouter: $openrouter_usage 次"
    echo "   - OpenAI: $openai_usage 次"
    echo "   - Anthropic: $anthropic_usage 次"
    
    if [ "$openrouter_usage" -gt 0 ] || [ "$openai_usage" -gt 0 ] || [ "$anthropic_usage" -gt 0 ]; then
        echo ""
        echo "   ⚠️  检测到付费 API 调用！"
        RECENT_PROVIDERS="PAID"
    elif [ "$custom_usage" -gt 0 ]; then
        echo ""
        echo "   ✅ 只检测到本地模型调用"
        RECENT_PROVIDERS="LOCAL"
    else
        echo ""
        echo "   ℹ️  无法从日志确定 provider 使用"
        RECENT_PROVIDERS="UNKNOWN"
    fi
fi

echo ""

# ============================================================================
# 检查 4：API Key 配置
# ============================================================================

echo "🔍 检查 4：API Key 配置"
echo "-" | awk '{for(i=0;i<40;i++)printf "-"} END {print ""}'

env_file="$HOME/.hermes/.env"

if [ ! -f "$env_file" ]; then
    echo "ℹ️  .env 文件不存在（或使用内联 API key）"
else
    openrouter_key=$(grep -i "openrouter.*api.*key" "$env_file" 2>/dev/null || echo "")
    openai_key=$(grep -i "openai.*api.*key" "$env_file" 2>/dev/null || echo "")
    anthropic_key=$(grep -i "anthropic.*api.*key" "$env_file" 2>/dev/null || echo "")
    
    if [ -n "$openrouter_key" ]; then
        echo "⚠️  找到 OpenRouter API key 配置"
        key_len=$(echo "$openrouter_key" | grep -o "=" | wc -l)
        if [ $key_len -gt 0 ]; then
            echo "   [已配置，实际值已隐藏]"
        fi
    fi
    
    if [ -n "$openai_key" ]; then
        echo "⚠️  找到 OpenAI API key 配置"
    fi
    
    if [ -n "$anthropic_key" ]; then
        echo "⚠️  找到 Anthropic API key 配置"
    fi
fi

if [ -z "$openrouter_key" ] && [ -z "$openai_key" ] && [ -z "$anthropic_key" ]; then
    echo "✅ 未配置付费 API keys"
    API_KEYS_STATUS="NONE"
else
    API_KEYS_STATUS="CONFIGURED"
fi

echo ""

# ============================================================================
# 汇总报告
# ============================================================================

echo "=========================================="
echo "📊 汇总报告"
echo "=========================================="
echo ""

RISK_LEVEL="LOW"

case $FALLBACK_STATUS in
    DISABLED)
        echo "✅ Fallback 状态：已禁用"
        ;;
    ENABLED)
        echo "⚠️  Fallback 状态：已启用"
        RISK_LEVEL="MEDIUM"
        ;;
esac

case $OPENROUTER_CONFIGURED in
    NO)
        echo "✅ OpenRouter 配置：无"
        ;;
    YES)
        echo "⚠️  OpenRouter 配置：已存在"
        if [ "$FALLBACK_STATUS" = "ENABLED" ]; then
            RISK_LEVEL="HIGH"
        fi
        ;;
esac

case $RECENT_PROVIDERS in
    LOCAL)
        echo "✅ 最近使用：本地模型"
        ;;
    PAID)
        echo "🚨 最近使用：付费 API"
        RISK_LEVEL="HIGH"
        ;;
    UNKNOWN)
        echo "ℹ️  最近使用：无法确定"
        ;;
esac

case $API_KEYS_STATUS in
    NONE)
        echo "✅ API Keys：未配置"
        ;;
    CONFIGURED)
        echo "⚠️  API Keys：已配置（可能的成本来源）"
        if [ "$FALLBACK_STATUS" = "ENABLED" ]; then
            RISK_LEVEL="HIGH"
        fi
        ;;
esac

echo ""
echo "╔════════════════════════════════════════╗"
echo "║  风险等级：$RISK_LEVEL"
echo "╚════════════════════════════════════════╝"
echo ""

case $RISK_LEVEL in
    LOW)
        echo "✅ 低风险 - Hermes 配置安全"
        echo "   推荐：定期检查日志以确保无意外 API 调用"
        ;;
    MEDIUM)
        echo "⚠️  中风险 - 检测到 fallback 配置"
        echo "   建议："
        echo "   1. 查看 QUICK_FIX_PAID_MODELS.md 了解详情"
        echo "   2. 运行 'bash scripts/disable-paid-models.sh'"
        echo "   3. 设置 OpenRouter 支出限额：https://openrouter.ai/account/billing/limits"
        ;;
    HIGH)
        echo "🚨 高风险 - 检测到可能的付费 API 使用"
        echo "   立即行动："
        echo "   1. 备份配置："
        echo "      cp ~/.hermes/config.yaml ~/.hermes/config.yaml.backup"
        echo "   2. 禁用 fallback："
        echo "      bash scripts/disable-paid-models.sh"
        echo "   3. 检查成本："
        echo "      - OpenRouter: https://openrouter.ai/account/usage"
        echo "      - OpenAI: https://platform.openai.com/account/usage/overview"
        echo "      - Anthropic: https://console.anthropic.com/account/usage"
        echo "   4. 如已产生成本，考虑联系供应商请求退款"
        ;;
esac

echo ""
echo "=========================================="
echo ""

# ============================================================================
# 输出详细建议
# ============================================================================

if [ "$FALLBACK_STATUS" = "ENABLED" ]; then
    echo "📖 推荐阅读：PREVENT_PAID_MODELS.md"
    echo ""
    echo "快速修复："
    echo "  cp config-no-fallback.yaml ~/.hermes/config.yaml"
    echo ""
fi

if [ "$RECENT_PROVIDERS" = "PAID" ]; then
    echo "🚨 检测到付费 API 使用"
    echo ""
    echo "下一步："
    echo "  1. 检查最近的 API 账单"
    echo "  2. 禁用 fallback/API keys"
    echo "  3. 运行 'tail -f ~/.hermes/logs/agent.log' 监控使用"
    echo ""
fi

echo "完整报告见：PREVENT_PAID_MODELS.md"
echo ""
