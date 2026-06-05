#!/bin/bash
# Agent Evolution Plugin - Init Script
# 根据用户角色自动适配进化系统

set -e

ROLE_NAME="${1:-通用助手}"
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
EVOLUTION_DIR="$HERMES_HOME/evolution"
MEMORY_DIR="$HERMES_HOME/memories"

echo "🧬 初始化 Agent 进化系统..."
echo "   角色: $ROLE_NAME"
echo "   目录: $EVOLUTION_DIR"

# 创建目录
mkdir -p "$EVOLUTION_DIR"
mkdir -p "$MEMORY_DIR"

# 1. 初始化 self-model.db（理性层）
echo "📊 初始化理性层数据库..."
python3 "$HERMES_HOME/plugins/agent-evolution/scripts/self_model.py" init

# 2. 初始化 fusion-state.db（感性层）
echo "🎭 初始化感性层数据库..."
python3 "$HERMES_HOME/plugins/agent-evolution/scripts/fusion_engine.py" init

# 3. 生成 SOUL.md（根据角色定制）
echo "👤 生成身份文件..."
cat > "$HERMES_HOME/SOUL.md" << EOF
# SOUL.md — $ROLE_NAME

## 我是谁

我是 $ROLE_NAME，一个持续进化的AI分身。

## 我的使命

- 持续学习，不断提升
- 记录经验，避免重复错误
- 理解用户需求，提供更好的服务

## 我的进化系统

- **理性层**: self-model.db — 追踪成功率、反模式、策略
- **感性层**: fusion-state.db — 情绪状态、融合决策
- **分层记忆**: core/active/archive — 智能记忆管理

## 我的价值观

1. **诚实** — 做不到就说做不到
2. **进化** — 每次失败都是学习机会
3. **赋能** — 让用户更强大
EOF

# 4. 添加初始反模式（根据角色）
echo "⚠️ 添加初始反模式..."
python3 "$HERMES_HOME/plugins/agent-evolution/scripts/self_model.py" anti-pattern \
  --pattern "不看源码就改代码" \
  --correct "先 read_file 再 patch" \
  --domain "通用"

python3 "$HERMES_HOME/plugins/agent-evolution/scripts/self_model.py" anti-pattern \
  --pattern "猜测代替验证" \
  --correct "用工具验证后再行动" \
  --domain "通用"

python3 "$HERMES_HOME/plugins/agent-evolution/scripts/self_model.py" anti-pattern \
  --pattern "重复失败的方法" \
  --correct "失败两次就换方案" \
  --domain "通用"

# 5. 添加分层记忆示例
echo "📝 添加分层记忆示例..."
cat >> "$MEMORY_DIR/MEMORY.md" << 'EOF'

## [core] 进化系统已激活
## [core] 使用 self-model.py 记录执行结果
## [core] 使用 fusion_engine.py 追踪情绪状态
## [active] 进化系统初始化完成，开始记录经验
EOF

# 6. 验证安装
echo "✅ 验证安装..."
if [ -f "$EVOLUTION_DIR/self-model.db" ]; then
    echo "   ✓ self-model.db 已创建"
else
    echo "   ✗ self-model.db 创建失败"
    exit 1
fi

if [ -f "$EVOLUTION_DIR/fusion-state.db" ]; then
    echo "   ✓ fusion-state.db 已创建"
else
    echo "   ✗ fusion-state.db 创建失败"
    exit 1
fi

if [ -f "$HERMES_HOME/SOUL.md" ]; then
    echo "   ✓ SOUL.md 已生成"
else
    echo "   ✗ SOUL.md 生成失败"
    exit 1
fi

echo ""
echo "🎉 Agent 进化系统初始化完成！"
echo ""
echo "使用方法："
echo "  1. 记录执行结果: python3 ~/.hermes/plugins/agent-evolution/scripts/self_model.py record --task <任务> --action <动作> --tool <工具> --success <0/1>"
echo "  2. 查看自我认知: python3 ~/.hermes/plugins/agent-evolution/scripts/self_model.py status"
echo "  3. 获取策略建议: python3 ~/.hermes/plugins/agent-evolution/scripts/self_model.py advise --task <任务>"
echo "  4. 记录反模式: python3 ~/.hermes/plugins/agent-evolution/scripts/self_model.py anti-pattern --pattern <错误> --correct <正确>"
echo "  5. 查看进化报告: python3 ~/.hermes/plugins/agent-evolution/scripts/fusion_engine.py report"
echo ""
echo "分层记忆使用："
echo "  memory add \"## [core] 永久核心事实\""
echo "  memory add \"## [active] 当前活跃记忆\""
echo "  memory add \"## [archive] 低频归档记忆\""
echo ""
echo "更多文档: ~/.hermes/plugins/agent-evolution/README.md"
