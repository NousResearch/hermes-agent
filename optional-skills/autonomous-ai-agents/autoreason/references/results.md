# Autoreason 论文核心结果速查

## 规模缩放曲线（CodeContests private-test）

- **Haiku 3.5**: 单次 ~31% → Autoreason ~40%（+9%）
- **Haiku 4.5**: 单次 ~60% → Autoreason ~60%（~0%，转折点）
- **Sonnet 4**: 单次 ~61% → Autoreason ~64%（+3%）
- **Sonnet 4.6**: 单次 ~73% → Autoreason ~77%（+4%）

关键洞察: Haiku 4.5 处 gain 消失——模型"生成能力"≈"评判能力"时迭代优化空间闭合。

## 为什么 autoreason 有效

- "不做改动"（A）是一等选项
- 每个 agent 都是 fresh（无上下文污染）
- 盲审 + Borda count 消除位置偏差
- 三个独立声音而非"自己找问题自己修"

## 消融实验

- 去掉 B 或 AB: 收敛极快（2-3轮）但质量差
- Judge 数量: 7 > 3 > 1
- Borda vs Majority: Borda 更稳定
