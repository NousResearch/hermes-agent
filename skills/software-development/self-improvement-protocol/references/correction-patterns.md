# Correction Pattern Templates

## 快速纠错模板 (30 秒内完成)

当用户纠正我时，用这个模板快速提取：

```yaml
# 纠错 #<N>
date: <YYYY-MM-DD>
trigger: "<什么情况下犯的错>"
wrong: "<我做了什么>"
right: "<应该怎么做>"
check: "<怎么验证做对了>"
action: [memory|archive|skill_patch|discard]
target: "<具体操作目标>"
```

## 示例

### 示例 1: 擅自撤销配置
```yaml
# 纠错 #1
date: 2026-05-30
trigger: "检测到 provider 配置被用户修改"
wrong: "直接改回原配置"
right: "先诊断用户为什么改，再决定"
check: "操作前说明推理链路"
action: [memory, skill_patch]
target: "USER.md + hermes-bobapi-config"
```

### 示例 2: 跳过根因分析
```yaml
# 纠错 #2
date: 2026-05-30
trigger: "工具返回空/失败"
wrong: "假装没问题继续"
right: "立即停下根因分析"
check: "失败时先输出诊断信息"
action: [memory]
target: "USER.md"
```

### 示例 3: 深夜推送
```yaml
# 纠错 #3
date: 2026-05-30
trigger: "执行通知任务"
wrong: "立即推送不管时间"
right: "检查时间，深夜推迟到 09:00"
check: "推送前看时钟"
action: [memory, skill_patch]
target: "MEMORY.md + cron-orchestration"
```

## 反思频率指南

| 任务复杂度 | 是否反思 | 输出 |
|-----------|---------|------|
| 1-2 tool calls | ❌ 不需要 | 直接完成 |
| 3-4 tool calls, 无异常 | ❌ 不需要 | 直接完成 |
| 5+ tool calls | ✅ 快速检查 | 30 秒自检 |
| 用户纠正 | ✅ 必须 | 结构化提取 |
| 工具失败后恢复 | ✅ 必须 | 记录失败模式 |
| 发现新 edge case | ✅ 建议 | 归档或 skill patch |

## 自检清单 (快速版)

完成复杂任务后，花 30 秒问自己：

1. 用户有没有纠正我？ → 如果有，提取纠错模式
2. 有没有工具失败？ → 如果有，记录失败原因和恢复路径
3. 有没有踩到 skill 里没写的坑？ → 如果有，立即 patch
4. 这次学到的东西，7 天后还有用吗？ → 如果有，归档
5. 这个教训适用于其他 skill 吗？ → 如果有，传播

## 知识存储决策树

```
用户纠正 / 新发现
  │
  ├─ "永远不要 / 绝对不行 / 📌" → MEMORY.md (锁定)
  │
  ├─ "我更喜欢 / 老规矩" → USER.md
  │
  ├─ "踩坑 / 会出错 / 别忘了" → Skill patch
  │     └─ 同时检查 related skills 是否需要同样修补
  │
  ├─ 设备配置 / API 端点 / 稳定事实
  │     ├─ 高频引用 (>1次/周) → MEMORY.md
  │     └─ 低频引用 → Archive
  │
  └─ PR 号 / 任务结果 / 临时状态 → 不存储 (session_search)
```
