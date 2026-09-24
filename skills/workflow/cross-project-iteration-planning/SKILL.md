---
name: cross-project-iteration-planning
description: 当需要为一个项目（A）推导后续迭代方向时，参考另一个项目（B）的架构模式。典型场景：项目A遇到了架构瓶颈，想要借鉴项目B的成熟设计。
trigger: "当用户说\"结合X为Y提供后续迭代方向\"或类似需求时"
---

# Cross-Project Iteration Planning Skill

## When to Use

当需要为一个项目（项目A）推导后续迭代方向时，参考另一个项目（项目B）的架构模式。

典型场景：
- 项目A遇到了架构瓶颈，想要借鉴项目B的成熟设计
- 需要快速为项目A提出5个迭代方向（P0-P4）
- 想从项目B的模块设计中找到项目A可以复用的模式

## Core Approach

### Step 1: 分析项目A的现有架构

列出：
- 项目A的现有模块和它们的功能
- 各模块的局限性（独立调用/无共享状态/无法协作等）
- 项目的技术栈和基础设施

### Step 2: 分析项目B的架构模式

找出项目B中可以借鉴的模块：
- Registry/注册模式
- 协作/Collaboration模式
- 共享上下文模式
- 任务编排/Orchestration模式
- 代理/Proxy模式

### Step 3: 建立映射关系

| 项目B的模块 | 项目A的借鉴 | 说明 |
|------------|-------------|------|
| acp_registry/ | agentRegistry | 角色注册表 |
| collaboration/ | CollaborationOrchestrator | 任务编排器 |

### Step 4: 提出迭代方向

通常提出5个方向，按价值/难度分级：
- P0: 最核心的差异化功能
- P1: 高价值中难度
- P2: 增强功能
- P3: 工具/体验优化
- P4: 未来探索

### Step 5: PRD起草 + 迭代确认

1. 起草PRD，包含：背景、架构图、核心类型、技术方案、目录结构
2. Boss确认细节（最多3轮问题）
3. 调整PRD（如精简Agent数量、改变复用策略等）
4. 进入开发阶段

## Key Decisions to Capture

当Boss确认PRD细节时，关键决策点包括：

1. **Agent数量**：精简 vs 完整（如5个→3个）
2. **复用策略**：
   - 抽取system prompt到registry（侵入性强）
   - registry代理模式（保持模块独立）
   - 直接复用（不做封装）
3. **UI范围**：哪些在当前版本完成
4. **向后兼容**：是否保留单AI模式

## 案例：ai-novel-assistant × hermes-agent

- 项目A: ai-novel-assistant（小说AI助手）
- 项目B: hermes-agent（CLI Agent框架）

**现有局限**：AI模块独立调用，无法协作

**借鉴模式**：
- hermes-agent的acp_registry → agentRegistry
- hermes-agent的collaboration → CollaborationOrchestrator
- hermes-agent的SharedContext → WritingContext

**PRD决策**：
- Agent精简为3个（PlotExpert/DialogueMaster/StyleGuard）
- 复用策略：registry代理模式（调用dialogueGenerator.ts等现有模块）
- Phase 4 UI全部完成