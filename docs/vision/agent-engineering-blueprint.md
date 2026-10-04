# Agent 工程化开发执行架构

> 目标：把 Codex、Claude Code、OpenCode 等 Agent 开发工具，从“会写代码的助手”升级为一套可重复、可验证、可维护的软件开发执行系统。
>
> 本文档作为长期建设蓝图使用，适合逐步落地，不要求一次性完成全部能力。

---

# 1. 建设目标

当前已有一套较完整的全局开发规则，覆盖：

- 需求理解
- 事实与假设区分
- 修改前检查
- 最小改动原则
- 长期维护
- 基于证据排障
- 高风险操作控制
- 完成前验证
- 输出规范

这些规则主要解决：

> Agent 应该以什么原则工作。

下一阶段需要进一步解决：

> Agent 如何稳定、自动、可重复地执行开发任务。

最终目标是形成：

```text
User Requirement
        │
        ▼
Global Development Rules
        │
        ▼
Project Context / AGENTS.md
        │
        ▼
Task Workflow / Skills
        │
        ▼
Planning / ExecPlan
        │
        ▼
Implementation
        │
        ▼
Automated Verification
        │
        ▼
Review / CI
        │
        ▼
Result Report
```

核心原则：

```text
规则负责约束
项目上下文负责提供事实
Skill 负责定义流程
工具负责提供真实数据
Hook/CI 负责强制验证
Subagent 负责分工
ExecPlan 负责复杂任务
```

---

# 2. 总体架构

推荐最终架构：

```text
Developer
    │
    ▼
Agent Runtime
Codex / Claude Code / OpenCode
    │
    ├── Global Rules
    │
    ├── Project Context
    │     └── AGENTS.md
    │
    ├── Skills
    │     ├── bugfix
    │     ├── feature
    │     ├── payment-integration
    │     ├── api-integration
    │     ├── database-change
    │     ├── deploy
    │     └── code-review
    │
    ├── Planning
    │     └── ExecPlan / PLANS.md
    │
    ├── Subagents
    │     ├── explorer
    │     ├── debugger
    │     ├── tester
    │     ├── reviewer
    │     └── database-reviewer
    │
    ├── Tools / MCP
    │     ├── GitHub
    │     ├── Documentation
    │     ├── Database ReadOnly
    │     ├── Logs
    │     └── Browser
    │
    ├── Automation
    │     ├── Taskfile / Makefile
    │     ├── Hooks
    │     ├── pre-commit
    │     └── Scripts
    │
    └── CI/CD
          ├── Build
          ├── Test
          ├── Lint
          ├── Security Check
          └── Deployment Gate
```

---

# 3. 分层设计

## 3.1 Global Rules

### 职责

Global Rules 只定义：

- 思考方式
- 风险控制
- 开发原则
- 验证要求
- 输出要求

不要加入大量项目细节。

例如不应该放入：

```text
本项目使用 PostgreSQL
支付模块位于 payment-business
运行 mvn clean package
```

这些属于项目级上下文。

### Global Rules 应保持稳定

推荐更新频率：

```text
低频更新
```

只有发现长期通用问题时才修改。

---

# 4. 项目级上下文：AGENTS.md

## 4.1 作用

`AGENTS.md` 负责回答：

```text
这个项目是什么？
如何运行？
如何构建？
代码在哪里？
架构是什么？
哪些规则是项目特有的？
哪些地方不能随便修改？
```

推荐目录：

```text
project/
├── AGENTS.md
├── README.md
├── docs/
├── scripts/
├── src/
└── ...
```

大型项目可以进一步拆分：

```text
project/
├── AGENTS.md
│
├── payment/
│   └── AGENTS.md
│
├── admin/
│   └── AGENTS.md
│
└── web/
    └── AGENTS.md
```

---

## 4.2 推荐 AGENTS.md 模板

```md
# Project Overview

项目名称：
项目用途：

## Technology Stack

- Java 17
- Spring Boot
- MyBatis Plus
- PostgreSQL
- Redis

## Project Structure

说明主要模块职责。

## Development Commands

Build:

```bash
./mvnw clean package
```

Test:

```bash
./mvnw test
```

Run:

```bash
./mvnw spring-boot:run
```

## Architecture Rules

- Controller 不放业务逻辑。
- Service 负责业务编排。
- Repository / Mapper 负责数据访问。
- 优先复用已有公共组件。
- 禁止无必要创建平行业务流程。

## Security Rules

- 不记录密钥。
- 不输出完整支付凭证。
- 敏感字段日志需要脱敏。

## Definition of Done

- 编译通过
- 相关测试通过
- Git Diff 已检查
- 无明显兼容性问题
- 无敏感信息泄露
```

---

# 5. Definition of Done

建议每个项目统一定义完成标准。

例如：

```md
## Definition of Done

A task is not complete until:

- Relevant existing implementation has been inspected.
- The root cause or requirement has been identified.
- Changes are minimal and scoped.
- Code compiles successfully.
- Relevant tests pass.
- No obvious security regression exists.
- Git diff has been reviewed.
- No secret has been introduced.
- API compatibility has been evaluated where applicable.
- Database compatibility has been evaluated where applicable.
- Verification results have been reported.
```

目标是防止：

```text
代码写完 = 任务完成
```

正确应该是：

```text
实现
+
测试
+
验证
+
Review
=
任务完成
```

---

# 6. Skills：把重复工作流程化

## 6.1 Skills 的定位

Skills 不负责存储项目事实。

Skills 负责：

> 某一类任务应该按照什么流程执行。

推荐目录：

```text
.agent/
└── skills/
    ├── bugfix/
    │   └── SKILL.md
    │
    ├── feature/
    │   └── SKILL.md
    │
    ├── api-integration/
    │   └── SKILL.md
    │
    ├── payment-integration/
    │   └── SKILL.md
    │
    ├── database-change/
    │   └── SKILL.md
    │
    ├── deploy/
    │   └── SKILL.md
    │
    └── code-review/
        └── SKILL.md
```

---

# 7. 推荐的核心 Skills

## 7.1 bugfix

标准流程：

```text
1. 收集错误信息
2. 找到第一处有效异常
3. 根据日志 / traceId 还原调用链
4. 定位相关代码
5. 分析 Root Cause
6. 检查是否已有类似实现
7. 设计最小修复方案
8. 修改代码
9. 执行测试
10. 检查 Git Diff
11. 输出：
    - Root Cause
    - Fix
    - Verification
    - Remaining Risk
```

---

# 8. feature Skill

功能开发推荐流程：

```text
Requirement
    │
    ▼
Existing Implementation Search
    │
    ▼
Impact Analysis
    │
    ▼
Design
    │
    ▼
Implementation
    │
    ▼
Tests
    │
    ▼
Compatibility Check
    │
    ▼
Review
```

重点检查：

- 是否已有类似功能
- 是否可以扩展已有模块
- 是否影响历史数据
- 是否影响已有 API
- 是否需要数据库迁移
- 是否需要权限控制
- 是否需要配置项

---

# 9. payment-integration Skill

对于支付类项目，建议单独建设。

标准流程：

```text
第三方文档
    │
    ▼
接口识别
    │
    ├── Payin
    ├── Payout
    ├── Query
    ├── Callback
    └── Refund
    │
    ▼
认证方式分析
    │
    ├── API Key
    ├── Signature
    ├── HMAC
    └── RSA
    │
    ▼
字段映射
    │
    ▼
状态映射
    │
    ▼
项目现有支付架构分析
    │
    ▼
实现
    │
    ▼
签名测试
    │
    ▼
下单测试
    │
    ▼
查询测试
    │
    ▼
回调测试
    │
    ▼
失败场景测试
```

必须检查：

```text
amount unit
currency
idempotency
merchantOrderId
channelOrderId
signature
callback validation
callback replay
duplicate notification
timeout
query compensation
status mapping
```

---

# 10. api-integration Skill

适用于普通第三方 API。

流程：

```text
1. 阅读官方文档
2. 识别认证方式
3. 找现有 HTTP Client
4. 找现有 SDK Wrapper
5. 定义 DTO
6. 定义错误映射
7. 配置 timeout
8. 配置 retry
9. 配置日志脱敏
10. 实现
11. 测试
```

避免每次重复创建：

```text
HttpClient
JSON Parser
Retry Logic
Auth Wrapper
```

应优先复用项目已有组件。

---

# 11. database-change Skill

数据库变更属于高风险操作。

执行前必须检查：

```text
表结构
历史数据
索引
约束
默认值
NULL
兼容性
迁移脚本
重复执行安全
回滚
```

标准流程：

```text
Schema Inspection
       │
       ▼
Existing Data Analysis
       │
       ▼
Migration Design
       │
       ▼
Backward Compatibility
       │
       ▼
Index / Constraint
       │
       ▼
Rollback
       │
       ▼
Execute
       │
       ▼
Verify
```

---

# 12. deploy Skill

部署流程建议标准化：

```text
Build
  │
  ▼
Artifact Check
  │
  ▼
Backup
  │
  ▼
Deploy
  │
  ▼
Health Check
  │
  ▼
Logs
  │
  ▼
API Smoke Test
  │
  ▼
Rollback Check
```

禁止直接：

```text
kill
rm
覆盖 jar
restart
```

而没有备份与回滚。

---

# 13. ExecPlan / PLANS.md

## 13.1 什么时候需要

简单任务：

```text
直接执行
```

例如：

```text
修改一个文案
修正一个变量名
调整单个配置
```

复杂任务：

```text
先建立 ExecPlan
```

例如：

- 新支付渠道
- 新认证流程
- 数据库结构变更
- 核心架构调整
- 跨模块功能
- 大型重构
- 数据迁移

---

# 14. ExecPlan 推荐模板

```md
# Objective

这次任务真正要解决什么问题。

# Existing Implementation

当前相关代码和架构。

# Scope

会修改：

- xxx
- xxx

不会修改：

- xxx

# Risks

- 数据兼容
- API 兼容
- 权限
- 安全

# Proposed Solution

推荐方案。

# Files

预计涉及：

- xxx.java
- xxx.xml
- xxx.sql

# Database Changes

是否需要数据库修改。

# Implementation Steps

1.
2.
3.

# Verification

- build
- test
- API test
- logs

# Rollback

回滚方式。

# Open Questions

尚未确认的问题。
```

---

# 15. Taskfile / Makefile

## 15.1 目的

不要让 Agent 每次自己猜：

```text
如何构建
如何测试
如何启动
如何格式化
```

统一成：

```text
task build
task test
task lint
task verify
task dev
task docker:up
task docker:down
```

---

# 16. 推荐 Taskfile 结构

```yaml
version: '3'

tasks:

  build:
    cmds:
      - ./mvnw clean package

  test:
    cmds:
      - ./mvnw test

  lint:
    cmds:
      - ./mvnw spotless:check

  verify:
    deps:
      - lint
      - test
      - build

  dev:
    cmds:
      - ./mvnw spring-boot:run
```

对于多技术栈项目：

```text
backend
frontend
docker
database
```

都可以统一入口。

---

# 17. scripts 目录

推荐：

```text
scripts/
├── verify.sh
├── build.sh
├── deploy.sh
├── rollback.sh
├── health-check.sh
└── db-migrate.sh
```

Agent 最终只需要执行：

```bash
./scripts/verify.sh
```

而不用每次理解复杂命令。

---

# 18. Hooks

Global Rules 属于：

```text
Soft Constraint
```

Hooks 属于：

```text
Hard Constraint
```

例如：

```text
修改 Java 文件
     │
     ▼
Formatter
     │
     ▼
Lint
```

提交前：

```text
git commit
    │
    ▼
lint
test
secret scan
```

---

# 19. pre-commit

推荐加入：

```text
format
lint
secret detection
large file detection
```

避免 Agent：

```text
误提交
.env
private key
token
password
```

---

# 20. CI

本地 Agent 验证之外，还需要 CI 二次验证。

推荐：

```text
Pull Request
    │
    ▼
Build
    │
    ▼
Unit Tests
    │
    ▼
Integration Tests
    │
    ▼
Lint
    │
    ▼
Security Scan
    │
    ▼
Review
    │
    ▼
Merge
```

原则：

```text
Agent 说成功
≠
真正成功

CI 通过
+
Review
=
可以合并
```

---

# 21. MCP / Tool Layer

Agent 应尽可能直接读取真实信息。

目标：

```text
Agent
 ├── GitHub
 ├── Docs
 ├── Database
 ├── Logs
 ├── Browser
 └── Internal API
```

减少：

```text
Developer
复制日志
复制 SQL
复制 Issue
复制 API 文档
复制 Git Diff
```

---

# 22. MCP 权限等级

建议分三级。

## Level 1：Read Only

默认开放：

```text
GitHub Read
Database Read
Logs Read
Docs Read
```

风险较低。

---

## Level 2：Development Write

可以有限开放：

```text
create branch
modify files
run tests
create PR
```

---

## Level 3：Production

默认要求人工批准：

```text
Database Write
Deploy
Delete
Restart
Secret Change
Permission Change
```

原则：

```text
Prompt Constraint
+
Tool Permission
+
CI Gate
```

形成多层安全控制。

---

# 23. Subagents

不要让一个 Agent 承担全部工作。

推荐：

```text
Main Agent
    │
    ├── Explorer
    │
    ├── Debugger
    │
    ├── Tester
    │
    ├── Reviewer
    │
    └── Database Reviewer
```

---

# 24. Explorer Agent

职责：

```text
查找相关代码
分析项目结构
追踪调用链
寻找已有实现
```

限制：

```text
Read Only
```

不修改代码。

---

# 25. Debugger Agent

职责：

```text
日志分析
异常定位
配置检查
调用链分析
Root Cause 分析
```

输出：

```text
Evidence
Root Cause
Affected Code
Suggested Fix
```

---

# 26. Tester Agent

职责：

```text
分析修改
补测试
执行测试
寻找遗漏场景
```

重点覆盖：

```text
normal case
boundary case
failure case
duplicate request
timeout
concurrency
```

---

# 27. Reviewer Agent

职责：

```text
Review Git Diff
```

检查：

```text
security
compatibility
maintainability
unnecessary refactor
edge cases
error handling
logging
```

---

# 28. Database Reviewer

专门检查：

```text
migration
index
constraint
default
NULL
existing data
rollback
idempotency
```

数据库任务建议必经该 Agent。

---

# 29. 推荐开发生命周期

最终统一成：

```text
                 User Task
                     │
                     ▼
              1. Understand
                     │
                     ▼
               2. Explore
                     │
            ┌────────┴────────┐
            │                 │
         Simple            Complex
            │                 │
            │            3. ExecPlan
            │                 │
            └────────┬────────┘
                     ▼
               4. Implement
                     │
                     ▼
                  5. Test
                     │
                     ▼
                  6. Build
                     │
                     ▼
                 7. Review
                     │
                     ▼
               8. Git Diff
                     │
                     ▼
                9. Report
```

---

# 30. 标准 Bugfix 生命周期

```text
Bug Report
    │
    ▼
Logs / Error
    │
    ▼
Root Cause
    │
    ▼
Existing Code
    │
    ▼
Minimal Fix
    │
    ▼
Test
    │
    ▼
Review
    │
    ▼
Verification
```

---

# 31. 标准 Feature 生命周期

```text
Requirement
    │
    ▼
Existing Architecture
    │
    ▼
Impact Analysis
    │
    ▼
ExecPlan
    │
    ▼
Implementation
    │
    ▼
Unit Test
    │
    ▼
Integration Test
    │
    ▼
Compatibility
    │
    ▼
Review
```

---

# 32. 项目文档体系

推荐：

```text
docs/
├── architecture/
│   ├── overview.md
│   ├── auth.md
│   ├── payment.md
│   └── database.md
│
├── api/
│
├── runbooks/
│   ├── deploy.md
│   ├── rollback.md
│   └── incident.md
│
└── adr/
    ├── ADR-001-xxx.md
    ├── ADR-002-xxx.md
    └── ADR-003-xxx.md
```

---

# 33. Architecture Docs

记录：

```text
系统模块
模块职责
调用关系
核心数据流
基础设施
外部依赖
```

Agent 可以快速理解项目。

---

# 34. ADR

ADR：

```text
Architecture Decision Record
```

主要回答：

```text
为什么这样设计？
```

例如：

```text
ADR-001-use-postgresql.md
ADR-002-payment-channel-abstraction.md
ADR-003-use-redis-cache.md
```

推荐模板：

```md
# Context

为什么需要做这个决定。

# Decision

最终采用什么方案。

# Alternatives

有哪些方案。

# Reason

为什么选择当前方案。

# Consequences

长期影响。
```

---

# 35. Runbook

Runbook 负责运维操作：

```text
deploy
rollback
restart
backup
incident
database recovery
```

这样 Agent 不应该自己临时创造生产操作流程。

---

# 36. Prompt Shortcuts

可建立快捷任务：

```text
/bugfix
/feature
/review
/refactor
/api-integration
/payment-integration
/database-change
/deploy
/incident
```

例如：

```text
/bugfix
```

自动执行：

```text
Reproduce
→ Evidence
→ Root Cause
→ Existing Implementation
→ Minimal Fix
→ Test
→ Review
→ Report
```

---

# 37. 推荐项目目录

完整目录示例：

```text
project/
│
├── AGENTS.md
├── README.md
│
├── .agent/
│   │
│   ├── PLANS.md
│   │
│   ├── skills/
│   │   ├── bugfix/
│   │   ├── feature/
│   │   ├── api-integration/
│   │   ├── payment-integration/
│   │   ├── database-change/
│   │   ├── deploy/
│   │   └── review/
│   │
│   └── prompts/
│       ├── bugfix.md
│       ├── feature.md
│       └── review.md
│
├── docs/
│   ├── architecture/
│   ├── api/
│   ├── adr/
│   └── runbooks/
│
├── scripts/
│   ├── build.sh
│   ├── test.sh
│   ├── verify.sh
│   ├── deploy.sh
│   └── rollback.sh
│
├── Taskfile.yml
│
└── src/
```

---

# 38. 第一阶段实施计划

目标：

> 先解决 Agent 不理解项目的问题。

实施：

```text
1. 保留当前 Global Rules
2. 每个主项目建立 AGENTS.md
3. 添加 Definition of Done
4. 建立 docs/architecture/overview.md
```

工作量较低，但收益很高。

验收标准：

```text
Agent 第一次进入项目
无需大量提问
可以知道：
- 技术栈
- 项目结构
- 构建方式
- 测试方式
- 架构限制
```

---

# 39. 第二阶段实施计划

目标：

> 统一执行命令。

实施：

```text
1. Taskfile / Makefile
2. scripts/build
3. scripts/test
4. scripts/verify
```

最终目标：

```bash
task verify
```

可以完成：

```text
format
lint
test
build
```

验收：

```text
Agent 无需自己拼构建命令。
```

---

# 40. 第三阶段实施计划

目标：

> 把高频任务标准化。

先做 4 个 Skill：

```text
bugfix
feature
api-integration
payment-integration
```

不要一次做太多。

优先做每天最常出现的任务。

---

# 41. 第四阶段实施计划

目标：

> 建立自动质量门禁。

增加：

```text
pre-commit
hooks
CI
```

最少要求：

```text
lint
test
build
secret scan
```

---

# 42. 第五阶段实施计划

目标：

> 给 Agent 接真实数据源。

逐步增加：

```text
GitHub
Docs
Database ReadOnly
Logs
```

原则：

```text
Read First
Write Later
```

优先只读。

---

# 43. 第六阶段实施计划

目标：

> 引入多 Agent 协同。

最初只增加：

```text
Explorer
Reviewer
```

稳定后增加：

```text
Debugger
Tester
Database Reviewer
```

避免一开始设计过度复杂。

---

# 44. 第七阶段实施计划

目标：

> 建立长期项目知识库。

完善：

```text
architecture
ADR
runbook
API docs
```

使 Agent 能理解：

```text
What
How
Why
```

---

# 45. 推荐建设顺序

按投入产出比：

```text
Phase 1
Global Rules
+
AGENTS.md
+
Definition of Done

        ↓

Phase 2
Taskfile
+
verify scripts

        ↓

Phase 3
Skills

        ↓

Phase 4
Hooks
+
CI

        ↓

Phase 5
MCP / Tools

        ↓

Phase 6
Subagents

        ↓

Phase 7
Architecture Docs
+
ADR
+
Runbook
```

---

# 46. 不建议一开始做的事情

不要：

```text
一次创建 20 个 Skill
```

维护成本很高。

不要：

```text
给 Agent 开 production root 权限
```

风险太大。

不要：

```text
所有任务都强制生成复杂 Plan
```

会降低小任务效率。

不要：

```text
把所有知识写进 Global Rules
```

上下文会越来越重。

不要：

```text
多个 Agent 同时修改同一区域代码
```

容易产生冲突。

---

# 47. 一个比较合理的成熟状态

最终理想流程：

```text
Developer：

新增 xxx 支付渠道

        │
        ▼

Agent：

读取 Global Rules

        │
        ▼

读取 AGENTS.md

        │
        ▼

调用 payment-integration Skill

        │
        ▼

Explorer 查现有支付架构

        │
        ▼

生成 ExecPlan

        │
        ▼

实现代码

        │
        ▼

task verify

        │
        ▼

Tester 验证

        │
        ▼

Reviewer Review Diff

        │
        ▼

CI

        │
        ▼

生成实施报告
```

这时候开发者主要负责：

```text
业务决策
架构决策
风险批准
最终 Review
```

Agent 负责：

```text
搜索
分析
实现
测试
验证
文档
```

---

# 48. 长期可以继续扩展

未来可以增加：

## 自动 Issue → PR

```text
GitHub Issue
    │
    ▼
Agent
    │
    ▼
Branch
    │
    ▼
Implementation
    │
    ▼
Tests
    │
    ▼
Draft PR
```

---

## 自动 PR Review

```text
Pull Request
    │
    ▼
Reviewer Agent
    │
    ├── Security
    ├── Compatibility
    ├── Tests
    └── Architecture
```

---

## 自动 Incident Analysis

```text
Alert
  │
  ▼
Logs
  │
  ▼
Debugger Agent
  │
  ▼
Root Cause
  │
  ▼
Suggested Fix
```

---

## 自动依赖升级

```text
Dependency Update
       │
       ▼
Compatibility Check
       │
       ▼
Build
       │
       ▼
Tests
       │
       ▼
PR
```

---

# 49. 成熟度模型

## Level 0

```text
Chat
```

Agent 只是聊天。

---

## Level 1

```text
Rules
```

Agent 有开发规范。

---

## Level 2

```text
Project Context
```

Agent 理解项目。

---

## Level 3

```text
Workflow
```

Skills 标准化任务。

---

## Level 4

```text
Automation
```

Hooks + Task + CI。

---

## Level 5

```text
Tool Integration
```

Agent 能直接读取：

```text
GitHub
DB
Logs
Docs
```

---

## Level 6

```text
Multi-Agent
```

Agent 分工协作。

---

## Level 7

```text
Engineering System
```

完整：

```text
Plan
Implement
Verify
Review
Deploy
Observe
```

---

# 50. 当前最推荐的第一步

现阶段优先执行：

```text
1. 保留当前 Global Development Rules
2. 给主项目建立 AGENTS.md
3. 定义 Definition of Done
4. 建立 task verify
5. 做第一个 bugfix Skill
6. 做 payment-integration Skill
```

先不要急着做：

```text
复杂 MCP
大量 Subagent
自动生产部署
```

原因是前三项主要解决：

```text
Agent 执行一致性
```

而 MCP / 多 Agent 更多解决：

```text
效率和规模
```

正确顺序应该是：

```text
先稳定
再自动化
最后扩规模
```

---

# 51. 核心设计原则

最终整个体系始终遵循：

```text
Global Rules
    ↓
告诉 Agent 应该怎么思考

AGENTS.md
    ↓
告诉 Agent 项目实际是什么

Skills
    ↓
告诉 Agent 某类任务怎么执行

ExecPlan
    ↓
告诉 Agent 这一次具体准备怎么做

MCP / Tools
    ↓
让 Agent 获取真实数据

Task / Scripts
    ↓
提供稳定执行入口

Hooks / CI
    ↓
强制执行质量标准

Subagents
    ↓
拆分复杂任务

Docs / ADR / Runbook
    ↓
保存长期工程知识
```

最终目标不是：

```text
让 Agent 写更多代码
```

而是：

```text
让 Agent 以稳定、可验证、可维护的方式完成工程任务。
```

---

# 52. 实施 Checklist

可以按照下面顺序逐项完成。

## 基础层

- [ ] Global Rules 已整理
- [ ] 主项目 AGENTS.md
- [ ] Definition of Done
- [ ] architecture overview

## 执行层

- [ ] Taskfile
- [ ] build task
- [ ] test task
- [ ] lint task
- [ ] verify task

## Skill 层

- [ ] bugfix
- [ ] feature
- [ ] api-integration
- [ ] payment-integration
- [ ] database-change
- [ ] deploy
- [ ] review

## 自动化层

- [ ] formatter
- [ ] lint
- [ ] pre-commit
- [ ] secret scan
- [ ] CI

## Tool 层

- [ ] GitHub
- [ ] Docs
- [ ] Database ReadOnly
- [ ] Logs
- [ ] Browser

## Multi-Agent

- [ ] Explorer
- [ ] Reviewer
- [ ] Debugger
- [ ] Tester
- [ ] Database Reviewer

## Knowledge

- [ ] architecture
- [ ] ADR
- [ ] runbooks
- [ ] API docs

---

# 53. 后续维护原则

建议每隔一段时间检查：

```text
哪些规则 Agent 经常违反？
哪些任务每天重复出现？
哪些 Skill 使用率很低？
哪些脚本已经过期？
哪些 CI 检查耗时过高？
哪些文档与实际代码不一致？
```

只有高频、稳定、可重复的流程才值得 Skill 化。

只有确定性要求才值得 Hook 化。

只有长期有效的知识才值得写入项目文档。

---

# 54. 最终判断标准

如果以后能够做到：

```text
一个新 Agent 第一次进入项目
```

只需要读取：

```text
Global Rules
AGENTS.md
Architecture Docs
Skills
```

就能：

```text
理解项目
找到代码
设计方案
实施修改
运行验证
Review Diff
报告结果
```

说明这套 Agent 工程开发体系已经基本成熟。
