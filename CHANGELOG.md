# Hermes Agent - 变更记录 (Changelog)

> 本文档记录 Hermes Agent 的重要变更历史，包括功能更新、bug 修复、文档改进等。

---

## 2026-04-19 - 同步上游最新变更 (v2026.4.19) 🚀

### 🔧 核心修复
- **Gateway 竞态条件修复**：修复 base adapter 中的 pending-drain 和 late-arrival 竞态条件
  - R5 (HIGH)：修复可能导致重复 agent spawn 的竞态条件
  - R6 (MED-HIGH)：修复可能导致消息丢失的 finally cleanup 问题

### ⚙️ 新增功能
- **Cron 审批模式**：新增 `approvals.cron_mode` 配置选项
  - `deny`（默认）：阻止危险命令，提示 agent 寻找替代方案
  - `approve`：自动批准所有危险命令（旧行为）
  - 21 个新测试覆盖配置解析和行为

### 🔐 安全改进
- **Codex 认证改进**：Hermes 现在拥有自己的 Codex 认证状态
  - 移除与 Codex CLI 共享 `~/.codex/auth.json` 的机制
  - 修复并发使用导致的 token 刷新竞态条件
  - 改进错误处理和 401/403 响应对应

### 🧹 内容处理
- **思考标签处理改进**：全面改进各种推理标签的处理
  - 修复 `<thinking>`、`<thought>`、`<reasoning>` 标签泄漏到下游
  - 在存储边界统一处理，影响消息平台、会话转录、上下文压缩等
  - 添加对未闭合标签和孤立关闭标签的处理
  - 新增回归测试覆盖各种标签变体

### 🗑️ 用户体验
- **卸载流程改进**：改进 `hermes uninstall` 命令
  - 正确停止和销毁 gateway
  - 提供移除命名配置文件的选项

### 🎨 TUI 改进
- 新增 `LIGHT_THEME` 预设，支持白色/浅色终端背景
- `/clear` 和 `/new` 命令的双击确认机制
- 改进确认窗口超时（3s → 30s）
- `/model` 选择器中稳定 React key 和行歧义处理
- 状态栏中显示 git 分支
- Ctrl+C 在输入选择时复制到剪贴板
- 改进 `/skills` 浏览为格式化面板

### 🔧 其他修复
- 修复斜杠命令无法中断运行中的 agent
- 修复 Telegram 中 `from_user=None` 的回退处理
- 改进上下文压缩器中的 JSON 有效性
- 修复 Gemini 模型的 Bearer auth 和低 TPM 模型隐藏
- 防止流式累加器中的工具名称重复

### 🎯 新增技能
- **TouchDesigner MCP**：重写 TouchDesigner 集成，移至 optional-skills
- **X URL**：替换 xitter，使用官方 X API CLI
- **Baoyu Infographic**：21 种布局 × 21 种样式

### 📚 文档更新
- 更新 Anthropic 控制台 URL 到 platform.claude.com

---

## 2026-04-16 - 同步上游 v2026.4.13 版本变更 🚀

### ✅ 新增功能
- **备份和导入系统**：完整的 `hermes backup` 和 `hermes import` 命令
- **快照功能**：`hermes backup --quick` 和 `/snapshot` 斜杠命令
- **OAuth 提供商管理**：Web Dashboard 中的 OAuth 登录功能
- **Kimi/Moonshot 提供商**：支持中国地区的模型提供商
- **调试命令**：`/debug` 斜杠命令和 `hermes debug share`
- **提示功能**：会话开始时显示随机提示（279 条提示）
- **模型选择器**：原生 `/model` 选择器模态框
- **网络配置**：`network.force_ipv4` 配置选项
- **组件化日志**：带会话上下文和过滤功能的日志系统

### 🔧 修复和改进
- 修复 SQLite 备份安全：使用 `sqlite3.Connection.backup()` API
- 修复 78 个 CI 测试：移除死代码和修复测试失败
- 修复 30+ 文档错误：全面的文档更新

---

## 2026-04-13 - 代码变更同步更新 🔄

### ✅ 新增功能
- **Web UI Dashboard**：基于 React 19 + Vite + Tailwind CSS v4 的浏览器管理界面
- **9 个页面**：Status、Config、Env、Sessions、Skills、Cron、Logs、Analytics 页面
- **新命令**：`hermes web` 启动 Web UI Dashboard

### 🔧 改进和修复
- **Gateway 改进**：重启通知机制、服务重启支持（systemd）
- **安全修复**：Home Assistant 工具的路径遍历验证
- **模型名称修复**：保留 OpenCode Zen 和 ZAI 提供商模型名称中的点
- **WhatsApp UX 改进**：分块、格式化、流式输出改进
- **Telegram 修复**：使用 UTF-16 代码单元进行消息长度分割
- **Web 服务器**：FastAPI 后端 + React 前端的完整 Web 界面

### 📖 文档更新
- 更新技术栈、模块结构图和核心命令

---

## 2026-04-08 - 完善 ADR 体系 📋

### ✅ 新增内容
- **新增 3 个 ADR**：命令系统、会话管理、技能系统
- **添加视觉化图表**：为所有 ADR 添加 10 个 Mermaid 图表
- **创建 ADR 导航图**：在根 CLAUDE.md 中添加 ADR 索引和思维导图
- **建立反馈机制**：添加 ADR 投票、评分和贡献指南
- **创建变更日志**：记录 ADR 的演进历史和未来计划

---

## 2026-04-08 - 添加模块交互流程图 🎨

为以下流程添加了详细的 Mermaid 图表：
- 工具调用流程
- 消息调度流程
- 配置加载流程
- 记忆管理流程

---

## 2026-04-08 - 模块文档完成 🎉

### ✅ 文档创建
- **创建所有模块文档**：为 8 个核心模块创建完整的 CLAUDE.md
- **文档覆盖率 100%**：所有核心模块都有详细文档
- **面包屑导航**：每个模块文档都包含面包屑导航
- **接口文档**：详细的接口说明和使用示例

### 🎯 改进
- **索引更新**：更新索引文件反映当前文档状态
- **下一步建议**：创建测试套件覆盖报告和性能优化指南

---

## 2026-04-08 - 初始化 AI 上下文文档 🚀

### ✅ 基础文档
- **创建根级文档**：生成项目级 CLAUDE.md
- **项目分析**：识别 9 个核心模块
- **架构文档**：详细说明技术栈、架构模式和设计原则
- **开发指南**：提供运行、测试、编码规范和 AI 使用指引
- **模块索引**：创建模块结构图和索引表

### 🎯 下一步
- 需要为每个模块创建详细的 CLAUDE.md 文档（已完成）

---

*提示：本变更记录仅包含重要的功能和文档更新。完整的提交历史请查看 Git 仓库。*
