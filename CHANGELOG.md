# Hermes Agent - 变更记录 (Changelog)

> 本文档记录 Hermes Agent 的重要变更历史，包括功能更新、bug 修复、文档改进等。

---

## 2026-04-24 - 同步上游最新变更 (v0.11.0/v2026.4.23) 🚀

### 🎉 重大版本 - "The Interface release"
- **1,556 个提交 · 761 个 PR · 290 位贡献者**
- 这是 Hermes Agent 的一个重大里程碑版本

### 🖥️ 全新的 Ink-based TUI
- **完整的 React/Ink 重写的交互式 CLI**
  - 通过 `hermes --tui` 或 `HERMES_TUI=1` 启动
  - ~310 个提交到 `ui-tui/` 和 `tui_gateway/`
- **核心特性**：
  - 粘性编辑器（Sticky composer）
  - 实时流式传输
  - OSC-52 剪贴板支持（跨 SSH 会话复制）
  - 稳定的选择器键
  - 状态栏显示 git 分支和每轮计时器
  - `/clear` 确认、浅色主题预设
  - 子代理生成可观察性覆盖层
  - 虚拟化历史渲染
  - 斜杠命令自动补全
  - 路径自动补全

### 🏗️ 传输层架构重构
- **可插拔的 `agent/transports/` 层**
  - 抽象格式转换和 HTTP 传输
  - 从 `run_agent.py` 中提取传输逻辑
- **新传输类型**：
  - `AnthropicTransport` - Anthropic Messages API
  - `ChatCompletionsTransport` - OpenAI 兼容提供商默认路径
  - `ResponsesApiTransport` - OpenAI Responses API + Codex
  - `BedrockTransport` - AWS Bedrock Converse API

### 🌐 五个新的推理路径
- **NVIDIA NIM** - 原生提供商支持
- **Arcee AI** - 直接提供商
- **Step Plan** - 新推理提供商
- **Google Gemini CLI OAuth** - 推理提供商
- **Vercel ai-gateway** - 带定价、归因和动态发现
- **Gemini AI Studio API** - 原生路由以获得更好性能

### 🤖 GPT-5.5 支持
- **通过 Codex OAuth 支持 OpenAI GPT-5.5 推理模型**
- 实时模型发现集成到模型选择器中
- 新的 OpenAI 发布无需目录更新即可显示

### 📱 QQBot - 第 17 个消息平台
- **原生 QQBot 适配器**
  - QQ 官方 API v2
  - QR 扫描配置向导
  - 流式游标、表情反应
  - DM/群组策略控制（与 WeCom/Weixin 对等）

### 🔌 大幅扩展的插件系统
插件现在可以：
- **注册斜杠命令** (`register_command`)
- **直接调度工具** (`dispatch_tool`)
- **阻止工具执行** (`pre_tool_call` 可以否决)
- **重写工具结果** (`transform_tool_result`)
- **转换终端输出** (`transform_terminal_output`)
- **提供 image_gen 后端**
- **添加自定义 dashboard 标签**
- **内置磁盘清理插件**作为参考实现（默认启用）

### 🎯 新功能
- **`/steer <prompt>`** - 运行时代程调整
  - 在下一次工具调用后注入提示
  - 不中断轮次或破坏提示缓存
  - 适合飞行中纠正代理行为
- **Shell hooks** - 生命周期钩子
  - 无需编写 Python 插件
  - 支持所有生命周期事件（pre_tool_call, post_tool_call 等）
- **Webhook 直接投递模式**
  - Webhook 订阅可以直接转发载荷到平台聊天
  - 零 LLM 推送通知用于警报、监控和事件流
- **更智能的委派**
  - 子代理现在有明确的 `orchestrator` 角色
  - 可配置的 `max_spawn_depth`（默认为平面）
  - 并发兄弟子代理通过文件协调层共享文件系统状态
- **辅助模型配置 UI**
  - `hermes model` 有专门的"配置辅助模型"屏幕
  - 支持按任务覆盖（压缩、视觉、会话搜索、标题生成）
  - `auto` 路由现在默认所有用户使用主模型

### 🎨 Dashboard 改进
- **插件系统**
  - 第三方插件可以添加自定义标签、小部件和视图
  - 无需分叉即可扩展
- **实时主题切换**
  - 主题控制颜色、字体、布局和密度
  - 热交换 dashboard 外观无需重新加载
- **i18n 支持**（英文 + 中文）
- **react-router 侧边栏布局**
- **移动端响应式**
- **Vercel 部署支持**
- **每会话 API 调用跟踪**
- **一键更新 + 网关重启按钮**

### 🔧 其他重要更新
- **Cron**：
  - 支持每个作业的 `enabled_toolsets` 以减少令牌开销
  - 尊重 `hermes tools` 配置
- **Kimi K2.6**：
  - 在 OpenRouter、Nous Portal、原生 Kimi 和 HuggingFace 上可用
  - 替换所有列表中的 Kimi K2.5
- **MCP**：
  - 改进模式稳健性
  - 强制转换字符串化数组/对象
- **委派**：
  - 默认 `child_timeout_seconds` 提升到 600s
- **Xiaomi MiMo v2.5-pro + v2.5**：
  - 在 OpenRouter、Nous Portal 和原生提供商上可用
- **GLM-5V-Turbo**：
  - 用于 coding plan
- **Claude Opus 4.7**：
  - 在 Nous Portal 目录中可用
- **OpenRouter elephant-alpha**：
  - 在策展列表中可用

### 🐛 重要修复
- **TUI**：
  - 恢复 voice 和 panic 处理器
  - inline_diff 段空白行呼吸空间
  - @<name> 跨仓库模糊匹配文件名
  - 语音模式每次启动时关闭（CLI 对等性）
  - 忽略 SIGPIPE 以防止 stderr 反压杀死网关
  - 捕获信号触发的网关退出
  - 记录网关退出原因
  - 转储网关崩溃跟踪到日志文件
  - 打断 TTS→STT 反馈循环
  - 语音 TTS 回话 + 转录键错误 + 自动提交
  - 添加缺失的 hermes_cli.voice 网关 RPC 包装器
  - 路由 Ctrl+B 到语音切换，不是 composer 输入
  - 缓存文本测量跨 yoga flex 重传
- **MCP**：
  - 强制转换字符串化数组/对象在工具参数中
  - 重写定义引用 $ref 到输入模式中
- **Kimi/Moonshot**：
  - 模式清理器 + MCP 模式稳健性

### 🏗️ 架构改进
- **Agent Loop**：
  - 压缩器智能折叠、去重、反抖动
  - 模板升级、硬化
  - 压缩摘要尊重对话的语言
  - 压缩模型在永久 503/404 时回退到主模型
  - 网关重启后自动继续中断的代理工作
  - 活动心跳防止虚假网关不活动超时
  - 重置压缩后的重试计数器
  - 打断压缩耗尽无限循环并自动重置会话
  - 修复空响应后弱模型的过早循环退出
  - 改进中断期间并发工具执行的响应性
  - `/stop` 不再重置会话
  - 荣耀中断期间 MCP 工具等待
- **Session & Memory**：
  - 启动时自动修剪旧会话 + VACUUM state.db
  - Honcho 改革 - 上下文注入、5 工具表面、成本安全、会话隔离
  - Hindsight 更丰富的会话范围保留元数据
  - 去重记忆提供者工具以防止严格提供商上的 400
  - 从 `$HERMES_HOME/plugins/` 发现用户安装的记忆提供者
  - 添加 `on_memory_write` 桥接到顺序工具执行路径
  - 保留 `session_id` 跨 `/v1/responses` 中的 `previous_response_id` 链

### 🔨 CI/CD 改进
- **Nix 支持**：
  - 自动 lockfile 修复以保持 main 在 nix 上构建
  - 在所有 lockfile 更改上运行 CI
  - 添加 nix-lockfile-check 和 nix-lockfile-fix 工作流

### 📚 文档更新
- 更新所有模块文档以反映新架构
- 更新技术栈和模块结构图
- 添加传输层和 TUI 模块文档

---

## 2026-04-21 - 同步上游最新变更 (v2026.4.21) 🚀

### 🎨 TUI 改进
- **剪贴板和快捷键改进**：
  - 修复 macOS 上的 Ctrl+C 回归问题，移除双重剪贴板写入
  - 在 macOS 上使用 pbcopy 进行复制操作
  - 在 macOS 输入字段中启用剪贴板快捷键
  - 在 clarify 模式下恢复剪贴板快捷键
  - 在 macOS 上使用 Command 快捷键
  - 启用右键粘贴功能
- **时间显示改进**：
  - 在会话总时长旁边显示距上次用户消息的时间
  - 将上次消息时间从状态栏移至提示右侧
  - 改进 elapsed 计时器，仅在 FaceTicker 中显示
- **性能优化**：
  - 重构内存和 resize 助手，清理传递
  - debounce resize RPC + column-aware useVirtualHistory
  - 修复 Node V8 OOM + GatewayClient 内存泄漏
- **显示修复**：
  - 将 MEDIA: 渲染为可点击的文件芯片，移除音频指令
  - 修复 markdown 中单词内下划线的斜体显示
  - 使 "/tools list" 显示真实颜色而非 "?[32m" 等
  - 自动展开错误时的 Activity
- **主题和横幅**：
  - 通过主题路由 update-behind 横幅，自动检测连字支持

### 🔧 模型和提供商改进
- **新增 Kimi-K2.6 模型**：
  - 在 HuggingFace provider 添加 moonshotai/Kimi-K2.6
  - 在 kimi-coding、kimi-coding-cn 和 moonshot provider 添加 kimi-k2.6
  - 在 OpenRouter 和 Nous Portal 用 kimi-k2.6 替换 kimi-k2.5
- **Kimi 模型修复**：
  - 为 Kimi/Moonshot 模型完全省略 temperature 参数
- **新适配器**：
  - 提取 codex_responses 逻辑到专用适配器
  - 添加 Gemini 原生适配器支持

### 📱 平台改进
- **Signal**：添加媒体投递支持
- **WhatsApp**：添加与 WeCom/WeiXin 相同的 dm_policy 和 group_policy 对等性
- **Telegram**：
  - 在 MarkdownV2 链接中正确处理括号
  - 为 DM topics 添加可操作的错误提示（Topics 模式未启用时）
  - 缓存入站视频并接受 mp4 上传
- **Discord**：在频道中正确处理 /slash 命令
- **插件系统**：
  - 在 Telegram 菜单中发布插件斜杠命令
  - 在 config.yaml 中启用插件以进行延迟发现测试

### 🔐 安全改进
- **上下文压缩**：从上下文压缩输入和输出中编辑敏感信息
- **文件工具**：为工作目录解析相对路径（TERMINAL_CWD）

### ⚙️ Cron 和调度改进
- **并行作业执行**：运行到期作业以防止串行 tick 饥饿
- **修复清理孤立的云浏览器守护进程**

### 🐛 其他修复
- **Gateway**：使用持久化的会话来源进行关闭通知
- **Agent**：在 API 发送前修复格式错误的 tool_call 参数
- **Compression**：从压缩触发器中排除完成令牌
- **Session Search**：当消息 id 为空时恢复同会话上下文
- **Steer**：在每次 API 调用前排空 pending steer（而不仅仅是之后）
- **Install**：对带空格的路径引用 PYTHON_PATH 和 UV_CMD
- **Model Switch**：在 Section 3 中为 custom: slug 注册 seen_slugs

### 🔨 CI/CD 改进
- **Nix 支持**：
  - 自动 lockfile 修复以保持 main 在 nix 上构建
  - 在所有 lockfile 更改上运行 CI
  - 添加 nix-lockfile-check 和 nix-lockfile-fix 工作流

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
