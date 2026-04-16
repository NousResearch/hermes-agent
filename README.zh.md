<p align="center">
  <img src="assets/banner.png" alt="Hermes Agent" width="100%">
</p>

# Hermes Agent ☤

<p align="center">
  <a href="https://hermes-agent.nousresearch.com/docs/"><img src="https://img.shields.io/badge/文档-hermes--agent.nousresearch.com-FFD700?style=for-the-badge" alt="Documentation"></a>
  <a href="https://discord.gg/NousResearch"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://github.com/NousResearch/hermes-agent/blob/main/LICENSE"><img src="https://img.shields.io/badge/许可证-MIT-green?style=for-the-badge" alt="License: MIT"></a>
  <a href="https://nousresearch.com"><img src="https://img.shields.io/badge/由-Nous%20Research-开发-blueviolet?style=for-the-badge" alt="Built by Nous Research"></a>
</p>

**由 [Nous Research](https://nousresearch.com) 开发的自改进 AI 助手。** 它是唯一一个内置学习循环的助手 — 它从经验中创建技能，在使用过程中改进它们，自我激励以持久化知识，搜索自己过去的对话，并在会话中构建对你的深入了解模型。你可以在 5 美元的 VPS、GPU 集群或空闲时几乎不花钱的无服务器基础设施上运行它。它不局限于你的笔记本电脑 — 当它在云 VM 上工作时，你可以通过 Telegram 与它交谈。

使用任何你想要的模型 — [Nous Portal](https://portal.nousresearch.com)、[OpenRouter](https://openrouter.ai)（200+ 模型）、[小米 MiMo](https://platform.xiaomimimo.com)、[z.ai/GLM](https://z.ai)、[Kimi/月之暗面](https://platform.moonshot.ai)、[MiniMax](https://www.minimax.io)、[Hugging Face](https://huggingface.co)、OpenAI 或你自己的端点。使用 `hermes model` 切换 — 无需代码更改，无锁定。

<table>
<tr><td><b>真正的终端界面</b></td><td>完整的 TUI，支持多行编辑、斜杠命令自动完成、对话历史、中断和重定向，以及流式工具输出。</td></tr>
<tr><td><b>随时随地陪伴你</b></td><td>Telegram、Discord、Slack、WhatsApp、Signal 和 CLI — 全部来自单个网关进程。语音备忘录转录，跨平台对话连续性。</td></tr>
<tr><td><b>闭合学习循环</b></td><td>代理策划的记忆，定期自我激励。复杂任务后自主创建技能。技能在使用过程中自我改进。FTS5 会话搜索与 LLM 摘要，实现跨会话回忆。<a href="https://github.com/plastic-labs/honcho">Honcho</a> 辩证用户建模。兼容 <a href="https://agentskills.io">agentskills.io</a> 开放标准。</td></tr>
<tr><td><b>定时自动化</b></td><td>内置 cron 调度器，可将内容投递到任何平台。每日报告、夜间备份、每周审计 — 全部以自然语言呈现，无人值守运行。</td></tr>
<tr><td><b>委派和并行化</b></td><td>为并行工作流生成隔离的子代理。编写通过 RPC 调用工具的 Python 脚本，将多步管道压缩为零上下文成本的回合。</td></tr>
<tr><td><b>随处运行，不只是你的笔记本电脑</b></td><td>六种终端后端 — 本地、Docker、SSH、Daytona、Singularity 和 Modal。Daytona 和 Modal 提供无服务器持久性 — 你的代理环境在空闲时休眠，按需唤醒，会话之间几乎不花费任何成本。在 5 美元的 VPS 或 GPU 集群上运行它。</td></tr>
<tr><td><b>研究就绪</b></td><td>批量轨迹生成、Atropos RL 环境、用于训练下一代工具调用模型的轨迹压缩。</td></tr>
</table>

---

## 快速安装

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

适用于 Linux、macOS、WSL2 和通过 Termux 的 Android。安装程序会为你处理平台特定的设置。

> **Android / Termux：** 测试过的手动路径记录在 [Termux 指南](https://hermes-agent.nousresearch.com/docs/getting-started/termux) 中。在 Termux 上，Hermes 安装一个精选的 `.[termux]` 扩展，因为完整的 `.[all]` 扩展目前会拉取与 Android 不兼容的语音依赖项。
>
> **Windows：** 不支持原生 Windows。请安装 [WSL2](https://learn.microsoft.com/zh-cn/windows/wsl/install) 并运行上面的命令。

安装后：

```bash
source ~/.bashrc    # 重新加载 shell（或：source ~/.zshrc）
hermes              # 开始聊天！
```

---

## 入门

```bash
hermes              # 交互式 CLI — 开始对话
hermes model        # 选择你的 LLM 提供商和模型
hermes tools        # 配置启用哪些工具
hermes config set   # 设置单个配置值
hermes gateway      # 启动消息网关（Telegram、Discord 等）
hermes setup        # 运行完整的设置向导（一次性配置所有内容）
hermes claw migrate # 从 OpenClaw 迁移（如果来自 OpenClaw）
hermes update       # 更新到最新版本
hermes doctor       # 诊断任何问题
```

📖 **[完整文档 →](https://hermes-agent.nousresearch.com/docs/)**

## CLI 与消息平台快速参考

Hermes 有两个入口点：使用 `hermes` 启动终端 UI，或运行网关并通过 Telegram、Discord、Slack、WhatsApp、Signal 或电子邮件与它交谈。一旦进入对话，许多斜杠命令在两个界面中都是共享的。

| 操作 | CLI | 消息平台 |
|---------|-----|---------------------|
| 开始聊天 | `hermes` | 运行 `hermes gateway setup` + `hermes gateway start`，然后向机器人发送消息 |
| 开始新对话 | `/new` 或 `/reset` | `/new` 或 `/reset` |
| 更改模型 | `/model [provider:model]` | `/model [provider:model]` |
| 设置个性 | `/personality [name]` | `/personality [name]` |
| 重试或撤销上一轮 | `/retry`, `/undo` | `/retry`, `/undo` |
| 压缩上下文 / 检查使用情况 | `/compress`, `/usage`, `/insights [--days N]` | `/compress`, `/usage`, `/insights [days]` |
| 浏览技能 | `/skills` 或 `/<skill-name>` | `/skills` 或 `/<skill-name>` |
| 中断当前工作 | `Ctrl+C` 或发送新消息 | `/stop` 或发送新消息 |
| 平台特定状态 | `/platforms` | `/status`, `/sethome` |

完整的命令列表，请参阅 [CLI 指南](https://hermes-agent.nousresearch.com/docs/user-guide/cli) 和 [消息网关指南](https://hermes-agent.nousresearch.com/docs/user-guide/messaging)。

---

## 文档

所有文档都位于 **[hermes-agent.nousresearch.com/docs](https://hermes-agent.nousresearch.com/docs/)**：

| 部分 | 涵盖内容 |
|---------|---------------|
| [快速入门](https://hermes-agent.nousresearch.com/docs/getting-started/quickstart) | 安装 → 设置 → 2 分钟内的第一次对话 |
| [CLI 使用](https://hermes-agent.nousresearch.com/docs/user-guide/cli) | 命令、按键绑定、个性、会话 |
| [配置](https://hermes-agent.nousresearch.com/docs/user-guide/configuration) | 配置文件、提供商、模型、所有选项 |
| [消息网关](https://hermes-agent.nousresearch.com/docs/user-guide/messaging) | Telegram、Discord、Slack、WhatsApp、Signal、Home Assistant |
| [安全](https://hermes-agent.nousresearch.com/docs/user-guide/security) | 命令批准、DM 配对、容器隔离 |
| [工具与工具集](https://hermes-agent.nousresearch.com/docs/user-guide/features/tools) | 40+ 工具、工具集系统、终端后端 |
| [技能系统](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills) | 程序记忆、技能中心、创建技能 |
| [记忆](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory) | 持久记忆、用户配置文件、最佳实践 |
| [MCP 集成](https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp) | 连接任何 MCP 服务器以扩展功能 |
| [Cron 调度](https://hermes-agent.nousresearch.com/docs/user-guide/features/cron) | 带平台投递的定时任务 |
| [上下文文件](https://hermes-agent.nousresearch.com/docs/user-guide/features/context-files) | 塑造每个对话的项目上下文 |
| [架构](https://hermes-agent.nousresearch.com/docs/developer-guide/architecture) | 项目结构、代理循环、关键类 |
| [贡献](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing) | 开发设置、PR 流程、代码风格 |
| [CLI 参考](https://hermes-agent.nousresearch.com/docs/reference/cli-commands) | 所有命令和标志 |
| [环境变量](https://hermes-agent.nousresearch.com/docs/reference/environment-variables) | 完整的环境变量参考 |

---

## 从 OpenClaw 迁移

如果你来自 OpenClaw，Hermes 可以自动导入你的设置、记忆、技能和 API 密钥。

**首次设置期间：** 设置向导 (`hermes setup`) 会自动检测 `~/.openclaw` 并在配置开始前提供迁移选项。

**安装后随时：**

```bash
hermes claw migrate              # 交互式迁移（完整预设）
hermes claw migrate --dry-run    # 预览将要迁移的内容
hermes claw migrate --preset user-data   # 迁移不含密钥
hermes claw migrate --overwrite  # 覆盖现有冲突
```

导入的内容：
- **SOUL.md** — 角色文件
- **记忆** — MEMORY.md 和 USER.md 条目
- **技能** — 用户创建的技能 → `~/.hermes/skills/openclaw-imports/`
- **命令允许列表** — 批准模式
- **消息设置** — 平台配置、允许的用户、工作目录
- **API 密钥** — 允许的密钥（Telegram、OpenRouter、OpenAI、Anthropic、ElevenLabs）
- **TTS 资产** — 工作区音频文件
- **工作区说明** — AGENTS.md（使用 `--workspace-target`）

有关所有选项，请参阅 `hermes claw migrate --help`，或使用 `openclaw-migration` 技能进行交互式代理引导的迁移，包括干运行预览。

---

## 贡献

我们欢迎贡献！请参阅 [贡献指南](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing) 了解开发设置、代码风格和 PR 流程。

贡献者快速入门：

```bash
git clone https://github.com/NousResearch/hermes-agent.git
cd hermes-agent
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv venv --python 3.11
source venv/bin/activate
uv pip install -e ".[all,dev]"
python -m pytest tests/ -q
```

> **RL 训练（可选）：** 要处理 RL/Tinker-Atropos 集成：
> ```bash
> git submodule update --init tinker-atropos
> uv pip install -e "./tinker-atropos"
> ```

---

## 社区

- 💬 [Discord](https://discord.gg/NousResearch)
- 📚 [技能中心](https://agentskills.io)
- 🐛 [问题](https://github.com/NousResearch/hermes-agent/issues)
- 💡 [讨论](https://github.com/NousResearch/hermes-agent/discussions)
- 🔌 [HermesClaw](https://github.com/AaronWong1999/hermesclaw) — 社区微信桥接：在同一个微信账号上运行 Hermes Agent 和 OpenClaw。

---

## 许可证

MIT — 见 [LICENSE](LICENSE)。

由 [Nous Research](https://nousresearch.com) 开发。
