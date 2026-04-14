<p align="center">
  <img src="assets/banner.png" alt="Hermes Agent" width="100%">
</p>

# Hermes Agent ☤

<p align="center">
  <a href="https://hermes-agent.nousresearch.com/docs/"><img src="https://img.shields.io/badge/Docs-hermes--agent.nousresearch.com-FFD700?style=for-the-badge" alt="Documentation"></a>
  <a href="https://discord.gg/NousResearch"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://github.com/NousResearch/hermes-agent/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License: MIT"></a>
  <a href="https://nousresearch.com"><img src="https://img.shields.io/badge/Built%20by-Nous%20Research-blueviolet?style=for-the-badge" alt="Built by Nous Research"></a>
</p>

**由 [Nous Research](https://nousresearch.com) 打造的可自我改进 AI Agent。** 它是目前少见的内置学习闭环的 Agent：能从经验中创建技能，在使用中持续优化技能，主动提醒自己沉淀知识，检索过往对话，并在跨会话中不断加深对你的理解。你可以把它跑在 5 美元 VPS、GPU 集群，或几乎空闲零成本的无服务器基础设施上。它不绑定你的笔记本电脑，你可以在 Telegram 上与它对话，同时让它在云端虚拟机上工作。

模型可自由选择：[Nous Portal](https://portal.nousresearch.com)、[OpenRouter](https://openrouter.ai)（200+ 模型）、[Xiaomi MiMo](https://platform.xiaomimimo.com)、[z.ai/GLM](https://z.ai)、[Kimi/Moonshot](https://platform.moonshot.ai)、[MiniMax](https://www.minimax.io)、[Hugging Face](https://huggingface.co)、OpenAI，或你自己的端点。使用 `hermes model` 即可切换，无需改代码、无平台锁定。

<table>
<tr><td><b>真正可用的终端界面</b></td><td>完整 TUI，支持多行编辑、斜杠命令自动补全、会话历史、打断并重定向、以及工具输出流式展示。</td></tr>
<tr><td><b>在你常用的平台中工作</b></td><td>Telegram、Discord、Slack、WhatsApp、Signal 与 CLI 全部由同一个网关进程接入。支持语音消息转写，并可跨平台连续对话。</td></tr>
<tr><td><b>闭环学习能力</b></td><td>Agent 策展式记忆并带周期性提醒。复杂任务后可自主创建技能，技能在使用中可自我优化。基于 FTS5 的会话检索结合 LLM 摘要实现跨会话回忆。集成 <a href="https://github.com/plastic-labs/honcho">Honcho</a> 辩证式用户建模。兼容 <a href="https://agentskills.io">agentskills.io</a> 开放标准。</td></tr>
<tr><td><b>定时自动化</b></td><td>内置 cron 调度，可投递到任意平台。日报、夜间备份、每周审计均可通过自然语言配置并无人值守运行。</td></tr>
<tr><td><b>可委派并行执行</b></td><td>可创建隔离子代理并行处理工作流。还能编写 Python 脚本通过 RPC 调用工具，把多步骤流程压缩为零上下文成本回合。</td></tr>
<tr><td><b>可在任意环境运行，而不只是本机</b></td><td>支持六种终端后端：local、Docker、SSH、Daytona、Singularity、Modal。Daytona 和 Modal 提供无服务器持久化能力：Agent 环境空闲时休眠、按需唤醒，会话间成本极低。可运行在 5 美元 VPS 或 GPU 集群。</td></tr>
<tr><td><b>面向研究</b></td><td>支持批量轨迹生成、Atropos RL 环境，以及用于训练下一代工具调用模型的轨迹压缩。</td></tr>
</table>

---

## 快速安装

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

支持 Linux、macOS、WSL2，以及通过 Termux 的 Android。安装脚本会自动处理平台差异。

> **Android / Termux：** 已验证的手动安装路径见 [Termux 指南](https://hermes-agent.nousresearch.com/docs/getting-started/termux)。在 Termux 上，Hermes 会安装精简的 `.[termux]` 依赖，因为完整 `.[all]` 当前会拉取与 Android 不兼容的语音依赖。
>
> **Windows：** 不支持原生 Windows。请先安装 [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install)，再执行上面的命令。

安装完成后：

```bash
source ~/.bashrc    # 重新加载 shell（或：source ~/.zshrc）
hermes              # 开始对话！
```

---

## 快速开始

```bash
hermes              # 交互式 CLI：开始会话
hermes model        # 选择你的 LLM 提供商和模型
hermes tools        # 配置启用哪些工具
hermes config set   # 设置单个配置项
hermes gateway      # 启动消息网关（Telegram、Discord 等）
hermes setup        # 运行完整初始化向导（一次性配置全部）
hermes claw migrate # 从 OpenClaw 迁移（若你来自 OpenClaw）
hermes update       # 更新到最新版本
hermes doctor       # 诊断问题
```

📖 **[完整文档 →](https://hermes-agent.nousresearch.com/docs/)**

## CLI 与消息平台对照速查

Hermes 有两个入口：通过 `hermes` 启动终端 UI，或运行网关后从 Telegram、Discord、Slack、WhatsApp、Signal、Email 与它对话。进入会话后，很多斜杠命令在两种界面中是共通的。

| 操作 | CLI | 消息平台 |
|---------|-----|---------------------|
| 开始对话 | `hermes` | 运行 `hermes gateway setup` + `hermes gateway start`，然后给机器人发消息 |
| 开启全新会话 | `/new` 或 `/reset` | `/new` 或 `/reset` |
| 切换模型 | `/model [provider:model]` | `/model [provider:model]` |
| 设置人格 | `/personality [name]` | `/personality [name]` |
| 重试或撤销上一轮 | `/retry`, `/undo` | `/retry`, `/undo` |
| 压缩上下文 / 查看用量 | `/compress`, `/usage`, `/insights [--days N]` | `/compress`, `/usage`, `/insights [days]` |
| 浏览技能 | `/skills` 或 `/<skill-name>` | `/skills` 或 `/<skill-name>` |
| 打断当前任务 | `Ctrl+C` 或发送新消息 | `/stop` 或发送新消息 |
| 平台相关状态 | `/platforms` | `/status`, `/sethome` |

完整命令列表请见 [CLI 指南](https://hermes-agent.nousresearch.com/docs/user-guide/cli) 与 [消息网关指南](https://hermes-agent.nousresearch.com/docs/user-guide/messaging)。

---

## 文档

所有文档集中在 **[hermes-agent.nousresearch.com/docs](https://hermes-agent.nousresearch.com/docs/)**：

| 部分 | 涵盖内容 |
|---------|---------------|
| [Quickstart](https://hermes-agent.nousresearch.com/docs/getting-started/quickstart) | 2 分钟完成安装 → 配置 → 首次会话 |
| [CLI Usage](https://hermes-agent.nousresearch.com/docs/user-guide/cli) | 命令、快捷键、人格、会话 |
| [Configuration](https://hermes-agent.nousresearch.com/docs/user-guide/configuration) | 配置文件、提供商、模型、全部选项 |
| [Messaging Gateway](https://hermes-agent.nousresearch.com/docs/user-guide/messaging) | Telegram、Discord、Slack、WhatsApp、Signal、Home Assistant |
| [Security](https://hermes-agent.nousresearch.com/docs/user-guide/security) | 命令审批、私聊配对、容器隔离 |
| [Tools & Toolsets](https://hermes-agent.nousresearch.com/docs/user-guide/features/tools) | 40+ 工具、工具集系统、终端后端 |
| [Skills System](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills) | 过程记忆、Skills Hub、技能创建 |
| [Memory](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory) | 持久记忆、用户画像、最佳实践 |
| [MCP Integration](https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp) | 连接任意 MCP 服务器扩展能力 |
| [Cron Scheduling](https://hermes-agent.nousresearch.com/docs/user-guide/features/cron) | 支持平台投递的定时任务 |
| [Context Files](https://hermes-agent.nousresearch.com/docs/user-guide/features/context-files) | 影响每次会话的项目上下文文件 |
| [Architecture](https://hermes-agent.nousresearch.com/docs/developer-guide/architecture) | 项目结构、Agent 循环、关键类 |
| [Contributing](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing) | 开发环境、PR 流程、代码风格 |
| [CLI Reference](https://hermes-agent.nousresearch.com/docs/reference/cli-commands) | 全部命令与参数 |
| [Environment Variables](https://hermes-agent.nousresearch.com/docs/reference/environment-variables) | 完整环境变量参考 |

---

## 从 OpenClaw 迁移

如果你从 OpenClaw 迁移，Hermes 可以自动导入你的设置、记忆、技能和 API 密钥。

**首次初始化期间：** 初始化向导（`hermes setup`）会自动检测 `~/.openclaw`，并在开始配置前提供迁移选项。

**安装后任意时间：**

```bash
hermes claw migrate              # 交互式迁移（完整预设）
hermes claw migrate --dry-run    # 预览将要迁移的内容
hermes claw migrate --preset user-data   # 不含密钥的迁移
hermes claw migrate --overwrite  # 覆盖现有冲突项
```

导入内容包括：
- **SOUL.md** — 人设文件
- **Memories** — MEMORY.md 与 USER.md 条目
- **Skills** — 用户自建技能 → `~/.hermes/skills/openclaw-imports/`
- **Command allowlist** — 命令批准模式
- **Messaging settings** — 平台配置、允许用户、工作目录
- **API keys** — 白名单密钥（Telegram、OpenRouter、OpenAI、Anthropic、ElevenLabs）
- **TTS assets** — 工作区音频文件
- **Workspace instructions** — AGENTS.md（配合 `--workspace-target`）

查看全部选项请运行 `hermes claw migrate --help`，或使用 `openclaw-migration` 技能获得带 dry-run 预览的交互式引导迁移。

---

## 贡献

欢迎贡献！请查看 [Contributing Guide](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing) 了解开发环境、代码风格和 PR 流程。

贡献者快速开始：

```bash
git clone https://github.com/NousResearch/hermes-agent.git
cd hermes-agent
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv venv --python 3.11
source venv/bin/activate
uv pip install -e ".[all,dev]"
python -m pytest tests/ -q
```

> **RL 训练（可选）：** 若要参与 RL/Tinker-Atropos 集成：
> ```bash
> git submodule update --init tinker-atropos
> uv pip install -e "./tinker-atropos"
> ```

---

## 社区

- 💬 [Discord](https://discord.gg/NousResearch)
- 📚 [Skills Hub](https://agentskills.io)
- 🐛 [Issues](https://github.com/NousResearch/hermes-agent/issues)
- 💡 [Discussions](https://github.com/NousResearch/hermes-agent/discussions)
- 🔌 [HermesClaw](https://github.com/AaronWong1999/hermesclaw) — 社区版微信桥接：在同一个微信账号上同时运行 Hermes Agent 与 OpenClaw。

---

## 许可证

MIT — 见 [LICENSE](LICENSE)。

由 [Nous Research](https://nousresearch.com) 打造。
