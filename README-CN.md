<p align="center">
  <img src="assets/banner.png" alt="Hermes Agent" width="100%">
</p>

# Hermes Agent ☤

<p align="center">
  <a href="https://hermes-agent.nousresearch.com/docs/"><img src="https://img.shields.io/badge/Docs-hermes--agent.nousresearch.com-FFD700?style=for-the-badge" alt="文档"></a>
  <a href="https://discord.gg/NousResearch"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://github.com/NousResearch/hermes-agent/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="许可证: MIT"></a>
  <a href="https://nousresearch.com"><img src="https://img.shields.io/badge/Built%20by-Nous%20Research-blueviolet?style=for-the-badge" alt="由Nous Research构建"></a>
</p>

**由[Nous Research](https://nousresearch.com)构建的自我改进AI代理。** 它是唯一具有内置学习循环的代理——它从经验中创造技能，在使用过程中不断改进，推动自己保持知识，搜索自己的历史对话，并在会话中建立对你越来越深的理解。它可以运行在一个5美元的VPS、GPU集群或几乎不花钱的无服务器基础设施上。它不依赖于你的笔记本电脑——你可以通过Telegram与它对话，它在云端虚拟机上运行。

使用你想要的任何模型——[Nous Portal](https://portal.nousresearch.com)、[OpenRouter](https://openrouter.ai)（200+模型）、[z.ai/GLM](https://z.ai)、[Kimi/Moonshot](https://platform.moonshot.ai)、[MiniMax](https://www.minimax.io)、OpenAI或你自己的端点。通过 `hermes model` 切换——无需修改代码，无锁定。

<table>
<tr><td><b>真实终端界面</b></td><td>完整的TUI，支持多行编辑、斜杠命令自动完成、对话历史、打断和重定向、流式工具输出。</td></tr>
<tr><td><b>与你共存</b></td><td>Telegram、Discord、Slack、WhatsApp、Signal和CLI——都通过一个单一的网关进程进行管理。语音备忘录转录、跨平台对话连续性。</td></tr>
<tr><td><b>闭环学习</b></td><td>由代理策划的记忆，定期提示。复杂任务后的自主技能创建。技能在使用过程中自我改进。使用FTS5会话搜索和LLM摘要进行跨会话回忆。<a href="https://github.com/plastic-labs/honcho">Honcho</a>方言用户建模。兼容<a href="https://agentskills.io">agentskills.io</a>开放标准。</td></tr>
<tr><td><b>定时自动化</b></td><td>内置cron调度程序，支持任何平台的任务调度。每日报告、夜间备份、每周审计——都以自然语言运行，无需人工干预。</td></tr>
<tr><td><b>委派与并行处理</b></td><td>生成独立的子代理以进行并行工作流。编写Python脚本，通过RPC调用工具，将多步流程压缩成零上下文成本的回合。</td></tr>
<tr><td><b>可以运行在任何地方，不仅仅是你的笔记本电脑</b></td><td>六种终端后端——本地、Docker、SSH、Daytona、Singularity和Modal。Daytona和Modal提供无服务器持久性——代理的环境在空闲时进入休眠，按需唤醒，几乎不产生任何费用。你可以在一个5美元的VPS或GPU集群上运行它。</td></tr>
<tr><td><b>研究就绪</b></td><td>批量轨迹生成、Atropos RL环境、用于训练下一代工具调用模型的轨迹压缩。</td></tr>
</table>

---

## 快速安装

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

支持Linux、macOS和WSL2。安装程序会自动处理所有内容——Python、Node.js、依赖项和 `hermes` 命令。除了git之外，没有其他先决条件。

> **Windows：** 不支持原生Windows。请安装[WSL2](https://learn.microsoft.com/en-us/windows/wsl/install)并运行上述命令。

安装完成后：

```bash
source ~/.bashrc    # 重新加载shell（或者：source ~/.zshrc）
hermes              # 开始对话！
```

---

## 入门

```bash
hermes              # 交互式CLI — 开始对话
hermes model        # 选择你的LLM提供者和模型
hermes tools        # 配置启用的工具
hermes config set   # 设置单个配置项
hermes gateway      # 启动消息网关（Telegram、Discord等）
hermes setup        # 运行完整的设置向导（一次性配置所有内容）
hermes claw migrate # 从OpenClaw迁移（如果你是从OpenClaw过来的）
hermes update       # 更新到最新版本
hermes doctor       # 诊断任何问题
```

📖 **[完整文档 →](https://hermes-agent.nousresearch.com/docs/)**

## CLI与消息平台简要对照

Hermes有两个入口：通过`hermes`启动终端界面，或者通过网关与它在Telegram、Discord、Slack、WhatsApp、Signal或电子邮件中对话。一旦你进入对话，许多斜杠命令在两个界面中是共享的。

| 操作           | CLI                                         | 消息平台                                                         |
| ------------ | ------------------------------------------- | ------------------------------------------------------------ |
| 开始对话         | `hermes`                                    | 运行`hermes gateway setup` + `hermes gateway start`，然后发送消息给机器人 |
| 开始新对话        | `/new` 或 `/reset`                           | `/new` 或 `/reset`                                            |
| 更改模型         | `/model [provider:model]`                   | `/model [provider:model]`                                    |
| 设置个性         | `/personality [name]`                       | `/personality [name]`                                        |
| 重试或撤销上一回合    | `/retry`，`/undo`                            | `/retry`，`/undo`                                             |
| 压缩上下文/检查使用情况 | `/compress`，`/usage`，`/insights [--days N]` | `/compress`，`/usage`，`/insights [days]`                      |
| 浏览技能         | `/skills` 或 `/<skill-name>`                 | `/skills` 或 `/<skill-name>`                                  |
| 打断当前工作       | `Ctrl+C` 或发送新消息                             | `/stop` 或发送新消息                                               |
| 平台特定状态       | `/platforms`                                | `/status`，`/sethome`                                         |

完整的命令列表，请参见[CLI指南](https://hermes-agent.nousresearch.com/docs/user-guide/cli)和[消息网关指南](https://hermes-agent.nousresearch.com/docs/user-guide/messaging)。

---

## 文档

所有文档都位于**[hermes-agent.nousresearch.com/docs](https://hermes-agent.nousresearch.com/docs/)**：

| 部分                                                                                    | 内容                                                    |
| ------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| [快速入门](https://hermes-agent.nousresearch.com/docs/getting-started/quickstart)         | 安装 → 设置 → 2分钟内开始第一次对话                                 |
| [CLI使用](https://hermes-agent.nousresearch.com/docs/user-guide/cli)                    | 命令、快捷键、个性、会话                                          |
| [配置](https://hermes-agent.nousresearch.com/docs/user-guide/configuration)             | 配置文件、提供者、模型、所有选项                                      |
| [消息网关](https://hermes-agent.nousresearch.com/docs/user-guide/messaging)               | Telegram、Discord、Slack、WhatsApp、Signal、Home Assistant |
| [安全](https://hermes-agent.nousresearch.com/docs/user-guide/security)                  | 命令审批、DM配对、容器隔离                                        |
| [工具与工具集](https://hermes-agent.nousresearch.com/docs/user-guide/features/tools)        | 40+工具、工具集系统、终端后端                                      |
| [技能系统](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills)         | 程序化记忆、技能中心、创建技能                                       |
| [记忆](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory)           | 持久记忆、用户档案、最佳实践                                        |
| [MCP集成](https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp)           | 连接任何MCP服务器以扩展功能                                       |
| [Cron调度](https://hermes-agent.nousresearch.com/docs/user-guide/features/cron)         | 平台交付的定时任务                                             |
| [上下文文件](https://hermes-agent.nousresearch.com/docs/user-guide/features/context-files) | 影响每次对话的项目上下文                                          |
| [架构](https://hermes-agent.nousresearch.com/docs/developer-guide/architecture)         | 项目结构、代理循环、关键类                                         |
| [贡献](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing)         | 开发设置、PR流程、代码风格                                        |
| [CLI参考](https://hermes-agent.nousresearch.com/docs/reference/cli-commands)            | 所有命令和标志                                               |
| [环境变量](https://hermes-agent.nousresearch.com/docs/reference/environment-variables)    | 完整的环境变量参考                                             |

---

## 从OpenClaw迁移

如果你是从OpenClaw过来的，Hermes可以自动导入你的设置、记忆、技能和API密钥。

**首次设置期间：** 设置向导（`hermes setup`）会自动检测`~/.openclaw`并在配置开始前提供迁移选项。

**安装后任何时候：**

```bash
hermes claw migrate              # 交互式迁移（完整预设）
hermes claw migrate --dry-run    # 预览将迁移的内容
hermes claw migrate --preset user-data   # 只迁移无机密的部分
hermes claw migrate --overwrite  # 覆盖现有的冲突
```

导入内容：

* **SOUL.md** — 人物文件
* **记忆** — MEMORY.md 和 USER.md 条目
* **技能** — 用户创建的技能 → `~/.hermes/skills/openclaw-imports/`
* **命令白名单** — 批准模式
* **消息设置** — 平台配置、允许用户、工作目录
* **API密钥** — 白名单的密钥（Telegram、OpenRouter、OpenAI、Anthropic、ElevenLabs）
* **TTS资源** — 工作区音频文件
* **工作区指令** — AGENTS.md（带有`--workspace-target`）

查看 `hermes claw migrate --help` 获取所有选项，或者使用 `openclaw-migration` 技能进行交互式代理引导的迁移，并进行干运行预览。

---

## 贡献

我们欢迎贡献！请参阅[贡献指南](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing)了解开发设置、代码风格和PR流程。

贡献者的快速入门：

```bash
git clone https://github.com/NousResearch/hermes-agent.git
cd hermes-agent
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv venv --python 3.11
source venv/bin/activate
uv pip install -e ".[all,dev]"
python -m pytest tests/ -q
```

> **RL训练（可选）：** 如果要进行RL/Tinker-Atropos集成开发：
>
> ```bash
> git submodule update --init tinker-atropos
> uv pip install -e "./tinker-atropos"
> ```

---

## 社区

* 💬 [Discord](https://discord.gg/NousResearch)
* 📚 [技能中心](https://agentskills.io)
* 🐛 [问题](https://github.com/NousResearch/hermes-agent/issues)
* 💡 [讨论](https://github.com/NousResearch/hermes-agent/discussions)

---

## 许可证

MIT — 见[许可证](LICENSE)。

由[Nous Research](https://nousresearch.com)构建。

---
