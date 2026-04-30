# Hermes Agent 文档中心

> 这套文档的目标不是“把所有实现细节一次讲完”，而是让新人能按正确顺序上手 Hermes Agent，并在需要时快速跳到更深的模块说明。

## 从哪里开始

如果你是第一次进入仓库，建议按这个顺序读：

1. [新人 Onboarding](./simple.md)：先知道项目是什么、第一天该做什么、哪些文件最值得先看。
2. [开发者指南](./development-guide.md)：解决本地安装、测试、提改动的基本问题。
3. [系统架构](./architecture.md)：建立对 `AIAgent`、工具系统、CLI、gateway 的整体认识。
4. [CLI 命令指南](./cli-guide.md)：知道平时怎么跑、怎么调、怎么切模型和工具。
5. [Hermes Desk 使用 SOP](./desk-sop.md)：用本地 Web Desk 管理项目、会话、记忆和工具运行过程。
6. [配置参考](./configuration-reference.md)：需要改行为时再查具体配置项。

之后按目标分支阅读：

- 扩工具：看 [工具参考手册](./tools-reference.md)
- 写技能：看 [Skills 技能系统](./skills-guide.md)
- 做网关：看 [消息网关指南](./gateway-guide.md)
- 做插件：看 [Plugin 集成指南](./plugin-integration-guide.md)
- 深挖 `run_agent.py`：看 [run_agent 导读地图](./run_agent-reading-map.md)

---

## 文档导航

| 文档 | 说明 | 适合谁读 |
|------|------|----------|
| [新人 Onboarding](./simple.md) | 第一天看什么、先跑什么、先改什么 | 第一次进入仓库的开发者 |
| [Onboarding 别名入口](./onboarding-guide.md) | 给外部引用保留的更直观入口名 | 从旧链接或外部导航进入的人 |
| [系统架构](./architecture.md) | 核心循环、模块关系、设计模式 | 想理解代码整体结构的开发者 |
| [CLI 命令指南](./cli-guide.md) | 交互式终端、斜杠命令、常见工作流 | 日常使用者 / 开发者 |
| [Hermes Desk 使用 SOP](./desk-sop.md) | 本地 Web Desk 的启动、项目、会话、运行和记忆管理流程 | 想通过浏览器使用 Hermes 的用户 |
| [工具参考手册](./tools-reference.md) | 核心工具名、toolset、注册和调试方法 | 开发者 / 高级用户 |
| [Skills 技能系统](./skills-guide.md) | Skills 的使用、编写和维护 | 想沉淀 SOP 或扩展提示词能力的用户 |
| [消息网关指南](./gateway-guide.md) | Telegram/Discord/Slack 等平台接入 | 想部署到聊天平台的用户 |
| [配置参考](./configuration-reference.md) | `config.yaml`、`.env`、profiles、覆盖顺序 | 需要自定义行为的用户 |
| [开发者指南](./development-guide.md) | 环境搭建、测试、提改动、常见坑 | 贡献者 / 二次开发者 |
| [Plugin 集成指南](./plugin-integration-guide.md) | 通用 plugin、memory provider、context engine 的接入方式 | 需要做深度扩展的开发者 |
| [run_agent 导读地图](./run_agent-reading-map.md) | 大文件 `run_agent.py` 的阅读路线 | 想深入主循环的人 |
| [system prompt 深读](./run_agent-build-system-prompt.md) | `_build_system_prompt` 的细节分析 | 想改 prompt 组装逻辑的人 |

---

## 新人第一天清单

如果你的目标是“先把项目跑起来，再做第一个小改动”，建议按下面的顺序来：

1. 激活虚拟环境：`source venv/bin/activate`
2. 跑 `hermes doctor`
3. 跑一次 `hermes`
4. 打开 [simple.md](./simple.md)
5. 找一个小改动：补文档、修帮助文案、给现有工具补测试，或者修一个小配置问题
6. 跑最小相关测试，而不是一上来就全量回归

---

## 快速上手指南

### 1. 环境要求

- Python 3.11+
- Git
- Node.js

补充说明：

- 如果你只是想先把 CLI 跑起来，Node.js 不是最小必需。
- 如果你要用浏览器工具、部分前端能力或 bridge 相关功能，再安装 Node.js 更合适。

### 2. 安装

**方式一：使用 `uv`（推荐）**

```bash
git clone <repo-url> hermes-agent
cd hermes-agent

uv venv venv --python 3.11
source venv/bin/activate
uv pip install -e ".[all,dev]"
```

**方式二：使用 `pip`**

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[all,dev]"
```

> 仓库里的开发约定是：只要要跑 Python 命令，先执行 `source venv/bin/activate`。

### 3. 配置 API 密钥

```bash
cp .env.example ~/.hermes/.env
```

至少准备一个模型提供商的密钥，例如：

```env
OPENROUTER_API_KEY=sk-or-...
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
NOUS_API_KEY=...
```

### 4. 验证安装

```bash
hermes doctor
```

这个命令通常能帮你快速发现：

- Python 环境问题
- 缺失依赖
- API key 未配置
- 可选工具的外部依赖未安装

### 5. 第一次运行

```bash
hermes
```

或者：

```bash
hermes chat -q "你好，请介绍一下你自己"
```

### 6. 使用交互式设置

如果你不确定怎么配置，可以直接运行：

```bash
hermes setup
```

它会引导你完成模型、工具、gateway 等常见配置。

---

## 这个项目到底是什么

从实现角度看，你可以把 Hermes Agent 理解成 4 层：

1. **Agent 主循环**：`run_agent.py` 里的 `AIAgent` 负责“问模型 -> 调工具 -> 再问模型”。
2. **工具系统**：`tools/` + `tools/registry.py` + `model_tools.py` 负责能力暴露与分发。
3. **运行入口**：`cli.py` 和 `gateway/run.py` 负责把同一个 agent 跑在终端或聊天平台里。
4. **扩展机制**：skills、plugins、MCP、memory provider、context engine 让它不必把所有能力写死在核心代码里。

---

## 项目目录结构概览

```text
hermes-agent/
├── run_agent.py          # 主编排器 AIAgent
├── cli.py                # 交互式 CLI/TUI
├── model_tools.py        # 工具发现与分发
├── toolsets.py           # toolset 分组定义
├── agent/                # prompt、compression、memory、模型元数据等
├── tools/                # 内置工具实现
├── gateway/              # 消息平台网关
├── hermes_cli/           # 子命令、配置、skins、plugins UI
├── skills/               # 内置技能
├── optional-skills/      # 可选技能
├── plugins/              # 仓库内 provider / engine 等扩展
├── cron/                 # 定时任务
├── tests/                # 测试套件
└── readme/               # 你现在正在看的文档
```

如果你刚开始读代码，最值得先看的 5 个文件通常是：

- `run_agent.py`
- `model_tools.py`
- `tools/registry.py`
- `cli.py`
- `toolsets.py`

---

## 下一步建议

按你的目标继续：

- 想理解项目：去看 [architecture.md](./architecture.md)
- 想开始开发：去看 [development-guide.md](./development-guide.md)
- 想找日常命令：去看 [cli-guide.md](./cli-guide.md)
