# pdr-agent

PDR 的智能体引擎，由 PDR 团队自行维护和开发。

它源自开源项目 Hermes Agent（MIT 许可，原版权声明保留在 [`LICENSE`](LICENSE)）。从本版本起独立演进：PDR 使用它的全部核心能力，后续的权限、数据接入等都在这里实现。管理端、用户端不在这里，在仓库的 `pdr/web/` 下自建。

## 在 PDR 中的位置

```
<仓库根>/
├── start_run.py          # 一键启动 PDR（安装依赖、起基础服务、建表、拉起引擎与前端）
├── pyproject.toml        # 后端 Python 环境（uv 管理）：uv sync 安装 PDR 服务 + 本引擎（可编辑）
├── package.json          # 前端工作区（pnpm 管理）：pdr/web/*
├── pdr/                  # PDR 业务后端、管理端、用户端、配置（pdr/config.yaml）
└── pdr-agent/            # 本目录：智能体引擎
```

PDR 后端通过引擎的消息网关（`gateway/`，api_server 平台，默认端口 8642）调用智能体。每个租户对应一个 profile，请求走 `/p/<profile>/` 前缀。PDR 管理端「智能体配置」页里的 967 个配置项，来自本目录的 `hermes_cli/config_defaults.py`，由 `pdr/scripts/gen_config_yaml.py` 同步到 `pdr/config.yaml`。

## 能力与目录

| 目录 | 内容 |
|---|---|
| `hermes_agent/` | 核心入口模块：`run_agent`（智能体对话循环入口）、`cli`、`model_tools`（工具编排）、`toolsets`、`hermes_constants`（路径与 profile）、`hermes_bootstrap` 等；`compat/` 为模块改名兼容层 |
| `agent/` | 对话回合、上下文压缩、提示词组装、模型与辅助模型解析、记忆、多语言 |
| `tools/` | 内置工具：终端（本地 / Docker / SSH 等后端）、文件、浏览器、网页、视觉、代码执行、MCP、子任务委派 |
| `skills/`、`optional-skills/` | 内置技能与可选技能库 |
| `gateway/` | 消息网关：api_server（PDR 接入点）、会话、流式输出、通知；各消息平台适配器在 `plugins/platforms/` |
| `plugins/`、`plugin-catalog/` | 插件（消息平台、记忆提供方、看板等）与插件目录 |
| `hermes_state.py`、`hermes_agent/state/` | 会话存储（SQLite）：存储本体与会话、消息、全文检索、压缩、修复等主题模块 |
| `hermes_cli/` | `hermes` 命令行：配置系统、profile、网关与定时任务管理、模型与提供商设置、插件 |
| `cron/` | 定时任务与看板调度 |
| `providers/` | 模型提供商注册表 |
| `tui_gateway/`、`ui-tui/` | 终端界面（后端与前端），供开发调试与运维使用 |
| `acp_adapter/` | 编辑器 ACP 接入 |
| `pm/` | 运行时与工具链的包管理（Python、Node、浏览器等的锁定版本） |
| `docker/` | 镜像初始化脚本与 `docker-compose*.yml` |
| `locales/`、`assets/` | 多语言文案与资源 |
| `docs/` | 开发说明（`CONTRIBUTING.md`）、安全说明（`SECURITY.md`）、默认人格参考（`SOUL.md`） |
| `website/` | 上游的完整功能文档，作为参考资料保留，会逐步改写 |
| `scripts/`、`tests/`、`evals/` | 开发与构建脚本、自动化测试、模型评测工具 |

## 本版本相对上游的变化

- 项目名改为 `pdr-agent`（`pyproject.toml`、`package.json`）。
- 移除与 PDR 无关的部分：桌面客户端（`apps/`）、网页仪表盘前端（`web/`）及对应的打包脚本、测试和 CI，还有贡献者名录、外语翻译、Nix 打包、独立安装脚本、上游评审机器人配置。
- 文件归类：
  - 根目录 20 个单文件模块移入 `hermes_agent/`，例如 `import run_agent` 改为 `from hermes_agent import run_agent`；
  - 67 个 `hermes_state_*.py` 移入 `hermes_agent/state/`，例如 `import hermes_state_sessions` 改为 `import hermes_agent.state.sessions`；存储本体 `hermes_state.py` 留在根目录。`hermes_agent/state/` 刻意不加载存储本体，导入单个主题模块不会引起循环导入；
  - 说明文档移入 `docs/`，配置模板移入 `hermes_cli/data/`。
- 旧模块名仍可导入：`hermes_agent/compat/moves.py` 登记了所有"旧名 → 新名"，运行时由兼容查找器指向同一个模块对象，第三方插件不受影响。仓库内代码一律使用新名，测试里的严格模式会拒绝旧名。

## 开发

日常在 PDR 中使用引擎：在仓库根目录执行 `uv sync`，根目录的 `.venv` 里即可直接 `import hermes_agent`、运行 `hermes` 命令。

单独开发、测试引擎本身时，需要 Python 3.14。按 `docs/CONTRIBUTING.md` 的「手动开发与测试环境」构建独立的测试环境：

```bash
cd pdr-agent
python -m pm.build_env --source . --out .venv --group dev --group test --extra messaging
```

运行测试（每个测试文件在独立进程中运行），必须给出路径：

```bash
scripts/run_tests.sh -j 4 tests/hermes_state tests/gateway/test_api_server.py
```

代码检查：

```bash
python scripts/check
```

## 开发约定

- 新增或移动模块：仓库内引用全部改为新名，并在 `hermes_agent/compat/moves.py` 登记旧名，不要删除已有条目。
- 新增配置项：写入 `hermes_cli/config_defaults.py`，再运行 `python pdr/scripts/gen_config_yaml.py`，并在 `pdr/server/pdr_server/agent/config_zh.json` 补上中文名称和说明。
- 多租户相关代码：一个进程同时服务多个 profile（租户）。凡在会话回合之外运行的代码，都必须显式绑定所属 profile，详见 `gateway/AGENTS.md` 的「Profile scope」。
- 依赖：修改 `pyproject.toml` 后用 `python -m pm.cli lock` 重新生成 `uv.lock`，并一起提交。

## 许可

MIT。原项目 Hermes Agent 的版权声明见 [`LICENSE`](LICENSE)，按 MIT 许可要求随本项目保留。
