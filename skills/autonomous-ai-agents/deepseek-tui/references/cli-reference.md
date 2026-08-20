# DeepSeek CLI 完整参考手册

> DeepSeek TUI 版本: 0.8.33
> 生成日期: 2026-05-14
> 配置文件: `~/.deepseek/config.toml`

---

## 目录

1. [快速入门](#快速入门)
2. [全局选项](#全局选项)
3. [命令分类索引](#命令分类索引)
4. [会话管理](#会话管理)
5. [Agent 执行](#agent-执行)
6. [配置与认证](#配置与认证)
7. [模型管理](#模型管理)
8. [服务与传输](#服务与传输)
9. [诊断与监控](#诊断与监控)
10. [实用工具](#实用工具)
11. [常用工作流示例](#常用工作流示例)
12. [环境变量](#环境变量)
13. [配置文件参考](#配置文件参考)

---

## 快速入门

```bash
# 最简单的调用：直接传 prompt
deepseek "解释这段代码的作用"

# 非交互式 agent 模式（最常用）
deepseek exec "帮我创建一个 Python 项目" --auto --yolo

# 管道输入
echo "这段代码有什么问题？" | deepseek exec --auto
cat error.log | deepseek exec "分析这些错误" --auto --yolo

# 启动交互式 TUI
deepseek-tui
```

---

## 全局选项

所有子命令都可以使用以下全局选项：

| 选项 | 说明 |
|------|------|
| `--config <CONFIG>` | 指定配置文件路径 |
| `--profile <PROFILE>` | 指定配置 profile |
| `--provider <PROVIDER>` | 指定 AI provider：`deepseek`, `nvidia-nim`, `openai`, `atlascloud`, `openrouter`, `novita`, `fireworks`, `sglang`, `vllm`, `ollama` |
| `--model <MODEL>` | 指定模型名称 |
| `--api-key <API_KEY>` | 指定 API key（覆盖配置文件） |
| `--base-url <BASE_URL>` | 自定义 API 基础 URL |
| `--output-mode <MODE>` | 输出模式 |
| `--log-level <LEVEL>` | 日志级别 |
| `--telemetry <BOOL>` | 遥测开关：`true` / `false` |
| `--approval-policy <POLICY>` | 审批策略 |
| `--sandbox-mode <MODE>` | 沙箱模式 |
| `--yolo` | YOLO 模式：自动批准所有工具调用 |
| `--mouse-capture` / `--no-mouse-capture` | 鼠标捕获开关 |
| `--skip-onboarding` | 跳过首次引导 |
| `-p, --prompt <PROMPT>` | 通过参数传入 prompt |
| `-h, --help` | 打印帮助信息 |
| `-V, --version` | 打印版本号 |

---

## 命令分类索引

### 会话管理
| 命令 | 说明 |
|------|------|
| `sessions` | 列出已保存的 TUI 会话 |
| `resume` | 恢复一个已保存的 TUI 会话 |
| `fork` | 复刻一个已保存的 TUI 会话 |
| `thread` | 管理线程/会话元数据及恢复/复刻流程 |

### Agent 执行
| 命令 | 说明 |
|------|------|
| `run` | 运行交互式/非交互式流程 |
| `exec` | **非交互式 agent 命令（主力 CLI 入口）** |
| `review` | 对 git diff 进行 AI 代码审查 |
| `eval` | 运行离线 TUI 评估框架 |

### 配置与认证
| 命令 | 说明 |
|------|------|
| `config` | 读取/写入/列出配置值 |
| `login` | 保存 provider API key 到配置 |
| `logout` | 删除已保存的认证状态 |
| `auth` | 管理认证凭据和 provider 模式 |
| `init` | 在当前目录创建默认 `AGENTS.md` |
| `setup` | 引导初始化 MCP 配置和/或 skills 目录 |

### 模型管理
| 命令 | 说明 |
|------|------|
| `models` | 列出可用的 DeepSeek API 模型 |
| `model` | 解析或列出各 provider 的可用模型 |

### 服务与传输
| 命令 | 说明 |
|------|------|
| `serve` | 运行本地 TUI 服务器 |
| `app-server` | 运行 app-server 传输层 |
| `mcp-server` | 通过 stdio 运行 MCP 服务器模式 |
| `mcp` | 管理 TUI MCP 服务器 |

### 诊断与监控
| 命令 | 说明 |
|------|------|
| `doctor` | 运行 DeepSeek TUI 诊断 |
| `metrics` | 打印用量汇总（来自审计日志和会话存储） |
| `sandbox` | 评估沙箱/审批策略决策 |
| `features` | 检查 TUI 功能标志 |

### 实用工具
| 命令 | 说明 |
|------|------|
| `apply` | 将 patch 文件或 stdin 应用到工作树 |
| `completion` / `completions` | 生成 shell 补全脚本 |
| `update` | 检查并应用 `deepseek` 二进制更新 |

---

## 会话管理

### `deepseek sessions`

列出已保存的 TUI 会话。

```bash
deepseek sessions
```

### `deepseek resume`

恢复一个已保存的 TUI 会话。

```bash
deepseek resume [ARGS]...
```

### `deepseek fork`

复刻（复制）一个已保存的 TUI 会话。

```bash
deepseek fork [ARGS]...
```

### `deepseek thread`

管理线程/会话元数据。

```bash
deepseek thread <SUBCOMMAND>

子命令:
  list        列出所有 thread
  read        读取指定 thread 详情
  resume      恢复一个 thread
  fork        复刻一个 thread
  archive     归档一个 thread
  unarchive   取消归档
  set-name    设置 thread 名称
```

示例：

```bash
deepseek thread list                          # 列出所有 thread
deepseek thread read <THREAD_ID>              # 查看 thread 详情
deepseek thread resume <THREAD_ID>            # 恢复指定 thread
deepseek thread fork <THREAD_ID>              # 复刻指定 thread
deepseek thread set-name <THREAD_ID> "名称"   # 重命名 thread
deepseek thread archive <THREAD_ID>           # 归档 thread
```

---

## Agent 执行

### `deepseek run`

运行交互式/非交互式流程（通用入口）。

```bash
deepseek run [ARGS]...
```

### `deepseek exec` ⭐ 主力 CLI 命令

非交互式 agent 模式，适合脚本集成和自动化。

```bash
deepseek exec [ARGS]...
```

**核心标志：**

| 标志 | 说明 |
|------|------|
| `--auto` | **启用 agentic 模式**，允许调用工具（读写文件、执行命令等） |
| `--json` | 输出结构化 JSON 摘要 |
| `--resume <SESSION_ID>` | 通过 ID 或前缀恢复之前的会话 |
| `--session-id <SESSION_ID>` | 同上 |
| `--continue` | 继续当前工作空间最近的会话 |
| `--output-format <FORMAT>` | 输出格式：`text` 或 `stream-json` |

**典型用法：**

```bash
# 基础 agent 调用
deepseek exec "分析项目结构" --auto

# 自动批准所有操作（全自动模式）
deepseek exec "重构 fanren_project 的模块" --auto --yolo

# 输出 JSON 结果供脚本消费
deepseek exec "列出所有 Python 文件" --auto --json --yolo

# 流式 JSON 输出（适合实时管道处理）
deepseek exec "..." --auto --output-format stream-json

# 继续最近的会话
deepseek exec "继续上一步的工作" --continue --auto

# 恢复指定会话
deepseek exec "..." --resume abc123 --auto

# 管道输入
cat requirements.txt | deepseek exec "根据依赖生成 Dockerfile" --auto --yolo

# 指定模型和 provider
deepseek exec "..." --auto --model deepseek-v4-flash --provider deepseek --yolo
```

### `deepseek review`

对 git diff 进行 AI 代码审查。

```bash
deepseek review [ARGS]...
```

**典型用法：**

```bash
# 审查暂存区的变更
deepseek review

# 审查指定文件
deepseek review path/to/file.py

# 审查最近的提交
git diff HEAD~1 | deepseek review
```

### `deepseek eval`

运行离线 TUI 评估框架。

```bash
deepseek eval [ARGS]...
```

---

## 配置与认证

### `deepseek config`

读取/写入/列出配置值。

```bash
deepseek config <SUBCOMMAND>

子命令:
  list        列出所有配置项
  path        显示配置文件路径
  set <key> <value>  设置配置值
  get <key>   获取配置值
```

示例：

```bash
deepseek config list                    # 列出所有配置
deepseek config path                    # 查看配置文件路径
deepseek config set provider deepseek   # 设置默认 provider
deepseek config get default_text_model  # 获取默认模型
```

### `deepseek login`

保存 provider API key 到配置。

```bash
deepseek login --api-key sk-xxx                    # 默认 provider
deepseek login --provider openai --api-key sk-xxx  # 指定 provider
```

### `deepseek logout`

删除已保存的认证状态。

```bash
deepseek logout                     # 删除所有
deepseek logout --provider openai   # 删除指定 provider
```

### `deepseek auth`

管理认证凭据和 provider 模式。

```bash
deepseek auth status                # 查看认证状态
deepseek auth list                  # 列出所有 provider 认证状态
deepseek auth set --api-key sk-xxx  # 保存 API key
deepseek auth set --provider openai --api-key sk-xxx
```

### `deepseek init`

在当前目录创建默认 `AGENTS.md` 文件。

```bash
deepseek init                       # 使用默认模板
deepseek init --template python     # 指定模板
```

### `deepseek setup`

引导初始化配置。

```bash
deepseek setup                      # 引导式设置
```

---

## 模型管理

### `deepseek models`

列出可用的 DeepSeek API 模型。

```bash
deepseek models
```

### `deepseek model`

解析或列出各 provider 的可用模型。

```bash
deepseek model list                          # 列出所有 provider 的模型
deepseek model list --provider deepseek      # 列出 DeepSeek 的模型
deepseek model resolve                       # 显示当前解析出的模型
```

---

## 服务与传输

### `deepseek serve`

运行本地 TUI 服务器。

```bash
deepseek serve [ARGS]...

常用标志:
  --http          启用 HTTP/SSE 服务器
  --port <PORT>   绑定端口（默认 7878）
  --host <HOST>   绑定地址（默认 127.0.0.1）
  --workers <N>   后台 worker 数量（1-8，默认 2）
  --auth-token    认证 token
  --insecure      跳过认证（仅用于 localhost）
```

### `deepseek app-server`

运行 app-server 传输层。

```bash
deepseek app-server [ARGS]...

常用标志:
  --port <PORT>   绑定端口（默认 8787）
  --host <HOST>   绑定地址
```

### `deepseek mcp-server`

通过 stdio 运行 MCP 服务器模式。

```bash
deepseek mcp-server
```

### `deepseek mcp`

管理 TUI MCP 服务器。

```bash
deepseek mcp list                    # 列出配置的服务器
deepseek mcp add <name> -- <cmd>     # 添加服务器
deepseek mcp connect                 # 测试连接
deepseek mcp tools                   # 列出发现的工具
deepseek mcp enable <name>           # 启用服务器
deepseek mcp disable <name>          # 禁用服务器
deepseek mcp add-self                # 注册为 MCP stdio 服务器
```

---

## 诊断与监控

### `deepseek doctor`

运行诊断检查。

```bash
deepseek doctor
```

### `deepseek metrics`

打印用量汇总。

```bash
deepseek metrics --since 7d                    # 最近 7 天
deepseek metrics --since 30d --json            # JSON 格式
```

### `deepseek sandbox`

评估沙箱/审批策略决策。

```bash
deepseek sandbox [ARGS]...
```

### `deepseek features`

检查 TUI 功能标志。

```bash
deepseek features
```

---

## 实用工具

### `deepseek apply`

将 patch 文件或 stdin 应用到工作树。

```bash
deepseek apply <patch_file>      # 应用 patch 文件
cat diff.patch | deepseek apply  # 从 stdin 读取
```

### `deepseek completion`

生成 shell 补全脚本。

```bash
deepseek completion bash         # Bash 补全
deepseek completion zsh          # Zsh 补全
```

### `deepseek update`

检查并应用更新。

```bash
deepseek update
```

---

## 常用工作流示例

### 代码重构

```bash
# 分析项目结构
deepseek exec "分析项目结构，找出可以优化的模块" --auto --yolo

# 重构指定模块
deepseek exec "重构 auth 模块，使用 JWT 替代 session" --auto --yolo

# 继续重构工作
deepseek exec "继续重构工作" --continue --auto --yolo
```

### 错误诊断

```bash
# 分析错误日志
cat /var/log/app/error.log | deepseek exec "分析这些错误，找出根本原因" --auto --yolo

# 生成修复方案
deepseek exec "根据上面的错误分析，生成修复方案" --continue --auto --yolo
```

### 代码审查

```bash
# 审查当前变更
deepseek review --json

# 审查特定文件
deepseek review src/auth/ --json

# 审查 PR
deepseek review --pr 123 --json
```

### 多 provider 切换

```bash
# 使用 OpenAI
deepseek exec "..." --auto --provider openai --model gpt-4.1 --yolo

# 使用本地 Ollama
deepseek exec "..." --auto --provider ollama --model deepseek-coder:1.3b --yolo
```

---

## 环境变量

| 变量 | 说明 |
|------|------|
| `DEEPSEEK_API_KEY` | DeepSeek API key（优先级最高） |
| `DEEPSEEK_CORS_ORIGINS` | CORS 允许的源（HTTP 服务器用） |
| `DEEPSEEK_RUNTIME_TOKEN` | 运行时 API 认证 token |

---

## 配置文件参考

**路径:** `~/.deepseek/config.toml`

```toml
# 默认 provider
provider = "deepseek"

# API key（也可通过 deepseek login 设置）
api_key = "sk-xxx"

# 默认模型
default_text_model = "deepseek-v4-pro"

# 推理努力程度: low, medium, high, max
reasoning_effort = "max"

# 项目特定配置
[projects."/root/my-project"]
trust_level = "trusted"
default_model = "deepseek-v4-flash"

[projects."/root/sandbox"]
trust_level = "sandbox"
```

**认证查找顺序:** config → secret store → env var (`DEEPSEEK_API_KEY`)
