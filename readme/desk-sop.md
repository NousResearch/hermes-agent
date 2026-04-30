# Hermes Desk 使用 SOP

Hermes Desk 是 Hermes 的本地 Web 工作台。它和传统 dashboard 的区别是：dashboard 偏“管理 Hermes”，Desk 偏“用 Hermes 工作”。

当前 Desk 覆盖四件事：

1. 在浏览器里和 Hermes 对话。
2. 为本地项目绑定工作目录。
3. 查看一次任务的工具调用过程。
4. 管理长期记忆和常用配置。

---

## 1. 启动 Desk

在仓库根目录执行：

```bash
source venv/bin/activate
hermes desk
```

默认会启动本地服务：

```text
http://127.0.0.1:9119
```

如果不想自动打开浏览器：

```bash
hermes desk --no-open
```

如果默认端口被占用：

```bash
hermes desk --port 9120 --no-open
```

然后在浏览器打开：

```text
http://127.0.0.1:9120
```

补充说明：

- `hermes dashboard`、`hermes web`、`hermes desk` 当前指向同一个 Web UI 入口。
- Desk 默认只绑定 `127.0.0.1`，适合本机使用。
- 不建议随意用 `--host 0.0.0.0`，因为 Desk 能读写配置和 API keys。

---

## 2. 第一次进入 Desk

打开页面后，默认进入 `Desk` 页。

页面分三列：

| 区域 | 用途 |
|------|------|
| 左侧 Projects | 添加和选择本地项目目录 |
| 中间 Desk | 与 Hermes 对话、发送任务、查看回复 |
| 右侧 Run Timeline | 查看 cwd、工具调用、运行状态和最近 Desk 会话 |

如果你只是想普通聊天，也可以不添加项目，直接在中间输入框发消息。

如果你希望 Hermes 在某个代码仓库里工作，先添加项目。

---

## 3. 添加项目

在左侧 `Projects` 区域：

1. `Project name` 填一个易识别的名字，例如：

   ```text
   Hermes Agent
   ```

2. 路径输入框填本地项目的绝对路径，例如：

   ```text
   /Users/xuji/Code/ai/hermes-agent
   ```

3. 点击 `Add`。

添加成功后，项目会出现在左侧列表里。

Desk 会自动显示一些项目元信息：

- Git branch
- 是否有未提交变更
- 是否存在 `AGENTS.md`、`HERMES.md`、`.hermes.md`、`CLAUDE.md`、`.cursorrules` 等上下文文件

这些信息只用于帮助你确认 Hermes 当前会在哪个项目上下文里工作。

---

## 4. 创建或选择会话

选择一个项目后，可以直接发消息；如果当前没有会话，Desk 会自动创建一个新会话。

也可以手动点击中间区域右上角的 `New`：

1. Desk 创建一个 `source=desk` 的新 session。
2. 这个 session 会写入 Hermes 的 `state.db`。
3. 后续可以在 `Sessions` 页和右侧 `Recent Desk sessions` 里看到它。

右侧 `Recent Desk sessions` 可以选择最近的 Desk 会话。

注意：当前 MVP 选择旧会话后主要用于继续使用该 session id；完整历史回填到中间聊天窗口可以作为后续增强。

---

## 5. 发送任务

在中间输入框输入任务，例如：

```text
请阅读这个项目的结构，并告诉我 Desk 相关代码在哪些文件里。
```

然后点击 `Send`。

Desk 会做这些事：

1. 把当前项目路径绑定为该 session 的工作目录。
2. 创建 Hermes `AIAgent`。
3. 使用该 session 的历史消息作为上下文。
4. 通过 SSE 把回复和工具事件实时推到浏览器。

如果项目路径是：

```text
/Users/xuji/Code/ai/hermes-agent
```

那么 terminal/file/context 相关工具会围绕这个 cwd 工作。

---

## 6. 查看运行过程

右侧 `Run Timeline` 会显示任务过程。

常见事件：

| 事件 | 含义 |
|------|------|
| `Run started` | 任务已开始 |
| `Tool started` | Hermes 开始调用某个工具 |
| `Tool completed` | 工具调用结束 |
| `Reasoning available` | 模型提供了可展示的推理摘要 |
| `Run completed` | 任务完成 |
| `Run failed` | 任务失败 |

如果 Hermes 正在长时间执行，可以点击 `Stop`。

`Stop` 会向后端发出 interrupt 请求，让当前 agent 尽快中断。

---

## 7. 管理记忆

点击顶部导航里的 `Memory`。

当前 Desk 支持编辑两类内置 curated memory：

| 标签 | 文件 | 用途 |
|------|------|------|
| Agent Memory | `~/.hermes/memories/MEMORY.md` | Hermes 对环境、项目、工具习惯的长期记忆 |
| User Profile | `~/.hermes/memories/USER.md` | 用户偏好、沟通风格、稳定需求 |

操作流程：

1. 进入 `Memory` 页。
2. 切换 `Agent Memory` 或 `User Profile`。
3. 点击 `Add Entry` 新增条目。
4. 编辑文本。
5. 点击 `Save`。

保存时会复用内置 memory 的安全规则：

- 空条目会被忽略。
- 重复条目会去重。
- 超过长度限制会被拒绝。
- 疑似 prompt injection 或 secret exfiltration 的内容会被拒绝。

重要：memory 会在新 session 启动时注入 system prompt。为了不破坏 prompt caching，Desk 中保存的 memory 不会强行改写正在运行的 session prompt。

---

## 8. 管理配置、Keys、Skills 和工具

Desk 保留原有 Web UI 的管理页面：

| 页面 | 用途 |
|------|------|
| Status | 查看 Hermes 版本、gateway、活跃会话 |
| Sessions | 查看、搜索、删除历史会话 |
| Analytics | 查看 token 和成本统计 |
| Logs | 查看日志 |
| Cron | 管理定时任务 |
| Skills | 查看和启停 skills |
| Config | 编辑 `config.yaml` |
| Keys | 管理 `.env` API keys |

注意：

- 配置和工具变更通常对新 session 生效。
- 不要期待它们在一个已经启动的 agent session 中实时改变工具面。
- 这是为了避免破坏 Hermes 的 prompt caching 和 toolset 稳定性。

---

## 9. 推荐日常工作流

### 代码项目工作流

```text
1. hermes desk
2. 打开 Desk 页面
3. 添加项目路径
4. 选择项目
5. 点击 New
6. 输入任务
7. 观察 Run Timeline
8. 需要时去 Sessions 查看历史
```

适合任务：

- 让 Hermes 熟悉项目结构
- 查找某个功能入口
- 规划改动
- 执行简单代码任务
- 查看工具调用过程

### 记忆维护工作流

```text
1. 打开 Memory
2. 阅读 Agent Memory / User Profile
3. 删除过期条目
4. 合并重复条目
5. 新增稳定偏好或长期事实
6. 保存
7. 新开 session 生效
```

适合保存的内容：

- 用户长期偏好
- 项目固定约定
- 工具使用注意事项
- 经常重复纠正 Hermes 的规则

不适合保存的内容：

- 临时任务进度
- 一次性结论
- 已完成工作的流水账
- 敏感密钥和 token

---

## 10. 停止 Desk

如果是在终端前台运行：

```text
Ctrl+C
```

如果用了自定义端口，可以检查：

```bash
lsof -i :9120
```

再按需结束进程。

---

## 11. 常见问题

### 打不开页面

先确认服务是否启动：

```bash
curl http://127.0.0.1:9119/api/status
```

如果使用了自定义端口，把 `9119` 换成你的端口。

### 添加项目失败

确认路径满足：

- 是绝对路径
- 目录存在
- 当前用户有读取权限

例如：

```text
/Users/xuji/Code/ai/hermes-agent
```

不要填：

```text
./hermes-agent
```

### 发送任务后没有回复

检查：

1. `.env` 中是否配置了可用模型 provider key。
2. `Config` 中默认模型是否可用。
3. `Logs` 页面是否有 provider 或网络错误。
4. 终端里启动 Desk 的进程是否还在运行。

### Memory 保存失败

常见原因：

- 条目超过字符限制。
- 内容包含疑似 prompt injection。
- 内容为空。

可以先缩短条目，只保留稳定事实。

---

## 12. 当前 MVP 边界

当前 Desk 已经能完成核心工作流，但仍是 MVP。

已支持：

- 本地项目 registry
- Desk session 创建
- 项目 cwd 绑定
- 流式消息输出
- 工具调用 timeline
- interrupt 请求
- curated memory 编辑
- 现有 dashboard 管理页面复用

后续可增强：

- 选择旧 session 后自动回填完整聊天历史
- 项目级默认 model/toolsets 的前端编辑
- 更完整的 tool call 参数和结果详情
- 文件树、diff、终端面板
- 外部 memory provider 的统一搜索和审计
