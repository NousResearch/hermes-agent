# Hermes × Codex Desktop 协作桥

## 这是什么

这是一个让 **Hermes 和 Codex Desktop 围绕同一个 Codex 任务线程协作**
的本地集成方案。

它不是让 Hermes 模拟鼠标键盘、自动点击 Codex Desktop，也不是在两个
Agent 之间复制 Prompt。它让 Hermes 通过 Codex App Server 创建和推进
真实的 Codex thread，并保存 Hermes session 与 Codex thread 的准确映射。
Codex Desktop 则继续作为这个 thread 的可视化操作界面。

可以把三者理解为：

- **Hermes**：主要聊天入口、上下文管理者和任务调度者。
- **Codex App Server**：实际承载 Codex thread、turn、执行事件和审批的运行时。
- **Codex Desktop**：查看 diff、终端、文件变化和人工接管任务的高级控制台。

```text
用户
 │
 ▼
Hermes
聊天、业务上下文、任务调度、结果汇总
 │
 ▼
Codex App Server
thread / turn / 执行 / 审批 / 历史
 │
 ├──────────────► Hermes 持续读取和推进
 │
 └──────────────► Codex Desktop 查看和人工接管
```

核心原则是：

> Hermes 和 Codex Desktop 使用同一个 Codex thread，而不是各自运行一份
> 相互独立的任务。

## 它解决什么问题

过去同时使用 Hermes 和 Codex Desktop 时，通常需要人工搬运上下文：

1. 在 Hermes 中讨论需求。
2. 把需求重新整理成 Prompt。
3. 复制到 Codex Desktop。
4. Codex 完成后，再把结果搬回 Hermes。
5. 下一次继续任务时，再次解释项目背景和已经完成的工作。

这种方式容易出现：

- 两边的任务状态不一致。
- 决策和约束在复制时遗漏。
- 同一件事生成多个互不关联的会话。
- 无法准确判断 Codex 当前执行到哪里。
- 人工接管后，Hermes 不知道发生了什么。
- 长任务需要反复复制日志、diff 和最终结果。

协作桥使用 Codex thread ID 作为唯一关联依据，避免通过标题、Prompt 文本
或模糊时间匹配来猜测两个任务是否相同。

## 做完之后有什么帮助

### 1. 大多数时候只需要和 Hermes 对话

Hermes 可以把适合交给 Codex 的代码任务送入 Codex App Server。用户不必
先打开 Codex Desktop，也不必重复编写任务 Prompt。

### 2. 打开 Desktop 时看到的是真实任务

需要查看代码 diff、终端输出、文件变化或进行人工干预时，可以直接打开
对应的 Codex thread，而不是进入一个空白窗口重新描述任务。

### 3. Hermes 重启后仍能继续原任务

Hermes 会把 session 与 thread ID 的映射持久化。运行时进程退出、客户端
被回收或者 Hermes 重启后，下一轮会调用 `thread/resume` 恢复原 thread，
而不是静默创建一个新任务。

### 4. 减少上下文重复与状态分裂

Codex 的执行历史保存在同一 thread 中。Hermes 可以继续推进它，Desktop
也可以读取和操作它，因此不需要在两个系统之间手动复制完整上下文。

### 5. 保留各自最擅长的体验

Hermes 继续负责长期业务上下文、聊天入口、记忆、渠道接入和任务调度；
Codex 继续负责代码执行、终端、补丁、沙箱和工程化开发体验。

## 适用谁

### 个人开发者和独立创业者

平时希望通过 Hermes 管理产品、内容和开发任务，但在复杂代码修改时需要
Codex Desktop 的 diff、终端和项目视图。

### 同时使用 Hermes 与 Codex 的重度用户

已经把 Hermes 当作主要 AI 入口，又不想放弃 Codex Desktop 的代码工作流，
并且厌倦在两个客户端之间反复复制 Prompt。

### 多项目负责人

需要由 Hermes 识别项目和任务上下文，再将具体工程工作交给对应目录中的
Codex thread，并保留可追踪的任务关系。

### 长任务和人工审查较多的团队

任务可能持续较长时间，中途需要人查看 diff、修改约束、处理审批或接管，
之后还希望 Hermes 能继续跟踪和总结。

## 典型使用场景

### 场景一：Hermes 全程代办普通代码任务

用户对 Hermes 说：

> 修复图片队列的并发竞争，并补充测试。

理想工作流：

1. Hermes 确定项目目录和任务约束。
2. Hermes 创建或恢复对应 Codex thread。
3. Hermes 启动 Codex turn。
4. Codex 在项目目录中检查代码、修改文件并运行测试。
5. Hermes 接收最终结果，并向用户汇总修改和验证情况。

用户可以只停留在 Hermes 中。

### 场景二：切换到 Codex Desktop 查看细节

当用户需要查看：

- 文件 diff；
- 终端输出；
- 具体修改过程；
- 测试失败信息；
- 多文件变更；
- 长任务实时状态；
- Codex 的项目操作界面；

可以打开同一个 Codex thread。无需复制 Prompt，也无需重新说明项目背景。

### 场景三：在 Desktop 中人工接管

用户在 Codex Desktop 中补充约束：

> 保留现有数据库结构，不进行 migration。

Codex 在原 thread 中继续执行。之后 Hermes 再恢复这个 thread 时，读取和
推进的仍然是同一个任务，而不是一份旧副本。

### 场景四：Hermes 或 App Server 重启

Hermes 进程退出后，内存中的 Codex 客户端会消失，但 Codex thread 本身
仍由 Codex 持久化。

Hermes 根据本地映射文件找到 thread ID，并使用 `thread/resume` 恢复任务。

### 场景五：审批回到 Hermes

Codex 执行命令、应用文件补丁或请求额外权限时，App Server 会发出双向
审批请求。Hermes 当前运行时已经能够把命令执行和文件变更审批接入 Hermes
的审批流程。

适合需要“自动执行低风险操作，高风险操作由人确认”的工作流。

## 当前已经实现的能力

截至 2026 年 7 月，当前代码已经具备：

- 通过 `codex app-server` 启动 Codex 本地运行时。
- 执行 `initialize` 握手。
- 使用 `thread/start` 创建真实 Codex thread。
- 使用 `turn/start` 启动任务轮次。
- 接收流式通知、工具事件、结果和 token usage。
- 处理命令执行、文件修改和权限相关审批。
- 将 Codex 事件投影回 Hermes 消息记录。
- 在同一个 Hermes 进程中复用同一 Codex thread。
- 返回 `codex_thread_id` 和 `codex_turn_id`。
- 持久化 Hermes session 到 Codex thread 的映射。
- Hermes 客户端重建后使用 `thread/resume` 恢复原 thread。
- 使用 `thread/read` 重新读取持久化 thread。
- 在当前 Codex Desktop 版本中按 thread ID 打开 App Server 创建的任务。

映射文件位置：

```text
~/.hermes/codex_thread_mappings.json
```

示例：

```json
{
  "20260716_120000_abcd1234": {
    "codex_thread_id": "019f6a62-8651-7ab3-a55b-ab42ec1879a8",
    "cwd": "/Users/example/Projects/image-platform"
  }
}
```

## 当前如何启用

先确保 Codex CLI 已安装并登录：

```bash
codex --version
codex login
```

在 Hermes 中启用 App Server 运行时：

```text
/codex-runtime codex_app_server
```

该设置在下一次 Hermes session 生效。查看当前状态：

```text
/codex-runtime
```

关闭并恢复 Hermes 默认运行时：

```text
/codex-runtime auto
```

## 当前边界

以下能力不能描述为已经完成：

- Hermes 尚未提供完整的“任务中心”来浏览所有 Codex thread 映射。
- 普通 Hermes 聊天界面尚未统一提供“在 Codex Desktop 中打开”按钮。
- Hermes 尚未自动读取 Desktop 中每一次人工操作并生成结构化决策记录。
- 当前映射主要是 Hermes session 与 Codex thread 的一对一关系，还不是完整的
  Project → Task → Thread 数据模型。
- 尚未实现面向多个独立 App Server 客户端的实时事件广播。
- 尚未实现跨机器共享 thread；当前方案是本机集成。
- Hermes 不控制 Codex Desktop 的窗口、按钮、布局或其他 GUI 状态。
- Desktop 的 thread 跳转能力依赖当前 Codex Desktop 版本，升级后仍需做兼容
  性回归测试。

## 不适用的情况

以下场景不需要或不适合使用这个协作桥：

- 完全不使用 Codex，只使用其他模型提供商。
- 只执行一次性的简单问答，没有代码项目或持续任务。
- 需要跨多台机器共享同一运行中任务。
- 希望 Hermes 自动操作 Codex Desktop 的 GUI。
- 任务只需要 Hermes 独有的 agent-loop 工具，并不需要 Codex 工程能力。
- 项目不能由本机 Codex Runtime 访问。

## 与“遥控 Desktop”方案的区别

| 方案 | 工作方式 | 主要问题 |
|---|---|---|
| GUI 遥控 | Hermes 模拟点击、输入和窗口操作 | 脆弱、依赖界面布局、难同步真实状态 |
| Prompt 复制 | Hermes 生成 Prompt，用户复制给 Codex | 上下文重复、任务分裂、结果需要搬运 |
| 共享 thread | Hermes 与 Desktop 使用同一 Codex thread | 状态统一，可恢复，可直接接管 |

本项目选择第三种方案。

## 主要代码位置

- `agent/transports/codex_app_server.py`：App Server JSON-RPC 客户端。
- `agent/transports/codex_app_server_session.py`：thread、turn、事件和审批生命周期。
- `agent/codex_runtime.py`：Hermes 对话循环与 Codex Runtime 的连接层。
- `agent/codex_thread_store.py`：Hermes session 与 Codex thread 的持久化映射。
- `hermes_cli/codex_runtime_switch.py`：运行时启用和关闭入口。
- `website/docs/user-guide/features/codex-app-server-runtime.md`：用户运行时文档。

## 验证状态

当前原型已经实际验证：

1. App Server 创建 thread。
2. 启动并完成 turn。
3. `thread/read` 读取原 thread。
4. Codex Desktop 按 thread ID 打开该任务。
5. 新 App Server 进程通过 `thread/resume` 恢复该任务。
6. 恢复后仍可读取原有 turn。
7. 相关自动化测试通过。

这证明核心闭环已经成立：

```text
Hermes 创建
  → Codex 执行
  → Desktop 打开/接管
  → Hermes 按 thread ID 恢复
```

## 后续产品化方向

建议按以下顺序继续建设：

1. 增加 Project Registry，明确项目名称与本地路径的映射。
2. 将当前 JSON 映射升级为 Hermes Task 数据模型。
3. 在 Hermes 回复中展示 Codex thread ID、状态和 Desktop 打开入口。
4. 增加 thread 列表、读取、归档和重新绑定命令。
5. 将 Desktop 中的重要人工指令同步为 Hermes 任务决策记录。
6. 完善审批网关的风险分级和各聊天渠道中的交互按钮。
7. 增加 Codex Desktop 升级后的自动兼容性测试。

## 一句话总结

Hermes × Codex Desktop 协作桥，把 Hermes 变成 Codex 任务的主要入口和
调度层，把 Codex Desktop 保留为高级工程控制台；两边通过同一个 Codex
thread 连续工作，从而减少 Prompt 搬运、上下文丢失和任务状态分裂。
