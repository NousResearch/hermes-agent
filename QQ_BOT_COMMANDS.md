# Hermes Agent — QQ 开放平台指令配置

> 适用于 QQ 机器人 v2（q.qq.com）「指令」配置页面
> 最后更新：2026-06-09

---

## 快速配置方法

在 QQ 开放平台后台 → 你的机器人 → **指令配置** 中，逐条添加以下指令。
每条指令需要填写：**指令名称**（不含 `/`）、**指令描述**、**参数**（如有）。

---

## 一、会话管理（Session）

### 1. start
- **名称**: `start`
- **描述**: 确认平台启动消息（静默模式，不回复）
- **参数**: 无

### 2. new
- **名称**: `new`
- **描述**: 开始新会话，清除历史记录
- **别名**: `reset`
- **参数**: `[name]` — 可选，为新会话命名

### 3. retry
- **名称**: `retry`
- **描述**: 重试上一条消息（将最后一条消息重新发送给 AI 处理）
- **参数**: 无

### 4. undo
- **名称**: `undo`
- **描述**: 撤回最近 N 轮用户消息并重新提问（默认 1 轮）
- **参数**: `[N]` — 可选，撤回轮数（默认 1）

### 5. title
- **名称**: `title`
- **描述**: 设置当前会话的标题
- **参数**: `[name]` — 标题名称，不填则显示当前标题

### 6. branch
- **名称**: `branch`
- **描述**: 分支当前会话（从当前状态分叉，探索不同方向）
- **别名**: `fork`
- **参数**: `[name]` — 可选，分支名称

### 7. compress
- **名称**: `compress`
- **描述**: 压缩对话上下文，节省 Token；可指定保留最近 N 轮
- **参数**: `[here N]` 或 `[focus topic]` — 可选

### 8. rollback
- **名称**: `rollback`
- **描述**: 列出或恢复文件系统检查点
- **参数**: `[number]` — 可选，要恢复的检查点编号

### 9. stop
- **名称**: `stop`
- **描述**: 终止所有正在后台运行的进程
- **参数**: 无

### 10. approve
- **名称**: `approve`
- **描述**: 批准待执行的危险命令（当 AI 请求执行敏感操作时）
- **参数**: `[session|always]` — session 本次允许，always 始终允许

### 11. deny
- **名称**: `deny`
- **描述**: 拒绝待执行的危险命令
- **参数**: 无

### 12. background
- **名称**: `background`
- **描述**: 在后台运行提示词（不阻塞当前对话）
- **别名**: `bg`, `btw`
- **参数**: `<prompt>` — 必填，后台执行的提示词内容

### 13. agents
- **名称**: `agents`
- **描述**: 显示当前活跃的代理和正在运行的任务
- **别名**: `tasks`
- **参数**: 无

### 14. queue
- **名称**: `queue`
- **描述**: 将提示词排队到下一轮处理（不打断当前任务）
- **别名**: `q`
- **参数**: `<prompt>` — 必填，要排队的提示词

### 15. steer
- **名称**: `steer`
- **描述**: 在下一个工具调用之后注入消息（不中断当前操作）
- **参数**: `<prompt>` — 必填，注入的消息内容

### 16. goal
- **名称**: `goal`
- **描述**: 设置持续目标，AI 跨轮次自动推进直到完成
- **参数**: `[text | pause | resume | clear | status]` — 设置文本、暂停、恢复、清除或查看状态

### 17. subgoal
- **名称**: `subgoal`
- **描述**: 添加或管理当前目标的子条件
- **参数**: `[text | remove N | clear]` — 添加子条件、移除第 N 条、或清除全部

### 18. status
- **名称**: `status`
- **描述**: 显示当前会话的详细信息
- **参数**: 无

### 19. sethome
- **名称**: `sethome`
- **描述**: 将当前聊天设为 Home 频道（默认消息接收频道）
- **别名**: `set-home`
- **参数**: 无

### 20. resume
- **名称**: `resume`
- **描述**: 恢复之前命名的会话
- **参数**: `[name]` — 会话名称

### 21. sessions
- **名称**: `sessions`
- **描述**: 浏览和恢复历史会话列表
- **参数**: 无

### 22. restart
- **名称**: `restart`
- **描述**: 优雅重启网关（等待当前任务完成后再重启）
- **参数**: 无

---

## 二、配置管理（Configuration）

### 23. model
- **名称**: `model`
- **描述**: 切换当前会话使用的 AI 模型
- **参数**: `[model] [--provider name] [--global] [--refresh]`

### 24. codex-runtime
- **名称**: `codex-runtime`
- **描述**: 切换 OpenAI/Codex 模型的 Codex 应用服务器运行时
- **别名**: `codex_runtime`
- **参数**: 无

### 25. personality
- **名称**: `personality`
- **描述**: 设置预设人格风格
- **参数**: `[name]` — 人格名称，不填则显示可用列表

### 26. verbose
- **名称**: `verbose`
- **描述**: 循环切换工具进度显示模式（关 -> 新 -> 全部 -> 详细）
- **参数**: 无
- **注意**: 需在配置中开启 `display.tool_progress_command` 后才可用

### 27. footer
- **名称**: `footer`
- **描述**: 切换 AI 回复底部的网关运行时元数据页脚
- **参数**: `[on|off|status]`

### 28. yolo
- **名称**: `yolo`
- **描述**: 切换 YOLO 模式（跳过所有危险命令审批，直接执行）
- **参数**: 无

### 29. reasoning
- **名称**: `reasoning`
- **描述**: 管理推理强度和显示方式
- **参数**: `[level|show|hide]`

### 30. fast
- **名称**: `fast`
- **描述**: 切换快速模式（OpenAI 优先处理 / Anthropic 快速模式）
- **参数**: `[normal|fast|status]`

### 31. voice
- **名称**: `voice`
- **描述**: 切换语音模式（AI 回复转为语音）
- **参数**: `[on|off|tts|status]`

---

## 三、工具与技能（Tools & Skills）

### 32. bundles
- **名称**: `bundles`
- **描述**: 列出可用技能包（技能组合的别名）
- **参数**: 无

### 33. curator
- **名称**: `curator`
- **描述**: 后台技能维护（查看状态、运行、固定、归档技能）
- **参数**: `[subcommand]` — status / run / pin / archive / list-archived

### 34. kanban
- **名称**: `kanban`
- **描述**: 多配置文件协作看板（任务管理、链接、评论）
- **参数**: `[subcommand]`

### 35. reload-mcp
- **名称**: `reload-mcp`
- **描述**: 重新加载 MCP（Model Context Protocol）服务器配置
- **别名**: `reload_mcp`
- **参数**: 无

### 36. reload-skills
- **名称**: `reload-skills`
- **描述**: 重新扫描技能目录，加载新安装或已移除的技能
- **别名**: `reload_skills`
- **参数**: 无

---

## 四、信息查看（Info）

### 37. whoami
- **名称**: `whoami`
- **描述**: 显示你的命令访问权限（管理员/普通用户）
- **参数**: 无

### 38. profile
- **名称**: `profile`
- **描述**: 显示当前活跃的配置文件和主目录路径
- **参数**: 无

### 39. commands
- **名称**: `commands`
- **描述**: 浏览所有可用命令和技能（分页显示）
- **参数**: `[page]` — 页码

### 40. help
- **名称**: `help`
- **描述**: 显示所有可用命令的简要列表
- **参数**: 无

### 41. usage
- **名称**: `usage`
- **描述**: 显示当前会话的 Token 用量和速率限制
- **参数**: 无

### 42. insights
- **名称**: `insights`
- **描述**: 显示用量洞察和使用分析
- **参数**: `[days]` — 查看最近 N 天的数据

### 43. platform
- **名称**: `platform`
- **描述**: 暂停、恢复或列出网关消息平台状态
- **参数**: `<pause|resume|list> [name]`

### 44. update
- **名称**: `update`
- **描述**: 更新 Hermes Agent 到最新版本
- **参数**: 无

### 45. version
- **名称**: `version`
- **描述**: 显示当前 Hermes Agent 版本号
- **别名**: `v`
- **参数**: 无

### 46. debug
- **名称**: `debug`
- **描述**: 上传调试报告（系统信息+日志）并获取分享链接
- **参数**: 无

---

## 五、QQ 特有交互（自动触发，无需配置指令）

| 场景 | 按钮 | 说明 |
|------|------|------|
| 危险命令审批 | ✅ 允许一次 | 允许本次执行 |
| 危险命令审批 | ⭐ 始终允许 | 本次会话内始终允许 |
| 危险命令审批 | ❌ 拒绝 | 拒绝执行 |
| 版本更新确认 | Yes / No | 确认是否执行更新 |

---

## 六、配置速查表

在 QQ 开放平台后台快速填写时参考：

| 指令名称 | 指令描述 | 参数 |
|----------|----------|------|
| start | 确认启动消息（静默） | 无 |
| new | 开始新会话 | [name] |
| retry | 重试上一条消息 | 无 |
| undo | 撤回 N 轮并重新提问 | [N] |
| title | 设置会话标题 | [name] |
| branch | 分支当前会话 | [name] |
| compress | 压缩对话上下文 | [here N] |
| rollback | 列出/恢复检查点 | [number] |
| stop | 终止后台进程 | 无 |
| approve | 批准危险命令 | [session/always] |
| deny | 拒绝危险命令 | 无 |
| background | 后台运行提示词 | prompt |
| agents | 显示活跃代理和任务 | 无 |
| queue | 排队提示词 | prompt |
| steer | 注入消息到工具调用后 | prompt |
| goal | 设置持续目标 | [text/pause/resume/clear/status] |
| subgoal | 管理目标子条件 | [text/remove N/clear] |
| status | 显示会话信息 | 无 |
| sethome | 设为 Home 频道 | 无 |
| resume | 恢复历史会话 | [name] |
| sessions | 浏览历史会话列表 | 无 |
| restart | 优雅重启网关 | 无 |
| model | 切换 AI 模型 | [model] |
| codex-runtime | 切换 Codex 运行时 | 无 |
| personality | 设置预设人格 | [name] |
| verbose | 切换工具进度显示 | 无 |
| footer | 切换回复页脚 | [on/off/status] |
| yolo | 切换 YOLO 模式 | 无 |
| reasoning | 管理推理强度 | [level/show/hide] |
| fast | 切换快速模式 | [normal/fast/status] |
| voice | 切换语音模式 | [on/off/tts/status] |
| bundles | 列出技能包 | 无 |
| curator | 技能维护 | [subcommand] |
| kanban | 协作看板 | [subcommand] |
| reload-mcp | 重载 MCP 配置 | 无 |
| reload-skills | 重载技能 | 无 |
| whoami | 查看权限 | 无 |
| profile | 查看配置文件 | 无 |
| commands | 浏览所有命令 | [page] |
| help | 显示帮助 | 无 |
| usage | Token 用量 | 无 |
| insights | 用量分析 | [days] |
| platform | 平台状态管理 | pause/resume/list [name] |
| update | 更新 Hermes | 无 |
| version | 显示版本 | 无 |
| debug | 上传调试报告 | 无 |

---

## 七、说明

1. **共 51 条指令**可在 QQ 网关中使用
2. 指令名称**不含** `/` 前缀（QQ 后台自动处理）
3. `<参数>` 为必填，`[参数]` 为可选
4. 别名（如 `bg`、`q`、`v`）在 QQ 中同样可用，但**不需要**在后台单独配置
5. `verbose` 指令需要先在 `config.yaml` 中设置 `display.tool_progress_command: true` 才会生效
6. 按钮交互（审批、更新确认）由系统自动处理，无需手动配置

> **参考**: 指令定义来源 -> `hermes_cli/commands.py` 中的 `COMMAND_REGISTRY`
