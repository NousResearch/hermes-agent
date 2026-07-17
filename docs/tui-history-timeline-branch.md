# TUI 历史消息 Timeline / Branch 交互规格

## 目标与范围

目标：在 Hermes Agent TUI 的当前会话内提供一个类似 opencode 的历史 user / assistant 消息导航界面，让用户可以快速上下切换过去的提问和回答、预览选中消息的完整内容，并从选中位置开出新的会话分支继续。

范围仅覆盖 **TUI 当前会话的 user / assistant 历史消息导航与分支语义**：

- 展示当前 live transcript 中已经进入会话上下文的历史 user / assistant 消息。
- 允许从选中的 user message 或 assistant message 生成新分支。
- 分支后进入新的 TUI live session，原会话保持可恢复。
- MVP 不要求修改 gateway 平台的点选历史交互，也不要求实现文件系统 checkpoint 回滚。

## 当前代码入口映射

实现任务应优先映射到以下现有入口，而不是重新发明一套 session 机制：

- Slash 命令注册：`hermes_cli/commands.py` 已有 `CommandDef("history", ...)`、`CommandDef("branch", aliases=("fork",))`、`CommandDef("retry", ...)`、`CommandDef("undo", ...)`。
- TUI `/history` 现状：`ui-tui/src/app/slash/commands/core.ts` 当前把 `ctx.local.getHistoryItems()` 格式化后交给 `ctx.transcript.page(...)`；这是新 overlay 的最小替换入口。
- TUI `/branch` 现状：`ui-tui/src/app/slash/commands/session.ts` 调用 JSON-RPC `session.branch`，成功后切换 `sid` 并关闭旧 live session。
- TUI backend branch：`tui_gateway/server.py::session.branch` 当前复制 live `session["history"]` 到新 SessionDB 记录，并创建新的 agent/session。
- TUI pager / overlay 键位参考：`ui-tui/src/app/useInputHandlers.ts` 已为 pager 支持 `↑↓/j/k`、`PgUp/PgDn`、`g/G`、`Esc/q`；history timeline overlay 应复用同类交互模型。
- SessionDB 完整字段参考：gateway `/branch` 的 `gateway/slash_commands.py::_handle_branch_command` 复制了 `tool_calls`、`tool_call_id`、`finish_reason`、`reasoning*`、`codex_*` 等字段；TUI branch 后续实现需要补齐同等级字段，避免只复制 `role/content`。

## 入口

MVP 必须提供至少一个稳定入口：

1. `/history` 打开 History Timeline overlay。
2. 可选快捷键入口：后续可增加，例如 `Ctrl+H` 或其他不冲突键位；快捷键不是 MVP 必须项。

打开 overlay 后默认选中最近一条 user / assistant 消息。若当前会话没有历史 user / assistant 消息，显示空状态并提示先发送一轮对话。

## Timeline 展示模型

Timeline item 是当前会话中 role 为 `user` 或 `assistant` 的消息。每个 item 至少包含：

- 顺序编号：按会话顺序递增。
- role：`user` 显示为 `You` 或 `User`；`assistant` 显示为 `Hermes` 或 `Assistant`。
- 简短预览：单行或少量行截断。
- 状态标记：若 assistant message 包含 tool calls / reasoning / 空 content，应以占位摘要显示，例如 `(3 tool calls)`、`(assistant response with tool calls)`，不可显示成不可理解的空行。

Overlay 推荐左右或上下分栏：

- 左侧 / 上方：timeline 列表，用于快速移动选中项。
- 右侧 / 下方：选中消息完整预览，用于阅读全文。

MVP 不要求 branch tree 可视化；timeline 只需表达当前会话线性历史。

## 导航键位

MVP done criteria 必须覆盖：

- `↑` / `k`：选中上一条消息。
- `↓` / `j`：选中下一条消息。
- `PgUp`：向上翻页，移动多个 timeline items 或滚动完整预览；实现需保证用户能快速跳过长历史。
- `PgDn`：向下翻页。
- `Home` / `g`：可选，跳到第一条。
- `End` / `G`：可选，跳到最后一条。
- `Esc` / `q`：关闭 overlay，不改变会话。

`/` 过滤/匹配用于在当前会话 timeline 内快速定位消息：输入 `/` 进入过滤输入，按 role + message text 匹配当前会话的 timeline items；不承诺跨 session 全文搜索。

## 默认动作与 action 菜单

按 `Enter` 对选中消息执行默认动作；也可以显示显式 action 菜单。MVP 必须保证不同 role 的语义不同。

### 选中 user message

选中 user message 表示“从这次用户提问处重新走一条路”。MVP 必须支持以下动作语义：

1. **Edit & branch**
   - 复制从会话开始到该 user message 之前的上下文。
   - 以该 user message 的内容作为可编辑草稿。
   - 用户提交编辑后的 prompt 后，在新分支中继续。
   - 原会话不被删除或覆盖。

2. **Retry in branch**
   - 复制从会话开始到该 user message（包含原 user message）的 cut point。
   - 不复制该 user message 之后的 assistant/tool 结果。
   - 在新分支中重新请求 assistant 生成回答。
   - 用于“同一个问题换一种回答”，不同于当前 `/retry` 对最后一轮的就地撤回重发。

默认动作建议：`Enter` 打开 user message action 菜单；若 MVP 只做一个快捷默认，则优先 `Retry in branch`，并提供可发现的 `e` 或菜单入口进入 `Edit & branch`。

### 选中 assistant message

选中 assistant message 表示“接受这条回答作为分支前缀，然后继续往下探索”。MVP 必须支持：

1. **Branch after this answer**
   - 复制从会话开始到该 assistant answer 结束的上下文。
   - 新分支的下一步等待用户输入，不自动重新生成该 assistant answer。
   - 若该 assistant answer 后面紧跟 tool result blocks，这些 tool result 是否属于 answer 的完整 cut point 必须按 tool-call block 规则处理，见风险章节。

默认动作建议：`Enter` 直接执行 `Branch after this answer`，或先弹确认 / action 菜单。执行成功后切换到新 branch live session，并显示新 session title / parent session 简短提示。

## Branch cut point 规则

实现必须用稳定 message id / index 映射选中项到真实历史消息，不能只靠预览文本反查。

Cut point 规则：

- 对 user `Retry in branch`：保留该 user message 及其之前消息；丢弃其之后所有 assistant/tool 消息，然后触发一次新生成。
- 对 user `Edit & branch`：保留该 user message 之前消息；编辑后的 prompt 作为新分支的下一条 user message。
- 对 assistant `Branch after this answer`：保留该 assistant answer 及其之前完整上下文；新分支等待下一条 user prompt。
- 若 assistant message 含 `tool_calls`，其后匹配的 `tool` role messages 是 assistant turn 的结构完整性要求。cut point 不能截断在 assistant tool_call 与对应 tool result 之间，否则 OpenAI / Anthropic / OpenRouter 消息重放会变成非法或语义不完整。

## MVP done criteria

后续实现任务可直接按以下验收：

1. `/history` 或明确快捷入口打开 History Timeline overlay，而不是只打开普通只读 pager。
2. overlay 展示当前会话 user / assistant 历史列表，并展示选中消息完整内容预览。
3. `↑↓/j/k` 能逐条移动选中消息。
4. `PgUp/PgDn` 能翻页或快速移动，长历史可用。
5. `/` 搜索若未实现，必须标注为后续扩展；不得作为已完成能力声称。
6. 选中 user message 时支持 `Edit & branch` 和 `Retry in branch` 的产品语义。
7. 选中 assistant message 时支持 `Branch after this answer` 的产品语义。
8. branch 成功后创建新 session，并保持 parent lineage / branch 可在 `/sessions` 或 session browser 中恢复。
9. 新 branch replay 保持必要消息字段完整：`tool_calls`、`tool_call_id`、`finish_reason`、`reasoning`、`reasoning_content`、`reasoning_details`、`codex_reasoning_items`、`codex_message_items` 等不应无故丢失。
10. 对含 tool call 的 assistant turn，不允许产生 assistant tool_call 与 tool result 不匹配的截断历史。

- `type / to filter role + message text in this session only`：未过滤状态下的提示；明确仅搜索当前会话 timeline，不承诺跨 session 全文搜索。
- `filter /<query> · M/N current-session matches`：过滤状态下展示当前 query、匹配数与当前会话总数；列表/预览中匹配片段用高亮文本强调。
- `Esc`：过滤状态下先清空过滤并恢复进入过滤前的原 timeline selection；未过滤状态下关闭 overlay。
- `Enter`：过滤输入模式中结束输入并保留 filtered item list；过滤输入结束后对当前匹配项执行既有 jump。`n/N` 或 `Tab/Shift+Tab` 在过滤输入中切换下/上一个匹配项。

## 非 MVP / 暂不做

以下能力明确不属于本轮 MVP：

- 文件系统 checkpoint 自动回滚或把 branch 与 `/rollback` 自动绑定。
- 跨平台 gateway（Telegram / Discord / Slack 等）点选历史消息并 branch。
- branch tree 高级可视化、图形化 lineage diff、复杂 merge UI。
- 跨 session 全文搜索；`/history` Timeline 过滤仅覆盖当前会话 timeline 的 role + message text。
- 对旧 session 数据做批量迁移或修复历史字段。

## 关键风险与实现注意事项

1. **message id 映射**
   - Timeline selection 必须保存原始 history index / DB message id / stable client id。
   - 不得用截断 preview、role + 文本内容等非唯一字段定位 cut point。

2. **tool-call block cut point**
   - Assistant tool call 与后续 tool result 是结构耦合块。
   - Branch after assistant answer 时要么包含完整 tool block，要么拒绝在不完整位置 branch 并给出解释。
   - Retry from user message 时应安全丢弃该 user 之后的 assistant/tool tail。

3. **prompt cache / agent 重建**
   - 分支会创建新 agent / live session；必须更新 `sid`、session context、active session file、UI status。
   - 新 agent 的 history 必须与 DB replay 一致，否则 prompt cache、reasoning continuation、tool-call replay 会出现 CLI/TUI 不一致。

4. **branch 复制字段完整性**
   - 当前 TUI `session.branch` 仅复制 `role/content`，与 gateway branch 的完整字段复制不一致。
   - 实现 timeline branch 时必须补齐字段，尤其是 provider reasoning 与 codex message items，否则会降低模型连续性或破坏 Responses API replay。

5. **运行中会话并发**
   - 若当前 turn 正在运行，history branch / edit / retry 会读写 live history。应与 `/undo`、`/compress` 一样拒绝或要求先 interrupt，避免 post-run 写回覆盖分支 cut point。

6. **长消息与空消息展示**
   - Assistant 可能有 reasoning、tool calls 或 provider-specific items 但 content 为空。UI 必须给可理解占位，而不是显示空白可选项。

## 后续实现建议

推荐分两步实现：

1. 前端先将 `/history` 从只读 pager 升级为 timeline overlay：复用现有 overlay store / input handler 风格，完成列表、预览、导航、action 菜单。
2. 后端新增或扩展 branch RPC，支持按 cut point 和 action 类型创建 branch；同时补齐 TUI branch 字段复制逻辑，使其与 gateway branch replay 字段保持一致。
