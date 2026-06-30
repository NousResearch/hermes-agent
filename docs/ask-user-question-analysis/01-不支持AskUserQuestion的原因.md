# 当前接入方式不支持 AskUserQuestion 的原因分析

> **面向**：技术 + 产品
> **结论先行**：公司「通用层」通过 hermes 的 `api_server`（OpenAI 兼容 Chat Completions）方式接入，在这条链路上 **hermes 本身就没有提供 AskUserQuestion（内部名 `clarify`）能力**。因此通用层和业务层都拿不到。**这不是通用层的实现疏漏，而是底层接入方式的能力边界。**

---

## 1. 名词对齐

| 术语 | 含义 |
|------|------|
| **AskUserQuestion** | 产品需求名。指 agent 在执行中「停下来向用户提问（单选/多选/填空）」的交互卡片 |
| **`clarify`** | hermes 内部对应的工具名。AskUserQuestion 在 hermes 里就叫 `clarify` |
| **`api_server`** | hermes 作为服务端对外提供的 OpenAI 兼容 HTTP 接口层（`/v1/chat/completions`、`/v1/responses`、`/v1/runs`） |
| **通用层** | 公司基于 `api_server` 再包一层的内部 SDK / 网关，业务应用通过它调用 hermes |
| **SSE 单向通道** | Server-Sent Events，服务端向客户端的单向流式推送。当前接入用它接收 agent 的实时输出 |

---

## 2. 接入链路

```
ins-liexiaoxia-platform（业务应用层，我们的代码）
        │  调用
        ▼
公司「通用层」（HTTP 客户端封装，无 hermes 仓库权限）
        │  OpenAI 兼容 Chat Completions（+ SSE 流式）
        ▼
hermes api_server（hermes-api-server 工具集）   ← clarify 在这一层就被裁掉
        │
        ▼
上游 LLM Provider（OpenAI / 公司内部模型 / …）
```

需求落地的前提是：**链路上每一层都具备该能力**。而最底下的 `api_server` 这一层就已经不具备，上面的通用层和业务层自然无米下锅。

---

## 3. 证据链（5 条，均来自 hermes 源码）

### 证据 ① 工具集白名单明确排除 `clarify`

`toolsets.py:396-397`，`hermes-api-server` 这个工具集的描述直接写死：

> "OpenAI-compatible API server — full agent tools accessible via HTTP **(no interactive UI tools like clarify or send_message)**"

对比同时存在的交互式工具集（CLI / Telegram / Discord / WhatsApp / Slack / Signal / Cron 共享的 `_HERMES_CORE_TOOLS`，`toolsets.py:31`），它们的工具列表里**包含** `clarify`（`toolsets.py:33`）。

**结论**：在 `api_server` 接入模式下，模型根本看不到 `clarify` 这个工具，无从调用。

### 证据 ② api_server 创建 agent 时不注入 `clarify` 回调

`gateway/platforms/api_server.py:1120` 构造 `AIAgent(...)` 时，只传了 `platform="api_server"` 和几个流式/工具回调，**没有传 `clarify_callback`**。而 `run_agent.py:417` 这个参数的默认值是 `None`，最终写入 `agent.clarify_callback = None`（`agent/agent_init.py:435`）。

全代码库里 `clarify_callback` 的设置点只有三处，全部是「有人的」交互场景：
- CLI 模式：`hermes_cli/cli_agent_setup_mixin.py:374`
- 消息平台模式：`gateway/run.py:16698`（`agent.clarify_callback = _clarify_callback_sync`）
- 无人值守模式：`hermes_cli/oneshot.py:363`（返回哨兵，让 agent 自行决定）

**`api_server` 不在其中。**

### 证据 ③ 即便强行调用，也是返回错误

`agent/agent_init.py:268` 的文档注释：

> "If None, the clarify tool returns an error."

对应 `tools/clarify_tool.py:95-99`：

```python
if callback is None:
    return json.dumps(
        {"error": "Clarify tool is not available in this execution context."},
        ensure_ascii=False,
    )
```

**结论**：退一万步，即使把 `clarify` 工具强行加回 `api_server` 的工具列表，模型调用后拿到的也是一句报错，无法真正提问。

### 证据 ④ 同层 `approval` 有完整异步通道，`clarify` 一根线都没接

`api_server` 的 `/v1/runs` 接口为「危险命令审批」接了完整闭环：

- SSE 事件：`approval.request`（`api_server.py:4041`）
- 回填端点：`POST /v1/runs/{run_id}/approval`（`api_server.py:1277`、`20`）
- notify 接线：`register_gateway_notify`（`api_server.py:4076`）

而 `clarify` **没有任何对应的端点、事件、notify**。全代码库 grep 不到 `clarify.request`、`/clarify` 端点、api_server 侧的 clarify notify。

**这是一个关键的不对称**——见第 5 节。

### 证据 ⑤ Chat Completions 是单轮请求-响应，通道在 turn 结束即拆

`gateway/platforms/api_server.py:762-766` 的注释：

> `/v1/chat/completions`、`/v1/responses`、`/v1/runs` 这些通道 "**tears down its channel when the turn ends**"（turn 一结束就拆除通道）。

Chat Completions 的语义是「客户端发请求 → 服务端跑完 → 一次性返回响应」。**没有「中途暂停、反向问客户端、拿到答案后继续」的语义通道。**

---

## 4. 技术根因：为什么这条链路天然难支持

`clarify`（AskUserQuestion）的本质是一次**双向交互**：

```
agent 执行中  ──暂停──►  把问题推给用户
                            │
   ◄──唤醒──  用户选了「选项B」
agent 继续
```

它依赖一个「**能阻塞 agent 线程、等用户回话、再唤醒继续**」的双向通道。hermes 为此专门建了一套线程安全原语（`tools/clarify_gateway.py`：`Event` + `register/wait_for_response/resolve_gateway_clarify`）。

而当前接入方式的两个特征决定了它天然不适配：

1. **请求-响应模型**：一次请求对应一次响应，agent 必须在「这一次响应」里跑完。中途没法「暂停去问用户」。
2. **SSE 是单向的**：服务端能向客户端推流（输出 token、工具进度），但**客户端无法在同一个流里「回答」agent 的提问**。要回答，得另起一个 HTTP 请求——而那时上一轮 agent 已经结束、通道已拆。

> 给产品的一句话：**AskUserQuestion 需要「对话中途回合」，而当前接入是「一问一答、答完即止」，两者模型对不上。**

---

## 5. 关键反例：`approval` 能做到，`clarify` 为什么没做？

这是最容易引起误解、也最需要向技术澄清的一点。

同一个 `api_server` 里，**`approval`（危险命令审批）实现了和 `clarify` 完全相同的「暂停-等待-继续」语义**，而且跑通了。它的做法是：

1. agent 要执行危险命令前，**不阻塞 HTTP 响应线程**，而是把「待审批」状态挂起；
2. 通过 SSE 向客户端推一个 `approval.request` 事件；
3. 客户端收到后渲染审批 UI，用户决策后，**另起一个** `POST /v1/runs/{run_id}/approval` 请求把结果回填；
4. agent 线程被唤醒，继续执行。

**这套机制 `clarify` 完全可以照搬**（机制上 100% 同构）。所以：

- **技术上完全可做**——并非 `api_server` 的能力禁区；
- hermes 当前**只是没给 `clarify` 接这根线**，属于实现优先级 / 未排期的选择，不是架构上做不到。

> 给技术的判断：**这条需求在工程上可行，前提是推动 hermes 侧（或 fork）按 approval 的范式给 clarify 补齐「SSE 事件 + POST 回填端点 + notify 接线 + 工具集加回 clarify + 注入 clarify_callback」五件套。** 详见文档 03 的落地路径。

---

## 6. 对业务的影响

| 角度 | 影响 |
|------|------|
| **需求可行性** | 当前链路下无法实现真正意义上的 AskUserQuestion（agent 自主中途提问）。 |
| **通用层责任** | 不是通用层的锅。通用层面对的 hermes 端点既不暴露 clarify 工具，也没有 clarify 的事件/回填端点可订阅。 |
| **用户体验差异** | 在 CLI / 消息平台接入下，agent 遇到歧义会主动澄清；在当前接入下，agent 只能「自行假设并继续」，或把疑问写进最终回复里（无结构化卡片）。 |
| **风险** | 若不澄清就继续，agent 可能基于错误假设执行（如在「部署到哪个环境」上猜错）。当前靠 system prompt 约束可缓解，但无工具级可靠性。 |

---

## 7. 可选出路（概要）

> 详细对比与决策建议见 [文档 03](./03-接入方式功能差异对比.md)。这里仅罗列方向。

| 路径 | 改造点 | 可靠性 | 代价 |
|------|--------|--------|------|
| **A. 推动 hermes 侧补齐 clarify（正道）** | hermes 仿 approval 接 5 件套；通用层订阅事件、渲染卡片、回填 | 高（工具级） | 需 hermes 仓库权限 / 推动排期 |
| **B. 通用层自建「模拟澄清」协议** | system prompt 约束模型用结构化文本表达疑问，通用层在流里拦截、渲染卡片、下一轮带入答案 | 中（依赖模型守约） | 通用层工作量中等；多选/超时语义需自复刻 |
| **C. 业务层纯前端前置表单** | 业务流程先弹表单，用户选完拼进 prompt 再调 hermes | 高 | 把「agent 自主澄清」降级为「表单前置」，语义弱化 |

---

## 附：核心源码定位

| 关注点 | 位置 |
|--------|------|
| clarify 工具定义与 schema | `tools/clarify_tool.py:56, 125` |
| clarify 在工具集中的归属 | `toolsets.py:31`（core 含）、`toolsets.py:396`（api_server 裁掉） |
| api_server 创建 agent（未传 clarify_callback） | `gateway/platforms/api_server.py:1120` |
| clarify_callback=None 的兜底报错 | `tools/clarify_tool.py:95`、`agent/agent_init.py:268` |
| clarify 的双向交互原语 | `tools/clarify_gateway.py` |
| approval 的完整异步闭环（可参照） | `gateway/platforms/api_server.py:4030-4088`、`1277` |
| api_server 各端点清单 | `gateway/platforms/api_server.py:5-21` |
