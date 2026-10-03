# ADR-001: Engineering Runtime Strategy

- **Status:** Accepted
- **Date:** 2026-08-15
- **Decision scope:** V1 Engineering Agent runtime composition
- **Research basis:** `docs/research/hermes-runtime-reconnaissance.md`

## 1. Decision question

在不重写 Hermes Runtime、不过早实现 V2/V3 能力的前提下，V1 应把 Engineering lifecycle、verification、review 和 completion control 放在哪里？本文比较：

- **A. Modify Hermes Agent Loop directly**
- **B. Subclass / wrap Hermes Agent**
- **C. Outer EngineeringOrchestrator + Hermes Runtime**
- **D. Plugin-only implementation**

本文先按源码事实评估四种方案，再给出推荐；不把任何方案预设为正确答案。

## 2. Source-grounded constraints

### 2.1 Hermes 已有共享 agent runtime

Hermes 的实际执行主干是 `AIAgent.run_conversation()` 调用模块化 conversation loop；provider 返回 tool calls 时循环执行现有 tool runtime，无 tool calls 时才产生 candidate final response，随后经过 stop guards 和 `finalize_turn()`。

- **Status:** CONFIRMED
- **Evidence:** `run_agent.py:8017-8194`; `agent/conversation_loop.py:1494-1709,6480-7165,7580-7911`; `agent/turn_finalizer.py:70-798`
- **Symbol:** `AIAgent.run_conversation()`, `agent.conversation_loop.run_conversation()`, `finalize_turn()`
- **Implication:** 四个方案都必须证明为什么需要改变或包裹这一主干；重新实现 provider/tool loop 没有事实依据。

### 2.2 Hermes completion 不等于 engineering completion

`finalize_turn()` 的 `completed` 表示一次 agent turn 正常返回，不读取测试证据、review verdict 或工程 policy。现有 `verify_on_stop` 和 `pre_verify` 能 bounded re-entry，但不是 fail-closed completion veto；预算耗尽时还可能恢复 gate 前的 candidate。

- **Status:** CONFIRMED
- **Evidence:** `agent/turn_finalizer.py:95-143,194-203,655-690`; `agent/conversation_loop.py:7602-7721`; `agent/verification_stop.py:95-134,233-313`; `agent/verify_hooks.py:21-42`
- **Symbol:** `finalize_turn()`, `verify_on_stop_enabled()`, `get_pre_verify_continue_message()`
- **Implication:** V1 必须定义独立的 engineering workflow terminal state；不能把 `result.completed` 或模型 final text 直接当作工程完成。

### 2.3 Plugin 能执行 policy，但不能提供通用 hard completion gate

`pre_tool_call` 可以 block 或进入统一 approval，middleware 可以包裹 tool/LLM 执行，`transform_tool_result` 和 `transform_llm_output` 可以改写结果；但 `post_tool_call`、`post_llm_call` 和 session/subagent hooks 的返回值不形成 completion veto。`pre_verify` 只能按现有 bounded continuation contract 工作。

- **Status:** CONFIRMED
- **Evidence:** `hermes_cli/plugins.py:156-244,3109-3381,4910-4953,5806-6055`; `model_tools.py:1116-1167,1347-1530`; `agent/turn_finalizer.py:551-596,775-793`
- **Symbol:** `PluginContext.register_hook()`, `resolve_pre_tool_block()`, `PluginManager.invoke_hook()`, `finalize_turn()`
- **Implication:** Plugin 适合 enforcement、registration 和 runtime bridge，但 plugin-only 不能独立保证严格 engineering completion。

### 2.4 Provider/tool/session/context 已有可复用扩展面

Provider 已由 `ProviderProfile`、resolver 和 transport 隔离；tool 已由 registry、approval、terminal/file environments 实现；context、memory、session 和 skills 也已有正式或相对明确的 contract。

- **Status:** CONFIRMED
- **Evidence:** `providers/base.py:1-9,38-148`; `providers/__init__.py:45-323`; `agent/transports/base.py:1-89`; `tools/registry.py:737-1142`; `hermes_cli/plugins.py:1388-3381`; `agent/context_engine.py:89-260`; `agent/memory_provider.py:104-251`; `hermes_state.py:2691`
- **Symbol:** `ProviderProfile`, `ProviderTransport`, `ToolRegistry`, `PluginContext`, `ContextEngine`, `MemoryProvider`, `SessionDB`
- **Implication:** Engineering architecture 应消费这些 runtime，不应复制 provider client、tool dispatch、terminal backend、session 或 memory。

### 2.5 Codex app-server 是正常 hook 路径的旁路

`codex_app_server` 把整 turn 交给外部 subprocess，其 terminal/file/patch/sandbox 不经过 Hermes 正常 tool dispatch，并在正常 loop finalizer 之前返回专用结果。

- **Status:** CONFIRMED
- **Evidence:** `hermes_cli/runtime_provider.py:385-395,419-442`; `agent/conversation_loop.py:1695-1707`; `agent/codex_runtime.py:677-757,795-934`
- **Symbol:** `_maybe_apply_codex_app_server_runtime()`, `run_codex_app_server_turn()`, `CodexAppServerSession`
- **Implication:** 仅依赖 Hermes plugin hooks 的方案无法覆盖该路径；V1 必须明确限制或隔离它。

## 3. Evaluation method

评分采用五级：

- **5 — Strong:** 该方案天然满足，且不依赖脆弱内部实现。
- **4 — Good:** 可满足，但需要薄 adapter、明确边界或少量配套机制。
- **3 — Mixed:** 可以实现，但存在明显限制或耦合。
- **2 — Weak:** 只能部分满足，或需要绕过既有 contract。
- **1 — Poor:** 与源码事实冲突，或需要重写/长期绑定内部实现。

分数是基于当前源码的架构比较，不是精确成本估算。

## 4. Option comparison

| Evaluation dimension | A. Modify loop | B. Subclass/wrap agent | C. Outer orchestrator | D. Plugin-only |
|---|---:|---:|---:|---:|
| Engineering lifecycle control | 5 | 3 | 5 | 2 |
| Verification gating | 5 | 3 | 5 | 2 |
| Review gating | 5 | 3 | 5 | 2 |
| Model independence | 4 | 4 | 5 | 4 |
| Reuse of Hermes runtime | 4 | 5 | 5 | 5 |
| Upstream merge difficulty | 1 | 3 | 5 | 5 |
| Testability | 3 | 3 | 5 | 3 |
| Maintainability | 2 | 3 | 5 | 3 |
| Delegation to Codex/Claude Code/OpenCode | 3 | 3 | 4 | 3 |
| Future multi-agent support | 3 | 3 | 5 | 3 |
| **Total / 50** | **35** | **33** | **49** | **32** |

分数不是决策的唯一理由。关键淘汰条件是：V1 能否在不把 Hermes turn completion 错当 engineering completion 的情况下，确定性拥有 workflow terminal transition。

## 5. Option A — Modify Hermes Agent Loop directly

### Strengths

- 可以在 candidate final response 与 `finalize_turn()` 之间放置真正的 completion decision。
- 可以让 verification/review 直接 `continue` 同一个 tool loop。
- 如果 normal loop 和 Codex app-server 两条路径都修改，理论上能覆盖所有 surface。

### Weaknesses

- 必须耦合 `agent.conversation_loop`、`agent.turn_finalizer` 和可能的 `agent.codex_runtime`；reconnaissance 已把这些判定为内部实现。
- completion、provider fallback、streaming、budget、compression、session persistence 和 plugin lifecycle 在该区域交错，回归面大。
- 上游 Hermes 持续拆分大文件和调整内部 forwarder，Controlled Fork 的合并冲突会集中在最热代码区。
- 若把 Engineering-specific state 写进 loop，会扩大 Hermes narrow waist，并使非工程 surface 支付复杂度。

- **Status:** INFERENCE，基于已确认内部调用路径
- **Evidence:** `agent/conversation_loop.py:1494-7911`; `agent/turn_finalizer.py:70-798`; `agent/codex_runtime.py:677-934`; `run_agent.py:7223-7226,8017-8194`
- **Symbol:** `run_conversation()`, `finalize_turn()`, `run_codex_app_server_turn()`
- **Implication:** A 的控制力最高，但 V1 的 upstream merge difficulty 和维护风险不可接受；只有通用且无法从外部表达的最小 seam 才可能成为后续 core patch。

## 6. Option B — Subclass / wrap Hermes Agent

本方案包含两种不同做法，不能混为一谈：

1. **Subclass `AIAgent`:** override `run_conversation()` 或内部 helper。
2. **Thin wrapper around `AIAgent`:** 在每次调用前后增加逻辑。

### Strengths

- 继续使用 Hermes provider、tool、context、memory 和 session runtime。
- wrapper 可以检查完整 result dict，而不是只看 final string。
- 对单次 turn 的前后处理比直接修改 loop 更容易隔离。

### Weaknesses

- subclass 无法通过公开 override point 插入 candidate completion 与 finalizer 之间；要做到 hard veto，最终仍会 override 私有/内部路径。
- 单纯 wrapper 只能在 `run_conversation()` 返回后拒绝结果并发起下一 turn；若没有独立 workflow state，它只是把 orchestration 隐藏进 agent facade。
- `AIAgent` 构造和 result surface 很宽，直接继承会放大上游变更影响。
- agent-centric wrapper 不自然拥有 project inspection、multi-step workflow、review actor、外部 worker 和跨 turn evidence lineage。

- **Status:** CONFIRMED（缺少公开 in-loop completion override）+ INFERENCE（维护性判断）
- **Evidence:** `run_agent.py:412,8017-8194`; `agent/conversation_loop.py:7580-7911`; `agent/turn_finalizer.py:70-798`
- **Symbol:** `AIAgent`, `AIAgent.run_conversation()`, `finalize_turn()`
- **Implication:** 应保留一个 composition-based `HermesRuntimeAdapter`，但不把 `AIAgent` subclass 本身作为 Engineering architecture。

## 7. Option C — Outer EngineeringOrchestrator + Hermes Runtime

### Strengths

- Orchestrator 可以拥有独立 workflow state machine，并把 Hermes 的 `completed` 解释为“本次 turn 返回”，而非工程完成。
- VerificationEngine 和 ReviewGate 可以在 Hermes turn 返回后读取结构化 evidence，决定 `PASS`、`NEEDS_WORK`、`BLOCKED`，只有 gate 通过才进入 workflow `COMPLETED`。
- gate 失败时，orchestrator 可以保持 workflow 非终态，并用同一 session/AIAgent 发起后续 turn；即使重入次数耗尽，也不会把未验收任务标记为完成。
- Provider/model 仍由 Hermes resolver/transport 选择，orchestrator 只依赖自己的 runtime port，因此天然 model-independent。
- ProjectInspector、workflow state、evidence store、review 和 delegated-worker coordination 可以独立单元测试，不需要启动完整 CLI。
- 外部 coding agents 与 Hermes-native subagents 可以被视为不同 execution adapters，而不是误当 provider。

### Weaknesses and constraints

- Orchestrator 不能假装拥有 Hermes loop 内部状态；它只控制 engineering workflow 的外层 terminal transition。
- Tool-level enforcement 仍应通过 plugin `pre_tool_call`/approval/middleware 进入现有 choke point，不能由 orchestrator 旁路执行 handler。
- `codex_app_server` 不受 V1 Engineering Surface 支持，也不允许作为 V1 delegated runtime；其集成仅作为未来可能性保留。
- 同一 AIAgent/session 的重入必须遵守 prompt caching、role alternation、compression 和 session lifecycle，不能直接篡改 past messages。

- **Status:** INFERENCE，所有依赖能力均有已确认源码基础
- **Evidence:** `run_agent.py:8017-8194`; `agent/conversation_loop.py:7134-7795`; `hermes_cli/plugins.py:1388-3381,5806-6055`; `tools/registry.py:1018-1142`; `agent/system_prompt.py:265-281,689-715`; `hermes_state.py:2691,4237-4240,8856-8922`
- **Symbol:** `AIAgent.run_conversation()`, `PluginContext`, `resolve_pre_tool_block()`, `ToolRegistry.dispatch()`, `SessionDB`
- **Implication:** C 在不修改主 loop 的情况下同时取得生命周期控制、runtime 复用和低 upstream coupling，是 V1 最均衡的结构。

## 8. Option D — Plugin-only implementation

### Strengths

- 最符合 Hermes 的 edge-extension 方向，上游合并成本最低。
- `pre_tool_call`、approval 和 middleware 足以实现大量 database/payment/production tool policy。
- plugin 可注册 tools、skills、commands、prompt sections 和 context，并由 ownership ledger 卸载。

### Weaknesses

- 当前没有通用 `before_complete` 或 completion veto hook。
- `pre_verify` 是 bounded continuation，且预算耗尽时 pending candidate 可被恢复，不能构成严格 fail-closed gate。
- `transform_llm_output` 只能改变输出，`post_llm_call` 返回值被忽略；两者都不能代替 review verdict。
- plugin session hooks 不拥有 surface 的真实 workflow lifecycle；`on_session_end` 实际是 per-turn observer。
- Codex app-server 的 tool execution 不经过正常 Hermes plugin tool hooks。

- **Status:** CONFIRMED
- **Evidence:** `hermes_cli/plugins.py:156-244,4910-4953,5806-6055`; `agent/conversation_loop.py:7602-7721`; `agent/turn_finalizer.py:95-143,551-596,775-793`; `agent/codex_runtime.py:677-757`
- **Symbol:** `VALID_HOOKS`, `get_pre_verify_continue_message()`, `finalize_turn()`, `run_codex_app_server_turn()`
- **Implication:** D 是必要的集成层，但不是完整 Engineering runtime strategy；把全部生命周期责任放进 plugin 会把 observer/continuation 误当 hard gate。

## 9. Capability ownership

### 9.1 KEEP UPSTREAM

以下能力继续由 Hermes 提供，本 fork 不重新实现：

| Capability | Upstream owner / evidence | V1 usage |
|---|---|---|
| Agent/tool calling loop | `AIAgent.run_conversation()`; `agent.conversation_loop.run_conversation()` | 通过 runtime adapter 调用 |
| Provider resolution and model selection | `ProviderProfile`; `resolve_runtime_provider()`; transports | Engineering 层只接收 resolved runtime |
| API modes and provider SDK adapters | `agent/transports/*`; `agent/chat_completion_helpers.py` | 不直接调用 vendor SDK |
| Tool registry/schema/dispatch | `ToolRegistry`; `handle_function_call()` | 复用 schema、middleware 和 dispatch |
| Permission and approval | `resolve_pre_tool_block()`; `tools/approval.py` | policy 在既有 choke point 叠加 |
| Terminal/filesystem environments | `terminal_tool()`; `BaseEnvironment`; `tools/file_tools.py` | verification 记录 backend/cwd，不另建 shell runtime |
| Session and conversation persistence | `SessionDB`; surface session stores | 保存对话；不承担 engineering workflow truth |
| Memory and context compression | `MemoryProvider`; `ContextEngine`; `ContextCompressor` | recall/compression 保持 upstream 语义 |
| Skills and plugin infrastructure | `PluginContext`; skill loader | 注册 Engineering edge capabilities |
| Hermes-native delegation | `tools/delegate_tool.py` | 可作为 V1 内建 delegation 路径 |

- **Status:** CONFIRMED
- **Evidence:** `run_agent.py:8017-8194`; `providers/base.py:38-148`; `hermes_cli/runtime_provider.py:1665-2292`; `tools/registry.py:737-1142`; `model_tools.py:1170-1585`; `tools/terminal_tool.py:1571-2132,2533-2561`; `hermes_state.py:2691`; `tools/delegate_tool.py:1471-1893`
- **Implication:** 这些模块是代码地基；Engineering V1 只能适配和消费，不能平行重建。

### 9.2 PLUGIN

以下能力通过 Hermes plugin surface 接入：

- Engineering tools、skills 和 CLI command 的注册。
- `pre_tool_call` 的 deterministic block/escalate/approval bridge。
- tool/LLM middleware 中与 runtime request 相关、但不拥有 workflow truth 的横切行为。
- cache-safe 的 Engineering context/prompt section 注入。
- tool result normalization、事件转发和 observability bridge。
- 将 Hermes runtime events 关联到 engineering workflow run id 的薄 adapter。

Plugin 不保存唯一 completion truth，也不单独决定 workflow terminal state。

- **Status:** CONFIRMED（注册与 enforcement 能力）
- **Evidence:** `hermes_cli/plugins.py:1388-3381,4910-4953,5806-6004`; `model_tools.py:1347-1530`
- **Symbol:** `PluginContext.register_tool()`, `register_hook()`, `register_middleware()`, `register_skill()`, `resolve_pre_tool_block()`
- **Implication:** Plugin 是 Engineering module 与 Hermes tool/context surface 的桥，不是 orchestrator 的替代品。

### 9.3 NEW ENGINEERING MODULE

以下能力不属于 Hermes 通用 runtime，应进入独立 Engineering module：

- `EngineeringOrchestrator` 和显式 workflow state machine。
- 独立的 workflow run id、step/attempt lineage 和 terminal-state predicate。
- `ProjectInspector` 及其结构化 project snapshot。
- `VerificationEngine`、verification plan、command/input/output/status/evidence schema。
- `ReviewGate` 及 machine-readable verdict。
- `EngineeringPolicy` 的领域规则；plugin 只负责把 decision 接到 enforcement point。
- Engineering Context 的结构化事实层。
- Engineering Skills 与 Workflows 的定义和版本边界。
- Database Safety、Payment Engineering、Production Guardrails 的领域 policy/evidence 规则。
- Software Engineering Evals。
- delegated execution 的 workflow coordination，以及在实际 V1 scope 要求时才加入的 external worker adapter。

- **Status:** INFERENCE，由 Hermes completion GAP 和现有公开扩展面推导
- **Evidence:** `agent/turn_finalizer.py:194-203`; `agent/conversation_loop.py:7602-7721`; `hermes_cli/plugins.py:1388-3381`; `tools/delegate_tool.py:1471-1893`; `skills/autonomous-ai-agents/codex/SKILL.md`; `skills/autonomous-ai-agents/claude-code/SKILL.md`; `skills/autonomous-ai-agents/opencode/SKILL.md`
- **Symbol:** `finalize_turn()`, `PluginContext`, `delegate_task`
- **Implication:** 这些能力拥有独立产品语义和测试矩阵，放进 core 或 plugin lifecycle 都会混淆 Hermes turn 与 engineering workflow。

### 9.4 CORE PATCH

**V1 默认：不需要 Hermes Core patch。** Outer orchestrator 拥有 workflow terminal transition，plugin bridge 提供 tool-level enforcement，因此 V1 不必修改 conversation loop。

仅当后续源码验证证明外层控制无法满足明确需求时，才考虑一个通用、非 Engineering-specific 的最小 patch：

- 在 candidate response 被接受前提供 typed completion decision contract，例如 `accept | continue | reject`。
- contract 必须覆盖 normal loop 和 `codex_app_server` path。
- decision 必须有明确的 budget-exhaustion/fail-closed 语义。
- patch 不包含 VerificationEngine、ReviewGate、payment/database rules 或 workflow state。
- patch 必须由 upstream-style behavior tests 证明，不以 prompt 约定代替。

- **Status:** GAP（当前不存在该通用 contract）；V1 decision 是不添加 patch
- **Evidence:** `agent/conversation_loop.py:7580-7795`; `agent/turn_finalizer.py:95-143,194-203`; `agent/codex_runtime.py:677-934`
- **Symbol:** 当前无 `before_complete` / generic completion veto symbol
- **Implication:** CORE PATCH 是有证据触发的最后手段，不是 V1 的预建扩展点。

## 10. V1 recommended architecture

### 10.1 Decision

V1 选择 **C. Outer EngineeringOrchestrator + Hermes Runtime**，同时使用 **D 的 plugin 能力作为 integration layer**。不 subclass `AIAgent`，不把 Engineering-specific lifecycle 写入 Hermes loop，V1 不做 core patch。

该决定不是因为“外层编排”抽象上更流行，而是因为源码同时证明：

1. Hermes 已有完整且可复用的 provider/tool/session/context runtime；
2. Hermes turn completion 没有严格 engineering gate；
3. plugin 有真实 tool enforcement，却没有通用 hard completion veto；
4. outer owner 可以在不改变 Hermes turn 语义的情况下独立控制 workflow terminal state。

### 10.2 Component boundary

```text
Engineering Entry Point
        |
        v
EngineeringOrchestrator ---------------- Engineering Workflow Store
        |                                      |
        |                                      +-- workflow/step/attempt state
        |                                      +-- evidence + review verdict
        |
        +-- ProjectInspector
        +-- VerificationEngine
        +-- ReviewGate
        +-- EngineeringPolicy
        +-- Delegation coordination
        |
        v
HermesRuntimeAdapter
        |
        +-- AIAgent.run_conversation()
        +-- resolved provider/model/api_mode
        +-- SessionDB / ContextEngine / MemoryProvider
        |
        v
Engineering Plugin Bridge
        |
        +-- pre_tool_call / approval / middleware
        +-- tools / skills / commands / context
        v
Hermes Tool Runtime -> terminal / file / delegate_task
```

`HermesRuntimeAdapter` 使用 composition，不继承 `AIAgent`。它只稳定以下边界：turn request、turn result、session/task correlation、interrupt/cancel 和 runtime error mapping；不暴露 conversation-loop 私有状态。

### 10.3 Completion sequence

1. Orchestrator 创建独立 `workflow_run_id`，记录目标、scope 和初始 project snapshot。
2. Hermes Runtime 执行一个或多个 agent turns；每个 Hermes `completed` 只标记该 turn 已返回。
3. Candidate result 产生后，orchestrator 收集 changed paths、tool evidence、backend、cwd 和命令结果。
4. VerificationEngine 按实际变更风险执行确定性 checks，并生成结构化 verdict。
5. ReviewGate 消费目标、diff、verification evidence 和 policy result，生成 `PASS | NEEDS_WORK | BLOCKED`。
6. `NEEDS_WORK` 时 workflow 保持非终态；orchestrator 将结构化缺口转换成下一 turn 输入并重入 Hermes Runtime。
7. 达到 attempts/budget 上限时 workflow 进入显式 `BLOCKED` 或 `FAILED`，不恢复为 `COMPLETED`。
8. 只有 completion predicate、verification 和 review 全部通过，orchestrator 才写入 workflow `COMPLETED` 并发布最终回答。

### 10.4 Model and delegation policy

- 主模型继续由 Hermes provider resolver 选择；Engineering module 不绑定 OpenAI、Anthropic、Gemini 或 OpenAI-compatible SDK。
- Codex 作为 `openai-codex` provider 时仍运行 Hermes normal loop，可以进入 V1 主路径。
- `codex_app_server` 不受 V1 Engineering Surface 支持。
- V1 不允许把 `codex_app_server` 作为 delegated runtime。
- delegated `codex_app_server` integration 仅作为延期的未来可能性保留，待 policy/evidence contract 明确后重新决策。
- Hermes-native `delegate_task` 是 V1 可依赖的 subagent 能力。
- Codex CLI、Claude Code CLI、OpenCode CLI 当前是 skill + subprocess，不视为 typed Hermes subagent。V1 只有在明确产品需求存在时才为实际采用的 worker 增加 adapter；不预建无消费者的统一框架。
- 无论执行者是谁，只有 outer orchestrator 能改变 engineering workflow terminal state。

### 10.5 V1 scope boundaries

- V1 只保证通过 Engineering entry point 创建的 workflow；不宣称改造所有普通 Hermes chat surfaces。
- V1 不改变 Hermes `result.completed` 的含义。
- V1 不修改 provider、tool registry、terminal backend、SessionDB 或 memory contract。
- V1 不要求同时支持所有外部 coding-agent CLI。
- V1 不把 LLM review 当作唯一确定性 verification。
- V1 不提前添加 generic core hook；只有真实消费者和失败证据出现后再评估。

## 11. Consequences

### Positive

- 最大化复用 Hermes Runtime，同时把 Engineering completion 从聊天完成中分离。
- tool policy 继续经过 Hermes approval/dispatch，避免安全旁路。
- workflow、verification 和 review 可以使用 deterministic unit/contract tests。
- provider/model 更换不影响 Engineering lifecycle。
- 上游合并主要发生在 adapter/plugin 边界，而不是 conversation loop 热点。
- 未来 multi-agent 可以围绕 workflow steps 和 evidence contract 扩展，而不改主 agent loop。

### Negative

- 系统存在 conversation session 与 engineering workflow 两套关联但不同的 lifecycle，需要明确 lineage。
- outer re-entry 是新 turn，而不是在 candidate response 内部无缝 veto；prompt/cache/session contract 必须受测试保护。
- plugin bridge 与 orchestrator 必须共享稳定 correlation id，否则 policy/evidence 无法可靠归属。
- `codex_app_server` 在 V1 Engineering Surface 中保持禁用；未来 delegated integration 仍需先解决 Hermes tool hooks 的旁路问题。

### Risks and mitigations

| Risk | Mitigation |
|---|---|
| Adapter 随 `AIAgent` 宽接口漂移 | 只封装最小 request/result protocol，并用 contract tests 固定 |
| Verification 被模型文本伪造 | evidence 由 VerificationEngine/执行端生成，不解析自述作为通过依据 |
| Plugin hook 触发语义被误读 | 对每个使用的 hook 固定 payload、返回消费与触发次数测试 |
| Session compression 丢失工程事实 | workflow/evidence 独立持久化，不只写 transcript/memory |
| 外部 worker 绕过 Hermes policy | 作为独立 runtime adapter，执行前后由 outer policy/evidence gate 包围 |
| 重入耗尽仍错误完成 | attempts/budget exhaustion 映射为 `BLOCKED`/`FAILED`，不映射 `COMPLETED` |

## 12. Reconsideration triggers

以下任一事实出现时重新评估本 ADR：

1. V1 被要求覆盖所有 Hermes surfaces，而不仅是 Engineering entry point。
2. 外层重入无法在保持 prompt cache、role alternation 或 session persistence 的前提下工作。
3. upstream 提供稳定、typed、fail-closed 的 completion decision API。
4. `codex_app_server` 必须成为 V1 主 runtime，且无法通过 outer adapter 获得足够 policy/evidence。
5. 实际 contract tests 证明 plugin enforcement 或 `AIAgent.run_conversation()` adapter 无法满足 V1。

触发重新评估不自动意味着选择 A；仍应按 existing capability → plugin/hook → new engineering module → minimal core patch 的顺序重新取证。

## 13. Decision summary

| Area | V1 decision |
|---|---|
| Runtime strategy | **C: Outer EngineeringOrchestrator + Hermes Runtime** |
| Integration | 使用 plugin bridge，但拒绝 plugin-only completion |
| Agent inheritance | 不 subclass；使用 composition adapter |
| Engineering lifecycle truth | 独立 Engineering workflow store/state machine |
| Verification/review terminal control | Outer orchestrator owns it |
| Hermes core changes | V1 none |
| Provider/model | KEEP UPSTREAM；由 Hermes resolution/transport 管理 |
| Tool/terminal/session/context | KEEP UPSTREAM |
| Engineering policy/tools/context bridge | PLUGIN |
| Orchestrator/verification/review/evidence/workflows/evals | NEW ENGINEERING MODULE |
| Codex/Claude Code/OpenCode | 区分 provider、Hermes subagent、app-server、skill+subprocess；V1 Engineering Surface 不支持 `codex_app_server`，其 delegated integration 延期 |
| Future core seam | 仅在真实证据触发后考虑 generic typed completion decision |
