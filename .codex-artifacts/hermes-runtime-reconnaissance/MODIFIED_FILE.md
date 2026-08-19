# Hermes Runtime Architecture Reconnaissance

> 调查范围：当前 Controlled Fork 工作树中的 Hermes Runtime 源码。
> 调查日期：2026-08-15。
> 本文只记录 Runtime 事实、推断与缺口，不提出或实现 Engineering Agent。

## 0. 结论摘要

Hermes 的共享运行核心不是 CLI 本身，而是 `AIAgent` 对模块化 conversation loop 的包装：入口层解析会话、provider、model 和 surface，`AIAgent.run_conversation()` 建立 task/turn 上下文，`agent.conversation_loop.run_conversation()` 负责 LLM/tool 循环，`agent.turn_finalizer.finalize_turn()` 负责结果组装、持久化和 turn-end observer。CLI、gateway、TUI、ACP 等 surface 复用这一核心，但拥有各自的 session/surface orchestration。

**Status:** CONFIRMED
**Evidence:** `hermes_cli/main.py:2592-2857`; `cli.py:14664-14739`; `run_agent.py:8017-8138`; `agent/conversation_loop.py:1494-1709`; `agent/turn_finalizer.py:70-92`
**Symbol:** `cmd_chat()`, `HermesCLI.chat()`, `AIAgent.run_conversation()`, `agent.conversation_loop.run_conversation()`, `finalize_turn()`
**Implication:** Engineering Agent 应在共享 runtime 之上编排，不应另建 agent loop；surface-specific 行为不能假设只由 CLI 决定。

Hermes 已有两种“阻止本轮立即结束并重入主循环”的机制：opt-in 的 `verify_on_stop` 和插件 `pre_verify`。两者都是有限次数、通过 synthetic user nudge 执行的 continuation gate；前者默认关闭，后者只有注册 hook 且本轮检测到编辑路径时才运行。它们不是通用、fail-closed 的 completion veto。

**Status:** CONFIRMED
**Evidence:** `agent/conversation_loop.py:7602-7721`; `agent/verification_stop.py:95-134,233-313`; `agent/verify_hooks.py:21-42`; `hermes_cli/plugins.py:6007-6055`; `agent/turn_finalizer.py:95-143`
**Symbol:** `verify_on_stop_enabled()`, `build_verify_on_stop_nudge()`, `get_pre_verify_continue_message()`, `max_verify_nudges()`, `finalize_turn()`
**Implication:** 可复用 bounded re-entry 语义，但当前能力不足以直接表达 Engineering Agent 的确定性“验证未通过则不可完成”；特别是预算耗尽时，finalizer 可以恢复 gate 前的候选答案。

Hermes 最成熟且面向第三方的扩展面是 `PluginContext`、插件 hooks/middleware、provider `ProviderProfile`、`ContextEngine`、`MemoryProvider` 和 skills。conversation loop、turn context/finalizer、直接 tool registry 操作以及 CLI/gateway 私有状态属于内部实现，不应成为新模块的硬耦合点。

**Status:** CONFIRMED（公开注册入口）+ INFERENCE（稳定性分级）
**Evidence:** `hermes_cli/plugins.py:1-31,1388-3381`; `providers/base.py:1-9,38-148`; `agent/context_engine.py:1-25,89-177`; `agent/memory_provider.py:104-251`; `agent/conversation_loop.py`; `agent/turn_finalizer.py`
**Symbol:** `PluginContext`, `ProviderProfile`, `ContextEngine`, `MemoryProvider`, `run_conversation()`, `finalize_turn()`
**Implication:** Engineering 能力优先通过公开 ABC、plugin、hook、middleware、skill 和独立 engineering module 接入；只有明确缺口才考虑最小 core seam。

---

## 1. Agent 主循环

### 1.1 入口与主对象

经典 CLI 的 `hermes chat` 路径由 `hermes_cli.main.main()` 分派到 `cmd_chat()`；`cmd_chat()` 处理 cwd、resume、surface、配置开关后调用 `cli.main()`。TUI 分支在此之前改走 `_launch_tui()`，所以 `cli.py` 不是所有 surface 的唯一入口。

**Status:** CONFIRMED
**Evidence:** `hermes_cli/main.py:2580-2594,2623-2694,2811-2857`; `hermes_cli/main.py:11583`
**Symbol:** `main()`, `cmd_chat()`, `_launch_tui()`
**Implication:** Runtime 调查和未来集成必须以共享 `AIAgent` 路径为中心，而不是把 `HermesCLI` 当成唯一宿主。

主 agent 类型是 `run_agent.py` 中的 `AIAgent`。公开的简单接口 `chat()` 调用 `run_conversation()` 并返回 `result["final_response"]`；完整接口返回包含消息、完成状态、usage、provider/model、session id 等字段的 dict。

**Status:** CONFIRMED
**Evidence:** `run_agent.py:412`; `run_agent.py:8017-8194`; `agent/turn_finalizer.py:655-690`
**Symbol:** `AIAgent`, `AIAgent.run_conversation()`, `AIAgent.chat()`
**Implication:** Engineering orchestration 可把完整 result dict 作为事实输入，不能只依赖最终字符串。

### 1.2 user input → LLM

经典 CLI 中，`HermesCLI.chat()` 确保 credentials、route 和 agent 已初始化，将 clean user input 暂存进 history，然后在 agent thread 中调用 `AIAgent.run_conversation(user_message=..., conversation_history=..., task_id=self.session_id)`。

**Status:** CONFIRMED
**Evidence:** `cli.py:14390-14549`; `cli.py:14664-14739`
**Symbol:** `HermesCLI.chat()`, nested `run_agent()`
**Implication:** CLI 把 session id 作为 task id 传入，但 core 允许独立 task id；二者不能在架构上视为永远同一概念。

`AIAgent.run_conversation()` 生成缺省 task id，绑定 relay/accounting/subagent context，调用实际的 `agent.conversation_loop.run_conversation()`，然后根据 result 标记 logical turn outcome 并调用 `finish_task_run()`。

**Status:** CONFIRMED
**Evidence:** `run_agent.py:8045-8138,8155-8180`
**Symbol:** `AIAgent.run_conversation()`
**Implication:** task-run telemetry 的 finished 不等于工程任务经过验证，只表示此 runtime 调用返回了一个 outcome。

实际 loop 在进入 provider 前先执行 `build_turn_context()`：构建/恢复 system prompt、处理输入、预压缩、memory prefetch、`pre_llm_call` context 注入和持久化。

**Status:** CONFIRMED
**Evidence:** `agent/conversation_loop.py:1588-1617`; `agent/turn_context.py:1-8,1145-1204,1248-1310`
**Symbol:** `agent.turn_context.build_turn_context()`
**Implication:** Engineering Context 应复用 turn-context 或 context-engine/plugin seam，避免在历史消息中另造一套不持久或破坏 cache 的上下文注入。

每次迭代由 `_build_api_kwargs()` 根据 active transport 构造请求，经 LLM request/execution middleware 后，优先走 `_interruptible_streaming_api_call()`，必要时走 `_interruptible_api_call()`。统一调用分派最终落到 Responses、Anthropic Messages、Bedrock Converse、MoA facade 或 OpenAI-compatible `chat.completions.create()`。

**Status:** CONFIRMED
**Evidence:** `agent/conversation_loop.py:2560-2638,2715-2831`; `agent/chat_completion_helpers.py:500-557,854-867,2761-2775`
**Symbol:** `AIAgent._build_api_kwargs()`, `_perform_api_call()`, `interruptible_streaming_api_call()`, `_dispatch_nonstreaming_api_request()`
**Implication:** Provider 差异应留在 provider/transport 层；Engineering orchestration 不应直接调用某家 SDK。

### 1.3 tool calling loop

loop 条件同时受 `max_iterations` 和共享 `iteration_budget` 控制，并保留一次 grace-call 语义。provider 返回 tool calls 时，loop 构建 assistant tool-call message，调用 `AIAgent._execute_tool_calls()`，将 tool results 写回消息并 `continue` 到下一次 LLM 调用。

**Status:** CONFIRMED
**Evidence:** `agent/conversation_loop.py:1709`; `agent/conversation_loop.py:6480-7132`; `run_agent.py:7848-7981`
**Symbol:** `agent.conversation_loop.run_conversation()`, `AIAgent._execute_tool_calls()`
**Implication:** Engineering workflow 可以把动作实现为现有 tools/skills/workflows，但不应复制“LLM → tools → LLM”的循环。

没有 tool calls 时，`assistant_message.content` 首先成为 `final_response`，之后才经过空响应恢复、verification continuation、持久化和 finalizer。

**Status:** CONFIRMED
**Evidence:** `agent/conversation_loop.py:7134-7165,7580-7795`
**Symbol:** `agent.conversation_loop.run_conversation()` no-tool-call branch
**Implication:** “模型给出文本”只是 candidate completion；真正返回还要经过 stop guards 和 finalizer。

---

## 2. Provider Runtime

### 2.1 abstraction、discovery 与 resolution

Provider 的声明抽象是 `ProviderProfile`：描述 identity、auth、base URL、`api_mode`、model catalog、request quirks 等；源码明确说明 profile 不负责 client construction、credential rotation 或 streaming，这些仍由 `AIAgent` 负责。

**Status:** CONFIRMED
**Evidence:** `providers/base.py:1-9,38-148`
**Symbol:** `ProviderProfile`
**Implication:** 新 provider 行为应优先扩展 profile/transport，不应塞进 Engineering Agent。

Provider registry 通过 `register_provider()`、`get_provider_profile()`、`list_providers()` 管理 profile。发现源包括 pip entry point、bundled `plugins/model-providers/`、`$HERMES_HOME/plugins/model-providers/` 和 legacy `providers/*.py`；模块 import 时自行调用 `register_provider(profile)`，后注册者覆盖前者。

**Status:** CONFIRMED
**Evidence:** `providers/__init__.py:45-97,100-179,271-323`; `plugins/model-providers/openai-codex/__init__.py:3-15`
**Symbol:** `register_provider()`, `get_provider_profile()`, `_discover_providers()`
**Implication:** Provider plugin 是独立的已存在扩展系统；Engineering 模块只消费 resolved runtime。

运行时权威 resolver 是 `resolve_runtime_provider()`。CLI 在首个 turn 前通过 `CLIAgentSetupMixin._ensure_runtime_credentials()` 延迟解析 provider/credential/base URL/API mode，并在 route 或 credential 变化时重建 agent。model 首先来自 CLI 参数或 `config.yaml model.default/model`，空 model 再按 resolved provider 选择默认模型。

**Status:** CONFIRMED
**Evidence:** `hermes_cli/runtime_provider.py:1665-1703,2187-2292`; `hermes_cli/cli_agent_setup_mixin.py:25-185`; `cli.py:4484-4555`
**Symbol:** `resolve_runtime_provider()`, `CLIAgentSetupMixin._ensure_runtime_credentials()`
**Implication:** 不应从 env、model 名称或 URL 自行猜 provider；应消费 resolver 结果 `{provider, api_mode, base_url, api_key, ...}`。

### 2.2 API mode 与 transport

当前合法 API mode 是 `chat_completions`、`codex_responses`、`anthropic_messages`、`bedrock_converse`、`codex_app_server`。`ProviderTransport` 统一 `convert_messages`、`convert_tools`、`build_kwargs`、`normalize_response`；transport registry 按 API mode 懒发现对应实现。

**Status:** CONFIRMED
**Evidence:** `hermes_cli/runtime_provider.py:385-405`; `agent/transports/base.py:1-65`; `agent/transports/__init__.py:17-67`
**Symbol:** `_VALID_API_MODES`, `ProviderTransport`, `register_transport()`, `get_transport()`
**Implication:** API mode 是 wire protocol 选择，不等同于商业 provider identity。

OpenAI-compatible provider 使用 `ChatCompletionsTransport` 和 OpenAI client；OpenAI Codex 使用 `ResponsesApiTransport`/Responses adapter；Anthropic 使用 `AnthropicTransport` 和 Anthropic SDK；Bedrock 使用 `BedrockTransport`/boto3 Converse。

**Status:** CONFIRMED
**Evidence:** `agent/transports/chat_completions.py:207,363`; `agent/transports/codex.py:266-281,787`; `agent/transports/anthropic.py:13-22,251`; `agent/transports/bedrock.py:15-68,154`; `agent/chat_completion_helpers.py:513-557`
**Symbol:** `ChatCompletionsTransport`, `ResponsesApiTransport`, `AnthropicTransport`, `BedrockTransport`
**Implication:** Review/verification 对 provider 应保持协议无关；provider-specific failure 交给 transport/error classifier/fallback。

Gemini profile 虽声明 `api_mode="chat_completions"`，实际 client 是 `GeminiNativeClient`，它将 OpenAI 风格调用翻译到 Gemini `generateContent`/`streamGenerateContent` HTTP API。

**Status:** CONFIRMED
**Evidence:** `plugins/model-providers/gemini/__init__.py:1-9,18-61`; `agent/gemini_native_adapter.py:1053-1093,1096-1124`
**Symbol:** `GeminiProfile`, `GeminiNativeClient`, `AsyncGeminiNativeClient`
**Implication:** 不能仅凭 `api_mode` 推断底层 SDK；稳定抽象是 client/transport contract。

OpenAI Codex 还有 opt-in `codex_app_server` 路径：当 `model.openai_runtime: codex_app_server` 且 provider 为 `openai`/`openai-codex` 时，整个 turn 交给复用的 `codex app-server` subprocess，其 terminal/file/patch/sandbox 不走 Hermes 正常 tool dispatch。

**Status:** CONFIRMED
**Evidence:** `hermes_cli/runtime_provider.py:385-395,419-442`; `agent/conversation_loop.py:1695-1707`; `agent/codex_runtime.py:677-757`
**Symbol:** `_maybe_apply_codex_app_server_runtime()`, `run_codex_app_server_turn()`, `CodexAppServerSession`
**Implication:** Engineering Policy/Verification 若只挂 Hermes tool hooks，会漏掉此模式；未来设计必须明确是禁用该模式、桥接其事件/approval，还是把它视作独立 delegated runtime。

### 2.3 auxiliary model

辅助模型由 `agent.auxiliary_client` 统一路由，服务 compression、session search、web extraction、vision 等 side tasks。主 turn 用 `set_runtime_main()` 把当前 provider/model/base URL/API mode 绑定到 `ContextVar`，防止并发 gateway sessions 相互覆盖。per-task 配置使用 `auxiliary.<task>.provider/model/base_url/api_mode`，显式参数优先，然后是 task 配置，再是 `auto`。

**Status:** CONFIRMED
**Evidence:** `agent/auxiliary_client.py:1-38,3363-3411,7727-7901,8962-9016`
**Symbol:** `set_runtime_main()`, `_resolve_task_provider_model()`, `resolve_provider_client()`, `call_llm()`
**Implication:** Verification/review 的轻量模型需求应优先声明为 auxiliary task，而不是自行创建 provider client；但确定性 gate 本身不能依赖辅助 LLM 的主观判断。

---

## 3. Tool Runtime

### 3.1 registry、schema 与 dispatch

内建 tool 模块通过顶层 `registry.register()` 自注册；`discover_builtin_tools()` 先用 AST 找出真正含顶层注册调用的 `tools/*.py` 再 import。`ToolRegistry.register()` 保存 schema、handler、toolset、availability `check_fn`、async flag 和 result limit，并对 plugin override 做显式授权检查。

**Status:** CONFIRMED
**Evidence:** `tools/registry.py:90-155,737-856`
**Symbol:** `discover_builtin_tools()`, `ToolRegistry.register()`
**Implication:** 新 Engineering tools 若确有必要，应通过 plugin context 或独立模块注册，并使用 `check_fn` 控制暴露；不能假设文件存在就自动进入 schema。

`ToolRegistry.get_definitions()` 只返回被请求且 availability check 通过的 OpenAI function schemas。`model_tools.get_tool_definitions()` 在 toolset/enable/disable 层选择它们，形成每次 provider 请求携带的 model tool surface。

**Status:** CONFIRMED
**Evidence:** `tools/registry.py:1018-1065`; `model_tools.py:307-636`
**Symbol:** `ToolRegistry.get_definitions()`, `get_tool_definitions()`
**Implication:** 每个新 model-facing tool 都增加请求 schema 成本；优先使用现有 terminal/file、skill 或 service-gated tool。

调用链为 `AIAgent._execute_tool_calls()` → `model_tools.handle_function_call()` → middleware → `ToolRegistry.dispatch()` → handler。registry 将返回值规范为 string 或唯一允许的 multimodal envelope，并把异常转成 tool error。

**Status:** CONFIRMED
**Evidence:** `run_agent.py:7848-7981`; `model_tools.py:1170-1585`; `tools/registry.py:1071-1142`
**Symbol:** `AIAgent._execute_tool_calls()`, `handle_function_call()`, `ToolRegistry.dispatch()`
**Implication:** Engineering Policy 的 tool-level 决策应在现有 pre-tool/middleware/approval seam 上做，不应绕过 `handle_function_call()` 直调 handler。

### 3.2 permission / approval

`pre_tool_call` 可返回 `block` 或 `approve`。`resolve_pre_tool_block()` 对 `approve` 调用统一的人类 approval gate，并在 gate 出错、拒绝或超时时 fail-closed；`handle_function_call()` 在 dispatch 前执行此判断。ACP 另有针对 `write_file`/`patch` 的 edit approval guard。

**Status:** CONFIRMED
**Evidence:** `hermes_cli/plugins.py:5806-6004`; `model_tools.py:1351-1436`; `acp_adapter/edit_approval.py:1-5`
**Symbol:** `resolve_pre_tool_block()`, `handle_function_call()`, `maybe_require_edit_approval()`
**Implication:** EngineeringPolicy 可用 `pre_tool_call` 实现确定性的 block/escalate；该 hook 是真实 enforcement point，不是 observer。

terminal 在执行前运行 consolidated command guards，包含 user deny rules、dangerous command detection、smart/manual approval、sudo/path/self-repo 等保护；本地显式 `force` 仅表示此前已获确认，且某些 non-bypassable guard 仍在更早处执行。

**Status:** CONFIRMED
**Evidence:** `tools/approval.py:3504-3574,3982-4463`; `tools/terminal_tool.py:376-381,2886-2972`
**Symbol:** `check_all_command_guards()`, `prompt_dangerous_approval()`, `terminal_tool()`
**Implication:** Database Safety/Production Guardrails 应叠加在统一 policy/approval 层，而不是只在 prompt 中要求模型谨慎。

### 3.3 terminal、filesystem 与 shell

`terminal_tool()` 根据 config/env 解析 backend，并缓存 task/session 对应的 environment。工厂支持 local、Docker、SSH、Singularity、Modal、Daytona、Vercel Sandbox；所有 backend 实现 `BaseEnvironment` 共同 shell contract。

**Status:** CONFIRMED
**Evidence:** `tools/terminal_tool.py:1090-1105,1571-1647,1755-1947,2046-2132,2533-2561`; `tools/environments/__init__.py:1-13`; `tools/environments/base.py`
**Symbol:** `_get_env_config()`, `_create_environment()`, `ensure_task_env()`, `terminal_tool()`, `BaseEnvironment`
**Implication:** ProjectInspector/VerificationEngine 应通过已选 environment 执行，不能默认命令一定在 host local filesystem 上。

local backend 使用受控 subprocess shell、持久 cwd、环境快照和输出限制；SSH/container backends 有各自 cwd/path 语义。file tools 解析路径时也读取 task 的 active terminal environment，以避免把 container path 当 host path 解引用。

**Status:** CONFIRMED
**Evidence:** `tools/environments/local.py:1015-1067,1414-1671`; `tools/file_tools.py:152-213,365`; `tools/terminal_tool.py:2493-2530`
**Symbol:** `LocalEnvironment`, `_resolve_path_for_task()`, `_resolve_command_cwd()`
**Implication:** Verification evidence 必须记录 backend、cwd 和命令；“运行过”不能脱离 execution environment。

filesystem surface 由 `read_file`、`write_file`、`patch`、`search_files` 提供，包含 read size、device/special-file、binary、Hermes internal/cross-profile、write approval 等 guards。

**Status:** CONFIRMED
**Evidence:** `tools/file_tools.py:52-167,152-260,1620,2173,2262,2472,2746-2749`
**Symbol:** `read_file_tool()`, `write_file_tool()`, `patch_tool()`, `search_tool()`
**Implication:** Engineering file operations 应复用这些 guards；直接 Python 文件写入会绕过 runtime policy 与 observability。

---

## 4. Plugin System

插件来源为 bundled、user、project（显式 opt-in）和 pip entry point。directory plugin 必须有 `plugin.yaml` 和带 `register(ctx)` 的 `__init__.py`。general plugin、model-provider、memory/context exclusive plugin 有不同 loader ownership，不能混为一个 discovery path。

**Status:** CONFIRMED
**Evidence:** `hermes_cli/plugins.py:1-31,3758-3993`; `providers/__init__.py:271-323`; `plugins/memory/__init__.py`
**Symbol:** `PluginManager.discover_and_load()`, `PluginManager._discover_and_load_inner()`, `_discover_providers()`
**Implication:** Engineering plugin 应选择正确 category；不要让 general loader 与 provider/context/memory 专用 loader 重复加载。

实际 load 过程 import module、查找 `register`、创建 `PluginContext` 并调用 `register_fn(ctx)`，同时通过 ownership ledger 记录 tools、hooks、middleware 等，以支持 unload/reload 和 collision restoration。

**Status:** CONFIRMED
**Evidence:** `hermes_cli/plugins.py:4590-4716`
**Symbol:** `PluginManager._load_plugin()`, `PluginManager._load_plugin_scoped()`
**Implication:** 长期扩展应通过 `PluginContext` 注册，让宿主管理生命周期，而不是 import-time 修改 core globals。

`PluginContext` 可注册 tool、顶层 CLI subcommand、会话 slash command、hook、system prompt section、middleware、skill、context engine 等；`dispatch_tool()` 是明确标注的 plugin public interface，可在 slash command 中复用已有 tool。

**Status:** CONFIRMED
**Evidence:** `hermes_cli/plugins.py:1388-3381`; 特别是 `1700-1789,2061-2200,3109-3209,3289-3381`
**Symbol:** `PluginContext.register_tool()`, `register_cli_command()`, `register_command()`, `dispatch_tool()`, `register_hook()`, `register_system_prompt_section()`, `register_middleware()`, `register_skill()`
**Implication:** Engineering workflows/commands/skills 可先作为 plugin-facing capability 存在，只有需要共享 deterministic state machine 时再引入专门 engineering module。

Plugin skill 是 namespaced、只读、显式加载的，不进入 flat skills tree，也不自动出现在 system prompt 的 `<available_skills>` 索引中。

**Status:** CONFIRMED
**Evidence:** `hermes_cli/plugins.py:3319-3381`
**Symbol:** `PluginContext.register_skill()`
**Implication:** Engineering Skills 若需要默认可发现，不能只依赖 plugin skill 的隐式曝光；需明确 built-in/user skill 与 plugin namespaced skill 的产品策略。

---

## 5. Hook Lifecycle：行为影响与 observer

### 5.1 真实行为影响 hook

| Hook / seam | 真实效果 | Status | Evidence / Symbol | Engineering implication |
|---|---|---|---|---|
| `pre_tool_call` | 可 `block`，也可 `approve` 进入人类审批；失败关闭 | CONFIRMED | `hermes_cli/plugins.py:5806-6004` / `resolve_pre_tool_block()` | 可承载 tool policy enforcement |
| `pre_llm_call` | 返回字符串或 `{context}`，注入本轮 user message；不改 system prompt、不持久化 | CONFIRMED | `hermes_cli/plugins.py:4910-4931`; `agent/turn_context.py:1152-1204` / `invoke_hook()`, `build_turn_context()` | 适合动态 Engineering Context，不适合强制完成规则 |
| `transform_tool_result` | 第一条有效 string 替换 tool result，再进入模型上下文 | CONFIRMED | `model_tools.py:1525-1538` 及后续 transform 分支 / `handle_function_call()` | 可做规范化/脱敏；不应伪造执行事实 |
| `transform_llm_output` | 第一条非空 string 替换最终返回文本 | CONFIRMED | `agent/turn_finalizer.py:551-575` / `finalize_turn()` | 能改呈现，不能证明工程完成 |
| `pre_verify` | 在检测到编辑路径时，可 bounded continuation 并重入 loop | CONFIRMED | `hermes_cli/plugins.py:6007-6055`; `agent/conversation_loop.py:7661-7721` | 是现成 verification continuation seam，但不是通用 hard gate |
| LLM/tool middleware | request 可重写，execution 可包裹真实 callback | CONFIRMED | `hermes_cli/plugins.py:3289-3314`; `hermes_cli/middleware.py`; `agent/conversation_loop.py:2617-2638,2804-2831`; `model_tools.py:1491-1503` | 比 observer hook 更适合结构化 policy/interception |

### 5.2 observer-only 或返回值未被消费

`post_tool_call` 明确是 observer；其返回值不参与 result。真正能变换 result 的是独立 `transform_tool_result`。

**Status:** CONFIRMED
**Evidence:** `model_tools.py:1116-1167,1512-1530`
**Symbol:** `_emit_post_tool_call_hook()`, `handle_function_call()`
**Implication:** 不应在 `post_tool_call` 中实现 veto 或补救性修改。

`post_llm_call` 在 final response 已生成并经过 output transform 后调用，调用结果被忽略，用于同步/观测。

**Status:** CONFIRMED
**Evidence:** `agent/turn_finalizer.py:577-596`
**Symbol:** `finalize_turn()` post-LLM block
**Implication:** `post_llm_call` 不能阻止 completion。

`on_session_start` 的返回值被忽略；`on_session_end` 名称容易误导，它在每次 `run_conversation()` 结束时触发，也就是 per-turn，而不是真实 session boundary。真实 boundary 另有 `on_session_finalize`，由 CLI/gateway/TUI 的 close/reset/expiry 路径调用。

**Status:** CONFIRMED
**Evidence:** `agent/conversation_loop.py:716-728`; `agent/turn_finalizer.py:768-793`; `hermes_cli/lifecycle.py:40-63`; `cli.py:8621-8752`; `gateway/run.py:12794-12836`; `tui_gateway/server.py:694-792`
**Symbol:** `invoke_hook("on_session_start")`, `finalize_turn()`, `finalize_session()`
**Implication:** Engineering task lifecycle 不能直接映射到 `on_session_end`；需定义自己的 task/workflow boundary。

`subagent_start`/`subagent_stop` 在 Hermes child `AIAgent` 创建和结果汇总时发出，调用返回值未被消费，因此是 observer。

**Status:** CONFIRMED
**Evidence:** `tools/delegate_tool.py:1878-1893,3208-3242`
**Symbol:** `_build_child_agent()` 内 hook 调用、`delegate_task()` 结果汇总
**Implication:** 当前 subagent hooks 可做追踪，不能阻止 spawn 或修改 child outcome；控制应放在 delegate tool policy/config/dispatch 前。

`pre_api_request`/`post_api_request` 是观测型 lifecycle hook；真正的请求变换由 LLM middleware 承担。`transform_api_error_classification` 是例外：它可覆盖 error classification。

**Status:** CONFIRMED
**Evidence:** `agent/conversation_loop.py:2640-2704`; `hermes_cli/plugins.py:185-211`; `agent/error_classifier.py`
**Symbol:** `pre_api_request`, `post_api_request`, `transform_api_error_classification`
**Implication:** 不要通过 observer 偷改 request；使用明确 middleware/transform contract。

---

## 6. Context

System prompt 由 `agent.system_prompt.build_system_prompt_parts()` 分为 `stable`、`context`、`volatile` 三层，再由 `build_system_prompt()` 连接。完整 prompt 缓存在 `agent._cached_system_prompt`，正常 session 中不重建，compression/restore 才允许重建，以保持 prompt cache prefix。

**Status:** CONFIRMED
**Evidence:** `agent/system_prompt.py:265-281,689-715`; `agent/agent_init.py:1618-1656`
**Symbol:** `build_system_prompt_parts()`, `build_system_prompt()`, `invalidate_system_prompt()`
**Implication:** Engineering Context 的动态信息优先注入 user-turn/context engine；不要每 turn 修改 system prompt。

Project context 类型按“第一类命中即停止”选择：`.hermes.md/HERMES.md` → `AGENTS.md/agents.md` → `CLAUDE.md/claude.md` → Cursor rules。`AGENTS.md` 在 git root 到 cwd 的目录链合并，深层规则排在后；`CLAUDE.md` 只读 cwd。`SOUL.md` 独立处理。

**Status:** CONFIRMED
**Evidence:** `agent/prompt_builder.py:2176-2255,2258-2274,2310-2384`
**Symbol:** `_agents_md_directory_chain()`, `_load_agents_md()`, `_load_claude_md()`, `build_context_files_prompt()`
**Implication:** 当 `.hermes.md` 存在时，AGENTS/CLAUDE project context 不会同时加载；Engineering Agent 不能假定所有 instruction files 都已进入 prompt。

Skills index 由 `build_skills_system_prompt()` 生成 `<available_skills>`，skill 正文按需由 skill tools 读取；CLI slash skill command 也能把 skill 内容作为 user message 注入。Plugin skill 则为显式 namespaced load。

**Status:** CONFIRMED
**Evidence:** `agent/prompt_builder.py:1488-2006`; `tools/skills_tool.py`; `agent/skill_commands.py`; `hermes_cli/plugins.py:3319-3381`
**Symbol:** `build_skills_system_prompt()`, `skill_view`, `scan_skill_commands()`, `PluginContext.register_skill()`
**Implication:** Engineering Skills 应保持按需加载，避免把完整 playbook 永久塞入 system prompt。

`ContextEngine` 是可插拔 ABC：负责 usage 更新、`should_compress()`、`compress()`，并可实现 per-turn `select_context()`。默认实现 `ContextCompressor` 进行有损总结和 deterministic tool-result pruning；host 在 `context.engine` 选择替代引擎。

**Status:** CONFIRMED
**Evidence:** `agent/context_engine.py:1-25,89-177,213-260`; `agent/context_compressor.py:1578-1595`; `agent/agent_init.py:2504-2617`
**Symbol:** `ContextEngine`, `ContextCompressor`, `select_context()`, `compress()`
**Implication:** Engineering Context 应评估 ContextEngine/plugin seam；但 project inspection facts 若是确定性状态，必须有独立结构化存储，不能只存在于 lossy compression summary。

---

## 7. Session / Memory

SQLite `SessionDB` 是 conversation persistence 的核心：惰性创建 session row，消息按 insertion id 保存/读取，`end_session()` 只标记结束且 first end reason wins。`AIAgent._ensure_db_session()` 在首用时写入 model、system prompt、cwd、parent session 等。

**Status:** CONFIRMED
**Evidence:** `run_agent.py:630-669,1902-2012`; `hermes_state.py:2691,4237-4240,5147-5163,8856-8922`
**Symbol:** `SessionDB`, `AIAgent._ensure_db_session()`, `SessionDB.create_session()`, `get_messages()`, `end_session()`
**Implication:** Engineering evidence 可引用 session/task/turn，但不应把普通 chat transcript 当作唯一 workflow database。

Gateway 另外使用 `SessionStore`/`SessionEntry` 管理 platform session key、expiry、routing、resume、compression rotation 和 transcript cache，再与 `SessionDB` 对齐。

**Status:** CONFIRMED
**Evidence:** `gateway/session.py:777-922,1238-1641,2428-2836,3226-3882`
**Symbol:** `SessionEntry`, `SessionStore`, `get_or_create_session()`, `reset_session()`, `advance_compression_session()`
**Implication:** session lifecycle 是 surface-owned；Engineering task lifecycle 需要独立于 gateway expiry 和 compression session rotation。

`session_id` 表示持久 conversation/session；`task_id` 表示一次 `run_conversation()` 的执行关联 id。CLI 常把 session id 传作 task id，但 core 在未传时生成 UUID；每次调用还生成独立 `turn_id`/relay turn id。

**Status:** CONFIRMED
**Evidence:** `cli.py:14732-14739`; `run_agent.py:8045-8055`; `agent/turn_context.py` 中 turn id 创建与传播
**Symbol:** `AIAgent.run_conversation()`, `build_turn_context()`
**Implication:** Engineering workflow run id 不应复用 session id；至少需要 session / workflow run / turn 三层标识。

Memory 通过 `MemoryProvider` ABC 和 `MemoryManager` 聚合：turn start 后进行 prefetch 并把 recall 作为 ephemeral user context 注入；turn finalizer 异步 sync 当前 user/assistant exchange 并预取下一轮；真正 session boundary 才调用 provider `on_session_end()` 和 shutdown。

**Status:** CONFIRMED
**Evidence:** `agent/memory_provider.py:104-251`; `agent/memory_manager.py:364-398,525-659,915-962`; `agent/turn_context.py:1248-1310`; `agent/turn_finalizer.py:740-773`; `run_agent.py:4133-4168`
**Symbol:** `MemoryProvider`, `MemoryManager.prefetch_all()`, `sync_all()`, `on_session_end()`
**Implication:** Memory 适合 recall/learning，不是 VerificationEngine 的事实账本；验证证据要可审计、结构化、不可被总结覆盖。

---

## 8. Completion：真实代码路径与缺口

### 8.1 正常路径

正常完成路径为：provider 返回无 tool-call assistant message → `assistant_message.content` 成为 candidate `final_response` → verify-on-stop / `pre_verify` / kanban stop guard 可注入 synthetic user message 并 `continue` → 接受的 final assistant row 先 flush 到 session DB → 跳出 loop → `finalize_turn()` 计算 `completed`、补齐 transcript、持久化、transform output、调用 observers、构造 result → `AIAgent.run_conversation()` 调用 `finish_task_run()` → surface 展示 `result.final_response`。

**Status:** CONFIRMED
**Evidence:** `agent/conversation_loop.py:7134-7147,7580-7795,7892-7911`; `agent/turn_finalizer.py:70-203,300-420,551-690,775-798`; `run_agent.py:8110-8138`; `cli.py:14943-14962,15138-15186`
**Symbol:** `run_conversation()`, `finalize_turn()`, `AIAgent.run_conversation()`, `HermesCLI.chat()`
**Implication:** “task/session finished”分三层：turn result returned、telemetry task run finished、真实 session boundary finalized；它们都不是自动的工程验收结论。

`finalize_turn()` 的 `completed` 主要由 `final_response != None`、未 failed、iteration condition/normal text response 计算；它没有查询测试结果、review verdict 或工程 policy。

**Status:** CONFIRMED
**Evidence:** `agent/turn_finalizer.py:194-203,655-690`
**Symbol:** `finalize_turn()`
**Implication:** 当前 `completed=True` 只能解释为 agent turn 完成，不能解释为 Engineering task 完成。

### 8.2 completion capability matrix

| 能力 | 判定 | Evidence / Symbol | 说明与 implication |
|---|---|---|---|
| `before_complete` | **GAP** | 全仓库源码无 `before_complete` symbol；最终 stop chokepoint 位于 `agent/conversation_loop.py:7580-7795` | 没有命名且通用的 completion lifecycle contract |
| 通用 completion veto | **GAP** | `finalize_turn()` 不消费 veto；`post_llm_call` 返回值忽略；`agent/turn_finalizer.py:194-203,577-596` | 现有机制不能表达任意 workflow 的 fail-closed terminal predicate |
| verification gate | **CONFIRMED，但有限** | `agent/verification_stop.py:95-134,233-313`; `agent/conversation_loop.py:7602-7721` | `verify_on_stop` 默认关；`pre_verify` 需 hook+edited paths；两者都 bounded |
| finalization gate | **GAP（通用语义）** | `agent/turn_finalizer.py:70-798` | `finalize_turn()` 是 post-loop result/persistence finalizer，不是可 veto gate |
| retry / re-enter agent loop | **CONFIRMED** | `agent/conversation_loop.py:7620-7659,7694-7721,7740-7771` | 通过 synthetic user message + `continue` 可有限重入；普通 provider retry/fallback 也在同一 loop 内 |
| completion output transform | **CONFIRMED** | `agent/turn_finalizer.py:551-575` / `transform_llm_output` | 只能改变文本，不能补足验证事实 |
| real session-finalize observer | **CONFIRMED** | `hermes_cli/lifecycle.py:40-63` / `finalize_session()` | 是 boundary notification，返回值未作为 veto 消费 |

当 verification continuation 已保存 candidate response、但后续耗尽 iteration budget 时，`finalize_turn()` 会恢复该 pending candidate，而不是保持 incomplete。这证明当前 continuation gate 不是严格 fail-closed completion veto。

**Status:** CONFIRMED
**Evidence:** `agent/turn_finalizer.py:95-143`
**Symbol:** `finalize_turn()` 的 `continuation_budget_exhausted` 分支
**Implication:** Engineering Review Gate 若要求“未验收绝不完成”，需要新的确定性 workflow state/finalization contract，不能仅依赖现有 `pre_verify` nudge。

Codex app-server path 提前绕过普通 conversation-loop finalizer，并在 `agent.codex_runtime.run_codex_app_server_turn()` 中自行镜像结果/持久化语义。

**Status:** CONFIRMED
**Evidence:** `agent/conversation_loop.py:1695-1707`; `agent/codex_runtime.py:677-691,795-934`
**Symbol:** `run_codex_app_server_turn()`
**Implication:** 任何未来 completion gate 必须覆盖 normal loop 和 codex app-server 两条路径，否则会存在旁路。

---

## 9. Delegated Coding Agents

### 9.1 Hermes-native subagents

`delegate_task` 创建新的 child `AIAgent`，带隔离 context、继承/收窄后的 toolsets、独立 model/provider override、shared iteration budget 和可选 worktree isolation。它是 Hermes tool + Hermes subagent，不是外部 coding-agent CLI。

**Status:** CONFIRMED
**Evidence:** `tools/delegate_tool.py:1-14,50-125,1471-1893,3205-3242`
**Symbol:** `delegate_task`, `_build_child_agent()`, `AIAgent`
**Implication:** EngineeringOrchestrator 可优先调度 Hermes-native subagents；其生命周期与外部进程代理不同。

### 9.2 Codex

Codex 有三种不同身份：

1. `openai-codex` 是 provider profile，经 OAuth + Responses API 运行主 Hermes loop；
2. `codex_app_server` 是 opt-in subprocess runtime，整 turn 委托给 Codex app-server；
3. `skills/autonomous-ai-agents/codex/SKILL.md` 指导 Hermes 通过 terminal 执行 `codex exec`，这是 skill-orchestrated external subprocess。

**Status:** CONFIRMED
**Evidence:** `plugins/model-providers/openai-codex/__init__.py:1-15`; `agent/transports/codex.py:266-787`; `agent/codex_runtime.py:677-757`; `skills/autonomous-ai-agents/codex/SKILL.md:44-56,147`
**Symbol:** `ProviderProfile(name="openai-codex")`, `ResponsesApiTransport`, `run_codex_app_server_turn()`
**Implication:** “调用 Codex”必须明确是 provider、app-server runtime，还是 CLI worker；三者的 policy/approval/evidence 路径不同。

### 9.3 Claude Code

`anthropic`/`claude-code` 可以是 provider alias，Hermes 可读取 Claude Code OAuth credential 并用 Anthropic Messages API；这不是启动 Claude Code agent。真正 delegated Claude Code 由 `claude-code` skill 指导 terminal/tmux 启动 `claude` CLI，因此分类为 skill + subprocess，而不是 Hermes subagent。

**Status:** CONFIRMED
**Evidence:** `plugins/model-providers/anthropic/__init__.py:14-54`; `agent/anthropic_adapter.py:402-477,985-1474,2885-2944`; `skills/autonomous-ai-agents/claude-code/SKILL.md:14-31,58-83,721`
**Symbol:** `AnthropicProfile`, Claude credential/auth helpers, `claude-code` skill
**Implication:** Provider-level Claude 与 Claude Code CLI worker 必须分别建模；后者的 tool actions 不经过 Hermes `handle_function_call()`。

### 9.4 OpenCode

`opencode-zen`/`opencode-go` 是 Hermes provider profiles，按 model 在 OpenAI-compatible 与 Anthropic Messages wire 间选择；这不是启动 OpenCode agent。真正 delegated OpenCode 由 `opencode` skill 通过 terminal 执行 `opencode run` 或后台 TUI，分类为 skill + subprocess。

**Status:** CONFIRMED
**Evidence:** `plugins/model-providers/opencode-zen/__init__.py:1-15,52-161`; `hermes_cli/models.py:4430-4521`; `hermes_cli/runtime_provider.py:546-570,2247-2271`; `skills/autonomous-ai-agents/opencode/SKILL.md:14-29,45-79,141-146`
**Symbol:** `OpenCodeGoProfile`, `opencode_zen`, `opencode_model_api_mode()`, `opencode` skill
**Implication:** OpenCode provider 和 OpenCode CLI worker 同样必须拆分；provider selection 不能被当成 delegated coding-agent orchestration。

### 9.5 programmatic delegation gap

除 Codex app-server 的专用 runtime 外，仓库没有统一的 typed “ExternalCodingAgent” API 来直接 programmatically 调度 Codex CLI、Claude Code CLI、OpenCode CLI；当前通用方式是模型按 skill 指导调用 terminal/process tools。

**Status:** GAP
**Evidence:** 外部 CLI 的调用说明位于 `skills/autonomous-ai-agents/codex/SKILL.md`; `skills/autonomous-ai-agents/claude-code/SKILL.md`; `skills/autonomous-ai-agents/opencode/SKILL.md`；核心 typed delegation 实现 `tools/delegate_tool.py` 只创建 child `AIAgent`; Codex 特例位于 `agent/codex_runtime.py`
**Symbol:** `delegate_task`, `run_codex_app_server_turn()`；不存在统一 external coding-agent registry/interface
**Implication:** 若 V1 需要 delegated coding agents，应先定义最小、可验证的 adapter/workflow contract，而不是把 skill prose 误认为稳定 programmatic API；是否进入 V1 仍需产品范围确认。

---

## 10. Public / Stable Extension Surface

以下“稳定性”是基于源码中明确 public/ABC/registration contract、兼容处理和 ownership lifecycle 的工程判断，不是上游 semantic-version 承诺。

### 10.1 可以优先依赖

| Surface | Status | Evidence / Symbol | 判断 |
|---|---|---|---|
| `PluginContext` registration APIs | CONFIRMED | `hermes_cli/plugins.py:1388-3381` / `PluginContext` | 最明确的 general extension surface；有 register/unload ownership |
| `PluginContext.dispatch_tool()` | CONFIRMED | `hermes_cli/plugins.py:2171-2200` | 源码明确称 public interface |
| `ProviderProfile` + `register_provider()` | CONFIRMED | `providers/base.py:1-9,38`; `providers/__init__.py:56-78` | provider 专用声明扩展面 |
| `ContextEngine` ABC | CONFIRMED | `agent/context_engine.py:1-25,89-260` | 有 lifecycle/core interface 与 backwards-compatible optional methods |
| `MemoryProvider` ABC | CONFIRMED | `agent/memory_provider.py:104-251` | memory provider 的正式 interface |
| plugin skills / commands / middleware / hooks | CONFIRMED | `hermes_cli/plugins.py:2061-2167,3109-3381` | 有显式注册与兼容 payload 过滤 |
| `AIAgent.chat()` / `run_conversation()` | CONFIRMED（公开调用形状） | `run_agent.py:8017-8194` | 可作为 host API 使用，但构造参数和 result 细节较宽，耦合需薄封装 |

**Implication:** Engineering modules 应在这些 surface 外再放一层项目自己的 adapter/protocol，避免上游更新扩散到业务模块。

### 10.2 可用但语义受限

`pre_tool_call`、`pre_llm_call`、`pre_verify`、transform hooks 是公开 plugin hook，但每个 hook 的影响能力不同；不能把 hook 名称当作通用 middleware。`on_session_end` 实际 per-turn，更应避免按名字推断。

**Status:** CONFIRMED
**Evidence:** `hermes_cli/plugins.py:156-244,4910-4953,5806-6055`; `agent/turn_finalizer.py:775-793`
**Symbol:** `VALID_HOOKS`, `PluginManager.invoke_hook()`
**Implication:** 建立 Engineering hook adapter 时必须固定并测试每个 hook 的真实输入、返回消费规则和触发次数。

Tool registry 可被 import，但 plugin 已有 `PluginContext.register_tool()`/`dispatch_tool()`；直接依赖 `tools.registry.registry` 会绑定内部 scope、override、availability cache 和 result normalization。

**Status:** INFERENCE
**Evidence:** `tools/registry.py:737-1142`; `hermes_cli/plugins.py:1700-1789,2171-2200`
**Symbol:** `ToolRegistry`, `PluginContext.register_tool()`, `PluginContext.dispatch_tool()`
**Implication:** 新模块优先走 PluginContext；只有 core-owned built-in tool 才直接注册 registry。

### 10.3 内部实现，不应强耦合

`agent.conversation_loop`、`agent.turn_context`、`agent.turn_finalizer`、`agent.chat_completion_helpers`、`HermesCLI` 私有字段、gateway `SessionStore` 私有 routing、terminal `_active_environments` 都是为拆分 god-file/内部状态而存在的实现模块或私有符号。

**Status:** INFERENCE
**Evidence:** `run_agent.py:8030,7223-7226`; `agent/turn_finalizer.py:88-91`; `hermes_cli/cli_agent_setup_mixin.py:1-12`; 私有 symbol 命名与 forwarder 注释遍布上述模块
**Symbol:** `agent.conversation_loop.run_conversation()`, `finalize_turn()`, `_build_api_kwargs()`, `_active_environments`
**Implication:** 若必须加入 completion seam，应做极薄、通用、带测试的 core contract；Engineering 逻辑本身留在新 module/plugin。

直接修改 past messages、mid-session system prompt、全局 toolset 或 process-env surface detection 都会破坏 prompt caching/session isolation。

**Status:** CONFIRMED
**Evidence:** `agent/system_prompt.py:265-281,689-715`; `hermes_cli/plugins.py:4919-4931`; `agent/context_engine.py:243-260`; `tools/registry.py:1018-1027`
**Symbol:** `build_system_prompt()`, `PluginManager.invoke_hook()`, `ContextEngine.select_context()`, `ToolRegistry.get_definitions()`
**Implication:** Engineering Context 和 tool availability 必须分别遵循 cache-stable 与 session-scoped contract。

---

## 11. 对 Engineering Agent 架构的直接约束（仅研究结论）

1. **复用 runtime，新增 orchestration。** `AIAgent`、provider transports、tool dispatch、session/memory/context 已存在；重新实现它们会制造重复 runtime。
2. **Completion 必须单独建模。** 当前 `completed` 是 turn completion；严格 Engineering completion 是 GAP。
3. **Policy 放在可执行 choke point。** tool policy 可用 `pre_tool_call`/approval/middleware；只写 prompt 不构成 enforcement。
4. **Verification evidence 必须结构化。** 现有 verify-on-stop 是 bounded nudge，默认关闭且可在预算耗尽时退回 candidate。
5. **Review 与 finalization 要区分。** output transform/post hooks 不是 review gate；真实 gate 需能返回 machine-readable verdict 并控制 terminal transition。
6. **外部 coding agent 是独立 execution mode。** skill+subprocess、provider、Hermes subagent、Codex app-server 必须分别建模。
7. **覆盖旁路。** normal loop 与 Codex app-server 都必须受一致的 Engineering completion/policy contract 约束。
8. **不把 session 当 task。** 至少区分 conversation session、engineering workflow run、turn/tool call。

**Status:** INFERENCE（由以上 CONFIRMED/GAP 事实推导）
**Evidence:** 本文第 1、3、5、8、9、10 节列出的源码路径
**Symbol:** `AIAgent.run_conversation()`, `handle_function_call()`, `finalize_turn()`, `run_codex_app_server_turn()`
**Implication:** 下一阶段在设计 EngineeringOrchestrator/VerificationEngine 前，应先定义 workflow state、completion predicate、evidence schema 与两条 runtime 路径的接入边界；本阶段不实现。

---

## 12. 仍需后续确认的未知点

以下不是本次源码调查可以直接确认的 Runtime 事实：

- V1 是否允许启用外部 Codex/Claude Code/OpenCode CLI worker，还是仅支持 Hermes-native subagents。
- Engineering completion gate 应否覆盖所有 Hermes surfaces，还是先只覆盖明确的 engineering surface。
- Codex app-server 在 V1 中应禁用、降级为独立 worker，还是纳入统一 policy/evidence bridge。
- Verification evidence 的持久化载体是扩展 `SessionDB`、独立 engineering store，还是 plugin-owned store。
- Review Gate 的 actor 是 deterministic rules、同模型复核、auxiliary model、独立 reviewer，还是组合；源码只证明现有 runtime 不替产品做该决定。
- 工程 workflow 与 chat session 的恢复、压缩 rotation、branching 之间需要怎样的 lineage contract。

**Status:** GAP（产品/架构决策尚未在当前 Runtime 中形成稳定 contract）
**Evidence:** 当前源码仅提供通用 session、hook、verification nudge 和 delegation primitive；无 `EngineeringOrchestrator`、通用 completion veto 或 external coding-agent interface
**Symbol:** N/A
**Implication:** 这些问题应在实现前形成受控 V1 决策，不能通过提前编码默认答案。
