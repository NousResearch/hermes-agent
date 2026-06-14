# Fork notes - Eynzof/hermes-agent-cn

英文版：[`FORK_NOTES.md`](./FORK_NOTES.md)

本文记录 `main` 分支相对官方上游 `NousResearch/hermes-agent` 的 fork 专属改动。新的行为补丁应使用 `[CN-fork] P-NNN: ...` 提交信息，并在本文登记。

## 补丁总览

| ID | 目标文件 | 做了什么 | 为什么需要 | 上游状态 |
|---|---|---|---|---|
| **P-001** | `tui_gateway/server.py` | provider 配置 dict/list 不一致修复 | 早期 fork 需要兼容用户配置形态 | 已由上游修复，本 fork 不再携带 |
| **P-002** | `hermes_cli/web_server.py` | 增加 `POST /api/upload` 附件上传接口 | desktop / web composer 拖拽上传依赖它 | 未进入上游 |
| **P-003** | `hermes_cli/web_server.py` | 去掉 `/api/ws` 的 `_DASHBOARD_EMBEDDED_CHAT_ENABLED` 门禁 | desktop 以 headless dashboard 方式运行，不带 `--tui` 时仍需要 gateway WS | **基本被上游解决** —— v0.16.0(#38591)默认把该标志设为 `True` 并移除了 `--tui`；fork 仍保留 `/api/ws` 上的显式去门禁作为纵深防御 |
| **P-004** | `hermes_cli/web_server.py` | 增加 `GET /api/fs/list` 文件夹浏览接口 | web 工作区选择器需要列目录，避免让用户手输路径 | 未进入上游 |
| **P-005** | `hermes_cli/web_server.py` | 增加 `GET /api/mcp-servers` 只读 MCP 列表 | desktop 健康检查需要 MCP 数量，但不能泄露 command/args/env | 可考虑上游 |
| **P-006** | `hermes_cli/config.py` | 为 CN provider 注册 `OPTIONAL_ENV_VARS` | 模型设置页需要展示 ARK、QIANFAN、HUNYUAN、SiliconFlow 等密钥项 | CN 专属，通常不向上游提交 |
| **P-007** | `tui_gateway/ws.py` | 捕获并记录 gateway dispatch 异常，返回 JSON-RPC error | 否则前端只看到 WebSocket closed，缺少诊断信息 | 建议上游 |
| **P-008** | `hermes_cli/web_server.py` | 增加 `GET/PUT /api/profiles/active` | desktop profile 切换器需要读写 sticky active profile | 建议上游 |
| **P-009** | `hermes_cli/web_server.py`, `tui_gateway/sse.py` | 增加 `/api/v2/events` SSE 和 `/api/v2/rpc` POST transport | ~~desktop 默认使用 EventSource + POST~~ → desktop ≥ 0.4 已改用原生 `/api/ws` WebSocket（与官方桌面端一致），此 transport 只服务旧版外壳 | **已弃用** —— 为 ≤ 0.3.x 旧外壳保留（外壳无自更新而 runtime 热更新），旧外壳 EOL 后移除。不上游。 |
| **P-010** | `hermes_cli/config.py` | 注册 `LONGCAT_API_KEY` | CN 模型设置需要 LongCat 密钥入口 | CN 专属，除非上游支持 LongCat |
| **P-011** | `tui_gateway/server.py` | 给 `model.options` 增加 `slug_filter`，并增加 `provider.probe` RPC | desktop 需要过滤模型选择器，并轻量探测 provider 状态 | 可考虑上游 |
| **P-012** | `hermes_cli/main.py` | `_model_flow_anthropic()` 支持保留或自定义 `base_url`，不再无条件删除 | 使用 Anthropic 兼容代理或私有端点的用户需要在模型设置流程中保留自定义 `base_url` | 建议上游 |
| **P-013** | `model_tools.py`, `tests/run_agent/test_repair_tool_arg_keys.py` | 在 `handle_function_call` 中增加自动参数键修复：全局别名表、工具级覆盖、模糊匹配、嵌套对象/数组递归修复，以及可选回调通知 | LLM 经常把参数名写错（如 `file`→`path`、`cmd`→`command`），此前会直接报 "unknown parameter"；该补丁在不放宽 JSON Schema 的前提下提高工具调用的容错率 | 建议上游 |
| **P-014** | `.github/workflows/release-runtime.yml`, `tools/mcp_tool.py`, `hermes_cli/config.py`, `docs/RUNTIME_RELEASES.md`, `tests/tools/test_mcp_tool.py` | 把原生 MCP 客户端 SDK 打进冻结 runtime（安装入口后并入 `cn-desktop` extra，见 P-015；外加 `--collect-submodules/--copy-metadata mcp` + CI 断言 `mcp-*.dist-info` 存在）；并让 `discover_mcp_tools()` 在已配置 `mcp_servers` 但 SDK 缺失时输出一次 WARNING，而不是在 debug 级别静默跳过 | issue #16：desktop runtime 打包时缺少 `mcp` extra，导致 `_MCP_AVAILABLE=False`，已配置的 `mcp_servers` 不注册任何工具且 INFO 日志无任何提示。打包改动是 CN 特有，诊断日志与已知根键则是通用改进 | 打包改动 CN 特有；`mcp_tool.py` 告警与 `mcp_servers` 根键建议上游 |
| **P-015** | `pyproject.toml`, `.github/workflows/release-runtime.yml`, `docs/RUNTIME_RELEASES.md`, `uv.lock` | 新增 `cn-desktop` 聚合 extra，把冻结 runtime 暴露的所有后端预打包（`web`、`anthropic`、`mcp`、`feishu`、`dingtalk`、`wecom`，以及微信用的 `aiohttp`/`qrcode`/`cryptography`）。发布流程改为安装 `.[cn-desktop]`，收集各 IM SDK 子模块与元数据，新增"构建环境 import 冒烟"，并断言每个后端的 `dist-info` 出现在冻结产物中 | 桌面反馈：飞书/钉钉/企微/微信适配器因 SDK（`lark-oapi`、`dingtalk-stream` 等）从未被打包、且冻结环境无法懒安装而静默降级为"不可用"。根因同 P-014，推广到所有桌面后端 | 打包 CN 特有；不上游（上游不构建这些产物） |
| **P-016** | `tools/terminal_tool.py`, `tools/environments/local.py`, `tools/environments/proccess_pwsh.py`, `tools/environments/base.py`, `model_tools.py`, `tests/tools/test_terminal_dynamic_description.py` | PowerShell 原生执行：Windows 上使用 `pwsh.exe`（PS7）作为主 shell，`powershell.exe`（PS5.1）作为回退，支持完整生命周期管理；删除 Git Bash 自动安装。增加运行时自适应的 terminal 工具描述和 shell 指纹缓存键。增加 pwsh_transform 警告传递 | Windows 上 agent 原本硬编码为 Git Bash；PowerShell 启动更快，路径处理更原生。Git for Windows 自动安装已删除。静态 terminal 描述中的 Linux 命令引用在 PS 下会产生误导 | 被 P-019 取代 |
| **P-019** | `tools/environments/local.py`, `tools/terminal_tool.py`, `agent/prompt_builder.py`, `cli.py`, `apps/desktop/electron/main.cjs`, `scripts/install.ps1`, `hermes_cli/uninstall.py`, `cron/scheduler.py`, `tools/environments/base.py`, `tools/file_operations.py`, `tools/browser_tool.py`, `tests/*`, `website/docs/*`, `FORK_NOTES*.md`、`hermes_bootstrap.py`、`tools/environments/windows_env.py`、`scripts/check-windows-footguns.py`、相关测试、`scripts/verify_windows_utf8.py` | 完成 Git-Bash→PowerShell 迁移：移除全部 Git Bash 发现逻辑（7策略 `_find_bash`）、WSL 启动器过滤和 `HERMES_GIT_BASH_PATH`。Windows 上仅使用 **Windows PowerShell 5.1**（`powershell.exe`，每套 Windows 10/11 自带）——无需 `pwsh.exe`、无需下载、无需安装。`HERMES_SHELL_TYPE=bash` 在 Windows 上抛 RuntimeError。重命名多个函数和变量。`pwsh_transform` 改为始终开启。替换桌面端 `findGitBash` 为 `findPowerShell`。移除安装脚本的 Git Bash 安装逻辑。清理所有 Git Bash 注释、文档和测试；并新增 PowerShell UTF-8 编码加固（`ps_with_utf8()` 辅助函数、控制台 CP_UTF8 引导、仅对 PowerShell 子进程保留 `encoding='utf-8'`）。 | `powershell.exe` (5.1) 每套 Windows 10/11 自带——零安装零下载。比 Git Bash 启动更快，路径处理原生，无需 POSIX 翻译。删除约 400 行死代码（7 策略 bash 发现、WSL 启动器过滤、PortableGit 自动安装）。Agent 在 Windows 上拥有唯一、可预测、始终可用的 shell。P-016 的 `pwsh.exe` 探测是不必要的复杂度——5.1 全覆盖 | 取代 P-016；应上游化 |
| **P-017** | `agent/tool_dedup.py`, `agent/agent_init.py`, `agent/conversation_loop.py`, `agent/tool_executor.py` | 增加 `ToolDedupTracker`，在跨 API 迭代间检测重复的相同工具调用，并在重复次数达到 3、5、8 次时注入逐级升级的 `<system-reminder>` 提示以打破无限循环 | Agent 在处理复杂任务时可能陷入无限循环，反复调用相同工具和参数——现有同轮去重 `_deduplicate_tool_calls` 无法检测跨迭代模式 | 内部机制——解决行为健壮性缺口；机制通用，但集成点与 fork 架构耦合 |
| **P-018** | `agent/agent_init.py`, `tests/run_agent/test_init_fallback_on_exhausted_pool.py` | 增加 `_api_key_required` 辅助函数，并在 OpenAI / Anthropic SDK 客户端构造前加入空 key 保护；当 api_key 为空且 provider 需要密钥时，抛出 `RuntimeError: no API key (param empty, env vars unset)` | 此前空 key（参数为空且环境变量未设置）会触发底层 SDK 认证异常，在 TUI/gateway 后台线程中表现为 panic 且无堆栈信息 | 建议上游 |
| **P-020** | `tools/environments/windows_env.py`（新建）, `tools/environments/local.py`, `hermes_cli/claw.py`, `hermes_cli/managed_uv.py`, `hermes_cli/gateway.py`, `hermes_cli/dep_ensure.py`, `hermes_cli/clipboard.py`, `skills/creative/comfyui/scripts/hardware_check.py` | 新增 `refresh_env_from_registry()` 函数，从 Windows 注册表（HKLM + HKCU）刷新 `os.environ["PATH"]` 和 `os.environ["PATHEXT"]`，在每次 PowerShell 子进程调用前执行，使进程启动后安装的工具（如 WinGet、MSI）可被发现。参考 `kimi-cli/src/kimi_cli/utils/environment.py` 的实现。非 Windows 平台无操作。 | 如果不刷新，agent 无法发现进程启动后安装的二进制文件（例如通过 WinGet 安装的工具）— `shutil.which` 和 `subprocess.Popen` 只能看到进程创建时捕获的 PATH。当 agent 在会话中安装自己的依赖（node、uv 等）时尤其痛苦。 | 建议上游 |
| **P-021** | `gateway/run.py`、`cron/scheduler.py`、`cron/jobs.py`、`hermes_time.py` | 四项 cron "静默停摆" 根因修复：(1) `_start_cron_ticker` 初始化包在 try/except 中，防止 daemon 线程静默死亡；(2) 僵尸 `.tick.lock` 自动清理——锁文件 mtime 超过 `lock_stale_seconds`（默认 120s）则删除；(3) `_validate_cron_startup()` 启动前校验 `jobs.json` 可解析性；(4) `_ensure_aware` 按配置时区解释无时区时间戳；修复 `hermes_time.py` 缺失的 `def now()`；每次 tick 调用 `reset_cache()` 使时区配置热生效。 | `jobs.json` 损坏 → ticker 线程崩溃 → daemon 静默死亡。僵尸 `.tick.lock` → 所有后续 tick 永久阻塞。ticker 初始化 `ImportError` → 线程零日志死亡。服务器时区 ≠ 配置时区 → 调度时间静默偏移。 | 建议上游 |
| **P-022** | `agent/agent_runtime_helpers.py`、`tests/run_agent/test_agent_guardrails.py`、`tests/run_agent/test_session_meta_filtering.py` | 在 `sanitize_api_messages` 中增加空内容过滤：丢弃 `content` 为 `""` 且无有效载荷的 `assistant`/`user`/`function` 消息；保留仍携带 `tool_calls`、`codex_reasoning_items`、`codex_message_items` 或 `reasoning_content` 的 `assistant` 消息。 | MiMo v2.5 及严格的 OpenAI 兼容网关会拒收空 `content` 消息（HTTP 400 / "text is not set"）。长会话（如飞书 3-13h）在上下文压缩/截断后可能留下这类消息。 | 建议上游 |
| **P-023** | `tui_gateway/server.py` | 网关回合执行器现在会把"漏接"的 `/steer` 作为下一轮用户输入投递。`run_conversation()` 只能把 steer 注入到*后续*的工具结果里；落在最后一个工具批次之后（或纯文本回合）的 steer 会以 `result["pending_steer"]` 返回。`cli.py` 会重新投递它，但网关此前直接丢弃——导致桌面端（运行时输入行为默认 "引导/steer"）发出的 steer 静默丢失。仿照已有的 `goal_followup` 链路：在 `finally` 释放 `session["running"]` 后，用 steer 文本发起一次嵌套 `_run_prompt_submit`（受 `running` 保护，真实用户输入优先；优先级高于 goal 续跑）。 | 桌面端反馈（#193）："引导功能不好用……等到任务执行完，我引导的东西也没插入进去"——晚到的 steer 被 `agent.steer()` 接受却从未生效，因为网关忽略了 `pending_steer`。 | 建议上游（通用可靠性修复） |

## 发布和维护支撑

这些不是运行时行为补丁，但属于 fork 维护能力：

| 范围 | 目标文件 | 做了什么 |
|---|---|---|
| 上游同步 | `scripts/sync-upstream.sh`, `.github/workflows/upstream-watch.yml`, `MAINTAINING.md` | 固化“临时同步分支 + PR 回 main”的同步流程，避免直接在 `main` 合上游 |
| managed runtime | `.github/workflows/release-runtime.yml`, `scripts/sign_runtime_manifest.py`, `docs/RUNTIME_RELEASES.md` | 构建 PyInstaller runtime，签名 manifest，并发布给 desktop 下载 |

## 补丁详情

### P-001：provider dict/list 不一致修复

这个补丁已被上游等价修复，本 fork 不再携带。当前 `_apply_model_switch` 中 `user_provs = cfg.get("providers")` 已能处理所需配置形态。

---

### P-002：`POST /api/upload`

**现象**：desktop 或 web composer 拖拽上传文件时，请求 `/api/upload` 返回 404。

**原因**：上游曾经加入过 dashboard 附件上传接口，后来又移除；desktop 仍需要这个能力。

**改动**：增加 FastAPI handler，接收 multipart `file` 和 `session_id`，写入 `~/.hermes/sessions/<id>/attachments/`，返回 `{ok, filename, path, size, mime_type}`。文件名冲突复用上游 `_next_unique_path`。

**风险和约束**：
- 走 dashboard session token 鉴权。
- 只写入指定 session 的 attachments 目录。
- 不覆盖已有文件。
- 不做会触发执行语义的 content-type 处理。

**是否上游**：可以考虑，但需要先确认上游当初移除该接口的原因。

---

### P-003：去掉 `/api/ws` 的 embedded TUI 门禁

**现象**：desktop 运行 `hermes dashboard --no-open` 时，`/api/ws` upgrade 会被关闭，聊天不可用。

**原因**：上游的 `_DASHBOARD_EMBEDDED_CHAT_ENABLED` 只在 `hermes dashboard --tui` 模式下打开。desktop 是 headless dashboard + 独立 UI，不会启用这个标志。

**改动**：移除 `/api/ws` 对 `_DASHBOARD_EMBEDDED_CHAT_ENABLED` 的检查。接口仍受 session token 和 loopback host 约束。

**风险和约束**：持有同源 session token 的 Web UI 可以在非 `--tui` 模式访问 gateway。这和 `/api/pty`、`/api/pub`、`/api/events` 的安全边界一致。

**是否上游**：建议上游。当前门禁会阻断合法的外部 Web UI 用法。

**v0.16.0 同步更新**：上游 #38591 现在默认开启 embedded chat(`_DASHBOARD_EMBEDDED_CHAT_ENABLED` 默认 `True`)并移除了 dashboard 的 `--tui` 标志,原始现象默认不再出现。fork 仍保留 `/api/ws` 上的显式去门禁,以便即使将来 embedded chat 被关闭,gateway RPC 通道(v2 web UI / 桌面端使用)仍可达。

---

### P-004：`GET /api/fs/list`

**现象**：web 工作区选择器没有目录浏览能力，只能退化为 `window.prompt()` 让用户输入路径。

**原因**：纯 Web UI 无法调用系统文件夹选择对话框；上游 dashboard 也没有文件夹浏览 API。

**改动**：增加 `GET /api/fs/list?path=<dir>&include_hidden=<bool>`，返回 `{path, parent, home, entries: [{name, path, is_dir}]}`。

路径处理规则：
- 支持 `~` 展开。
- 使用 `Path.resolve(strict=False)` 折叠 `..`。
- 限制在用户 home 子树内。
- 响应最多 5000 项。
- 默认隐藏隐藏文件。

**风险和约束**：这是目录枚举接口，因此必须保留 token 鉴权、home 子树限制和大目录上限。

**是否上游**：取决于上游是否希望 browser-only Web UI 成为一等场景。

---

### P-005：`GET /api/mcp-servers`

**现象**：desktop 健康检查需要知道 MCP server 总数和启用数，但不应读取完整 MCP 配置。

**原因**：MCP 配置中的 `command`、`args`、`env` 可能包含敏感信息。上游没有只读摘要接口。

**改动**：返回 `{summary: {total, enabled}, servers: [{name, enabled}]}`，刻意不返回 `command`、`args`、`env`。

**风险和约束**：只读摘要，风险低。必须继续避免暴露密钥和启动参数。

**是否上游**：建议上游，其他 dashboard frontend 也会用到。

---

### P-006：CN provider 的 `OPTIONAL_ENV_VARS`

**现象**：desktop 模型设置页列出 CN provider，但 env 面板没有对应 `*_API_KEY` 输入项。

**原因**：上游 metadata 主要覆盖 OpenAI、Anthropic、Google、DeepSeek 等全球 provider。

**改动**：为 ARK、QIANFAN、HUNYUAN、SILICONFLOW、MODELSCOPE、AI302、COMPSHARE 等注册 provider 类环境变量，并补充中文说明和官方文档链接。

**风险和约束**：设置页会多出一批高级 provider 配置项，不改变现有解析逻辑。

**是否上游**：部分 provider 也许可以单独上游，但整体是 CN 专属。

---

### P-007：gateway WS dispatch 异常可观测性

**现象**：前端偶发只显示 “WebSocket closed”，后端没有足够上下文定位 dispatch 异常。

**原因**：`tui_gateway/ws.py` 中 dispatch/write 发生异常时会跳出循环并关闭连接，客户端只能看到连接断开。

**改动**：
- 包裹 `server.dispatch` 和 `transport.write_async`。
- 将 traceback 写入 `~/.hermes/logs/dispatch_exceptions.log`。
- 返回 JSON-RPC error（code `-32000`）。
- 保持连接继续可用。

**风险和约束**：异常日志会增长；客户端应把 `-32000` 视为通用服务端错误。

**是否上游**：强烈建议。正常路径行为不变，主要提升诊断能力。

---

### P-008：`GET/PUT /api/profiles/active`

**现象**：desktop profile 切换器需要读取和设置 sticky active profile。

**原因**：上游有 profile 列表、创建、删除、重命名、SOUL 读写，但没有对 `~/.hermes/active_profile` 的 HTTP getter/setter。

**改动**：
- `GET /api/profiles/active` 返回 `{name}`，文件不存在时返回 `default`。
- `PUT /api/profiles/active` 接收 `{name}` 并写入 sticky 设置。

**风险和约束**：该接口只影响下次启动默认 profile，不改变当前 dashboard 进程正在使用的 `HERMES_HOME`。desktop 需要提示用户重启。

**是否上游**：建议上游，属于明显的 API 对称性缺口。

---

### P-009：SSE+POST gateway transport —— **已弃用**

> **弃用说明（2026-06-09）**：桌面端自 0.4 起已切换到 runtime 原生的
> `/api/ws` JSON-RPC WebSocket（与官方 Electron 桌面端 `apps/desktop` 同一
> transport）——SSE+POST 路径无心跳、每个 RPC 一次 HTTP 往返、异步 ack 拆分
> 导致在途回合脆弱。**这两个端点必须保留**到 ≤ 0.3.x 旧外壳 EOL：Tauri 外壳
> 没有自更新，而 runtime 会在其下热更新，新 runtime 必须继续服务旧外壳。
> `/api/v2/events` 每次连接会打一条弃用日志，便于在移除前从 runtime 日志
> 量化残留用量。

**现象**：desktop 需要稳定、浏览器友好的流式 transport。只依赖 `/api/ws` 时，桌面壳和网络环境下的故障更难诊断。

**原因**：上游 gateway 主要通过 WebSocket 暴露。desktop 希望服务端到客户端走 EventSource，客户端到服务端走普通 HTTP POST。

**改动**：
- 增加 `GET /api/v2/events` 推送 SSE frame。
- 增加 `POST /api/v2/rpc` 发送 gateway JSON-RPC 请求。
- 增加 `tui_gateway/sse.py` transport 实现。

**风险和约束**：新增一个经过鉴权的 gateway transport 面。鉴权应继续复用 dashboard session token。

**是否上游**：可以考虑。它对 browser-hosted dashboard 和桌面壳有价值，但会扩大上游需要维护的 transport 矩阵。

---

### P-010：`LONGCAT_API_KEY`

**现象**：CN 模型设置包含 LongCat，但 env metadata 没有 `LONGCAT_API_KEY`。

**原因**：上游 provider metadata 未覆盖 LongCat。

**改动**：将 `LONGCAT_API_KEY` 加入 `OPTIONAL_ENV_VARS`。

**风险和约束**：设置页多一个 provider credential 输入项。

**是否上游**：只有在上游正式支持 LongCat 时才适合提交。

---

### P-011：模型过滤和 provider probe

**现象**：desktop 需要按 provider slug 过滤模型选择器，并在不启动完整 agent turn 的情况下轻量探测 provider。

**原因**：上游 `model.options` 返回较宽泛的选项；没有专用的 provider 探测 RPC。

**改动**：
- `model.options` 增加 `slug_filter`。
- 增加 `provider.probe` gateway RPC。

**风险和约束**：`provider.probe` 不应返回密钥、原始配置或敏感错误细节。

**是否上游**：可以考虑，但需要先审定 probe 的返回结构和错误语义。

---

### P-012：`_model_flow_anthropic()` 支持可选自定义 `base_url`

**现象**：在交互式添加 Anthropic 模型时，代码无条件执行 `model.pop("base_url", None)`，导致任何预配置或期望的自定义 `base_url` 被静默丢弃。

**原因**：`_model_flow_anthropic()` 原本假设所有 Anthropic 请求都应走官方 `https://api.anthropic.com`，未考虑使用兼容代理、OpenRouter 或私有端点的场景。

**改动**：
- 移除无条件的 `model.pop("base_url", None)`。
- 在模型选择后增加交互式提示，显示当前 `base_url`（默认 `https://api.anthropic.com`）。
- 用户输入自定义地址则保存到 `model["base_url"]`。
- 用户直接回车则保留已有 `base_url`；仅在原本不存在时才将其移除，让运行时回退到硬编码的官方地址。

**风险和约束**：无。`runtime_provider.py` 对 `anthropic` provider 已使用 `model_cfg.get("base_url")` 读取配置，无需额外运行时改动。

**是否上游**：建议上游。该改动向后兼容，且能支持合法的第三方 Anthropic 兼容端点场景。

---

### P-013：`handle_function_call` 自动修复工具参数键名

**现象**：LLM 发起工具调用时经常使用同义词或拼写错误的参数名（如 `file` 代替 `path`、`cmd` 代替 `command`、`backgroud` 代替 `background`），导致工具层返回 "unknown parameter" 或直接失败。

**原因**：Hermes 的 JSON Schema 较为严格，LLM 对字段名的漂移会直接透传给工具 handler，而 handler 通常不认识这些别名。

**改动**：
- 在 `model_tools.py` 中引入 `repair_tool_arg_keys()` 与 `_repair_nested_args()`。
- 定义全局别名表 `TOOL_FIELD_ALIASES`，覆盖通用、文件、Shell、Web、任务、待办、输入、搜索、记忆、定时任务、技能等多类参数名。
- 定义 `TOOL_SPECIFIC_ALIASES` 实现工具级覆盖（如 `delegate_task` 将 `task` 映射到 `goal` 而非全局的 `prompt`；`cronjob` 将 `command` 映射到 `action`）。
- 当别名表未命中时，使用 `difflib.get_close_matches` 对拼写错误进行模糊匹配。
- 根据 schema 中的 `properties` 与 `items` 定义，递归修复嵌套对象和对象数组内部的键名。
- 提供可选回调钩子 `set_arg_repair_callback`，供外部系统（TUI、ACP）在顶层键名被修复时得到通知。
- 在 `handle_function_call()` 中于 `coerce_tool_args()` 之前调用修复逻辑，因此修复后的键仍会正常经历类型强制转换。
- 新增完整测试 `tests/run_agent/test_repair_tool_arg_keys.py`。

**风险和约束**：极低。该函数是纯键名映射变换，无法识别的键保持原样；模糊匹配仅对长度 ≥4 且相似度 ≥0.75–0.80 的键生效，随机字段不会被误改名。

**是否上游**：建议上游。这是与平台、provider 无关的通用健壮性提升，对所有 Hermes 部署都有价值。

---

### P-014：冻结 desktop runtime 缺失原生 MCP 客户端

**现象**（issue #16）：用户在 `~/.hermes/config.yaml` 正确配置了 `mcp_servers`，MCP server 脚本独立运行正常，但 CN Desktop agent 启动后从不连接它——`agent.log` 中没有任何 MCP 发现/连接日志，工具列表里也没有 `mcp_*` 工具。在宿主机执行 `pip install mcp` 也无济于事。

**根因**：原生 MCP 客户端其实已完整实现（`tools/mcp_tool.py`、`discover_mcp_tools()`），但其 SDK 是只存在于 `[mcp]` extra 的可选依赖。runtime 发布流程当时只安装 `.[web,anthropic]`，因此冻结后的 PyInstaller 产物**没有**打进 `mcp` 包。于是冻结 runtime 内 `_MCP_AVAILABLE` 为 `False`，`discover_mcp_tools()` 仅以 `debug` 级别记录后返回 `[]`——在默认 INFO 日志级别下完全不可见。宿主机的 `pip install mcp` 无关紧要，因为冻结 runtime 自带独立解释器和依赖。

**改动内容**：
- `release-runtime.yml`：把 `mcp` SDK 打进产物（安装入口后并入 `cn-desktop` extra，见 P-015），PyInstaller 增加 `--collect-submodules mcp` 与 `--copy-metadata mcp`，并扩展校验步骤——若缺少 `mcp-*.dist-info` 则直接让构建失败（防止再次悄悄回归）。
- `tools/mcp_tool.py`：当已配置 `mcp_servers` 但 SDK 不可用时，`discover_mcp_tools()` 改为输出一次 `WARNING`（“mcp_servers are configured but the MCP SDK is not available …”），而非静默的 debug。未配置 MCP 的用户仍走安静的 debug 分支。
- `hermes_cli/config.py`：把 `mcp_servers` 加入 `_KNOWN_ROOT_KEYS`，让根级 schema 文档保持准确。
- `docs/RUNTIME_RELEASES.md`：将 MCP 列为 runtime 必备依赖，并更新手动 dry-run 命令。
- `tests/tools/test_mcp_tool.py`：覆盖“已配置则告警 / 未配置则安静 / 仅告警一次”三种行为。

**风险和约束**：冻结 runtime 体积增加 `mcp` SDK 及其传递依赖（`anyio`/`httpx-sse`/`sse-starlette`，均已随 `web`/`anthropic` 存在）。对已包含 `[mcp]` extra 的源码安装无行为变化。

**是否上游**：打包改动是 CN runtime 特有（上游不构建这些 PyInstaller 产物）；`mcp_tool.py` 的诊断日志与 `mcp_servers` 根键属于通用改进，值得上游。

---

### P-015：冻结 desktop runtime 缺失 IM 平台后端

**现象**：桌面用户正确填了飞书 App ID/Secret 到 `.env`，在 `config.yaml` 加了飞书平台，网关进程也在跑——但就是连不上飞书；打包应用内 `lark-oapi`"无法安装"。钉钉、企业微信、微信同理。

**根因**：和 P-014 完全同源，只是范围更广。IM 适配器（`gateway/platforms/feishu.py`、`dingtalk.py`、`wecom*.py`、`weixin.py`）都在 `try/except` 里导入 SDK，包缺失时降级为 `*_AVAILABLE = False`。这些 SDK 只存在于可选 extra（`[feishu]`→`lark-oapi`、`[dingtalk]`→`dingtalk-stream`+`alibabacloud-*`、`[wecom]`→`defusedxml`；微信**没有** extra，需要 `aiohttp`/`qrcode`/`cryptography`）。`[all]` 的策略故意排除它们，因为它们能通过 `tools/lazy_deps.py` 懒安装——但**冻结的 PyInstaller 二进制里懒安装根本跑不了**（没有可用 pip），而 desktop runtime 当时只装了 `.[web,anthropic,mcp]`，于是一个都没带。用户在宿主机执行的 `pip install lark-oapi` 写进的是系统 Python，冻结 runtime 从不使用。

**改动内容**：
- `pyproject.toml`：新增 `cn-desktop` 聚合 extra，列出冻结 runtime 必须预打包的所有后端——`web`、`anthropic`、`mcp`、`feishu`、`dingtalk`、`wecom`，外加微信用的 `aiohttp`/`qrcode`/`cryptography`（pin 与现有 extra 对齐）。这是"桌面端打包什么"的单一事实来源，刻意区别于 `[all]` 的懒安装策略。
- `release-runtime.yml`：安装 `.[cn-desktop]`；为 `lark_oapi`、`dingtalk_stream`、`alibabacloud_dingtalk`（+`alibabacloud_tea_openapi`/`alibabacloud_tea_util`）、`aiohttp`、`qrcode` 增加 `--collect-submodules`/`--copy-metadata`；新增**构建环境 import 冒烟**——逐个 import 适配器并断言其 `*_AVAILABLE` 为 True（缺依赖立即失败）；并把校验步骤推广为断言每个打包后端的 `dist-info` 都在冻结产物里。
- `docs/RUNTIME_RELEASES.md`：把 `cn-desktop` extra 记录为"以后新增桌面后端"的入口，并标注 `alibabacloud_*` 收集较脆（首次发版需对真实钉钉机器人做连通冒烟）。
- `uv.lock`：为新 extra 重新生成（`uv lock --check` 通过）。

**风险和约束**：冻结 runtime 体积增加 IM SDK 及其传递依赖（尤其是纯 Python 的 `alibabacloud_*` 链）。它们都是纯 Python、有跨平台 wheel/sdist——不像 `matrix` 的 `python-olm` 需要 C 工具链，那个仍刻意排除。对源码安装无影响。

**是否上游**：否。上游不构建这些 PyInstaller 产物，`cn-desktop` extra 与打包均为 CN runtime 特有。

---

### P-016：PowerShell 原生执行 + 运行时自适应终端工具描述

> **由 P-019 更新**：P-019 完成了迁移，移除了所有剩余 Git Bash 发现逻辑，专注于仅使用 **Windows PowerShell 5.1**（`powershell.exe`）。详见下方 P-019。

**现象**：Windows 上 agent 硬编码使用 Git Bash。PowerShell 启动更快（`-NoProfile`），原生处理 Windows 路径（无需 `/c/foo` 翻译）。此外，terminal 工具描述包含 Linux/bash 命令引用，在原生 PS 中不存在。

**原因**：上游 `LocalEnvironment` 只支持 bash。

**改动内容**：

1. **`tools/environments/local.py`** — 新增 `_resolve_shell()`：Windows 上检测 `pwsh.exe`（PS7）优先，回退到 `powershell.exe`（PS5.1）或 Git Bash。新增 `_run_pwsh()`、`_wrap_command_pwsh()`，覆写 `init_session()`、`_run_bash()`、`_wrap_command()`。支持 `HERMES_SHELL_TYPE` 和 `HERMES_PWSH_PATH`。

2. **`tools/terminal_tool.py`** — 动态描述：`_detect_shell_for_description()` + `_build_dynamic_terminal_description()`，将 Linux/bash 命令引用替换为 PS cmdlet。

3. **`model_tools.py`** — 将 `_shell_fp` 加入 `get_tool_definitions()` 缓存键。

4. **`tools/environments/proccess_pwsh.py`** — `pwsh_transform()` 将 PS7+ 语法（`?:`、`??`、`&&`、`||`、`?.`、`?[`）降级为 PS5.1 兼容的 `if/else`，带警告传递。

**风险和约束**：Windows 上 terminal 命令在 PS 中执行。Git Bash 自动安装已移除，但 Python 层 bash 回退（`_find_bash()`）仍保留为 7 策略发现链。

**是否上游**：建议上游——被 P-019 取代并完成迁移。

---

### P-019：完成 Git-Bash→PowerShell 迁移（仅 Windows PowerShell 5.1）

**现象**：P-016 为代码库增加了 PowerShell 支持，但留下了混合状态：`pwsh.exe`（PS7）被优先探测，`powershell.exe`（PS5.1）作为回退，而 7 策略 `_find_bash()` Git Bash 发现链（环境覆盖 → PortableGit → git.exe 推导 → 注册表 → PATH → 常见路径 → 自动安装）仍然存在。`HERMES_GIT_BASH_PATH`、`HERMES_PWSH_PATH` 和 `_install_git` 导入（不存在的模块）都是死代码。

**原因**：P-016 专注于将 PowerShell 添加为主 shell，但未完全移除 Git Bash 机制。`pwsh.exe`（PS7）的要求是不必要的——Windows PowerShell 5.1（`powershell.exe`）随每套 Windows 10/11 系统自带，始终可用。

**改动内容**：

1. **`tools/environments/local.py`** — 核心 shell 解析（约 ~400 行删除）：移除 `_find_bash()`，替换为最小化的 `_find_bash_posix()`。移除 `_is_windows_wsl_launcher()`。`_find_pwsh_simple` → `_find_powershell()`。重写 `_resolve_shell()`：Windows 上始终返回 `("powershell", path)`。`HERMES_SHELL_TYPE=bash` 在 Windows 上抛 `RuntimeError`。函数重命名：`_run_pwsh` → `_run_powershell`，`_wrap_command_pwsh` → `_wrap_command_powershell`。`pwsh_transform` 改为始终开启。所有 `"pwsh"` → `"powershell"`。

2. **`tools/terminal_tool.py`** — 移除 "Windows Git Bash" 描述分支。简化 `_detect_shell_for_description()`。

3. **`agent/prompt_builder.py`** — `_WINDOWS_BASH_SHELL_HINT` → `_WINDOWS_POWERSHELL_SHELL_HINT`。

4. **`cli.py`** — `_normalize_git_bash_path` → `_normalize_msys_path`。

5. **`apps/desktop/electron/main.cjs`** — `findGitBash()` → `findPowerShell()`。更新预检。

6. **`scripts/install.ps1`** — 移除 `Install-Git` bash 发现 + `Set-GitBashEnvVar`（约 210 行）。简化 `Stage-Git`。增加 `powershell.exe` 防御性检查。

7. **`hermes_cli/uninstall.py`** — 移除 `HERMES_GIT_BASH_PATH`。

8. **`cron/scheduler.py`** — 更新 `.sh`/`.bash` 错误消息。

9. **注释清理**：`base.py`、`file_operations.py`、`browser_tool.py`。

10. **测试**：更新 4 个测试文件。

11. **文档**：更新 3 个英文文档页面。

12. **PowerShell UTF-8 编码加固** —— 让 Windows 上 PowerShell 子进程输出按 UTF-8 解码：
    - 在 `tools/environments/windows_env.py` 中新增 `ps_with_utf8()` 辅助函数，为 PowerShell 命令字符串前置 `[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; $OutputEncoding=[System.Text.Encoding]::UTF8;`。幂等，非 Windows 平台无操作。
    - 在 `tools/environments/local.py` 中于 `pwsh_transform()` 之后调用 `ps_with_utf8()`。
    - 仅对 PowerShell 子进程调用保留 `encoding="utf-8", errors="replace"`：`hermes_cli/claw.py`、`hermes_cli/clipboard.py`、`hermes_cli/gateway.py`、`hermes_cli/managed_uv.py`。
    - `hermes_bootstrap.py` 将 Windows 控制台输入/输出代码页设为 CP_UTF8（65001），并新增 `HERMES_DISABLE_WINDOWS_UTF8=1` 逃逸开关。
    - 撤销所有非 PowerShell 子进程上的 `encoding="utf-8"` 添加（tasklist、ssh、docker、ffmpeg、singularity、ripgrep、termux、comfyui auto-fix、check-windows-footguns 中的 git 辅助，以及若干测试）。
    - 新增测试：`tests/tools/test_clipboard.py::TestClipboardPowershellEncoding`、`tests/tools/test_local_pwsh_warnings.py::TestRunPowershellUtf8Encoding` / `TestPwshTransformAndUtf8Compose`、`tests/tools/test_windows_encoding.py`、`scripts/verify_windows_utf8.py`。

**为什么需要**：`powershell.exe` (5.1) 随每套 Windows 10/11 系统自带——零安装、零下载。比 Git Bash 启动更快，路径处理原生，避免 POSIX 翻译开销。删除约 400 行死代码。Agent 在 Windows 上拥有唯一、可预测、始终可用的 shell。P-016 的 `pwsh.exe`（PS7）探测是不必要的复杂度——5.1 全覆盖。

**风险和约束**：`HERMES_SHELL_TYPE=bash` 现在在 Windows 上抛清晰的 `RuntimeError`。`HERMES_PWSH_PATH` 和 `HERMES_GIT_BASH_PATH` 环境变量不再被识别。所有命令无条件经过 `pwsh_transform`。PowerShell 命令现在能可靠地往返非 ASCII 输出（中文、emoji、带重音符号字符）；非 PowerShell 子进程仍使用系统 locale，这是 fork 有意保持的保守范围。

**是否上游**：建议上游。完成 P-016 开始的迁移，使 Hermes 成为零依赖的 Windows 程序。

---

### P-017：跨迭代重复工具调用检测（无限循环断路器）

**现象**：在复杂任务（长时间构建、多步骤重构）中，agent 有时会陷入无限循环，跨连续 API 迭代反复调用相同的工具和相同参数——例如反复读取同一文件，或使用相同命令反复调用 `run`。现有的 `_deduplicate_tool_calls()` 仅移除**同一次**工具批次中的精确重复，完全无法检测跨迭代重复。

**根因**：此前没有跨步骤去重机制。每个 API 迭代的工具结果进入下一次 LLM 调用时，对之前尝试过什么完全没有历史感知。

**改动内容**：

1. **`agent/tool_dedup.py`** — 新增 `ToolDedupTracker` 类模块：
   - 通过 `_canonical_tool_arguments()` 对工具调用键规范化（字典递归排序、回退到 `str()`）。
   - 跟踪 `_seen_call_keys`（所有跨步骤见过的调用）和 `_consecutive_key`/`_consecutive_count`（连续调用计数）。
   - `begin_step(previous_calls, step_no, turn_id)`：从上一步的工具调用结果中植入状态。
   - `end_step()`：返回本步的调用列表供下一次迭代使用，并更新连续计数。
   - `check_and_register(tool_name, arguments)`：在工具执行期间调用；若调用键在前序步骤中已出现过，则在重复计数达到 3、5、8 时返回逐步升级的提示文本。
   - 逐级提示：计数 3 时温和提醒（`<system-reminder>`：“你在重复完全相同工具调用…”）。计数 5 和 8 时更强提示，明确给出工具名、重复次数和参数。

2. **`agent/agent_init.py`** — 在 `AIAgent` 实例上初始化 `_tool_dedup_tracker`。

3. **`agent/conversation_loop.py`** — 步骤生命周期：
   - 每次 API 调用前：`begin_step()` 从上一次迭代的调用结果植入跨步骤状态。
   - 所有工具结果收集完成后：`end_step()` 捕获本次迭代的调用供下一次使用。

4. **`agent/tool_executor.py`** — 去重检查注入：
   - 在 `execute_tool_calls_concurrent()` 中：每次工具执行后调用 `check_and_register()`，将提示文本追加到结果中。
   - 在 `execute_tool_calls_sequential()` 中：相同模式。

**风险和约束**：
- 触发去重时，工具结果可能增加数百字符（`<system-reminder>` 文本）。
- LLM 可见提示文本，可能影响其下一步决策——这正是预期行为。
- 线程安全：`check_and_register()` 使用 `threading.Lock()` 保护并发执行路径中的共享状态。

**是否上游**：机制是通用的，但集成点（`agent_init.py`、`conversation_loop.py`、`tool_executor.py`）与 fork 的 agent 架构高度耦合。可作为通用可观测性钩子提出。

---

### P-018：`agent/agent_init.py` 空 API key 保护

**现象**：当 API key 为空（参数显式传入 `""`，环境变量未设置）时，agent 会以底层 OpenAI 或 Anthropic SDK 认证异常的形式 panic，而不是给出清晰可操作的错误提示。在 TUI/gateway 后台线程中，堆栈信息不会暴露给用户，看起来像静默崩溃。

**根因**：`init_agent()` 在将 `api_key` 交给 `_create_openai_client()` 或 `build_anthropic_client()` 之前，没有显式验证其非空。空字符串流入 SDK 构造函数后产生令人困惑的异常。

**改动内容**：
- 新增 `_api_key_required(provider, api_key, base_url)` 辅助函数，对真正不需要字面量密钥的 provider（Azure Entra ID callable token、`"aws-sdk"` / `"no-key-required"`、Bedrock）返回 `False`。
- 在 `anthropic_messages` 分支的 `build_anthropic_client()` 调用前插入保护。
- 在 `chat_completions` 分支的 `_create_openai_client()` 调用前插入保护。
- 两个保护都在 key 为空且 provider 需要密钥时抛出 `RuntimeError("no API key (param empty, env vars unset)")`。
- 新增两个 pytest 用例分别覆盖 `chat_completions` 和 `anthropic_messages` 的空 key 路径。

**风险和约束**：对真正不需要密钥的 provider（本地端点 `"no-key-required"`、Bedrock、Azure Entra ID）无影响。fallback 循环（`fallback_model` / `fallback_providers`）仍在保护之前执行。

**是否上游**：建议上游。改动纯增量、与 provider 无关，能同时改善 CLI、TUI、gateway 和直接 `AIAgent()` 调用的用户体验。

---

### P-021：Cron 调度器可靠性修复 — 防止静默停摆

**现象**：定时任务在默认日志级别下无任何错误提示就停止执行。Gateway 仍在运行且健康，但 `hermes cron list` 显示任务的 `next_run_at` 已过期却一直不触发。

**根因**：四个相互独立的故障模式：

1. **Daemon 线程静默死亡** — ticker 线程顶部的导入语句在 try/except 之外，`ImportError` 会直接杀死 daemon 线程且零日志。
2. **僵尸锁文件** — 进程被 `SIGKILL` 或内核 panic 后 `.tick.lock` 永不清理，后续进程永远获取不到锁。
3. **损坏的 `jobs.json`** — 首次 tick 中 `load_jobs()` 抛 `RuntimeError`，线程在产生任何输出前死亡。
4. **时区解释漂移** — 旧的无时区时间戳按系统本地时间解释，与配置时区不一致时所有调度时间静默偏移。

同时修复了 `hermes_time.py` 中 `def now():` 缺失的既有 bug。

**改动**：`gateway/run.py`（F-1/F-4）、`cron/scheduler.py`（F-3/F-7）、`cron/jobs.py`（F-5）、`hermes_time.py`（F-7）。详见英文版 Fork Notes。

**是否上游**：建议上游。通用可靠性修复，与平台和 provider 无关。

---

### P-022：`sanitize_api_messages` 空内容消息过滤

**现象**：长会话（如飞书 3-13h）在调用模型 API 时偶发 HTTP 400，错误信息如 MiMo 的 `"text is not set"` 或某些严格 OpenAI 兼容网关的空内容拒绝。出错的请求里包含 `content` 被压缩/截断为空字符串的 `assistant` 或 `user` 消息。

**根因**：部分 provider（MiMo v2.5、严格的 OpenAI 兼容网关）拒绝 `content` 为 `""` 且没有工具载荷的消息。Agent 的上下文压缩器会留下这类空消息；此前的预调用清理器只修复了孤儿 tool result 并去掉了 `session_meta` 角色消息，但没有清理空内容的 `assistant`/`user`/`function` 消息。

**改动**：

- 在 `sanitize_api_messages` 的孤儿修复之后新增一轮过滤：角色属于 `{assistant, user, function}`、`content` 严格等于 `""` 且没有有效载荷的消息会被丢弃。
- 可保留 `assistant` 消息的有效载荷包括：
  - `tool_calls`
  - `codex_reasoning_items`
  - `codex_message_items`
  - `reasoning_content`
- 这样 codex / DeepSeek reasoning 回放和工具调用链保持完整，同时剔除触发网关校验错误的空消息。
- `system` 消息故意保留不处理（各 provider 行为不同，避免误删）。
- 完全没有 `content` 键的消息也保留不动，以便需要时让 API 按自身规则报错，避免掩盖其他 bug。

**涉及文件**：
- `agent/agent_runtime_helpers.py` — 在 `sanitize_api_messages` 中加入空内容过滤。
- `tests/run_agent/test_agent_guardrails.py` — 新增 11 个聚焦回归测试，覆盖 assistant/user/function 空内容丢弃、带 tool calls / codex reasoning / reasoning content 时的保留、system 保留、多连续空消息丢弃、幂等性等。
- `tests/run_agent/test_session_meta_filtering.py` — 新增 `TestSanitizeApiMessagesEmptyContentFilter` 类，提供端到端回归测试，包括 MiMo "text is not set" 场景。

**副作用**：
- 重度压缩后到达 API 的消息数会略少，这是预期行为，因为这些消息本身没有可用内容。
- 如果上游调用方出于某种协议目的故意传入空 `assistant` 消息，现在除非携带上述识别载荷，否则会被丢弃。

**是否上游**：建议上游。过滤逻辑与 provider 无关，能防范一类真实的网关拒绝，且已有较全面的测试覆盖。

---

## Windows 兼容性补丁

以下补丁由 Maxwell Geng 贡献，用于提升 Windows 平台的一等支持体验，均可向上游提交。

### `282cfeeca` — 为 `shlex.split` 增加 `posix` 选项以兼容 Windows

**做了什么**：在代码库所有涉及到 `subprocess` 的 `shlex.split()` 调用中增加 `posix=os.name == "posix"` 参数，防止 Windows 路径中的反斜杠被误解析为转义字符。

**涉及文件**：
- `agent/copilot_acp_client.py`
- `agent/shell_hooks.py`
- `agent/subdirectory_hints.py`
- `cli.py`
- `gateway/run.py`
- `hermes_cli/auth.py`
- `hermes_cli/gateway_windows.py`
- `hermes_cli/memory_setup.py`
- `tools/transcription_tools.py`

**上游状态**：建议上游。纯 Windows bug 修复，POSIX 下行为无变化。

### `ada59ec36` — 修复 10 个在 Windows 上失败的测试，使其跨平台

**做了什么**：让 10 个测试用例在 Windows 上正确通过或优雅跳过：

| 测试 | 修复方式 |
|---|---|
| `test_make_run_env_appends_homebrew_on_minimal_path` | Windows 下跳过（POSIX PATH 注入在该平台被有意跳过）。 |
| `test_returns_root_when_only_root_exists` | Windows 下对 cwd 做 `os.path.normpath()`，使带正斜杠的路径能正确走到文件系统根目录。 |
| `test_close_stdin_allows_eof_driven_process_to_finish` | 用 `cat` 代替 `python3`；PTY 库缺失时跳过；winpty 传 `str`、ptyprocess 传 `bytes`。 |
| `test_popen_killed_when_thread_creation_fails` | 仅在 `os.getpgid` 存在时（POSIX）patch。 |
| `test_popen_killed_when_write_checkpoint_fails` | 仅在 `os.getpgid` 存在时（POSIX）patch。 |
| `test_kill_detached_session_uses_host_pid` | 直接 mock `_terminate_host_pid`，不再依赖内部 `psutil` 调用。 |
| `test_windows_does_not_call_psutil` | 增加 `pytest.importorskip("psutil")`。 |
| `test_posix_walks_tree_and_terminates_children_then_parent` | 增加 `pytest.importorskip("psutil")`。 |
| `test_posix_no_such_process_swallowed` | 增加 `pytest.importorskip("psutil")`。 |
| `test_posix_oserror_falls_back_to_os_kill` | 增加 `pytest.importorskip("psutil")`。 |

**涉及文件**：
- `tests/tools/test_local_env_blocklist.py`
- `tests/tools/test_process_registry.py`
- `tools/environments/local.py`
- `tools/process_registry.py`

**上游状态**：建议上游。扩展 CI 到 Windows 覆盖，不改变生产行为。

### `1a75a7672` — ~~Windows 下自动安装 Git-Bash，并将 Windows 风格命令转换为 POSIX 风格~~ **已删除**

**状态**：已移除。Git for Windows 自动安装与 Git Bash 回退支持已被删除，改为原生 PowerShell 执行（见 P-016）。以下文件已移除：
- `tools/environments/_install_git.py`
- `tools/environments/_process_bash_command.py`

Windows 平台现在要求使用 PowerShell 7（`pwsh`）或 Windows PowerShell（系统 PowerShell）。Shell 通过 `_find_pwsh` 解析，不再自动安装——PowerShell 属于 Windows 标准组件，默认已可用。
