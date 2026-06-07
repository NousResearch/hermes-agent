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
| **P-009** | `hermes_cli/web_server.py`, `tui_gateway/sse.py` | 增加 `/api/v2/events` SSE 和 `/api/v2/rpc` POST transport | desktop 默认使用 EventSource + POST，减少 WebSocket 兼容问题 | 可考虑上游 |
| **P-010** | `hermes_cli/config.py` | 注册 `LONGCAT_API_KEY` | CN 模型设置需要 LongCat 密钥入口 | CN 专属，除非上游支持 LongCat |
| **P-011** | `tui_gateway/server.py` | 给 `model.options` 增加 `slug_filter`，并增加 `provider.probe` RPC | desktop 需要过滤模型选择器，并轻量探测 provider 状态 | 可考虑上游 |
| **P-012** | `hermes_cli/main.py` | `_model_flow_anthropic()` 支持保留或自定义 `base_url`，不再无条件删除 | 使用 Anthropic 兼容代理或私有端点的用户需要在模型设置流程中保留自定义 `base_url` | 建议上游 |
| **P-013** | `model_tools.py`, `tests/run_agent/test_repair_tool_arg_keys.py` | 在 `handle_function_call` 中增加自动参数键修复：全局别名表、工具级覆盖、模糊匹配、嵌套对象/数组递归修复，以及可选回调通知 | LLM 经常把参数名写错（如 `file`→`path`、`cmd`→`command`），此前会直接报 "unknown parameter"；该补丁在不放宽 JSON Schema 的前提下提高工具调用的容错率 | 建议上游 |
| **P-014** | `.github/workflows/release-runtime.yml`, `tools/mcp_tool.py`, `hermes_cli/config.py`, `docs/RUNTIME_RELEASES.md`, `tests/tools/test_mcp_tool.py` | 把原生 MCP 客户端 SDK 打进冻结 runtime（安装入口后并入 `cn-desktop` extra，见 P-015；外加 `--collect-submodules/--copy-metadata mcp` + CI 断言 `mcp-*.dist-info` 存在）；并让 `discover_mcp_tools()` 在已配置 `mcp_servers` 但 SDK 缺失时输出一次 WARNING，而不是在 debug 级别静默跳过 | issue #16：desktop runtime 打包时缺少 `mcp` extra，导致 `_MCP_AVAILABLE=False`，已配置的 `mcp_servers` 不注册任何工具且 INFO 日志无任何提示。打包改动是 CN 特有，诊断日志与已知根键则是通用改进 | 打包改动 CN 特有；`mcp_tool.py` 告警与 `mcp_servers` 根键建议上游 |
| **P-015** | `pyproject.toml`, `.github/workflows/release-runtime.yml`, `docs/RUNTIME_RELEASES.md`, `uv.lock` | 新增 `cn-desktop` 聚合 extra，把冻结 runtime 暴露的所有后端预打包（`web`、`anthropic`、`mcp`、`feishu`、`dingtalk`、`wecom`，以及微信用的 `aiohttp`/`qrcode`/`cryptography`）。发布流程改为安装 `.[cn-desktop]`，收集各 IM SDK 子模块与元数据，新增"构建环境 import 冒烟"，并断言每个后端的 `dist-info` 出现在冻结产物中 | 桌面反馈：飞书/钉钉/企微/微信适配器因 SDK（`lark-oapi`、`dingtalk-stream` 等）从未被打包、且冻结环境无法懒安装而静默降级为"不可用"。根因同 P-014，推广到所有桌面后端 | 打包 CN 特有；不上游（上游不构建这些产物） |
| **P-016** | `tools/terminal_tool.py`, `tools/environments/local.py`, `model_tools.py`, `tests/tools/test_terminal_dynamic_description.py` | PowerShell (pwsh) 原生执行：Windows 上检测 pwsh 并将其作为主 shell，支持完整生命周期管理（spawn、wrap、init_session、cwd 跟踪）；增加运行时自适应的 terminal 工具描述，当激活 shell 为 pwsh 时自动将 Linux/bash 命令引用替换为 pwsh cmdlet；将 shell 指纹加入工具定义缓存键，使会话中期 pwsh 自动安装能刷新描述缓存 | Windows 上 agent 原本硬编码为 Git Bash；pwsh 启动更快（-NoProfile）、Windows 路径处理更原生、无需 POSIX 翻译。静态 `TERMINAL_TOOL_DESCRIPTION` 中包含的 Linux 命令引用（`cat/head/tail`、`grep/rg/find`、`echo/cat heredoc`）在 pwsh 下会产生误导 | 建议上游 |

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

### P-009：SSE+POST gateway transport

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
### P-014：PowerShell (pwsh) 原生执行 + 运行时自适应终端工具描述

**现象**：Windows 上 agent 硬编码使用 Git Bash。PowerShell 7 (pwsh) 启动更快（`-NoProfile` 跳过 profile 加载），原生处理 Windows 路径（无需 `/c/foo` 翻译），是现代 Windows 的默认 shell。此外，terminal 工具的静态 `TERMINAL_TOOL_DESCRIPTION` 包含 Linux/bash 命令引用（`cat/head/tail`、`grep/rg/find`、`echo/cat heredoc`、`Pipe git output to cat`），这些命令在原生 pwsh 中不存在——LLM 要么尝试执行失败，要么不知道应避开哪些 pwsh 等价命令。

**原因**：上游 `LocalEnvironment` 只支持 bash。terminal 工具描述是硬编码的静态字符串，假定 Linux 环境。

**改动内容**：

1. **`tools/environments/local.py`** — Shell 检测与 pwsh 执行：
   - 新增 `_resolve_shell()`：检测 pwsh vs bash。支持 `HERMES_SHELL_TYPE`（auto/pwsh/bash）和 `HERMES_PWSH_PATH` 环境变量覆盖。pwsh 不可用时回退到 Git Bash。复用已有的 `_find_pwsh`（含自动安装）来自 `tools/environments/_find_pwsh.py`。
   - `LocalEnvironment.__init__` 调用 `_resolve_shell()` 并保存 `self._shell_type` / `self._shell_path`。
   - 新增 `_run_pwsh()`：使用 `-NoProfile -Command <script>` 启动 pwsh，处理 stdin 管道和 Windows creation flags。
   - 新增 `_wrap_command_pwsh()`：构建 PowerShell 脚本，执行 `Set-Location`、`Invoke-Expression`，捕获 `$LASTEXITCODE`，通过 `Get-Location | Out-File` 写入 CWD，并输出框架所需的 CWD 标记。
   - 覆写 `init_session()`：pwsh 路径跳过 bash 环境快照（Windows 环境变量通过 `os.environ` 自然继承），仅写入初始 CWD 文件。
   - 覆写 `_run_bash()` 和 `_wrap_command()`：当 `self._shell_type == "pwsh"` 时分发到 pwsh 变体，保持原有 bash 代码路径不变。

2. **`tools/terminal_tool.py`** — 动态描述：
   - 新增 `_detect_shell_for_description()`：快速、无副作用的 shell 探测（不触发自动安装）。使用 `@lru_cache(maxsize=1)` 缓存。
   - 新增 `_build_dynamic_terminal_description()`：返回 `{"description": ...}`，包含平台适配的首句和 pwsh 专用的禁用手命令引用。在 pwsh 下替换：
     - `cat/head/tail` → `Get-Content/cat/type`
     - `grep/rg/find` → `Select-String/findstr`
     - `ls` → `Get-ChildItem/ls/dir`
     - `echo/cat heredoc` → `echo/Set-Content/Out-File`
     - `Pipe git output to cat` → `Pipe git output to Out-Host -Paging`
   - 在 terminal 工具的 `registry.register()` 调用中注册 `dynamic_schema_overrides=_build_dynamic_terminal_description`，使每次组装工具定义时描述都会重新生成。

3. **`model_tools.py`** — 缓存失效：
   - 将 `_shell_fp`（当前 shell 类型）加入 `get_tool_definitions()` 缓存键。确保会话中期 pwsh 自动安装（会改变 terminal 描述）能使缓存的工具定义失效。

4. **`tests/tools/test_terminal_dynamic_description.py`** — 14 个测试覆盖：
   - Shell 检测（Windows pwsh 存在/不存在、非 Windows、macOS、环境变量覆盖）。
   - 描述构建（pwsh、Windows bash、非 Windows）。
   - pwsh 适配命令引用存在且 Linux 原始引用不存在。
   - Registry 集成。
   - LRU 缓存行为。

**风险和约束**：
- Windows 上有 pwsh 时：所有本地 terminal 命令现在在 pwsh 中执行而非 bash。这意味着 shell 语法（`;` 而非 `&&`、`$env:VAR` 而非 `$VAR`、脚本应使用 `Get-ChildItem` 而非 `ls`）必须是 pwsh 兼容的。Agent 的 prompt 已指示使用当前活动的 shell。
- `_detect_shell_for_description` LRU 缓存意味着会话中期 pwsh 自动安装不会更新描述，直到缓存被清除。缓解方案：`_find_pwsh` 自动安装路径应调用 `_detect_shell_for_description.cache_clear()`。
- Docker/SSH/Modal 后端不受影响——它们始终在容器内使用 bash，不经过 `_resolve_shell()`。

**是否上游**：建议上游。这使 Hermes 成为 Windows 一等公民。改动是模块化的：shell 检测和分发遵循已有的 `_run_bash` / `_wrap_command` 模式，动态描述复用已有的 `dynamic_schema_overrides` 机制。

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

### `1a75a7672` — Windows 下自动安装 Git-Bash，并将 Windows 风格命令转换为 POSIX 风格

**做了什么**：
1. **自动安装 Git for Windows**：当本地终端后端找不到可用的 `bash.exe` 时，按优先级尝试：Chocolatey、Scoop、PortableGit 自解压包、官方安装程序静默安装。安装成功后将目录写入用户 PATH 注册表项。
2. **Windows 命令转 POSIX**：在将命令发送给 bash 前，把 Windows 风格命令转换为 POSIX 风格。包括盘符路径（`C:\foo` → `/c/foo`）、反斜杠目录分隔符、Windows 特有引号/转义规则以及常见 Windows 专用语法。

**新增文件**：
- `tools/environments/_install_git.py` — Windows 下 Git 的下载与安装策略。
- `tools/environments/_process_bash_command.py` — Windows 到 POSIX 的命令翻译。

**涉及文件**：
- `tools/environments/local.py` — 在本地环境后端中集成自动安装和命令转换逻辑。

**上游状态**：建议上游。让 terminal 工具链在 Windows 上开箱即用，无需手动安装 Git。
