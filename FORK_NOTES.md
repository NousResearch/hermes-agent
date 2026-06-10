# Fork notes — Eynzof/hermes-agent-cn

Simplified Chinese: [`FORK_NOTES.zh-CN.md`](./FORK_NOTES.zh-CN.md)

This document explains the fork-specific changes on `main` that diverge from upstream `NousResearch/hermes-agent`. New behavioral patches should use `[CN-fork] P-NNN` in the commit message and be listed below.

| ID | Target file | What it does | Why we need it | Upstream status |
|---|---|---|---|---|
| **P-002** | `hermes_cli/web_server.py` | Adds `POST /api/upload` for dashboard attachment uploads | v2 web composer's drag-to-upload depends on it; upstream had it once (`e7c3cd772`) then reverted | Not in upstream |
| **P-003** | `hermes_cli/web_server.py` | Drops the `_DASHBOARD_EMBEDDED_CHAT_ENABLED` gate on `/api/ws` | v2 runs `hermes dashboard` without `--tui`, the gate would close gateway WS | **Largely addressed upstream** — v0.16.0 (#38591) defaults the flag to `True` and removes the dashboard `--tui` flag; fork keeps the explicit gate removal on `/api/ws` as defense-in-depth |
| **P-004** | `hermes_cli/web_server.py` | Adds `GET /api/fs/list` for the v2 web workspace picker | v2 `/new` task page browses directories instead of `window.prompt()` for path; restricted to user home subtree | Not in upstream |
| **P-005** | `hermes_cli/web_server.py` | Adds `GET /api/mcp-servers` (read-only `{summary, servers:[{name,enabled}]}`) — handler `list_mcp_servers_summary` | v2 panel "健康检查" cell needs MCP count without leaking command/args/env (which embed secrets) | Distinct from upstream's `/api/mcp/servers` (exposes url/command/args); fork handler renamed in 2026-06-04 sync to avoid an operationId clash |
| **P-006** | `hermes_cli/config.py` | Registers `OPTIONAL_ENV_VARS` for CN providers (ARK / QIANFAN / HUNYUAN / SILICONFLOW / MODELSCOPE / AI302 / COMPSHARE) | Dashboard env panel is metadata-driven; upstream only knows global providers (OpenAI / Anthropic / Google / DeepSeek) | Won't be upstreamed (CN-specific) |
| ~~**P-007**~~ | `tui_gateway/ws.py` | ~~Wraps the dispatch handler in a try/except that logs traceback + returns a JSON-RPC error response instead of silently closing the WS~~ | Without this, any unhandled handler exception or json.dumps serialization failure shows up in the client as "WebSocket closed" with zero diagnostic context | **Superseded by upstream** — dropped in 2026-06-04 sync |
| **P-008** | `hermes_cli/web_server.py` | ~~Adds `GET/PUT /api/profiles/active`~~ → upstream shipped its own `GET/POST /api/profiles/active`; fork now keeps only a **compat layer**: adds `name` to the GET response (desktop reads `.name`) + a `PUT` alias (desktop sets via PUT) | v2 web profile switcher reads `.name` and writes via `PUT`; upstream returns `{active,current}` and only has `POST` | **Upstreamed (GET/POST)** + fork compat (2026-06-04 sync) |
| **P-009** | `hermes_cli/web_server.py`, `tui_gateway/sse.py` | Adds SSE+POST gateway transport at `/api/v2/events` and `/api/v2/rpc` | desktop uses EventSource for streaming and POST for JSON-RPC to avoid WebSocket edge cases in packaged desktop runtimes | Maybe upstream |
| **P-010** | `hermes_cli/config.py` | Registers `LONGCAT_API_KEY` in `OPTIONAL_ENV_VARS` | CN model settings need first-class LongCat credentials in the env panel | Won't be upstreamed unless upstream adopts LongCat |
| **P-011** | `tui_gateway/server.py` | Adds `slug_filter` to `model.options` and `provider.probe` RPC | desktop needs filtered model picker options and a lightweight provider health probe | Maybe upstream |
| **P-012** | `hermes_cli/main.py` | `_model_flow_anthropic()` prompts for optional custom `base_url` instead of unconditionally removing it | Users running Anthropic-compatible proxies or alternative endpoints need to preserve a custom `base_url` during model setup | Should be upstreamed |
| **P-013** | `model_tools.py`, `tests/run_agent/test_repair_tool_arg_keys.py` | Adds automatic tool argument key repair (`repair_tool_arg_keys`) with alias tables, per-tool overrides, fuzzy fallback, nested object/array recursion, and an optional callback hook; integrated into `handle_function_call` before type coercion | LLMs often misname arguments (e.g. "file"→"path", "cmd"→"command"); this makes tool dispatch resilient to common drift without weakening JSON Schemas | Should be upstreamed |
| **P-014** | `.github/workflows/release-runtime.yml`, `tools/mcp_tool.py`, `hermes_cli/config.py`, `docs/RUNTIME_RELEASES.md`, `tests/tools/test_mcp_tool.py` | Bundles the native MCP client SDK into the frozen runtime (install entry later folded into the `cn-desktop` extra — see P-015 — plus `--collect-submodules/--copy-metadata mcp` and a CI assert on `mcp-*.dist-info`), and makes `discover_mcp_tools()` warn once when `mcp_servers` is configured but the SDK is absent instead of silently no-op'ing at debug | Issue #16: the desktop runtime shipped without the `mcp` extra, so `_MCP_AVAILABLE=False` and configured `mcp_servers` registered no tools with no INFO-level log. The packaging fix is fork-specific; the diagnostic + known-root-key are generic | Packaging change is CN-specific; the `mcp_tool.py` warning and `mcp_servers` known-root-key should be upstreamed |
| **P-015** | `pyproject.toml`, `.github/workflows/release-runtime.yml`, `docs/RUNTIME_RELEASES.md`, `uv.lock` | Adds a `cn-desktop` aggregate extra that pre-bakes every backend the frozen runtime exposes (`web`, `anthropic`, `mcp`, `feishu`, `dingtalk`, `wecom`, plus 微信's `aiohttp`/`qrcode`/`cryptography`). The release workflow installs `.[cn-desktop]`, collects the IM SDK submodules + metadata, runs a build-env import smoke test, and asserts each backend's `dist-info` in the frozen output | Desktop report: the 飞书/钉钉/企微/微信 adapters silently degraded to "unavailable" because their SDKs (`lark-oapi`, `dingtalk-stream`, …) were never bundled and the frozen build can't lazy-install. Same root cause as P-014, generalized to all desktop backends | Packaging is CN-specific; not upstreamed (upstream doesn't build these artifacts) |
| **P-016** | `tools/terminal_tool.py`, `tools/environments/local.py`, `tools/environments/proccess_pwsh.py`, `tools/environments/base.py`, `model_tools.py`, `tests/tools/test_terminal_dynamic_description.py` | PowerShell (pwsh / Windows PowerShell) native execution: on Windows, uses pwsh as the primary local shell with full lifecycle support (spawn, wrap, init_session, cwd tracking); removes Git Bash auto-install and fallback. Adds runtime-adaptive terminal tool description that replaces Linux/bash command references with pwsh cmdlets when the active shell is pwsh; adds shell-fingerprint to tool-definitions cache key. Adds pwsh_transform warning propagation so the LLM is notified when its PowerShell 7.x syntax was down-leveled to 5.1 | Agent on Windows was hardcoded to Git Bash; pwsh is faster (-NoProfile), has better Windows-native path handling, and avoids the POSIX-translation overhead. Git for Windows auto-install and Git Bash fallback have been deleted — the agent now requires pwsh or system PowerShell on Windows. The static `TERMINAL_TOOL_DESCRIPTION` contained Linux-only command references that are misleading under pwsh | Should be upstreamed |
| **P-017** | `agent/tool_dedup.py`, `agent/agent_init.py`, `agent/conversation_loop.py`, `agent/tool_executor.py` | Adds `ToolDedupTracker` that detects consecutive identical tool calls across API iterations and injects escalating reminders (`<system-reminder>`) at repeat counts 3, 5, and 8 to break infinite loops | Agent on complex tasks can enter infinite loops calling the same tool with the same arguments repeatedly — the existing same-turn dedup (`_deduplicate_tool_calls`) doesn't catch this cross-iteration pattern | Internal — addresses a behavioral robustness gap; the mechanism is generic but integration points are fork-specific |
| **P-018** | `agent/agent_init.py`, `tests/run_agent/test_init_fallback_on_exhausted_pool.py` | Adds `_api_key_required` helper and empty-key guards before OpenAI / Anthropic SDK client construction. Raises `RuntimeError: no API key (param empty, env vars unset)` instead of letting a low-level SDK auth exception bubble up | Empty key (param empty, env vars unset) previously triggered confusing low-level SDK exceptions that looked like panics, especially in TUI/gateway background threads where stack traces are not surfaced to the user | Should be upstreamed |

> **P-001** (provider dict-vs-list mismatch in `tui_gateway/server.py`) — **dropped from this fork**. Upstream has since fixed it; the line `user_provs = cfg.get("providers")` in `_apply_model_switch` already does the right thing.

## Release/support changes

These are fork maintenance changes, not runtime behavior patches:

| Area | Target file | What it does |
|---|---|---|
| Upstream sync | `scripts/sync-upstream.sh`, `.github/workflows/upstream-watch.yml`, `MAINTAINING.md` | Keeps upstream syncs on temporary PR branches instead of merging directly into `main` |
| Managed runtime | `.github/workflows/release-runtime.yml`, `scripts/sign_runtime_manifest.py`, `docs/RUNTIME_RELEASES.md` | Builds PyInstaller runtime artifacts, signs manifests, and publishes GitHub Releases consumed by desktop. Bundles the `[web,anthropic,mcp]` extras and asserts each SDK's `dist-info` is present in the frozen output (see P-014 for the MCP gap) |

## Per-patch detail

### P-002: `POST /api/upload` for dashboard attachment uploads

**Symptom**: v2 web composer drags a file → upload fails with 404 because `/api/upload` doesn't exist. v2 stack trace shows `XMLHttpRequest` returning HTTP 404 on the upload URL.

**Root cause**: Upstream `e7c3cd772` (commit "Add dashboard attachment upload endpoint") added this endpoint, then it was reverted in a later commit. The endpoint itself is small and self-contained — we just bring it back.

**What the patch does**: Adds a single FastAPI handler that takes a multipart `file` + `session_id`, writes it under `~/.hermes/sessions/<id>/attachments/`, and returns `{ok, filename, path, size, mime_type}`. Reuses upstream's `_next_unique_path` helper for naming collisions.

**Side effects**: Adds an attachment-upload attack surface. Mitigated by:
- Gated by the same session token as all other `/api/` routes
- Never overwrites: collisions resolved via `_next_unique_path`
- Writes only inside the session's own attachments directory (validated)
- No content-type sniffing that could trigger executable behavior

**Should we upstream?** Yes, but the original revert reason isn't documented in upstream's commit log. Worth a thread before sending a PR.

---

### P-003: Drop `_DASHBOARD_EMBEDDED_CHAT_ENABLED` gate on `/api/ws`

**Symptom**: v2 web app `/api/ws` upgrade closes immediately with 4001. Gateway never connects, all chat is broken.

**Root cause**: Upstream v0.12.0 added a module-level flag `_DASHBOARD_EMBEDDED_CHAT_ENABLED` that's only set to `True` when running `hermes dashboard --tui` (the embedded TUI mode). v2 runs `hermes dashboard --no-open` without `--tui` for headless dashboard + Web UI, so the gate stays closed.

**What the patch does**: Removes the gate from the `/api/ws` route's preconditions. The route is still gated by token + loopback host check, which is sufficient.

**Side effects**: WebSocket gateway is now reachable from any same-origin web UI that has the session token, regardless of `--tui` mode. This matches the security posture of `/api/pty`, `/api/pub`, and `/api/events`, all of which work without `--tui`.

**Should we upstream?** Yes. The gate seems to have been added defensively, but it breaks legitimate Web UI use cases.

**Update (v0.16.0 sync)**: upstream #38591 now always enables embedded chat (`_DASHBOARD_EMBEDDED_CHAT_ENABLED = True` by default) and removed the dashboard `--tui` flag, so the original symptom no longer occurs out of the box. The fork retains the explicit gate removal on `/api/ws` so the gateway RPC channel (used by the v2 web UI / desktop) stays reachable even if embedded chat is ever disabled.

---

### P-004: `GET /api/fs/list` for v2 web workspace picker

**Symptom**: v2 `/new` task page → "选择 workspace" → falls back to `window.prompt()` asking the user to type a path. UX is bad on a desktop OS.

**Root cause**: Upstream has no filesystem browse endpoint. Electron desktop shells use the OS native dialog, but a pure web UI can't.

**What the patch does**: Adds `GET /api/fs/list?path=<dir>&include_hidden=<bool>` returning `{path, parent, home, entries: [{name, path, is_dir}]}`. Path is resolved through:
- `~` expansion
- `..` folding via `Path.resolve(strict=False)`
- enforced subtree of `Path.home()` (raises 400 if outside)

Plus a 5000-entry cap to bound responses on huge directories.

**Side effects**: Adds a directory-listing attack surface. Mitigated by:
- Token gate (same as all `/api/` routes)
- Enforced home subtree — picker can't wander into `/Library`, `/private`, `/System`, etc.
- Permissions tolerant — entries that fail `is_dir()` (broken symlinks, denied access) are silently skipped
- Hidden-file filter defaults to off

**Should we upstream?** Maybe — depends on whether upstream wants browser-only Web UI to be a first-class deployment target.

---

### P-005: `GET /api/mcp-servers` (read-only list)

**Symptom**: v2 task panel has a 5-cell health-check grid. One cell is "MCP" (configured / enabled). Upstream's `/api/tools/toolsets` returns toolsets and MCP servers blended together — extracting just the MCP count is awkward.

**Root cause**: MCP server config is in `config.yaml`'s `mcp_servers` key. Upstream doesn't expose it via REST.

**What the patch does**: Returns `{summary: {total, enabled}, servers: [{name, enabled}]}`. **Deliberately does not return** `command` / `args` / `env` because those routinely embed secrets.

**Side effects**: None. Read-only.

**Should we upstream?** Upstream added a *different* `/api/mcp/servers` (slash) in the 2026-06-04 sync that returns full per-server config (url/command/args, env redacted). The fork keeps `/api/mcp-servers` (hyphen) with the minimal `{name, enabled}` shape the desktop health-check expects; the handler was renamed `list_mcp_servers_summary` so the two endpoints don't collide on the generated OpenAPI operationId.

---

### P-006: `OPTIONAL_ENV_VARS` for CN providers

**Symptom**: v2 Models settings page lists CN providers (alibaba / deepseek / kimi / volcengine-ark / minimax-cn / baidu-qianfan / tencent-hunyuan / siliconflow / modelscope / ai302) in its catalog, but the env panel doesn't expose `*_API_KEY` entries for them — users have to manually `vim ~/.hermes/.env`.

**Root cause**: Upstream `OPTIONAL_ENV_VARS` is the metadata dict that drives the env panel UI. It only registers global providers (OpenAI / Anthropic / Google / DeepSeek / Groq / etc.). CN providers were never added.

**What the patch does**: Adds 7 `*_API_KEY` entries plus 1 `ARK_BASE_URL`, all `category="provider"`. `ARK_API_KEY` is top-5 (always visible), the rest are `advanced=True`. Chinese description / prompt / official docs URL.

**Side effects**: Env panel grows by 8 entries. Doesn't change parsing of any existing entry.

**Should we upstream?** Maybe, on a per-provider basis. Some are obscure and upstream might decline.

---

### P-007: Surface gateway WS dispatch exceptions

**Symptom**: v2 sometimes shows "WebSocket closed" Toast with no diagnostic info. Refresh, retry — the issue is intermittent and unreproducible.

**Root cause**: `tui_gateway/ws.py` wraps `server.dispatch` + `transport.write_async` in a bare `try/finally`. Any unhandled exception (from an inline handler or from `json.dumps` of a non-serializable response) escapes the loop, hits `finally → ws.close()`, and the client sees "WebSocket closed" with zero context.

**What the patch does**:
- Wraps dispatch + write in an explicit `try/except`
- Logs traceback to `~/.hermes/logs/dispatch_exceptions.log`
- Converts the crash into a JSON-RPC error response (code -32000) sent back to the client
- Keeps the connection alive for subsequent calls

**Side effects**: Log file grows on dispatch crashes (rotate via standard logrotate if needed). Error responses use a non-standard error code; clients should treat -32000 as a generic server error.

**Should we upstream?** Done — as of the 2026-06-04 upstream sync, upstream ships equivalent dispatch-exception handling (try/except around `dispatch`, a JSON-RPC `-32603` "internal error" response, structured `dispatch_crashes` logging via `_log.exception`, and the connection kept alive for subsequent calls). The fork implementation — including the dedicated `~/.hermes/logs/dispatch_exceptions.log` file and the `-32000` error code — was dropped in favor of upstream's version, which the merged `handle_ws` observability counters already depend on. The standard hermes log now captures the traceback.

---

### P-008: `GET/PUT /api/profiles/active`

**Symptom**: v2 wants to build a profile switcher UI. Upstream has `GET /api/profiles` (list), `POST /api/profiles` (create), `DELETE /api/profiles/{name}`, `PATCH /api/profiles/{name}` (rename), `GET/PUT /api/profiles/{name}/soul` — but **no way to read or write the sticky active profile** (`~/.hermes/active_profile`).

**Root cause**: Upstream's dashboard binds `HERMES_HOME` at process startup; "switching the active profile mid-session" isn't part of its model. Switching requires restarting hermes. But the *sticky* setting (which profile to use *next* time) does need a getter/setter.

**What the patch does**:
- `GET /api/profiles/active` → `{name: <sticky default>}`. Reads `~/.hermes/active_profile` (or returns `default` if file missing).
- `PUT /api/profiles/active` body `{name}` → writes the file. **Does not affect the currently running dashboard process** — the client (v2) is responsible for prompting the user to restart hermes.

**Side effects**: None. File-backed sticky preference, mirroring `hermes profile use <name>` CLI behavior.

**Should we upstream?** Done — upstream shipped `GET/POST /api/profiles/active` in the 2026-06-04 sync (GET returns `{active, current}`; POST sets via `ProfileActiveUpdate`). The fork's standalone GET/PUT were removed to avoid a duplicate route. To keep the existing desktop client working without a coordinated release, two minimal compat shims now ride on upstream's endpoint: the GET response also carries `name` (= `active`; the desktop's `useActiveProfile` reads `.name`), and a `@app.put("/api/profiles/active")` alias is stacked on the setter (the desktop sets via `PUT`). Both can be dropped once the desktop migrates to `{active,current}` + `POST`.

---

### P-009: SSE+POST gateway transport

**Symptom**: desktop's packaged runtime needs a stable browser-friendly
streaming transport. Relying only on `/api/ws` makes failures harder to
debug and interacts poorly with some desktop shell/network setups.

**Root cause**: Upstream exposes the TUI gateway over WebSocket. desktop
wants EventSource for server-to-client events and normal HTTP POST for
client-to-server JSON-RPC.

**What the patch does**:
- Adds `GET /api/v2/events` for SSE frames.
- Adds `POST /api/v2/rpc` for gateway JSON-RPC requests.
- Adds `tui_gateway/sse.py` transport plumbing.

**Side effects**: Adds another authenticated gateway transport surface.
It uses the same session token model as the dashboard API.

**Should we upstream?** Maybe. It is useful for browser-hosted dashboards
and desktop shells, but it changes the supported gateway transport matrix.

---

### P-010: `LONGCAT_API_KEY`

**Symptom**: CN model settings include LongCat, but the dashboard env
metadata had no first-class `LONGCAT_API_KEY` entry.

**Root cause**: Upstream provider metadata focuses on global providers and
does not include this CN-specific key.

**What the patch does**: Adds `LONGCAT_API_KEY` to `OPTIONAL_ENV_VARS`.

**Side effects**: Env settings show one additional provider credential.

**Should we upstream?** Only if upstream decides to support LongCat.

---

### P-011: Gateway model filtering and provider probe

**Symptom**: desktop needs to filter model picker options by provider
slug and run a lightweight provider health check without starting a full
agent turn.

**Root cause**: Upstream `model.options` returns broad picker data, and
there was no small JSON-RPC method for provider probing.

**What the patch does**:
- Adds `slug_filter` support to `model.options`.
- Adds a `provider.probe` gateway RPC.

**Side effects**: Minimal. The new RPC should avoid returning secrets or
raw provider config.

**Should we upstream?** Maybe, but the probe shape should be reviewed before
opening an upstream PR.

---

### P-012: Optional custom `base_url` in `_model_flow_anthropic()`

**Symptom**: When adding an Anthropic model through the interactive setup flow, any pre-configured or desired custom `base_url` is silently discarded because the code unconditionally calls `model.pop("base_url", None)`.

**Root cause**: `_model_flow_anthropic()` hardcoded `model.pop("base_url", None)` with the assumption that all Anthropic traffic should go to the official `https://api.anthropic.com` endpoint. This breaks users who need to point at Anthropic-compatible proxies, OpenRouter, or private endpoints.

**What the patch does**:
- Removes the unconditional `model.pop("base_url", None)`.
- After model selection, prompts the user with the current `base_url` (or `https://api.anthropic.com` as the default).
- If the user types a custom URL, it is saved to `model["base_url"]`.
- If the user presses Enter without input, the existing `base_url` is kept; only when none existed before is it popped so the runtime falls back to the hardcoded Anthropic URL.

**Side effects**: None. The runtime (`runtime_provider.py`) already reads `model_cfg.get("base_url")` for the `anthropic` provider, so no runtime changes are required.

**Should we upstream?** Yes. The change is backward-compatible and enables legitimate use cases for alternative Anthropic-compatible endpoints.

---

### P-013: Automatic tool argument key repair in `handle_function_call`

**Symptom**: LLM tool calls frequently fail with "unknown parameter" because the model uses synonyms or typos for argument names (e.g. `file` instead of `path`, `cmd` instead of `command`, `backgroud` instead of `background`).

**Root cause**: Hermes' JSON Schemas are strict. When an LLM drifts from the canonical field name, `handle_function_call` passes the bad key straight through to the tool handler, which often rejects it.

**What the patch does**:
- Introduces `repair_tool_arg_keys()` and `_repair_nested_args()` in `model_tools.py`.
- Defines `TOOL_FIELD_ALIASES` — a large global alias table covering general, file, shell, web, task, todo, input, search, memory, cronjob, and skill argument names.
- Defines `TOOL_SPECIFIC_ALIASES` for per-tool overrides (e.g. `delegate_task` maps `task`→`goal` instead of `task`→`prompt`; `cronjob` maps `command`→`action`).
- Uses `difflib.get_close_matches` as a fuzzy fallback for typos when no alias matches.
- Recursively repairs keys inside nested objects and arrays of objects, guided by the schema's `properties` and `items` definitions.
- Adds an optional callback hook (`set_arg_repair_callback`) so external systems (TUI, ACP) can be notified of top-level key repairs.
- Hooks the repair into `handle_function_call()` so it runs *before* `coerce_tool_args()`, meaning repaired keys are still type-coerced as usual.
- Ships comprehensive tests in `tests/run_agent/test_repair_tool_arg_keys.py`.

**Side effects**: Minimal. The function is a pure key-mapping transform; unknown keys are left untouched. The fuzzy matcher only kicks in for keys ≥4 chars with a similarity ratio ≥0.75–0.80, so random fields are unlikely to be falsely renamed.

**Should we upstream?** Yes. This is a generic robustness improvement that benefits every Hermes deployment regardless of platform or provider.

---

### P-014: Native MCP client missing in the frozen desktop runtime

**Symptom** (issue #16): A user configures `mcp_servers` correctly in `~/.hermes/config.yaml`, the MCP server script works standalone, but the CN Desktop agent never connects to it — `agent.log` shows no MCP discovery/connection lines and no `mcp_*` tools appear. `pip install mcp` on the host does not help.

**Root cause**: The native MCP client is fully implemented (`tools/mcp_tool.py`, `discover_mcp_tools()`), but the SDK is an *optional* dependency that lives only in the `[mcp]` extra. The runtime release workflow installed just `.[web,anthropic]`, so the frozen PyInstaller artifact shipped **without** the `mcp` package. Inside the frozen runtime `_MCP_AVAILABLE` is therefore `False`, and `discover_mcp_tools()` returns `[]` after logging only at `debug` level — invisible at the default INFO log level. The host's `pip install mcp` is irrelevant because the frozen runtime bundles its own interpreter and packages.

**What the patch does**:
- `release-runtime.yml`: bundles the `mcp` SDK (install entry later folded into the `cn-desktop` extra — P-015), adds `--collect-submodules mcp` + `--copy-metadata mcp` to PyInstaller, and extends the verify step to fail the build if `mcp-*.dist-info` is absent (so this can't silently regress).
- `tools/mcp_tool.py`: when `mcp_servers` is configured but the SDK is unavailable, `discover_mcp_tools()` now emits a one-time `WARNING` ("mcp_servers are configured but the MCP SDK is not available …") instead of a silent debug line. Users without MCP config keep the quiet debug path.
- `hermes_cli/config.py`: adds `mcp_servers` to `_KNOWN_ROOT_KEYS` so the documented root schema is accurate.
- `docs/RUNTIME_RELEASES.md`: documents MCP bundling as a required runtime dep and updates the manual dry-run command.
- Tests in `tests/tools/test_mcp_tool.py` cover the warn-when-configured, stay-quiet-when-unconfigured, and warn-once behaviors.

**Side effects**: The frozen runtime grows by the `mcp` SDK and its transitive deps (`anyio`/`httpx-sse`/`sse-starlette`, all already present via `web`/`anthropic`). No behavior change for source installs that already include the `[mcp]` extra.

**Should we upstream?** The packaging change is CN-runtime-specific (upstream doesn't build these PyInstaller artifacts). The `mcp_tool.py` diagnostic and the `mcp_servers` known-root-key are generic and worth upstreaming.

---

### P-015: IM platform backends missing in the frozen desktop runtime

**Symptom**: A desktop user correctly sets the Feishu App ID/Secret in `.env`, adds the Feishu platform to `config.yaml`, and the gateway process runs — but it never connects to Feishu. `lark-oapi` "cannot be installed" inside the packaged app. The same applies to DingTalk, WeCom, and WeChat.

**Root cause**: Identical to P-014, generalized. The IM adapters (`gateway/platforms/feishu.py`, `dingtalk.py`, `wecom*.py`, `weixin.py`) import their SDKs under `try/except` and degrade to an `*_AVAILABLE = False` state when the package is missing. Those SDKs live only in optional extras (`[feishu]` → `lark-oapi`, `[dingtalk]` → `dingtalk-stream` + `alibabacloud-*`, `[wecom]` → `defusedxml`; 微信 has **no** extra and needs `aiohttp`/`qrcode`/`cryptography`). `[all]`'s policy deliberately excludes these because they're lazy-installable via `tools/lazy_deps.py` — but **lazy install can't run inside a frozen PyInstaller binary** (no working pip), so the desktop runtime, which installed only `.[web,anthropic,mcp]`, shipped without any of them. The host-side `pip install lark-oapi` the user tried writes to system Python, which the frozen runtime never uses.

**What the patch does**:
- `pyproject.toml`: adds a `cn-desktop` aggregate extra listing every backend the frozen runtime must pre-bake — `web`, `anthropic`, `mcp`, `feishu`, `dingtalk`, `wecom`, plus 微信's `aiohttp`/`qrcode`/`cryptography` (pinned to match the existing extras). This is the single source of truth for "what the desktop ships", deliberately diverging from `[all]`'s lazy-install policy.
- `release-runtime.yml`: installs `.[cn-desktop]`; adds `--collect-submodules`/`--copy-metadata` for `lark_oapi`, `dingtalk_stream`, `alibabacloud_dingtalk` (+ `alibabacloud_tea_openapi`/`alibabacloud_tea_util`), `aiohttp`, `qrcode`; adds a **build-env import smoke test** that imports each adapter and asserts its `*_AVAILABLE` flag is True (fails fast on a missing extra dep); and generalizes the verify step to assert every bundled backend's `dist-info` is present in the frozen output.
- `docs/RUNTIME_RELEASES.md`: documents the `cn-desktop` extra as the place to add future desktop backends, and flags the `alibabacloud_*` collection as fragile (smoke-test against a live DingTalk bot on first release).
- `uv.lock`: regenerated for the new extra (`uv lock --check` passes).

**Side effects**: The frozen runtime grows by the IM SDKs and their transitive deps (notably the pure-Python `alibabacloud_*` chain). All are pure-Python with cross-platform wheels/sdists — unlike `matrix`'s `python-olm`, which needs a C toolchain and is intentionally still excluded. No change to source installs.

**Should we upstream?** No — upstream doesn't build these PyInstaller artifacts. The `cn-desktop` extra and packaging are CN-runtime-specific.
### P-016: PowerShell (pwsh) native execution + runtime-adaptive terminal description

**Symptom**: On Windows, the agent was hardcoded to always use Git Bash. PowerShell 7 (pwsh) is faster to start (`-NoProfile`), handles Windows paths natively (no `/c/foo` translation), and is the default shell on modern Windows. Additionally, the terminal tool's static `TERMINAL_TOOL_DESCRIPTION` referenced Linux/bash commands (`cat/head/tail`, `grep/rg/find`, `echo/cat heredoc`, `Pipe git output to cat`) that don't exist on native pwsh — the LLM would either try them and fail, or not know the pwsh equivalents to avoid.

**Root cause**: Upstream's `LocalEnvironment` is bash-only. The terminal tool description is a hardcoded static string assuming a Linux environment.

**What the patch does**:

1. **`tools/environments/local.py`** — Shell resolution and pwsh execution:
   - Adds `_resolve_shell()`: on Windows, detects pwsh (PowerShell 7) or falls back to Windows PowerShell (system PowerShell). Respects `HERMES_SHELL_TYPE` (auto/pwsh/powershell) and `HERMES_PWSH_PATH` env overrides. Git Bash support and auto-install have been removed. Uses `_find_pwsh` from `tools/environments/_find_pwsh.py` to locate the PowerShell executable.
   - `LocalEnvironment.__init__` calls `_resolve_shell()` and stores `self._shell_type` / `self._shell_path`.
   - Adds `_run_pwsh()`: spawns pwsh with `-NoProfile -Command <script>`, handles stdin piping, Windows creation flags.
   - Adds `_wrap_command_pwsh()`: builds a PowerShell script that executes `Set-Location`, `Invoke-Expression`, captures `$LASTEXITCODE`, writes CWD via `Get-Location | Out-File`, and emits a CWD marker for the framework.
   - Overrides `init_session()`: pwsh path skips the bash env-snapshot dance (Windows env vars propagate through `os.environ` naturally). Just writes the initial CWD file.
   - Overrides `_run_bash()` and `_wrap_command()`: dispatch to pwsh variants when `self._shell_type == "pwsh"` or `"powershell"`, preserving the original bash code paths unchanged for non-Windows platforms.

2. **`tools/terminal_tool.py`** — Dynamic description:
   - Adds `_detect_shell_for_description()`: fast, side-effect-free shell probe (no auto-install). Uses `@lru_cache(maxsize=1)`.
   - Adds `_build_dynamic_terminal_description()`: returns `{"description": ...}` with platform-appropriate first sentence and pwsh-adapted forbidden-command references. On pwsh, replaces:
     - `cat/head/tail` → `Get-Content/cat/type`
     - `grep/rg/find` → `Select-String/findstr`
     - `ls` → `Get-ChildItem/ls/dir`
     - `echo/cat heredoc` → `echo/Set-Content/Out-File`
     - `Pipe git output to cat` → `Pipe git output to Out-Host -Paging`
   - Registers `dynamic_schema_overrides=_build_dynamic_terminal_description` in the terminal tool's `registry.register()` call so the description is rebuilt each time tool definitions are assembled.

3. **`model_tools.py`** — Cache invalidation:
   - Adds `_shell_fp` (current shell type) to `get_tool_definitions()` cache key. This ensures that a mid-session shell change (which changes the terminal description) invalidates the cached tool definitions.

4. **`tests/tools/test_terminal_dynamic_description.py`** — 16 tests covering:
   - Shell detection (Windows pwsh found/not-found, non-Windows, macOS, env overrides).
   - Description building (pwsh, Windows bash, non-Windows).
   - pwsh-adapted command references present and Linux-only references absent.
   - Registry integration.
   - LRU cache behavior.

5. **`tools/environments/proccess_pwsh.py` + `tools/environments/base.py`** — pwsh_transform warning propagation:
   - Changes `pwsh_transform()` return type from `str` to `tuple[str, list[str]]` — the second element is a list of human-readable warnings describing each transformation applied (e.g. `"Line 3: ternary operator rewritten to if/else"`).
   - Each internal `_transform_*_line` function collects warnings for transformations it performs.
   - `_run_pwsh()` in `local.py` captures warnings from the single `pwsh_transform()` call and stores them on `self._pwsh_warnings`.
   - `execute()` in `base.py` attaches `_pwsh_warnings` to the result dict under the key `"pwsh_warnings"`.
   - `terminal_tool()` surfaces `pwsh_warnings` in the final JSON response returned to the LLM, so the agent can see when its PowerShell 7.x syntax was rewritten to 5.1-compatible code.
   - Tests in `tests/tools/test_pwsh_transform.py` (18 tests) and `tests/tools/test_local_pwsh_warnings.py` (8 tests) cover all layers.

**Side effects**:
- On Windows with pwsh: all local terminal commands now execute in pwsh instead of bash. This means shell syntax (`;` not `&&`, `$env:VAR` not `$VAR`, `Get-ChildItem` not `ls` for scripts) must be pwsh-compatible. The agent's prompt already instructs it to use the active shell. Git Bash is no longer supported or auto-installed.
- The `_detect_shell_for_description` LRU cache means a mid-session shell change won't update the description until cache is cleared. Mitigation: callers can invoke `_detect_shell_for_description.cache_clear()`.
- Docker/SSH/Modal backends are unaffected — they always use bash inside containers and don't go through `_resolve_shell()`.

**Should we upstream?** Yes. This makes Hermes a first-class Windows citizen. The changes are modular: shell detection and dispatch follow the existing `_run_bash` / `_wrap_command` pattern, and the dynamic description reuses the existing `dynamic_schema_overrides` mechanism.

---

### P-017: Consecutive identical tool call dedup (infinite loop breaker)

**Symptom**: On complex tasks (long-running builds, multi-step refactors), the agent sometimes enters an infinite loop, calling the same tool with the same arguments across consecutive API iterations — e.g. repeatedly reading the same file, or calling `run` with the same command. The existing `_deduplicate_tool_calls()` in `run_agent.py` only removes exact duplicates within a **single** turn's tool batch, missing cross-iteration repeats entirely.

**Root cause**: No cross-step dedup mechanism existed. Each API iteration's tool results feed into the next LLM call without any history awareness of what was already tried.

**What the patch does**:

1. **`agent/tool_dedup.py`** — New module with `ToolDedupTracker` class:
   - Normalizes tool call keys via `_canonical_tool_arguments()` (recursive key-sorting for dicts, fallback to `str()`).
   - Tracks `_seen_call_keys` (all calls seen across steps) and `_consecutive_key`/`_consecutive_count` (streak tracking).
   - `begin_step(previous_calls, step_no, turn_id)`: seeds state from previous step's tool calls.
   - `end_step()`: returns the list of calls made this step for the next iteration, and advances the consecutive streak.
   - `check_and_register(tool_name, arguments)`: called during tool execution; if the call key was seen in a previous step, returns escalating reminder text at repeat counts 3, 5, and 8.
   - Escalating reminders: at count 3, a gentle nudge (`<system-reminder>`: "You are repeating the exact same tool call…"). At counts 5 and 8, a stronger message naming the tool, repeat count, and arguments.

2. **`agent/agent_init.py`** — Initializes `_tool_dedup_tracker` on the `AIAgent` instance.

3. **`agent/conversation_loop.py`** — Step lifecycle:
   - Before each API call: `begin_step()` seeds cross-step state from the previous iteration's calls.
   - After all tool results are collected: `end_step()` captures the current step's calls for the next iteration.

4. **`agent/tool_executor.py`** — Dedup check injection:
   - In `execute_tool_calls_concurrent()`: after each tool execution, calls `check_and_register()` and appends any reminder text to the result.
   - In `execute_tool_calls_sequential()`: same pattern.

**Side effects**:
- Tool results may grow by a few hundred characters (the `<system-reminder>` text) when dedup is triggered.
- The reminder text is visible to the LLM, which may influence its next action — this is the intended behavior.
- Thread safety: `check_and_register()` uses a `threading.Lock()` to protect shared state in the concurrent execution path.

**Should we upstream?** The mechanism is generic, but the integration points (`agent_init.py`, `conversation_loop.py`, `tool_executor.py`) are specific to this fork's agent architecture. Could be proposed as a generalized observability hook.

---

### P-018: Empty API key guard in `agent/agent_init.py`

**Symptom**: When the API key is empty (parameter explicitly ` ""`, environment variables unset), the agent panics with a low-level OpenAI or Anthropic SDK auth exception instead of a clean, actionable error message. In TUI/gateway background threads the stack trace is not surfaced to the user, making the failure look like a silent crash.

**Root cause**: `init_agent()` had no explicit validation that `api_key` is non-empty before handing it to `_create_openai_client()` or `build_anthropic_client()`. Empty strings reached the SDK constructors and produced confusing exceptions.

**What the patch does**:
- Adds `_api_key_required(provider, api_key, base_url)` helper that returns `False` for providers that genuinely do not need a literal key (Azure Entra ID callable tokens, `"aws-sdk"` / `"no-key-required"`, Bedrock).
- Inserts a guard in the `anthropic_messages` branch right before `build_anthropic_client()`.
- Inserts a guard in the `chat_completions` branch right before `_create_openai_client()`.
- Both guards raise `RuntimeError("no API key (param empty, env vars unset)")` when the key is empty and the provider requires one.
- Adds two pytest cases covering the `chat_completions` and `anthropic_messages` empty-key paths.

**Side effects**: None for providers that legitimately need no key (local endpoints with `"no-key-required"`, Bedrock, Azure Entra ID). The fallback loop (`fallback_model` / `fallback_providers`) still executes before the guard.

**Should we upstream?** Yes. The change is purely additive, provider-agnostic, and prevents a poor user experience across CLI, TUI, gateway, and direct `AIAgent()` usage.

---

## Windows compatibility patches

These patches improve first-class Windows support. They are authored by Maxwell Geng and are candidates for upstreaming.

### `282cfeeca` — Add `posix` option for `shlex.split` (Windows compatible)

**What it does**: Passes `posix=os.name == "posix"` to every `shlex.split()` call about `subprocess` usage in the codebase so that backslashes in Windows paths are not misinterpreted as escape characters.

**Files touched**:
- `agent/copilot_acp_client.py`
- `agent/shell_hooks.py`
- `agent/subdirectory_hints.py`
- `cli.py`
- `gateway/run.py`
- `hermes_cli/auth.py`
- `hermes_cli/gateway_windows.py`
- `hermes_cli/memory_setup.py`
- `tools/transcription_tools.py`

**Upstream status**: Should be upstreamed — pure bug-fix for Windows, no behavior change on POSIX.

### `ada59ec36` — Fix 10 Windows-failing tests to be cross-platform

**What it does**: Makes 10 test cases pass (or skip gracefully) on Windows:

| Test | Fix |
|---|---|
| `test_make_run_env_appends_homebrew_on_minimal_path` | Skip on Windows (POSIX PATH injection is intentionally skipped there). |
| `test_returns_root_when_only_root_exists` | `os.path.normpath()` the cwd on Windows so forward-slash paths walk up to the filesystem root correctly. |
| `test_close_stdin_allows_eof_driven_process_to_finish` | Use `cat` instead of `python3`; skip when PTY library is missing; pass `str` to winpty and `bytes` to ptyprocess. |
| `test_popen_killed_when_thread_creation_fails` | Only patch `os.getpgid` when it exists (POSIX-only). |
| `test_popen_killed_when_write_checkpoint_fails` | Only patch `os.getpgid` when it exists (POSIX-only). |
| `test_kill_detached_session_uses_host_pid` | Mock `_terminate_host_pid` directly instead of internal `psutil` calls. |
| `test_windows_does_not_call_psutil` | Add `pytest.importorskip("psutil")`. |
| `test_posix_walks_tree_and_terminates_children_then_parent` | Add `pytest.importorskip("psutil")`. |
| `test_posix_no_such_process_swallowed` | Add `pytest.importorskip("psutil")`. |
| `test_posix_oserror_falls_back_to_os_kill` | Add `pytest.importorskip("psutil")`. |

**Files touched**:
- `tests/tools/test_local_env_blocklist.py`
- `tests/tools/test_process_registry.py`
- `tools/environments/local.py`
- `tools/process_registry.py`

**Upstream status**: Should be upstreamed — expands CI coverage to Windows without changing production behavior.

### `1a75a7672` — ~~Auto-install Git-Bash on Windows, transform Windows-style commands to POSIX for bash~~ **DELETED**

**Status**: Removed. Git for Windows auto-install and Git Bash fallback support have been deleted in favor of native PowerShell execution (see P-016). The following files have been removed:
- `tools/environments/_install_git.py`
- `tools/environments/_process_bash_command.py`

Windows platform now requires PowerShell 7 (`pwsh`) or Windows PowerShell (system PowerShell). The shell is resolved via `_find_pwsh` without auto-installation — users are expected to have PowerShell available as part of a standard Windows installation.
