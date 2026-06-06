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

> **P-001** (provider dict-vs-list mismatch in `tui_gateway/server.py`) — **dropped from this fork**. Upstream has since fixed it; the line `user_provs = cfg.get("providers")` in `_apply_model_switch` already does the right thing.

## Release/support changes

These are fork maintenance changes, not runtime behavior patches:

| Area | Target file | What it does |
|---|---|---|
| Upstream sync | `scripts/sync-upstream.sh`, `.github/workflows/upstream-watch.yml`, `MAINTAINING.md` | Keeps upstream syncs on temporary PR branches instead of merging directly into `main` |
| Managed runtime | `.github/workflows/release-runtime.yml`, `scripts/sign_runtime_manifest.py`, `docs/RUNTIME_RELEASES.md` | Builds PyInstaller runtime artifacts, signs manifests, and publishes GitHub Releases consumed by desktop |

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

### `1a75a7672` — Auto-install Git-Bash on Windows, transform Windows-style commands to POSIX for bash

**What it does**:
1. **Auto-install Git for Windows** when the local terminal backend can't find a usable `bash.exe`. Tries, in order: Chocolatey, Scoop, PortableGit self-extracting archive, official Git installer silent setup. Adds the install directory to the user's PATH registry entry.
2. **Transforms Windows-style shell commands to POSIX style** before sending them to bash. Handles drive-letter paths (`C:\foo` → `/c/foo`), backslash directory separators, Windows-specific quoting/escaping, and common Windows-only idioms.

**New files**:
- `tools/environments/_install_git.py` — Download & install strategies for Git on Windows.
- `tools/environments/_process_bash_command.py` — Windows-to-POSIX command translation.

**Files touched**:
- `tools/environments/local.py` — Integrates auto-install and command transformation into the local environment backend.

**Upstream status**: Should be upstreamed — makes the terminal toolset work out-of-the-box on Windows without requiring manual Git installation.
