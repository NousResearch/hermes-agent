# Dashboard 会话令牌固定配置说明

本文记录 Hermes Agent **Web Dashboard**（`hermes dashboard`）会话令牌 **`_SESSION_TOKEN`** 可通过环境变量固定的行为，以及与 **Hermes Workspace** 的对接方式。

## 背景

- 默认情况下，Dashboard 每次进程启动会生成随机会话令牌，并注入到 SPA 首页（`window.__HERMES_SESSION_TOKEN__`），REST/API 请求使用 `Authorization: Bearer …` 或 `X-Hermes-Session-Token`。
- 集成方（例如 Hermes Workspace）若不想依赖抓取首页 HTML，可将令牌设为 **跨重启不变的共享密钥**。

## Agent / Dashboard 侧

| 环境变量 | 作用 |
|----------|------|
| `HERMES_DASHBOARD_SESSION_TOKEN` | 可选。设为 **≥16 字符**、且通过占位符校验的非空字符串时，用作 **`_SESSION_TOKEN`**（固定会话密钥）。未设置或不合规时，行为与原先一致：**每次启动随机生成**。 |

配置位置：`~/.hermes/.env`，或由 systemd / Docker / `hermes dashboard` 启动脚本注入进程环境（与 `API_SERVER_KEY` 用法类似）。

校验逻辑：`hermes_cli.auth.has_usable_secret(value, min_length=16)`（含过短拒绝与常见占位符拒绝）。

SPA 注入：首页脚本中为 `window.__HERMES_SESSION_TOKEN__=` 使用 **`json.dumps(token)`**，避免自定义令牌中含引号等字符破坏页面脚本。

实现代码：[hermes_cli/web_server.py](../hermes_cli/web_server.py) 中的 `_resolve_dashboard_session_token()` 与 `mount_spa` 内 `_serve_index`。

配置注册：[hermes_cli/config.py](../hermes_cli/config.py) → `OPTIONAL_ENV_VARS["HERMES_DASHBOARD_SESSION_TOKEN"]`。

官方文档索引：

- [环境变量参考](../website/docs/reference/environment-variables.md)（Messaging 一节表格）
- [Web Dashboard 功能说明](../website/docs/user-guide/features/web-dashboard.md)（REST API 节前会话鉴权段落）

## Hermes Workspace 侧

| 环境变量 | 作用 |
|----------|------|
| `HERMES_DASHBOARD_TOKEN` | Workspace 调用 Dashboard **`HERMES_DASHBOARD_URL`** 下受保护 `/api/*` 时携带的 Bearer；应与 **`HERMES_DASHBOARD_SESSION_TOKEN` 完全一致**。 |

可选：`HERMES_DASHBOARD_URL` 指向 Dashboard 基地址（例如 `http://127.0.0.1:9112`）。

若未设置 `HERMES_DASHBOARD_TOKEN`，Workspace 可能回退为请求 Dashboard 根路径 `/` 并从 HTML 解析 `__HERMES_SESSION_TOKEN__`（遗留行为）；显式配置两端同名密钥可避免重启后令牌漂移带来的对接问题。

## 与 Gateway API 密钥的区别

| 用途 | Agent / env | Workspace / env |
|------|-------------|-----------------|
| OpenAI 兼容 HTTP API（默认 `:8642`） | `API_SERVER_KEY` | `HERMES_API_TOKEN`（须一致） |
| Web Dashboard HTTP API | `HERMES_DASHBOARD_SESSION_TOKEN` | `HERMES_DASHBOARD_TOKEN`（须一致） |

两套密钥 **不可混用**：Workspace 源码中注明 Dashboard 与 Gateway 使用独立的 Bearer 方案。

## 运维步骤摘要

1. 生成足够长的随机串（例如 `openssl rand -hex 32`）。
2. 写入 `~/.hermes/.env`：`HERMES_DASHBOARD_SESSION_TOKEN=<同一字符串>`。
3. 在 Workspace `.env`（或容器环境）中：`HERMES_DASHBOARD_TOKEN=<同一字符串>`。
4. **重启** `hermes dashboard` 与 Workspace 进程使环境变量生效。

## 安全提示

- 固定令牌等价于长期密码；监听 `0.0.0.0` 或非可信网络时风险更高，建议最小暴露面并结合防火墙或反向代理。
- 令牌泄露将导致他人具备与浏览器会话等效的 Dashboard API 访问能力（读写配置与环境变量等敏感能力）。

## 测试与回归

单元测试：[tests/hermes_cli/test_dashboard_session_token.py](../tests/hermes_cli/test_dashboard_session_token.py)（`pytest` + `monkeypatch` 覆盖 `_resolve_dashboard_session_token()`）。

运行示例：

```bash
cd /path/to/hermes-agent
uv run pytest tests/hermes_cli/test_dashboard_session_token.py -v
```
