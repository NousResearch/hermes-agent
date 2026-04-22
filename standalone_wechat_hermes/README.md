# wechat-hermes-bridge

Standalone Python package extracted from **Hermes Agent**: connects to **WeChat personal accounts** via Tencent **iLink Bot API** (`ilink/bot/getupdates` long-poll), the same implementation as `gateway/platforms/weixin.py`.

This repo folder is **not** the full Hermes agent (no SQLite sessions, tools, skills, or full gateway). It ships:

- `wechat_hermes.weixin_adapter` — full `WeixinAdapter` copy
- Minimal copies of `BasePlatformAdapter`, gateway config/session types, helpers, URL safety
- Optional echo / OpenAI demo CLI (`wechat-hermes-bridge`)

## Install

```bash
cd standalone_wechat_hermes
pip install -e ".[dev]"
```

## Configure

Required environment variables (same as Hermes gateway):

| Variable | Meaning |
|----------|---------|
| `WEIXIN_TOKEN` | iLink bot token |
| `WEIXIN_ACCOUNT_ID` | Account id |
| `HERMES_HOME` | Optional; defaults to `~/.hermes` for context-token cache files |

Optional:

| Variable | Meaning |
|----------|---------|
| `CHAT_MODE` | `echo` (default) or `llm` |
| `OPENAI_API_KEY` | Required when `CHAT_MODE=llm` |
| `OPENAI_MODEL` | Default `gpt-4.1-mini` |
| `OPENAI_BASE_URL` | Compatible API base URL |

## Run

```bash
export WEIXIN_TOKEN=...
export WEIXIN_ACCOUNT_ID=...
export CHAT_MODE=echo   # or llm + OPENAI_API_KEY
wechat-hermes-bridge
```

## Integrate your own backend

Implement an async handler:

```python
async def handle(event: MessageEvent) -> str | None:
    ...
adapter.set_message_handler(handle)
await adapter.connect()
```

The handler receives normalized Hermes `MessageEvent`; return text to send back on WeChat.

## Legal / ToS

You are responsible for complying with Tencent / WeChat terms of service and applicable law when using iLink or automation.

## Full Hermes Agent on WeChat only (recommended for “whole agent”)

For the **complete** gateway stack — `AIAgent`, SQLite sessions, tools/skills,
slash commands, memory — use the **main Hermes Agent** repo and run::

```bash
export HERMES_GATEWAY_WECHAT_ONLY=1
export WEIXIN_TOKEN=...
export WEIXIN_ACCOUNT_ID=...
hermes gateway
```

Setting `HERMES_GATEWAY_WECHAT_ONLY` removes every **other** messaging platform from
gateway config after env merges; only Weixin stays enabled.

This folder (`standalone_wechat_hermes`) remains a **thin SDK / echo demo** without
the full runner.

## Source

Derived from [Hermes Agent](https://github.com/NousResearch/hermes-agent) (`gateway/platforms/weixin.py` and dependencies). Hermes upstream remains the authoritative source for bugfixes.
