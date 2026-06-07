# WeChat Relay / 微信中转 platform

This plugin implements the independent Android WeChat relay channel for Hermes.
It is intentionally separate from Hermes' existing `weixin` platform adapter.
Do not import, patch, replace, or configure `weixin` when operating this relay.

Boundary:
- Platform name: `wechat_relay`
- Default transport port: `9797`
- Android WebSocket: `ws://<Mac LAN IP>:9797/_openclaw/notify/ws`
- Health: `GET /_openclaw/notify/healthz`
- Connections: `GET /_openclaw/notify/connections`
- Mock/HTTP inbound: `POST /_openclaw/wechat/inbound`
- Outbound compatibility endpoint: `POST /_openclaw/wechat/send`
- Refuses port `8787`, which belongs to OpenClaw's normal mobile notification path.

Config examples:

```yaml
wechat_relay:
  enabled: true
  host: 127.0.0.1      # use LAN IP or 0.0.0.0 only on a trusted network
  port: 9797
  auto_reply: true
  # Optional. If omitted, outbound replies are pushed to the connected Android
  # relay WebSocket through this same adapter.
  # send_url: http://127.0.0.1:9797/_openclaw/wechat/send
  # shared_secret: "..."
  # allowed_logical_keys: ["wechat-room-1"]
```

Equivalent env vars:
- `WECHAT_RELAY_ENABLED=true`
- `WECHAT_RELAY_HOST=127.0.0.1`
- `WECHAT_RELAY_PORT=9797`
- `WECHAT_RELAY_SEND_URL=...`
- `WECHAT_RELAY_SHARED_SECRET=...`
- `WECHAT_RELAY_ALLOWED_LOGICAL_KEYS=key1,key2`
- `WECHAT_RELAY_ALLOW_ALL_USERS=true`
- `WECHAT_RELAY_HOME_CHANNEL=<logicalKey>`

Session identity:
- Inbound `logicalKey` is normalized to Hermes chat id `wechat:logical:<logicalKey>`.
- `title` / `chatTitle` is display-only and must not be treated as the stable key.

Non-text guardrail:
- Media and placeholders such as `[图片]`, `[文件]`, `[语音]`, `[视频]` are acknowledged as
  `wechat.media_inbound` and are not dispatched to the agent as normal text.
  A safe artifact pipeline can be added later.

Rollback:
- Disable `wechat_relay.enabled` or unset `WECHAT_RELAY_ENABLED`.
- Restart the Hermes gateway.
- Existing `weixin` config/code is untouched and does not need rollback.
