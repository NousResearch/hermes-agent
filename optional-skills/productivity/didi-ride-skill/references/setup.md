# DiDi MCP Server 安装配置指南

本文档说明如何安装和配置 DiDi MCP Server，以便使用 didi-ride-skill skill。

## 1. 安装 mcporter

mcporter 是用于调用 MCP Server 的命令行工具。

```bash
npm install -g mcporter
```

验证安装：
```bash
mcporter --version
```

## 2. 获取 MCP KEY

访问 https://mcp.didichuxing.com/claw 获取您的 MCP KEY，或扫描下方二维码直达官网注册页面。

![滴滴出行APP扫码获取MCP Key，解锁一键打车](https://s3-yspu-cdn.didistatic.com/mcp-web/qrcode/didi_ride_skill_qrcode.png)


## 3. 配置 MCP KEY

**推荐方式：通过 OpenClaw 持久化（所有 isolated session 自动生效）**

```bash
openclaw config set 'skills.entries.didi-ride-skill.apiKey' 'YOUR_MCP_KEY'
```

**备用方式：环境变量（仅当前 shell 会话有效）**

```bash
export DIDI_MCP_KEY="YOUR_MCP_KEY"
```

## 4. 验证连接

> 本 skill 使用 mcporter 的 **URL 直连模式**，不依赖 `config/mcporter.json` 配置文件。如果 mcporter 启动时报 JSON 校验错误（`invalid_type` 或 `Failed to parse JSON`），说明存在格式异常的 `config/mcporter.json`。不要删除它（可能包含其他应用配置），改用 `--config` 绕过——详见 SKILL.md §3.2 第 3 条。

**Step 1 — 查看所有工具签名（同时验证 Key 和连通性）：**

```bash
MCP_URL="https://mcp.didichuxing.com/mcp-servers?key=$DIDI_MCP_KEY"
mcporter list "$MCP_URL"
```

预期输出完整的工具列表（`maps_textsearch`、`taxi_estimate` 等 13 个工具及参数签名）。如果返回鉴权错误，说明 Key 无效。

**Step 2 — 测试地址解析功能：**

```bash
mcporter call "$MCP_URL" maps_textsearch --args '{"keywords":"西二旗地铁站","city":"北京市"}'
```

**Step 3 — 测试价格预估功能：**

```bash
mcporter call "$MCP_URL" taxi_estimate --args '{"from_lng":"116.322","from_lat":"39.893","from_name":"北京西站","to_lng":"116.482","to_lat":"40.004","to_name":"首都机场"}'
```

## 5. OpenClaw 配置

如果使用 OpenClaw，还需要以下配置：

- 确保已安装 `openclaw` CLI
- 验证 mcporter 已安装：`which mcporter`
- 若未找到，执行 `npm install -g mcporter` 后重新验证

## 常见问题

**Q: MCP KEY 无效**
A: 请检查 MCP KEY 是否正确，以及是否已启用相应权限

**Q: 调用超时**
A: 检查网络连接，稍后重试

**Q: 地理位置限制**
A: 部分功能仅支持中国大陆地区
