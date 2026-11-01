# 滴滴打车详细工作流程

## 全局执行约束

1. 参数优先级：`用户 query` > `assets/PREFERENCE.md` > 默认值。
2. 每次调用前先核对 [api_references.md](./api_references.md) 参数名。
3. 所有 `--args` 值必须为字符串。
4. `taxi_create_order` 必须使用最近一次预估返回的 `traceId`。
5. 起终点缺失时按优先级补全：① 读 assets/PREFERENCE.md，有地址别名且值非空则推断；② 无可用别名时询问用户当前位置，不得用历史记忆补齐。
6. 用户拒绝提供当前位置时固定回复：不提供当前位置信息则无法满足您的需求。
7. mcporter 使用 URL 直连模式，**禁止添加 `--server` 标志**，**禁止创建或修改 `config/mcporter.json`**。遇到 `StatusCode=400` 先核对参数名（见 SKILL.md §3.2 第 4 条），遇到 JSON 校验错误用 `--config` 绕过（见 SKILL.md §3.2 第 3 条）。

## Phase 1：地址解析

调用：`maps_textsearch`

```bash
MCP_URL="https://mcp.didichuxing.com/mcp-servers?key=$DIDI_MCP_KEY"
mcporter call "$MCP_URL" maps_textsearch --args '{"keywords":"北京西站","city":"北京市"}'
```
调用时需要注意:

- city **必须使用完整格式，如"北京市"而非"北京"**
- keywords 和 city 参数值是否为字符串格式（加引号）

解析规则：

- 用户给完整起终点：直接进入预估。
- "我要上班/下班了"：允许按家↔公司偏好直接解析。
- "回家/去公司"但缺起点：先问当前位置。

## Phase 2：价格预估

调用：`taxi_estimate`

```bash
mcporter call "$MCP_URL" taxi_estimate --args '{"from_lat":"39.894","from_lng":"116.321","from_name":"北京西站","to_lat":"40.053","to_lng":"116.297","to_name":"西二旗"}'
```

执行要点：

- 记录 `traceId`，后续发单必须使用该值。
- 若重新预估，覆盖旧 `traceId`，只认最新一份。

## Phase 3：创建订单

调用：`taxi_create_order`

```bash
mcporter call "$MCP_URL" taxi_create_order --args '{"estimate_trace_id":"TRACE_ID","product_category":"1"}'
```

创建策略：

- 用户已明确车型（当前消息含"叫快车""叫专车"等）时允许直发，不必二次确认车型；用户未指定车型且偏好为空时，**必须先展示预估结果让用户选择车型**，禁止默认选择。
- 用户明确车型时按用户选择发单。
- 用户未明确车型时，可按偏好车型直发；偏好也未配置时，须向用户询问车型，不要自行推荐。
- 若缺少 `estimate_trace_id` 或 `product_category`，禁止发单。
- 若返回 `Streamable HTTP error: Unexpected content type: text/plain`，立即停止流程，按 `error_handling.md` 的「taxi_create_order 调用失败」章节输出固定文案（不要重试、不要切换 Key、不要继续后续步骤）。

成功输出模板：

```text
✅ 订单已创建！

🚖 订单号: [orderId]
📍 [起点] → [终点]
🚗 车型: [车型名称]
💰 预估: 约 [价格] 元
📱 手机尾号: [phoneNumberSuffix]

⏳ 正在为您匹配司机...
💡 发送「查询订单」可了解当前订单状态
⏱️ 将在 5 分钟后自动为您回查订单状态
```

## Phase 4：查询订单

触发词：`查询订单` / `查询订单 <orderId>`

调用：`taxi_query_order`

```bash
# MCP_URL 沿用 Phase 1 已定义的变量
mcporter call "$MCP_URL" taxi_query_order --args '{"order_id":"ORDER_ID"}'
```

### 状态码与输出格式

| code | 含义 | 输出建议 |
|------|------|----------|
| 0 | 匹配中 | ⏳ 正在为您匹配司机，请稍候 |
| 1 | 司机已接单 | 展示司机信息（姓名、车型、车牌、电话）及距上车点距离/ETA |
| 2 | 司机已到达 | 🔔 司机已到达上车点，请前往上车 |
| 4 | 行程进行中 | 🚗 行程已开始，祝您旅途愉快 |
| 5 | 订单完成 | ✅ 行程结束，展示费用（如有） |
| 6 | 订单已被系统取消 | ❌ 订单已被系统取消 |
| 7 | 订单已被取消 | ❌ 订单已取消 |
| 3/8-12 | 其他终态 | 显示对应状态描述 |

## Phase 5：取消订单

调用顺序：

1. 向用户确认是否取消（即使用户说了"取消订单"，仍需明确询问"确认取消吗？"，用户的取消意图 ≠ 取消确认）；
2. 用户确认后调用 `taxi_cancel_order`；
3. 调用 `taxi_query_order` 确认取消结果。

## Phase 6：预约出行（cron 托管）

说明：MCP API 是实时发单，预约由 OpenClaw cron 托管叫车需求，到点由 isolated agent 独立执行完整打车流程。

```bash
# ⚠️ 替换占位符：
#   FROM_NAME    → 带城市前缀的起点全称（如"北京市西二旗地铁站"）
#   TO_NAME      → 带城市前缀的终点全称（如"北京市佰嘉城小区"）
#   VEHICLE      → 车型（如"快车"）
#   TIME         → 见下方时间规则
#   CHANNEL_NAME → 当前会话 metadata 中的 channel 字段（如 feishu、telegram），CHANNEL_NAME 不需要带引号，例如: feishu ✅, "feishu" ❌。不允许使用 last 作为参数值。
#   CHAT_ID      → 当前会话 metadata 中的 chat_id 字段

openclaw cron add \
  --name "didi-ride-skill:$(date +%s)" \
  --at "TIME" \
  --session isolated \
  --message "执行定时打车：起点「FROM_NAME」，终点「TO_NAME」，车型「VEHICLE」。请完整执行打车流程：地址解析 → 价格预估（获取最新 traceId）→ 创建订单。订单创建成功后，输出订单信息并提示用户可发送「查询订单」了解订单状态，同时创建 5 分钟后自动回查 cron（模板见 SKILL.md 第 3.8 节「发单后自动回查」）。" \
  --announce \
  --channel CHANNEL_NAME \
  --to "CHAT_ID"
```

### TIME 填写规则

| 场景 | 写法 | 示例 |
|------|------|------|
| 相对时间（X 分钟/小时后） | duration 格式 | `15m` / `2h` / `1h30m` |
| 绝对时间（具体时刻） | 本地时区 ISO 格式 | `$(date -d '明天 09:00' '+%Y-%m-%dT%H:%M:%S+08:00')` |

- 相对时间（如 `15m`）无需格式化，直接使用
- 绝对时间使用带时区的 ISO 8601 格式：`YYYY-MM-DDTHH:MM:SS+08:00`（北京时间东八区）

**系统兼容性说明：**
- Linux (GNU date): `date -d '明天 09:00' '+%Y-%m-%dT%H:%M:%S+08:00'`
- macOS (BSD date): `TZ=Asia/Shanghai date -j -v+1d -f '%H:%M' '09:00' '+%Y-%m-%dT%H:%M:%S+08:00'`

## 查询司机位置

调用顺序：

1. `taxi_get_driver_location`
2. `maps_regeocode`

返回内容建议包含：地址、距离、预计到达时间、司机信息（如可得）。
