---
name: evotraders-stock-analysis
description: 使用 EvoTraders WinAPI 工具链（tqcenter + upstream proxy）进行 A 股个股快照/指标/走势分析，避免纯网页检索。
version: 0.2.0
platforms: [linux, windows]
required_environment_variables:
  - name: EVOTRADERS_WINAPI_BASE
    prompt: WinAPI 基础地址（例如 http://192.168.100.168:18880）
    help: 需要在 Windows 机器上运行 evotraders/winapi/server.py
metadata:
  hermes:
    tags: [stocks, a-share, quant, evotraders]
---

## 目标

当用户让你分析/诊断 A 股个股（如“分析 300111”）时，优先使用 **EvoTraders 工具链**获得结构化数据与可复现依据，而不是只做网页资料汇总。

你可用的关键工具：
- `evotraders_health`: 检查 WinAPI 中转是否可用
- `evotraders_tq_call`: 调用 `tqcenter` 或自动回退到 upstream Evo bridge（支持 `formula_*`）
- `evotraders_proxy_call`: 透传任意 upstream Evo API（用于扩展端点）
- `evotraders_market_mainline`: A 股主线情绪上下文快捷接口（proxy-first）

## 工作流（必须遵守）

1) **连通性**：先调用 `evotraders_health`。不可用时，明确告知用户需要配置/启动 WinAPI。

2) **标的规范化**：
   - 输入如果是 `300111`，默认当作 A 股，优先尝试 `300111.SZ`，不行再尝试 `300111.SH`。
   - 输出中必须明确你最终使用的代码（例如 `300111.SZ`）。

3) **最小必需数据（至少做这些）**：
   - **行情快照**：`evotraders_tq_call(method="get_market_snapshot", params={...})`
   - **K线序列**：`evotraders_tq_call(method="formula_kline_series", params={...})`（默认日线，count=60）
   - **指标序列**：`evotraders_tq_call(method="formula_indicator_series", params={...})`（至少 MACD/CCI/换手）

4) **结论输出格式（必须结构化）**：
   - **结论**：一句话方向（偏强/偏弱/震荡）+ 关键理由
   - **关键数据**：列出 5-10 个核心字段（现价/涨跌/量额/近 N 日高低/均线/指标拐点）
   - **技术面**：趋势、支撑/压力、量价配合、指标共振/背离
   - **风险提示**：至少 3 条（如量能不足、假突破、消息面不确定等）
   - **下一步观察**：给出 2-3 个可执行的观察条件（例如“放量突破 xx”）

## 推荐参数模板

- 快照：
  - `{"stock_code":"000001.SZ"}`
  - 如果调用方传了 `stock_list`，工具/中转会做兼容映射。

- 日线 K 线（默认）：
  - `{"stock_code":"000001.SZ","period":"D","count":60}`

- 指标序列（默认）：
  - `{"stock_code":"000001.SZ","count":60}`

## 失败处理（不要瞎编）

- 若 `tq` 方法不存在或参数签名不匹配：`evotraders_tq_call` 会自动回退到 upstream proxy；你只需继续用同一工具重试一次并记录返回的 `relay`/`url` 信息（如果返回里有）。
- 若仍失败：用 `evotraders_proxy_call` 调 `v1/tq/methods` / `v1/health` 做诊断，并在回答中给出“哪一步失败 + 返回的 error/状态码”。
- 若 WinAPI 链路持续失败：允许降级到公网信息源（`web_search`/`web_extract`）补充“主线线索”，但必须明确标注“公网替代口径（非通达信实时链路）”，并说明恢复后需要补哪些 Evo 证据。

