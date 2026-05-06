---
name: tqlex-function-playbook
description: 基于 TQLEX 详细文档的函数调用手册（Entry/Params/body 契约、wendaQuery/InfoSelectV2、常见模板与失败兜底）。
version: 0.2.0
platforms: [linux, windows]
required_environment_variables:
  - name: TDX_API_DATA_ENDPOINT
    prompt: TQLEX 公网枢纽地址（例如 http://tdxhub.icfqs.com:7615/TQLEX）
    help: TQLEX 公网直连地址，可选；默认使用 tdxhub.icfqs.com:7615
  - name: TDX_API_KEY
    prompt: TQLEX API Token（请求头 token 字段）
    help: 部分环境可无 token 探测，生产建议配置
optional_environment_variables: []
metadata:
  hermes:
    tags: [tqlex, tdx, wenda, infoselect, params, function-chain]
---

## 目标

当用户要求使用 TQLEX 能力时，优先按"函数链路"调用，而非只走 UI 端点。
本技能用于统一三类契约：

1) Params 数组契约：`POST .../TQLEX?Entry=...` + `{"Params":[...]}`
2) 行情/K 线结构化 body 契约：如 `TdxShare.PBHQInfo` / `TdxShare.PBFXT`
3) NLP 契约：`JNLPSE:wendaQuery`（问达）与 `NLPSE:InfoSelectV2`（自然语言指标）

---

## 工具映射（Hermes 侧）

**TQLEX 唯一链路：公网直连**

- **地址**：`http://tdxhub.icfqs.com:7615/TQLEX`（可通过 `TDX_API_DATA_ENDPOINT` 覆盖）
- **鉴权**：`TDX_API_KEY` 环境变量 → 请求头 `token`
- **实现**：`/opt/evotraders/backend/tdx_http/tqlex_client.py` + `tqlex_public_payload.py`
- **特点**：直连外网枢纽，不经过 WinAPI

---

### 工具选择优先级

1. **问达选股**：`evotraders_wenda_query`（优先函数链 `tdx_wenda_query_tool`）
2. **自然语言指标**：`evotraders_indicator_select`（优先函数链 `tdx_indicator_select_tool`）
3. **通用 TQLEX**：直接调用公网枢纽（`TDX_API_DATA_ENDPOINT`）
4. **兜底**：`evotraders_proxy_call`（仅当以上都不可用）

---

## 关键硬规则

1) **先区分通道**：`wendaQuery` 与 `InfoSelectV2` 不是一回事，不能混用。  
2) **Params 顺序不可乱**：TQLEX 很多 Entry 对参数顺序敏感。  
3) **优先函数链**：问达/指标优先 `tdx_wenda_query_tool` / `tdx_indicator_select_tool`。
4) **失败要可审计**：输出中必须写明 `entry`、`params/body`、状态码/错误信息。  
5) **不要编造字段**：仅基于返回的 `ResultSets` / `response` 下结论。

---

## 常用模板

### A) 问达选股（JNLPSE:wendaQuery）

优先：
- `evotraders_wenda_query(message="涨停", rang="AG", page_no=1, page_size=30)`

通用 body 形态（仅排障时手工）：
- `full_body_json` 为单元素数组：
  - `[{"message":"涨停","rang":"AG","pageNo":"1","pageSize":"30"}]`

---

### B) 指标查询（NLPSE:InfoSelectV2）

优先：
- `evotraders_indicator_select(query="上证指数 今日涨跌家数 市场情绪")`

注意：
- 单词"市场情绪"常不足以稳定命中，建议使用更完整自然语言问句。

---

### C) Params 入口（示例：研报评级）

- `entry`: `TdxSharePCCW.tdxf10_gg_ybpj`
- `params_json`: `["000001","yzyq"]`

调用：
- `evotraders_tqlex_public_call(entry="TdxSharePCCW.tdxf10_gg_ybpj", params_json="[\"000001\",\"yzyq\"]")`

---

### D) 交易数据（示例：资金流向）

- `entry`: `TdxSharePCCW.tdxf10_gg_jyds`
- 常见 `fixedTag`: `zjlx` / `rzrq` / `zrq` / `ztfx` / `dtfx`
- 常见 Params 形态：`[code, fixedTag, extra]`

---

## 输出规范

每次 TQLEX 取数后，回答里至少带：

- `data_plane`（若有）
- `entry`（或具体函数名）
- `params/body`（可脱敏但结构要保留）
- 成功/失败与关键返回字段（如 `ErrorCode`、`ResultSets` 数量）
- 基于证据的简洁结论（不要只贴原始 JSON）

---

## 失败兜底顺序

1) 同一工具重试一次（保留错误上下文）
2) 问达/指标走对应函数链工具（若之前走的是通用入口）
3) `evotraders_tqlex_public_call` 与 `evotraders_proxy_call` 交叉验证
4) 若仍失败，输出"失败步骤 + 错误码 + 下一步排障建议"

---

## 参考来源

- `/opt/evotraders/backend/tdx_http/tqlex_client.py` — TQLEX 公网直连客户端
- `/opt/evotraders/backend/tdx_http/tqlex_public_payload.py` — 公网 Payload 构建
- `/opt/hermes-agent/tools/evotraders_tool.py` — Hermes 工具封装
- `evotraders/backend/tools/TQLEX-API-capability-analysis.md`
- `evotraders/docs/tdx_tqlex_mcp_capabilities.md`
