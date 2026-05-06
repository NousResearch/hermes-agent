---
name: a-share-mainline-analysis
description: 使用 EvoTraders 工具链优先分析 A 股市场主线，失败时降级公网线索并显式标注口径。
version: 0.1.0
platforms: [linux, windows]
required_environment_variables:
  - name: EVOTRADERS_WINAPI_BASE
    prompt: WinAPI 基础地址（例如 http://192.168.100.168:18880）
    help: 需要在 Windows 机器上运行 evotraders/winapi/server.py
metadata:
  hermes:
    tags: [a-share, mainline, market, evotraders, sentiment]
---

## 目标

当用户询问“接下来 A 股主线是什么 / 当前主线板块 / 市场主线轮动”时，优先使用 EvoTraders 数据链路给出结构化判断。

## 必须调用的工具顺序

1) `evotraders_health`：先检查链路可用性  
2) `evotraders_market_mainline(max_age_sec=60)`：优先拉取主线情绪上下文  
3) 如需补证据，再调用：
   - `evotraders_wenda_query(message="涨停")` / `evotraders_wenda_query(message="连板")`
   - `evotraders_indicator_select(query="主线 强势板块 资金流入", topk=10)`
   - `evotraders_iwencai_query(query="今日主线板块资金净流入排名")`（问财链路）
   - `evotraders_ths_bigorder(code="000001")`（同花顺大单）
   - `evotraders_proxy_call(subpath="v1/quant/runtime/metrics")`

## 失败降级策略

如果 `evotraders_*` 连续失败（网络、上游 5xx、方法不存在）：
- 明确写出失败环节与状态码；
- 使用公网数据源（`web_search`/`web_extract`）补充“主线线索”；
- 在结论处明确标注：**公网替代口径（非通达信实时链路）**；
- 给出恢复 Evo 链路后需要补的验证项（资金流、量价、龙头联动）。

## 输出模板

1) 主线判断：1-2 句（主线/次主线/混沌）  
2) 证据清单：3-5 条（工具名 + 关键字段）  
3) 交易观察点：2-3 条（触发条件）  
4) 风险：至少 3 条（风格切换、缩量、消息扰动等）  
