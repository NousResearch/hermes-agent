# Hermes T Runtime

`hermes_t` 是第二阶段开始后的**通用入口 facade**。

当前定位：
- 默认仍兼容欧林 profile（`olin-688319`）
- 但对外入口语义已经去欧林化
- CLI 默认工作目录改为 `~/.hermes_t_runtime`
- `hermes_t` / `hermes_olin` 已共用 shared CLI builder
- 已提供多 profile orchestrator 骨架（JSON config -> 多 `RuntimeProfile` 顺序执行）
- 已提供最小 `TechDataProvider` 插件层：支持 fixed fallback、`--tech-data-config` JSON provider、`--quote-data-config` -> `quote_payload.tech_data` 适配，以及 `--quote-snapshot-config` -> `quote snapshot/real source` 适配
- 已提供多 profile CLI 编排入口（`--profiles-config` + 可选 `--tech-data-config` / `--quote-data-config` / `--quote-snapshot-config`）
- 内部继续复用已验证的 `RuntimeProfile` / `TradingStateStore` / `run_runtime_cycle`

## 推荐入口

### 单 profile

```bash
cd /Users/wj/hermes-agent
python -m hermes_t --trade-date 20260501 --signal hold
```

### 多 profile 编排

```bash
cd /Users/wj/hermes-agent
python -m hermes_t \
  --profiles-config ./profiles.json \
  --tech-data-config ./tech_data.json \
  --trade-date 20260501
```

### quote/realtime 适配输入

```bash
cd /Users/wj/hermes-agent
python -m hermes_t \
  --quote-data-config ./quote_data.json \
  --trade-date 20260501
```

### quote snapshot / live source 适配输入

```bash
cd /Users/wj/hermes-agent
python -m hermes_t \
  --quote-snapshot-config ./quote_snapshot_config.json \
  --trade-date 20260501
```

其中：
- `profiles.json`：定义多个 `profile_id/symbol/trade_unit/max_trades`
- `tech_data.json`：可选，定义 `symbol -> tech_data`
- `quote_data.json`：可选，定义 `symbol -> quote payload`，当前最小口径只读取其中的 `tech_data` 子字段
- `quote_snapshot_config.json`：可选，可为原始 `symbol -> snapshot` map，或 factory config（如 `{"source":"file"}` / `{"source":"mock"}` / `{"source":"tdx"}`）
- 若某 symbol 未在 `tech_data.json` / `quote_data.json` / `quote_snapshot_config.json` 中产出可用 `tech_data`，则回退到 CLI 传入的 `--signal/--score`
- 若同时提供 `--tech-data-config` 与 `--quote-data-config`，当前优先使用 `--tech-data-config`
- 若同时提供 `--quote-data-config` 与 `--quote-snapshot-config`，当前优先使用 `--quote-data-config`

## quote snapshot source 当前口径

当前 factory 支持：
- `file`：从 JSON 文件读取 snapshot map
- `mock`：从 config 内联 `snapshots_by_symbol` 读取
- `tdx`：最小真实源，使用 `pytdx` 通过 TDX TCP 拉取实时 quote snapshot
- `eastmoney`：仅占位，当前 `get()` 仍会抛 `NotImplementedError`

### TDX 依赖

若要使用 `{"source":"tdx"}`：
- 项目依赖已纳入 `pytdx>=1.72,<2`
- 推荐在项目根目录执行：

```bash
cd /Users/wj/hermes-agent
uv sync
```

### TDX 失败语义（当前正式口径）

- `TdxQuoteSnapshotSource.get(symbol)` 自身仍保持严格语义：
  - `pytdx` 不可用 -> 抛 `RuntimeError("pytdx unavailable")`
  - 全部 server 连接失败 / 空 quote / `price <= 0` -> 抛运行时错误
- 但 `QuoteSnapshotTechDataAdapter` 在 runtime 适配层会兜底：
  - 若上游 snapshot source 抛异常，则回退到默认 `tech_data`
  - 因此 `--quote-snapshot-config {"source":"tdx"}` 在运行期临时取数失败时，不会直接把 runtime 主链打崩，而是回退到默认信号口径

## 当前边界

这还不是最终的多股票自动编排系统；当前只是把正式入口从“欧林专用包”推进到“通用 facade + 欧林兼容层 + 最小多 profile 编排入口”。

当前 `quote/realtime` 插件化仍是**最小适配层**：
- `run_runtime_cycle()` 继续只消费 `tech_data`
- `hermes_t` 入口层负责把外部 quote/realtime 原始输入适配成 `tech_data`
- 后续若接 TDX / 东财 / 真实实时源，再在这一层继续扩 provider，而不侵入 runtime 主链

## 兼容关系

- `hermes_t`：今后优先使用的通用入口
- `hermes_olin`：保留，继续服务现有欧林兼容调用
