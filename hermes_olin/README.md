# Hermes Olin Runtime (minimal, profile-aware)

这是 Hermes 主仓内从欧林做T抽出来的最小正式 runtime 内核；当前默认 profile 仍是欧林，但状态存储与 runtime 主链已经支持按 profile 复用。

> 第二阶段开始后，**通用入口优先使用 `hermes_t`**；本目录 `hermes_olin` 继续保留为欧林兼容入口与默认欧林口径文档。

- `execution_state.json`：执行账本事实源
- `pending_signal.json`：待 dispatch 信号槽
- `push_state.json`：最近一次 dispatch 摘要
- `signal_send_history.jsonl`：发送流水
- `dispatch_ledger.jsonl`：dispatch ledger

当前最小可运行闭环：

1. `build_execution_suggestion()` 根据执行状态计算下一笔建议
2. `stage_pending_signal()` 把建议挂起为 pending，并同步 `active_signal`
3. `recover_signal_runtime()` 在每个 cycle 开始前执行最小自愈：清孤儿 active signal、跨日过期、超时升级、不可重试失败升级
4. `deliver_pending_signal()` 在需要时发出 pending，并自动走 dispatch-confirm
5. `confirm_dispatch_sent()` 在 sent 后提交账本、更新 push state、清空 pending
6. `run_runtime_cycle()` 串起单次 cycle，可直接作为 Hermes 自有最小入口

## 直接运行

如果你现在是按“通用做T系统”口径启动，优先用新的 generic facade：

```bash
cd /Users/wj/hermes-agent
uv run python3 -m hermes_t --signal sell --score 20 --trade-date 20260422
```

`hermes_olin` 则继续保留给默认欧林兼容入口。

默认 profile 仍保持欧林兼容值：

- `profile_id=olin-688319`
- `symbol=688319`
- `trade_unit=10000`
- `max_trades=4`

```bash
cd /Users/wj/hermes-agent
uv run python3 -m hermes_olin --signal sell --score 20 --trade-date 20260422
```

也可以显式切到其他标的 profile；第一阶段已经支持把运行时画像从欧林默认值中抽离出来：

```bash
uv run python3 -m hermes_olin \
  --profile-id demo-600519 \
  --symbol 600519 \
  --trade-unit 200 \
  --max-trades 6 \
  --signal sell \
  --score 20 \
  --trade-date 20260501
```

显式指定发送目标时可直接带：

```bash
uv run python3 -m hermes_olin --signal sell --score 20 --trade-date 20260422 --dispatch --chat-id oc_xxx --thread-id omt_xxx
```

目前 dispatch channel 仍只支持 `feishu`；`--channel` / `HERMES_OLIN_CHANNEL` 现阶段主要用于显式声明该默认值。

也支持环境变量回退：

- `HERMES_OLIN_CHANNEL`（当前仅支持 `feishu`）
- `HERMES_OLIN_CHAT_ID`
- `HERMES_OLIN_THREAD_ID`

默认只做本地状态演算，不真实发送；输出单次 cycle 的 JSON 结果。CLI 默认 `--base-dir` 为稳定的 home 目录口径：

- `~/.hermes_olin_runtime/`

各 profile 的状态文件会按独立命名空间落盘：

- 默认欧林：`~/.hermes_olin_runtime/profiles/olin-688319/state/realtime/`
- 其他 profile：`~/.hermes_olin_runtime/profiles/<profile-id>/state/realtime/`

加 `--dispatch` 时，会尝试通过 Hermes gateway 走真实 dispatch；是否真的发到 Feishu 取决于当前 gateway 可用性以及目标 chat/channel 配置是否可解析。

现阶段 `--base-dir` 仍建议遵守两条口径：

1. 省略时，走稳定的 `Path.home() / ".hermes_olin_runtime"` 默认值，不随 `cwd` 漂移。
2. 做隔离验证/无扰动联调时，显式传独立目录，例如 `/tmp/hermes_olin_probe`，不要直接写入正式状态树。

## 当前最小自愈口径

`run_runtime_cycle()` 现在会先执行 `recover_signal_runtime()`，当前正式规则如下：

- `pending_signal` 不存在但 `active_signal` 还挂着：清掉孤儿 `active_signal`
- `pending_signal.trade_date != effective_trade_date`：标记 `expired_cross_day`，并清掉匹配的 `active_signal`
- `pending_signal` 超过 300 秒仍处于 `pending/failed`：标记 `timed_out`
- `pending_signal.status=failed` 且 `last_error_retryable=false`：标记 `failed_exhausted`
- `pending_signal.status=failed` 且仍有剩余重试次数：恢复为 `pending`，同次 cycle 允许继续走重试 dispatch
- 当 recovery 结果为 `timed_out` 或 `failed_exhausted` 时，当次 cycle 不再重建新信号，而是返回 `suggestion.next_action=hold` 与 `reason=recovery_blocked`
- 当 recovery 刚把 retryable failed pending 重新排回 `pending` 时，当次 cycle 直接复用该 `pending_signal`；dry-run 返回里保留原 failed 结果对象，store 内状态则已恢复为 `pending`
- 当日已经发过同方向信号、且当前没有新的 pending 时，不会仅凭同一 `summary_signal` 直接推进到下一笔；当次 cycle 返回 `reason=duplicate_sent_signal`
- 当 recovery 结果为跨日过期时，允许当次 cycle 在清理后继续按新交易日生成新 pending

## 验证

```bash
cd /Users/wj/hermes-agent
/Users/wj/.hermes/hermes-agent/venv/bin/python3 -m pytest tests/test_hermes_olin_runtime.py -q
```

当前定向回归：`60 passed`

## 当前推进阶段（2026-05-01）

这轮第一阶段通用化 + 第二阶段入口去欧林化首批工作已经落到正式代码：

- 已抽出 `RuntimeProfile`
- 已引入 `TradingStateStore`，按 `profile_id` 隔离状态目录
- `build_execution_suggestion()` / `stage_pending_signal()` / `run_runtime_cycle()` 已改为优先读取 profile 的 `trade_unit` / `max_trades`
- CLI 已支持 `--profile-id` / `--symbol` / `--trade-unit` / `--max-trades`
- 已新增 `hermes_t` 作为通用 facade 包与 generic CLI 入口
- `python -m hermes_t` 默认使用 `~/.hermes_t_runtime/`，并统一走 `TradingStateStore`
- `hermes_olin` / `hermes_t` CLI 已改为共用 shared CLI builder，避免参数解析逻辑双份漂移
- 已新增多 profile orchestrator 骨架，可从 JSON 配置加载多个 `RuntimeProfile` 并逐个执行 runtime cycle
- 默认欧林口径保持不变，兼容值仍是 `688319 / 10000 股 / 最多 4 笔`

当前仍属于“通用 runtime 骨架的第二阶段”，还没有完成：

- 内部模块/测试命名与文档口径的彻底去欧林化
- 实时行情输入与策略插件化
- 日级参数自适应进化、自愈守护闭环
- 完整盘中生产调度链

结论口径：Hermes 自有侧已经从“欧林专用最小 runtime”推进到“默认兼容欧林、但底层已 profile-aware 的最小正式内核”，不过离可迁移的完整通用做T系统还有后续阶段。

这个目录仍然不接 OpenClaw 线上 daemon、实时 quote、runtime panel、自愈恢复全链路；当前只是在 Hermes 自有侧先立住正式最小状态核与最小自愈语义。
