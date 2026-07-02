# Hermes Agent 0.18 升级评审报告

日期：2026-07-02

## 升级结论

已将本地 `main` rebase 到上游正式发布 `v2026.7.1`，对应 Hermes Agent `v0.18.0`。生产目录保持不变，仍为 `/home/liao/.hermes/hermes-agent`；user systemd 服务 `hermes-gateway.service` 已刷新并重启。

当前 CLI 版本基线：

```text
Hermes Agent v0.18.0 (2026.7.1)
```

说明：`origin/main` 在 release tag 后还有 1 个 post-release 提交；本次按“0.18 正式版”要求停在 `v2026.7.1` 基线上。

## 本地改动保留判断

### Feishu / Gateway media

结论：保留，并已补一个 rebase 后发现的修复。

- Feishu Markdown table 原生卡片未被 0.18 覆盖。0.18 基础实现仍偏向 plain text fallback，本地实现能生成 Feishu native table card。
- `reply_in_thread=False`、编辑消息禁用 interactive 更新、interactive stub refetch 仍有保留价值。
- `.md` 作为 media attachment 的基础识别在 0.18 已有覆盖，但本地 `SendResult(success=False)` 失败日志仍有价值。
- subagent 发现 fallback 判断直接读取 `metadata["thread_id"]`，与 `reply_in_thread=False` 的实际发送语义不一致。已新增提交 `0877b444 fix(feishu): respect thread opt-out in reply fallback`，统一按“thread metadata 是否实际启用”判断，并补回归测试。

### Hindsight rolling session summary

结论：保留。

- 0.18 上游没有等价的 rolling session summary。上游有 `recall/reflect`、`recall_types=observation`、`update_mode=append`，但没有“本地滚动会话摘要 + 用摘要增强 recall query”。
- 本地实现默认关闭，且目前限制在 recall query 增强路径，不注入 prompt、不进入 retain context，风险相对可控。
- retain-pending 语义仍优于上游 `_last_retained_turn_count` 水位线方案：pending buffer、legacy baseline、失败回滚、single writer 串行都有测试覆盖。

后续可优化：

- `compose_summary_recall_query()` 的 budget 语义可调整为“最新 query 优先，summary 只吃剩余预算”。
- 长会话 `_session_summary_messages` 可在成功摘要后裁剪窗口，避免 hash/JSON 成本随会话无限增长。
- `SessionSummaryStore` 可改为短连接或 `check_same_thread=False` 加锁，降低跨线程调用时 SQLite 静默失效风险。
- LLM generator 配置目前仍是 reserved/fake generator，可在真实 generator 前收窄暴露面。

### Codex gpt-5.5 压缩提示降噪

结论：保留。

0.18 上游没有等价解决。该提交减少 gateway/CLI 场景中 Codex gpt-5.5 自动抬阈值提示的重复噪音，并已有针对测试。

### systemd service `--replace`

结论：不保留。

生产服务在升级前已经带 `gateway run --replace`，但上游测试明确要求 systemd supervisor 管理生命周期时不要在 unit 模板里无条件带 `--replace`。subagent 评审也指出 legacy 双 service、user/system 双 scope、`Restart=always` 场景有 flap 风险。

处理结果：

- 已回退并清理临时提交，当前本地历史不再包含 service 模板 `--replace` 改动。
- `hermes gateway restart` 已刷新 user service definition；当前 unit 为 `gateway run`，不带 `--replace`。

## 验证结果

最终 targeted 测试：

```text
scripts/run_tests.sh \
  tests/gateway/test_feishu.py \
  tests/gateway/test_feishu_markdown_table.py \
  tests/gateway/test_send_image_file.py \
  tests/gateway/test_stream_media_delivery_failure.py \
  tests/plugins/memory/test_hindsight_provider.py \
  tests/plugins/memory/test_hindsight_session_summary.py \
  tests/plugins/memory/test_hindsight_session_summary_assembly.py \
  tests/plugins/memory/test_hindsight_session_summary_generator.py \
  tests/agent/test_codex_gpt55_autoraise_notice.py \
  tests/hermes_cli/test_gateway_service.py -q
```

结果：

```text
606 tests passed, 0 failed
```

其他检查：

- `git diff --check` 通过。
- `.venv` 已按 lockfile 同步：`UV_PROJECT_ENVIRONMENT=.venv uv sync --extra all --extra dev --locked`。
- 生产 gateway 已重启并运行：`hermes-gateway.service` active，主进程 PID `71512`。
- 当前 unit `ExecStart`：`/home/liao/.hermes/hermes-agent/.venv/bin/python -m hermes_cli.main gateway run`。

## 当前状态

- 本地 `main` 干净，保留 22 个 `v2026.7.1` 之上的本地提交（含本报告）。
- 备份分支：`backup/pre-0.18-rebase-20260702-082305`。
- 尚未推送到 GitHub fork；`de1tydev/main` 仍是 rebase 前远端状态。
