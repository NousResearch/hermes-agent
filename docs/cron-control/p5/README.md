# P5 Rollout Contract Pack

這個資料包對應 `cron-failover-work-tasks-model-routing.md` 的 Phase 5。

涵蓋內容：

- `labeled-dataset.jsonl`：30 筆標記資料
- `canary-allowlist.json`：只允許 `openai-codex/*` 路由
- `auto-quarantine-canary.md`
- `auto-reset-canary.md`
- `model-switch-canary.md`
- `rollback-rehearsal.md`
- `production-readiness-review.md`
- `production-readiness-signoff.md`
- `validate_phase5.py`

驗證方式：

```bash
python /Users/ryanchao/.hermes/worktrees/cron-control-p0/docs/cron-control/p5/validate_phase5.py
python /Users/ryanchao/.hermes/worktrees/cron-control-p0/scripts/cron-control-p5-canary.py
```

判定原則：

- 路由只允許 `openai-codex/<model>`
- canary 一律在 temp home / temp store 跑，不碰真實 runtime
- `production-readiness-review.md` 只能到 `ready_for_signoff`
