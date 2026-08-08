# P5-T05 Auto-reset Canary

目標：

- 在 temp home / temp store 中執行 `reset_job`
- 驗證 stale-running 的 job 會被重置為 `scheduled`
- 不碰真實 jobs.json

執行方式：

- canary module 會先建立 temp Hermes home
- 寫入 `openai-codex` only fallback chain
- 以 `execute_verdict_action(..., approved=True)` 跑 `reset_job`

驗證重點：

- action outcome = `verified`
- job `state == scheduled`
- `run_claim` 與 `fire_claim` 清空

回退：

- 刪除 temp home 即可

參考入口：

```bash
python /Users/ryanchao/.hermes/worktrees/cron-control-p0/scripts/cron-control-p5-canary.py
```
