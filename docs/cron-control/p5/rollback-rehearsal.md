# P5-T07 Rollback Rehearsal

目標：

- 驗證 model switch 之後可回復原始 model
- 只在 temp store 跑，不碰真實 runtime

驗證重點：

- `execute_rollback(job_id)` 有回傳 record
- `primary_provider == openai-codex`
- `primary_model == gpt-5.4`
- rollback 後 job 回到原始狀態

失敗條件：

- rollback 失敗
- rollback 後 model 不是原始值

回退：

- 砍掉 temp home 即可
