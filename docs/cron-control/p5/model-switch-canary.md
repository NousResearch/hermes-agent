# P5-T06 OpenAI Codex Model-Switch Canary

目標：

- 在 temp home 中驗證 `openai-codex` 同 provider 的模型切換
- 先從 `gpt-5.4` 切到 `gpt-5.6-terra`
- 再做 rollback rehearsal 回到 `gpt-5.4`

執行方式：

- canary 使用 temp home + allowlist patch
- fallback 被固定成 `openai-codex/gpt-5.6-terra`
- action 層仍走現有 `switch_provider` 入口，但結果必須是同 provider 的 model CAS

驗證重點：

- `job_after_switch.provider == "openai-codex"`
- `job_after_switch.model == "gpt-5.6-terra"`
- rollback 後回到 `gpt-5.4`

回退：

- 刪除 temp home 即可

參考入口：

```bash
python /Users/ryanchao/.hermes/worktrees/cron-control-p0/scripts/cron-control-p5-canary.py
```
