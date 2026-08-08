# P5-T04 Auto-quarantine Canary

目標：

- 驗證影子判讀能穩定把 receipt conflict 類型的 evidence 判成 `quarantined`
- 不執行 runtime 寫入

固定證據：

- `docs/cron-control/p0/fixtures/receipt-conflict-429.json`

驗證重點：

- `state == quarantined`
- `recommended_action == escalate_to_human`
- `automatic_action_allowed == false`
- `blocked_by` 含 `delivery_receipt_conflict`

執行入口：

```bash
python /Users/ryanchao/.hermes/worktrees/cron-control-p0/scripts/cron-control-p5-canary.py
```

失敗條件：

- 任何自動修復被標成 true
- 判讀落到 `healthy` 或 `recoverable`

回退：

- 無 runtime 寫入；失敗只需修正判讀規則或 fixture 對照。
