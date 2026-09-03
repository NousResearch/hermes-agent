# Demo log (2026-08-13)

Acting wallet printed by `taskmarket address` (public):

```text
0xEb76f3442EAF192EBdDbc1F7EeBD59CdE5e00D35
```

Commands run (read-only except where noted):

```bash
taskmarket address
taskmarket stats
taskmarket task list --status open --limit 6
taskmarket task get 0xe9fb8fe4e6f83b54d4850efd1c5b6aef6d1bbd7f9f91921d4329f404e64c5682
```

Observed:

- CLI JSON envelope `{ ok: true, data: ... }`
- Open bounties with `pendingActions` including worker `submit` (`requiresPayment: false`)
- Requester `accept` remains paid (0.001 USDC) and is not invoked by this skill without explicit approval

Worker submit demo (artifact delivery, unpaid):

```bash
taskmarket task submit 0xe9fb8fe4e6f83b54d4850efd1c5b6aef6d1bbd7f9f91921d4329f404e64c5682 \
  --file AUDIT-REPORT.md --role final \
  --file FINDINGS.json --role final \
  --file METHOD.md --role attachment
```

Result:

```json
{"ok":true,"data":{"submissionId":"b871bea6-93e4-45b9-bbf5-4587e0dd7a44"}}
```

The skill never calls `task accept`. Submissions stay in review until the human requester authorizes payout.
