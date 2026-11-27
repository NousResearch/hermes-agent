# Post-SEAL Verification Checklist

After SEAL, Neon Genie does **not** execute. The operator (or Wayfinder) verifies outcomes. Use this checklist so packets remain outcome-dense.

## Immediate (before handoff)

1. **Completion proof defined** — packet has `completion_proof` (externally checkable).
2. **Proof path present** — ordered steps to obtain proof (or honest `NOT_COMPUTABLE`).
3. **Open blocking DataRequests** — listed; promotion capped until satisfied/waived.
4. **Authority** — `advisory_only`; `grants_execution: false`.
5. **Wayfinder** — if handoff emitted, `product_intent_changes_require_neon_genie_review: true`.
6. **Claim labels** — no uncited `OBSERVED`; offline runs do not invent market facts.

## After real-world contact (operator)

7. Record whether proof was obtained or failed.
8. Append a **learning ledger** entry (`PROPOSED` / `OBSERVATION` only — never auto-canon):

```bash
python scripts/neon_genie.py do learn \
  --class proof_obtained \
  --summary "First paid diagnostic booked" \
  --source-run out/neon-genie/product-audit/run-receipt.json \
  --ledger out/neon-genie/learning-ledger.jsonl
```

9. Failed opportunities, brittle integrations, buyer/distribution/anti-capture failures become ledger observations so Neon becomes harder to impress over time.

## Forbidden

- Auto-applying ledger entries to skill corpus or promotion canon
- Treating unverified proof as `OBSERVED` success
- SEAL that grants spend/publish/execute
