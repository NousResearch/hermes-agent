# Spec 087 B1b — First-Fire Checklist

Cron: `0 18 * * 5` (Friday 18:00 ET)
Expected first fire: 2026-05-08T18:00:00-04:00
Alert deadline: 2026-05-09T09:00:00-04:00

---

## Pass criteria (all must be true)

- [ ] `output/hedge_report/hedge_report_2026-05-08.json` exists
- [ ] `output/hedge_report/BIOSHORT_VERDICT.json` as_of_date == "2026-05-08"
- [ ] BIOSHORT_VERDICT.json recommendation line is non-empty
- [ ] `logs/biotech_hedge_report.log` updated after 18:00 ET on 2026-05-08
- [ ] No MASSIVE_API_KEY warnings in biotech_hedge_report.log
- [ ] No Exception/Traceback in biotech_hedge_report.log for this run
- [ ] bioshort_watch LLM cron remains commented out (no active line)

## Fail criteria (any one → FAIL)

- [ ] hedge_report_2026-05-08.json missing by 2026-05-09 09:00 ET
- [ ] BIOSHORT_VERDICT as_of_date != "2026-05-08"
- [ ] Exception or Traceback in log for this run date

## Special case — WSL2 sleep

If laptop was asleep at 18:00 ET on Friday:
- Status: MISSED_DUE_TO_WSL_SLEEP
- Action: check if @reboot catchup cron fired after next boot
- Check: `logs/biotech_hedge_report.log` for a run after reboot
- Do NOT run a manual extra producer run without operator approval

## On PASS — next actions

1. Run build tool to flip first-fire status: `python3 tools/build_hermes_knowledge_layer.py`
2. Update HELD_ITEMS_SEED: spec_087_b1b → status CLOSED
3. Unblock spec_087_b2: remove "B1b first-fire validation must pass" blocker
4. Commit: "ops: Spec 087 B1b first-fire PASS — advance to B2"
5. Do NOT reactivate bioshort_watch LLM — that requires a separate spec

## On FAIL/MISSED — next actions

1. Run build tool: status will show FAIL_ARTIFACT_MISSING_PAST_DEADLINE or MISSED_DUE_TO_WSL_SLEEP
2. Surface to operator. Do NOT advance to B2.
3. Determine root cause:
   - Check cron fired: `grep 'biotech_hedge_report' logs/biotech_hedge_report.log | tail -20`
   - Check portfolio CSV: `ls data/snapshots/2026-05-08/portfolio_positions.csv`
   - Check MASSIVE API key: `grep -i 'massive\|api_key\|error' logs/biotech_hedge_report.log | tail -20`
4. File NEEDS_OPERATOR_DECISION if cause unclear

## Related artifacts

- Cron install: artifacts/audit/spec_087_b1b_env_readiness_2026_05_07.md
- Phase A governance: artifacts/audit/spec_087_phase_a_bioshort_hedge_governance_decision_2026_05_06.md
- First-fire ledger: artifacts/ops/first_fire_ledger/latest.json
- Held spec ledger: artifacts/ops/held_spec_ledger/latest.json
