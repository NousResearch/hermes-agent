# Cron Profile Assignment Runbook

This runbook records the intended ownership model and the safe operating path for moving default-profile cron jobs into domain profiles.

## Ownership Summary

| Profile | Active jobs | Ownership |
|---|---:|---|
| `default` | 29 | Core Hermes runtime, backups, GBrain bridge, global diagnostics, default-owned `hermes-loop:*` checks. |
| `rva-leads` | 13 | Lead intake, CRM lifecycle, consultation follow-up, quote suggestions, SMS opt-out handling. |
| `rva-profit-pulse` | 10 | Content CMS, Square sales, topic discovery, weekly digest, profit/content loop checks. |
| `rva-firm-ops` | 5 | Client email capture, TaxDome alerting, calendar nudges, Gmail watch refresh, meeting-note processing. |
| `cpa-tax-researcher` | 2 | Tax research archive lint and tax research integrity. |
| `personal` | 0 | No active cron jobs currently match this profile. |
| `rva-dev` | 0 | No active cron jobs currently match this profile. |

The paused `stale-draft-alert` job is not part of the active migration. Its candidate owner is `rva-firm-ops` after the existing shadow-readiness blocker is resolved.

## Tools

Read-only audit:

```bash
python3 scripts/cron_profile_assignment_audit.py --format markdown
python3 scripts/cron_profile_assignment_audit.py --format json
```

Dry-run migration manifest:

```bash
python3 scripts/cron_profile_rebalance.py --format markdown
python3 scripts/cron_profile_rebalance.py --format json
```

Use `--target-profile <profile>` to limit a dry-run or stage operation to one receiving profile. Repeat the flag to include multiple receiving profiles.

Stage scripts and disabled target job copies:

```bash
python3 scripts/cron_profile_rebalance.py --apply-stage --format json
```

Do not combine `--apply-stage` with `--cutover-verified`; the CLI rejects that combination so profile-local verification cannot be skipped accidentally.

Cut over verified script-only jobs:

```bash
python3 scripts/cron_profile_rebalance.py --cutover-verified <job-id> [<job-id> ...] --format json
```

Cut over accepted agent-driven shadow jobs:

```bash
python3 scripts/cron_profile_rebalance.py \
  --cutover-verified <job-id> \
  --accepted-agent <job-id> \
  --format json
```

Use `--remove-source` only after you intentionally want the default-profile source records deleted instead of paused.

## Required Profile Gateways

Only profiles that receive active jobs need profile-scoped gateways for this migration:

```bash
hermes -p cpa-tax-researcher gateway start
hermes -p rva-firm-ops gateway start
hermes -p rva-leads gateway start
hermes -p rva-profit-pulse gateway start
```

Use foreground debugging only when needed:

```bash
hermes -p <profile> gateway run
```

Before starting additional gateways, check multi-gateway dispatcher settings so only the intended profile owns singleton dispatcher behavior.

## Migration Checklist

1. Run the read-only audit and confirm the active counts match the Ownership Summary.
2. Run the dry-run rebalance manifest and confirm there are no blockers.
3. Review any `missing_target_skills`, `id_collision`, `output_history_collision`, `missing_source_script`, or `source_script_outside_profile_scripts` blockers before applying.
4. Run `--apply-stage`. This copies required scripts into target profile `scripts/` directories and writes disabled target job copies.
5. Start the required profile-scoped gateways.
6. For each target profile, verify the target cron store and gateway status:

   ```bash
   hermes -p <profile> cron list
   hermes -p <profile> cron status
   hermes -p <profile> gateway status
   ```

7. Trigger or dry-run one migrated job per target profile and confirm output lands under that profile's cron output directory.
8. For script-only jobs, cut over only after target execution is verified.
9. For agent-driven jobs, compare the target-profile shadow output with the default-profile behavior and pass `--accepted-agent` only after the output quality is accepted.
10. Re-run the audit and confirm default keeps only default-owned jobs plus any intentionally shadowing jobs.

## Rollback

Every apply-stage and cutover operation backs up existing target cron stores under each touched profile's cron backup directory.

To roll back a failed profile batch:

1. Stop or pause the target profile gateway if duplicate firing is possible.
2. Restore that profile's latest pre-change `jobs.json` backup.
3. Restore the default profile `jobs.json` backup if source jobs were paused or removed.
4. Remove copied target scripts only if they were introduced by the failed batch and no other job references them.
5. Run:

   ```bash
   hermes cron list
   hermes -p <target-profile> cron list
   ```

6. Confirm the default copy is active before re-attempting migration.

## Verification Evidence Format

For each profile batch, record:

- Target profile name.
- Jobs staged.
- Scripts copied.
- Gateway command used.
- `cron list` result.
- `cron status` result.
- Job IDs verified.
- Agent-driven jobs accepted or left shadowing.
- Cutover command run.
- Rollback backup path if created.

## Follow-up Items

- Decide whether every profile should eventually get its own cloned `hermes-loop:*` diagnostic suite. This is a separate design change from moving domain-owned active jobs.
- Resolve the `stale-draft-alert` paused blocker before migrating it to `rva-firm-ops`.
- Add new assignments for `personal` and `rva-dev` only after cron jobs exist that match those domains.
