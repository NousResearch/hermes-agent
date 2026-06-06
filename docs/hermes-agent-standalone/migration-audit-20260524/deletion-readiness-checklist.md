# Deletion Readiness Checklist

Do not delete Hermes Nous or Hermes Lab until every checkbox is complete and the user gives explicit final approval.

## AI Execution Status

| Area | Status | Evidence |
|---|---:|---|
| Manifest coverage | 100% | `migration-status.json` records 27,176 / 27,176 |
| Security triage | 100% | `security-triage-report.md` triages 100 / 100 |
| Knowledge import | 100% | 449 / 449 knowledge candidates imported |
| Runtime disposition | 100% | `runtime-port-plan.md` dispositions 513 / 513 |
| Synerry workflows | 100% | 6 playbooks + 5 runtime skills created |
| Verification | 100% | `95 passed`; 0 symlinks; 0 prompt-visible secret matches |
| Destructive deletion | pending human review | not performed by design |

## Legacy Folders

- [ ] Confirm Hermes Nous source path: `/Users/rattanasak/Documents/Viber Project/Tech Tools/_archived-hermes-20260524/HermesNous.retired-20260524`
- [ ] Confirm Hermes Lab source path: `/Users/rattanasak/Documents/Viber Project/Tech Tools/_archived-hermes-20260524/Hermes Labs.retired-20260524`
- [ ] Confirm no additional "Hermes Nous", "Hermes Lab", or old Hermes folders are outside the audited paths.
- [ ] Confirm `migration-manifest.json` exists and records 27,176 files.
- [ ] Confirm manifest SHA-256 hashes were generated after the final legacy freeze.

## Security Gate

- [ ] Review all 100 `secret-review` manifest records.
- [ ] Separate actual secrets from dependency-name false positives.
- [ ] Securely preserve any required `.env`, key, token, or credential files outside prompt-visible docs.
- [ ] Rotate or revoke any exposed/obsolete credentials if needed.
- [ ] Confirm no secret content was copied into Hermes Agent docs, Obsidian notes, skills, reports, or prompts.
- [ ] Confirm employee/owner/finance/client materials have private-scope handling.

## Knowledge Gate

- [ ] Review all 449 `knowledge-candidate` records.
- [ ] Import approved HermesNous knowledge with source lineage.
- [ ] Import approved HermesNous lessons.
- [ ] Import approved HermesNous patterns.
- [ ] Import approved HermesNous playbooks.
- [ ] Import approved Hermes Lab team facts/rules/incidents.
- [ ] Reject or archive stale knowledge explicitly.
- [ ] Populate `~/ObsidianVault/HermesAgent/knowledge/domain` with reviewed domain notes or document why each domain remains empty.
- [ ] Create or update Obsidian indexes so imported notes are discoverable from `MOC.md`.

## Synerry Business Gate

- [ ] Migrate owner/company context from MD Assist and HermesNous into private Synerry knowledge.
- [ ] Migrate Pitching War Room workflow.
- [ ] Migrate TOR Analyzer workflow.
- [ ] Migrate Proposal Builder workflow.
- [ ] Migrate Case Study Writer workflow.
- [ ] Migrate Market Research Pack workflow.
- [ ] Migrate Finance/Margin Check workflow.
- [ ] Verify AI no longer asks repeated company/owner questions already answered in source documents.

## Runtime Gate

- [ ] Review all 513 `runtime-candidate` records.
- [ ] Decide which legacy skills still matter.
- [ ] Port selected skills into Hermes Agent skill format.
- [ ] Port selected scripts/tools with tests.
- [ ] Confirm no legacy runtime service is still required.
- [ ] Confirm Hermes Agent profiles know how to route work across architect, orchestrator, knowledge, security, devex, qa, wow, and sunset roles.

## Verification Gate

- [ ] Run `scripts/run_tests.sh tests/knowledge_center/ -q` and record output.
- [ ] Run any added migration/skill tests.
- [ ] Start Hermes Agent dashboard if required and verify `/chat`.
- [ ] Run Synerry company-context scenario.
- [ ] Run TOR analysis scenario.
- [ ] Run Pitch War Room scenario.
- [ ] Run Market Research for pitch support scenario.
- [ ] Run old incident/lesson lookup scenario.
- [ ] Run project context lookup for HermesNous and Hermes Lab project cards.

## Archive Gate

- [ ] Create final compressed backup of Hermes Nous.
- [ ] Create final compressed backup of Hermes Lab.
- [ ] Store backup location in a non-prompt-visible operations note.
- [ ] Confirm backup can be listed/read.
- [ ] Confirm disk cleanup plan does not remove backups.

## Final Human Approval

- [ ] User reviews migration audit pack.
- [ ] User reviews migrated knowledge sample.
- [ ] User reviews security summary.
- [ ] User reviews acceptance scenario results.
- [ ] User explicitly says old Hermes Nous and Hermes Lab folders may be deleted.

## Stop Conditions

Stop deletion immediately if any of these are true:

- Any source path is uncertain.
- Any manifest scan error remains unexplained.
- Any secret-review item is unresolved.
- Any Synerry business workflow still depends on old folders.
- Any acceptance scenario fails.
- User has not given explicit deletion approval.
