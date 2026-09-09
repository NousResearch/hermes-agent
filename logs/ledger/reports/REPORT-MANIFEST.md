# North-Forge Report Manifest

The index the completeness check uses. **Every `RUN-YYYY-MM-DD-NNN` id that
appears anywhere in this ledger must have exactly one row below.** A row either
names a session report that exists in the handoff out-dir (`D:\logs\` — the
`<OutDir>` `scripts/collect-logs.{ps1,sh}` builds) **and** mentions that run id,
or declares the run `ledger-only` (it legitimately produced no report).

`scripts/lib/report_completeness.py` (invoked by both collectors) enforces this:

| Condition | Result |
| --- | --- |
| a ledger `RUN-` id with **no row** here | **FAIL** |
| a row names a report **not present** in the out-dir | **FAIL** |
| the report exists but does **not mention** its run id | **FAIL** |
| a row's `sha256` != the report on disk (report edited) | WARN — refresh the row |
| a row for a `RUN-` id no ledger entry uses | WARN — stale row |
| an unresolved `[[wiki-link]]` in the ledger | WARN — confirm it's intentional |

**Append-only in spirit.** Correct a wrong row by editing it in place (and
refreshing its `sha256`); never delete a row for a run that still exists in the
ledger. Reports dated `2026-09-07` and earlier were written before the
`Run:`/`Covers:`/`Source-Of-Truth:` header convention (`CHG-2026-09-08-008`); they
carry their run id in the body, which is what the check verifies. New reports use
`templates/SESSION-REPORT-TEMPLATE.md`.

`Source-Of-Truth` = `in-bundle` when the `D:\logs\` file *is* the authoritative
write-up; an absolute path / URL when that file is a redacted mirror of a fuller
report kept elsewhere; `ledger-only` when the run produced no report.

To refresh a `sha256` after editing a report:
`python -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" D:\logs\<file>`

---

| RUN id | Report file (in `D:\logs\`) | Covers | Source-Of-Truth | sha256 |
| --- | --- | --- | --- | --- |
| RUN-2026-09-06-001 | LEDGER_TEMPLATE_UPDATE_2026-09-06.md | ledger-schema v2 setup, CHG-2026-09-06-015..019 | in-bundle | ce407d912baa39ac5513b43aee29fb742abf57630d907c799fdb4ca732c8c236 |
| RUN-2026-09-06-002 | REBRAND_2026-09-06.md | CHG-2026-09-06-020, CHG-2026-09-06-021, CHG-2026-09-06-022, CHG-2026-09-06-023, CHG-2026-09-06-024, DECISION-2026-09-06-001 | in-bundle | 480f5d20755c4f44a4a1ee8ce93f2f0f644242ff6803bcb74d94ea94f5959ed6 |
| RUN-2026-09-06-003 | BRANDING_PASS_2026-09-06.md | CHG-2026-09-06-025, CHG-2026-09-06-026 | in-bundle | de1814090505c4923d5b6b1afc2f22cffd474f5c2acd0ade5745c452e9a16404 |
| RUN-2026-09-07-001 | CONSOLIDATED_PASS_2026-09-07.md | consolidated pass (step 4 → DECISION-2026-09-06-003 opened) | in-bundle | 9352dafea4d9ad3330b89f2ca064518d31658c7620743a154290e1674f8e12bc |
| RUN-2026-09-07-002 | IDENTITY-RUNTIME_2026-09-07.md | CHG-2026-09-07-007 | in-bundle | 732cd9a7051b83605c1809941cef26d514e4c8fad04d893b769670904776f404 |
| RUN-2026-09-07-003 | PUSH-SYNC_2026-09-07.md | CHG-2026-09-07-009, CHG-2026-09-07-010, ERR-2026-09-07-002 | in-bundle | 6661f45d60da2dff3dd8466cff46562cb0c97a92b114f9ab175c992009191731 |
| RUN-2026-09-07-004 | DRIVE-NATIVE-ONBOARDING_2026-09-07.md | CHG-2026-09-07-011, CHG-2026-09-07-012, CHG-2026-09-07-013, CHG-2026-09-07-014, DECISION-2026-09-07-001 | in-bundle | ebfc8b655591de920e61b15ad3509735541d3c310b01f0bf34c343be364a5ebe |
| RUN-2026-09-07-005 | BOOTSTRAP-F04-DATALOSS-FIX_2026-09-07.md | CHG-2026-09-07-015, ERR-2026-09-07-003 | in-bundle | b6d5ab21168697977a3f8beda98108155cc67fbd5458d34a11d48704b5ab3167 |
| RUN-2026-09-07-006 | CODEX-AUDIT-CHEAP-FIXES-AND-BACKLOG_2026-09-07.md | CHG-2026-09-07-016, CHG-2026-09-07-017, CHG-2026-09-07-018, ERR-2026-09-07-004, ERR-2026-09-07-006, ERR-2026-09-07-007, DECISION-2026-09-07-002 | in-bundle | c88c57275831a6513fe23d3719bb68893f5d9b09b41d3ac7cb58d194add1b902 |
| RUN-2026-09-07-007 | CONTENT-SCAN-F03-WHITESPACE-BYPASS_2026-09-07.md | CHG-2026-09-07-019, ERR-2026-09-07-005 | in-bundle | de7820866fb2ae0ec4736a924ad15c06728e9748abbc51f3dac6ae4a8ad00c94 |
| RUN-2026-09-07-008 | LAUNCHER-READINESS-PROBE_2026-09-07.md | CHG-2026-09-07-020, ERR-2026-09-07-006 | in-bundle | e14f46c2b242f33e5cf124d1033c1f7c2b7283af8462bf2570d484f5f1d7a8f3 |
| RUN-2026-09-07-009 | SKIN-REBUILD-V2_2026-09-07.md | CHG-2026-09-07-021 | in-bundle | 1af8afb2fd7b36ea88ed0614c604b5e19034cb98bf43a21a268e5772a48ae75d |
| RUN-2026-09-07-010 | TIER-PIN-MECHANISM_2026-09-07.md | CHG-2026-09-07-022, DECISION-2026-09-07-003 | in-bundle | 99ca88ffe8484aa657884f518b71555e9be5ee986df6f86219e063e0209c0e5e |
| RUN-2026-09-07-011 | LEDGER-RATIFICATION_2026-09-07.md | CHG-2026-09-07-023, DECISION-2026-09-06-002, DECISION-2026-09-06-003 | in-bundle (backfilled RUN-2026-09-08-004) | c90917d00d9a26565f58a532ef6ee35b19b9e5df431e48b407af930c636cb211 |
| RUN-2026-09-08-001 | NF-v0.7.0-INSTALLER-GUI-SKIN-DOCS_2026-09-08.md | CHG-2026-09-08-001, CHG-2026-09-08-002, CHG-2026-09-08-003, CHG-2026-09-08-004 | in-bundle (backfilled RUN-2026-09-08-004) | 053e582d956e0100962aa8d7c637bc67e72edfa17bd3a4ffb212c93bc91a3a61 |
| RUN-2026-09-08-002 | PREVIEW-PAGE-REAL-ART_2026-09-08.md | CHG-2026-09-08-005 | in-bundle (backfilled RUN-2026-09-08-004) | a9a23734aa9639f47a52b8502b85279b83fbf169317f2033e296432e82622974 |
| RUN-2026-09-08-003 | RECONCILE-PUSH-EDITIONS_2026-09-08.md | CHG-2026-09-08-006, CHG-2026-09-08-007 | in-bundle (backfilled RUN-2026-09-08-004) | 2d366d9aa977eed3679bdd64671d64f4d79e2f7d0aa43f0116d035adc10e7697 |
| RUN-2026-09-08-004 | HYGIENE-CLEANUP_2026-09-08.md | CHG-2026-09-08-008, CHG-2026-09-08-009, CHG-2026-09-08-010, CHG-2026-09-08-011, CHG-2026-09-08-012, CHG-2026-09-08-013, CHG-2026-09-08-014, CHG-2026-09-08-015, CHG-2026-09-08-016, ERR-2026-09-07-004, ERR-2026-09-07-007, ERR-2026-09-08-001, ERR-2026-09-08-002, ERR-2026-09-08-003, ERR-2026-09-08-004, DECISION-2026-09-07-002 | in-bundle | b41cbf6e3b81f623823f69b51b129f740220937e661e96d1124381a54e3e9064 |
