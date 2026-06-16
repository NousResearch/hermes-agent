# PRD-6 LCM Aegis-to-Apollo Cutover Runbook

This runbook covers the Phase-5 Aegis stress/QA gate and Phase-6 Apollo hash/config preflight. It is deliberately split between read-only worker tooling and the privileged operational gate: Daedalus/Aegis tooling produces reports only; Apollo owns the privileged live cutover and any gateway lifecycle action.

## Aegis activation

1. Start from an Aegis profile that is already isolated from Apollo traffic.
2. Enable the LCM context engine only in the Aegis profile config:

```yaml
context:
  engine: lcm
lcm:
  context_threshold: 0.35
  sensitive_patterns_enabled: true
  sensitive_patterns:
    - api_key
    - bearer_token
    - password_assignment
    - private_key
  encryption_enabled: true
```

3. Preserve the Aegis LCM store before testing unless Apollo explicitly chooses a purge for a clean-room rehearsal. The default decision is preserve.
4. Run the Aegis stress gate in dry-run or live-harness mode and archive the report. Dry-run command:

```bash
python scripts/lcm_aegis_stress.py --dry-run --out /tmp/lcm-aegis-stress.md
```

5. A FAIL-LOUD stress status routes to #alerts and blocks cutover. The loud triggers are degraded compactions >=3 consecutive or degraded rate >5%; any fail-closed recall also blocks cutover.

## Config/hash preflight

Aegis QA must emit a manifest with these fields:

```json
{
  "plugin_artifact_sha256": "<sha256 or omit when plugin_artifact_path is present>",
  "plugin_artifact_path": "<path used to compute sha256 when hash is omitted>",
  "threshold_config": {"context_threshold": 0.35, "leaf_chunk_tokens": 20000},
  "redaction_ruleset": ["api_key", "bearer_token", "password_assignment", "private_key"],
  "schema_version": 4,
  "encryption_mode": "aead-v1"
}
```

Apollo target emits the same manifest for the target profile/artifact. Run:

```bash
python scripts/lcm_config_preflight.py --aegis-report /path/to/aegis-report.md --apollo-target /path/to/apollo-target.json --out /tmp/lcm-config-preflight.md
```

Preflight PASS requires byte-identical plugin artifact hash plus identical threshold config, redaction ruleset, schema version, and encryption mode. Any drift is FAIL-LOUD and blocks cutover.

## Apollo cutover

Apollo owns the privileged cutover gate. No worker script in this phase restarts or controls a live gateway. Apollo must review the stress report, config/hash preflight report, and store preserve/purge decision before applying a config diff.

Cutover diff:

```diff
 context:
-  engine: compressor
+  engine: lcm
+lcm:
+  context_threshold: 0.35
+  sensitive_patterns_enabled: true
+  sensitive_patterns:
+    - api_key
+    - bearer_token
+    - password_assignment
+    - private_key
+  encryption_enabled: true
```

After the diff is applied, Apollo performs the privileged process lifecycle step from the operator console only after printing this runbook section and the rollback diff below.

## Rollback diff

Exact rollback diff:

```diff
 context:
-  engine: lcm
+  engine: compressor
-lcm:
-  context_threshold: 0.35
-  sensitive_patterns_enabled: true
-  sensitive_patterns:
-    - api_key
-    - bearer_token
-    - password_assignment
-    - private_key
-  encryption_enabled: true
```

Rollback keeps the LCM store on disk by default for forensic inspection and possible re-cutover. Purge only when Apollo records an explicit store purge decision and confirms there is no open incident or recovery need.

## Store preserve/purge decision points

- Preserve before Aegis activation when testing against existing profile history.
- Purge before Aegis activation only for a clean-room rehearsal.
- Preserve during Apollo cutover so rollback can inspect what LCM ingested.
- Purge after rollback only with Apollo approval and only after exporting any incident evidence.
- Preserve after a FAIL-LOUD report until Apollo triages degraded/fail-closed evidence.

## Safety contract

- Aegis stress and config preflight tooling is read-only with respect to live Hermes profiles.
- Apollo owns the privileged process lifecycle and cutover gate.
- Any operational wrapper that is later allowed to perform a live lifecycle action must require an explicit dangerous flag and must print both the intended diff and the rollback diff before taking action.
