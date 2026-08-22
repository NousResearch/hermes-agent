# Protocol

### 1. Discover architecture

Identify Hermes homes, versions, profiles, gateways, adapters, schedulers, repositories, memory providers, credential stores, and declared operating expectations.

### 2. Check installation and update health

Verify executables, runtime paths, package metadata, repository state, version consistency, and partial-update indicators.

### 3. Check gateway and delivery health

Verify actual processes, adapters, credential posture, logs, conflicts, and recent delivery evidence.

### 4. Check cron and automation

Inspect enabled jobs, schedules, last status, delivery errors, stale next-run state, model overrides, and missing skills.

### 5. Check profiles and access

Review role files, configuration, skill loading, memory posture, token ownership, and least privilege.

### 6. Check skills and repositories

Run lightweight bundle integrity, duplicate-name, dirty-tree, divergence, CI, and readiness checks.

### 7. Check cost and persistence

Surface runaway usage, unavailable state, stale durable records, and backup or rollback gaps.

### 8. Issue verdict

Set the overall status to the worst confirmed Health Matrix status. Any confirmed `RED` subsystem forces overall `RED`. Confirmed gateway or message-delivery failure is `RED`, even when the process itself is running. Use `YELLOW` only when no subsystem is confirmed `RED` and required delivery and integrity remain available.

Prioritize delivery and integrity failures over cleanup findings, record evidence conflicts, and recommend the smallest focused follow-up skill.

## Evidence discipline

Record the source, timestamp when relevant, scope, access limitations, and any contradiction for each load-bearing finding.
