# Protocol

### 1. Resolve gateway scope

Identify profile, Hermes home, configured platforms, delivery targets, service manager, and expected architecture.

### 2. Inspect state surfaces

Read state and PID files, timestamps, adapter inventory, and recent gateway status output.

### 3. Verify processes

Confirm the reported process exists and matches the expected command line. Detect duplicate or orphaned gateway processes.

### 4. Inspect credential posture

Check required keys as set or not set and identify conflicting profile ownership without exposing values.

### 5. Inspect logs

Trace connection, authentication, network, reconnect, polling, rate-limit, shutdown, and delivery patterns.

### 6. Test safely

Use non-mutating platform diagnostics when available and avoid tests that steal polling sessions.

### 7. Correlate delivery

Compare gateway evidence with recent message or cron delivery results.

### 8. Classify and hand off

Report the smallest supported diagnosis and one approval-gated recovery path.

## Evidence discipline

Record the source, timestamp when relevant, scope, access limitations, and any contradiction for each load-bearing finding.
