# Protocol

### 1. Resolve installation

Identify the Hermes executable, repository, virtual environment, editable-install location, active profile, and reported version.

### 2. Inspect process state

List Hermes Desktop, gateway, serve, worker, WebUI, and sidecar processes that may hold the installation open.

### 3. Inspect Git topology

Record HEAD, branch, tracking branch, remotes, ahead and behind counts, shallow state, dirty state, and recent reflog.

### 4. Compare update paths

Determine which remotes and references the check path and apply path use. Explain contradictions with evidence.

### 5. Inspect installation integrity

Check launcher presence, package metadata, partial backups, native-extension locks, and stale bytecode indicators.

### 6. Classify root cause

Choose the smallest diagnosis supported by the evidence.

### 7. Propose recovery

Provide one approval-gated recovery sequence with rollback and verification commands.

## Evidence discipline

Record the source, timestamp when relevant, scope, access limitations, and any contradiction for each load-bearing finding.
