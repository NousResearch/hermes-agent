# HAD-1158 Docker named-volume backup and restore plan

Issue: HAD-1158, Docker named-volume backup and restore plan for ontology and
Hadto services.

This runbook covers service data stored in Docker named volumes that are outside
the `/data/hermes` home-path migration boundary. It is a planning and acceptance
artifact for the uncovered storage boundary identified by HAD-1155.

## Scope and safety invariants

This document does not execute backup or restore. It records the required
inventory, discovery evidence, backup plan, restore plan, and acceptance gates
for later operator work.

Safety invariants:

- Discovery commands are read-only with respect to Docker volumes and services;
  optional file redirects write evidence outside the source volumes.
- No service stop, start, restart, or container replacement is part of this artifact.
- No Docker backup, restore, volume create, volume remove, or volume overwrite command was run to produce this artifact.
- Backup execution requires an approved maintenance window and service-specific
  writer quiescence before archive capture.
- A read-only Docker mount of a named volume is not a consistency guarantee if a
  writer is still active. Quiescence is mandatory.
- Production restore must not overwrite a non-empty live volume. Rehearse on a
  scratch host or with rehearsal volume names first, then restore to an empty
  production target during an approved recovery window.
- Evidence must not print secrets, MinIO credentials, cookies, API tokens, or
  credential-bearing environment variables.

## Named-volume inventory

Current read-only Docker inspection on 2026-05-31 found these required volumes
and labels:

| Volume | Compose project | Compose volume | Service ownership | Current container | Mount target | Backup class | Notes |
|---|---|---|---|---|---|---|---|
| `hadto-pipeline_pipeline-data` | `hadto-pipeline` | `pipeline-data` | Hadto pipeline service | `hadto-pipeline` | `/app/data` | `critical-durable` | Pipeline runtime state, generated data, and any non-reproducible pipeline artifacts must be captured before migration closeout. |
| `ontology-platform_archivebox-data` | `ontology-platform` | `archivebox-data` | Ontology ArchiveBox service | `ontology-archivebox` | `/data` | `important-durable` | Archive index, fetched material, metadata, and crawler history may be source evidence for ontology work. |
| `ontology-platform_oxigraph-data` | `ontology-platform` | `oxigraph-data` | Ontology Oxigraph triplestore | `ontology-triplestore` | `/data` | `critical-durable` | RDF graph store state is required to restore ontology API/query behavior. |
| `ontology-platform_source-materials-minio-data` | `ontology-platform` | `source-materials-minio-data` | Ontology source-materials MinIO service | `ontology-source-materials-blob-store` | `/data` | `critical-durable` | Object-store buckets may contain the only durable copy of source materials and import payloads. |

These volumes live under Docker's data root, currently surfaced by Docker as
`/var/lib/docker/volumes/<volume>/_data`. They are not covered by
`/home/david/.hermes`, `/home/david/stacks`, `/home/david/.ctx-data`, or
`/home/david/.codex` bind-mount coverage.

## Discovery commands

Use these commands during the later backup issue to refresh inventory and store
evidence. They are inspection commands only.

```bash
MIGRATION_ID="$(date -u +%Y%m%dT%H%M%SZ)"
EVIDENCE_DIR="/data/hermes/.migration-evidence/docker-volumes/${MIGRATION_ID}"
mkdir -p "${EVIDENCE_DIR}"
```

```bash
docker volume inspect \
  hadto-pipeline_pipeline-data \
  ontology-platform_archivebox-data \
  ontology-platform_oxigraph-data \
  ontology-platform_source-materials-minio-data \
  > "${EVIDENCE_DIR}/volume-inspect.json"
```

```bash
docker inspect \
  hadto-pipeline \
  ontology-archivebox \
  ontology-triplestore \
  ontology-source-materials-blob-store \
  > "${EVIDENCE_DIR}/container-inspect.json"
```

```bash
docker ps \
  --filter "volume=hadto-pipeline_pipeline-data" \
  --filter "volume=ontology-platform_archivebox-data" \
  --filter "volume=ontology-platform_oxigraph-data" \
  --filter "volume=ontology-platform_source-materials-minio-data" \
  --format '{{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Mounts}}' \
  > "${EVIDENCE_DIR}/volume-users.tsv"
```

```bash
docker volume ls \
  --filter "label=com.docker.compose.project=hadto-pipeline" \
  --format '{{.Name}}' \
  > "${EVIDENCE_DIR}/hadto-pipeline-volumes.txt"
```

```bash
docker volume ls \
  --filter "label=com.docker.compose.project=ontology-platform" \
  --format '{{.Name}}' \
  > "${EVIDENCE_DIR}/ontology-platform-volumes.txt"
```

```bash
docker compose ls --format json > "${EVIDENCE_DIR}/compose-projects.json"
```

If the Compose working directory must be recovered from Docker labels, inspect
the project labels on a container without printing environment variables:

```bash
docker inspect \
  --format '{{ index .Config.Labels "com.docker.compose.project" }} {{ index .Config.Labels "com.docker.compose.project.working_dir" }} {{ index .Config.Labels "com.docker.compose.project.config_files" }}' \
  hadto-pipeline ontology-archivebox ontology-triplestore ontology-source-materials-blob-store \
  > "${EVIDENCE_DIR}/compose-labels.txt"
```

## Backup plan

### Universal backup procedure

For each named volume:

1. Refresh discovery evidence and confirm the volume is still owned by the
   expected Compose project and service.
2. Prove application-level writer quiescence for the owning service.
3. Record the quiescence evidence in the evidence directory without secrets.
4. Capture the archive from a read-only mount of the Docker volume into backup
   storage that is not inside the source volume.
5. Record archive filename, byte size, checksum, helper image digest, Docker
   version, and UTC timestamp in a manifest.
6. Read the archive back from backup storage and verify the checksum.
7. Keep the source volume unchanged until a scratch-host restore rehearsal
   passes and the manager accepts the evidence.

Use a pinned helper image digest selected by the operator at execution time.
Record that digest in evidence. The backup template is:

```bash
VOLUME="<docker-volume-name>"
ARCHIVE="${VOLUME}.${MIGRATION_ID}.tar"
HELPER_IMAGE="<approved-helper-image-pinned-by-digest>"
BACKUP_DIR="<backup-target-directory>"

docker run --rm \
  --mount "type=volume,source=${VOLUME},target=/source,readonly" \
  --mount "type=bind,source=${BACKUP_DIR},target=/backup" \
  "${HELPER_IMAGE}" \
  sh -c 'cd /source && tar --numeric-owner -cpf "/backup/'"${ARCHIVE}"'" .'

sha256sum "${BACKUP_DIR}/${ARCHIVE}" \
  > "${EVIDENCE_DIR}/${ARCHIVE}.sha256"
```

The command above is a future execution template, not a command run by this
artifact.

### Per-volume backup plan

| Volume | Quiescence dependency | Backup evidence to capture | Minimum archive checks |
|---|---|---|---|
| `hadto-pipeline_pipeline-data` | Hadto pipeline must have no active ingestion, transform, export, or worker job writing `/app/data`. Strongest guarantee is maintenance-window shutdown of the pipeline writer before archive capture. | `docker ps` status, Compose project labels, pipeline idle proof or approved shutdown window, recent non-secret logs showing no active job, archive checksum. | Archive contains the expected top-level pipeline data tree, preserves ownership, and can be listed from backup storage. |
| `ontology-platform_archivebox-data` | ArchiveBox must not be crawling, importing, indexing, or writing snapshots. Scheduled or queued crawler work must be paused or proven idle. | `docker ps` status, ArchiveBox container labels, non-secret app status or log excerpt showing idle crawler state, archive checksum. | Archive includes ArchiveBox data/index content and preserves metadata needed by the service. |
| `ontology-platform_oxigraph-data` | Oxigraph must not receive writes, imports, compactions, or graph mutations. For consistency, use a maintenance window that blocks ontology API writes before archive capture. | `docker ps` health/status, triplestore container labels, write-quiescence note, optional read-only SPARQL count/query result, archive checksum. | Archive includes the Oxigraph store files and can be restored into an empty test volume that passes a triplestore smoke query. |
| `ontology-platform_source-materials-minio-data` | MinIO must not receive object uploads, deletes, lifecycle changes, or bucket policy changes. Block source-material ingestion before archive capture. | `docker ps` status, MinIO container labels, bucket/object inventory or sampled object manifest with credentials redacted, archive checksum. | Archive includes bucket/object storage layout and can restore sampled source objects with matching checksums. |

## Restore plan

### Scratch-host rehearsal

Before any production restore, perform a scratch-host rehearsal:

1. Provision a scratch host or isolated Docker data root with the same Docker
   major version when practical.
2. Copy backup archives and checksums to the scratch host.
3. Verify archive checksums before extraction.
4. Create empty rehearsal volumes, preferably with a suffix such as
   `-restore-rehearsal`, unless the scratch host is disposable and can use the
   production names.
5. Restore each archive into its empty rehearsal volume.
6. Start the matching Compose services only against the rehearsal volumes and an
   isolated network.
7. Run service smoke checks and record outputs without secrets.
8. Destroy the scratch host or clearly mark rehearsal volumes as non-production
   after evidence is accepted.

Restore template for a scratch volume:

```bash
SOURCE_ARCHIVE="<volume>.<migration-id>.tar"
RESTORE_VOLUME="<volume>-restore-rehearsal"
HELPER_IMAGE="<approved-helper-image-pinned-by-digest>"
BACKUP_DIR="<backup-target-directory>"

docker volume create "${RESTORE_VOLUME}"

docker run --rm \
  --mount "type=volume,source=${RESTORE_VOLUME},target=/restore" \
  --mount "type=bind,source=${BACKUP_DIR},target=/backup,readonly" \
  "${HELPER_IMAGE}" \
  sh -c 'cd /restore && tar --numeric-owner -xpf "/backup/'"${SOURCE_ARCHIVE}"'"'
```

The restore template mutates the named restore volume and belongs only in the
later approved restore or rehearsal task.

### Per-volume restore plan

| Volume | Production restore target | Rehearsal checks | Production acceptance |
|---|---|---|---|
| `hadto-pipeline_pipeline-data` | Restore into an empty `hadto-pipeline_pipeline-data` volume before the Hadto pipeline container starts. | Container can mount `/app/data`; pipeline health check passes; expected run metadata or generated data is visible; no jobs auto-start without operator approval. | Pipeline starts healthy, can read prior state, and manager accepts sampled restored data. |
| `ontology-platform_archivebox-data` | Restore into an empty `ontology-platform_archivebox-data` volume before ArchiveBox starts. | ArchiveBox can list or index restored entries; sampled archived item metadata is present; no crawler starts during validation. | Archive UI or CLI sees expected snapshots and sampled data matches manifest. |
| `ontology-platform_oxigraph-data` | Restore into an empty `ontology-platform_oxigraph-data` volume before Oxigraph starts. | Oxigraph starts healthy; read-only SPARQL smoke query returns expected graph/count sample; no write traffic during validation. | Ontology API/query path can read restored triplestore data and sampled counts match evidence. |
| `ontology-platform_source-materials-minio-data` | Restore into an empty `ontology-platform_source-materials-minio-data` volume before MinIO starts. | MinIO starts healthy; expected buckets are visible; sampled source objects read back with matching checksums. | Source-material ingest/read paths can access restored buckets and sampled objects match manifest. |

## Boundary with `/data/hermes` coverage

HAD-1155 defines backup coverage for the planned `/data/hermes` roots that
preserve home-path contracts:

- `/data/hermes/profile-default` for `/home/david/.hermes`
- `/data/hermes/stacks` for `/home/david/stacks`
- `/data/hermes/ctx-data` for `/home/david/.ctx-data`
- `/data/hermes/codex-home` for `/home/david/.codex`

Those roots cover home-path bind mounts and repo-adjacent data under
`/home/david/stacks`, such as bind-mounted service directories that physically
live in the stacks tree. They do not cover Docker named volumes stored under Docker's data root.

Backup manifests must list each Docker named volume as a separate source. Do
not infer coverage from a Compose file, image tag, container name, or
`/data/hermes` backup success. The backup boundary is accepted only when both
home-path backups and the four named-volume backups have independent evidence.

## Failure modes and rollback notes

| Failure mode | Risk | Rollback or response |
|---|---|---|
| Writer active during archive capture | Archive may contain inconsistent databases, partial objects, or half-written indexes. | Reject the backup, restore nothing from it, re-run after quiescence proof. |
| Restore into non-empty production volume | Live data can be merged, overwritten, or made unrecoverable. | Do not restore over non-empty volumes. Keep original volumes intact until replacement passes checks. |
| Missing or stale Compose labels | Backup may target the wrong volume or old project. | Re-run discovery, compare active containers, and require manager acceptance for any renamed or duplicate volume. |
| Ownership or mode drift after restore | Containers may fail to read or write restored files. | Use `tar --numeric-owner`, record helper image digest, and validate ownership/mode in rehearsal. |
| Oxigraph store inconsistency | Triplestore may fail to start or return corrupt query results. | Prefer stopped/quiesced backup, run service health and SPARQL smoke checks before acceptance. |
| MinIO object-store inconsistency | Buckets may list but sampled objects can be missing or corrupt. | Capture object inventory/checksum samples and validate them after scratch restore. |
| ArchiveBox absolute path or index mismatch | Archive UI may start but restored entries may be inaccessible. | Run ArchiveBox list/index smoke checks and sampled item readback during rehearsal. |
| Backup target unavailable or checksum mismatch | Backup artifact is not restorable. | Treat backup as failed; keep source volumes and retry to a verified target. |
| Docker data-root migration handled separately | Volumes may be moved by Docker rather than restored from archives. | Still require named-volume evidence or an accepted Docker data-root migration plan with equivalent restore proof. |

Rollback posture:

- Keep original production volumes until restored services pass smoke checks.
- Keep backup archives and evidence until the migration rollback window closes.
- If restored service validation fails, stop using the restored target and return
  to the original untouched volume or the previous host snapshot.
- Do not delete stale-looking duplicate volumes until discovery proves they are
  not active and the manager accepts deletion.

## Verification checklist and acceptance evidence

Checklist for this runbook artifact:

- The four required Docker named volumes are inventoried with service ownership.
- Discovery commands cover volumes, active containers, Compose labels, and
  Compose project listing.
- Backup plan names quiescence dependencies and evidence capture for each
  volume.
- Restore plan requires scratch-host rehearsal before production restore.
- Boundary with `/data/hermes`, bind mounts, and home-path backup coverage is
  explicit.
- Failure modes and rollback notes are documented.
- Unknowns and follow-ups are listed.
- No migration, backup, restore, service lifecycle, cron, webhook, or
  notification command is required by this artifact.

Acceptance evidence for the later backup/restore execution:

- `volume-inspect.json` for the four required volumes.
- `container-inspect.json` and `volume-users.tsv` for active volume consumers.
- Compose project evidence and config labels.
- Per-volume quiescence evidence with secrets redacted.
- Per-volume backup archive name, size, checksum, timestamp, and helper image
  digest.
- Backup readback checksum verification.
- Scratch-host restore log for every volume.
- Service smoke-check output for Hadto pipeline, ArchiveBox, Oxigraph, and
  MinIO source-materials storage.
- Manager acceptance that Docker named-volume coverage closes the HAD-1155
  service-storage boundary.

## Unknowns and follow-ups

- Confirm the canonical Compose working directories for `hadto-pipeline` and
  `ontology-platform` from Compose labels or repo runbooks before execution.
- Define exact app-level quiescence commands for Hadto pipeline, ArchiveBox,
  Oxigraph, and MinIO without exposing secrets.
- Decide the backup target, encryption method, retention policy, and restore
  rehearsal host.
- Decide whether backup archives should be plain tar, compressed tar, or a
  backup-system-native artifact. Record the chosen tool and version in evidence.
- Determine whether older similarly named volumes, such as `server_*`,
  `ont040-*`, or anonymous hash-named volumes, are stale or require separate
  owner signoff.
- Confirm whether Docker's data root itself will later move to `/data`; if so,
  align that migration with this named-volume restore evidence instead of
  treating Docker storage as implicitly covered by `/data/hermes`.
- Create or confirm a follow-up restore rehearsal issue if production recovery
  cannot be accepted from documentation-only evidence.
