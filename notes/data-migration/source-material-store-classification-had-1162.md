# HAD-1162 source-material store classification

Issue: HAD-1162, Classify service-specific source-material stores outside
`/data/hermes`.

This artifact follows HAD-1155 and classifies ontology and Hadto source-material
stores that are not currently under `/data/hermes`. It is planning-only: no
migration, backup, mount, copy, delete, chmod/chown, Docker stop/start, or
systemd stop/start command was run to produce it.

## Scope and assumptions

- Current dry-run `/data/hermes` roots from HAD-1155 are
  `/data/hermes/profile-default`, `/data/hermes/stacks`,
  `/data/hermes/ctx-data`, and `/data/hermes/codex-home`.
- Repo-adjacent stores under `/home/david/stacks` should become covered by the
  planned `/data/hermes/stacks` path contract, but they still need explicit
  source-material classification so they are not treated as caches or build
  artifacts.
- Docker named volumes under `/var/lib/docker/volumes` are outside those four
  dry-run roots and need a separate Docker-volume backup and restore path.
- Hermes temporary source queues under `/home/david/.hermes/tmp` are under the
  planned profile root, but HAD-1155 allows temporary paths to be excluded. Any
  source files still in these queues must be promoted or explicitly retained
  before backup acceptance.
- Secrets were not printed or queried. MinIO bucket contents were not listed
  because credentialed object-store access is outside this read-only artifact.

## Evidence commands

Read-only commands used for this classification:

```bash
git status --short --branch
rg --files notes/data-migration
sed -n '1,240p' notes/data-migration/backup-boundary-restore-readiness-closeout.md
docker volume ls --format '{{.Name}}' | sort | rg -i 'ontology|hadto|source|material|minio|archive|oxigraph|pipeline|blob|object|ctx|hermes'
docker volume inspect hadto-pipeline_pipeline-data ont040-archivebox-data ontology-platform_archivebox-data ontology-platform_evolver-logs ontology-platform_evolver-orsd ontology-platform_evolver-proposals ontology-platform_oxigraph-data ontology-platform_source-materials-minio-data server_oxigraph-data server_pipeline-data --format '{{.Name}}\t{{.Mountpoint}}\t{{json .Labels}}'
docker inspect ontology-archivebox hadto-ontology-workbench ontology-triplestore hadto-pipeline ontology-source-materials-blob-store ontology-validator ontology-api --format '{{.Name}}\t{{.Config.Image}}\t{{json .Mounts}}'
docker system df -v
find /home/david/stacks -maxdepth 2 -mindepth 1 -type d \( -iname '*ontology*' -o -iname '*hadto*' -o -iname '*source*' -o -iname '*archive*' -o -iname '*minio*' -o -iname '*blob*' -o -iname '*data*' -o -iname '*materials*' \) -printf '%p\n' | sort
find /home/david/stacks/smb-ontology-platform/research/source_store -type f | wc -l
find /home/david/stacks/smb-ontology-platform/research/manifests -type f | wc -l
du -sh /home/david/stacks/smb-ontology-platform/research/source_store /home/david/stacks/smb-ontology-platform/research/archivebox_exports /home/david/stacks/smb-ontology-platform/orsd /home/david/stacks/smb-ontology-platform/proposals /home/david/stacks/hadto-decision-room-data/decision-room /home/david/stacks/hadto-ontology-workbench/data /home/david/stacks/hadto-pipeline/.local/data
find /home/david/.hermes/tmp -maxdepth 1 -type d \( -iname 'ontology-source*' -o -iname 'ontology-sources*' -o -iname 'ontology-research*' -o -iname '*ontology*source*' \) -printf '%p\n' | sort
du -sh /home/david/.hermes/tmp/ontology-source-candidates /home/david/.hermes/tmp/ontology-source-transfer-20260531 /home/david/.hermes/tmp/ontology-sources-2026-05-26 /home/david/.hermes/tmp/ontology-research-cycle /home/david/.hermes/tmp/ontology-research-oklahoma /home/david/.hermes/tmp/ontology-oklahoma-2026-05-24
du -sh /home/david/Downloads/taildrop/Keet* /home/david/code/de-novo/ontology
curl -fsS http://127.0.0.1:9100/minio/health/live
curl -fsS -G --data-urlencode 'query=ASK { ?s ?p ?o }' http://127.0.0.1:7878/query -H 'Accept: text/csv'
curl -fsSI http://127.0.0.1:9111
curl -fsS http://127.0.0.1:5100/api/health
curl -fsS http://127.0.0.1:5101/health
```

Direct `ls -ld` probes of selected Docker volume mountpoints returned
`Permission denied`, so file-level Docker-volume contents remain an uncovered
region for a later root or container-assisted read-only verifier.

## Classified stores

| Store | Owner/service | Current path or volume | Mutability | Backup class | Migration posture | Verification probe |
|---|---|---|---|---|---|---|
| MinIO source-material object store | `ontology-platform` / `ontology-source-materials-blob-store` | Docker volume `ontology-platform_source-materials-minio-data` at `/var/lib/docker/volumes/ontology-platform_source-materials-minio-data/_data`, mounted as `/data` | Mutable by MinIO writes and source publication jobs | `important-durable` | Not covered by `/data/hermes/stacks`. Move through a Docker-volume backup/restore plan or re-home the volume under a data-disk service-volume path before service start. Quiesce writers before snapshot. | `docker volume inspect`, `docker inspect ontology-source-materials-blob-store`, `docker system df -v`, `curl -fsS http://127.0.0.1:9100/minio/health/live`; later credential-safe bucket inventory without logging secrets. |
| ArchiveBox service store | `ontology-platform` / `ontology-archivebox` | Docker volume `ontology-platform_archivebox-data` at `/var/lib/docker/volumes/ontology-platform_archivebox-data/_data`, mounted as `/data` | Mutable by ArchiveBox capture and indexing | `important-durable` | Backup and restore with Docker volumes before ArchiveBox starts. Do not assume repo `archivebox_exports` contains the full service state. | `docker inspect ontology-archivebox`, `docker system df -v`, `curl -fsSI http://127.0.0.1:9111`; later read-only ArchiveBox data inventory. |
| Oxigraph triplestore | `ontology-platform` / `ontology-triplestore` | Docker volume `ontology-platform_oxigraph-data` at `/var/lib/docker/volumes/ontology-platform_oxigraph-data/_data`, mounted as `/data` | Mutable by ontology loader and triplestore writes | `critical-durable` | Restore before validator/API/workbench consumers start. Prefer service-specific dump or cold volume backup plus post-restore SPARQL smoke check. | `docker inspect ontology-triplestore`, `docker system df -v`, `curl -fsS -G --data-urlencode 'query=ASK { ?s ?p ?o }' http://127.0.0.1:7878/query -H 'Accept: text/csv'`. |
| Hadto pipeline SQLite store | `hadto-pipeline` | Docker volume `hadto-pipeline_pipeline-data` at `/var/lib/docker/volumes/hadto-pipeline_pipeline-data/_data`, mounted as `/app/data` | Mutable by the pipeline app | `critical-durable` | Restore before pipeline starts. Treat SQLite WAL/SHM as a writer-quiescence concern, not disposable noise. | `docker inspect hadto-pipeline`, `docker system df -v`, `curl -fsS http://127.0.0.1:5100/api/health`; later `PRAGMA integrity_check` against a restored copy. |
| Ontology evolver named volumes | `ontology-platform` / `ontology-evolver` | Docker volumes `ontology-platform_evolver-orsd`, `ontology-platform_evolver-proposals`, and `ontology-platform_evolver-logs` | Intermittently mutable when evolver jobs run; currently unlinked in `docker system df -v` evidence | `important-durable` for ORSD/proposals, `operational-history` for logs | Preserve until compared with repo paths under `/home/david/stacks/smb-ontology-platform`. Do not delete zero-size/unlinked volumes without owner signoff. | `docker volume inspect`, `docker system df -v`; later root/container-assisted read-only listing and content comparison to repo `orsd/`, `proposals/`, and `evolution/logs/`. |
| SMB ontology content-addressed source store | `smb-ontology-platform` research loop | `/home/david/stacks/smb-ontology-platform/research/source_store` | Mutable by source capture and research-cycle tooling | `important-durable` | Covered only if `/home/david/stacks` is migrated and backup excludes do not treat it as cache. Preserve as source material, including untracked objects. | `du -sh` showed about `121M`; `find ... -type f` found `130` files, `129` non-empty content blobs; verify manifest references resolve after restore. |
| SMB ontology source manifests | `smb-ontology-platform` research loop | `/home/david/stacks/smb-ontology-platform/research/manifests` | Mutable by source capture and research-cycle tooling | `important-durable` | Must move with `/home/david/stacks`; manifests are the provenance index for `research/source_store` and MinIO publication. | `find .../research/manifests -type f | wc -l` found `109`; run `python3 tools/validate_source_manifests.py` after restore. |
| Repo-local ArchiveBox exports | `smb-ontology-platform` research examples/export path | `/home/david/stacks/smb-ontology-platform/research/archivebox_exports` | Low-frequency writes from export/capture tooling | `important-durable` | Covered by `/home/david/stacks` if not excluded. Treat as a partial export, not a substitute for the Docker ArchiveBox volume. | `du -sh` showed about `236K`; restore probe should compare expected export files and manifests. |
| ORSD and proposal queues | `smb-ontology-platform` ontology evolution | `/home/david/stacks/smb-ontology-platform/orsd`, `/home/david/stacks/smb-ontology-platform/proposals`, `/home/david/stacks/smb-ontology-platform/evolution/logs` | Mutable by humans, evolver jobs, and review workflows | `critical-durable` for ORSD/proposals; `operational-history` for logs | Covered by `/home/david/stacks`; explicitly protect from cache/build exclusions. Compare with evolver Docker volumes before deciding which copy is authoritative. | `du -sh` showed about `616K` for `orsd` and `96K` for `proposals`; run ontology validation and proposal queue listing after restore. |
| Workbench decision-room bind data | `hadto-ontology-workbench` | `/home/david/stacks/hadto-decision-room-data/decision-room`, bind-mounted RW as `/data/decision-room` | Mutable by workbench flows and operator edits | `important-durable` | Under `/home/david/stacks`, but not a git repository. Ensure raw directory backup, ownership, and modes survive migration. | `docker inspect hadto-ontology-workbench`; `du -sh` showed about `32K`; restore probe should load decision-room route and list JSON exchanges. |
| Workbench local data directory | `hadto-ontology-workbench` | `/home/david/stacks/hadto-ontology-workbench/data` | Mutable in local/demo workflows | `important-durable` until owner declares it duplicate | Covered by `/home/david/stacks`; reconcile with `hadto-decision-room-data` so stale duplicates are not mistaken for the service source of truth. | `du -sh` showed about `32K`; file probe found client-interview and decision-room JSON examples. |
| Hadto pipeline host-local SQLite residue | `hadto-pipeline` development/local path | `/home/david/stacks/hadto-pipeline/.local/data/data.db` | Unknown; live container uses Docker volume `/app/data` | `important-durable` pending owner review, likely demotable if empty/stale | Covered by `/home/david/stacks`, but should be reconciled against `hadto-pipeline_pipeline-data` before restore docs call it authoritative. | `du -sh` showed `0`; later `sqlite3 ... 'PRAGMA integrity_check'` if non-empty. |
| Hermes ontology source candidate queues | Hermes research operators | `/home/david/.hermes/tmp/ontology-source-candidates`, `/home/david/.hermes/tmp/ontology-source-transfer-20260531`, `/home/david/.hermes/tmp/ontology-sources-2026-05-26`, plus ontology research temp directories | Mutable scratch and transfer queues | `important-durable` until promoted or explicitly discarded | High-risk because HAD-1155 permits tmp exclusions. Promote accepted files into `smb-ontology-platform/research/source_store` plus manifests or MinIO before migration; otherwise add an explicit backup exception and owner signoff. | `du -sh` showed about `10M`, `2.8M`, and `792K` for the named source queues, plus ontology temp research directories. Probe file names without printing contents. |
| Hermes ontology research notes | Hermes notes | `/home/david/.hermes/notes/ontology-research-cycle` | Mutable by cron/research runs | `important-durable` | Covered by `/data/hermes/profile-default`; keep out of tmp/cache exclusions. | `find ... -maxdepth 2 -type f` showed dated research notes through `2026-05-31`; restore probe should confirm note count and latest dated note. |
| Taildrop ontology book/source copies | Manual source intake | `/home/david/Downloads/taildrop/Keet*Ontology*` | Usually immutable after download/decryption | `important-durable` if used by ontology research; otherwise owner-decision holdback | Outside all four dry-run roots. Import into a durable source store or add an explicit backup include/exclusion decision before migration acceptance. | `du -sh /home/david/Downloads/taildrop/Keet*` listed PDF and text copies totaling roughly `22M`. |
| De Novo ontology root | `de-novo` / cross-project ontology | `/home/david/code/de-novo/ontology` | Mutable by De Novo ontology work, not the live Hadto service path | `important-durable` for De Novo; `uncovered-by-Hermes-dry-run` | Outside all four dry-run roots. Do not silently fold into Hadto restore order, but record as an ontology root requiring a separate owner decision. | `find /home/david/code/de-novo/ontology -maxdepth 2 -type f`; `du -sh` showed about `40K`. |

## Legacy or unknown Docker volumes

These volumes matched the ontology/Hadto search but were not linked to a
currently running container in the collected evidence:

| Volume | Evidence | Classification | Required follow-up |
|---|---|---|---|
| `server_oxigraph-data` | Docker label `com.docker.compose.project=server`; `docker system df -v` showed `0` links and about `6.527MB` | Unknown ontology/triplestore residue, default `important-durable` until owner review | Root/container-assisted read-only listing, compare to `ontology-platform_oxigraph-data`, then decide backup, archive, or deletion. |
| `server_pipeline-data` | Docker label `com.docker.compose.project=server`; `0` links and about `106.9kB` | Unknown pipeline residue, default `important-durable` until owner review | Inspect for SQLite or exported state before excluding. |
| `ont040-archivebox-data` | No compose labels; `0` links and about `528.8kB` | Unknown ArchiveBox residue, default `important-durable` until owner review | Identify origin or snapshot contents before excluding. |

## Migration posture by store family

- Docker named volumes are not covered by the four dry-run `/data/hermes`
  subvolumes. They need a Docker-volume backup/restore plan, a data-disk-backed
  volume location, or a cold export/import procedure with writer quiescence.
- Repo-adjacent stores under `/home/david/stacks` are covered only if the stacks
  path contract preserves raw working trees, untracked files, and non-git
  directories. Backup filters must not exclude `research/source_store`, ORSD,
  proposals, decision-room data, or source manifests as build artifacts.
- Hermes temporary ontology source queues must not be swept up by generic tmp
  exclusions until every accepted source is promoted to a durable store or
  explicitly discarded by the owner.
- External source intake under `/home/david/Downloads/taildrop` and
  `/home/david/code/de-novo/ontology` is outside the current dry-run boundary.
  These paths need either import into a managed root or an explicit accepted
  omission.

## Uncovered regions

- Docker volume file-level contents were not readable by the unprivileged
  inspection path. Selected `/var/lib/docker/volumes/.../_data` probes returned
  `Permission denied`.
- MinIO bucket inventory was not collected because it requires object-store
  credentials; only local health and Docker-volume evidence were collected.
- Old agent worktrees under `/home/david/stacks/agent-*ontology*` and
  `/home/david/stacks/agent-*hadto*` were enumerated only at top level. They are
  covered by the stacks path contract, but their deep source-material contents
  were not exhaustively classified here.
- Active backup jobs and backup-storage readback were not inspected. This
  artifact classifies source stores; it does not prove recurring backup
  coverage.
- PhoneItIn MinIO/Postgres volumes matched broad storage keywords but are not
  Hadto or ontology service stores for this issue. They need separate owner
  classification if the data-disk migration boundary expands.

## Acceptance checklist

- Docker-backed object/blob stores are explicitly identified:
  `ontology-platform_source-materials-minio-data`,
  `ontology-platform_archivebox-data`, `ontology-platform_oxigraph-data`, and
  `hadto-pipeline_pipeline-data`.
- Repo-adjacent source/material stores are explicitly identified:
  `research/source_store`, `research/manifests`, `research/archivebox_exports`,
  `orsd`, `proposals`, `evolution/logs`, `hadto-decision-room-data`,
  `hadto-ontology-workbench/data`, and `hadto-pipeline/.local/data`.
- Temporary and external source-intake regions are not guessed away:
  `/home/david/.hermes/tmp/ontology-source-*`,
  `/home/david/Downloads/taildrop/Keet*Ontology*`, and
  `/home/david/code/de-novo/ontology` are classified with explicit migration
  decisions still required.
- Unknown legacy Docker volumes are held as `important-durable` until owner
  review proves they are stale or safely excluded.
