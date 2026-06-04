# Data Migration Maintenance Window and Writer Quiescence Runbook

Issue: HAD-1150

## 1. Scope and Safety Invariants

This document is a runbook for planning and operating a short maintenance
window before a Hermes data migration. It does not execute the migration and it
does not replace the migration command plan. Use it to prove that writers are
quiet before copying state.

The protected path contracts for the first migration pass are:

- `/home/david/.hermes`
- `/home/david/stacks`
- `/home/david/.ctx-data`
- `/home/david/.codex`
- planned `/data/hermes` paths used as migration targets or bind-mount sources

Safety invariants:

- Do not copy SQLite databases, WAL files, SHM files, repo `.git` state, Codex
  metadata, ctx state, or Docker bind-mounted files while their writers are
  active.
- Do not start copy, rename, mount, bind-mount, delete, ownership, or permission
  changes until the rollback plan in section 2 is filled in and verified.
- Do not rewrite application-visible paths as the first migration step. Preserve
  `/home/david/.hermes`, `/home/david/stacks`, `/home/david/.ctx-data`, and
  `/home/david/.codex` through bind mounts or equivalent path-contract
  preservation.
- Treat missing inventory as blocking unless the operator explicitly records
  the uncovered region in section 10 before proceeding.
- Treat active unknown writers as blocking. The preflight gate below fails
  closed by design.

Dependency note: HAD-1147 is expected to produce a path-contract inventory. No
path-contract inventory file was present in the checked repository when this
runbook was written. Until HAD-1147 lands, validate this runbook independently
with the discovery commands below and record any additional path contracts in
section 10 before the maintenance window.

Current-host evidence used while drafting this runbook:

- Inspected checkout paths: `/home/david/stacks/hermes-agent` and
  `/home/david/stacks/had-1150-hermes-agent`.
- Inspected migration-root existence: `/home/david/.hermes`,
  `/home/david/stacks`, `/home/david/.ctx-data`, `/home/david/.codex`, and
  `/data/hermes`.
- Inspected command surfaces: `git status --short --branch`,
  `rg --files`, `systemctl --user list-units`, `systemctl --user list-timers`,
  `docker ps`, `docker inspect`, `command -v systemctl`, `command -v docker`,
  `command -v sqlite3`, `command -v codex`, `command -v ctx`,
  `fuser --version`.
- Host observations at drafting time: `hermes-gateway.service` existed as an
  active user service; `systemctl --user list-timers --all` showed operator
  timers including `ops-heartbeat.timer`, `hourly-status.timer`,
  `lenovo-backup.timer`, `lenovo-backup-verify.timer`,
  `hadto-ops-*.timer`, and `hermes-codex-auth-health.timer`; Docker had active
  containers with bind mounts under `/home/david/stacks`; `lsof` was not
  installed, so the file-handle gate uses `/proc` and `fuser`-available hosts.

## 2. Rollback Plan Before First Mutating Step

Fill these variables before stopping any writer. The first mutating step in this
runbook is stopping services, timers, or containers.

```bash
export MIGRATION_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
export MIGRATION_EVIDENCE_DIR="/home/david/.hermes/data-migration-evidence/${MIGRATION_RUN_ID}"
export MIGRATION_TARGET_ROOT="/data/hermes"
export HERMES_HOLDBACK="/home/david/.hermes.holdback-${MIGRATION_RUN_ID}"
export STACKS_HOLDBACK="/home/david/stacks.holdback-${MIGRATION_RUN_ID}"
export CTX_HOLDBACK="/home/david/.ctx-data.holdback-${MIGRATION_RUN_ID}"
export CODEX_HOLDBACK="/home/david/.codex.holdback-${MIGRATION_RUN_ID}"
```

Rollback is possible while all of these are true:

- Original source directories still exist or their holdback directories are
  intact.
- No target path under `/data/hermes` has been accepted as the only writable
  source of truth.
- Bind mounts and `/etc/fstab` changes have not been made permanent without a
  verified restore path.
- No writer has been restarted against partially copied or partially mounted
  state.

Do not proceed past these points:

- Do not stop writers until this section has a run ID, evidence directory, and
  holdback paths.
- Do not start copying until the preflight gate in section 4 passes.
- Do not activate bind mounts until copy checksums, SQLite checks, git status,
  and Docker bind-mount inventory are captured.
- Do not remove holdbacks until post-window verification in section 9 passes and
  the operator explicitly closes the rollback window.

Rollback commands if only writers were stopped:

```bash
docker start hadto-ontology-workbench phoneitin-monitoring-prometheus-1 phoneitin-monitoring-node-exporter-1
systemctl --user start ops-heartbeat.timer hourly-status.timer lenovo-backup.timer lenovo-backup-verify.timer hadto-ops-backup-restore-refresh.timer hadto-ops-capacity-refresh.timer hadto-ops-dashboard-feeds.timer hadto-ops-engineering-velocity-refresh.timer hadto-ops-hermes-health.timer hermes-codex-auth-health.timer
systemctl --user start hermes-gateway.service
systemctl --user --no-pager --plain status hermes-gateway.service
docker ps --format '{{.Names}} {{.Status}}'
```

Rollback commands if copy occurred but bind mounts were not activated:

```bash
rsync -aHAX --numeric-ids --dry-run "${MIGRATION_TARGET_ROOT}/profile-default/" /home/david/.hermes/
rsync -aHAX --numeric-ids --dry-run "${MIGRATION_TARGET_ROOT}/stacks/" /home/david/stacks/
rsync -aHAX --numeric-ids --dry-run "${MIGRATION_TARGET_ROOT}/ctx-data/" /home/david/.ctx-data/
rsync -aHAX --numeric-ids --dry-run "${MIGRATION_TARGET_ROOT}/codex-home/" /home/david/.codex/
```

If those dry-runs show unexpected reverse changes, do not restart writers. Keep
the maintenance window open and capture the evidence in section 9.

Rollback commands if bind mounts were activated:

```bash
findmnt -R /home/david/.hermes /home/david/stacks /home/david/.ctx-data /home/david/.codex
sudo umount /home/david/.hermes
sudo umount /home/david/stacks
sudo umount /home/david/.ctx-data
sudo umount /home/david/.codex
sudo mv "${HERMES_HOLDBACK}" /home/david/.hermes
sudo mv "${STACKS_HOLDBACK}" /home/david/stacks
sudo mv "${CTX_HOLDBACK}" /home/david/.ctx-data
sudo mv "${CODEX_HOLDBACK}" /home/david/.codex
systemctl --user start ops-heartbeat.timer hourly-status.timer lenovo-backup.timer lenovo-backup-verify.timer hadto-ops-backup-restore-refresh.timer hadto-ops-capacity-refresh.timer hadto-ops-dashboard-feeds.timer hadto-ops-engineering-velocity-refresh.timer hadto-ops-hermes-health.timer hermes-codex-auth-health.timer
systemctl --user start hermes-gateway.service
docker start hadto-ontology-workbench phoneitin-monitoring-prometheus-1 phoneitin-monitoring-node-exporter-1
```

If `/etc/fstab` was edited, restore the previous fstab from the migration
evidence copy before rebooting:

```bash
sudo cp "${MIGRATION_EVIDENCE_DIR}/fstab.before" /etc/fstab
sudo findmnt --verify
```

## 3. Writer Inventory

Run these discovery commands before the window. The expected quiet states apply
after the stop order in section 5 has completed.

### Hermes Gateway

Discovery:

```bash
systemctl --user list-units --type=service --all --no-pager --plain 'hermes*'
systemctl --user --no-pager --plain status hermes-gateway.service
python - <<'PY'
from pathlib import Path
for p in (Path('/home/david/.hermes/gateway.pid'), Path('/home/david/.hermes/gateway_state.json'), Path('/home/david/.hermes/gateway.lock')):
    print(f'{p}: {"present" if p.exists() else "missing"}')
PY
```

Expected quiet state:

- `hermes-gateway.service` is inactive.
- No live process owns `/home/david/.hermes/gateway.pid` or
  `/home/david/.hermes/gateway.lock`.
- No Hermes gateway process has an open handle under the copied roots.

### Hermes Cron Jobs and User Timers

Discovery:

```bash
systemctl --user list-timers --all --no-pager --plain
systemctl --user list-units --type=service --all --no-pager --plain '*cron*' '*timer*' 'hadto-ops-*' 'ops-*' 'hourly-*' 'lenovo-*' 'hermes-codex-*'
python - <<'PY'
from pathlib import Path
for p in (Path('/home/david/.hermes/cron'), Path('/home/david/.hermes/cron/jobs.json'), Path('/home/david/.hermes/cron/.tick.lock')):
    print(f'{p}: {"present" if p.exists() else "missing"}')
PY
```

Expected quiet state:

- Relevant timers are inactive for the whole copy window.
- No cron tick lock is held.
- No `hermes cron`, `cron/scheduler.py`, or gateway-dispatched cron worker is
  running.

Timers observed on this host that must be classified before the window:

```bash
systemctl --user --no-pager --plain status ops-heartbeat.timer hourly-status.timer lenovo-backup.timer lenovo-backup-verify.timer hadto-ops-backup-restore-refresh.timer hadto-ops-capacity-refresh.timer hadto-ops-dashboard-feeds.timer hadto-ops-engineering-velocity-refresh.timer hadto-ops-hermes-health.timer hermes-codex-auth-health.timer
```

### Tracked Codex Runs and Process Registry

Discovery:

```bash
python - <<'PY'
import os
from pathlib import Path

roots = [Path('/home/david/.codex'), Path('/home/david/stacks')]
for pid in filter(str.isdigit, os.listdir('/proc')):
    try:
        comm = Path('/proc') / pid / 'comm'
        cmd = (Path('/proc') / pid / 'cmdline').read_bytes().replace(b'\0', b' ').decode('utf-8', 'replace')
    except OSError:
        continue
    if 'codex' not in cmd.lower():
        continue
    cwd = None
    try:
        cwd = os.readlink(Path('/proc') / pid / 'cwd')
    except OSError:
        pass
    print(f'pid={pid} comm={comm.read_text(errors="replace").strip()} cwd={cwd}')
PY
find /home/david/stacks -path '*/.git/hermes-codex' -type d -prune -print
find /home/david/stacks -path '*/.git/hermes-codex/*' -type f -mmin -5 -print
find /home/david/.codex -type f -mmin -5 -print
```

Expected quiet state:

- No `codex exec`, delegated Codex worker, or Codex process has cwd or file
  handles under the copied roots.
- `.git/hermes-codex` registry files and `/home/david/.codex` metadata are not
  changing during the window.
- Any active Codex task is complete, cancelled by its owner before the window,
  or explicitly excluded from copied roots.

### Hermes Process Sessions

Discovery:

```bash
python - <<'PY'
import os
from pathlib import Path

needles = ('hermes', 'run_agent.py', 'hermes_cli', 'tui_gateway', 'gateway/run.py', 'cron/scheduler.py')
for pid in filter(str.isdigit, os.listdir('/proc')):
    try:
        cmd = (Path('/proc') / pid / 'cmdline').read_bytes().replace(b'\0', b' ').decode('utf-8', 'replace')
        comm = (Path('/proc') / pid / 'comm').read_text(errors='replace').strip()
    except OSError:
        continue
    if any(n in cmd for n in needles):
        try:
            cwd = os.readlink(Path('/proc') / pid / 'cwd')
        except OSError:
            cwd = None
        print(f'pid={pid} comm={comm} cwd={cwd}')
PY
find /home/david/.hermes/sessions -type f -mmin -5 -print
find /home/david/.hermes/logs -type f -mmin -5 -print
```

Expected quiet state:

- No live Hermes CLI, TUI gateway, ACP, dashboard PTY, batch runner, cron
  worker, or gateway process writes under `/home/david/.hermes`.
- Session JSONL and log files are not changing during the copy window.

### ctx Daemon and Sessions

Discovery:

```bash
command -v ctx
python - <<'PY'
import os
from pathlib import Path

for pid in filter(str.isdigit, os.listdir('/proc')):
    try:
        cmd = (Path('/proc') / pid / 'cmdline').read_bytes().replace(b'\0', b' ').decode('utf-8', 'replace')
        comm = (Path('/proc') / pid / 'comm').read_text(errors='replace').strip()
    except OSError:
        continue
    if 'ctx' in cmd.lower():
        try:
            cwd = os.readlink(Path('/proc') / pid / 'cwd')
        except OSError:
            cwd = None
        print(f'pid={pid} comm={comm} cwd={cwd}')
PY
find /home/david/.ctx-data -type f -mmin -5 -print
```

Expected quiet state:

- No ctx daemon, ctx session worker, or ctx-backed service is writing under
  `/home/david/.ctx-data`.
- If ctx has no daemon on the host, record that as evidence and still require
  the file-handle gate to pass.

### Docker Containers and Services With Bind Mounts

Discovery:

```bash
docker ps --format '{{.ID}} {{.Names}} {{.Status}}'
docker inspect --format '{{.Name}} {{range .Mounts}}{{.Type}}:{{.Source}}->{{.Destination}} {{end}}' $(docker ps -q)
python - <<'PY'
import json
import subprocess
from pathlib import Path

roots = [Path('/home/david/.hermes'), Path('/home/david/stacks'), Path('/home/david/.ctx-data'), Path('/home/david/.codex'), Path('/data/hermes')]
roots = [r.resolve() for r in roots if r.exists()]

def touches_copied_root(source_resolved, root):
    # Catch bind sources under a copied root and ancestor bind sources
    # such as /home/david or / exposing /home/david/.hermes.
    return (
        source_resolved == root
        or source_resolved in root.parents
        or root in source_resolved.parents
    )

ids = subprocess.run(['docker', 'ps', '-q'], check=True, text=True, capture_output=True).stdout.split()
if not ids:
    raise SystemExit(0)
data = json.loads(subprocess.run(['docker', 'inspect', *ids], check=True, text=True, capture_output=True).stdout)
for container in data:
    name = container.get('Name', '').lstrip('/')
    for mount in container.get('Mounts', []):
        if mount.get('Type') != 'bind':
            continue
        source = Path(mount.get('Source', ''))
        try:
            source_resolved = source.resolve()
        except OSError:
            source_resolved = source
        if any(touches_copied_root(source_resolved, root) for root in roots):
            print(f'{name}: {source}->{mount.get("Destination")}')
PY
```

Expected quiet state:

- Every running container with a bind mount touching a copied root is either
  stopped or explicitly classified as read-only and safe.
- Current-host bind mounts observed during drafting that touch copied roots and
  need classification include:
  `hadto-ontology-workbench`, `phoneitin-monitoring-prometheus-1`, and
  `phoneitin-monitoring-node-exporter-1`.

### SQLite WAL and SHM Files

Discovery:

```bash
find /home/david/.hermes /home/david/stacks /home/david/.ctx-data /home/david/.codex -type f \( -name '*-wal' -o -name '*-shm' -o -name '*.db-wal' -o -name '*.db-shm' \) -print
find /home/david/.hermes /home/david/stacks /home/david/.ctx-data /home/david/.codex -type f \( -name '*.db' -o -name '*.sqlite' -o -name '*.sqlite3' \) -print
```

Expected quiet state:

- No SQLite DB has an active writer.
- WAL and SHM files are either absent after checkpointing or explicitly tied to
  a stopped writer and copied as part of a documented SQLite-safe snapshot.
- `sqlite3 "$db" 'PRAGMA quick_check;'` returns `ok` for every DB in scope.

### Open File Handles Under Copied Paths

Discovery:

```bash
python - <<'PY'
import os
from pathlib import Path

roots = [Path('/home/david/.hermes'), Path('/home/david/stacks'), Path('/home/david/.ctx-data'), Path('/home/david/.codex'), Path('/data/hermes')]
roots = [r.resolve() for r in roots if r.exists()]
seen = set()
for pid in filter(str.isdigit, os.listdir('/proc')):
    proc = Path('/proc') / pid
    for label in ('cwd', 'root', 'exe'):
        try:
            target = Path(os.readlink(proc / label)).resolve()
        except OSError:
            continue
        if any(target == root or root in target.parents for root in roots):
            seen.add((pid, label, str(target)))
    fd_dir = proc / 'fd'
    try:
        entries = list(fd_dir.iterdir())
    except OSError:
        continue
    for fd in entries:
        try:
            target = Path(os.readlink(fd)).resolve()
        except OSError:
            continue
        if any(target == root or root in target.parents for root in roots):
            seen.add((pid, f'fd/{fd.name}', str(target)))
for pid, label, target in sorted(seen, key=lambda item: int(item[0])):
    try:
        comm = (Path('/proc') / pid / 'comm').read_text(errors='replace').strip()
    except OSError:
        comm = 'unknown'
    print(f'pid={pid} comm={comm} handle={label} path={target}')
PY
```

Expected quiet state:

- The command prints no unexpected PIDs.
- If it prints the operator's current shell, tmux, or read-only inspection
  process, record the PID and reason before proceeding.

### Dirty Git and Repo State

Discovery:

```bash
python - <<'PY'
import subprocess
from pathlib import Path

for git_dir in sorted(Path('/home/david/stacks').glob('*/.git')):
    repo = git_dir.parent
    status = subprocess.run(['git', '-C', str(repo), 'status', '--porcelain=v1'], text=True, capture_output=True)
    branch = subprocess.run(['git', '-C', str(repo), 'status', '--short', '--branch'], text=True, capture_output=True)
    if status.returncode != 0:
        print(f'FAIL {repo}: git status failed')
        continue
    if status.stdout.strip():
        print(f'DIRTY {repo}')
        print(status.stdout, end='')
    else:
        print(f'CLEAN {repo}: {branch.stdout.splitlines()[0] if branch.stdout else "branch unknown"}')
PY
```

Expected quiet state:

- Every repo under copied roots is clean, intentionally excluded, or has its
  dirty state captured in the evidence bundle.
- Do not copy a repo `.git` directory while Codex, Hermes, an IDE, or another
  writer is changing that repo.

## 4. Preflight Gate

Run this after the stop order in section 5 and before any copy, rename, mount,
or bind-mount step. It fails closed when unexpected writers are active.
Run it from `/tmp` or another directory outside the copied roots so the
operator shell does not create a false open-handle hit.

```bash
set -euo pipefail

: "${MIGRATION_RUN_ID:?set MIGRATION_RUN_ID before the window}"
: "${MIGRATION_EVIDENCE_DIR:?set MIGRATION_EVIDENCE_DIR before the window}"
mkdir -p "${MIGRATION_EVIDENCE_DIR}"

COPY_ROOTS=(
  /home/david/.hermes
  /home/david/stacks
  /home/david/.ctx-data
  /home/david/.codex
  /data/hermes
)

TIMER_UNITS=(
  ops-heartbeat.timer
  hourly-status.timer
  lenovo-backup.timer
  lenovo-backup-verify.timer
  hadto-ops-backup-restore-refresh.timer
  hadto-ops-capacity-refresh.timer
  hadto-ops-dashboard-feeds.timer
  hadto-ops-engineering-velocity-refresh.timer
  hadto-ops-hermes-health.timer
  hermes-codex-auth-health.timer
)

fail() {
  printf 'PREFLIGHT FAIL: %s\n' "$*" >&2
  exit 1
}

systemctl --user is-active --quiet hermes-gateway.service && fail 'hermes-gateway.service is still active'

for unit in "${TIMER_UNITS[@]}"; do
  if systemctl --user list-unit-files --no-pager --plain "$unit" | grep -q "$unit"; then
    systemctl --user is-active --quiet "$unit" && fail "$unit is still active"
  fi
done

for root in /home/david/.hermes /home/david/stacks /home/david/.ctx-data /home/david/.codex; do
  test -d "$root" || fail "source root missing: $root"
done

python - <<'PY' >"${MIGRATION_EVIDENCE_DIR}/open-handles.txt"
import os
from pathlib import Path

roots = [Path('/home/david/.hermes'), Path('/home/david/stacks'), Path('/home/david/.ctx-data'), Path('/home/david/.codex'), Path('/data/hermes')]
roots = [r.resolve() for r in roots if r.exists()]
allowed_pids = {str(os.getpid()), str(os.getppid())}
hits = []
for pid in filter(str.isdigit, os.listdir('/proc')):
    if pid in allowed_pids:
        continue
    proc = Path('/proc') / pid
    for label in ('cwd', 'root', 'exe'):
        try:
            target = Path(os.readlink(proc / label)).resolve()
        except OSError:
            continue
        if any(target == root or root in target.parents for root in roots):
            hits.append((pid, label, str(target)))
    fd_dir = proc / 'fd'
    try:
        entries = list(fd_dir.iterdir())
    except OSError:
        continue
    for fd in entries:
        try:
            target = Path(os.readlink(fd)).resolve()
        except OSError:
            continue
        if any(target == root or root in target.parents for root in roots):
            hits.append((pid, f'fd/{fd.name}', str(target)))
for pid, label, target in sorted(set(hits), key=lambda item: int(item[0])):
    try:
        comm = (Path('/proc') / pid / 'comm').read_text(errors='replace').strip()
    except OSError:
        comm = 'unknown'
    print(f'pid={pid} comm={comm} handle={label} path={target}')
PY

test ! -s "${MIGRATION_EVIDENCE_DIR}/open-handles.txt" || {
  cat "${MIGRATION_EVIDENCE_DIR}/open-handles.txt" >&2
  fail 'unexpected open handles under copied roots'
}

python - <<'PY' >"${MIGRATION_EVIDENCE_DIR}/docker-bind-writers.txt"
import json
import subprocess
from pathlib import Path

roots = [Path('/home/david/.hermes'), Path('/home/david/stacks'), Path('/home/david/.ctx-data'), Path('/home/david/.codex'), Path('/data/hermes')]
roots = [r.resolve() for r in roots if r.exists()]

def touches_copied_root(source_resolved, root):
    # Catch bind sources under a copied root and ancestor bind sources
    # such as /home/david or / exposing /home/david/.hermes.
    return (
        source_resolved == root
        or source_resolved in root.parents
        or root in source_resolved.parents
    )

ids = subprocess.run(['docker', 'ps', '-q'], check=True, text=True, capture_output=True).stdout.split()
if ids:
    data = json.loads(subprocess.run(['docker', 'inspect', *ids], check=True, text=True, capture_output=True).stdout)
    for container in data:
        name = container.get('Name', '').lstrip('/')
        for mount in container.get('Mounts', []):
            if mount.get('Type') != 'bind':
                continue
            source = Path(mount.get('Source', ''))
            try:
                source_resolved = source.resolve()
            except OSError:
                source_resolved = source
            if any(touches_copied_root(source_resolved, root) for root in roots):
                print(f'{name}: {source}->{mount.get("Destination")}')
PY

test ! -s "${MIGRATION_EVIDENCE_DIR}/docker-bind-writers.txt" || {
  cat "${MIGRATION_EVIDENCE_DIR}/docker-bind-writers.txt" >&2
  fail 'running Docker containers still bind copied roots'
}

python - <<'PY' >"${MIGRATION_EVIDENCE_DIR}/active-runtime-processes.txt"
import os
from pathlib import Path

needles = ('codex', 'hermes', 'run_agent.py', 'hermes_cli', 'tui_gateway', 'gateway/run.py', 'cron/scheduler.py', 'ctx')
for pid in filter(str.isdigit, os.listdir('/proc')):
    try:
        cmd = (Path('/proc') / pid / 'cmdline').read_bytes().replace(b'\0', b' ').decode('utf-8', 'replace')
        comm = (Path('/proc') / pid / 'comm').read_text(errors='replace').strip()
    except OSError:
        continue
    if any(n in cmd.lower() for n in needles):
        try:
            cwd = os.readlink(Path('/proc') / pid / 'cwd')
        except OSError:
            cwd = None
        print(f'pid={pid} comm={comm} cwd={cwd}')
PY

test ! -s "${MIGRATION_EVIDENCE_DIR}/active-runtime-processes.txt" || {
  cat "${MIGRATION_EVIDENCE_DIR}/active-runtime-processes.txt" >&2
  fail 'runtime process inventory is not quiet'
}

find /home/david/.hermes /home/david/stacks /home/david/.ctx-data /home/david/.codex -type f \( -name '*-wal' -o -name '*-shm' -o -name '*.db-wal' -o -name '*.db-shm' \) -print >"${MIGRATION_EVIDENCE_DIR}/sqlite-wal-shm.txt"
test ! -s "${MIGRATION_EVIDENCE_DIR}/sqlite-wal-shm.txt" || {
  cat "${MIGRATION_EVIDENCE_DIR}/sqlite-wal-shm.txt" >&2
  fail 'SQLite WAL/SHM files remain after writer stop and checkpoint'
}

python - <<'PY' >"${MIGRATION_EVIDENCE_DIR}/dirty-repos.txt"
import subprocess
from pathlib import Path

dirty = False
for git_dir in sorted(Path('/home/david/stacks').glob('*/.git')):
    repo = git_dir.parent
    status = subprocess.run(['git', '-C', str(repo), 'status', '--porcelain=v1'], text=True, capture_output=True)
    if status.returncode != 0:
        print(f'FAIL {repo}: git status failed')
        dirty = True
        continue
    if status.stdout.strip():
        print(f'DIRTY {repo}')
        print(status.stdout, end='')
        dirty = True
raise SystemExit(1 if dirty else 0)
PY

printf 'PREFLIGHT PASS: writer quiescence gate passed for %s\n' "${MIGRATION_RUN_ID}"
```

If the gate fails, use the blocked Slack message in section 8, capture the
evidence file named by the failure, and do not copy.

## 5. Stop Order and Per-Step Verification Probes

Stop writers from highest-level intake to lowest-level storage writers.

### 5.1 Announce the Window

Paste the start message from section 8. Record the exact timestamp:

```bash
date -Is
```

### 5.2 Stop Hermes Gateway Intake

```bash
systemctl --user stop hermes-gateway.service
systemctl --user is-active --quiet hermes-gateway.service && exit 1 || true
systemctl --user --no-pager --plain status hermes-gateway.service
```

Verification probe:

```bash
python - <<'PY'
from pathlib import Path
for p in (Path('/home/david/.hermes/gateway.pid'), Path('/home/david/.hermes/gateway.lock'), Path('/home/david/.hermes/gateway_state.json')):
    print(f'{p}: {"present" if p.exists() else "missing"}')
PY
```

### 5.3 Stop Hermes Cron and User Timers

```bash
systemctl --user stop ops-heartbeat.timer hourly-status.timer lenovo-backup.timer lenovo-backup-verify.timer hadto-ops-backup-restore-refresh.timer hadto-ops-capacity-refresh.timer hadto-ops-dashboard-feeds.timer hadto-ops-engineering-velocity-refresh.timer hadto-ops-hermes-health.timer hermes-codex-auth-health.timer
systemctl --user stop ops-heartbeat.service hourly-status.service lenovo-backup.service lenovo-backup-verify.service hadto-ops-backup-restore-refresh.service hadto-ops-capacity-refresh.service hadto-ops-dashboard-feeds.service hadto-ops-engineering-velocity-refresh.service hadto-ops-hermes-health.service hermes-codex-auth-health.service
```

Verification probe:

```bash
systemctl --user list-timers --all --no-pager --plain
systemctl --user --no-pager --plain status ops-heartbeat.timer hourly-status.timer lenovo-backup.timer lenovo-backup-verify.timer hadto-ops-backup-restore-refresh.timer hadto-ops-capacity-refresh.timer hadto-ops-dashboard-feeds.timer hadto-ops-engineering-velocity-refresh.timer hadto-ops-hermes-health.timer hermes-codex-auth-health.timer
```

### 5.4 Drain or Cancel Tracked Codex, Hermes, and ctx Sessions

Do not kill processes as a normal migration step. Ask owners to let tasks
complete or cancel them from their owning terminal/session. If a process remains
after the agreed drain period, treat the window as blocked unless the manager
approves a process-specific intervention.

Verification probe:

```bash
python - <<'PY'
import os
from pathlib import Path

needles = ('codex', 'hermes', 'run_agent.py', 'hermes_cli', 'tui_gateway', 'gateway/run.py', 'cron/scheduler.py', 'ctx')
for pid in filter(str.isdigit, os.listdir('/proc')):
    try:
        cmd = (Path('/proc') / pid / 'cmdline').read_bytes().replace(b'\0', b' ').decode('utf-8', 'replace')
        comm = (Path('/proc') / pid / 'comm').read_text(errors='replace').strip()
    except OSError:
        continue
    if any(n in cmd.lower() for n in needles):
        try:
            cwd = os.readlink(Path('/proc') / pid / 'cwd')
        except OSError:
            cwd = None
        print(f'pid={pid} comm={comm} cwd={cwd}')
PY
```

### 5.5 Stop Docker Containers With Relevant Bind Mounts

Only stop containers whose bind mounts touch copied roots or planned target
paths. At drafting time, the known containers requiring classification were
`hadto-ontology-workbench`, `phoneitin-monitoring-prometheus-1`, and
`phoneitin-monitoring-node-exporter-1`.

```bash
docker stop hadto-ontology-workbench phoneitin-monitoring-prometheus-1 phoneitin-monitoring-node-exporter-1
```

Verification probe:

```bash
docker ps --format '{{.Names}} {{.Status}}'
docker inspect --format '{{.Name}} {{range .Mounts}}{{.Type}}:{{.Source}}->{{.Destination}} {{end}}' $(docker ps -q)
```

### 5.6 Checkpoint SQLite and Verify WAL/SHM Quiet

Run only after all corresponding writers are stopped.

```bash
find /home/david/.hermes /home/david/stacks /home/david/.ctx-data /home/david/.codex -type f \( -name '*.db' -o -name '*.sqlite' -o -name '*.sqlite3' \) -print0 |
while IFS= read -r -d '' db; do
  printf 'Checking %s\n' "$db"
  sqlite3 "$db" 'PRAGMA quick_check;'
  sqlite3 "$db" 'PRAGMA wal_checkpoint(TRUNCATE);'
done
find /home/david/.hermes /home/david/stacks /home/david/.ctx-data /home/david/.codex -type f \( -name '*-wal' -o -name '*-shm' -o -name '*.db-wal' -o -name '*.db-shm' \) -print
```

Expected probe result: `quick_check` prints `ok` for each DB and the final
`find` prints nothing.

### 5.7 Run the Preflight Gate

Run section 4 exactly. Do not copy unless it prints:

```text
PREFLIGHT PASS: writer quiescence gate passed for <run-id>
```

## 6. Copy-Window Guardrails

The copy window is open only after section 4 passes.

Guardrails:

- SQLite DBs and their `-wal` or `-shm` files must not be copied while the
  owning Hermes, ctx, Codex, plugin, dashboard, gateway, cron, or Docker writer
  is active.
- Repo `.git` directories must not be copied while Codex delegates, Hermes
  workers, IDEs, git commands, or Docker services are changing the repo.
- Codex metadata under `/home/david/.codex` and `.git/hermes-codex` must not be
  copied while Codex processes are active or registry files are changing.
- ctx state under `/home/david/.ctx-data` must not be copied while ctx daemons
  or ctx-backed sessions are active.
- Docker bind-mounted files under copied roots must not be copied while their
  containers are running unless the mount is proven read-only and recorded in
  the evidence bundle.
- If any writer restarts unexpectedly, stop the copy, preserve evidence, and
  treat the run as blocked or rollback-only.

Suggested copy command shape for the migration plan, shown here only to define
the guardrail target:

```bash
rsync -aHAX --numeric-ids --info=progress2 /home/david/.hermes/ /data/hermes/profile-default/
rsync -aHAX --numeric-ids --info=progress2 /home/david/stacks/ /data/hermes/stacks/
rsync -aHAX --numeric-ids --info=progress2 /home/david/.ctx-data/ /data/hermes/ctx-data/
rsync -aHAX --numeric-ids --info=progress2 /home/david/.codex/ /data/hermes/codex-home/
```

Do not run those copy commands from this runbook unless the migration command
plan owns the current step and the preflight gate is still passing.

## 7. Start Order and Per-Step Verification Probes

Start storage-dependent services before top-level intake, in reverse of the stop
order.

### 7.1 Verify Mounts and Path Contracts First

```bash
findmnt -R /home/david/.hermes /home/david/stacks /home/david/.ctx-data /home/david/.codex
test -d /home/david/.hermes
test -d /home/david/stacks
test -d /home/david/.ctx-data
test -d /home/david/.codex
```

### 7.2 Start Docker Containers With Relevant Bind Mounts

```bash
docker start hadto-ontology-workbench phoneitin-monitoring-prometheus-1 phoneitin-monitoring-node-exporter-1
docker ps --format '{{.Names}} {{.Status}}'
docker inspect --format '{{.Name}} {{range .Mounts}}{{.Type}}:{{.Source}}->{{.Destination}} {{end}}' hadto-ontology-workbench phoneitin-monitoring-prometheus-1 phoneitin-monitoring-node-exporter-1
```

### 7.3 Start User Timers

```bash
systemctl --user start ops-heartbeat.timer hourly-status.timer lenovo-backup.timer lenovo-backup-verify.timer hadto-ops-backup-restore-refresh.timer hadto-ops-capacity-refresh.timer hadto-ops-dashboard-feeds.timer hadto-ops-engineering-velocity-refresh.timer hadto-ops-hermes-health.timer hermes-codex-auth-health.timer
systemctl --user list-timers --all --no-pager --plain
```

### 7.4 Start Hermes Gateway Last

```bash
systemctl --user start hermes-gateway.service
systemctl --user --no-pager --plain status hermes-gateway.service
journalctl --user -u hermes-gateway.service -n 100 --no-pager
```

Verification probes:

```bash
python - <<'PY'
from pathlib import Path
for p in (Path('/home/david/.hermes/state.db'), Path('/home/david/.hermes/gateway.pid'), Path('/home/david/.hermes/gateway_state.json')):
    print(f'{p}: {"present" if p.exists() else "missing"}')
PY
sqlite3 /home/david/.hermes/state.db 'PRAGMA quick_check;'
```

## 8. Slack-Facing Operator Messages

These are message bodies for the human operator to paste into the existing
Slack incident or ops thread. This runbook does not send Slack messages.

Start:

```text
Starting Hermes data-migration maintenance window now. Intake, cron/timers, Codex/Hermes/ctx sessions, and Docker bind mounts will be quiesced before any copy starts. I will post again before rollback or completion.
```

Blocked:

```text
Blocked before copy. Writer-quiescence preflight found active writers or unresolved state, so no migration copy is running. I am preserving the evidence bundle and will retry only after the listed blockers are cleared.
```

Rollback:

```text
Rolling back the migration window. Writers remain stopped while I restore the previous path state, verify mounts and SQLite checks, then restart Docker, timers, and Hermes gateway in order.
```

Complete:

```text
Maintenance window complete. Writers were stopped before copy, path contracts are verified, SQLite and git checks passed, Docker/timers/Hermes gateway are back up, and the evidence bundle is captured.
```

## 9. Post-Window Verification and Evidence Capture Checklist

Create the evidence directory and capture read-only state:

```bash
mkdir -p "${MIGRATION_EVIDENCE_DIR}"
date -Is | tee "${MIGRATION_EVIDENCE_DIR}/completed-at.txt"
git -C /home/david/stacks/hermes-agent status --short --branch >"${MIGRATION_EVIDENCE_DIR}/hermes-agent-git-status.txt"
systemctl --user --no-pager --plain status hermes-gateway.service >"${MIGRATION_EVIDENCE_DIR}/hermes-gateway-status.txt"
systemctl --user list-timers --all --no-pager --plain >"${MIGRATION_EVIDENCE_DIR}/systemd-user-timers.txt"
docker ps --format '{{.ID}} {{.Names}} {{.Status}}' >"${MIGRATION_EVIDENCE_DIR}/docker-ps.txt"
findmnt -R /home/david/.hermes /home/david/stacks /home/david/.ctx-data /home/david/.codex >"${MIGRATION_EVIDENCE_DIR}/findmnt-path-contracts.txt"
sqlite3 /home/david/.hermes/state.db 'PRAGMA quick_check;' >"${MIGRATION_EVIDENCE_DIR}/state-db-quick-check.txt"
find /home/david/.hermes /home/david/stacks /home/david/.ctx-data /home/david/.codex -type f \( -name '*-wal' -o -name '*-shm' -o -name '*.db-wal' -o -name '*.db-shm' \) -print >"${MIGRATION_EVIDENCE_DIR}/post-window-wal-shm.txt"
journalctl --user -u hermes-gateway.service -n 200 --no-pager >"${MIGRATION_EVIDENCE_DIR}/hermes-gateway-journal.txt"
```

Checklist:

- The rollback plan was recorded before stopping writers.
- Slack start message was posted before stopping writers.
- `hermes-gateway.service` stopped before copy and started last.
- Relevant user timers and services were stopped before copy and restarted
  after path-contract verification.
- Codex, Hermes, and ctx process inventories were quiet before copy.
- Docker containers with bind mounts touching copied roots were stopped or
  explicitly classified as read-only before copy.
- SQLite `quick_check` passed before and after copy.
- WAL/SHM files were absent after checkpointing or documented as safe.
- Git dirty-state inventory was captured before copy.
- Path contracts still resolve at `/home/david/.hermes`, `/home/david/stacks`,
  `/home/david/.ctx-data`, and `/home/david/.codex`.
- Slack complete or rollback message was posted after verification.

## 10. Contradictions, Unknowns, and Uncovered Regions

Contradictions and unknowns known at drafting time:

- HAD-1147 path-contract inventory was not present in the repository, so this
  runbook uses independently verifiable discovery commands instead of a
  checked-in contract file.
- `lsof` was not available on the inspected host. The runbook uses `/proc`
  handle scanning and Docker/systemd inventories instead.
- Docker named volumes are inventoried but not migrated by this runbook. Treat
  them as service-specific follow-up scope unless the migration plan explicitly
  includes them.
- `phoneitin-monitoring-node-exporter-1` bind-mounted `/` during drafting. It
  may be read-only operational telemetry, but it still touches copied roots and
  must be classified or stopped.
- Current active Codex work may include delegated tasks whose command lines
  reference `/home/david/stacks/hermes-agent`. Those must complete before a
  real migration window.
- User timers observed on the host are not all Hermes-specific. They are still
  included for classification because backup, heartbeat, status, and Hadto ops
  jobs can read or write copied roots.

Uncovered regions that must be resolved before migration:

- Any path added by HAD-1147 that is not one of `/home/david/.hermes`,
  `/home/david/stacks`, `/home/david/.ctx-data`, `/home/david/.codex`, or
  `/data/hermes`.
- Any system service outside `systemctl --user` with bind mounts or writes into
  copied roots.
- Any non-Docker VM, remote mount, sync daemon, editor, backup agent, or desktop
  indexer that holds open file handles under copied roots.
- Any repo under `/home/david/stacks` intentionally left dirty at copy time.
- Any SQLite DB outside the protected roots that must remain transactionally
  consistent with copied Hermes, Codex, ctx, or repo state.
