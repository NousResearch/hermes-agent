# Newest Container → Original Container Preservation Bundle

Generated: 2026-05-17T14:10:23Z
Newest container hostname/id: `0fab39ae3c6b`
Original container target: `hermes-agent-a4cy`

## Purpose

This bundle captures the repo/vault-visible state from the newest Hermes container so the original container can incorporate it deliberately, with dry-run checks before anything is applied.

This is intentionally **not** a blind container overwrite. The safe path is:

1. preserve newest-container work,
2. validate checksums,
3. inspect/dry-run on original container,
4. apply patches/extract untracked files,
5. run targeted tests,
6. verify Telegram/gateway behavior,
7. only then switch or retire the newest container.

## Captured artifacts

- `hermes-agent-tracked.patch`
  - Modified tracked files in `/opt/hermes`:
    - `gateway/platforms/telegram.py`
    - `hermes_cli/gateway.py`
    - `hermes_cli/main.py`
- `hermes-agent-untracked-files.tar.gz`
  - 75 untracked Hermes repo files/directories, including:
    - `agent/smart_model_routing.py`
    - `gateway/builtin_hooks/boot_md.py`
    - `gateway/platforms/qqbot.py`
    - new docs under `docs/`
    - new skills under `skills/`
    - new tests under `tests/`
    - new web UI components under `web/src/`
- `jarvis-vault-tracked.patch`
  - Modified tracked Obsidian/JARVIS vault notes:
    - `00 - Command Center/Active Priorities.md`
    - `04 - LOBOBYTE/LOBOBYTE HQ.md`
    - `04 - LOBOBYTE/LOBOBYTE Tasks.md`
- `jarvis-vault-untracked-files.tar.gz`
  - New Obsidian/JARVIS notes:
    - `00 - Command Center/Newest Container Preservation Plan.md`
    - `10 - Daily Notes/2026-05-16 - Recommended Actions Executed.md`
- `lobobyte-mockups-hub-git-status.txt`
  - Confirmed clean at capture time.
- `sanitized-env-manifest.md`
  - Captures environment key names and important paths without secret values.
- `process-snapshot.txt`, `system-snapshot.txt`, `node-npm-versions.txt`
  - Runtime context for comparison.
- `SHA256SUMS.txt`
  - Integrity checksums for the bundle.

## Important observations

- Current session has **no Docker socket** at `/var/run/docker.sock`; do not try to switch containers from inside this Telegram session.
- One live gateway process was present at capture time:
  - `/opt/hermes/.venv/bin/hermes gateway run --replace`
- Many defunct `[hermes]` zombies are present; based on current CPU/memory behavior this is maintenance/yellow, not an immediate red alarm.
- `pip freeze` could not be captured because `/opt/hermes/.venv/bin/python` reported `No module named pip`. Do not assume Python package drift is fully captured; rely on repo manifests and test execution in original container.
- No obvious secrets were found in the plaintext patch files by a simple keyword scan. Tarballs should still be treated as internal artifacts.

## What can go wrong if we are careless

### 1. Repo patch conflict

The original container may not be at the same commit/state as this newest container. A patch can fail or, worse, apply against stale context.

Mitigation:

```bash
cd /opt/hermes
git status --porcelain=v1 --branch --ignore-submodules=all
git apply --check /path/to/hermes-agent-tracked.patch
```

Do not apply if `--check` fails.

### 2. Untracked files overwrite original-container work

The tarball contains files that are currently untracked in newest container. If original already has files at those paths with different contents, extraction could clobber them.

Mitigation:

```bash
cd /opt/hermes
tar -tzf /path/to/hermes-agent-untracked-files.tar.gz | while read -r p; do
  [ -e "$p" ] && echo "EXISTS: $p"
done
```

If any paths exist, inspect before extraction.

### 3. Obsidian/JARVIS vault drift

The vault is a separate Git repo. Applying vault patches inside the original container can conflict if the vault has moved forward.

Mitigation:

```bash
cd /opt/data/jarvis-vault
git fetch origin
git status --porcelain=v1 --branch
git apply --check /path/to/jarvis-vault-tracked.patch
```

Prefer committing/pushing vault changes from one place, then pulling them in the original container.

### 4. Secrets/config are not migrated by patch

This bundle intentionally does not copy secret values. If the newest container changed `.env`, `config.yaml`, API tokens, bot tokens, SSH keys, or provider credentials, those must be migrated manually and securely.

Mitigation:

- Compare `sanitized-env-manifest.md` against original container keys.
- Do not paste secrets into chat or into repo files.
- Verify original container can still access GitHub, Telegram, OpenRouter/OpenAI/etc. after cutover.

### 5. Runtime/dependency drift

If newest container installed packages manually, those may not be represented in repo files.

Mitigation:

- Run tests inside original container after applying.
- If missing imports/packages appear, update proper dependency manifests rather than ad-hoc installing only in the container.

### 6. Gateway behavior regression

The tracked patch touches Telegram/gateway/CLI files. A bad merge could affect Telegram command handling or gateway startup.

Mitigation:

Run targeted tests before cutover:

```bash
cd /opt/hermes
TMPDIR=/opt/data/tmp scripts/run_tests.sh \
  tests/gateway/test_telegram_chat_id_arg.py \
  tests/gateway/test_plan_command.py \
  tests/gateway/test_async_memory_flush.py \
  tests/gateway/test_flush_memory_stale_guard.py \
  tests/cli/test_cli_plan_command.py \
  -q
```

Then verify Telegram inbound/outbound manually.

### 7. Memory/skills split-brain

Some useful state lives in Hermes memory/skills/vault rather than the code repo.

Mitigation:

- Built-in repo skills in `/opt/hermes/skills` are captured by the Hermes tarball.
- User skills under `/opt/data/skills` were not copied into this code patch because they are persistent volume state, not repo state; verify original container mounts the same `/opt/data`.
- Obsidian vault changes are captured separately.
- If original uses a different Hermes home, copy/pull the vault and skills explicitly.

### 8. Host/container targeting mistake

Accidentally applying to the newest container again, or to the wrong original container, would create false confidence.

Mitigation:

Every host-side script must print:

```bash
docker exec hermes-agent-a4cy hostname
docker exec hermes-agent-a4cy pwd
docker exec hermes-agent-a4cy bash -lc 'cd /opt/hermes && git rev-parse --show-toplevel && git status --porcelain=v1 --branch --ignore-submodules=all'
```

Proceed only if target identity is correct.

## Original-container apply checklist

Run from Docker host, not from this Telegram session, unless Docker socket access is added.

### Step 0: Verify bundle integrity

```bash
cd /opt/data/reports/container-preservation/20260517T141023Z
sha256sum -c SHA256SUMS.txt
```

### Step 1: Inspect original container

```bash
ORIGINAL_CONTAINER=hermes-agent-a4cy

docker ps --format '{{.Names}}' | grep -Fxq "$ORIGINAL_CONTAINER"
docker exec "$ORIGINAL_CONTAINER" hostname
docker exec "$ORIGINAL_CONTAINER" bash -lc 'cd /opt/hermes && git status --porcelain=v1 --branch --ignore-submodules=all'
docker exec "$ORIGINAL_CONTAINER" bash -lc 'cd /opt/data/jarvis-vault && git status --porcelain=v1 --branch || true'
```

### Step 2: Make a backup inside original container

```bash
docker exec "$ORIGINAL_CONTAINER" bash -lc 'mkdir -p /opt/data/reports/pre-restore-backups && cd /opt/hermes && git diff --binary > /opt/data/reports/pre-restore-backups/original-hermes-before-restore.patch'
docker exec "$ORIGINAL_CONTAINER" bash -lc 'cd /opt/data/jarvis-vault && git diff --binary > /opt/data/reports/pre-restore-backups/original-vault-before-restore.patch || true'
```

### Step 3: Dry-run Hermes patch

```bash
docker exec "$ORIGINAL_CONTAINER" bash -lc 'cd /opt/hermes && git apply --check /opt/data/reports/container-preservation/20260517T141023Z/hermes-agent-tracked.patch'
```

### Step 4: Check untracked path collisions

```bash
docker exec "$ORIGINAL_CONTAINER" bash -lc 'cd /opt/hermes && tar -tzf /opt/data/reports/container-preservation/20260517T141023Z/hermes-agent-untracked-files.tar.gz | while read -r p; do [ -e "$p" ] && echo "EXISTS: $p"; done'
```

If this prints any `EXISTS:` lines, inspect before extracting.

### Step 5: Apply Hermes changes

```bash
docker exec "$ORIGINAL_CONTAINER" bash -lc 'cd /opt/hermes && git apply /opt/data/reports/container-preservation/20260517T141023Z/hermes-agent-tracked.patch'
docker exec "$ORIGINAL_CONTAINER" bash -lc 'cd /opt/hermes && tar -xzf /opt/data/reports/container-preservation/20260517T141023Z/hermes-agent-untracked-files.tar.gz'
```

### Step 6: Apply/pull vault changes

If the vault changes have already been committed/pushed from newest container, prefer:

```bash
docker exec "$ORIGINAL_CONTAINER" bash -lc 'cd /opt/data/jarvis-vault && git pull --ff-only origin main'
```

Otherwise dry-run and apply the captured patch/tar:

```bash
docker exec "$ORIGINAL_CONTAINER" bash -lc 'cd /opt/data/jarvis-vault && git apply --check /opt/data/reports/container-preservation/20260517T141023Z/jarvis-vault-tracked.patch'
docker exec "$ORIGINAL_CONTAINER" bash -lc 'cd /opt/data/jarvis-vault && git apply /opt/data/reports/container-preservation/20260517T141023Z/jarvis-vault-tracked.patch'
docker exec "$ORIGINAL_CONTAINER" bash -lc 'cd /opt/data/jarvis-vault && tar -xzf /opt/data/reports/container-preservation/20260517T141023Z/jarvis-vault-untracked-files.tar.gz'
```

### Step 7: Run tests in original container

```bash
docker exec "$ORIGINAL_CONTAINER" bash -lc 'cd /opt/hermes && mkdir -p /opt/data/tmp && TMPDIR=/opt/data/tmp scripts/run_tests.sh tests/gateway/test_telegram_chat_id_arg.py tests/gateway/test_plan_command.py tests/gateway/test_async_memory_flush.py tests/gateway/test_flush_memory_stale_guard.py tests/cli/test_cli_plan_command.py tests/agent/test_smart_model_routing.py tests/run_agent/test_context_pressure.py tests/run_agent/test_flush_memories_codex.py -q'
```

### Step 8: Verify gateway and Telegram

```bash
docker exec "$ORIGINAL_CONTAINER" bash -lc "ps -eo pid,ppid,etime,stat,pcpu,pmem,cmd | grep -E 'hermes|gateway|telegram' | grep -v grep"
```

Then send a Telegram test message and verify response from the original container before shutting down/replacing newest container.

## Decision rule

Do not cut over if any of these are true:

- checksum verification fails,
- `git apply --check` fails,
- untracked tar extraction would overwrite files you have not inspected,
- targeted tests fail,
- Telegram inbound/outbound is not verified,
- original container is not definitely `hermes-agent-a4cy`.
