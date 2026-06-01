# HAD-1147 path-contract inventory

Issue: HAD-1147, Data migration: finalize path-contract inventory.

## Scope and non-goals

This is a first-pass path contract for moving bytes to `/data` without breaking the logical paths Hermes and Hadto automation already depend on. It covers Hermes source, host runtime config, user systemd units, Docker/Compose bind mounts, Codex delegate metadata, ctx metadata, and tool storage paths visible on the Lenovo host.

No migration commands were executed for this issue. This artifact only records decisions for a later migration issue.

Non-goals:

* Do not move, copy, rsync, delete, chmod, chown, mount, unmount, or restart runtime services.
* Do not edit `/home/david/.hermes/config.yaml`, systemd unit files, Docker Compose files, or live secrets.
* Do not resolve every historical checkout under `/home/david/stacks`; only first-pass contracts and repeated mount patterns are captured.

## Evidence inventory

### Hermes source path resolution

* `hermes_constants.py:14-68`: `get_hermes_home()` reads `HERMES_HOME`, otherwise falls back to `Path.home() / ".hermes"`. It warns if `~/.hermes/active_profile` names a non-default profile while `HERMES_HOME` is unset.
* `hermes_constants.py:71-107`: `get_default_hermes_root()` treats normal/profile paths under `~/.hermes` differently from Docker/custom roots such as `/opt/data`.
* `hermes_constants.py:124-142`: `get_hermes_dir(new_subpath, old_name)` preserves legacy directories under `HERMES_HOME` when they already exist.
* `hermes_constants.py:165-188`: `get_subprocess_home()` makes `{HERMES_HOME}/home` the subprocess `HOME` only when that directory exists.
* `hermes_constants.py:277-294`: `get_config_path()`, `get_skills_dir()`, and `get_env_path()` resolve to `HERMES_HOME/config.yaml`, `HERMES_HOME/skills`, and `HERMES_HOME/.env`.

### State and logging

* `hermes_state.py:36`: `DEFAULT_DB_PATH = get_hermes_home() / "state.db"`.
* `hermes_state.py:41-55`: SQLite WAL mode is preferred, with DELETE fallback for filesystems where WAL locks are unsafe. This matters for any `/data` filesystem choice.
* `hermes_logging.py:1-13`: logs live under `~/.hermes/logs/`, profile-aware through `get_hermes_home()`.
* `hermes_logging.py:198` and `hermes_logging.py:357-379`: logging creates log parents under the resolved Hermes home and reads logging config from `get_config_path()`.

### Cron storage

* `cron/jobs.py:1-6`: jobs are stored in `~/.hermes/cron/jobs.json`; output is saved under `~/.hermes/cron/output/{job_id}/{timestamp}.md`.
* `cron/jobs.py:37-45`: `CRON_DIR = HERMES_DIR / "cron"`, `JOBS_FILE = CRON_DIR / "jobs.json"`, `OUTPUT_DIR = CRON_DIR / "output"`.
* `cron/scheduler.py:1260-1303`: cron scripts must resolve inside `HERMES_HOME/scripts`; absolute and `~` paths are validated against that scripts directory.
* `cron/scheduler.py:1442-1461`: `context_from` reads prior outputs from `OUTPUT_DIR / source_job_id`.

### Gateway runtime

* `systemctl --user cat hermes-gateway.service`: `ExecStart=/home/david/stacks/hermes-agent/venv/bin/python -m hermes_cli.main gateway run --replace`; `WorkingDirectory=/home/david/stacks/hermes-agent`; `VIRTUAL_ENV=/home/david/stacks/hermes-agent/venv`; `HERMES_HOME=/home/david/.hermes`; `EnvironmentFile=/home/david/.hermes/.env`.
* `gateway/whatsapp_identity.py:85`: WhatsApp session data resolves under `get_hermes_home() / "whatsapp" / "session"`.
* `mcp_serve.py:66,102,366`: MCP/session helper paths use `HERMES_HOME/sessions`, `HERMES_HOME/channel_directory.json`, and `HERMES_HOME/state.db`.

### Terminal backend

* `tools/terminal_tool.py:1163-1230`: terminal backend config comes from environment variables such as `TERMINAL_CWD`, `TERMINAL_DOCKER_VOLUMES`, `TERMINAL_DOCKER_ENV`, and `TERMINAL_DOCKER_RUN_AS_HOST_USER`.
* `tools/terminal_tool.py:1165-1191`: Docker backend sanitizes host `TERMINAL_CWD`; only configured bind-mounted host paths should remain usable inside the container unless explicit `/workspace` mounting is enabled.
* `/home/david/.hermes/config.yaml` path assumptions captured by sanitized read: default terminal `cwd: /home/david/stacks`; Docker image `nikolaik/python-nodejs:python3.11-nodejs20`; `CODEX_HOME: /home/pn/.codex`; host bind list includes `/home/david/stacks`, `/home/david/.codex`, `/home/david/.hermes/bin/codex`, `/home/david/.hermes/backlog`, `/home/david/.hermes/codex`, `/home/david/.hermes/ctx`, `/home/david/.hermes/implementation_delegate`, `/home/david/.hermes/notes`, `/home/david/.hermes/self_improvement`, `/home/david/.hermes/cache/documents`, and GitHub/Stripe config paths.

### Codex delegate and ctx

* `/home/david/.hermes/plugins/hadto/hadto_hermes_plugin/tools/codex_delegate.py:1-5`: local Codex delegate is legacy inventory/status/resume surface; new repo-backed starts should use VM isolation in that plugin version.
* `/home/david/.hermes/plugins/hadto/hadto_hermes_plugin/tools/codex_delegate.py:150-160`: Codex delegate config comes from `load_config()` under the `codex_delegate` key.
* `/home/david/.hermes/plugins/hadto/hadto_hermes_plugin/tools/workspace_backlog.py:33-36`: coordinator defaults are `HERMES_HOME/codex/runs.json`, `HERMES_HOME/vm-workers/runs.json`, `HERMES_HOME/ctx/session_bindings.json`, and legacy ctx worktrees under `~/.ctx-data/worktrees`.
* `/home/david/.hermes/plugins/hadto/hadto_hermes_plugin/ctx_runtime.py:897-975`: ctx loads config, finds auth material, resolves the workspace, and builds worktree paths as `{CTX_DATA_DIR or config.data_dir or default}/worktrees/{workspace_id}/{worktree_id}`.
* `/home/david/.hermes/config.yaml` ctx assumption: `ctx.data_dir: /home/david/.ctx-data`.
* `/home/david/.hermes/config.yaml` VM worker assumptions: storage under `/home/david/.hermes/vm-workers/rootless-qemu`, toolchain under `/home/david/.hermes/vm-workers/toolchain`, base image under `/home/david/.hermes/vm-workers/rootless-qemu/base.qcow2`, Codex and ripgrep binaries under `/home/david/.local/share/mise/...`.

### Tool storage

* `tools/tool_result_storage.py:1-23`: oversized tool outputs are persisted into the sandbox temp dir for later `read_file` access.
* `tools/tool_result_storage.py:39-57`: default result storage is `/tmp/hermes-results`, or `{env.get_temp_dir()}/hermes-results` when the environment exposes a temp dir.
* `tools/code_execution_tool.py` search evidence: script/project resolution uses `TERMINAL_CWD`; generated script storage may read `~/.hermes/.env`; cache areas include `{HERMES_HOME}/home` when present.
* `tools/tts_tool.py` and `tools/vision_tools.py` search evidence: media cache helpers use `get_hermes_dir("cache/audio", "audio_cache")`, `get_hermes_dir("cache/vision", "temp_vision_images")`, and `get_hermes_dir("cache/video", "temp_video_files")`.

### Docker and Compose

* `docker-compose.yml:31-32` and `docker-compose.yml:65-66`: Hermes gateway and dashboard containers mount `~/.hermes:/opt/data`.
* `docker-compose.yml:24-71`: gateway and dashboard both use `network_mode: host`; dashboard binds localhost via command.
* Host `docker inspect` snapshot for live containers:
  * `hadto-pipeline`: Docker volume `hadto-pipeline_pipeline-data` mounted at `/app/data`.
  * `hadto-ontology-workbench`: `/home/david/stacks/hadto-decision-room-data/decision-room:/data/decision-room:rw` and `/home/david/stacks/smb-ontology-platform/ops:/data/smb-ops:ro`.
  * `ontology-archivebox`: Docker volume `ontology-platform_archivebox-data` mounted at `/data`.
  * `ontology-triplestore`: Docker volume `ontology-platform_oxigraph-data` mounted at `/data`.
  * `ontology-source-materials-blob-store`: Docker volume `ontology-platform_source-materials-minio-data` mounted at `/data`.
* Repo-wide `docker-compose*.yml` search under `/home/david/stacks` found repeated `/home/david/stacks/smb-ontology-platform/ops` bind defaults in Hadto ontology workbench checkouts, plus Hermes `~/.hermes:/opt/data` mounts in Hermes checkouts.

### Systemd user services and timers

* `systemctl --user list-timers --all --no-pager` showed active/found timers: `ops-heartbeat.timer`, `lenovo-backup.timer`, `lenovo-backup-verify.timer`, `hourly-status.timer`, `hadto-ops-*` refresh timers, and `hermes-codex-auth-health.timer`.
* `systemctl --user cat ops-heartbeat.service`: `ExecStart=/home/david/.ops-agent/ops_heartbeat.sh`.
* `systemctl --user cat hermes-codex-auth-health.service`: `WorkingDirectory=/home/david/.hermes/plugins/hadto`; `PYTHONPATH=/home/david/stacks/hermes-agent:/home/david/.hermes/plugins/hadto`; Python executable `/home/david/stacks/hermes-agent/.venv/bin/python`; script `/home/david/.hermes/plugins/hadto/scripts/check_ctx_delegate_health.py`.

## Path-contract table

| Logical path | Source/evidence | Owner | Mutability | Backup class | Migration strategy | Decision | Notes |
|---|---|---|---|---|---|---|---|
| `/home/david/.hermes` | `HERMES_HOME=/home/david/.hermes` in `hermes-gateway.service`; `get_hermes_home()` fallback | Hermes runtime | Hot mutable state, config, credentials, logs, jobs, caches | Critical, includes secrets and live state | Move physical bytes to `/data`, preserve logical path with bind-mount or symlink only after service-stop plan | bind-mount | Must remain visible as `/home/david/.hermes` to current source, systemd, Docker, cron, and plugin contracts. |
| `/home/david/.hermes/config.yaml` | `get_config_path()` and sanitized config read | Hermes runtime | Mutable config | Critical config, no secrets expected but may include operational routing | Move with `.hermes`; no separate path rewrite | bind-mount | Source assumes `HERMES_HOME/config.yaml`. |
| `/home/david/.hermes/.env` | `get_env_path()`; `hermes-gateway.service.d/10-hadto-host-env.conf` | Hermes runtime | Mutable secret file | Critical secret | Move with `.hermes`; keep exact logical path and systemd EnvironmentFile contract | bind-mount | Do not expose content in migration logs. |
| `/home/david/.hermes/state.db` plus WAL/shm files | `hermes_state.py:36`; WAL fallback comments | Hermes runtime | Hot SQLite | Critical live database | Move with `.hermes`; ensure target filesystem supports SQLite WAL locks or accept DELETE fallback | bind-mount | Stop gateway before physical move in later issue. |
| `/home/david/.hermes/logs` | `hermes_logging.py` | Hermes runtime | Append/rotate | Operational, medium retention | Move with `.hermes` or optionally split to `/data/hermes/logs` behind same logical path | bind-mount | Managed mode may chmod logs 0660. |
| `/home/david/.hermes/cron/jobs.json` | `cron/jobs.py` | Hermes scheduler | Mutable scheduler metadata | Critical operational | Move with `.hermes`; keep logical path | bind-mount | Gateway dispatch reads this path. |
| `/home/david/.hermes/cron/output` | `cron/jobs.py`; `cron/scheduler.py context_from` | Hermes scheduler | Append-only report archive | Useful history | Move with `.hermes`; can tier backup lower than jobs.json | bind-mount | Context chaining depends on existing output path. |
| `/home/david/.hermes/scripts` | `cron/scheduler.py:1260-1303` | Hermes scheduler/user | User-managed scripts | Critical if jobs depend on scripts | Move with `.hermes`; keep logical path | bind-mount | Absolute script paths are validated to stay inside this directory. |
| `/home/david/.hermes/skills` | `get_skills_dir()` | Hermes/user | Mutable procedural memory | Critical | Move with `.hermes` | bind-mount | Skills are durable user data. |
| `/home/david/.hermes/plugins` | config and systemd health service | Hermes/plugin runtime | Mutable plugin code | Critical runtime | Move with `.hermes`; keep path stable for PYTHONPATH and WorkingDirectory | bind-mount | `hermes-codex-auth-health.service` directly enters this tree. |
| `/home/david/.hermes/codex/runs.json` | Hadto plugin `workspace_backlog.py:33` | Hadto plugin | Mutable run registry | Critical coordination | Move with `.hermes`; keep path stable | bind-mount | Used to detect live/finished Codex runs. |
| `/home/david/.hermes/ctx/session_bindings.json` | Hadto plugin `workspace_backlog.py:35` | Hadto plugin | Mutable run registry | Critical coordination | Move with `.hermes`; keep path stable | bind-mount | Config binds this read-only into Docker terminal sandbox. |
| `/home/david/.hermes/vm-workers` | sanitized config VM worker storage/toolchain/base image | Hadto VM worker plugin | Large mutable VM artifacts | Critical but bulky | Move with `.hermes` or split to `/data/hermes/vm-workers` mounted back | bind-mount | Large QEMU/base-image storage is a prime data-disk candidate. |
| `/home/david/.hermes/backlog` | sanitized Docker volume list | Hadto plugin | Mutable coordinator snapshots | Critical coordination | Move with `.hermes`; keep logical path | bind-mount | Bound into Docker coding sandbox. |
| `/home/david/.hermes/implementation_delegate` | sanitized Docker volume list | Hadto plugin | Mutable delegation artifacts | Critical coordination | Move with `.hermes`; keep logical path | bind-mount | Bound into Docker coding sandbox. |
| `/home/david/.hermes/notes` | sanitized Docker volume list | Hermes/Hadto notes | Mostly mutable notes | Important | Move with `.hermes`; optionally read-only mounts remain read-only | bind-mount | Config binds it read-only into Docker terminal sandbox. |
| `/home/david/.hermes/self_improvement` | sanitized Docker volume list | Hadto plugin | Mutable benchmark/history | Important | Move with `.hermes`; keep logical path | bind-mount | Used by self-improvement pipeline. |
| `/home/david/.hermes/cache` | media/tool caches, config output mount | Hermes tools | Rebuildable cache with some user-visible generated media | Mixed, mostly cache | Move with `.hermes for compatibility`; backup selective | bind-mount | `cache/documents` is mounted to `/output` in Docker sandbox. |
| `/home/david/.hermes/home` | `get_subprocess_home()` | Hermes subprocesses | Tool credentials/configs if directory exists | Critical if present | Move with `.hermes`; keep exact path | bind-mount | Controls subprocess `HOME`; may contain git/gh/npm/ssh state. |
| `/home/david/stacks` | host convention, systemd WorkingDirectory, config terminal cwd, Docker volume | User/repos | Hot repos and working checkouts | Critical source/WIP | Move physical repo tree to `/data/stacks` only with `/home/david/stacks` bind-mount | bind-mount | Many systemd, Docker, terminal, and repo assumptions name this path directly. |
| `/home/david/stacks/hermes-agent` | `hermes-gateway.service` ExecStart/WorkingDirectory/PYTHONPATH | Hermes source runtime | Git checkout and venv | Critical runtime | Keep logical path via parent `/home/david/stacks` bind-mount | bind-mount | Gateway currently runs from this exact checkout and venv path. |
| `/home/david/stacks/hermes-agent/venv` | `hermes-gateway.service` | Hermes runtime | Python env, rebuildable but live | Rebuildable/important | Move with repo tree or rebuild after bind mount | bind-mount | Exact interpreter path is in systemd. |
| `/home/david/stacks/hermes-agent/.venv` | `hermes-codex-auth-health.service` | Hermes runtime | Python env, rebuildable but live | Rebuildable/important | Move with repo tree or rebuild after bind mount | bind-mount | Different systemd health service names `.venv`. |
| `/home/david/.codex` | sanitized Docker volume list, `CODEX_HOME=/home/pn/.codex`, VM worker code | Codex CLI | OAuth/auth/config | Critical secret | Move physical bytes to `/data/codex`, keep `/home/david/.codex` bind-mounted | bind-mount | Docker maps host `/home/david/.codex` to container `/home/pn/.codex`. |
| `/home/pn/.codex` | sanitized config Docker `CODEX_HOME` | Docker Codex runtime | Container-visible Codex state | Derived mount target | No host move; preserve container target mapping | no-move | Host-side source is `/home/david/.codex`. |
| `/home/david/.ctx-data` | sanitized config `ctx.data_dir`; ctx runtime worktree path formula | ctx daemon | ctx workspace/worktree data | Critical WIP | Move physical bytes to `/data/ctx-data`, keep `/home/david/.ctx-data` bind-mounted | bind-mount | ctx worktree paths are persisted in bindings and expected by coordinator hygiene. |
| `/tmp/hermes-results` | `tools/tool_result_storage.py:39-57` | Tool runtime | Ephemeral tool output spill | Ephemeral | Do not move in data migration | no-move | Sandbox temp, not durable host contract. |
| `{env.get_temp_dir()}/hermes-results` | `tools/tool_result_storage.py:44-57` | Tool runtime | Ephemeral tool output spill | Ephemeral | Do not move globally | no-move | Backend-owned temp path. |
| `/opt/data` | `docker-compose.yml` `~/.hermes:/opt/data`; `get_default_hermes_root()` Docker/custom root | Hermes Docker container | Container view of Hermes home | Derived mount target | Preserve mount target; move only host source | no-move | Container contract should stay `/opt/data` unless Compose is changed deliberately. |
| `/workspace` | `tools/terminal_tool.py` cwd remap when Docker cwd passthrough is enabled | Terminal Docker runtime | Container view of mounted CWD | Derived mount target | No host move | no-move | Only active when explicit mount-to-workspace is enabled. |
| `/home/david/.ops-agent/ops_heartbeat.sh` | `ops-heartbeat.service` | Ops automation | Executable script | Important | Either move with a separate `/home/david/.ops-agent` bind-mount or leave in place if small | bind-mount | Outside issue checklist but discovered in related timers. |
| `/home/david/stacks/smb-ontology-platform/ops` | Docker inspect and compose search | SMB ontology ops | Generated ops feed files | Important/live | Move with `/home/david/stacks` parent bind-mount | bind-mount | Mounted read-only into OWB. |
| `/home/david/stacks/hadto-decision-room-data/decision-room` | Docker inspect | OWB data | Mutable app data | Important | Move with `/home/david/stacks` parent bind-mount or separate data mount | bind-mount | Mounted read-write into OWB. |
| Docker named volumes under `/var/lib/docker/volumes/...` | `docker inspect` for hadto-pipeline and ontology containers | Docker daemon | Mutable app data | Critical per service | Out of scope for home path move; handle with Docker volume backup/restore plan | no-move | Not one of the requested home-path contracts, but must be backed up before disk work. |
| `/home/david/.local/share/uv/python` | sanitized Docker volume list | User toolchain | Installed Python runtimes | Rebuildable/important | Leave in place for first pass or move under broader home strategy later | no-move | Mounted read-only into Docker sandbox. |
| `/home/david/.config/gh` and `/home/david/.config/git` | sanitized Docker volume list | User tool config/secrets | Mutable credentials/config | Critical secret | Not part of requested move set; preserve and back up before any home-wide operation | no-move | Bound read-only into Docker sandbox. |

## First-pass decisions by class

* Move behind bind-mount: `/home/david/.hermes`, `/home/david/stacks`, `/home/david/.codex`, `/home/david/.ctx-data`, and likely `/home/david/.ops-agent` if ops scripts move with the same data-disk migration.
* Bind-mount not rewrite: keep logical paths stable because source, config, persisted run records, systemd units, Docker mounts, and comments already reference those paths.
* No-move: derived container targets such as `/opt/data`, `/home/pn/.codex`, `/workspace`, and ephemeral temp storage such as `/tmp/hermes-results`.
* Defer to service-specific backup: Docker named volumes under `/var/lib/docker/volumes` are not home-path contracts. They need a separate Docker volume backup/restore plan if the Docker data root moves.

## Contradictions and unknowns

* Hermes source is profile-aware and supports custom `HERMES_HOME`, but live systemd pins `HERMES_HOME=/home/david/.hermes`. Migration should preserve the pinned logical path unless a later issue deliberately updates systemd and all config surfaces together.
* `get_default_hermes_root()` supports Docker/custom roots such as `/opt/data`; live Compose implements that by mounting host `~/.hermes` to `/opt/data`. That is not a reason to point host `HERMES_HOME` at `/opt/data`.
* `hermes-gateway.service` uses `venv/bin/python`; `hermes-codex-auth-health.service` uses `.venv/bin/python`. Both venv paths must be accounted for, even if one is a compatibility artifact.
* Codex delegate in the live Hadto plugin now says new repo-backed starts should use VM isolation, while the Linear event policy asked Hermes to use Codex delegation first. I attempted Codex delegation, hit `codex_auth_token_expired`, and continued with native isolated fallback.
* The sanitized config read captured path assumptions without dumping secrets. Values under secret-like keys were redacted and are not fully enumerated here.
* The Docker Compose search under `/home/david/stacks` includes many historical agent checkouts. The durable live evidence is the `docker inspect` snapshot plus canonical repo Compose files. Historical duplicate compose paths should not be treated as active services without container or systemd evidence.
* I did not inspect every unit file under `/home/david/.config/systemd/user`; I inspected gateway, timer list, ops heartbeat, and Codex auth health because they match the issue scope and active Hermes/Hadto paths.

## Follow-up checklist for the migration issue

1. Stop or quiesce `hermes-gateway.service`, cron dispatch, Codex/VM workers, ctx daemon sessions, and any service writing under `/home/david/.hermes`, `/home/david/.codex`, `/home/david/.ctx-data`, or `/home/david/stacks`.
2. Take backups of `/home/david/.hermes`, `/home/david/.codex`, `/home/david/.ctx-data`, `/home/david/stacks`, and the listed Docker named volumes if Docker data is in scope.
3. Verify target `/data` filesystem supports SQLite WAL locking before moving `state.db`; otherwise explicitly accept DELETE fallback risk and lower concurrency.
4. Move bytes to `/data/...` only in the migration issue, then create bind mounts preserving `/home/david/.hermes`, `/home/david/.codex`, `/home/david/.ctx-data`, and `/home/david/stacks` logical paths.
5. Verify `systemctl --user cat hermes-gateway.service`, `systemctl --user cat hermes-codex-auth-health.service`, and Docker bind mounts still resolve after remount.
6. Restart services and verify Hermes gateway, cron jobs, Linear agent sessions, terminal Docker backend, Codex auth health, ctx bindings, and `/ops` feeds.
7. Update this manifest if the actual migration changes a decision from bind-mount to path rewrite.
