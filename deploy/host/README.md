# Matter host image

This directory builds and boots a litco-agent matter host: one DigitalOcean droplet per real matter, created from a versioned snapshot. The snapshot carries code and packages but no secrets. Each droplet gets its secrets and its matter identity from cloud-init user-data at first boot.

The design follows `MATTER_AGENT_VM_PILOT_2026-09-27.md` section 5 in the litkit-notes repo, and the turn server it runs is described in `docs/litco.md`.

## Files

| File | Role |
|---|---|
| `install-host.sh` | The install steps, run as root on the machine being built. The droplet build and the smoke container both run it. `--topology per_matter` (the default, also `LITCO_TOPOLOGY`) builds the per-matter image described below; `--topology machine` builds the machine host in "Machine topology". `--dry-run` prints every step and runs nothing. |
| `build-image.sh` | Runs on the operator's machine. It creates a throwaway builder droplet, runs `install-host.sh` there, cleans cloud-init state, snapshots `litco-agent-host-<version>`, and deletes the builder. `--dry-run` prints the plan and touches nothing. |
| `cloud-init.yaml.tmpl` | Per-matter user-data template. |
| `cloud-init.machine.yaml.tmpl` | Machine host user-data template ("Machine topology"). |
| `render-user-data.py` | Renders either template from the control plane's JSON spec; `"topology": "machine"` selects the machine one. Standard library only. |
| `litco-agent.service` | The per-matter systemd unit: the Hermes gateway with the `litco_turn` platform, run as `hermes`. |
| `litco-agent@.service` | The machine host's template unit: one slot per matter, run as that matter's user `m_<slot>`. |
| `litco-agent-init` | `ExecStartPre`. It renders `$HERMES_HOME/config.yaml` and `SOUL.md` from `profile/` using non-secret variables only. |
| `litco-agent-drain` | `ExecStop`. It asks the turn server to drain (new turns get `503 draining`) and waits until no turn is running. |
| `litco-host-update` | Machine host only. It builds a release beside the running one, flips `current`, and restarts slots one at a time. |
| `litco-supervisor` | Machine host only. The root daemon that creates, starts, stops and removes slots for the app. |
| `litco-slot-probe` | Machine host only. Checks from inside one slot's sandbox that it cannot read another slot's secrets or the metadata API. |
| `litco-slot-import` | Machine host only. Restores a migrated matter's homes into a stopped slot. |
| `profile/config.yaml.tmpl`, `profile/SOUL.md` | The matter profile templates. |
| `smoke.sh`, `smoke/` | Local container smoke tests: per-matter by default, `--topology machine` for two slots on a machine. |
| `Makefile` | `make -C deploy/host test`, `dry-run`, `smoke`. |

## What the image contains

| Path | Contents |
|---|---|
| `/opt/litco-agent/app` | The litco-agent checkout at the pinned ref, owned by root. |
| `/opt/litco-agent/app/.venv` | The Hermes venv, built with `uv sync --frozen --extra messaging` from `uv.lock`. |
| `/opt/litco-agent/python` | uv-managed CPython 3.14. Ubuntu 24.04's own SQLite (3.45) has the WAL-reset bug, and the uv build bundles SQLite 3.53. The install script refuses a Python whose SQLite is older than 3.51.3. |
| `/opt/litco-agent/tools` | The PM store: `agent-browser` and the PM-pinned Chromium, found by Hermes through `HERMES_RUNTIME_DIR`. |
| `/opt/litco-agent/toolenv` | A venv for the agent's own scripts: duckdb, pandas, matplotlib, python-docx, PyMuPDF, requests, openpyxl. It is first on the service's `PATH`, so `python` in the agent's terminal has these. |
| system packages | curl, git, jq, ripgrep, ffmpeg, 7zip, unrar, poppler-utils (`pdftotext`, `pdftoppm`), LibreOffice core/writer/calc, metric-compatible fonts, Node 24 LTS, Tailscale, ufw, fail2ban, unattended-upgrades. |
| `hermes` user | uid 10000, with linger enabled. Upstream's cron and Kanban workers use `systemd-run --user`, which needs hermes's own user manager. |

The unit is baked enabled and stopped, and no `/etc/litco-agent/env` exists in the image. `install-host.sh` fails if that file exists, and `build-image.sh` checks again before the snapshot.

## Versioning

The snapshot name is the version, and the version is a git tag on the fork.

1. Tag the commit you want on hosts, for example `git tag host-2026.09.27 && git push origin host-2026.09.27`.
2. Build: `deploy/host/build-image.sh --version host-2026.09.27 --ssh-key <fingerprint>`. The script refuses a version whose snapshot already exists, so a version always names one image.
3. Point the control plane at `litco-agent-host-host-2026.09.27`, and record that name on each host it creates.

`/opt/litco-agent/REF` on every host holds the exact commit it runs.

## How the control plane boots a host

The control plane builds a spec, renders user-data, and creates the droplet from the snapshot:

```
python3 deploy/host/render-user-data.py spec.json > user-data.yaml     # or: ... - < spec.json
```

It then calls DigitalOcean's droplet create with `image` set to the snapshot, `size` `s-4vcpu-8gb`, `region` `sfo3`, `monitoring` on, and `user_data` set to the output. The output is at most 64 KiB, which is DigitalOcean's limit; the renderer refuses anything larger. On a bad spec it exits 2 and names the field, and it never echoes a secret.

The A7 control plane is TypeScript. It can call this script, or port it. If it ports it, `tests/host/test_render_user_data.py` is the contract to keep passing.

### The spec

| Field | Required | Meaning |
|---|---|---|
| `matterId` | yes | The LitKit matter id. |
| `instanceUrl` | yes | The firm's instance, `https://<firm>.litco.ai`. |
| `hostSecret` | yes | Secret, at least 32 characters. LitKit sends it in `X-Host-Secret`, and it keys user assertions. |
| `agentToken` | yes | Secret. The matter-pinned LitKit token, `lkm_…`. |
| `modelProvider` | yes | A Hermes provider slug: `anthropic`, `openrouter`, `custom`, and so on. |
| `egressPolicy` | yes | `open` or `allowlist`. |
| `matterHome` | no | Default `/home/hermes/matter`. |
| `model` | no | Model name, such as `anthropic/claude-opus-4.6`. |
| `modelBaseUrl` | no | OpenAI-compatible base URL, used with provider `custom` (for example, a future LitKit model proxy). |
| `modelKey` | no | Secret. It lands in the provider's own key variable (`ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, …). For `custom` it lands in `LITCO_MODEL_API_KEY`. |
| `typesafeApiKey` | no | Secret. It lands in `TYPESAFE_API_KEY`, which `litkit_jev` (the Jev first-pass screen) reads; a machine host's supervisor takes the same field in its `PUT /slots/{id}` body. Without it the tool reports that Jev is not configured. |
| `tailscaleAuthKey` | no | Secret. With it, the host joins the tailnet with Tailscale SSH. |
| `hostname` | no | `matter-<shortid>`. By default it is derived from the first eight letters and digits of `matterId`. |
| `egressAllowlist` | no | `[{"cidr": "203.0.113.0/24", "port": 443, "proto": "tcp"}]`. Allowed only with `egressPolicy: "allowlist"`. |
| `approvalsMode` | no | Empty (the upstream default), `manual`, `smart`, or `off`. See below. |
| `turnHost`, `turnPort` | no | Default `0.0.0.0` and `8765`. The firewall admits inbound traffic only on `tailscale0`. |
| `channelEnv` | no | Native Slack or Telegram settings, limited to `SLACK_BOT_TOKEN`, `SLACK_APP_TOKEN`, `SLACK_ALLOWED_USERS`, `SLACK_HOME_CHANNEL`, `SLACK_IGNORED_CHANNELS`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_ALLOWED_USERS`, `TELEGRAM_GROUP_ALLOWED_USERS`, `TELEGRAM_GROUP_ALLOWED_CHATS`. |

### First boot, in order

1. cloud-init writes `/etc/litco-agent/env` (root:root, 0600), `/etc/litco-agent/matter.json` (non-secret), and, if a Tailscale key was given, `/etc/litco-agent/tailscale-authkey` (0600).
2. It sets the hostname to `matter-<shortid>` and creates the matter home, owned by `hermes`.
3. With a key, it runs `tailscale up --ssh --hostname=matter-<shortid> --auth-key=file:…` and then shreds the key file.
4. It resets ufw: deny all inbound, allow inbound on `tailscale0`, allow all outbound, plus the allowlist block when requested.
5. It runs `systemctl restart litco-agent.service`. It uses restart because the unit is baked enabled; the unit's `ConditionPathExists=/etc/litco-agent/env` kept it from running before this point.
6. The unit's `ExecStartPre` renders the profile, and the gateway starts with `litco_turn` on port 8765.

## The environment contract

systemd reads `/etc/litco-agent/env` as root for the unit's `EnvironmentFile=`. The `hermes` user cannot read the file. The gateway and the tools Hermes spawns get the values as environment variables.

| Variable | Secret | Read by |
|---|---|---|
| `LITCO_MATTER_ID` | no | turn server, init |
| `LITCO_INSTANCE_URL` | no | turn server, init, LitKit tools |
| `LITCO_MATTER_HOME` | no | turn server, init |
| `LITCO_TURN_HOST`, `LITCO_TURN_PORT` | no | turn server, init, drain |
| `LITCO_MODEL_PROVIDER`, `LITCO_MODEL`, `LITCO_MODEL_BASE_URL` | no | init (into `config.yaml`) |
| `LITCO_MODEL_KEY_ENV` | no | init. It is set to `LITCO_MODEL_API_KEY` only for provider `custom` with a key, and becomes `model.key_env`. |
| `LITCO_APPROVALS_MODE` | no | init |
| `LITCO_EGRESS_POLICY` | no | operators (informational) |
| `LITCO_HOST_SECRET` | yes | turn server |
| `LITCO_AGENT_TOKEN` | yes | turn server, LitKit tools |
| `<PROVIDER>_API_KEY` or `LITCO_MODEL_API_KEY` | yes | Hermes provider resolution |
| `SLACK_*`, `TELEGRAM_*` | yes | Hermes native adapters |

Two rules hold throughout. No script in this repo writes a secret; only cloud-init does, from user-data. And `litco-agent-init` reads only the variables in its `NON_SECRET_KEYS` list, so a template cannot pull in a secret: an unknown placeholder is an error. The tests check both.

Like the LitKit worker fleet, secrets ride in user-data, and DigitalOcean's metadata API serves user-data to processes on the droplet for its lifetime. The droplet serves one matter, and the agent on it already holds these secrets in its environment.

## Egress policy

`open` is the pilot posture: nothing inbound except over the tailnet, and all outbound traffic allowed (pilot plan section 7).

`allowlist` sets outbound to deny by default. It then allows the tailnet interface, DNS, NTP, Tailscale's direct UDP port 41641, and each `egressAllowlist` entry. ufw matches addresses, not names. Destinations behind DNS names (the model provider, the LitKit instance behind a CDN, Tailscale's control plane and relays, Westlaw and Lexis) need their current address ranges or a logging proxy. Without Tailscale's control-plane ranges, the host drops off the tailnet. The allowlist block is scaffolding for the day the pilot shows which destinations the agent needs; no host has run with it.

## Approvals: one knob, the owner decides

Hermes gates commands it classifies as dangerous. This image adds no override of its own. The single knob is `approvalsMode` in the spec, which becomes `LITCO_APPROVALS_MODE` and then `approvals.mode` in `config.yaml`:

| Value | Effect |
|---|---|
| empty (default) | No `approvals` block is rendered, so the upstream Hermes default applies (`smart` at fork time: a guardian model approves low-risk commands and escalates the rest). |
| `manual` | Every flagged command needs a human. |
| `smart` | The same as the upstream default, pinned. |
| `off` | No approval gate. This is the same as Hermes's `--yolo`. |

A turn that arrives through `litco_turn` has no approval channel. The turn server registers no approval notifier, and LitKit has no `/approve` surface. So when a command needs a human, the tool returns a "pending approval" result to the agent at once, and the agent sees that the command did not run. The turn does not hang for the approval timeout. (This comes from reading `tools/approval.py`: with no gateway notifier and no CLI, the gate returns `_pending_result`. The smoke test does not exercise it.) The native Slack and Telegram adapters, when configured, keep their own `/approve` flow.

Which posture a matter runs under is the owner's decision. The image ships the upstream default until the owner sets the knob.

## Stopping and restarting never kills a running turn

`systemctl stop` or `restart` first runs `litco-agent-drain`. It sends `POST /drain` with the host secret, so the turn server answers new turns with `503 draining` and LitKit sends them to its daemon. It then polls `GET /health` until `activeTurns` is 0. A turn server from before `/drain` answers 404, and the drain only polls. `TimeoutStopSec=infinity` means systemd waits as long as that takes. Then Hermes's stop marker runs, and `KillMode=mixed` sends SIGTERM to the gateway process alone. This mirrors the LitSpace worker rule. The drain counts turns on the turn server only. Native Slack and Telegram turns and cron jobs are drained by Hermes on SIGTERM. To stop with a turn still running, on purpose, use `systemctl kill litco-agent`. `LITCO_DRAIN_MAX_SECONDS` in the env file caps the wait if the owner ever wants a cap.

## Rolling the fork forward

New hosts: tag, rebuild the snapshot, and point the control plane at the new name. Hosts created from then on get the new code.

Existing hosts: update in place over Tailscale SSH, one host at a time. The restart waits for running turns.

```
ssh root@matter-<shortid>     # Tailscale SSH
REF=host-2026.10.04
cd /opt/litco-agent/app
git fetch --tags origin && git checkout --detach "$REF" && git rev-parse HEAD > /opt/litco-agent/REF
UV_PYTHON_INSTALL_DIR=/opt/litco-agent/python UV_PROJECT_ENVIRONMENT=/opt/litco-agent/app/.venv \
  uv sync --frozen --no-dev --extra messaging
install -m 0755 deploy/host/litco-agent-init /usr/local/bin/litco-agent-init
install -m 0755 deploy/host/litco-agent-drain /usr/local/bin/litco-agent-drain
install -m 0644 deploy/host/litco-agent.service /etc/systemd/system/litco-agent.service
systemctl daemon-reload && systemctl restart litco-agent
curl -s http://127.0.0.1:8765/health
```

The `uv sync` step is needed only when `uv.lock` changed, and it is harmless otherwise. A change to `install-host.sh` itself, such as a new system package, reaches existing hosts only by hand or by replacing the host from the new snapshot. Replacing a host means snapshotting the matter home first, as the control plane's idle path already does.

## Machine topology

`FIRM_AGENT_HOST_2026-09-30.md` in the litkit repo replaces one droplet per matter with one droplet per deployment. On it, each matter runs its own turn server under its own Unix user. `install-host.sh --topology machine` builds that image. The per-matter image stays the default until cutover, so today's droplets keep booting unchanged.

The two images differ as follows.

| | `per_matter` | `machine` |
|---|---|---|
| Code | `/opt/litco-agent/app` | `/opt/litco-agent/releases/<ref>`, behind the `current` symlink |
| Unit | `litco-agent.service`, baked enabled, run as `hermes` | `litco-agent@<slot>.service`, never enabled, run as `m_<slot>` |
| Secrets | `/etc/litco-agent/env`, from cloud-init | `/run/litco-agent/<slot>.env` (tmpfs, root 0600), from `litco-supervisor` |
| Started by | cloud-init's restart | `litco-supervisor`, when LitKit asks for the slot |
| Inbound firewall | ufw, set by cloud-init | `/etc/nftables.conf`: drop, except `tailscale0` (loaded at the next boot, so the build's SSH session survives) |

`/opt/litco-agent/TOPOLOGY` records which image a host runs.

### The slot unit

`litco-agent@.service` carries the hardening in section 3.2 of the spec. The slot runs as `m_<slot>` in `/srv/litco/m/<slot>`. `TemporaryFileSystem=` and `BindPaths=` hide every other slot's home, and `ProtectProc=invisible` hides other users' processes. `PrivateTmp=` and `PrivateIPC=` give it its own `/tmp` and IPC namespace. `IPAddressDeny=169.254.169.254` closes the metadata API, which serves the machine's user-data. `InaccessiblePaths=` removes the slot env files and the supervisor's state from the slot's view. Limits: `MemoryMax=2G`, `TasksMax=2048`, `CPUWeight=100`.

The env file the supervisor writes carries `LITCO_MATTER_ID`, `LITCO_SLOT_PORT`, `LITCO_INSTANCE_URL`, `LITCO_HOST_SECRET`, `LITCO_AGENT_TOKEN` and the model settings. It also sets `HERMES_HOME`, `LITCO_MATTER_HOME` and `LITCO_TURN_PORT`, to the same values the unit would give them: the two homes inside `/srv/litco/m/<slot>`, and the turn port equal to the slot port. `litco-agent-init` refuses a home outside the slot's home and a `LITCO_TURN_PORT` that differs from `LITCO_SLOT_PORT`, so a value that disagreed would stop the slot rather than cross into another. The machine smoke runs two slots with these files.

`ExecStart` resolves `current` once, so a running slot keeps importing from the release it started on. `litco-agent-init` resolves it too, so the rendered profile names that release's skills.

### The machine's user-data

The control plane boots a machine as it boots a per-matter droplet, from a spec and `render-user-data.py`. `"topology": "machine"` in the spec selects `cloud-init.machine.yaml.tmpl`. A spec without `topology`, or with `"per_matter"`, renders exactly the per-matter output, byte for byte (`tests/host/fixtures` pins it). A machine spec carries machine secrets only: no matter id, agent token, host secret or model key. The supervisor receives those for each slot.

| Field | Required | Meaning |
|---|---|---|
| `topology` | yes | `machine`. |
| `hostname` | yes | `litco-host-<id>`, matching `^litco-host-[a-z0-9]([a-z0-9-]{0,40}[a-z0-9])?$`. It becomes the tailnet name the app dials. |
| `instanceUrl` | yes | The firm's instance, `https://<firm>.litco.ai`. Non-secret. |
| `supervisorSecret` | yes | Secret, at least 32 characters. The app's bearer for `litco-supervisor`. |
| `tailscaleAuthKey` | yes | Secret. Required because the supervisor listens on `tailscale0` only. |
| `egressPolicy` | yes | `open`. The renderer refuses `allowlist` on a machine. |

Any other field is an error. The output writes three files:

| File | Owner, mode | Contents |
|---|---|---|
| `/etc/litco-supervisor.env` | root:root 0600 | `LITCO_SUPERVISOR_SECRET="…"` and nothing else secret. The supervisor reads it itself (`--env-file`). |
| `/etc/litco-agent/machine.json` | root:root 0644 | `{schema, topology, hostname, instanceUrl}`. Non-secret. |
| `/etc/litco-agent/tailscale-authkey` | root:root 0600 | The Tailscale key, shredded after use. |

Secrets appear in the user-data only inside these files' base64 content. The metadata API serves the user-data for the droplet's lifetime. Every slot unit sets `IPAddressDeny=169.254.169.254`, and so does the supervisor's unit, so no matter process can fetch it.

### First boot of a machine, in order

1. At boot, before cloud-init's commands run, `nftables.service` loads `/etc/nftables.conf`: inbound traffic is dropped except on `tailscale0`, Tailscale's UDP port 41641, and replies. `litco-supervisor.service` is enabled but does not start, because `/etc/litco-supervisor.env` does not exist yet (`ConditionPathExists=`).
2. cloud-init writes the three files above.
3. It sets the hostname to `litco-host-<id>`.
4. It re-asserts the supervisor env file's owner and mode. The supervisor refuses a file that group or others can read.
5. It checks that the `inet litco_host` table is live and loads `/etc/nftables.conf` if it is not, so the machine never runs open. It then prints the table to the cloud-init log.
6. It runs `tailscale up --ssh --hostname=litco-host-<id> --auth-key=file:…` and shreds the key file.
7. It runs `systemctl daemon-reload` and `systemctl restart litco-supervisor.service`. It uses restart because the unit is baked enabled. The supervisor binds port 8700 on `tailscale0`, which step 6 brought up. Its unit retries every five seconds if the interface is late.
8. No slot starts at boot. `/run/litco-agent` is empty after any boot, so the supervisor marks every registered slot stopped, and the app's next turn for a matter asks for its slot.

There is no ufw step on a machine. The image's nftables rule set is the firewall.

### The supervisor's API

The app calls `http://<machine>:8700` with `Authorization: Bearer <supervisorSecret>`. `/health` needs no bearer.

| Call | Answer |
|---|---|
| `GET /health` | `200 {ok, service, ref, uptimeSeconds}` |
| `GET /slots` | `200 {slots: [{id, matterId, port, unixUser, state}]}` |
| `PUT /slots/{id}` | Creates the user `m_<id>` and the home if absent, enables linger, writes the env file, starts the unit. `200 {slot, created, userCreated, started}`. A PUT to a running slot rewrites the env file and does not restart it. |
| `POST /slots/{id}/stop` | Drains and stops the unit, then deletes the env file. `202 {slot}` while it stops, `200 {slot}` if it was already down. The user and the home stay. |
| `DELETE /slots/{id}` | Removes the slot for good, in the background. `202 {slot}` with state `stopping`, `404 no_such_slot` for an unknown id. |

`DELETE` runs these steps in order: `systemctl stop` (the drain waits for running turns), delete the env file, `loginctl disable-linger`, `loginctl terminate-user` (which ends the user's manager and any Hermes cron job at once, since `userdel` refuses a user with processes), `userdel`, remove `/srv/litco/m/<id>`, and drop the registry entry. Dropping the entry frees the port, and the next new slot takes it. While the removal runs, a `PUT` for the id answers `409 slot_stopping`. If a step fails, the entry stays with `desired: "removing"` and a log line names the step. `PUT` still answers 409, and another `DELETE` finishes the job. The home is removed without following links: a home that is itself a link is unlinked, and `shutil.rmtree` works by file descriptor and never descends through one.

Slot ids are 1–30 lowercase letters and digits in the supervisor, the unit and `litco-agent-init`, so `m_<id>` fits Linux's 32-character user names. The app uses the first 12 hex digits of the matter id.

### Checking isolation, and importing a matter

`litco-slot-probe <a> <b>` runs four fixed reads as `m_<a>` from inside slot `a`'s sandbox: `b`'s env file, `/proc/<pid of b>/environ`, a listing of `b`'s home, and a GET of the metadata user-data with a three-second timeout. It prints `{"env": "denied" | "READ", "proc": …, "home": …, "metadata": …}`, gives the reason for each verdict on stderr, and exits 0 only when all four are `denied`. Both slots must be running. The script's header explains how it enters the unit's cgroup and namespaces, and why `sudo -u m_<a>` would not test the unit at all. `IPAddressDeny=` drops a blocked packet silently, so a blocked metadata request reports `timed out`. That verdict means something only if the address answers from outside the slots: on a droplet, `curl -m 3 http://169.254.169.254/metadata/v1/user-data` as root returns the user-data.

`litco-slot-import <slot> <tarball>` restores a matter migrated from a per-matter droplet. The tarball holds `matter/` (the old `/home/hermes/matter`) and `hermes/` (the old `/home/hermes/.hermes`). The tool refuses unless the slot is registered and its unit is stopped. It refuses the whole tarball if any member is absolute, contains `..`, lies outside those two directories, is a device node or FIFO, or is a link that points outside its own directory. It extracts into a staging directory in the home, moves any existing `matter` or `.hermes` aside to `<name>.pre-import-<time>`, moves the new ones in, and gives the home to `m_<slot>`. `--dry-run` prints the plan and writes nothing. The cutover order is: the app's `prepare` (create the slot, start it once, stop it), then this import, then `activate`.

### Capacity on `s-4vcpu-8gb`

Spec section 3.5 planned with estimates. Two runs of the machine smoke measured each slot's `MemoryCurrent` on 2026-09-30, in the Ubuntu 24.04 container on an aarch64 colima VM (4 CPUs, 8 GiB), with the stub model. `MemoryCurrent` is the unit's whole cgroup charge, page cache included, so it overstates what the process alone holds.

| Component | Spec 3.5 estimate | Measured |
|---|---|---|
| Idle slot (Hermes gateway and turn server), just started | 300–500 MB | 245–357 MiB (run 1: a 357, b 284; run 2: a 334, b 258; a after a restart: 245, 247) |
| Idle slot after one turn | 300–500 MB | 300–423 MiB (run 1: a 423, b 330; run 2: a 376, b 300) |
| Active Chromium | 300–800 MB | not measured |
| LibreOffice conversion | 200–400 MB | not measured |
| OS | about 0.5 GB | not measured |

Eight idle slots at the measured high of about 425 MiB come to about 3.3 GiB. Two browsers at the estimate's 800 MB, one conversion at 400 MB and the OS at 0.5 GB bring the total to about 5.8 GB, inside the droplet's 8 GB. Each slot's `MemoryMax=2G` caps any one matter. The resize rule in section 3.5 is unchanged: move to `s-8vcpu-16gb` if memory pressure (PSI) stays above 10% for a week, more than four matters are routinely active at once, or any unit is killed for memory. A droplet measurement under real models and real tools should replace these figures.

### Updating a machine host

```
ssh root@<machine>     # Tailscale SSH
litco-host-update --ref host-2026.10.04 --dry-run    # the plan
litco-host-update --ref host-2026.10.04
```

The script builds `releases/<ref>` beside the running release and flips `current`. It then installs that release's slot unit, supervisor and itself, and restarts the supervisor only if its code or unit changed; a supervisor restart stops no slot. Last, it restarts the running slots one at a time. Each restart drains first, and the next slot waits until the last one answers `/health`. If a slot does not come back, the update stops there, and the slots not yet restarted keep running their old release. Passing an older ref whose release is still on disk rolls back without a rebuild. Pass tags or commits, not branch names: a release directory is built once and then reused.

## The browser on Ubuntu 24.04

Ubuntu 24.04 restricts unprivileged user namespaces through AppArmor, so Chromium's own sandbox cannot start from the PM store path. Hermes detects this and launches Chromium with `--no-sandbox` (`tools/browser_tool_session.py`). That fits the permissive pilot posture. A later hardening step would be an AppArmor profile that grants `userns` to that binary, plus `AGENT_BROWSER_ARGS` set so that Hermes stops passing `--no-sandbox`.

## Testing locally

```
scripts/run_tests.sh tests/host tests/deploy -q
deploy/host/build-image.sh --version v0 --dry-run
deploy/host/build-image.sh --version v0 --topology machine --dry-run
deploy/host/smoke.sh                             # per-matter; needs a Docker daemon
deploy/host/smoke.sh --topology machine          # two slots on a machine; needs a Docker daemon
```

`build-image.sh --topology machine` passes the topology to `install-host.sh` and checks the builder for the supervisor's env file and `/opt/litco-agent/TOPOLOGY` before the snapshot. The default stays `per_matter`.

`smoke.sh` exports the working tree, builds `smoke/Dockerfile` (Ubuntu 24.04 running `install-host.sh --container`, which skips Tailscale, ufw, fail2ban, unattended-upgrades, and linger), and boots systemd in the container. It then:

1. runs `systemd-analyze verify` on the unit;
2. checks that the unit is enabled and does not start without an env file;
3. writes a fake env file as cloud-init would and restarts the unit;
4. waits for `/health`;
5. checks the rendered profile, and checks that no secret reached `$HERMES_HOME` and that `hermes` cannot read the env file;
6. posts one `/turn` and asserts the SSE stream ends in a `final` frame with the stub model's answer;
7. renders a page with the PM Chromium as `hermes`;
8. stops the unit and checks that the drain ran.

The model is a stub OpenAI-compatible server inside the container (`smoke/stub_model.py`), so the turn runs through the real gateway, the `litco_turn` platform, `HermesTurnRunner`, and `AIAgent`. The unit tests in `tests/litco/` already cover the turn server with the fake runner.

`smoke.sh --topology machine` builds `smoke/Dockerfile.machine` (`install-host.sh --container --topology machine`) and drives two slots, `a` and `b`, through the real supervisor API. It starts the supervisor with a test env file and no interface binding. It then checks the image (topology `machine`, no `litco-agent.service`, no secret on disk), the 401 without the bearer, both `PUT`s (ports 8800 and 8801, users `m_a` and `m_b`, each `/health` naming its own matter), `systemd-analyze verify litco-agent@a.service`, a turn on each slot, and `403 matter_mismatch` for `b`'s matter on slot `a`. It runs `litco-slot-probe a b` and `b a`. It holds a slow turn on `a` (the stub sleeps on `SMOKE-SLOW-<n>`), stops `a`, and checks that a new turn gets `503 draining`, the held turn finishes, the env file is gone and `b` still answers. It restarts `a` on the same port with its home intact, and `DELETE`s `b`. Last, it prints each idle slot's `MemoryCurrent`. `smoke/machine.sh` lists the steps.

## What the first cloud build verified, and what it did not

The first cloud build ran on 2026-09-28. `BUILD_LOG.md` records it with the snapshot id, times and cost.

**Verified on a real x86_64 droplet.** A second droplet booted from the snapshot with no user-data, and these checks passed on it:

- `build-image.sh` ran end to end: the doctl flags, the SSH wait, the install, the snapshot, and the builder delete.
- `install-host.sh` ran outside a container on the x86_64 package set. The droplet-only steps ran: the Tailscale install (installed, not joined), `loginctl enable-linger hermes` (`Linger=yes`), fail2ban and unattended-upgrades (both active), and the sshd drop-in (`sshd -t` passes).
- `cloud-init clean --logs --machine-id` worked. A droplet from the snapshot got a new machine id, and cloud-init applied its hostname and SSH key. So droplets from this image process their user-data.
- The unit is enabled and does not start without `/etc/litco-agent/env` (`ConditionResult=no`). No secret is in the image.
- journald is persistent, and `do-agent` and `droplet-agent` are active once first boot finishes. These were checked on rc2.

**Not verified:**

- Matter user-data on a real droplet has not run. That covers `write_files` with `b64`, the `matter-<shortid>` hostname, `tailscale up --auth-key=file:` (the `file:` form needs a recent Tailscale), the ufw reset on a real kernel, the restart ordering, and the gateway answering `/health` on a droplet. The rendered YAML is parsed in tests, not by cloud-init.
- The allowlist egress policy has never been applied anywhere.
- Linger is on, but nothing has run through hermes's user manager. Cron jobs that need `systemd-run --user` are untested.
- A real model provider. The smoke turn used the stub model, not a provider key.
- The browser tool end to end through a model's tool call. The container smoke renders a page with the same Chromium binary but does not drive `agent-browser` from a turn.
- The pending-approval behavior on `litco_turn`, which comes from reading the code.
- Native Slack and Telegram through `channelEnv`.
- The 100 GB block volume for productions (pilot section 5) is not attached or mounted by this tooling. The control plane or a later cloud-init block has to do that.
- The control plane's `s-4vcpu-8gb` size has not booted this image, though the snapshot's 80 GB minimum disk fits it.

**Machine topology, not verified on a droplet.** The machine smoke ran every step of `smoke.sh --topology machine` in a container. No machine image has been built, and no droplet has booted one. So these have not run on a real droplet: the machine user-data (the `litco-host-<id>` hostname, `nftables.service` loading the rule set at boot, the `nft -f` fallback, `tailscale up` on a machine); the supervisor binding `tailscale0`; `loginctl` linger and `terminate-user` under a real logind; `IPAddressDeny=` against the real metadata service, where the container showed the drop only against a refusing address; and the capacity figures under a real model and real tools.
